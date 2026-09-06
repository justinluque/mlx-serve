//! Per-input-channel activation statistics — an "imatrix" — collected from
//! REAL generations, for a calibrated re-quantization.
//!
//! Lives in its own module because both halves of an Ideogram 4 pack need it
//! and they sit on opposite sides of an import: `ideogram4.zig` reads
//! `flux.zig` for the shared text encoder, so the collector cannot live in
//! either one.
//!
//! Off unless armed, and the cost when off is one comparison per projection.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

const S = mlx.mlx_stream;

inline fn mulA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&o, a, b, s));
    return o;
}
inline fn addA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&o, a, b, s));
    return o;
}
inline fn reshape(x: mlx.mlx_array, shape: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, s));
    return o;
}
inline fn astype(x: mlx.mlx_array, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, s));
    return o;
}

/// Per-input-channel mean-squared activation for every projection in a DiT,
/// accumulated across REAL generations.
///
/// `tests/convert_ideogram4.py` quantizes with a bare `mx.quantize` —
/// round-to-nearest on a min/max group, which spends the same precision on a
/// channel the model barely excites as on one carrying the prompt. Weighting
/// the quantizer's error by E[x^2] per input channel needs that expectation,
/// and for a DiT there is nowhere else to get it: its activations depend on
/// the latents and the timestep, so unlike a language model it cannot be
/// calibrated by pushing text through a reference tree. This engine is the
/// only thing on the machine that runs the model, so the statistics are
/// collected HERE, on real prompts, real seeds and the real sampler.
///
/// Armed by `MLX_SERVE_IDEOGRAM_IMATRIX=<abs path>` and off otherwise — the
/// cost when off is one null check per projection. Keys are
/// `<component>/<module>` (`transformer/layers.0.attention.qkv`), which is
/// what the converter looks up, so ONE file calibrates both checkpoints.
///
/// Collected through a QUANTIZED pack, because that is the only Ideogram DiT
/// that runs here: an activation second moment is far more stable to weight
/// noise than the weights are, but it is a real approximation and the file
/// says so. The check is a fixed point — re-collect on the calibrated pack and
/// the statistics should barely move.
pub const Imatrix = struct {
    allocator: std.mem.Allocator,
    path: []u8,
    /// BORROWED from each `IgLinear.name`, which outlives the collector.
    keys: std.ArrayList([]const u8) = .empty,
    acc: std.ArrayList(mlx.mlx_array) = .empty,
    rows: std.ArrayList(u64) = .empty,

    pub fn init(allocator: std.mem.Allocator, path: []const u8) !*Imatrix {
        const self = try allocator.create(Imatrix);
        self.* = .{ .allocator = allocator, .path = try allocator.dupe(u8, path) };
        return self;
    }

    pub fn deinit(self: *Imatrix) void {
        for (self.acc.items) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        self.keys.deinit(self.allocator);
        self.acc.deinit(self.allocator);
        self.rows.deinit(self.allocator);
        self.allocator.free(self.path);
        self.allocator.destroy(self);
    }

    /// Reserve a slot for `name`. The slot INDEX lives on the linear, so the
    /// hot path never looks a string up.
    pub fn register(self: *Imatrix, name: []const u8) !i32 {
        try self.keys.append(self.allocator, name);
        try self.acc.append(self.allocator, .{ .ctx = null });
        try self.rows.append(self.allocator, 0);
        return @intCast(self.keys.items.len - 1);
    }

    /// Accumulate `sum(x^2)` over every row of one projection's input.
    ///
    /// In f32 deliberately: the sum runs over every token of every step of
    /// every generation, and a bf16 accumulator saturates long before the
    /// statistic converges.
    pub fn observe(self: *Imatrix, slot: usize, x: mlx.mlx_array, s: S) !void {
        const shape = mlx.getShape(x);
        if (shape.len == 0) return;
        const in_dim = shape[shape.len - 1];
        const f = try astype(x, .float32, s);
        defer _ = mlx.mlx_array_free(f);
        const flat = try reshape(f, &[_]c_int{ -1, in_dim }, s);
        defer _ = mlx.mlx_array_free(flat);
        const sq = try mulA(flat, flat, s);
        defer _ = mlx.mlx_array_free(sq);
        var sum = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sum_axis(&sum, sq, 0, false, s));
        const prev = self.acc.items[slot];
        if (prev.ctx == null) {
            self.acc.items[slot] = sum;
        } else {
            const merged = try addA(prev, sum, s);
            _ = mlx.mlx_array_free(sum);
            _ = mlx.mlx_array_free(prev);
            self.acc.items[slot] = merged;
        }
        _ = mlx.mlx_array_eval(self.acc.items[slot]);
        self.rows.items[slot] += @intCast(@divExact(numel(shape), in_dim));
    }

    fn numel(shape: []const c_int) c_int {
        var n: c_int = 1;
        for (shape) |d| n *= d;
        return n;
    }

    /// Write `sum(x^2)/rows` — the MEAN square, which is the weight
    /// `weighted_affine_quant` consumes unchanged. Called after every
    /// generation rather than at unload: a long calibration run that is
    /// interrupted keeps everything it had measured.
    pub fn save(self: *Imatrix) !void {
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        var written: usize = 0;
        for (self.keys.items, self.acc.items, self.rows.items) |name, a, rows| {
            if (a.ctx == null or rows == 0) continue;
            const den = mlx.mlx_array_new_float(@floatFromInt(rows));
            defer _ = mlx.mlx_array_free(den);
            var mean = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(mean);
            try mlx.check(mlx.mlx_divide(&mean, a, den, self.streamOf()));
            _ = mlx.mlx_array_eval(mean);
            const key = try std.fmt.allocPrintSentinel(self.allocator, "{s}", .{name}, 0);
            defer self.allocator.free(key);
            _ = mlx.mlx_map_string_to_array_insert(map, key.ptr, mean);
            written += 1;
        }
        const path = try std.fmt.allocPrintSentinel(self.allocator, "{s}", .{self.path}, 0);
        defer self.allocator.free(path);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        _ = mlx.mlx_map_string_to_string_insert(meta, "values", "mean-squared activation per INPUT channel (sum(x^2)/rows)");
        _ = mlx.mlx_map_string_to_string_insert(meta, "keys", "<component>/<module>, the names convert_ideogram4.py emits");
        _ = mlx.mlx_map_string_to_string_insert(meta, "collected_on", "a QUANTIZED pack — the only Ideogram DiT that runs on Apple Silicon");
        try mlx.check(mlx.mlx_save_safetensors(path.ptr, map, meta));
        log.info("[ideogram4] imatrix: {d} projections -> {s}\n", .{ written, self.path });
    }

    fn streamOf(self: *const Imatrix) S {
        _ = self;
        return mlx.mlx_default_gpu_stream_new();
    }
};

/// The process-wide collector, or null. Written once at load from the
/// environment and read on the inference thread, which is the only thread that
/// ever calls a forward.
pub var active: ?*Imatrix = null;


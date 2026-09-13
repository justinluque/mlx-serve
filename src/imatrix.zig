//! Per-input-channel activation statistics — an "imatrix" — collected from
//! REAL generations, for a calibrated re-quantization of a media checkpoint.
//!
//! A diffusion transformer cannot be calibrated offline the way a language
//! model can: its activations depend on the latents and the timestep, so the
//! statistics come from the one thing on the machine that runs the model, on
//! real prompts, seeds and the real sampler. `tests/image_quant.py` weights
//! each projection's reconstruction error by them.
//!
//! Armed by `MLX_SERVE_IMATRIX=<path>` and off otherwise; the cost when off is
//! one comparison per projection.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

const S = mlx.mlx_stream;

pub const ENV = "MLX_SERVE_IMATRIX";

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

/// Mean-squared activation per input channel for every armed projection,
/// accumulated across generations and saved after each one.
pub const Imatrix = struct {
    allocator: std.mem.Allocator,
    path: []u8,
    keys: std.ArrayList([]u8) = .empty,
    acc: std.ArrayList(mlx.mlx_array) = .empty,
    rows: std.ArrayList(u64) = .empty,

    pub fn init(allocator: std.mem.Allocator, path: []const u8) !*Imatrix {
        const self = try allocator.create(Imatrix);
        errdefer allocator.destroy(self);
        self.* = .{ .allocator = allocator, .path = try allocator.dupe(u8, path) };
        return self;
    }

    pub fn deinit(self: *Imatrix) void {
        for (self.acc.items) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        for (self.keys.items) |k| self.allocator.free(k);
        self.keys.deinit(self.allocator);
        self.acc.deinit(self.allocator);
        self.rows.deinit(self.allocator);
        self.allocator.free(self.path);
        self.allocator.destroy(self);
    }

    /// The slot filed under `key`, registered on first sight. Find-or-add, so
    /// an encoder freed and reloaded per request keeps accumulating into the
    /// same slots. The slot INDEX lives on the linear; the hot path never
    /// looks a string up.
    pub fn register(self: *Imatrix, key: []const u8) error{OutOfMemory}!i32 {
        for (self.keys.items, 0..) |k, i| {
            if (std.mem.eql(u8, k, key)) return @intCast(i);
        }
        const owned = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(owned);
        try self.keys.ensureUnusedCapacity(self.allocator, 1);
        try self.acc.ensureUnusedCapacity(self.allocator, 1);
        try self.rows.ensureUnusedCapacity(self.allocator, 1);
        self.keys.appendAssumeCapacity(owned);
        self.acc.appendAssumeCapacity(.{ .ctx = null });
        self.rows.appendAssumeCapacity(0);
        return @intCast(self.keys.items.len - 1);
    }

    /// Accumulate `sum(x^2)` over every row of one projection's input.
    ///
    /// In f32 deliberately: the sum runs over every token of every step of
    /// every generation, far past what bf16's precision can accumulate.
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
        // Realized per call: a lazy sum would hold every step's activations.
        _ = mlx.mlx_array_eval(self.acc.items[slot]);
        var n: u64 = 1;
        for (shape[0 .. shape.len - 1]) |d| n *= @intCast(d);
        self.rows.items[slot] += n;
    }

    /// Write `sum(x^2)/rows` — the MEAN square, which is the channel weight the
    /// quantizer consumes unchanged. Called after every generation, so an
    /// interrupted calibration run keeps everything it measured.
    pub fn save(self: *Imatrix) !void {
        const s = mlx.mlx_default_gpu_stream_new();
        defer _ = mlx.mlx_stream_free(s);
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        var means: std.ArrayList(mlx.mlx_array) = .empty;
        defer {
            for (means.items) |m| _ = mlx.mlx_array_free(m);
            means.deinit(self.allocator);
        }
        for (self.keys.items, self.acc.items, self.rows.items) |name, a, rows| {
            if (a.ctx == null or rows == 0) continue;
            const den = mlx.mlx_array_new_float(@floatFromInt(rows));
            defer _ = mlx.mlx_array_free(den);
            var mean = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_divide(&mean, a, den, s));
            try means.append(self.allocator, mean);
            _ = mlx.mlx_array_eval(mean);
            const key = try std.fmt.allocPrintSentinel(self.allocator, "{s}", .{name}, 0);
            defer self.allocator.free(key);
            _ = mlx.mlx_map_string_to_array_insert(map, key.ptr, mean);
        }
        const path = try std.fmt.allocPrintSentinel(self.allocator, "{s}", .{self.path}, 0);
        defer self.allocator.free(path);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        _ = mlx.mlx_map_string_to_string_insert(meta, "values", "mean-squared activation per INPUT channel (sum(x^2)/rows)");
        _ = mlx.mlx_map_string_to_string_insert(meta, "keys", "<component>/<module>, module = checkpoint key without .weight");
        try mlx.check(mlx.mlx_save_safetensors(path.ptr, map, meta));
        log.info("[imatrix] {d} projections -> {s}\n", .{ means.items.len, self.path });
    }
};

/// The process-wide collector, or null. Written once at load and read on the
/// inference thread, the only thread that ever calls a forward.
pub var active: ?*Imatrix = null;

/// Arm `active` from `MLX_SERVE_IMATRIX`, once per process.
pub fn armFromEnv(allocator: std.mem.Allocator) !void {
    if (active != null) return;
    const env = std.c.getenv(ENV) orelse return;
    const path = std.mem.sliceTo(env, 0);
    if (path.len == 0) return;
    active = try Imatrix.init(allocator, path);
    log.info("[imatrix] collecting activation statistics into {s}\n", .{path});
}

/// The key a projection's statistics are filed under: `<component>/<module>`.
/// A text encoder's wrapper prefix is dropped because packs disagree on it
/// (`language_model.`, `model.`, none), and a file collected through one pack
/// must calibrate a conversion that spells it another way.
pub fn moduleKey(allocator: std.mem.Allocator, component: []const u8, module: []const u8) error{OutOfMemory}![]u8 {
    var m = module;
    for ([_][]const u8{ "model.language_model.", "language_model.", "model." }) |p| {
        if (std.mem.startsWith(u8, m, p)) {
            m = m[p.len..];
            break;
        }
    }
    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ component, m });
}

/// Give every NAMED linear of type `L` reachable from `root` — through struct
/// fields, fixed arrays and slices — a slot under `component`. `L` carries
/// `name` (its checkpoint key) and `im_slot`. Unnamed linears are views a
/// backend derives from a named one and are skipped. Returns how many were armed.
pub fn armAll(comptime L: type, comptime T: type, root: *T, im: *Imatrix, component: []const u8) error{OutOfMemory}!usize {
    var n: usize = 0;
    try armWalk(L, T, root, im, component, &n);
    return n;
}

fn armWalk(comptime L: type, comptime T: type, ptr: *T, im: *Imatrix, component: []const u8, n: *usize) error{OutOfMemory}!void {
    if (T == L) {
        if (ptr.name.len == 0) return;
        const key = try moduleKey(im.allocator, component, ptr.name);
        defer im.allocator.free(key);
        ptr.im_slot = try im.register(key);
        n.* += 1;
        return;
    }
    switch (@typeInfo(T)) {
        .@"struct" => |st| inline for (st.field_names, st.field_types) |fname, ftype| {
            if (comptime reaches(L, ftype)) try armWalk(L, ftype, &@field(ptr.*, fname), im, component, n);
        },
        .array => |ar| for (ptr) |*e| try armWalk(L, ar.child, e, im, component, n),
        .pointer => |p| for (ptr.*) |*e| try armWalk(L, p.child, e, im, component, n),
        else => {},
    }
}

fn reaches(comptime L: type, comptime T: type) bool {
    if (T == L) return true;
    switch (@typeInfo(T)) {
        .@"struct" => |st| {
            for (st.field_types) |ft| if (reaches(L, ft)) return true;
            return false;
        },
        .array => |ar| return reaches(L, ar.child),
        .pointer => |p| return p.size == .slice and !p.attrs.@"const" and reaches(L, p.child),
        else => return false,
    }
}

const testing = std.testing;

test "an imatrix slot is found by its key, so a reloaded encoder keeps its slots" {
    const im = try Imatrix.init(testing.allocator, "/tmp/unused.safetensors");
    defer im.deinit();
    const first = try testing.allocator.dupe(u8, "text_encoder/layers.0.self_attn.q_proj");
    const a = try im.register(first);
    testing.allocator.free(first); // the collector owns its copy
    const b = try im.register("text_encoder/layers.0.self_attn.k_proj");
    const again = try im.register("text_encoder/layers.0.self_attn.q_proj");
    try testing.expect(a != b);
    try testing.expectEqual(a, again);
    try testing.expectEqual(@as(usize, 2), im.keys.items.len);
    try testing.expectEqualStrings("text_encoder/layers.0.self_attn.q_proj", im.keys.items[@intCast(a)]);
}

test "a module key names the component and drops the encoder's wrapper prefix" {
    const a = testing.allocator;
    const cases = [_][3][]const u8{
        .{ "text_encoder", "language_model.layers.3.mlp.down_proj", "text_encoder/layers.3.mlp.down_proj" },
        .{ "text_encoder", "model.layers.3.mlp.down_proj", "text_encoder/layers.3.mlp.down_proj" },
        .{ "text_encoder", "layers.3.mlp.down_proj", "text_encoder/layers.3.mlp.down_proj" },
        .{ "transformer", "blocks.0.attn.wq", "transformer/blocks.0.attn.wq" },
    };
    for (cases) |c| {
        const k = try moduleKey(a, c[0], c[1]);
        defer a.free(k);
        try testing.expectEqualStrings(c[2], k);
    }
}

test "arming reaches every named linear through fields, arrays and slices, and skips views" {
    const Lin = struct { name: []const u8 = "", im_slot: i32 = -1 };
    const Inner = struct { x: Lin, deep: struct { y: Lin } };
    const Holder = struct {
        cfg: u32 = 7,
        allocator: std.mem.Allocator,
        top: Lin,
        pair: [2]Lin,
        blocks: []Inner,
        view: Lin,
    };
    var inner = [_]Inner{
        .{ .x = .{ .name = "blocks.0.x" }, .deep = .{ .y = .{ .name = "blocks.0.y" } } },
        .{ .x = .{ .name = "blocks.1.x" }, .deep = .{ .y = .{ .name = "blocks.1.y" } } },
    };
    var h = Holder{
        .allocator = testing.allocator,
        .top = .{ .name = "top" },
        .pair = .{ .{ .name = "pair.0" }, .{ .name = "pair.1" } },
        .blocks = &inner,
        .view = .{},
    };
    const im = try Imatrix.init(testing.allocator, "/tmp/unused.safetensors");
    defer im.deinit();
    try testing.expectEqual(@as(usize, 7), try armAll(Lin, Holder, &h, im, "transformer"));
    try testing.expectEqual(@as(i32, -1), h.view.im_slot);
    try testing.expectEqualStrings("transformer/blocks.1.y", im.keys.items[@intCast(inner[1].deep.y.im_slot)]);
    const slot = h.pair[1].im_slot;
    try testing.expectEqual(@as(usize, 7), try armAll(Lin, Holder, &h, im, "transformer"));
    try testing.expectEqual(slot, h.pair[1].im_slot);
    try testing.expectEqual(@as(usize, 7), im.keys.items.len);
}

test "observe files the MEAN square per input channel across calls and leading axes" {
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const path = try std.fmt.allocPrintSentinel(a, "/tmp/mlx-serve-imatrix-test-{d}.safetensors", .{std.c.getpid()}, 0);
    defer a.free(path);
    defer _ = std.c.unlink(path.ptr);

    const im = try Imatrix.init(a, path);
    defer im.deinit();
    const slot: usize = @intCast(try im.register("transformer/blocks.0.attn.wq"));
    var b1 = [_]f32{ 1, 2, 3, 3, 2, 1 };
    const sh1 = [_]c_int{ 1, 2, 3 };
    const x1 = mlx.mlx_array_new_data(&b1, &sh1, 3, .float32);
    defer _ = mlx.mlx_array_free(x1);
    var b2 = [_]f32{ 2, 0, -2 };
    const sh2 = [_]c_int{ 1, 3 };
    const x2 = mlx.mlx_array_new_data(&b2, &sh2, 2, .float32);
    defer _ = mlx.mlx_array_free(x2);
    try im.observe(slot, x1, s);
    try im.observe(slot, x2, s);
    try im.save();

    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var tensors = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensors);
    var meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    try mlx.check(mlx.mlx_load_safetensors(&tensors, &meta, path.ptr, cpu));
    const it = mlx.mlx_map_string_to_array_iterator_new(tensors);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(it);
    var key: ?[*:0]const u8 = null;
    var got = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(got);
    try testing.expectEqual(@as(c_int, 0), mlx.mlx_map_string_to_array_iterator_next(&key, &got, it));
    try testing.expectEqualStrings("transformer/blocks.0.attn.wq", std.mem.span(key.?));
    _ = mlx.mlx_array_eval(got);
    const d = mlx.mlx_array_data_float32(got).?;
    const want = [_]f32{ 14.0 / 3.0, 8.0 / 3.0, 14.0 / 3.0 };
    for (want, 0..) |w, i| try testing.expectApproxEqAbs(w, d[i], 1e-5);
}

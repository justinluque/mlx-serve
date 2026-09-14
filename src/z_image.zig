//! Z-Image / Z-Image-Turbo (Tongyi-MAI, "S3-DiT" single-stream diffusion
//! transformer) native text→image. Architecture pulled directly from the
//! real checkpoint (`transformer/config.json`, 521 tensors) and from
//! diffusers `transformer_z_image.py` / `pipelines/z_image/pipeline_z_image.py`
//! (HEAD, Nov 2025) — see docs/reference.md "Z-Image" for the full trace.
//!
//! Pipeline: prompt -> Qwen3 text encoder (the SAME arch `transformer.zig`
//! already serves; captured via `CaptureLayers` like `ltx_video.gemmaCapture4`)
//! -> patchify+pad-to-32 (image + caption, separately) -> noise_refiner (2
//! modulated blocks, image only) + context_refiner (2 unmodulated blocks,
//! caption only) -> concat [image, caption] -> 30 adaLN-modulated main
//! layers -> FinalLayer -> unpatchify (image prefix only) -> Euler
//! flow-match step -> repeat -> VAE decode (the ORIGINAL flux-dev 16-channel
//! AutoencoderKL, NOT the FLUX.2 32-channel one in flux.zig).
//!
//! NO NUMERIC ORACLE (documented gap, same status as the SDXL work): built
//! against the real weight names/shapes/pipeline source, not run against a
//! local PyTorch reference (24GB+ download, no local torch in this
//! environment). Verify by generating a real image and looking at it.
//!
//! Z-Image (50 steps, CFG 5) and Z-Image-Turbo (8 steps, CFG 0, distilled —
//! no negative-prompt branch) are the SAME architecture; only the weights
//! and the sampling defaults (detected from the checkpoint dir name, same
//! convention as `mage_flow.dirIsEdit`) differ.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const transformer_mod = @import("transformer.zig");
const tok_mod = @import("tokenizer.zig");
const sse = @import("gen_sse.zig");
const mage_flow = @import("mage_flow.zig");

const Weights = model_mod.Weights;
const S = mlx.mlx_stream;
const MfLinear = mage_flow.MfLinear;

// ── Low-level array helpers (same primitives as flux.zig/mage_flow.zig) ──

inline fn matmul(x: mlx.mlx_array, w_t: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&o, x, w_t, s));
    return o;
}
inline fn addA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&o, a, b, s));
    return o;
}
inline fn mulA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&o, a, b, s));
    return o;
}
inline fn subA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_subtract(&o, a, b, s));
    return o;
}
inline fn reshape(x: mlx.mlx_array, shape: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, s));
    return o;
}
inline fn transpose(x: mlx.mlx_array, axes: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, s));
    return o;
}
inline fn astype(x: mlx.mlx_array, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, s));
    return o;
}
inline fn contig(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&o, x, false, s));
    return o;
}
inline fn rms(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&o, x, w, eps, s));
    return o;
}
inline fn tanhA(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_tanh(&o, x, s));
    return o;
}
inline fn silu(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, x, s));
    return mulA(x, sig, s);
}
fn scalarF(v: f32) mlx.mlx_array {
    return mlx.mlx_array_new_float(v);
}
fn concat(arrs: []const mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (arrs) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, axis, s));
    return o;
}
/// Slice the LAST axis of an arbitrary-rank array to `[idx, idx+1)`.
fn sliceLastIdx(x: mlx.mlx_array, idx: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const nd = sh.len;
    const a = std.heap.page_allocator;
    const lo = a.alloc(c_int, nd) catch return error.OutOfMemory;
    defer a.free(lo);
    const hi = a.alloc(c_int, nd) catch return error.OutOfMemory;
    defer a.free(hi);
    const st = a.alloc(c_int, nd) catch return error.OutOfMemory;
    defer a.free(st);
    for (0..nd) |i| {
        lo[i] = 0;
        hi[i] = sh[i];
        st[i] = 1;
    }
    lo[nd - 1] = idx;
    hi[nd - 1] = idx + 1;
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, lo.ptr, nd, hi.ptr, nd, st.ptr, nd, s));
    return o;
}
/// Slice axis 0 (the token axis) to `[start, stop)`, rank-agnostic.
fn sliceAxis0(x: mlx.mlx_array, start: c_int, stop: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const nd = sh.len;
    const a = std.heap.page_allocator;
    const lo = a.alloc(c_int, nd) catch return error.OutOfMemory;
    defer a.free(lo);
    const hi = a.alloc(c_int, nd) catch return error.OutOfMemory;
    defer a.free(hi);
    const st = a.alloc(c_int, nd) catch return error.OutOfMemory;
    defer a.free(st);
    for (0..nd) |i| {
        lo[i] = 0;
        hi[i] = sh[i];
        st[i] = 1;
    }
    lo[0] = start;
    hi[0] = stop;
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, lo.ptr, nd, hi.ptr, nd, st.ptr, nd, s));
    return o;
}

fn ownWeight(w: *const Weights, key: []const u8) !mlx.mlx_array {
    const a = w.get(key) orelse {
        log.err("[zimage] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingZImageWeight;
    };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&o, a));
    return o;
}
fn ownOpt(w: *const Weights, key: []const u8) ?mlx.mlx_array {
    const a = w.get(key) orelse return null;
    var o = mlx.mlx_array_new();
    mlx.check(mlx.mlx_array_set(&o, a)) catch return null;
    return o;
}
fn fmtKey(a: std.mem.Allocator, comptime f: []const u8, args: anytype) ![]u8 {
    return std.fmt.allocPrint(a, f, args);
}
/// Linear weight [out,in] -> pre-transposed [in,out], f32 (VAE only; dense).
fn loadLinT(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, s: S) !mlx.mlx_array {
    const wk = try fmtKey(a, "{s}.weight", .{prefix});
    defer a.free(wk);
    const raw = try ownWeight(w, wk);
    defer _ = mlx.mlx_array_free(raw);
    const t = try transpose(raw, &[_]c_int{ 1, 0 }, s);
    defer _ = mlx.mlx_array_free(t);
    const tc = try contig(t, s);
    defer _ = mlx.mlx_array_free(tc);
    return astype(tc, .float32, s);
}
fn loadVec(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, comptime suffix: []const u8, s: S) !mlx.mlx_array {
    const k = try fmtKey(a, "{s}." ++ suffix, .{prefix});
    defer a.free(k);
    const raw = try ownWeight(w, k);
    defer _ = mlx.mlx_array_free(raw);
    return astype(raw, .float32, s);
}
/// Conv weight OIHW -> OHWI, f32.
fn loadConvW(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, s: S) !mlx.mlx_array {
    const wk = try fmtKey(a, "{s}.weight", .{prefix});
    defer a.free(wk);
    const raw = try ownWeight(w, wk);
    defer _ = mlx.mlx_array_free(raw);
    const t = try transpose(raw, &[_]c_int{ 0, 2, 3, 1 }, s);
    defer _ = mlx.mlx_array_free(t);
    const tc = try contig(t, s);
    defer _ = mlx.mlx_array_free(tc);
    return astype(tc, .float32, s);
}

// ── JSON config helpers ──

fn readJson(io: std.Io, a: std.mem.Allocator, path: []const u8) !std.json.Parsed(std.json.Value) {
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var rb: [8192]u8 = undefined;
    var rs = file.reader(io, &rb);
    const content = try rs.interface.allocRemaining(a, .limited(4 * 1024 * 1024));
    defer a.free(content);
    return std.json.parseFromSlice(std.json.Value, a, content, .{});
}
fn objGet(v: std.json.Value, key: []const u8) ?std.json.Value {
    if (v != .object) return null;
    return v.object.get(key);
}
fn getU32(v: std.json.Value, key: []const u8, default: u32) u32 {
    const x = objGet(v, key) orelse return default;
    return switch (x) {
        .integer => |i| if (i >= 0) @intCast(i) else default,
        .float => |f| @intFromFloat(f),
        else => default,
    };
}
fn getF32(v: std.json.Value, key: []const u8, default: f32) f32 {
    const x = objGet(v, key) orelse return default;
    return switch (x) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => default,
    };
}
fn getU32Arr(a: std.mem.Allocator, v: std.json.Value, key: []const u8, default: []const u32) ![]u32 {
    const x = objGet(v, key) orelse return a.dupe(u32, default);
    if (x != .array) return a.dupe(u32, default);
    const out = try a.alloc(u32, x.array.items.len);
    for (x.array.items, 0..) |item, i| out[i] = switch (item) {
        .integer => |n| if (n >= 0) @intCast(n) else 0,
        .float => |f| @intFromFloat(f),
        else => 0,
    };
    return out;
}

// ── Config ──

pub const Config = struct {
    dim: u32 = 3840,
    n_layers: u32 = 30,
    n_refiner_layers: u32 = 2,
    n_heads: u32 = 30,
    n_kv_heads: u32 = 30,
    norm_eps: f32 = 1e-5,
    cap_feat_dim: u32 = 2560,
    rope_theta: f32 = 256.0,
    t_scale: f32 = 1000.0,
    in_channels: u32 = 16,
    patch_size: u32 = 2,
    axes_dims: [3]u32 = .{ 32, 48, 48 },
    // Text encoder (Qwen3) geometry, read straight off `text_encoder/config.json`.
    te_hidden: u32 = 2560,
    te_layers: u32 = 36,
    // Sampling defaults, resolved from the checkpoint dir name (Turbo vs base).
    is_turbo: bool = true,
    default_steps: u32 = 8,
    default_cfg: f32 = 0.0,

    pub fn headDim(self: Config) u32 {
        return self.dim / self.n_heads;
    }
    pub fn ffnHidden(self: Config) u32 {
        return @intCast(@as(u64, self.dim) / 3 * 8);
    }
};

fn dirLooksTurbo(model_dir: []const u8) bool {
    var buf: [512]u8 = undefined;
    const lower = std.ascii.lowerString(&buf, model_dir[0..@min(model_dir.len, buf.len)]);
    return std.mem.indexOf(u8, lower, "turbo") != null;
}

pub fn parseConfig(io: std.Io, a: std.mem.Allocator, model_dir: []const u8) !Config {
    var cfg = Config{};
    cfg.is_turbo = dirLooksTurbo(model_dir);
    if (!cfg.is_turbo) {
        cfg.default_steps = 50;
        cfg.default_cfg = 5.0;
    }
    {
        const path = try fmtKey(a, "{s}/transformer/config.json", .{model_dir});
        defer a.free(path);
        var p = try readJson(io, a, path);
        defer p.deinit();
        const o = p.value;
        cfg.dim = getU32(o, "dim", cfg.dim);
        cfg.n_layers = getU32(o, "n_layers", cfg.n_layers);
        cfg.n_refiner_layers = getU32(o, "n_refiner_layers", cfg.n_refiner_layers);
        cfg.n_heads = getU32(o, "n_heads", cfg.n_heads);
        cfg.n_kv_heads = getU32(o, "n_kv_heads", cfg.n_kv_heads);
        cfg.norm_eps = getF32(o, "norm_eps", cfg.norm_eps);
        cfg.cap_feat_dim = getU32(o, "cap_feat_dim", cfg.cap_feat_dim);
        cfg.rope_theta = getF32(o, "rope_theta", cfg.rope_theta);
        cfg.t_scale = getF32(o, "t_scale", cfg.t_scale);
        cfg.in_channels = getU32(o, "in_channels", cfg.in_channels);
        const axes = try getU32Arr(a, o, "axes_dims", &.{ 32, 48, 48 });
        defer a.free(axes);
        if (axes.len == 3) cfg.axes_dims = .{ axes[0], axes[1], axes[2] };
        const patches = try getU32Arr(a, o, "all_patch_size", &.{2});
        defer a.free(patches);
        if (patches.len > 0) cfg.patch_size = patches[0];
    }
    {
        const path = try fmtKey(a, "{s}/text_encoder/config.json", .{model_dir});
        defer a.free(path);
        var p = try readJson(io, a, path);
        defer p.deinit();
        const o = p.value;
        cfg.te_hidden = getU32(o, "hidden_size", cfg.te_hidden);
        cfg.te_layers = getU32(o, "num_hidden_layers", cfg.te_layers);
    }
    return cfg;
}

// ── Text encoder: the checkpoint's Qwen3 LM, captured pre-final-layer ──
// (diffusers: `hidden_states[-2]`, i.e. the residual AFTER layer index
// `n_layers-2`, skipping the last transformer layer AND the final norm —
// exactly `ltx_video.gemmaCapture4`'s pattern, one captured layer instead
// of all of them.)

pub const TextEncoder = struct {
    allocator: std.mem.Allocator,
    tok: tok_mod.Tokenizer,
    // `Transformer` binds views into `weights`' arrays rather than deep
    // copies (every other caller in this codebase keeps its `Weights` alive
    // for as long as it uses the resulting `Transformer` — see
    // `main.zig`'s offline path, `dflash.zig`, `diffusion.zig` — a
    // `weights.deinit()` right after `init` frees memory `xfm` still reads
    // on the NEXT forward, corrupting the embedding gather with a SIGSEGV
    // that only shows up later, once this TextEncoder actually encodes a
    // prompt). Freed together with `xfm` in `deinit`.
    weights: Weights,
    xfm: transformer_mod.Transformer,
    cap_layer_id: u32,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !TextEncoder {
        const te_dir = try fmtKey(allocator, "{s}/text_encoder", .{model_dir});
        defer allocator.free(te_dir);
        // The tokenizer (BPE + chat template) ships in a sibling `tokenizer/` dir.
        const tok_dir = try fmtKey(allocator, "{s}/tokenizer", .{model_dir});
        defer allocator.free(tok_dir);
        var tok = tok_mod.loadTokenizerAny(io, allocator, tok_dir) catch |e| {
            log.err("[zimage] failed to load tokenizer from {s}: {s}\n", .{ tok_dir, @errorName(e) });
            return e;
        };
        errdefer tok.deinit();

        var config = try model_mod.parseConfig(io, allocator, te_dir);
        var weights = try model_mod.loadWeights(io, allocator, te_dir);
        errdefer weights.deinit();
        model_mod.resolveWeightPrefix(&config, &weights);
        var xfm = try transformer_mod.Transformer.init(io, allocator, config, &weights);
        errdefer xfm.deinit();
        const cap_layer_id: u32 = if (config.num_hidden_layers >= 2) config.num_hidden_layers - 2 else 0;

        return .{ .allocator = allocator, .tok = tok, .weights = weights, .xfm = xfm, .cap_layer_id = cap_layer_id };
    }

    pub fn deinit(self: *TextEncoder) void {
        self.xfm.deinit();
        self.weights.deinit();
        self.tok.deinit();
    }

    /// Qwen3 chat template with no system message, no tools, a single user
    /// turn, `add_generation_prompt=True` — the exact rendering of the
    /// checkpoint's own `tokenizer/tokenizer_config.json` `chat_template`
    /// for this shape of input (verified against the real template source).
    fn formatPrompt(allocator: std.mem.Allocator, prompt: []const u8) ![]u8 {
        return std.fmt.allocPrint(allocator, "<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n", .{prompt});
    }

    /// Encode `prompt` -> caption feature `[Lc, cap_feat_dim]` bf16 (owned).
    /// Captures `hidden_states[-2]` — the residual after layer `n_layers-2`,
    /// skipping the last transformer layer AND the final norm — exactly
    /// `ltx_video.gemmaCapture4`'s pattern, one captured layer.
    pub fn encode(self: *TextEncoder, s: S, prompt: []const u8) !mlx.mlx_array {
        const allocator = self.allocator;
        const formatted = try formatPrompt(allocator, prompt);
        defer allocator.free(formatted);
        const enc = try self.tok.encode(allocator, formatted);
        defer allocator.free(enc);
        const ids = try allocator.alloc(i32, enc.len);
        defer allocator.free(ids);
        for (enc, 0..) |t, i| ids[i] = @intCast(t);
        {
            var min_id: u32 = std.math.maxInt(u32);
            var max_id: u32 = 0;
            for (enc) |t| {
                if (t < min_id) min_id = t;
                if (t > max_id) max_id = t;
            }
            log.info("[zimage] encoded {d} tokens, id range [{d}, {d}]\n", .{ enc.len, min_id, max_id });
        }

        var cap_ids = [_]u32{self.cap_layer_id};
        var out = [_]mlx.mlx_array{mlx.mlx_array_new()};
        var cl = transformer_mod.CaptureLayers{ .ids = &cap_ids, .out = &out };

        const id_shape = [_]c_int{ 1, @intCast(ids.len) };
        const ids_arr = mlx.mlx_array_new_data(ids.ptr, &id_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(ids_arr);

        var ctx = self.xfm.defaultCtx();
        ctx.capture_layers = &cl;
        ctx.skip_lm_head = true;
        const discard = try self.xfm.forwardWith(&ctx, ids_arr);
        _ = mlx.mlx_array_free(discard);

        if (out[0].ctx == null) return error.ZImageCaptureUnavailable;
        _ = mlx.mlx_array_eval(out[0]);
        // out[0] is [1, L, te_hidden]; drop the batch dim -> [L, te_hidden] bf16.
        const sh = mlx.getShape(out[0]);
        const squeezed = try reshape(out[0], &[_]c_int{ sh[1], sh[2] }, s);
        _ = mlx.mlx_array_free(out[0]);
        return astype(squeezed, .bfloat16, s);
    }
};

// ── VAE decoder: the ORIGINAL flux-dev 16-channel AutoencoderKL (dense f32,
// no quant_conv, no post_quant_conv) — NOT `flux.zig`'s FLUX.2 32-channel
// bn-wrapped one. `_name_or_path: "flux-dev"` in the shipped vae/config.json
// confirms it's byte-shaped like flux-dev's own VAE. ──

const VAE_SCALING_FACTOR: f32 = 0.3611;
const VAE_SHIFT_FACTOR: f32 = 0.1159;
const VAE_BLOCK_CHANNELS = [4]u32{ 128, 256, 512, 512 };

fn conv2dP(x: mlx.mlx_array, w: mlx.mlx_array, bias: mlx.mlx_array, pad: c_int, s: S) !mlx.mlx_array {
    var xc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xc);
    try mlx.check(mlx.mlx_contiguous(&xc, x, false, s));
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv2d(&o, xc, w, 1, 1, pad, pad, 1, 1, 1, s));
    const r = try addA(o, bias, s);
    _ = mlx.mlx_array_free(o);
    return r;
}
fn groupNorm32(x: mlx.mlx_array, weight: mlx.mlx_array, bias: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [1,H,W,C]
    const H = sh[1];
    const Wd = sh[2];
    const C = sh[3];
    const groups: c_int = 32;
    const cg = @divExact(C, groups);
    const xf = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    const r1 = try reshape(xf, &[_]c_int{ 1, H * Wd, groups, cg }, s);
    defer _ = mlx.mlx_array_free(r1);
    const t1 = try transpose(r1, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(t1);
    const flat = try reshape(t1, &[_]c_int{ 1, groups, H * Wd * cg }, s);
    defer _ = mlx.mlx_array_free(flat);
    var mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mean);
    try mlx.check(mlx.mlx_mean_axis(&mean, flat, -1, true, s));
    const xc = try subA(flat, mean, s);
    defer _ = mlx.mlx_array_free(xc);
    const sq = try mulA(xc, xc, s);
    defer _ = mlx.mlx_array_free(sq);
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_mean_axis(&v, sq, -1, true, s));
    const epsa = scalarF(1e-6);
    defer _ = mlx.mlx_array_free(epsa);
    var ve = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ve);
    try mlx.check(mlx.mlx_add(&ve, v, epsa, s));
    var rsq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rsq);
    try mlx.check(mlx.mlx_rsqrt(&rsq, ve, s));
    const norm = try mulA(xc, rsq, s);
    defer _ = mlx.mlx_array_free(norm);
    const b1 = try reshape(norm, &[_]c_int{ 1, groups, H * Wd, cg }, s);
    defer _ = mlx.mlx_array_free(b1);
    const b2 = try transpose(b1, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(b2);
    const b3 = try reshape(b2, &[_]c_int{ 1, H, Wd, C }, s);
    defer _ = mlx.mlx_array_free(b3);
    const sc = try mulA(b3, weight, s);
    defer _ = mlx.mlx_array_free(sc);
    return addA(sc, bias, s);
}

const VaeResnet = struct {
    n1w: mlx.mlx_array,
    n1b: mlx.mlx_array,
    c1w: mlx.mlx_array,
    c1b: mlx.mlx_array,
    n2w: mlx.mlx_array,
    n2b: mlx.mlx_array,
    c2w: mlx.mlx_array,
    c2b: mlx.mlx_array,
    sw: ?mlx.mlx_array = null,
    sb: ?mlx.mlx_array = null,

    fn deinit(self: *VaeResnet) void {
        inline for (.{ "n1w", "n1b", "c1w", "c1b", "n2w", "n2b", "c2w", "c2b" }) |f| _ = mlx.mlx_array_free(@field(self, f));
        if (self.sw) |x| _ = mlx.mlx_array_free(x);
        if (self.sb) |x| _ = mlx.mlx_array_free(x);
    }
    fn forward(self: *const VaeResnet, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const h0 = try groupNorm32(x, self.n1w, self.n1b, s);
        defer _ = mlx.mlx_array_free(h0);
        const a0 = try silu(h0, s);
        defer _ = mlx.mlx_array_free(a0);
        const c1 = try conv2dP(a0, self.c1w, self.c1b, 1, s);
        defer _ = mlx.mlx_array_free(c1);
        const h1 = try groupNorm32(c1, self.n2w, self.n2b, s);
        defer _ = mlx.mlx_array_free(h1);
        const a1 = try silu(h1, s);
        defer _ = mlx.mlx_array_free(a1);
        const c2 = try conv2dP(a1, self.c2w, self.c2b, 1, s);
        defer _ = mlx.mlx_array_free(c2);
        if (self.sw) |sw| {
            const sc = try conv2dP(x, sw, self.sb.?, 0, s);
            defer _ = mlx.mlx_array_free(sc);
            return addA(c2, sc, s);
        }
        return addA(c2, x, s);
    }
};

const VaeAttn = struct {
    gnw: mlx.mlx_array,
    gnb: mlx.mlx_array,
    qw: mlx.mlx_array,
    qb: mlx.mlx_array,
    kw: mlx.mlx_array,
    kb: mlx.mlx_array,
    vw: mlx.mlx_array,
    vb: mlx.mlx_array,
    ow: mlx.mlx_array,
    ob: mlx.mlx_array,

    fn deinit(self: *VaeAttn) void {
        inline for (.{ "gnw", "gnb", "qw", "qb", "kw", "kb", "vw", "vb", "ow", "ob" }) |f| _ = mlx.mlx_array_free(@field(self, f));
    }
    fn forward(self: *const VaeAttn, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const sh = mlx.getShape(x);
        const H = sh[1];
        const Wd = sh[2];
        const C = sh[3];
        const normed = try groupNorm32(x, self.gnw, self.gnb, s);
        defer _ = mlx.mlx_array_free(normed);
        const flat = try reshape(normed, &[_]c_int{ 1, H * Wd, C }, s);
        defer _ = mlx.mlx_array_free(flat);
        const q = try matmulBias(flat, self.qw, self.qb, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try matmulBias(flat, self.kw, self.kb, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try matmulBias(flat, self.vw, self.vb, s);
        defer _ = mlx.mlx_array_free(v);
        const qr = try reshape(q, &[_]c_int{ 1, 1, H * Wd, C }, s);
        defer _ = mlx.mlx_array_free(qr);
        const kr = try reshape(k, &[_]c_int{ 1, 1, H * Wd, C }, s);
        defer _ = mlx.mlx_array_free(kr);
        const vr = try reshape(v, &[_]c_int{ 1, 1, H * Wd, C }, s);
        defer _ = mlx.mlx_array_free(vr);
        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(C)));
        var attn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn);
        const null_a = mlx.mlx_array{ .ctx = null };
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qr, kr, vr, scale, "", null_a, null_a, false, s));
        const ar = try reshape(attn, &[_]c_int{ 1, H, Wd, C }, s);
        defer _ = mlx.mlx_array_free(ar);
        const ao = try matmulBias(ar, self.ow, self.ob, s);
        defer _ = mlx.mlx_array_free(ao);
        return addA(x, ao, s);
    }
};

fn matmulBias(x: mlx.mlx_array, w_t: mlx.mlx_array, bias: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&o, x, w_t, s));
    const r = try addA(o, bias, s);
    _ = mlx.mlx_array_free(o);
    return r;
}

pub const Vae = struct {
    allocator: std.mem.Allocator,
    s: S,
    conv_in_w: mlx.mlx_array,
    conv_in_b: mlx.mlx_array,
    mid_r0: VaeResnet,
    mid_attn: VaeAttn,
    mid_r1: VaeResnet,
    up_resnets: [4][3]VaeResnet,
    up_conv_w: [3]mlx.mlx_array,
    up_conv_b: [3]mlx.mlx_array,
    norm_out_w: mlx.mlx_array,
    norm_out_b: mlx.mlx_array,
    conv_out_w: mlx.mlx_array,
    conv_out_b: mlx.mlx_array,

    pub fn deinit(self: *Vae) void {
        inline for (.{ "conv_in_w", "conv_in_b", "norm_out_w", "norm_out_b", "conv_out_w", "conv_out_b" }) |f| _ = mlx.mlx_array_free(@field(self, f));
        self.mid_r0.deinit();
        self.mid_attn.deinit();
        self.mid_r1.deinit();
        for (&self.up_resnets) |*blk| for (blk) |*r| r.deinit();
        for (0..3) |i| {
            _ = mlx.mlx_array_free(self.up_conv_w[i]);
            _ = mlx.mlx_array_free(self.up_conv_b[i]);
        }
    }

    /// `latents` [1,16,h,w] -> image [1,3,h*8,w*8] f32 [-1,1] (undenormalized).
    pub fn decode(self: *Vae, latents: mlx.mlx_array) !mlx.mlx_array {
        const s = self.s;
        const lf = try astype(latents, .float32, s);
        defer _ = mlx.mlx_array_free(lf);
        const inv_scale = scalarF(1.0 / VAE_SCALING_FACTOR);
        defer _ = mlx.mlx_array_free(inv_scale);
        const shift = scalarF(VAE_SHIFT_FACTOR);
        defer _ = mlx.mlx_array_free(shift);
        const scaled = try mulA(lf, inv_scale, s);
        defer _ = mlx.mlx_array_free(scaled);
        const unshifted = try addA(scaled, shift, s);
        defer _ = mlx.mlx_array_free(unshifted);
        const nhwc = try transpose(unshifted, &[_]c_int{ 0, 2, 3, 1 }, s);
        defer _ = mlx.mlx_array_free(nhwc);
        var h = try conv2dP(nhwc, self.conv_in_w, self.conv_in_b, 1, s);
        { const nh = try self.mid_r0.forward(h, s); _ = mlx.mlx_array_free(h); h = nh; }
        { const nh = try self.mid_attn.forward(h, s); _ = mlx.mlx_array_free(h); h = nh; }
        { const nh = try self.mid_r1.forward(h, s); _ = mlx.mlx_array_free(h); h = nh; }
        for (0..4) |bi| {
            for (0..3) |ri| {
                const nh = try self.up_resnets[bi][ri].forward(h, s);
                _ = mlx.mlx_array_free(h);
                h = nh;
            }
            if (bi < 3) {
                var r1 = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_repeat_axis(&r1, h, 2, 1, s));
                _ = mlx.mlx_array_free(h);
                var r2 = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_repeat_axis(&r2, r1, 2, 2, s));
                _ = mlx.mlx_array_free(r1);
                h = try conv2dP(r2, self.up_conv_w[bi], self.up_conv_b[bi], 1, s);
                _ = mlx.mlx_array_free(r2);
            }
        }
        { const nh = try groupNorm32(h, self.norm_out_w, self.norm_out_b, s); _ = mlx.mlx_array_free(h); h = nh; }
        { const nh = try silu(h, s); _ = mlx.mlx_array_free(h); h = nh; }
        { const nh = try conv2dP(h, self.conv_out_w, self.conv_out_b, 1, s); _ = mlx.mlx_array_free(h); h = nh; }
        const out = try transpose(h, &[_]c_int{ 0, 3, 1, 2 }, s);
        _ = mlx.mlx_array_free(h);
        defer _ = mlx.mlx_array_free(out);
        return contig(out, s);
    }
};

fn loadVaeResnetProper(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, s: S) !VaeResnet {
    var r: VaeResnet = undefined;
    r.n1w = try loadVec(w, a, pfx, "norm1.weight", s);
    r.n1b = try loadVec(w, a, pfx, "norm1.bias", s);
    r.c1w = try loadConvW(w, a, try fmtKey(a, "{s}.conv1", .{pfx}), s);
    r.c1b = try loadVec(w, a, pfx, "conv1.bias", s);
    r.n2w = try loadVec(w, a, pfx, "norm2.weight", s);
    r.n2b = try loadVec(w, a, pfx, "norm2.bias", s);
    r.c2w = try loadConvW(w, a, try fmtKey(a, "{s}.conv2", .{pfx}), s);
    r.c2b = try loadVec(w, a, pfx, "conv2.bias", s);
    const swk = try fmtKey(a, "{s}.conv_shortcut.weight", .{pfx});
    defer a.free(swk);
    if (w.get(swk) != null) {
        r.sw = try loadConvW(w, a, try fmtKey(a, "{s}.conv_shortcut", .{pfx}), s);
        r.sb = try loadVec(w, a, pfx, "conv_shortcut.bias", s);
    } else {
        r.sw = null;
        r.sb = null;
    }
    return r;
}

pub fn loadVae(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8) !Vae {
    const dir = try fmtKey(allocator, "{s}/vae", .{model_dir});
    defer allocator.free(dir);
    var w = try model_mod.loadWeights(io, allocator, dir);
    defer w.deinit();
    var v: Vae = undefined;
    v.allocator = allocator;
    v.s = s;
    v.conv_in_w = try loadConvW(&w, allocator, "decoder.conv_in", s);
    v.conv_in_b = try loadVec(&w, allocator, "decoder.conv_in", "bias", s);
    v.mid_r0 = try loadVaeResnetProper(&w, allocator, "decoder.mid_block.resnets.0", s);
    v.mid_r1 = try loadVaeResnetProper(&w, allocator, "decoder.mid_block.resnets.1", s);
    v.mid_attn = .{
        .gnw = try loadVec(&w, allocator, "decoder.mid_block.attentions.0.group_norm", "weight", s),
        .gnb = try loadVec(&w, allocator, "decoder.mid_block.attentions.0.group_norm", "bias", s),
        .qw = try loadLinT(&w, allocator, "decoder.mid_block.attentions.0.to_q", s),
        .qb = try loadVec(&w, allocator, "decoder.mid_block.attentions.0.to_q", "bias", s),
        .kw = try loadLinT(&w, allocator, "decoder.mid_block.attentions.0.to_k", s),
        .kb = try loadVec(&w, allocator, "decoder.mid_block.attentions.0.to_k", "bias", s),
        .vw = try loadLinT(&w, allocator, "decoder.mid_block.attentions.0.to_v", s),
        .vb = try loadVec(&w, allocator, "decoder.mid_block.attentions.0.to_v", "bias", s),
        .ow = try loadLinT(&w, allocator, "decoder.mid_block.attentions.0.to_out.0", s),
        .ob = try loadVec(&w, allocator, "decoder.mid_block.attentions.0.to_out.0", "bias", s),
    };
    for (0..4) |bi| {
        for (0..3) |ri| {
            const pfx = try fmtKey(allocator, "decoder.up_blocks.{d}.resnets.{d}", .{ bi, ri });
            defer allocator.free(pfx);
            v.up_resnets[bi][ri] = try loadVaeResnetProper(&w, allocator, pfx, s);
        }
    }
    for (0..3) |bi| {
        const pfx = try fmtKey(allocator, "decoder.up_blocks.{d}.upsamplers.0.conv", .{bi});
        defer allocator.free(pfx);
        v.up_conv_w[bi] = try loadConvW(&w, allocator, pfx, s);
        v.up_conv_b[bi] = try loadVec(&w, allocator, pfx, "bias", s);
    }
    v.norm_out_w = try loadVec(&w, allocator, "decoder.conv_norm_out", "weight", s);
    v.norm_out_b = try loadVec(&w, allocator, "decoder.conv_norm_out", "bias", s);
    v.conv_out_w = try loadConvW(&w, allocator, "decoder.conv_out", s);
    v.conv_out_b = try loadVec(&w, allocator, "decoder.conv_out", "bias", s);
    return v;
}

// ── RoPE: 3-axis interleaved-complex, computed directly on the host (no
// gather — every position id used is known at call time). ──

/// Per-token cos/sin table `[seq, sum(axes_dims)/2]` for the given per-token
/// 3D position ids, bf16 mlx arrays. `theta` is shared across axes (matches
/// `RopeEmbedder(theta=256, axes_dims, axes_lens)`; axes_lens is a cache-size
/// bound in the reference, not part of the math).
fn buildRopeTable(allocator: std.mem.Allocator, pos: []const [3]i32, axes_dims: [3]u32, theta: f32, s: S) !struct { cos: mlx.mlx_array, sin: mlx.mlx_array } {
    const half: usize = (axes_dims[0] + axes_dims[1] + axes_dims[2]) / 2;
    const n = pos.len;
    const cos_h = try allocator.alloc(f32, n * half);
    defer allocator.free(cos_h);
    const sin_h = try allocator.alloc(f32, n * half);
    defer allocator.free(sin_h);
    var axis_off: usize = 0;
    for (axes_dims, 0..) |d, axis| {
        const pairs = d / 2;
        for (0..pairs) |j| {
            const inv_freq = std.math.pow(f64, @floatCast(theta), -@as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(d)));
            for (pos, 0..) |p, ti| {
                const angle = @as(f64, @floatFromInt(p[axis])) * inv_freq;
                cos_h[ti * half + axis_off + j] = @floatCast(@cos(angle));
                sin_h[ti * half + axis_off + j] = @floatCast(@sin(angle));
            }
        }
        axis_off += pairs;
    }
    const shape = [_]c_int{ 1, @intCast(n), 1, @intCast(half) };
    const cos_raw = mlx.mlx_array_new_data(cos_h.ptr, &shape, 4, .float32);
    const sin_raw = mlx.mlx_array_new_data(sin_h.ptr, &shape, 4, .float32);
    const cos_bf = try astype(cos_raw, .bfloat16, s);
    _ = mlx.mlx_array_free(cos_raw);
    const sin_bf = try astype(sin_raw, .bfloat16, s);
    _ = mlx.mlx_array_free(sin_raw);
    return .{ .cos = cos_bf, .sin = sin_bf };
}

/// Interleaved-pair (torch `view_as_complex`) RoPE on `x` `[1,seq,heads,hd]`,
/// `cos`/`sin` `[1,seq,1,hd/2]` (broadcast over heads).
fn applyRope(x: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [1,seq,heads,hd]
    const half = @divExact(sh[3], 2);
    const r5 = try reshape(x, &[_]c_int{ sh[0], sh[1], sh[2], half, 2 }, s);
    defer _ = mlx.mlx_array_free(r5);
    const xr5 = try sliceLastIdx(r5, 0, s);
    defer _ = mlx.mlx_array_free(xr5);
    const xi5 = try sliceLastIdx(r5, 1, s);
    defer _ = mlx.mlx_array_free(xi5);
    const xr = try reshape(xr5, &[_]c_int{ sh[0], sh[1], sh[2], half }, s);
    defer _ = mlx.mlx_array_free(xr);
    const xi = try reshape(xi5, &[_]c_int{ sh[0], sh[1], sh[2], half }, s);
    defer _ = mlx.mlx_array_free(xi);
    const xr_cos = try mulA(xr, cos, s);
    defer _ = mlx.mlx_array_free(xr_cos);
    const xi_sin = try mulA(xi, sin, s);
    defer _ = mlx.mlx_array_free(xi_sin);
    const out_r = try subA(xr_cos, xi_sin, s);
    defer _ = mlx.mlx_array_free(out_r);
    const xr_sin = try mulA(xr, sin, s);
    defer _ = mlx.mlx_array_free(xr_sin);
    const xi_cos = try mulA(xi, cos, s);
    defer _ = mlx.mlx_array_free(xi_cos);
    const out_i = try addA(xr_sin, xi_cos, s);
    defer _ = mlx.mlx_array_free(out_i);
    const out_r5 = try reshape(out_r, &[_]c_int{ sh[0], sh[1], sh[2], half, 1 }, s);
    defer _ = mlx.mlx_array_free(out_r5);
    const out_i5 = try reshape(out_i, &[_]c_int{ sh[0], sh[1], sh[2], half, 1 }, s);
    defer _ = mlx.mlx_array_free(out_i5);
    const stacked = try concat(&.{ out_r5, out_i5 }, 4, s);
    defer _ = mlx.mlx_array_free(stacked);
    const stacked_c = try contig(stacked, s);
    defer _ = mlx.mlx_array_free(stacked_c);
    return reshape(stacked_c, &[_]c_int{ sh[0], sh[1], sh[2], sh[3] }, s);
}

// ── DiT ──

const Attn = struct {
    to_q: MfLinear,
    to_k: MfLinear,
    to_v: MfLinear,
    to_out: MfLinear,
    norm_q: mlx.mlx_array,
    norm_k: mlx.mlx_array,

    fn deinit(self: *Attn) void {
        self.to_q.deinit();
        self.to_k.deinit();
        self.to_v.deinit();
        self.to_out.deinit();
        _ = mlx.mlx_array_free(self.norm_q);
        _ = mlx.mlx_array_free(self.norm_k);
    }

    fn forward(self: *const Attn, x: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, n_heads: u32, head_dim: u32, eps: f32, s: S) !mlx.mlx_array {
        const sh = mlx.getShape(x); // [1,seq,dim]
        const seq = sh[1];
        const q = try self.to_q.forward(x, null, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try self.to_k.forward(x, null, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try self.to_v.forward(x, null, s);
        defer _ = mlx.mlx_array_free(v);
        const hshape = [_]c_int{ 1, seq, @intCast(n_heads), @intCast(head_dim) };
        const qh = try reshape(q, &hshape, s);
        defer _ = mlx.mlx_array_free(qh);
        const kh = try reshape(k, &hshape, s);
        defer _ = mlx.mlx_array_free(kh);
        const vh = try reshape(v, &hshape, s);
        defer _ = mlx.mlx_array_free(vh);
        const qn = try rms(qh, self.norm_q, eps, s);
        defer _ = mlx.mlx_array_free(qn);
        const kn = try rms(kh, self.norm_k, eps, s);
        defer _ = mlx.mlx_array_free(kn);
        const qr = try applyRope(qn, cos, sin, s);
        defer _ = mlx.mlx_array_free(qr);
        const kr = try applyRope(kn, cos, sin, s);
        defer _ = mlx.mlx_array_free(kr);
        const qt = try transpose(qr, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(qt);
        const kt = try transpose(kr, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(kt);
        const vt = try transpose(vh, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(vt);
        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_dim)));
        var attn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn);
        const null_a = mlx.mlx_array{ .ctx = null };
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qt, kt, vt, scale, "", null_a, null_a, false, s));
        const ao = try transpose(attn, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(ao);
        const ao_c = try contig(ao, s);
        defer _ = mlx.mlx_array_free(ao_c);
        const flat = try reshape(ao_c, &[_]c_int{ 1, seq, @as(c_int, @intCast(n_heads)) * @as(c_int, @intCast(head_dim)) }, s);
        defer _ = mlx.mlx_array_free(flat);
        return self.to_out.forward(flat, null, s);
    }
};

const FeedForward = struct {
    w1: MfLinear,
    w2: MfLinear,
    w3: MfLinear,

    fn deinit(self: *FeedForward) void {
        self.w1.deinit();
        self.w2.deinit();
        self.w3.deinit();
    }
    fn forward(self: *const FeedForward, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const g = try self.w1.forward(x, null, s);
        defer _ = mlx.mlx_array_free(g);
        const gs = try silu(g, s);
        defer _ = mlx.mlx_array_free(gs);
        const u = try self.w3.forward(x, null, s);
        defer _ = mlx.mlx_array_free(u);
        const gated = try mulA(gs, u, s);
        defer _ = mlx.mlx_array_free(gated);
        return self.w2.forward(gated, null, s);
    }
};

/// A `ZImageTransformerBlock`. `modulated` blocks (main layers +
/// noise_refiner) carry `adaLN_modulation`; `context_refiner` blocks don't.
const Block = struct {
    attn: Attn,
    ffn: FeedForward,
    attn_norm1: mlx.mlx_array,
    attn_norm2: mlx.mlx_array,
    ffn_norm1: mlx.mlx_array,
    ffn_norm2: mlx.mlx_array,
    ada_w: ?MfLinear = null,
    ada_b: ?mlx.mlx_array = null,

    fn deinit(self: *Block) void {
        self.attn.deinit();
        self.ffn.deinit();
        inline for (.{ "attn_norm1", "attn_norm2", "ffn_norm1", "ffn_norm2" }) |f| _ = mlx.mlx_array_free(@field(self, f));
        if (self.ada_w) |*w| w.deinit();
        if (self.ada_b) |b| _ = mlx.mlx_array_free(b);
    }

    fn forward(self: *const Block, x: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, ada_input: ?mlx.mlx_array, cfg: Config, s: S) !mlx.mlx_array {
        if (self.ada_w) |*aw| {
            const mod_raw = try aw.forward(ada_input.?, self.ada_b, s);
            defer _ = mlx.mlx_array_free(mod_raw);
            const dim: c_int = @intCast(cfg.dim);
            // mod_raw is [1,4*dim]; unsqueeze to [1,1,4*dim] then split into 4 chunks of `dim`
            // (order: scale_msa, gate_msa, scale_mlp, gate_mlp — `mod.unsqueeze(1).chunk(4,dim=2)`).
            const mod3 = try reshape(mod_raw, &[_]c_int{ 1, 1, 4 * dim }, s);
            defer _ = mlx.mlx_array_free(mod3);
            var chunks: [4]mlx.mlx_array = undefined;
            for (0..4) |ci| chunks[ci] = try sliceLast4(mod3, dim, @intCast(ci), s);
            defer {
                for (chunks) |c| _ = mlx.mlx_array_free(c);
            }
            const one = scalarF(1.0);
            defer _ = mlx.mlx_array_free(one);
            const gate_msa = try tanhA(chunks[1], s);
            defer _ = mlx.mlx_array_free(gate_msa);
            const gate_mlp = try tanhA(chunks[3], s);
            defer _ = mlx.mlx_array_free(gate_mlp);
            const scale_msa_v = try addA(chunks[0], one, s);
            defer _ = mlx.mlx_array_free(scale_msa_v);
            const scale_mlp_v = try addA(chunks[2], one, s);
            defer _ = mlx.mlx_array_free(scale_mlp_v);

            const n1 = try rms(x, self.attn_norm1, cfg.norm_eps, s);
            defer _ = mlx.mlx_array_free(n1);
            const n1s = try mulA(n1, scale_msa_v, s);
            defer _ = mlx.mlx_array_free(n1s);
            const attn_out = try self.attn.forward(n1s, cos, sin, cfg.n_heads, cfg.headDim(), cfg.norm_eps, s);
            defer _ = mlx.mlx_array_free(attn_out);
            const attn_n2 = try rms(attn_out, self.attn_norm2, cfg.norm_eps, s);
            defer _ = mlx.mlx_array_free(attn_n2);
            const attn_g = try mulA(gate_msa, attn_n2, s);
            defer _ = mlx.mlx_array_free(attn_g);
            const x1 = try addA(x, attn_g, s);
            errdefer _ = mlx.mlx_array_free(x1);

            const n2 = try rms(x1, self.ffn_norm1, cfg.norm_eps, s);
            defer _ = mlx.mlx_array_free(n2);
            const n2s = try mulA(n2, scale_mlp_v, s);
            defer _ = mlx.mlx_array_free(n2s);
            const ffn_out = try self.ffn.forward(n2s, s);
            defer _ = mlx.mlx_array_free(ffn_out);
            const ffn_n2 = try rms(ffn_out, self.ffn_norm2, cfg.norm_eps, s);
            defer _ = mlx.mlx_array_free(ffn_n2);
            const ffn_g = try mulA(gate_mlp, ffn_n2, s);
            defer _ = mlx.mlx_array_free(ffn_g);
            const x2 = try addA(x1, ffn_g, s);
            _ = mlx.mlx_array_free(x1);
            return x2;
        } else {
            const n1 = try rms(x, self.attn_norm1, cfg.norm_eps, s);
            defer _ = mlx.mlx_array_free(n1);
            const attn_out = try self.attn.forward(n1, cos, sin, cfg.n_heads, cfg.headDim(), cfg.norm_eps, s);
            defer _ = mlx.mlx_array_free(attn_out);
            const attn_n2 = try rms(attn_out, self.attn_norm2, cfg.norm_eps, s);
            defer _ = mlx.mlx_array_free(attn_n2);
            const x1 = try addA(x, attn_n2, s);
            errdefer _ = mlx.mlx_array_free(x1);

            const n2 = try rms(x1, self.ffn_norm1, cfg.norm_eps, s);
            defer _ = mlx.mlx_array_free(n2);
            const ffn_out = try self.ffn.forward(n2, s);
            defer _ = mlx.mlx_array_free(ffn_out);
            const ffn_n2 = try rms(ffn_out, self.ffn_norm2, cfg.norm_eps, s);
            defer _ = mlx.mlx_array_free(ffn_n2);
            const x2 = try addA(x1, ffn_n2, s);
            _ = mlx.mlx_array_free(x1);
            return x2;
        }
    }
};

/// Slice a `[1,1,4*dim]` array's last axis to chunk `ci` of width `dim`,
/// returned as `[1,1,dim]`.
fn sliceLast4(x: mlx.mlx_array, dim: c_int, ci: c_int, s: S) !mlx.mlx_array {
    const lo = [_]c_int{ 0, 0, ci * dim };
    const hi = [_]c_int{ 1, 1, (ci + 1) * dim };
    const st = [_]c_int{ 1, 1, 1 };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, &lo, 3, &hi, 3, &st, 3, s));
    return o;
}

fn loadAttn(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, dim: u32, dtype: mlx.mlx_dtype, s: S) !Attn {
    const p = try fmtKey(a, "{s}.attention", .{pfx});
    defer a.free(p);
    return .{
        .to_q = try MfLinear.load(w, a, try fmtKey(a, "{s}.to_q", .{p}), dim, dtype, s),
        .to_k = try MfLinear.load(w, a, try fmtKey(a, "{s}.to_k", .{p}), dim, dtype, s),
        .to_v = try MfLinear.load(w, a, try fmtKey(a, "{s}.to_v", .{p}), dim, dtype, s),
        .to_out = try MfLinear.load(w, a, try fmtKey(a, "{s}.to_out.0", .{p}), dim, dtype, s),
        .norm_q = try astype(try ownWeight(w, try fmtKey(a, "{s}.norm_q.weight", .{p})), dtype, s),
        .norm_k = try astype(try ownWeight(w, try fmtKey(a, "{s}.norm_k.weight", .{p})), dtype, s),
    };
}

fn loadFfn(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, dim: u32, hidden: u32, dtype: mlx.mlx_dtype, s: S) !FeedForward {
    const p = try fmtKey(a, "{s}.feed_forward", .{pfx});
    defer a.free(p);
    return .{
        .w1 = try MfLinear.load(w, a, try fmtKey(a, "{s}.w1", .{p}), dim, dtype, s),
        .w3 = try MfLinear.load(w, a, try fmtKey(a, "{s}.w3", .{p}), dim, dtype, s),
        .w2 = try MfLinear.load(w, a, try fmtKey(a, "{s}.w2", .{p}), hidden, dtype, s),
    };
}

fn loadBlock(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, cfg: Config, modulated: bool, dtype: mlx.mlx_dtype, s: S) !Block {
    var b: Block = undefined;
    b.attn = try loadAttn(w, a, pfx, cfg.dim, dtype, s);
    b.ffn = try loadFfn(w, a, pfx, cfg.dim, cfg.ffnHidden(), dtype, s);
    b.attn_norm1 = try astype(try ownWeight(w, try fmtKey(a, "{s}.attention_norm1.weight", .{pfx})), dtype, s);
    b.attn_norm2 = try astype(try ownWeight(w, try fmtKey(a, "{s}.attention_norm2.weight", .{pfx})), dtype, s);
    b.ffn_norm1 = try astype(try ownWeight(w, try fmtKey(a, "{s}.ffn_norm1.weight", .{pfx})), dtype, s);
    b.ffn_norm2 = try astype(try ownWeight(w, try fmtKey(a, "{s}.ffn_norm2.weight", .{pfx})), dtype, s);
    if (modulated) {
        b.ada_w = try MfLinear.load(w, a, try fmtKey(a, "{s}.adaLN_modulation.0", .{pfx}), 256, dtype, s);
        b.ada_b = try astype(try ownWeight(w, try fmtKey(a, "{s}.adaLN_modulation.0.bias", .{pfx})), dtype, s);
    } else {
        b.ada_w = null;
        b.ada_b = null;
    }
    return b;
}

pub const Dit = struct {
    allocator: std.mem.Allocator,
    s: S,
    cfg: Config,
    dtype: mlx.mlx_dtype,
    x_embed_w: MfLinear,
    x_embed_b: mlx.mlx_array,
    x_pad_token: mlx.mlx_array,
    cap_embed_norm: mlx.mlx_array,
    cap_embed_w: MfLinear,
    cap_embed_b: mlx.mlx_array,
    cap_pad_token: mlx.mlx_array,
    t_embed_w1: MfLinear,
    t_embed_b1: mlx.mlx_array,
    t_embed_w2: MfLinear,
    t_embed_b2: mlx.mlx_array,
    noise_refiner: []Block,
    context_refiner: []Block,
    layers: []Block,
    final_norm_gate_w: MfLinear, // adaLN_modulation.1 in FinalLayer (SiLU already applied before)
    final_norm_gate_b: mlx.mlx_array,
    final_linear_w: MfLinear,
    final_linear_b: mlx.mlx_array,

    pub fn deinit(self: *Dit) void {
        self.x_embed_w.deinit();
        _ = mlx.mlx_array_free(self.x_embed_b);
        _ = mlx.mlx_array_free(self.x_pad_token);
        _ = mlx.mlx_array_free(self.cap_embed_norm);
        self.cap_embed_w.deinit();
        _ = mlx.mlx_array_free(self.cap_embed_b);
        _ = mlx.mlx_array_free(self.cap_pad_token);
        self.t_embed_w1.deinit();
        _ = mlx.mlx_array_free(self.t_embed_b1);
        self.t_embed_w2.deinit();
        _ = mlx.mlx_array_free(self.t_embed_b2);
        for (self.noise_refiner) |*b| b.deinit();
        self.allocator.free(self.noise_refiner);
        for (self.context_refiner) |*b| b.deinit();
        self.allocator.free(self.context_refiner);
        for (self.layers) |*b| b.deinit();
        self.allocator.free(self.layers);
        self.final_norm_gate_w.deinit();
        _ = mlx.mlx_array_free(self.final_norm_gate_b);
        self.final_linear_w.deinit();
        _ = mlx.mlx_array_free(self.final_linear_b);
    }

    pub fn load(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8, cfg: Config, dtype: mlx.mlx_dtype) !Dit {
        const dir = try fmtKey(allocator, "{s}/transformer", .{model_dir});
        defer allocator.free(dir);
        var w = try model_mod.loadWeights(io, allocator, dir);
        defer w.deinit();

        var d: Dit = undefined;
        d.allocator = allocator;
        d.s = s;
        d.cfg = cfg;
        d.dtype = dtype;
        const patch_in = cfg.patch_size * cfg.patch_size * cfg.in_channels;
        d.x_embed_w = try MfLinear.load(&w, allocator, "all_x_embedder.2-1", patch_in, dtype, s);
        d.x_embed_b = try astype(try ownWeight(&w, "all_x_embedder.2-1.bias"), dtype, s);
        d.x_pad_token = try astype(try ownWeight(&w, "x_pad_token"), dtype, s);
        d.cap_embed_norm = try astype(try ownWeight(&w, "cap_embedder.0.weight"), dtype, s);
        d.cap_embed_w = try MfLinear.load(&w, allocator, "cap_embedder.1", cfg.cap_feat_dim, dtype, s);
        d.cap_embed_b = try astype(try ownWeight(&w, "cap_embedder.1.bias"), dtype, s);
        d.cap_pad_token = try astype(try ownWeight(&w, "cap_pad_token"), dtype, s);
        d.t_embed_w1 = try MfLinear.load(&w, allocator, "t_embedder.mlp.0", 256, dtype, s);
        d.t_embed_b1 = try astype(try ownWeight(&w, "t_embedder.mlp.0.bias"), dtype, s);
        d.t_embed_w2 = try MfLinear.load(&w, allocator, "t_embedder.mlp.2", 1024, dtype, s);
        d.t_embed_b2 = try astype(try ownWeight(&w, "t_embedder.mlp.2.bias"), dtype, s);

        d.noise_refiner = try allocator.alloc(Block, cfg.n_refiner_layers);
        for (0..cfg.n_refiner_layers) |i| {
            const pfx = try fmtKey(allocator, "noise_refiner.{d}", .{i});
            defer allocator.free(pfx);
            d.noise_refiner[i] = try loadBlock(&w, allocator, pfx, cfg, true, dtype, s);
        }
        d.context_refiner = try allocator.alloc(Block, cfg.n_refiner_layers);
        for (0..cfg.n_refiner_layers) |i| {
            const pfx = try fmtKey(allocator, "context_refiner.{d}", .{i});
            defer allocator.free(pfx);
            d.context_refiner[i] = try loadBlock(&w, allocator, pfx, cfg, false, dtype, s);
        }
        d.layers = try allocator.alloc(Block, cfg.n_layers);
        for (0..cfg.n_layers) |i| {
            const pfx = try fmtKey(allocator, "layers.{d}", .{i});
            defer allocator.free(pfx);
            d.layers[i] = try loadBlock(&w, allocator, pfx, cfg, true, dtype, s);
        }

        d.final_norm_gate_w = try MfLinear.load(&w, allocator, "all_final_layer.2-1.adaLN_modulation.1", 256, dtype, s);
        d.final_norm_gate_b = try astype(try ownWeight(&w, "all_final_layer.2-1.adaLN_modulation.1.bias"), dtype, s);
        d.final_linear_w = try MfLinear.load(&w, allocator, "all_final_layer.2-1.linear", cfg.dim, dtype, s);
        d.final_linear_b = try astype(try ownWeight(&w, "all_final_layer.2-1.linear.bias"), dtype, s);
        return d;
    }

    /// Sinusoidal timestep embedding (`TimestepEmbedder.timestep_embedding`,
    /// freq dim 256) -> MLP(256->1024->SiLU->256) -> `[1,256]`.
    fn tEmbed(self: *const Dit, t: f32, s: S) !mlx.mlx_array {
        const half = 128;
        var freqs: [half]f32 = undefined;
        for (0..half) |j| {
            const exp = -@log(@as(f64, 10000.0)) * @as(f64, @floatFromInt(j)) / @as(f64, @floatFromInt(half));
            freqs[j] = @floatCast(@exp(exp));
        }
        var emb: [256]f32 = undefined;
        for (0..half) |j| {
            const angle = @as(f64, t) * @as(f64, freqs[j]);
            emb[j] = @floatCast(@cos(angle));
            emb[half + j] = @floatCast(@sin(angle));
        }
        const shape = [_]c_int{ 1, 256 };
        const raw = mlx.mlx_array_new_data(&emb, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(raw);
        const raw_dt = try astype(raw, self.dtype, s);
        defer _ = mlx.mlx_array_free(raw_dt);
        const h1 = try self.t_embed_w1.forward(raw_dt, self.t_embed_b1, s);
        defer _ = mlx.mlx_array_free(h1);
        const a1 = try silu(h1, s);
        defer _ = mlx.mlx_array_free(a1);
        return self.t_embed_w2.forward(a1, self.t_embed_b2, s);
    }

    /// One denoise step's model call. `latent` `[1,16,H,W]`, `cap` `[Lc,cap_feat_dim]`
    /// bf16 (text encoder output). `timestep` is the diffusers-convention
    /// `(1000 - scheduler_t)` value fed to `t_embedder` (already includes the
    /// `t_scale=1000` multiply done by the caller... no: `t_embedder(t *
    /// t_scale)` where `t` here is already in [0,1] — see `Engine.generateImage`
    /// for the exact scheduler-to-model timestep mapping).
    /// Returns velocity `[1,16,H,W]` (NOT yet negated — caller negates per the
    /// pipeline's `noise_pred = -model_out`).
    pub fn forward(self: *const Dit, latent: mlx.mlx_array, cap: mlx.mlx_array, model_t: f32, s: S) !mlx.mlx_array {
        const cfg = self.cfg;
        const lsh = mlx.getShape(latent); // [1,16,H,W]
        const H = lsh[2];
        const Wd = lsh[3];
        const p: c_int = @intCast(cfg.patch_size);
        const Ht = @divExact(H, p);
        const Wt = @divExact(Wd, p);
        const n_img: usize = @intCast(Ht * Wt);

        // -- patchify image --
        const l3 = try reshape(latent, &[_]c_int{ lsh[1], Ht, p, Wt, p }, s);
        defer _ = mlx.mlx_array_free(l3);
        const lt = try transpose(l3, &[_]c_int{ 1, 3, 2, 4, 0 }, s); // [Ht,Wt,p,p,C]
        defer _ = mlx.mlx_array_free(lt);
        const ltc = try contig(lt, s);
        defer _ = mlx.mlx_array_free(ltc);
        const patches = try reshape(ltc, &[_]c_int{ @as(c_int, @intCast(n_img)), p * p * lsh[1] }, s);
        defer _ = mlx.mlx_array_free(patches);
        const patches_dt = try astype(patches, self.dtype, s);
        defer _ = mlx.mlx_array_free(patches_dt);
        var x_tok = try self.x_embed_w.forward(patches_dt, self.x_embed_b, s); // [n_img, dim]

        // pad image tokens to a multiple of 32
        const img_pad = (32 - n_img % 32) % 32;
        const img_total = n_img + img_pad;
        if (img_pad > 0) {
            const pad_row = try broadcastRow(self.x_pad_token, @intCast(img_pad), s);
            defer _ = mlx.mlx_array_free(pad_row);
            const catted = try concat(&.{ x_tok, pad_row }, 0, s);
            _ = mlx.mlx_array_free(x_tok);
            x_tok = catted;
        }
        const x3 = try reshape(x_tok, &[_]c_int{ 1, @as(c_int, @intCast(img_total)), @as(c_int, @intCast(cfg.dim)) }, s);
        _ = mlx.mlx_array_free(x_tok);
        var ximg = x3;

        // -- text hidden -> cap_embedder --
        const csh = mlx.getShape(cap); // [Lc, cap_feat_dim]
        const Lc: usize = @intCast(csh[0]);
        const cap_normed = try rms(cap, self.cap_embed_norm, cfg.norm_eps, s);
        defer _ = mlx.mlx_array_free(cap_normed);
        var cap_tok = try self.cap_embed_w.forward(cap_normed, self.cap_embed_b, s); // [Lc, dim]
        const cap_pad = (32 - Lc % 32) % 32;
        const Lc_total = Lc + cap_pad;
        if (cap_pad > 0) {
            const pad_row = try broadcastRow(self.cap_pad_token, @intCast(cap_pad), s);
            defer _ = mlx.mlx_array_free(pad_row);
            const catted = try concat(&.{ cap_tok, pad_row }, 0, s);
            _ = mlx.mlx_array_free(cap_tok);
            cap_tok = catted;
        }
        const cap3 = try reshape(cap_tok, &[_]c_int{ 1, @as(c_int, @intCast(Lc_total)), @as(c_int, @intCast(cfg.dim)) }, s);
        _ = mlx.mlx_array_free(cap_tok);
        var xcap = cap3;

        // -- position ids + RoPE tables --
        const allocator = self.allocator;
        const img_pos = try allocator.alloc([3]i32, img_total);
        defer allocator.free(img_pos);
        {
            const axis0: i32 = @intCast(Lc_total + 1);
            var idx: usize = 0;
            for (0..@intCast(Ht)) |hh| {
                for (0..@intCast(Wt)) |ww| {
                    img_pos[idx] = .{ axis0, @intCast(hh), @intCast(ww) };
                    idx += 1;
                }
            }
            while (idx < img_total) : (idx += 1) img_pos[idx] = .{ 0, 0, 0 };
        }
        const cap_pos = try allocator.alloc([3]i32, Lc_total);
        defer allocator.free(cap_pos);
        for (0..Lc_total) |i| cap_pos[i] = .{ @intCast(1 + i), 0, 0 };

        const img_rope = try buildRopeTable(allocator, img_pos, cfg.axes_dims, cfg.rope_theta, s);
        defer _ = mlx.mlx_array_free(img_rope.cos);
        defer _ = mlx.mlx_array_free(img_rope.sin);
        const cap_rope = try buildRopeTable(allocator, cap_pos, cfg.axes_dims, cfg.rope_theta, s);
        defer _ = mlx.mlx_array_free(cap_rope.cos);
        defer _ = mlx.mlx_array_free(cap_rope.sin);

        // -- t embedding (global adaLN input) --
        const t_emb = try self.tEmbed(model_t, s);
        defer _ = mlx.mlx_array_free(t_emb);

        // -- refiners --
        for (self.noise_refiner) |*blk| {
            const nx = try blk.forward(ximg, img_rope.cos, img_rope.sin, t_emb, cfg, s);
            _ = mlx.mlx_array_free(ximg);
            ximg = nx;
        }
        for (self.context_refiner) |*blk| {
            const nx = try blk.forward(xcap, cap_rope.cos, cap_rope.sin, null, cfg, s);
            _ = mlx.mlx_array_free(xcap);
            xcap = nx;
        }

        // -- concat [image, caption] + concat RoPE tables --
        var unified = try concat(&.{ ximg, xcap }, 1, s);
        _ = mlx.mlx_array_free(ximg);
        _ = mlx.mlx_array_free(xcap);
        const cos_u = try concat(&.{ img_rope.cos, cap_rope.cos }, 1, s);
        defer _ = mlx.mlx_array_free(cos_u);
        const sin_u = try concat(&.{ img_rope.sin, cap_rope.sin }, 1, s);
        defer _ = mlx.mlx_array_free(sin_u);

        for (self.layers) |*blk| {
            const nx = try blk.forward(unified, cos_u, sin_u, t_emb, cfg, s);
            _ = mlx.mlx_array_free(unified);
            unified = nx;
        }

        // -- final layer (image-token prefix only). FinalLayer.adaLN_modulation
        // is Sequential(SiLU, Linear) — SiLU precedes the linear. --
        const gate_silu_in = try silu(t_emb, s);
        defer _ = mlx.mlx_array_free(gate_silu_in);
        const gate = try self.final_norm_gate_w.forward(gate_silu_in, self.final_norm_gate_b, s);
        defer _ = mlx.mlx_array_free(gate);
        const one = scalarF(1.0);
        defer _ = mlx.mlx_array_free(one);
        const scale3 = try reshape(gate, &[_]c_int{ 1, 1, @as(c_int, @intCast(cfg.dim)) }, s);
        defer _ = mlx.mlx_array_free(scale3);
        const scale_v = try addA(scale3, one, s);
        defer _ = mlx.mlx_array_free(scale_v);

        const normed = try layerNormNoAffine(unified, 1e-6, s);
        _ = mlx.mlx_array_free(unified);
        defer _ = mlx.mlx_array_free(normed);
        const scaled = try mulA(normed, scale_v, s);
        defer _ = mlx.mlx_array_free(scaled);
        const out_full = try self.final_linear_w.forward(scaled, self.final_linear_b, s); // [1, total, patch_dim]
        defer _ = mlx.mlx_array_free(out_full);

        // image tokens come first in the unified sequence; keep the real (non-pad) prefix.
        const out_img = try sliceAxis0Seq(out_full, 0, @intCast(n_img), s);
        defer _ = mlx.mlx_array_free(out_img);

        // -- unpatchify: [n_img, p*p*C] -> [C, H, W] -> [1,C,H,W] --
        const patch_dim = mlx.getShape(out_img)[2];
        const o2 = try reshape(out_img, &[_]c_int{ @as(c_int, @intCast(n_img)), patch_dim }, s);
        defer _ = mlx.mlx_array_free(o2);
        const o5 = try reshape(o2, &[_]c_int{ Ht, Wt, p, p, lsh[1] }, s);
        defer _ = mlx.mlx_array_free(o5);
        const ot = try transpose(o5, &[_]c_int{ 4, 0, 2, 1, 3 }, s); // [C,Ht,p,Wt,p]
        defer _ = mlx.mlx_array_free(ot);
        const otc = try contig(ot, s);
        defer _ = mlx.mlx_array_free(otc);
        const img = try reshape(otc, &[_]c_int{ 1, lsh[1], H, Wd }, s);
        return astype(img, .float32, s);
    }
};

/// Slice axis 1 (the sequence axis of a `[1,seq,dim]` tensor) to `[start,stop)`.
fn sliceAxis0Seq(x: mlx.mlx_array, start: c_int, stop: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    var lo = [_]c_int{ 0, 0, 0 };
    var hi = [_]c_int{ sh[0], sh[1], sh[2] };
    const st = [_]c_int{ 1, 1, 1 };
    lo[1] = start;
    hi[1] = stop;
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, &lo, 3, &hi, 3, &st, 3, s));
    return o;
}

/// Repeat a `[1,dim]` row vector `n` times -> `[n,dim]`.
fn broadcastRow(row: mlx.mlx_array, n: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(row);
    const dim = sh[sh.len - 1];
    const r2 = try reshape(row, &[_]c_int{ 1, dim }, s);
    defer _ = mlx.mlx_array_free(r2);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_repeat_axis(&o, r2, n, 0, s));
    return o;
}

/// LayerNorm over the last axis with NO affine params, computed in f32,
/// cast back to the input dtype (`FinalLayer.norm_final`, `elementwise_affine=False`).
fn layerNormNoAffine(x: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    const in_dt = mlx.mlx_array_dtype(x);
    const xf = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    var mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mean);
    try mlx.check(mlx.mlx_mean_axis(&mean, xf, -1, true, s));
    const centered = try subA(xf, mean, s);
    defer _ = mlx.mlx_array_free(centered);
    const sq = try mulA(centered, centered, s);
    defer _ = mlx.mlx_array_free(sq);
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_mean_axis(&v, sq, -1, true, s));
    const epsa = scalarF(eps);
    defer _ = mlx.mlx_array_free(epsa);
    var ve = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ve);
    try mlx.check(mlx.mlx_add(&ve, v, epsa, s));
    var rsq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rsq);
    try mlx.check(mlx.mlx_rsqrt(&rsq, ve, s));
    const out = try mulA(centered, rsq, s);
    if (in_dt == .float32) return out;
    defer _ = mlx.mlx_array_free(out);
    return astype(out, in_dt, s);
}

// ── Sampler ──

/// Z-Image's static-shift schedule: `sigmas = linspace(1, 1/N, N)`, scheduler
/// shift `sigma' = shift*sigma/(1+(shift-1)*sigma)` (shift=3.0,
/// `use_dynamic_shifting=false`), with a trailing 0 appended for the final
/// Euler target.
pub fn computeSigmas(allocator: std.mem.Allocator, steps: u32, shift: f32) ![]f32 {
    const n: usize = steps;
    const out = try allocator.alloc(f32, n + 1);
    const sh: f64 = shift;
    for (0..n) |i| {
        const base: f64 = if (n == 1) 1.0 else 1.0 - (1.0 - 1.0 / @as(f64, @floatFromInt(n))) * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n - 1));
        out[i] = @floatCast(sh * base / (1.0 + (sh - 1.0) * base));
    }
    out[n] = 0.0;
    return out;
}

fn denormImage(decoded: mlx.mlx_array, s: S) !mlx.mlx_array {
    const df = try astype(decoded, .float32, s);
    defer _ = mlx.mlx_array_free(df);
    const half = scalarF(0.5);
    defer _ = mlx.mlx_array_free(half);
    const sc = try mulA(df, half, s);
    defer _ = mlx.mlx_array_free(sc);
    const shifted = try addA(sc, half, s);
    defer _ = mlx.mlx_array_free(shifted);
    const lo = scalarF(0.0);
    defer _ = mlx.mlx_array_free(lo);
    const hi = scalarF(1.0);
    defer _ = mlx.mlx_array_free(hi);
    var clo = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(clo);
    try mlx.check(mlx.mlx_maximum(&clo, shifted, lo, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_minimum(&out, clo, hi, s));
    return out;
}

fn normalizeDim(size: u32) u32 {
    // Z-Image's vae_scale factor is 8, DiT patch 2 -> multiples of 32; clamp
    // to a sane image-gen range like the other backends.
    const clamped = std.math.clamp(size, 256, 2048);
    return (clamped / 32) * 32;
}

// ── Engine ──

pub const Engine = struct {
    allocator: std.mem.Allocator,
    s: S,
    model_dir: []u8,
    cfg: Config,
    dtype: mlx.mlx_dtype,
    te: TextEncoder,
    dit: Dit,
    vae: Vae,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*Engine {
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.s = mlx.mlx_default_gpu_stream_new();
        self.model_dir = try allocator.dupe(u8, model_dir);
        errdefer allocator.free(self.model_dir);
        self.cfg = try parseConfig(io, allocator, model_dir);
        self.dtype = .bfloat16;

        self.te = try TextEncoder.load(io, allocator, model_dir);
        errdefer self.te.deinit();
        self.dit = try Dit.load(io, allocator, self.s, model_dir, self.cfg, self.dtype);
        errdefer self.dit.deinit();
        self.vae = try loadVae(io, allocator, self.s, model_dir);
        errdefer self.vae.deinit();

        log.info(
            "[image] Z-Image ready (DiT dim {d} x{d} heads, {d} layers; TE Qwen3 {d}L; {s})\n",
            .{ self.cfg.dim, self.cfg.n_heads, self.cfg.n_layers, self.cfg.te_layers, if (self.cfg.is_turbo) "Turbo defaults (8 steps, CFG 0)" else "base defaults (50 steps, CFG 5)" },
        );
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.vae.deinit();
        self.dit.deinit();
        self.te.deinit();
        self.allocator.free(self.model_dir);
        self.allocator.destroy(self);
    }

    /// Text->image. Returns `[1,3,H,W]` f32 in `[0,1]` (owned; caller frees).
    pub fn generateImage(
        self: *Engine,
        allocator: std.mem.Allocator,
        prompt: []const u8,
        width: u32,
        height: u32,
        seed: u64,
        steps: u32,
        progress: ?sse.Progress,
    ) !mlx.mlx_array {
        const s = self.s;
        const n_steps: u32 = if (steps == 0) self.cfg.default_steps else steps;
        const W = normalizeDim(width);
        const H = normalizeDim(height);
        // prepare_latents: latent H/W = 2*(px // (vae_scale*2)) = px/8 (already
        // a multiple of 4 given `normalizeDim`'s 32-multiple clamp).
        const lat_h: c_int = @intCast(H / 8);
        const lat_w: c_int = @intCast(W / 8);

        const cap = try self.te.encode(s, prompt);
        defer _ = mlx.mlx_array_free(cap);
        // Classifier-free guidance (base Z-Image only; Turbo ships
        // `default_cfg=0` and skips this branch entirely — no negative
        // forward, matching the pipeline's `apply_cfg` gate).
        const guidance_scale = self.cfg.default_cfg;
        const neg_cap: ?mlx.mlx_array = if (guidance_scale > 0) try self.te.encode(s, "") else null;
        defer {
            if (neg_cap) |nc| _ = mlx.mlx_array_free(nc);
        }

        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, seed));
        const nsh = [_]c_int{ 1, @intCast(self.cfg.in_channels), lat_h, lat_w };
        var latent = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&latent, &nsh, 4, .float32, 0.0, 1.0, key, s));

        const sigmas = try computeSigmas(allocator, n_steps, 3.0);
        defer allocator.free(sigmas);

        for (0..n_steps) |i| {
            if (progress) |p| if (p.cancelled()) {
                _ = mlx.mlx_array_free(latent);
                return error.Cancelled;
            };
            const latent_dt = try astype(latent, self.dtype, s);
            defer _ = mlx.mlx_array_free(latent_dt);
            // diffusers: `timestep = (1000 - t)/1000`, then `t_embedder(timestep
            // * t_scale)` = `t_embedder(1000 - t)` where `t = sigmas[i]*1000`.
            const model_t: f32 = 1000.0 - sigmas[i] * 1000.0;
            const v = try self.dit.forward(latent_dt, cap, model_t, s);
            defer _ = mlx.mlx_array_free(v);
            // CFG: `pred = pos + scale*(pos - neg)`, computed on the RAW
            // (pre-negation) model outputs, matching the pipeline exactly.
            var pred = v;
            var pred_owned = false;
            if (neg_cap) |nc| {
                const vn = try self.dit.forward(latent_dt, nc, model_t, s);
                defer _ = mlx.mlx_array_free(vn);
                const diff = try subA(v, vn, s);
                defer _ = mlx.mlx_array_free(diff);
                const scale = scalarF(guidance_scale);
                defer _ = mlx.mlx_array_free(scale);
                const scaled_diff = try mulA(diff, scale, s);
                defer _ = mlx.mlx_array_free(scaled_diff);
                pred = try addA(v, scaled_diff, s);
                pred_owned = true;
            }
            defer {
                if (pred_owned) _ = mlx.mlx_array_free(pred);
            }
            // pipeline negates the model output before the Euler step.
            const neg_one = scalarF(-1.0);
            defer _ = mlx.mlx_array_free(neg_one);
            const noise_pred = try mulA(pred, neg_one, s);
            defer _ = mlx.mlx_array_free(noise_pred);
            const dt = scalarF(sigmas[i + 1] - sigmas[i]);
            defer _ = mlx.mlx_array_free(dt);
            const step = try mulA(noise_pred, dt, s);
            defer _ = mlx.mlx_array_free(step);
            const next = try addA(latent, step, s);
            _ = mlx.mlx_array_free(latent);
            latent = next;
            _ = mlx.mlx_array_eval(latent);
            if (progress) |p| p.emit("Generating", @intCast(i + 1), n_steps);
        }

        if (progress) |p| p.emit("Decoding image", n_steps, n_steps);
        const decoded = try self.vae.decode(latent);
        _ = mlx.mlx_array_free(latent);
        defer _ = mlx.mlx_array_free(decoded);
        return denormImage(decoded, s);
    }
};

// ── Tests ──

const testing = std.testing;

test "computeSigmas: static shift=3.0 formula, monotone decreasing, trailing 0" {
    const a = testing.allocator;
    const sigmas = try computeSigmas(a, 8, 3.0);
    defer a.free(sigmas);
    try testing.expectEqual(@as(usize, 9), sigmas.len);
    try testing.expectApproxEqAbs(@as(f32, 1.0), sigmas[0], 1e-6);
    try testing.expectEqual(@as(f32, 0.0), sigmas[8]);
    for (0..8) |i| try testing.expect(sigmas[i] > sigmas[i + 1]);
    // hand-computed: base at i=7 (last real step) = 1/8 = 0.125;
    // shift(0.125) = 3*0.125/(1+2*0.125) = 0.375/1.25 = 0.3
    try testing.expectApproxEqAbs(@as(f32, 0.3), sigmas[7], 1e-5);
}

test "dirLooksTurbo: case-insensitive substring match" {
    try testing.expect(dirLooksTurbo("/models/Tongyi-MAI/Z-Image-Turbo"));
    try testing.expect(dirLooksTurbo("/models/z-image-turbo-8bit"));
    try testing.expect(!dirLooksTurbo("/models/Tongyi-MAI/Z-Image"));
}

test "Config.ffnHidden / headDim match the checkpoint's real geometry" {
    const cfg = Config{};
    try testing.expectEqual(@as(u32, 128), cfg.headDim()); // 3840/30
    try testing.expectEqual(@as(u32, 10240), cfg.ffnHidden()); // 3840/3*8
}

test "normalizeDim clamps to [256,2048] and rounds to a multiple of 32" {
    try testing.expectEqual(@as(u32, 1024), normalizeDim(1024));
    try testing.expectEqual(@as(u32, 992), normalizeDim(1000));
    try testing.expectEqual(@as(u32, 256), normalizeDim(64));
    try testing.expectEqual(@as(u32, 2048), normalizeDim(4096));
}

// Live text-encoder check against a real converted checkpoint — isolates
// TextEncoder.load + .encode from the rest of the DiT/sampler/HTTP stack.
// `ZIMAGE_TEST_MODEL=<dir> zig build test -Dtest-filter="zimage text encoder live"`
test "zimage text encoder live: encode returns a finite, in-range embedding" {
    const model_dir = std.mem.span(std.c.getenv("ZIMAGE_TEST_MODEL") orelse return error.SkipZigTest);
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    var te = try TextEncoder.load(io, testing.allocator, model_dir);
    defer te.deinit();
    const cap = try te.encode(s, "a red fox in the snow");
    defer _ = mlx.mlx_array_free(cap);
    _ = mlx.mlx_array_eval(cap);

    const sh = mlx.getShape(cap);
    try testing.expect(sh.len == 2);
    try testing.expect(sh[0] > 0);
    try testing.expectEqual(@as(c_int, 2560), sh[1]);

    const cap_f32 = try astype(cap, .float32, s);
    defer _ = mlx.mlx_array_free(cap_f32);
    _ = mlx.mlx_array_eval(cap_f32);
    const data = mlx.mlx_array_data_float32(cap_f32) orelse return error.NullData;
    const n: usize = @intCast(sh[0] * sh[1]);
    var any_nonzero = false;
    for (0..n) |i| {
        const v = data[i];
        try testing.expect(std.math.isFinite(v));
        if (v != 0) any_nonzero = true;
    }
    try testing.expect(any_nonzero);
}

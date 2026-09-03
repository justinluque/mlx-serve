//! Ideogram 4 — native text→image (Qwen3-VL-8B 13-tap text encoder +
//! single-stream MRoPE DiT + Euler flow-matching with asymmetric CFG + the
//! Flux2 KL autoencoder), ported from `ideogram-oss/ideogram4` (Apache-2.0).
//!
//! Three things separate it from every other image backend here:
//!
//!  * TWO transformers. The negative branch is its OWN 9.3B checkpoint
//!    (`unconditional_transformer/`) run image-only with zeroed conditioning —
//!    not the same weights with an empty prompt. That is what "asymmetric CFG"
//!    means, and it doubles the DiT residency.
//!  * ONE stream. Text tokens and image tokens ride the same sequence through
//!    the same projections at every layer; there is no cross-attention and no
//!    context branch. The two roles are told apart by an indicator embedding
//!    and by 3D MRoPE positions (text at 0.., image at 65536+(h,w)).
//!  * The prompt is a JSON CAPTION. The model was trained exclusively on
//!    structured captions, so a bare sentence is out of distribution — see
//!    `ideogram4_prompt.zig` for the magic-prompt rewriter.
//!
//! The text encoder is `flux.TextEncoder` (the same Qwen3 block stack) with a
//! 13-tap list and tap-INNER flattening; the VAE is `flux.Vae.decodeLatent`
//! (byte-identical architecture) with Ideogram's published latent table
//! instead of the checkpoint's `bn` stats.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const sse = @import("gen_sse.zig");
const lora_mod = @import("lora.zig");
const flux = @import("flux.zig");

const Weights = model_mod.Weights;
const S = mlx.mlx_stream;

// ── Low-level helpers (per-file by convention; see krea.zig/hunyuan3d.zig) ──

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
inline fn rms(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&o, x, w, eps, s));
    return o;
}
inline fn astype(x: mlx.mlx_array, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, s));
    return o;
}
inline fn silu(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sigmoid(&o, x, s));
    defer _ = mlx.mlx_array_free(o);
    return mulA(x, o, s);
}
inline fn tanhA(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_tanh(&o, x, s));
    return o;
}
inline fn contig(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&o, x, false, s));
    return o;
}
fn concat(arrs: []const mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (arrs) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, axis, s));
    return o;
}
/// Half-open slice along `axis` of a rank-3 array.
fn slice3(x: mlx.mlx_array, axis: usize, start: c_int, stop: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    var lo = [_]c_int{ 0, 0, 0 };
    var hi = [_]c_int{ sh[0], sh[1], sh[2] };
    const st = [_]c_int{ 1, 1, 1 };
    lo[axis] = start;
    hi[axis] = stop;
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, &lo, 3, &hi, 3, &st, 3, s));
    return o;
}
/// Half-open slice along the LAST axis of a rank-4 array.
fn sliceLast4(x: mlx.mlx_array, start: c_int, stop: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const lo = [_]c_int{ 0, 0, 0, start };
    const hi = [_]c_int{ sh[0], sh[1], sh[2], stop };
    const st = [_]c_int{ 1, 1, 1, 1 };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, &lo, 4, &hi, 4, &st, 4, s));
    return o;
}
fn ownWeight(w: *const Weights, key: []const u8) !mlx.mlx_array {
    const a = w.get(key) orelse {
        log.err("[ideogram4] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingWeight;
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

// ── Linear: affine-quantized at any width, or dense bf16, plus stacked LoRA ──

/// One projection. The converter quantizes attention/MLP but can leave the
/// modulation and embedding projections dense (see `tests/convert_ideogram4.py
/// --precision mixed`), so BOTH arms have to exist in one type — a per-tensor
/// decision, read off whether `.scales` is present, never off a name list.
pub const IgLinear = struct {
    quantized: bool,
    /// quantized: packed u32 [out, in·bits/32]; dense: PRE-TRANSPOSED [in,out].
    w: mlx.mlx_array,
    scales: mlx.mlx_array = .{ .ctx = null },
    biases: mlx.mlx_array = .{ .ctx = null },
    add_bias: ?mlx.mlx_array = null,
    bits: u32 = 0,
    group_size: u32 = 0,
    // Runtime adapters (non-owning; gen.zig's lora.Stack owns the arrays).
    lora_refs: [lora_mod.MAX_LORAS]lora_mod.Ref = undefined,
    lora_count: u8 = 0,

    /// `in_features` is the module's logical input width, which is the only
    /// way to solve (bits, group_size) out of the packed geometry.
    pub fn load(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, in_features: u32, s: S) !IgLinear {
        const wk = try fmtKey(a, "{s}.weight", .{prefix});
        defer a.free(wk);
        const sk = try fmtKey(a, "{s}.scales", .{prefix});
        defer a.free(sk);
        const bk = try fmtKey(a, "{s}.biases", .{prefix});
        defer a.free(bk);
        const ak = try fmtKey(a, "{s}.bias", .{prefix});
        defer a.free(ak);

        if (ownOpt(w, sk)) |scales| {
            const weight = try ownWeight(w, wk);
            const bs = try ownWeight(w, bk);
            const w_cols: u32 = @intCast(mlx.getShape(weight)[1]);
            const s_cols: u32 = @intCast(mlx.getShape(scales)[1]);
            if (in_features == 0 or w_cols == 0 or s_cols == 0) return error.BadQuantGeometry;
            return .{
                .quantized = true,
                .w = weight,
                .scales = scales,
                .biases = bs,
                .add_bias = ownOpt(w, ak),
                .bits = @intCast(@divExact(32 * w_cols, in_features)),
                .group_size = @intCast(@divExact(in_features, s_cols)),
            };
        }
        const raw = try ownWeight(w, wk);
        defer _ = mlx.mlx_array_free(raw);
        const t = try transpose(raw, &[_]c_int{ 1, 0 }, s);
        defer _ = mlx.mlx_array_free(t);
        const tc = try contig(t, s);
        defer _ = mlx.mlx_array_free(tc);
        const wt = try astype(tc, .bfloat16, s);
        return .{ .quantized = false, .w = wt, .add_bias = ownOpt(w, ak) };
    }

    pub fn deinit(self: *IgLinear) void {
        _ = mlx.mlx_array_free(self.w);
        if (self.quantized) {
            _ = mlx.mlx_array_free(self.scales);
            _ = mlx.mlx_array_free(self.biases);
        }
        if (self.add_bias) |b| _ = mlx.mlx_array_free(b);
    }

    pub fn forward(self: *const IgLinear, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        var o = mlx.mlx_array_new();
        if (self.quantized) {
            try mlx.check(mlx.mlx_quantized_matmul(&o, x, self.w, self.scales, self.biases, true, mlx.mlx_optional_int.some(@intCast(self.group_size)), mlx.mlx_optional_int.some(@intCast(self.bits)), "affine", s));
        } else {
            try mlx.check(mlx.mlx_matmul(&o, x, self.w, s));
        }
        if (self.add_bias) |b| {
            const r = try addA(o, b, s);
            _ = mlx.mlx_array_free(o);
            o = r;
        }
        if (self.lora_count > 0) {
            const d = try lora_mod.deltaSum(x, self.lora_refs[0..self.lora_count], s);
            defer _ = mlx.mlx_array_free(d);
            const r = try addA(o, d, s);
            _ = mlx.mlx_array_free(o);
            o = r;
        }
        return o;
    }

    fn setLoraRefs(self: *IgLinear, refs: []const lora_mod.Ref) void {
        self.lora_count = @intCast(refs.len);
        @memcpy(self.lora_refs[0..refs.len], refs);
    }
    fn clearLoraRefs(self: *IgLinear) void {
        self.lora_count = 0;
    }
};

// ── Config ──

/// Model spec, from `docs/model_architecture.md` in the reference repo. Every
/// field that can be read off the checkpoint IS read off it (`configFrom`) —
/// a hardcoded layer count against a different build loads a prefix of the
/// blocks without erroring and generates a plausible WRONG image.
pub const Config = struct {
    emb_dim: u32 = 4608,
    num_layers: u32 = 34,
    num_heads: u32 = 18,
    intermediate_size: u32 = 12288,
    adaln_dim: u32 = 512,
    /// ae_channels (32) · patch_size² (4).
    in_channels: u32 = 128,
    /// Qwen3-VL hidden (4096) · number of taps (13).
    llm_features_dim: u32 = 4096 * 13,
    rope_theta: f64 = 5_000_000.0,
    /// Only sections [1] and [2] are consumed — see `buildMrope`.
    mrope_section: [3]u32 = .{ 24, 20, 20 },
    norm_eps: f32 = 1e-5,

    pub fn headDim(self: Config) u32 {
        return self.emb_dim / self.num_heads;
    }
};

/// Layers of Qwen3-VL whose hidden states the DiT consumes. The reference
/// counts "output of decoder layer i"; `flux.TextEncoder` counts capture slots
/// where slot 0 is the embedding, so every entry is one higher here.
pub const qwen3_vl_taps = [_]u32{ 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 36 };

/// The reference's own `QWEN3_VL_ACTIVATION_LAYERS`, kept beside the shifted
/// list so the off-by-one is visible rather than folded away.
pub const qwen3_vl_activation_layers = [_]u32{ 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35 };

/// Image grid coordinates start here so they never collide with text
/// positions (which start at 0 and stop below `max_text_tokens`).
pub const image_position_offset: i64 = 65536;
pub const max_text_tokens: usize = 2048;
/// patch_size (2) · ae_scale_factor (8): every side must be a multiple.
pub const dim_multiple: u32 = 16;

const latent_norm = @import("ideogram4_latent.zig");

// ── Sampler ──

pub const SamplerPreset = enum {
    turbo_12,
    default_20,
    quality_48,

    pub fn params(self: SamplerPreset) SamplerParameters {
        return switch (self) {
            // guidance_schedule is in LOOP-INDEX order: index 0 is the LAST
            // (polish) step. Each preset runs the bulk at gw=7 and finishes
            // with a few polish steps at gw=3.
            .turbo_12 => .{ .num_steps = 12, .cleanup_steps = 1, .mu = 0.5, .std = 1.75 },
            .default_20 => .{ .num_steps = 20, .cleanup_steps = 2, .mu = 0.0, .std = 1.75 },
            .quality_48 => .{ .num_steps = 48, .cleanup_steps = 3, .mu = 0.0, .std = 1.5 },
        };
    }

    /// The preset whose step count a request's `steps` asks for, so the mu/std
    /// pair travels WITH the step count instead of being silently mismatched.
    pub fn forSteps(steps: u32) SamplerPreset {
        if (steps <= 15) return .turbo_12;
        if (steps <= 33) return .default_20;
        return .quality_48;
    }
};

pub const SamplerParameters = struct {
    num_steps: u32,
    /// Trailing polish steps run at `guidance_cleanup` instead of `guidance`.
    cleanup_steps: u32,
    mu: f64,
    std: f64,
    guidance: f32 = 7.0,
    guidance_cleanup: f32 = 3.0,

    /// Guidance for loop index `i` (index 0 is the LAST step taken).
    pub fn guidanceAt(self: SamplerParameters, i: u32) f32 {
        return if (i < self.cleanup_steps) self.guidance_cleanup else self.guidance;
    }
};

/// Inverse standard-normal CDF (Acklam's rational approximation). Relative
/// error stays under ~1.15e-9 across the open interval, which is four orders
/// tighter than anything a 12–48-step noise schedule can resolve; the
/// endpoints go to ±inf, which the schedule's clamp then absorbs. (The usual
/// Halley refinement needs `erfc`, and Zig's std has none.)
pub fn ndtri(p: f64) f64 {
    if (p <= 0.0) return -std.math.inf(f64);
    if (p >= 1.0) return std.math.inf(f64);
    const a = [_]f64{ -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00 };
    const b = [_]f64{ -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01 };
    const c = [_]f64{ -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00 };
    const d = [_]f64{ 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00 };
    const p_low = 0.02425;
    var x: f64 = undefined;
    if (p < p_low) {
        const q = @sqrt(-2.0 * @log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= 1.0 - p_low) {
        const q = p - 0.5;
        const r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        const q = @sqrt(-2.0 * @log(1.0 - p));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    return x;
}

/// The resolution-aware logit-normal noise schedule. `mean` slides with the
/// pixel count, which is why a 2048² render is not the 512² schedule stretched.
pub const LogitNormalSchedule = struct {
    mean: f64,
    std: f64 = 1.0,
    logsnr_min: f64 = -15.0,
    logsnr_max: f64 = 18.0,

    pub fn forResolution(height: u32, width: u32, known_mean: f64, sd: f64) LogitNormalSchedule {
        const num_pixels: f64 = @floatFromInt(@as(u64, height) * @as(u64, width));
        const known_pixels: f64 = 512.0 * 512.0;
        return .{ .mean = known_mean + 0.5 * @log(num_pixels / known_pixels), .std = sd };
    }

    pub fn at(self: LogitNormalSchedule, t: f64) f64 {
        const z = ndtri(t);
        const y = self.mean + self.std * z;
        // 1 − expit(y), then clamp to the log-SNR window.
        const t_ = 1.0 - 1.0 / (1.0 + @exp(-y));
        const t_min = 1.0 / (1.0 + @exp(0.5 * self.logsnr_max));
        const t_max = 1.0 / (1.0 + @exp(0.5 * self.logsnr_min));
        return std.math.clamp(t_, t_min, t_max);
    }
};

// ── DiT ──

const Block = struct {
    qkv: IgLinear,
    o: IgLinear,
    norm_q: mlx.mlx_array,
    norm_k: mlx.mlx_array,
    w1: IgLinear,
    w2: IgLinear,
    w3: IgLinear,
    attention_norm1: mlx.mlx_array,
    attention_norm2: mlx.mlx_array,
    ffn_norm1: mlx.mlx_array,
    ffn_norm2: mlx.mlx_array,
    adaln_modulation: IgLinear,

    fn deinit(self: *Block) void {
        inline for (.{ "qkv", "o", "w1", "w2", "w3", "adaln_modulation" }) |f| @field(self, f).deinit();
        inline for (.{ "norm_q", "norm_k", "attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2" }) |f| _ = mlx.mlx_array_free(@field(self, f));
    }
};

/// The cos/sin tables for one packed sequence, [1,1,L,head_dim] bf16, ready to
/// broadcast over the head axis.
pub const Rope = struct {
    cos: mlx.mlx_array,
    sin: mlx.mlx_array,
    text_len: usize,
    image_len: usize,

    pub fn deinit(self: *Rope) void {
        _ = mlx.mlx_array_free(self.cos);
        _ = mlx.mlx_array_free(self.sin);
    }
};

pub const Transformer = struct {
    cfg: Config,
    allocator: std.mem.Allocator,
    s: S,
    input_proj: IgLinear,
    llm_cond_norm: mlx.mlx_array,
    llm_cond_proj: IgLinear,
    t_mlp_in: IgLinear,
    t_mlp_out: IgLinear,
    adaln_proj: IgLinear,
    /// [2, emb_dim] dense: row 0 = text token, row 1 = output image token.
    embed_image_indicator: mlx.mlx_array,
    layers: []Block,
    final_linear: IgLinear,
    final_adaln: IgLinear,

    pub fn deinit(self: *Transformer) void {
        inline for (.{ "input_proj", "llm_cond_proj", "t_mlp_in", "t_mlp_out", "adaln_proj", "final_linear", "final_adaln" }) |f| @field(self, f).deinit();
        _ = mlx.mlx_array_free(self.llm_cond_norm);
        _ = mlx.mlx_array_free(self.embed_image_indicator);
        for (self.layers) |*l| l.deinit();
        self.allocator.free(self.layers);
    }

    /// `t_embedding` + `adaln_proj`: one [1,1,adaln_dim] vector per step.
    /// Note the reference silu's it here AND again inside the final layer.
    fn adalnInput(self: *Transformer, t: f64) !mlx.mlx_array {
        const s = self.s;
        const dim = self.cfg.emb_dim;
        const half = dim / 2;
        // Ideogram4EmbedScalar over input_range (0,1): scaled = 1e4·t, then a
        // sinusoidal embedding whose own log-space is also 1e4.
        const scaled: f64 = 1.0e4 * t;
        const buf = try self.allocator.alloc(f32, dim);
        defer self.allocator.free(buf);
        const freq_step: f64 = @log(1.0e4) / @as(f64, @floatFromInt(half - 1));
        for (0..half) |i| {
            const f = @exp(-freq_step * @as(f64, @floatFromInt(i)));
            const ang = scaled * f;
            buf[i] = @floatCast(@sin(ang));
            buf[half + i] = @floatCast(@cos(ang));
        }
        const shape = [_]c_int{ 1, 1, @intCast(dim) };
        const raw = mlx.mlx_array_new_data(buf.ptr, &shape, 3, .float32);
        defer _ = mlx.mlx_array_free(raw);
        const emb = try astype(raw, .bfloat16, s);
        defer _ = mlx.mlx_array_free(emb);
        const h1 = try self.t_mlp_in.forward(emb, s);
        defer _ = mlx.mlx_array_free(h1);
        const a1 = try silu(h1, s);
        defer _ = mlx.mlx_array_free(a1);
        const t_cond = try self.t_mlp_out.forward(a1, s);
        defer _ = mlx.mlx_array_free(t_cond);
        const proj = try self.adaln_proj.forward(t_cond, s);
        defer _ = mlx.mlx_array_free(proj);
        return silu(proj, s);
    }

    /// Velocity for the image span. `llm_feat` is [1,n_text,llm_dim] (null for
    /// the unconditional branch, which carries no text tokens at all), `z` is
    /// [1,n_img,in_channels]. Returns [1,n_img,in_channels] f32.
    pub fn forward(self: *Transformer, llm_feat: ?mlx.mlx_array, z: mlx.mlx_array, t: f64, rope: *const Rope) !mlx.mlx_array {
        const s = self.s;
        const c = self.cfg;
        const n_img: c_int = @intCast(rope.image_len);
        const n_txt: c_int = @intCast(rope.text_len);
        const emb: c_int = @intCast(c.emb_dim);

        const adaln_input = try self.adalnInput(t);
        defer _ = mlx.mlx_array_free(adaln_input);

        // Indicator embedding rows: 0 = text, 1 = output image. Sliced once.
        const ind_txt = blk: {
            const lo = [_]c_int{ 0, 0 };
            const hi = [_]c_int{ 1, emb };
            const st = [_]c_int{ 1, 1 };
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&o, self.embed_image_indicator, &lo, 2, &hi, 2, &st, 2, s));
            break :blk o;
        };
        defer _ = mlx.mlx_array_free(ind_txt);
        const ind_img = blk: {
            const lo = [_]c_int{ 1, 0 };
            const hi = [_]c_int{ 2, emb };
            const st = [_]c_int{ 1, 1 };
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&o, self.embed_image_indicator, &lo, 2, &hi, 2, &st, 2, s));
            break :blk o;
        };
        defer _ = mlx.mlx_array_free(ind_img);

        // Image half: input_proj(z) + indicator[1]. The reference masks the
        // projection's BIAS off the text rows, which is the same thing as
        // never running the text rows through it.
        const zb = try astype(z, .bfloat16, s);
        defer _ = mlx.mlx_array_free(zb);
        const xi = try self.input_proj.forward(zb, s);
        defer _ = mlx.mlx_array_free(xi);
        const h_img = try addA(xi, ind_img, s);
        defer _ = mlx.mlx_array_free(h_img);

        // Text half: llm_cond_proj(llm_cond_norm(feat)) + indicator[0]. Same
        // reasoning — running 16k zeroed image rows through a 53248-wide
        // projection would dominate the whole forward for a guaranteed zero.
        var h: mlx.mlx_array = undefined;
        if (llm_feat) |feat| {
            const fb = try astype(feat, .bfloat16, s);
            defer _ = mlx.mlx_array_free(fb);
            const fn_ = try rms(fb, self.llm_cond_norm, 1e-6, s);
            defer _ = mlx.mlx_array_free(fn_);
            const fp = try self.llm_cond_proj.forward(fn_, s);
            defer _ = mlx.mlx_array_free(fp);
            const h_txt = try addA(fp, ind_txt, s);
            defer _ = mlx.mlx_array_free(h_txt);
            h = try concat(&[_]mlx.mlx_array{ h_txt, h_img }, 1, s);
        } else {
            std.debug.assert(n_txt == 0);
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&o, h_img));
            h = o;
        }
        errdefer _ = mlx.mlx_array_free(h);

        for (self.layers) |*layer| {
            const nh = try self.blockForward(h, layer, adaln_input, rope);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }

        // Final layer: LayerNorm (no affine) · (1 + adaln(silu(c))), then out.
        const cs = try silu(adaln_input, s);
        defer _ = mlx.mlx_array_free(cs);
        const fscale_raw = try self.final_adaln.forward(cs, s);
        defer _ = mlx.mlx_array_free(fscale_raw);
        const one = mlx.mlx_array_new_float(1.0);
        defer _ = mlx.mlx_array_free(one);
        const fscale = try addA(fscale_raw, one, s);
        defer _ = mlx.mlx_array_free(fscale);
        const null_arr = mlx.mlx_array{ .ctx = null };
        var normed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(normed);
        try mlx.check(mlx.mlx_fast_layer_norm(&normed, h, null_arr, null_arr, 1e-6, s));
        _ = mlx.mlx_array_free(h);
        const modulated = try mulA(normed, fscale, s);
        defer _ = mlx.mlx_array_free(modulated);
        const out = try self.final_linear.forward(modulated, s);
        defer _ = mlx.mlx_array_free(out);
        // Only the image span is meaningful.
        const img_span = if (n_txt == 0) blk: {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&o, out));
            break :blk o;
        } else try slice3(out, 1, n_txt, n_txt + n_img, s);
        defer _ = mlx.mlx_array_free(img_span);
        return astype(img_span, .float32, s);
    }

    fn blockForward(self: *Transformer, x: mlx.mlx_array, layer: *const Block, adaln_input: mlx.mlx_array, rope: *const Rope) !mlx.mlx_array {
        const s = self.s;
        const c = self.cfg;
        const emb: c_int = @intCast(c.emb_dim);

        const mod = try layer.adaln_modulation.forward(adaln_input, s);
        defer _ = mlx.mlx_array_free(mod);
        // chunk(4, dim=-1): scale_msa, gate_msa, scale_mlp, gate_mlp.
        const chunk = struct {
            fn get(m: mlx.mlx_array, i: c_int, width: c_int, st: S) !mlx.mlx_array {
                const sh = mlx.getShape(m);
                const lo = [_]c_int{ 0, 0, i * width };
                const hi = [_]c_int{ sh[0], sh[1], (i + 1) * width };
                const stride = [_]c_int{ 1, 1, 1 };
                var o = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_slice(&o, m, &lo, 3, &hi, 3, &stride, 3, st));
                return o;
            }
        };
        const one = mlx.mlx_array_new_float(1.0);
        defer _ = mlx.mlx_array_free(one);

        const scale_msa_raw = try chunk.get(mod, 0, emb, s);
        defer _ = mlx.mlx_array_free(scale_msa_raw);
        const scale_msa = try addA(scale_msa_raw, one, s);
        defer _ = mlx.mlx_array_free(scale_msa);
        const gate_msa_raw = try chunk.get(mod, 1, emb, s);
        defer _ = mlx.mlx_array_free(gate_msa_raw);
        const gate_msa = try tanhA(gate_msa_raw, s);
        defer _ = mlx.mlx_array_free(gate_msa);
        const scale_mlp_raw = try chunk.get(mod, 2, emb, s);
        defer _ = mlx.mlx_array_free(scale_mlp_raw);
        const scale_mlp = try addA(scale_mlp_raw, one, s);
        defer _ = mlx.mlx_array_free(scale_mlp);
        const gate_mlp_raw = try chunk.get(mod, 3, emb, s);
        defer _ = mlx.mlx_array_free(gate_mlp_raw);
        const gate_mlp = try tanhA(gate_mlp_raw, s);
        defer _ = mlx.mlx_array_free(gate_mlp);

        // Attention branch.
        const an1 = try rms(x, layer.attention_norm1, c.norm_eps, s);
        defer _ = mlx.mlx_array_free(an1);
        const am = try mulA(an1, scale_msa, s);
        defer _ = mlx.mlx_array_free(am);
        const attn_out = try self.attention(am, layer, rope);
        defer _ = mlx.mlx_array_free(attn_out);
        const an2 = try rms(attn_out, layer.attention_norm2, c.norm_eps, s);
        defer _ = mlx.mlx_array_free(an2);
        const ag = try mulA(an2, gate_msa, s);
        defer _ = mlx.mlx_array_free(ag);
        const x1 = try addA(x, ag, s);
        defer _ = mlx.mlx_array_free(x1);

        // Feed-forward branch: w2(silu(w1(h)) · w3(h)).
        const fn1 = try rms(x1, layer.ffn_norm1, c.norm_eps, s);
        defer _ = mlx.mlx_array_free(fn1);
        const fm = try mulA(fn1, scale_mlp, s);
        defer _ = mlx.mlx_array_free(fm);
        const g = try layer.w1.forward(fm, s);
        defer _ = mlx.mlx_array_free(g);
        const gs = try silu(g, s);
        defer _ = mlx.mlx_array_free(gs);
        const u = try layer.w3.forward(fm, s);
        defer _ = mlx.mlx_array_free(u);
        const gu = try mulA(gs, u, s);
        defer _ = mlx.mlx_array_free(gu);
        const ff = try layer.w2.forward(gu, s);
        defer _ = mlx.mlx_array_free(ff);
        const fn2 = try rms(ff, layer.ffn_norm2, c.norm_eps, s);
        defer _ = mlx.mlx_array_free(fn2);
        const fg = try mulA(fn2, gate_mlp, s);
        defer _ = mlx.mlx_array_free(fg);
        return addA(x1, fg, s);
    }

    /// QK-RMSNorm + MRoPE + full (non-causal) attention. Batch 1 with a single
    /// segment, so the reference's block-diagonal `segment_ids` mask is all
    /// True and carries no information — building it would be a 16k×16k array
    /// per layer for nothing.
    fn attention(self: *Transformer, x: mlx.mlx_array, layer: *const Block, rope: *const Rope) !mlx.mlx_array {
        const s = self.s;
        const c = self.cfg;
        const heads: c_int = @intCast(c.num_heads);
        const hd: c_int = @intCast(c.headDim());
        const sh = mlx.getShape(x);
        const seq: c_int = sh[1];

        const qkv = try layer.qkv.forward(x, s);
        defer _ = mlx.mlx_array_free(qkv);
        // view(B,L,3,H,D) then unbind(2) — contiguous thirds of the last axis.
        const q_raw = try slice3(qkv, 2, 0, heads * hd, s);
        defer _ = mlx.mlx_array_free(q_raw);
        const k_raw = try slice3(qkv, 2, heads * hd, 2 * heads * hd, s);
        defer _ = mlx.mlx_array_free(k_raw);
        const v_raw = try slice3(qkv, 2, 2 * heads * hd, 3 * heads * hd, s);
        defer _ = mlx.mlx_array_free(v_raw);

        const q4 = try reshape(q_raw, &[_]c_int{ 1, seq, heads, hd }, s);
        defer _ = mlx.mlx_array_free(q4);
        const qn = try rms(q4, layer.norm_q, c.norm_eps, s);
        defer _ = mlx.mlx_array_free(qn);
        const k4 = try reshape(k_raw, &[_]c_int{ 1, seq, heads, hd }, s);
        defer _ = mlx.mlx_array_free(k4);
        const kn = try rms(k4, layer.norm_k, c.norm_eps, s);
        defer _ = mlx.mlx_array_free(kn);
        const v4 = try reshape(v_raw, &[_]c_int{ 1, seq, heads, hd }, s);
        defer _ = mlx.mlx_array_free(v4);

        const qt = try transpose(qn, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(qt);
        const kt = try transpose(kn, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(kt);
        const vt = try transpose(v4, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(vt);

        const qr = try applyRope(qt, rope, s);
        defer _ = mlx.mlx_array_free(qr);
        const kr = try applyRope(kt, rope, s);
        defer _ = mlx.mlx_array_free(kr);

        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(c.headDim())));
        var attn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn);
        const null_arr = mlx.mlx_array{ .ctx = null };
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qr, kr, vt, scale, "", null_arr, null_arr, false, s));
        const at = try transpose(attn, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(at);
        const af = try reshape(at, &[_]c_int{ 1, seq, heads * hd }, s);
        defer _ = mlx.mlx_array_free(af);
        return layer.o.forward(af, s);
    }
};

/// q·cos + rotate_half(q)·sin, with `rotate_half` = cat(−x₂, x₁) over halves
/// of the head dim (the NeoX convention, not the interleaved one).
fn applyRope(x: mlx.mlx_array, rope: *const Rope, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const hd = sh[3];
    const half = @divExact(hd, 2);
    const x1 = try sliceLast4(x, 0, half, s);
    defer _ = mlx.mlx_array_free(x1);
    const x2 = try sliceLast4(x, half, hd, s);
    defer _ = mlx.mlx_array_free(x2);
    var neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg);
    try mlx.check(mlx.mlx_negative(&neg, x2, s));
    const rot = try concat(&[_]mlx.mlx_array{ neg, x1 }, 3, s);
    defer _ = mlx.mlx_array_free(rot);
    const a = try mulA(x, rope.cos, s);
    defer _ = mlx.mlx_array_free(a);
    const b = try mulA(rot, rope.sin, s);
    defer _ = mlx.mlx_array_free(b);
    return addA(a, b, s);
}

/// Build the MRoPE cos/sin tables for `n_text` text tokens followed by a
/// `grid_h × grid_w` image grid.
///
/// The interleave is the reference's, verbatim and deliberately odd: the t
/// frequencies fill the whole table, then H overwrites indices 1,4,…<3·s[1]
/// and W overwrites 2,5,…<3·s[2]. `mrope_section[0]` never appears — the
/// first section is whatever those two loops leave behind, so a "fix" that
/// partitions the table by all three sections is a different model.
pub fn buildRope(allocator: std.mem.Allocator, cfg: Config, n_text: usize, grid_h: u32, grid_w: u32, s: S) !Rope {
    const hd = cfg.headDim();
    const half = hd / 2;
    const n_img: usize = @as(usize, grid_h) * @as(usize, grid_w);
    const total = n_text + n_img;

    const cos_buf = try allocator.alloc(f32, total * hd);
    defer allocator.free(cos_buf);
    const sin_buf = try allocator.alloc(f32, total * hd);
    defer allocator.free(sin_buf);

    const inv_freq = try allocator.alloc(f64, half);
    defer allocator.free(inv_freq);
    for (0..half) |i| {
        const e = @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(hd));
        inv_freq[i] = 1.0 / std.math.pow(f64, cfg.rope_theta, e);
    }

    // Which position axis each of the first `half` frequency slots reads.
    const axis_of = try allocator.alloc(u2, half);
    defer allocator.free(axis_of);
    @memset(axis_of, 0);
    for (1..3) |axis| {
        const length: usize = @min(@as(usize, cfg.mrope_section[axis]) * 3, half);
        var idx: usize = axis;
        while (idx < length) : (idx += 3) axis_of[idx] = @intCast(axis);
    }

    for (0..total) |row| {
        // (t, h, w) for this token.
        var pos = [3]f64{ 0, 0, 0 };
        if (row < n_text) {
            const p: f64 = @floatFromInt(row);
            pos = .{ p, p, p };
        } else {
            const k = row - n_text;
            const off: f64 = @floatFromInt(image_position_offset);
            pos = .{
                off,
                off + @as(f64, @floatFromInt(k / grid_w)),
                off + @as(f64, @floatFromInt(k % grid_w)),
            };
        }
        for (0..half) |i| {
            const ang = pos[axis_of[i]] * inv_freq[i];
            const cv: f32 = @floatCast(@cos(ang));
            const sv: f32 = @floatCast(@sin(ang));
            // emb = cat(freqs, freqs): both halves of the head dim share it.
            cos_buf[row * hd + i] = cv;
            cos_buf[row * hd + half + i] = cv;
            sin_buf[row * hd + i] = sv;
            sin_buf[row * hd + half + i] = sv;
        }
    }

    const shape = [_]c_int{ 1, 1, @intCast(total), @intCast(hd) };
    const cos_raw = mlx.mlx_array_new_data(cos_buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(cos_raw);
    const sin_raw = mlx.mlx_array_new_data(sin_buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(sin_raw);
    return .{
        .cos = try astype(cos_raw, .bfloat16, s),
        .sin = try astype(sin_raw, .bfloat16, s),
        .text_len = n_text,
        .image_len = n_img,
    };
}

/// The image-only tail of a packed rope table, for the unconditional branch.
/// Non-owning views would alias the parent, so this copies the slice.
pub fn imageOnlyRope(rope: *const Rope, s: S) !Rope {
    const start: c_int = @intCast(rope.text_len);
    const stop: c_int = @intCast(rope.text_len + rope.image_len);
    const cut = struct {
        fn go(x: mlx.mlx_array, lo: c_int, hi: c_int, st: S) !mlx.mlx_array {
            const sh = mlx.getShape(x);
            const a = [_]c_int{ 0, 0, lo, 0 };
            const b = [_]c_int{ sh[0], sh[1], hi, sh[3] };
            const stride = [_]c_int{ 1, 1, 1, 1 };
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&o, x, &a, 4, &b, 4, &stride, 4, st));
            const c = try contig(o, st);
            _ = mlx.mlx_array_free(o);
            return c;
        }
    };
    return .{
        .cos = try cut.go(rope.cos, start, stop, s),
        .sin = try cut.go(rope.sin, start, stop, s),
        .text_len = 0,
        .image_len = rope.image_len,
    };
}

// ── Loading ──

/// First-axis extent of a loaded weight, or 0 when absent.
fn rowsOf(w: *const Weights, key: []const u8) u32 {
    const a = w.get(key) orelse return 0;
    const sh = mlx.getShape(a);
    if (sh.len == 0 or sh[0] < 0) return 0;
    return @intCast(sh[0]);
}

/// Highest `i` for which `fmt`-with-`{d}` names a present weight.
fn countIndexed(w: *const Weights, a: std.mem.Allocator, comptime fmt: []const u8) u32 {
    var n: u32 = 0;
    while (n < 256) : (n += 1) {
        const key = fmtKey(a, fmt, .{n}) catch return n;
        defer a.free(key);
        if (w.get(key) == null) return n;
    }
    return n;
}

/// Geometry read off the checkpoint. Zero = unreadable, which leaves the field
/// at its default and lets the subsequent load name what is actually missing.
fn configFrom(w: *const Weights, a: std.mem.Allocator, base: Config) Config {
    var c = base;
    const emb = rowsOf(w, "layers.0.attention_norm1.weight");
    if (emb != 0) c.emb_dim = emb;
    const hd = rowsOf(w, "layers.0.attention.norm_q.weight");
    if (hd != 0) c.num_heads = c.emb_dim / hd;
    const inter = rowsOf(w, "layers.0.feed_forward.w1.weight");
    if (inter != 0 and w.get("layers.0.feed_forward.w1.scales") == null) c.intermediate_size = inter;
    const adaln = rowsOf(w, "adaln_proj.weight");
    if (adaln != 0 and w.get("adaln_proj.scales") == null) c.adaln_dim = adaln;
    const llm = rowsOf(w, "llm_cond_norm.weight");
    if (llm != 0) c.llm_features_dim = llm;
    const in_ch = rowsOf(w, "final_layer.linear.weight");
    if (in_ch != 0 and w.get("final_layer.linear.scales") == null) c.in_channels = in_ch;
    const n = countIndexed(w, a, "layers.{d}.attention_norm1.weight");
    if (n != 0) c.num_layers = n;
    return c;
}

/// Load one Ideogram 4 transformer from `<model_dir>/<subdir>`.
pub fn loadTransformer(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8, subdir: []const u8) !Transformer {
    const dir = try fmtKey(allocator, "{s}/{s}", .{ model_dir, subdir });
    defer allocator.free(dir);
    var w = try model_mod.loadWeights(io, allocator, dir);
    defer w.deinit();

    var d: Transformer = undefined;
    d.cfg = configFrom(&w, allocator, .{});
    d.allocator = allocator;
    d.s = s;
    log.info("[ideogram4] {s}: emb={d} layers={d} heads={d}x{d} inter={d} adaln={d} llm={d}\n", .{
        subdir, d.cfg.emb_dim, d.cfg.num_layers, d.cfg.num_heads, d.cfg.headDim(), d.cfg.intermediate_size, d.cfg.adaln_dim, d.cfg.llm_features_dim,
    });

    const emb = d.cfg.emb_dim;
    d.input_proj = try IgLinear.load(&w, allocator, "input_proj", d.cfg.in_channels, s);
    d.llm_cond_norm = try ownWeight(&w, "llm_cond_norm.weight");
    d.llm_cond_proj = try IgLinear.load(&w, allocator, "llm_cond_proj", d.cfg.llm_features_dim, s);
    d.t_mlp_in = try IgLinear.load(&w, allocator, "t_embedding.mlp_in", emb, s);
    d.t_mlp_out = try IgLinear.load(&w, allocator, "t_embedding.mlp_out", emb, s);
    d.adaln_proj = try IgLinear.load(&w, allocator, "adaln_proj", emb, s);
    d.embed_image_indicator = try ownWeight(&w, "embed_image_indicator.weight");
    d.final_linear = try IgLinear.load(&w, allocator, "final_layer.linear", emb, s);
    d.final_adaln = try IgLinear.load(&w, allocator, "final_layer.adaln_modulation", d.cfg.adaln_dim, s);

    d.layers = try allocator.alloc(Block, d.cfg.num_layers);
    for (d.layers, 0..) |*b, i| {
        const pfx = try fmtKey(allocator, "layers.{d}", .{i});
        defer allocator.free(pfx);
        const ld = struct {
            fn lin(ww: *const Weights, aa: std.mem.Allocator, p: []const u8, sub: []const u8, in_f: u32, st: S) !IgLinear {
                const k = try fmtKey(aa, "{s}.{s}", .{ p, sub });
                defer aa.free(k);
                return IgLinear.load(ww, aa, k, in_f, st);
            }
            fn norm(ww: *const Weights, aa: std.mem.Allocator, p: []const u8, sub: []const u8) !mlx.mlx_array {
                const k = try fmtKey(aa, "{s}.{s}", .{ p, sub });
                defer aa.free(k);
                return ownWeight(ww, k);
            }
        };
        b.* = .{
            .qkv = try ld.lin(&w, allocator, pfx, "attention.qkv", emb, s),
            .o = try ld.lin(&w, allocator, pfx, "attention.o", emb, s),
            .norm_q = try ld.norm(&w, allocator, pfx, "attention.norm_q.weight"),
            .norm_k = try ld.norm(&w, allocator, pfx, "attention.norm_k.weight"),
            .w1 = try ld.lin(&w, allocator, pfx, "feed_forward.w1", emb, s),
            .w2 = try ld.lin(&w, allocator, pfx, "feed_forward.w2", d.cfg.intermediate_size, s),
            .w3 = try ld.lin(&w, allocator, pfx, "feed_forward.w3", emb, s),
            .attention_norm1 = try ld.norm(&w, allocator, pfx, "attention_norm1.weight"),
            .attention_norm2 = try ld.norm(&w, allocator, pfx, "attention_norm2.weight"),
            .ffn_norm1 = try ld.norm(&w, allocator, pfx, "ffn_norm1.weight"),
            .ffn_norm2 = try ld.norm(&w, allocator, pfx, "ffn_norm2.weight"),
            .adaln_modulation = try ld.lin(&w, allocator, pfx, "adaln_modulation", d.cfg.adaln_dim, s),
        };
    }
    return d;
}

// ── LoRA ──

/// Attach every adapter in `stack` to its matching linear. Returns the number
/// of (module, adapter) attachments — a module hit by two stacked LoRAs counts
/// twice, so it doubles as a "did anything match" signal.
pub fn attachLora(d: *Transformer, stack: *const lora_mod.Stack) u32 {
    detachLora(d);
    var matched: u32 = 0;
    var kbuf: [128]u8 = undefined;
    var rbuf: [lora_mod.MAX_LORAS]lora_mod.Ref = undefined;
    for (d.layers, 0..) |*b, i| {
        const mods = .{
            .{ "attention.qkv", &b.qkv },
            .{ "attention.o", &b.o },
            .{ "feed_forward.w1", &b.w1 },
            .{ "feed_forward.w2", &b.w2 },
            .{ "feed_forward.w3", &b.w3 },
            .{ "adaln_modulation", &b.adaln_modulation },
        };
        inline for (mods) |m| {
            const key = std.fmt.bufPrint(&kbuf, "layers.{d}.{s}", .{ i, m[0] }) catch "";
            const refs = stack.findAll(key, &rbuf);
            if (refs.len > 0) {
                m[1].setLoraRefs(refs);
                matched += @intCast(refs.len);
            }
        }
    }
    return matched;
}

pub fn detachLora(d: *Transformer) void {
    for (d.layers) |*b| {
        inline for (.{ &b.qkv, &b.o, &b.w1, &b.w2, &b.w3, &b.adaln_modulation }) |l| l.clearLoraRefs();
    }
}

// ── Generation ──

pub const GenOpts = struct {
    steps: u32 = 20,
    /// img2img: the source's latent MEAN [1,32,H/8,W/8] f32 from
    /// `flux.VaeEncoder.encodeLatent`. Patchified and normalized here — the
    /// packing order and the latent table are Ideogram's, not FLUX's.
    init_latent: ?mlx.mlx_array = null,
    /// How far to renoise the source (diffusers convention: 1 = ignore it,
    /// low = a small change). Only read with `init_latent`.
    strength: f32 = 0.6,
    guidance: f32 = 7.0,
    /// Trailing polish steps run at `guidance_cleanup`; -1 = the preset's.
    guidance_cleanup: f32 = 3.0,
    cleanup_steps: ?u32 = null,
    mu: ?f64 = null,
    sigma: ?f64 = null,
};

/// Stage 1 alone: the 13-tap Qwen3-VL encode, MATERIALIZED. The eval matters —
/// mlx is lazy, and a low-mem host frees the 8B encoder right after this
/// returns; an unevaluated graph would pin its weights through refcounts and
/// the free would reclaim nothing.
pub fn encodePrompt(te: *flux.TextEncoder, ids: []const i32, mask: []const i32) !mlx.mlx_array {
    const enc = try te.encode(ids, mask);
    _ = mlx.mlx_array_eval(enc);
    return enc;
}

/// The full pipeline: encode → denoise (cond + uncond per step) → VAE decode.
/// Returns [1,3,H,W] f32 in [0,1], matching every other image backend.
pub fn generate(
    te: *flux.TextEncoder,
    cond: *Transformer,
    uncond: *Transformer,
    vae: *flux.Vae,
    ids: []const i32,
    mask: []const i32,
    seed: u64,
    height: u32,
    width: u32,
    opts: GenOpts,
    progress: ?sse.Progress,
) !mlx.mlx_array {
    if (progress) |p| p.emit("Encoding prompt", 0, @max(opts.steps, 1));
    const enc = try encodePrompt(te, ids, mask);
    return generateFromCond(cond, uncond, vae, enc, ids.len, seed, height, width, opts, progress);
}

/// Stages 2+ from a pre-computed conditioning. TAKES OWNERSHIP of `enc_owned`.
/// `n_text` is the token count it was built from — it drives both the MRoPE
/// text span and where the image span starts in the packed sequence, so it can
/// never be re-derived from the array alone.
pub fn generateFromCond(
    cond: *Transformer,
    uncond: *Transformer,
    vae: *flux.Vae,
    enc_owned: mlx.mlx_array,
    n_text: usize,
    seed: u64,
    height: u32,
    width: u32,
    opts: GenOpts,
    progress: ?sse.Progress,
) !mlx.mlx_array {
    const s = cond.s;
    const a = cond.allocator;
    const enc = enc_owned;
    defer _ = mlx.mlx_array_free(enc);

    if (height % dim_multiple != 0 or width % dim_multiple != 0) return error.BadDimensions;
    if (n_text > max_text_tokens) return error.PromptTooLong;
    const grid_h = height / dim_multiple;
    const grid_w = width / dim_multiple;
    const n_img: usize = @as(usize, grid_h) * @as(usize, grid_w);

    const preset = SamplerPreset.forSteps(opts.steps).params();
    const steps = @max(opts.steps, 1);
    const cleanup = @min(opts.cleanup_steps orelse preset.cleanup_steps, steps);
    const sched = LogitNormalSchedule.forResolution(
        height,
        width,
        opts.mu orelse preset.mu,
        opts.sigma orelse preset.std,
    );

    var rope = try buildRope(a, cond.cfg, n_text, grid_h, grid_w, s);
    defer rope.deinit();
    var img_rope = try imageOnlyRope(&rope, s);
    defer img_rope.deinit();

    // Latents: [1, n_img, in_channels] standard normal.
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    const nsh = [_]c_int{ 1, @intCast(n_img), @intCast(cond.cfg.in_channels) };
    var z = mlx.mlx_array_new();
    errdefer if (z.ctx != null) {
        _ = mlx.mlx_array_free(z);
    };
    try mlx.check(mlx.mlx_random_normal(&z, &nsh, 3, .float32, 0.0, 1.0, key, s));

    // img2img. This schedule runs BACKWARDS from every other one here: t = 0
    // is noise and t = 1 is data (the loop walks `schedule(1)` up to
    // `schedule(0)`, and `z += v·(s−t)` with a POSITIVE step). So the flow
    // interpolation is x_t = (1−t)·noise + t·x₁, and starting late means
    // starting at a LARGE t — the opposite of the sigma convention in
    // `flux.zig`. Getting this backwards yields noise at strength 0.2 and a
    // copy of the source at 1.0, both of which look like a plausible bug
    // somewhere else.
    var run_steps = steps;
    if (opts.init_latent) |src| {
        run_steps = @max(strengthRunSteps(steps, opts.strength), 1);
        const z0 = try patchify(src, grid_h, grid_w, cond.cfg.in_channels, s);
        defer _ = mlx.mlx_array_free(z0);
        const t0: f32 = @floatCast(sched.at(@as(f64, @floatFromInt(run_steps)) / @as(f64, @floatFromInt(steps))));
        const ta = mlx.mlx_array_new_float(t0);
        defer _ = mlx.mlx_array_free(ta);
        const oma = mlx.mlx_array_new_float(1.0 - t0);
        defer _ = mlx.mlx_array_free(oma);
        const zs = try mulA(z0, ta, s);
        defer _ = mlx.mlx_array_free(zs);
        const ns = try mulA(z, oma, s);
        defer _ = mlx.mlx_array_free(ns);
        const mixed = try addA(zs, ns, s);
        _ = mlx.mlx_array_free(z);
        z = mixed;
        log.info("[ideogram4] img2img: strength={d:.2} -> {d}/{d} steps from t={d:.4}\n", .{ opts.strength, run_steps, steps, t0 });
    }

    // Loop index i counts DOWN; index 0 is the final polish step. img2img
    // starts partway down, which is why the guidance schedule is indexed by
    // `i` (its polish steps are the LAST ones) and the progress by `run_steps`.
    var i: u32 = run_steps;
    while (i > 0) {
        i -= 1;
        if (progress) |p| if (p.cancelled()) return error.Cancelled;
        const t_val = sched.at(@as(f64, @floatFromInt(i + 1)) / @as(f64, @floatFromInt(steps)));
        const s_val = sched.at(@as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(steps)));

        const pos_v = try cond.forward(enc, z, t_val, &rope);
        defer _ = mlx.mlx_array_free(pos_v);
        const neg_v = try uncond.forward(null, z, t_val, &img_rope);
        defer _ = mlx.mlx_array_free(neg_v);

        const gw: f32 = if (i < cleanup) opts.guidance_cleanup else opts.guidance;
        const gwa = mlx.mlx_array_new_float(gw);
        defer _ = mlx.mlx_array_free(gwa);
        const omg = mlx.mlx_array_new_float(1.0 - gw);
        defer _ = mlx.mlx_array_free(omg);
        const pw = try mulA(pos_v, gwa, s);
        defer _ = mlx.mlx_array_free(pw);
        const nw = try mulA(neg_v, omg, s);
        defer _ = mlx.mlx_array_free(nw);
        const v = try addA(pw, nw, s);
        defer _ = mlx.mlx_array_free(v);

        const dta = mlx.mlx_array_new_float(@floatCast(s_val - t_val));
        defer _ = mlx.mlx_array_free(dta);
        const step = try mulA(v, dta, s);
        defer _ = mlx.mlx_array_free(step);
        const nz = try addA(z, step, s);
        _ = mlx.mlx_array_free(z);
        z = nz;
        _ = mlx.mlx_array_eval(z);
        if (progress) |p| p.emit("Generating", run_steps - i, run_steps);
    }
    if (progress) |p| p.emit("Decoding image", run_steps, run_steps);

    const latent = try unpatchify(z, grid_h, grid_w, cond.cfg.in_channels, s);
    _ = mlx.mlx_array_free(z);
    z = .{ .ctx = null };
    defer _ = mlx.mlx_array_free(latent);

    const decoded = try vae.decodeLatent(latent);
    defer _ = mlx.mlx_array_free(decoded);
    // [-1,1] → [0,1], matching flux/krea/mage_flow.
    const half = mlx.mlx_array_new_float(0.5);
    defer _ = mlx.mlx_array_free(half);
    const df = try astype(decoded, .float32, s);
    defer _ = mlx.mlx_array_free(df);
    const scaled = try mulA(df, half, s);
    defer _ = mlx.mlx_array_free(scaled);
    const shifted = try addA(scaled, half, s);
    defer _ = mlx.mlx_array_free(shifted);
    const lo = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(lo);
    const hi = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(hi);
    var clo = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(clo);
    try mlx.check(mlx.mlx_maximum(&clo, shifted, lo, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_minimum(&out, clo, hi, s));
    return out;
}

/// Denormalize with Ideogram's published latent table, then unpack
/// [1, gh·gw, 128] → [1, 32, gh·2, gw·2].
///
/// The packing order is [ph, pw, channel] — the OPPOSITE of FLUX.2's
/// [channel, ph, pw], which is why `flux.unpatchify` is not reused here. Same
/// shapes either way; the wrong one is a 2×2-scrambled image.
fn unpatchify(z: mlx.mlx_array, grid_h: u32, grid_w: u32, in_channels: u32, s: S) !mlx.mlx_array {
    const patch: c_int = 2;
    const ae_ch: c_int = @intCast(in_channels / 4);
    const shape = [_]c_int{ 1, 1, @intCast(in_channels) };
    const shift_raw = mlx.mlx_array_new_data(&latent_norm.latent_shift, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(shift_raw);
    const scale_raw = mlx.mlx_array_new_data(&latent_norm.latent_scale, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(scale_raw);
    const zs = try mulA(z, scale_raw, s);
    defer _ = mlx.mlx_array_free(zs);
    const zn = try addA(zs, shift_raw, s);
    defer _ = mlx.mlx_array_free(zn);

    const r = try reshape(zn, &[_]c_int{ 1, @intCast(grid_h), @intCast(grid_w), patch, patch, ae_ch }, s);
    defer _ = mlx.mlx_array_free(r);
    const t = try transpose(r, &[_]c_int{ 0, 5, 1, 3, 2, 4 }, s);
    defer _ = mlx.mlx_array_free(t);
    const flat = try reshape(t, &[_]c_int{ 1, ae_ch, @intCast(grid_h * 2), @intCast(grid_w * 2) }, s);
    defer _ = mlx.mlx_array_free(flat);
    return contig(flat, s);
}

/// The inverse of `unpatchify`: a VAE latent [1, 32, gh·2, gw·2] → normalized
/// tokens [1, gh·gw, 128]. Same [ph, pw, channel] packing and the same
/// published table, run backwards — `(x − shift) / scale`.
fn patchify(latent: mlx.mlx_array, grid_h: u32, grid_w: u32, in_channels: u32, s: S) !mlx.mlx_array {
    const patch: c_int = 2;
    const ae_ch: c_int = @intCast(in_channels / 4);
    const lf = try astype(latent, .float32, s);
    defer _ = mlx.mlx_array_free(lf);
    const r = try reshape(lf, &[_]c_int{ 1, ae_ch, @intCast(grid_h), patch, @intCast(grid_w), patch }, s);
    defer _ = mlx.mlx_array_free(r);
    // Inverse of unpatchify's {0,5,1,3,2,4}.
    const t = try transpose(r, &[_]c_int{ 0, 2, 4, 3, 5, 1 }, s);
    defer _ = mlx.mlx_array_free(t);
    const flat = try reshape(t, &[_]c_int{ 1, @intCast(grid_h * grid_w), @intCast(in_channels) }, s);
    defer _ = mlx.mlx_array_free(flat);

    const shape = [_]c_int{ 1, 1, @intCast(in_channels) };
    const shift_raw = mlx.mlx_array_new_data(&latent_norm.latent_shift, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(shift_raw);
    const scale_raw = mlx.mlx_array_new_data(&latent_norm.latent_scale, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(scale_raw);
    var centered = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(centered);
    try mlx.check(mlx.mlx_subtract(&centered, flat, shift_raw, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_divide(&out, centered, scale_raw, s));
    return out;
}

/// How many of `steps` an img2img request actually runs. The diffusers
/// convention: strength 1 = the full schedule from pure noise, low = a few
/// steps and a small change. At least one step always runs.
pub fn strengthRunSteps(steps: u32, strength: f32) u32 {
    const fsteps: f32 = @floatFromInt(steps);
    const run: u32 = @intFromFloat(@round(fsteps * std.math.clamp(strength, 0.0, 1.0)));
    return std.math.clamp(run, 1, @max(steps, 1));
}

/// Round a requested side to the multiple the patchifier needs, inside the
/// model's own 256..2048 window.
pub fn clampDim(v: u32) u32 {
    const c = std.math.clamp(v, 256, 2048);
    return (c / dim_multiple) * dim_multiple;
}

// ── Tests (pure helpers only; the forward needs weights + Metal) ──

const testing = std.testing;

test "ndtri inverts the standard normal CDF at the reference's own anchors" {
    // Φ⁻¹(0.5)=0, Φ⁻¹(0.975)≈1.959964, Φ⁻¹(0.025)≈-1.959964, Φ⁻¹(0.8413…)≈1.
    try testing.expectApproxEqAbs(@as(f64, 0.0), ndtri(0.5), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 1.959963985), ndtri(0.975), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, -1.959963985), ndtri(0.025), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 1.0), ndtri(0.8413447461), 1e-6);
    // The tails Acklam switches branches on, either side of p_low = 0.02425.
    try testing.expectApproxEqAbs(@as(f64, -3.090232306), ndtri(0.001), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 3.090232306), ndtri(0.999), 1e-6);
    try testing.expect(std.math.isNegativeInf(ndtri(0.0)));
    try testing.expect(std.math.isPositiveInf(ndtri(1.0)));
}

test "the logit-normal schedule runs high→low and clamps at both endpoints" {
    const sch = LogitNormalSchedule.forResolution(1024, 1024, 0.0, 1.75);
    const t_min = 1.0 / (1.0 + @exp(0.5 * 18.0));
    const t_max = 1.0 / (1.0 + @exp(0.5 * -15.0));
    try testing.expectApproxEqAbs(t_max, sch.at(0.0), 1e-12);
    try testing.expectApproxEqAbs(t_min, sch.at(1.0), 1e-12);
    // Monotone decreasing in between — the loop walks it from 1 down to 0.
    var prev: f64 = 2.0;
    var i: u32 = 0;
    while (i <= 20) : (i += 1) {
        const v = sch.at(@as(f64, @floatFromInt(i)) / 20.0);
        try testing.expect(v <= prev);
        try testing.expect(v >= t_min and v <= t_max);
        prev = v;
    }
}

test "the schedule's mean slides with the pixel count, not the step count" {
    // known_mean + 0.5·ln(pixels / 512²): a 1024² render is 2× the area of a
    // 724², so the offsets differ by exactly 0.5·ln(4) at 512²→1024².
    const half = LogitNormalSchedule.forResolution(512, 512, 0.0, 1.0);
    const full = LogitNormalSchedule.forResolution(1024, 1024, 0.0, 1.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), half.mean, 1e-12);
    try testing.expectApproxEqAbs(0.5 * @log(4.0), full.mean, 1e-12);
    // mu rides on top of it rather than replacing it.
    const with_mu = LogitNormalSchedule.forResolution(1024, 1024, 0.5, 1.0);
    try testing.expectApproxEqAbs(0.5 + full.mean, with_mu.mean, 1e-12);
}

test "guidance schedule is loop-index ordered: the LAST steps polish at 3.0" {
    const p = SamplerPreset.default_20.params();
    try testing.expectEqual(@as(u32, 20), p.num_steps);
    // Index 0 is the final step; indices 0..cleanup-1 are the polish steps.
    try testing.expectEqual(@as(f32, 3.0), p.guidanceAt(0));
    try testing.expectEqual(@as(f32, 3.0), p.guidanceAt(1));
    try testing.expectEqual(@as(f32, 7.0), p.guidanceAt(2));
    try testing.expectEqual(@as(f32, 7.0), p.guidanceAt(19));
    // Every preset's cleanup count matches the reference's schedules.
    try testing.expectEqual(@as(u32, 1), SamplerPreset.turbo_12.params().cleanup_steps);
    try testing.expectEqual(@as(u32, 3), SamplerPreset.quality_48.params().cleanup_steps);
}

test "a step count picks the preset whose mu/std were measured with it" {
    try testing.expectEqual(SamplerPreset.turbo_12, SamplerPreset.forSteps(12));
    try testing.expectEqual(SamplerPreset.default_20, SamplerPreset.forSteps(20));
    try testing.expectEqual(SamplerPreset.quality_48, SamplerPreset.forSteps(48));
    // Turbo's mu is the odd one out (0.5); mismatching it with 48 steps is the
    // bug this mapping exists to prevent.
    try testing.expectEqual(@as(f64, 0.5), SamplerPreset.turbo_12.params().mu);
    try testing.expectEqual(@as(f64, 0.0), SamplerPreset.quality_48.params().mu);
}

test "the tap list is the reference's activation layers, shifted by one slot" {
    // flux.TextEncoder counts capture slots where slot 0 is the embedding; the
    // reference counts decoder-layer outputs. Off by exactly one, everywhere.
    try testing.expectEqual(qwen3_vl_activation_layers.len, qwen3_vl_taps.len);
    try testing.expectEqual(@as(usize, 13), qwen3_vl_taps.len);
    for (qwen3_vl_activation_layers, qwen3_vl_taps) |ref, ours| {
        try testing.expectEqual(ref + 1, ours);
    }
    // 13 taps × Qwen3-VL-8B's 4096 hidden is the DiT's declared input width.
    const cfg = Config{};
    try testing.expectEqual(@as(u32, 4096 * 13), cfg.llm_features_dim);
    try testing.expectEqual(@as(u32, 256), cfg.headDim());
}

test "MRoPE fills from t, then H and W overwrite every third slot" {
    // Reproduces the reference's interleave without MLX: axis 0 everywhere,
    // then indices offset..<3·section[axis] step 3 for axes 1 and 2.
    const cfg = Config{};
    const half = cfg.headDim() / 2; // 128
    var axis_of: [128]u2 = @splat(0);
    for (1..3) |axis| {
        const length: usize = @min(@as(usize, cfg.mrope_section[axis]) * 3, half);
        var idx: usize = axis;
        while (idx < length) : (idx += 3) axis_of[idx] = @intCast(axis);
    }
    // Sections [1] and [2] are both 20 → 60 interleaved slots, then all t.
    try testing.expectEqual(@as(u2, 0), axis_of[0]);
    try testing.expectEqual(@as(u2, 1), axis_of[1]);
    try testing.expectEqual(@as(u2, 2), axis_of[2]);
    try testing.expectEqual(@as(u2, 0), axis_of[3]);
    try testing.expectEqual(@as(u2, 1), axis_of[58]);
    try testing.expectEqual(@as(u2, 2), axis_of[59]);
    // Past 3·20 nothing is overwritten — `mrope_section[0]` is never read, and
    // a "fix" that partitions all 128 slots by all three sections is a
    // different model.
    for (60..128) |i| try testing.expectEqual(@as(u2, 0), axis_of[i]);
    var counts = [_]usize{ 0, 0, 0 };
    for (axis_of) |a| counts[a] += 1;
    try testing.expectEqual(@as(usize, 88), counts[0]);
    try testing.expectEqual(@as(usize, 20), counts[1]);
    try testing.expectEqual(@as(usize, 20), counts[2]);
}

test "image positions cannot collide with text positions" {
    // Text positions run 0..max_text_tokens; the image grid starts far above
    // it, and even a 2048² render's 128-wide grid stays clear of the next
    // decade. Both halves of the guarantee, not just the offset's value.
    try testing.expect(image_position_offset > @as(i64, @intCast(max_text_tokens)));
    const max_grid: i64 = 2048 / dim_multiple;
    try testing.expect(image_position_offset - @as(i64, @intCast(max_text_tokens)) > max_grid);
}

test "clampDim keeps the model's own resolution window and patch multiple" {
    try testing.expectEqual(@as(u32, 1024), clampDim(1024));
    try testing.expectEqual(@as(u32, 1024), clampDim(1039)); // down to the multiple
    try testing.expectEqual(@as(u32, 256), clampDim(1)); // floor of the window
    try testing.expectEqual(@as(u32, 2048), clampDim(99999)); // 256–2048 per the card
    // Every result is a multiple of patch_size·ae_scale_factor.
    for ([_]u32{ 1, 255, 257, 700, 1023, 1536, 4096 }) |v| {
        try testing.expectEqual(@as(u32, 0), clampDim(v) % dim_multiple);
    }
}

test "latent norm table is 128 wide and matches the packed channel count" {
    const cfg = Config{};
    try testing.expectEqual(cfg.in_channels, @as(u32, latent_norm.latent_shift.len));
    try testing.expectEqual(cfg.in_channels, @as(u32, latent_norm.latent_scale.len));
    // A scale of zero would silently erase a channel; the published table has
    // none, and a transcription slip is exactly what that would look like.
    for (latent_norm.latent_scale) |v| try testing.expect(v > 0.0);
    for (latent_norm.latent_shift) |v| try testing.expect(std.math.isFinite(v));
}

// ── Numeric oracle (env-gated) ────────────────────────────────────────────

/// Cosine similarity of two f32 arrays, materialized on the host.
fn cosineOf(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !f64 {
    const af = try astype(a, .float32, s);
    defer _ = mlx.mlx_array_free(af);
    const bf = try astype(b, .float32, s);
    defer _ = mlx.mlx_array_free(bf);
    const ac = try contig(af, s);
    defer _ = mlx.mlx_array_free(ac);
    const bc = try contig(bf, s);
    defer _ = mlx.mlx_array_free(bc);
    _ = mlx.mlx_array_eval(ac);
    _ = mlx.mlx_array_eval(bc);
    const n = mlx.mlx_array_size(ac);
    if (n != mlx.mlx_array_size(bc)) return error.ShapeMismatch;
    const ap = mlx.mlx_array_data_float32(ac) orelse return error.NoData;
    const bp = mlx.mlx_array_data_float32(bc) orelse return error.NoData;
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (0..n) |i| {
        const x: f64 = ap[i];
        const y: f64 = bp[i];
        if (!std.math.isFinite(x) or !std.math.isFinite(y)) return error.NonFinite;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if (na == 0 or nb == 0) return error.ZeroNorm;
    return dot / (@sqrt(na) * @sqrt(nb));
}

/// RMS ratio (ours / reference). A cosine cannot see a SCALE error, and this
/// forward has three places one could hide — the `1 + scale` AdaLN offsets,
/// the attention scale, and the latent table — so both are asserted.
fn rmsRatioOf(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !f64 {
    const ac = try contig(try astype(a, .float32, s), s);
    defer _ = mlx.mlx_array_free(ac);
    const bc = try contig(try astype(b, .float32, s), s);
    defer _ = mlx.mlx_array_free(bc);
    _ = mlx.mlx_array_eval(ac);
    _ = mlx.mlx_array_eval(bc);
    const n = mlx.mlx_array_size(ac);
    const ap = mlx.mlx_array_data_float32(ac) orelse return error.NoData;
    const bp = mlx.mlx_array_data_float32(bc) orelse return error.NoData;
    var sa: f64 = 0;
    var sb: f64 = 0;
    for (0..n) |i| {
        sa += @as(f64, ap[i]) * ap[i];
        sb += @as(f64, bp[i]) * bp[i];
    }
    if (sb == 0) return error.ZeroNorm;
    return @sqrt(sa / sb);
}

// The reference oracle. Built by `tests/dump_ideogram4_fixtures.py` from a
// TINY randomly-initialised `Ideogram4Transformer` — the published weights are
// gated and 27 GB, and what needs pinning here is the arithmetic, not a
// checkpoint.
test "ideogram4 fixture" {
    const dir_c = std.c.getenv("IDEOGRAM4_FIXTURE_DIR") orelse return error.SkipZigTest;
    const dir_env = std.mem.sliceTo(dir_c, 0);
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();

    const meta_path = try std.fmt.allocPrint(a, "{s}/meta.json", .{dir_env});
    defer a.free(meta_path);
    const io = std.Io.Threaded.global_single_threaded.io();
    const meta_file = try std.Io.Dir.openFileAbsolute(io, meta_path, .{});
    defer meta_file.close(io);
    var meta_rb: [4096]u8 = undefined;
    var meta_rs = meta_file.reader(io, &meta_rb);
    const meta_bytes = try meta_rs.interface.allocRemaining(a, .limited(1 << 20));
    defer a.free(meta_bytes);
    var meta = try std.json.parseFromSlice(std.json.Value, a, meta_bytes, .{});
    defer meta.deinit();
    const m = meta.value.object;
    const n_text: usize = @intCast(m.get("n_text").?.integer);
    const grid_h: u32 = @intCast(m.get("grid_h").?.integer);
    const grid_w: u32 = @intCast(m.get("grid_w").?.integer);
    const t_val = m.get("t").?.float;

    var cfg = Config{
        .emb_dim = @intCast(m.get("emb_dim").?.integer),
        .num_layers = @intCast(m.get("num_layers").?.integer),
        .num_heads = @intCast(m.get("num_heads").?.integer),
        .intermediate_size = @intCast(m.get("intermediate_size").?.integer),
        .adaln_dim = @intCast(m.get("adaln_dim").?.integer),
        .in_channels = @intCast(m.get("in_channels").?.integer),
        .llm_features_dim = @intCast(m.get("llm_features_dim").?.integer),
        .rope_theta = @floatFromInt(m.get("rope_theta").?.integer),
    };
    const sec = m.get("mrope_section").?.array.items;
    for (sec, 0..) |v, i| cfg.mrope_section[i] = @intCast(v.integer);

    // 1. The rope tables, on their own. They are pure functions of the
    //    position ids and the single most transcription-prone part of this
    //    architecture; a mismatch here is otherwise diffuse — every layer is
    //    slightly wrong and the end-to-end cosine still looks plausible.
    var rope = try buildRope(a, cfg, n_text, grid_h, grid_w, s);
    defer rope.deinit();

    const ref_path = try std.fmt.allocPrint(a, "{s}/reference.safetensors", .{dir_env});
    defer a.free(ref_path);
    var ref = try model_mod.loadWeightsSingleFile(a, ref_path);
    defer ref.deinit();

    const ref_cos = ref.get("rope_cos") orelse return error.MissingFixture;
    const ref_sin = ref.get("rope_sin") orelse return error.MissingFixture;
    // Our tables are [1,1,L,D] to broadcast over heads; the reference's are
    // [L,D]. Same elements in the same order.
    const c_cos = try cosineOf(rope.cos, ref_cos, s);
    const c_sin = try cosineOf(rope.sin, ref_sin, s);
    std.debug.print("[ideogram4 fixture] rope cos={d:.6} sin={d:.6}\n", .{ c_cos, c_sin });
    // The tables are built in f32 on the host and only then cast to bf16, so
    // the bar is tight — a wrong INTERLEAVE shows up as ~0.5, not as 0.999.
    try testing.expect(c_cos > 0.9999);
    try testing.expect(c_sin > 0.9999);

    // 2. The full forward. The fixture ships f32 weights; `loadTransformer`
    //    reads them through the dense arm of `IgLinear` (no `.scales`), which
    //    is the same path a `--precision mixed` pack takes for its dense
    //    tensors.
    var dit = try loadTransformer(io, a, s, dir_env, ".");
    defer dit.deinit();

    const feat = ref.get("llm_features_text") orelse return error.MissingFixture;
    const z = ref.get("z_image") orelse return error.MissingFixture;
    const feat3 = try reshape(feat, &[_]c_int{ 1, @intCast(n_text), @intCast(cfg.llm_features_dim) }, s);
    defer _ = mlx.mlx_array_free(feat3);
    const z3 = try reshape(z, &[_]c_int{ 1, @intCast(grid_h * grid_w), @intCast(cfg.in_channels) }, s);
    defer _ = mlx.mlx_array_free(z3);

    const got = try dit.forward(feat3, z3, t_val, &rope);
    defer _ = mlx.mlx_array_free(got);
    const want = ref.get("velocity_image") orelse return error.MissingFixture;

    const cos_v = try cosineOf(got, want, s);
    const rms_v = try rmsRatioOf(got, want, s);
    std.debug.print("[ideogram4 fixture] velocity cos={d:.6} rms_ratio={d:.6}\n", .{ cos_v, rms_v });
    // bf16 activations against an f32 reference over 3 layers. A cosine cannot
    // see a scale error — the `1 + scale` AdaLN offsets, the attention scale
    // and the tanh gates are all places one could hide — so the RMS ratio is
    // asserted too, and the bar is tight because both are structural, not
    // numerical: getting them wrong is a factor, not a rounding.
    try testing.expect(cos_v > 0.995);
    try testing.expect(rms_v > 0.95 and rms_v < 1.05);
}

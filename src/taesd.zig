//! Real tiny-decoder previews: a Zig/MLX port of madebyollin/taesd's
//! `Decoder` network (https://github.com/madebyollin/taesd, MIT), NOT the
//! linear Latent2RGB approximation in latent_preview.zig. This is an actual
//! distilled conv net (~1.2M params) trained to reproduce the real VAE
//! decode — noticeably sharper previews than the linear projection, at
//! still-trivial cost (a handful of 3x3 convs).
//!
//! Two checkpoints matter here, both published by the TAESD author:
//!   - `madebyollin/taef1` — FLUX.1-family 16-channel latent (used for Krea).
//!   - `madebyollin/taef2` — FLUX.2's 32-channel real latent (used for
//!     FLUX.2-klein 4B/9B). The "no bn-denorm" claim this comment used to
//!     make turned out to be wrong: `flux.zig`'s own `Vae.decode` applies a
//!     per-channel batchnorm denorm (`bn_mean`/`bn_var`) to the packed
//!     latent BEFORE unpatchifying, and the preview call site now mirrors
//!     that (see `Vae.bnDenorm`) before handing this decoder its input.
//!     Still unverified against taef2's actual HF card / checkpoint bytes
//!     (no network access here) whether it expects that same denormalized
//!     space or something else — treat this decode path as provisional
//!     until it's been run against a real downloaded checkpoint and the
//!     output visually checked against a real VAE decode of the same step.
//!
//! Architecture (from taesd.py's `Decoder`/`Block`, reproduced faithfully):
//!   Clamp(tanh(x/3)*3) → conv_in(C→64) → ReLU
//!   → [Block×3 (+ midblock-GN "pool" residual, taef2 only)] → nearest-2x → conv(64→64, no bias)
//!   → [Block×3]                                              → nearest-2x → conv(64→64, no bias)
//!   → [Block×3]                                              → nearest-2x → conv(64→64, no bias)
//!   → Block×1 → conv_out(64→3)
//! Block(x) = ReLU(conv3(ReLU(conv2(ReLU(conv1(x))))) + x)   (in==out==64 always here, skip is identity)
//! Pool(x) [taef2 blocks 0-2 only] = x + conv1x1(GroupNorm(4, conv1x1(x)))  (added BEFORE the block's own conv path)
//!
//! WEIGHT KEY NAMES: taef1 ships as a proper diffusers `AutoencoderTiny`
//! state dict (`diffusion_pytorch_model.safetensors`). taef2 does NOT — its
//! HF card says its architecture "isn't properly integrated into Diffusers
//! yet", so it ships as a bare `taef2.safetensors` (no `config.json`
//! either). Confirmed since (the original guess below was made with no
//! network access) against the real madebyollin/taef2 model card: its own
//! `convert_diffusers_sd_to_taesd` remapper takes keys shaped
//! `"{encdec}.layers.{i}.{suffix}"` (decoder indices offset by +1 in that
//! remapper — i.e. -1 the other way, from taesd.py's own numbering, since
//! diffusers' Sequential doesn't carry the leading `Clamp()` as an indexed
//! module), meaning taef2's raw checkpoint already keys itself that way —
//! so the reconstructed candidate list below (first guess:
//! `"decoder.layers.{d}.{s}"`) matches. `resolveConv`/`resolveGn`
//! below try that pattern FIRST, then a couple of plausible fallbacks, and
//! `load` fails with the exact list of keys it tried (plus a dump of the
//! keys actually present) if none match — loud and diagnosable rather than
//! silently loading garbage. Treat this loader as unverified until it's
//! actually run against a downloaded checkpoint.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");

const Weights = model_mod.Weights;
const S = mlx.mlx_stream;

// ── mlx helpers (file-local; mirrors flux.zig/krea.zig/nsfw.zig) ──

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
inline fn addA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&o, a, b, s));
    return o;
}
inline fn contig(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&o, x, false, s));
    return o;
}
inline fn relu(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    try mlx.check(mlx.mlx_maximum(&o, x, zero, s));
    return o;
}
/// 3x3 conv, NHWC in, OHWI weight [out,3,3,in], pad=1, optional bias.
fn conv3x3(x: mlx.mlx_array, w: mlx.mlx_array, b: ?mlx.mlx_array, s: S) !mlx.mlx_array {
    const xc = try contig(x, s);
    defer _ = mlx.mlx_array_free(xc);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv2d(&o, xc, w, 1, 1, 1, 1, 1, 1, 1, s));
    const bb = b orelse return o;
    defer _ = mlx.mlx_array_free(o);
    return addA(o, bb, s);
}
/// 1x1 conv, no bias (both Pool convs are bias=False).
fn conv1x1(x: mlx.mlx_array, w: mlx.mlx_array, s: S) !mlx.mlx_array {
    const xc = try contig(x, s);
    defer _ = mlx.mlx_array_free(xc);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv2d(&o, xc, w, 1, 1, 0, 0, 1, 1, 1, s));
    return o;
}
/// PyTorch-compatible GroupNorm on NHWC [1,H,W,C], f32, + affine. Mirrors
/// flux.zig's VAE groupNorm but with a configurable group count / eps
/// (PyTorch's `nn.GroupNorm` default eps is 1e-5, not the VAE's 1e-6).
fn groupNorm(x: mlx.mlx_array, weight: mlx.mlx_array, bias: mlx.mlx_array, groups: c_int, eps: f32, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const h = sh[1];
    const w_ = sh[2];
    const c = sh[3];
    const cg = @divExact(c, groups);
    const xf = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    const r1 = try reshape(xf, &[_]c_int{ 1, h * w_, groups, cg }, s);
    defer _ = mlx.mlx_array_free(r1);
    const t1 = try transpose(r1, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(t1);
    const flat = try reshape(t1, &[_]c_int{ 1, groups, h * w_ * cg }, s);
    defer _ = mlx.mlx_array_free(flat);
    var mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mean);
    try mlx.check(mlx.mlx_mean_axis(&mean, flat, -1, true, s));
    const xc = blk: {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_subtract(&o, flat, mean, s));
        break :blk o;
    };
    defer _ = mlx.mlx_array_free(xc);
    var sq = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&sq, xc, xc, s));
    defer _ = mlx.mlx_array_free(sq);
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_mean_axis(&v, sq, -1, true, s));
    const epsa = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(epsa);
    var ve = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ve);
    try mlx.check(mlx.mlx_add(&ve, v, epsa, s));
    var rsq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rsq);
    try mlx.check(mlx.mlx_rsqrt(&rsq, ve, s));
    var norm = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&norm, xc, rsq, s));
    defer _ = mlx.mlx_array_free(norm);
    const b1 = try reshape(norm, &[_]c_int{ 1, groups, h * w_, cg }, s);
    defer _ = mlx.mlx_array_free(b1);
    const b2 = try transpose(b1, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(b2);
    const b3 = try reshape(b2, &[_]c_int{ 1, h, w_, c }, s);
    defer _ = mlx.mlx_array_free(b3);
    const wf = try astype(weight, .float32, s);
    defer _ = mlx.mlx_array_free(wf);
    const bf = try astype(bias, .float32, s);
    defer _ = mlx.mlx_array_free(bf);
    var sc = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&sc, b3, wf, s));
    defer _ = mlx.mlx_array_free(sc);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, sc, bf, s));
    return out;
}
/// nearest 2x upsample on NHWC, then conv3x3 (no bias) — mirrors flux.zig's
/// VAE `upsample` (nn.Upsample(scale_factor=2) + conv).
fn upsample2x(x: mlx.mlx_array, w: mlx.mlx_array, s: S) !mlx.mlx_array {
    var r1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r1);
    try mlx.check(mlx.mlx_repeat_axis(&r1, x, 2, 1, s));
    var r2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r2);
    try mlx.check(mlx.mlx_repeat_axis(&r2, r1, 2, 2, s));
    return conv3x3(r2, w, null, s);
}

// ── Weights ──

const Pool = struct {
    c1w: mlx.mlx_array, // 1x1, n_in -> n_in*4, no bias
    gnw: mlx.mlx_array,
    gnb: mlx.mlx_array, // GroupNorm(4, n_in*4)
    c2w: mlx.mlx_array, // 1x1, n_in*4 -> n_in, no bias

    fn deinit(self: *Pool) void {
        _ = mlx.mlx_array_free(self.c1w);
        _ = mlx.mlx_array_free(self.gnw);
        _ = mlx.mlx_array_free(self.gnb);
        _ = mlx.mlx_array_free(self.c2w);
    }
    fn forward(self: *const Pool, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const h0 = try conv1x1(x, self.c1w, s);
        defer _ = mlx.mlx_array_free(h0);
        const h1 = try groupNorm(h0, self.gnw, self.gnb, 4, 1e-5, s);
        defer _ = mlx.mlx_array_free(h1);
        const h2 = try relu(h1, s);
        defer _ = mlx.mlx_array_free(h2);
        return conv1x1(h2, self.c2w, s);
    }
};

const Block = struct {
    c1w: mlx.mlx_array,
    c1b: mlx.mlx_array,
    c2w: mlx.mlx_array,
    c2b: mlx.mlx_array,
    c3w: mlx.mlx_array,
    c3b: mlx.mlx_array,
    pool: ?Pool = null,

    fn deinit(self: *Block) void {
        inline for (.{ "c1w", "c1b", "c2w", "c2b", "c3w", "c3b" }) |f| _ = mlx.mlx_array_free(@field(self, f));
        if (self.pool) |*p| p.deinit();
    }
    /// n_in == n_out == 64 for every Block in this decoder, so `skip` is
    /// always identity (taesd.py only makes it a real 1x1 conv when the
    /// channel counts differ, which never happens past conv_in here).
    fn forward(self: *const Block, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        var xin = x;
        var xin_owned = false;
        if (self.pool) |*p| {
            const pooled = try p.forward(x, s);
            defer _ = mlx.mlx_array_free(pooled);
            xin = try addA(x, pooled, s);
            xin_owned = true;
        }
        defer if (xin_owned) {
            _ = mlx.mlx_array_free(xin);
        };
        const h1 = try conv3x3(xin, self.c1w, self.c1b, s);
        const a1 = try relu(h1, s);
        _ = mlx.mlx_array_free(h1);
        defer _ = mlx.mlx_array_free(a1);
        const h2 = try conv3x3(a1, self.c2w, self.c2b, s);
        const a2 = try relu(h2, s);
        _ = mlx.mlx_array_free(h2);
        defer _ = mlx.mlx_array_free(a2);
        const h3 = try conv3x3(a2, self.c3w, self.c3b, s);
        defer _ = mlx.mlx_array_free(h3);
        const summed = try addA(h3, xin, s);
        defer _ = mlx.mlx_array_free(summed);
        return relu(summed, s);
    }
};

pub const Variant = enum {
    /// FLUX.1-family, 16 real channels, no midblock GN (`madebyollin/taef1`).
    taef1,
    /// FLUX.2, 32 real channels (post-unpack), midblock GN on the first
    /// three blocks (`madebyollin/taef2`).
    taef2,

    fn channels(self: Variant) u32 {
        return switch (self) {
            .taef1 => 16,
            .taef2 => 32,
        };
    }
    fn hasMidblockGn(self: Variant) bool {
        return self == .taef2;
    }
};

pub const Decoder = struct {
    allocator: std.mem.Allocator,
    s: S,
    variant: Variant,
    conv_in_w: mlx.mlx_array,
    conv_in_b: mlx.mlx_array,
    blocks: [10]Block,
    up_w: [3]mlx.mlx_array, // nearest-2x-then-conv, no bias
    conv_out_w: mlx.mlx_array,
    conv_out_b: mlx.mlx_array,

    pub fn deinit(self: *Decoder) void {
        _ = mlx.mlx_array_free(self.conv_in_w);
        _ = mlx.mlx_array_free(self.conv_in_b);
        for (&self.blocks) |*b| b.deinit();
        for (&self.up_w) |w| _ = mlx.mlx_array_free(w);
        _ = mlx.mlx_array_free(self.conv_out_w);
        _ = mlx.mlx_array_free(self.conv_out_b);
    }

    /// latent [1,C,H,W] (any float dtype, C == variant.channels()) → RGB
    /// [1,3,8H,8W] f32 clipped to [0,1] (NCHW, caller frees). Uses the
    /// decoder's own stream (set at load time), not the caller's — mirrors
    /// nsfw.zig's `Classifier.classify`: an auxiliary model runs on its own
    /// dedicated stream regardless of which stream produced its input.
    pub fn decode(self: *const Decoder, latent: mlx.mlx_array) !mlx.mlx_array {
        const sh = mlx.getShape(latent);
        if (sh[1] != self.variant.channels()) return error.ChannelMismatch;
        const s = self.s;

        const lf = try astype(latent, .float32, s);
        defer _ = mlx.mlx_array_free(lf);
        // Clamp: tanh(x/3)*3
        const third = mlx.mlx_array_new_float(1.0 / 3.0);
        defer _ = mlx.mlx_array_free(third);
        var scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&scaled, lf, third, s));
        defer _ = mlx.mlx_array_free(scaled);
        var tanhed = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_tanh(&tanhed, scaled, s));
        defer _ = mlx.mlx_array_free(tanhed);
        const three = mlx.mlx_array_new_float(3.0);
        defer _ = mlx.mlx_array_free(three);
        var clamped = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&clamped, tanhed, three, s));
        defer _ = mlx.mlx_array_free(clamped);

        // NCHW -> NHWC
        const nhwc = try transpose(clamped, &[_]c_int{ 0, 2, 3, 1 }, s);
        defer _ = mlx.mlx_array_free(nhwc);

        var h = try conv3x3(nhwc, self.conv_in_w, self.conv_in_b, s);
        {
            const a = try relu(h, s);
            _ = mlx.mlx_array_free(h);
            h = a;
        }
        var bi: usize = 0;
        for (0..3) |group| {
            for (0..3) |_| {
                const nh = try self.blocks[bi].forward(h, s);
                _ = mlx.mlx_array_free(h);
                h = nh;
                bi += 1;
            }
            const us = try upsample2x(h, self.up_w[group], s);
            _ = mlx.mlx_array_free(h);
            h = us;
        }
        {
            const nh = try self.blocks[bi].forward(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
            bi += 1;
        }
        std.debug.assert(bi == 10);
        const out_nhwc = try conv3x3(h, self.conv_out_w, self.conv_out_b, s);
        _ = mlx.mlx_array_free(h);
        defer _ = mlx.mlx_array_free(out_nhwc);

        // clip [0,1]
        const lo = mlx.mlx_array_new_float(0.0);
        defer _ = mlx.mlx_array_free(lo);
        const hi = mlx.mlx_array_new_float(1.0);
        defer _ = mlx.mlx_array_free(hi);
        var clo = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_maximum(&clo, out_nhwc, lo, s));
        defer _ = mlx.mlx_array_free(clo);
        var chi = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_minimum(&chi, clo, hi, s));
        defer _ = mlx.mlx_array_free(chi);

        // NHWC -> NCHW, materialized (a lazy transpose reads back in source order)
        const nchw = try transpose(chi, &[_]c_int{ 0, 3, 1, 2 }, s);
        defer _ = mlx.mlx_array_free(nchw);
        return contig(nchw, s);
    }
};

// ── Loading ──

fn fmtKey(a: std.mem.Allocator, comptime f: []const u8, args: anytype) ![]u8 {
    return std.fmt.allocPrint(a, f, args);
}

/// Tries every plausible key convention for one conv weight (see module
/// docs) and returns the first hit, OIHW→OHWI-transposed for our conv2d.
/// `taesd_idx` is taesd.py's own `decoder.<N>` numbering.
fn resolveConvW(w: *const Weights, a: std.mem.Allocator, taesd_idx: u32, sub: []const u8, s: S) !mlx.mlx_array {
    const diffusers_idx = taesd_idx - 1; // decoder.layers.N doesn't carry the leading Clamp()
    const candidates = [_][]const u8{
        try fmtKey(a, "decoder.layers.{d}.{s}", .{ diffusers_idx, sub }),
        try fmtKey(a, "decoder.{d}.{s}", .{ taesd_idx, sub }),
        try fmtKey(a, "{d}.{s}", .{ taesd_idx, sub }),
        try fmtKey(a, "decoder.decoder.{d}.{s}", .{ taesd_idx, sub }),
    };
    defer for (candidates) |k| a.free(k);

    for (candidates) |k| {
        if (w.get(k)) |raw| {
            var owned = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&owned, raw));
            // PyTorch OIHW [out,in,kh,kw] -> our OHWI [out,kh,kw,in]
            const rsh = mlx.getShape(owned);
            if (rsh.len != 4) {
                _ = mlx.mlx_array_free(owned);
                return error.UnexpectedTensorRank;
            }
            log.info("[taesd]   decoder[{d}].{s} <- \"{s}\" OIHW={any}\n", .{ taesd_idx, sub, k, rsh });
            const t = try transpose(owned, &[_]c_int{ 0, 2, 3, 1 }, s);
            _ = mlx.mlx_array_free(owned);
            const c = try contig(t, s);
            _ = mlx.mlx_array_free(t);
            return c;
        }
    }
    logMissing(w, taesd_idx, sub, &candidates);
    return error.MissingTaesdWeight;
}

fn resolveBiasOrGn(w: *const Weights, a: std.mem.Allocator, taesd_idx: u32, sub: []const u8) !mlx.mlx_array {
    const diffusers_idx = taesd_idx - 1;
    const candidates = [_][]const u8{
        try fmtKey(a, "decoder.layers.{d}.{s}", .{ diffusers_idx, sub }),
        try fmtKey(a, "decoder.{d}.{s}", .{ taesd_idx, sub }),
        try fmtKey(a, "{d}.{s}", .{ taesd_idx, sub }),
        try fmtKey(a, "decoder.decoder.{d}.{s}", .{ taesd_idx, sub }),
    };
    defer for (candidates) |k| a.free(k);

    for (candidates) |k| {
        if (w.get(k)) |raw| {
            var owned = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&owned, raw));
            log.info("[taesd]   decoder[{d}].{s} <- \"{s}\" shape={any}\n", .{ taesd_idx, sub, k, mlx.getShape(owned) });
            return owned;
        }
    }
    logMissing(w, taesd_idx, sub, &candidates);
    return error.MissingTaesdWeight;
}

fn logMissing(w: *const Weights, idx: u32, sub: []const u8, tried: []const []const u8) void {
    log.err("[taesd] none of these keys were found for decoder[{d}].{s}:\n", .{ idx, sub });
    for (tried) |k| log.err("[taesd]   tried: {s}\n", .{k});
    log.err("[taesd] this checkpoint's real key names differ from what this loader guesses — see the module doc comment in src/taesd.zig for how to fix the candidate list. Loaded {d} tensors total; a few for reference:\n", .{w.count()});
    var it = w.map.iterator();
    var shown: u32 = 0;
    while (it.next()) |entry| {
        if (shown >= 8) break;
        log.err("[taesd]   have: {s}\n", .{entry.key_ptr.*});
        shown += 1;
    }
}

fn loadBlock(w: *const Weights, a: std.mem.Allocator, taesd_idx: u32, with_pool: bool, s: S) !Block {
    var b: Block = .{
        .c1w = try resolveConvW(w, a, taesd_idx, "conv.0.weight", s),
        .c1b = try resolveBiasOrGn(w, a, taesd_idx, "conv.0.bias"),
        .c2w = try resolveConvW(w, a, taesd_idx, "conv.2.weight", s),
        .c2b = try resolveBiasOrGn(w, a, taesd_idx, "conv.2.bias"),
        .c3w = try resolveConvW(w, a, taesd_idx, "conv.4.weight", s),
        .c3b = try resolveBiasOrGn(w, a, taesd_idx, "conv.4.bias"),
    };
    errdefer b.deinit();
    if (with_pool) {
        b.pool = .{
            .c1w = try resolveConvW(w, a, taesd_idx, "pool.0.weight", s),
            .gnw = try resolveBiasOrGn(w, a, taesd_idx, "pool.1.weight"),
            .gnb = try resolveBiasOrGn(w, a, taesd_idx, "pool.1.bias"),
            .c2w = try resolveConvW(w, a, taesd_idx, "pool.3.weight", s),
        };
    }
    return b;
}

/// Load `<model_dir>/<filename>` — `diffusion_pytorch_model.safetensors`
/// for taef1 (the diffusers `AutoencoderTiny` layout), `taef2.safetensors`
/// for taef2 (not diffusers-wrapped — see the HF card) — into a `Decoder`.
/// Fails loudly (see module docs) rather than loading garbage.
pub fn loadDecoder(allocator: std.mem.Allocator, model_dir: []const u8, variant: Variant, s: S) !Decoder {
    const filename = switch (variant) {
        .taef1 => "diffusion_pytorch_model.safetensors",
        .taef2 => "taef2.safetensors",
    };
    const path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ model_dir, filename });
    defer allocator.free(path);
    var w = try model_mod.loadWeightsSingleFile(allocator, path);
    defer w.deinit();

    // Weight-bearing taesd.py indices: 1 (conv_in), {3,4,5} (block group 1,
    // midblock-GN if taef2), 7 (upsample conv), {8,9,10} (group 2), 12,
    // {13,14,15} (group 3), 17, 18 (final single block), 19 (conv_out).
    var dec: Decoder = .{
        .allocator = allocator,
        .s = s,
        .variant = variant,
        .conv_in_w = try resolveConvW(&w, allocator, 1, "weight", s),
        .conv_in_b = try resolveBiasOrGn(&w, allocator, 1, "bias"),
        .blocks = undefined,
        .up_w = undefined,
        .conv_out_w = try resolveConvW(&w, allocator, 19, "weight", s),
        .conv_out_b = try resolveBiasOrGn(&w, allocator, 19, "bias"),
    };
    errdefer {
        _ = mlx.mlx_array_free(dec.conv_in_w);
        _ = mlx.mlx_array_free(dec.conv_in_b);
        _ = mlx.mlx_array_free(dec.conv_out_w);
        _ = mlx.mlx_array_free(dec.conv_out_b);
    }

    const block_indices = [10]u32{ 3, 4, 5, 8, 9, 10, 13, 14, 15, 18 };
    var built: usize = 0;
    errdefer for (dec.blocks[0..built]) |*b| b.deinit();
    for (block_indices, 0..) |idx, i| {
        const with_pool = variant.hasMidblockGn() and i < 3;
        dec.blocks[i] = try loadBlock(&w, allocator, idx, with_pool, s);
        built = i + 1;
    }

    const up_indices = [3]u32{ 7, 12, 17 };
    var up_built: usize = 0;
    errdefer for (dec.up_w[0..up_built]) |uw| { _ = mlx.mlx_array_free(uw); };
    for (up_indices, 0..) |idx, i| {
        dec.up_w[i] = try resolveConvW(&w, allocator, idx, "weight", s);
        up_built = i + 1;
    }

    log.info("[taesd] loaded {s} decoder from {s} ({d} channels)\n", .{ @tagName(variant), model_dir, variant.channels() });
    return dec;
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "Decoder.decode rejects a channel-count mismatch" {
    // A zeroed-out (never loaded) decoder is fine for this check: the
    // mismatch is caught before any weight is touched (in particular,
    // before `self.s` is ever read).
    var dec: Decoder = undefined;
    dec.variant = .taef1;
    var buf: [1 * 4 * 2 * 2]f32 = @splat(0);
    const sh = [_]c_int{ 1, 4, 2, 2 };
    const lat = mlx.mlx_array_new_data(&buf, &sh, 4, .float32);
    defer _ = mlx.mlx_array_free(lat);
    try testing.expectError(error.ChannelMismatch, dec.decode(lat));
}

test "Variant channel counts and midblock-gn flags" {
    try testing.expectEqual(@as(u32, 16), Variant.taef1.channels());
    try testing.expectEqual(@as(u32, 32), Variant.taef2.channels());
    try testing.expect(!Variant.taef1.hasMidblockGn());
    try testing.expect(Variant.taef2.hasMidblockGn());
}

test "Block.forward on an identity-ish weight set is a shape no-op" {
    // Hermetic architecture check (no real checkpoint needed): a Block with
    // c1/c2 zeroed and c3 zeroed reduces to relu(0 + x) == relu(x), i.e. the
    // wiring (pool-add, 3-conv path, skip, final relu) is exercised without
    // needing real distilled weights.
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const C = 4;
    const H = 3;
    const W = 3;
    var zeros_w1: [C * 3 * 3 * C]f32 = @splat(0);
    var zeros_b: [C]f32 = @splat(0);
    const wsh = [_]c_int{ C, 3, 3, C };
    const bsh = [_]c_int{C};
    const zw = mlx.mlx_array_new_data(&zeros_w1, &wsh, 4, .float32);
    defer _ = mlx.mlx_array_free(zw);
    const zb = mlx.mlx_array_new_data(&zeros_b, &bsh, 1, .float32);
    defer _ = mlx.mlx_array_free(zb);

    const dupA = struct {
        fn f(a: mlx.mlx_array) mlx.mlx_array {
            var o = mlx.mlx_array_new();
            mlx.check(mlx.mlx_array_set(&o, a)) catch unreachable;
            return o;
        }
    }.f;

    var block: Block = .{
        .c1w = dupA(zw), .c1b = dupA(zb),
        .c2w = dupA(zw), .c2b = dupA(zb),
        .c3w = dupA(zw), .c3b = dupA(zb),
    };
    defer block.deinit();

    var buf: [1 * H * W * C]f32 = undefined;
    for (&buf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.1 - 1.0; // includes negatives
    const xsh = [_]c_int{ 1, H, W, C };
    const x = mlx.mlx_array_new_data(&buf, &xsh, 4, .float32);
    defer _ = mlx.mlx_array_free(x);

    const out = try block.forward(x, s);
    defer _ = mlx.mlx_array_free(out);
    _ = mlx.mlx_array_eval(out);
    const d = mlx.mlx_array_data_float32(out) orelse return error.NoData;
    for (buf, 0..) |v, i| {
        try testing.expectApproxEqAbs(@max(v, 0.0), d[i], 1e-5);
    }
}

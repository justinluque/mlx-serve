//! Live "ghost image" previews during diffusion sampling.
//!
//! This is NOT a learned decoder (TAESD/TAEF1) — it's the "Latent2RGB" trick
//! used by ComfyUI/Forge/Fooocus et al: a single fixed linear projection from
//! latent channels straight to RGB, fit once (by the community, off real
//! model activations) and reused forever. No weights to fetch, no VAE forward
//! pass — just a [C]→[3] matmul over the current latent every step, so it
//! costs microseconds against a real decode's tens of milliseconds. It won't
//! look as clean as a real (or distilled) decode, but it shows composition,
//! color, and "is this cooking into something or about to fail" at zero
//! marginal model cost, which is what a live per-step preview is actually for.
//!
//! Coefficients below are the published FLUX and FLUX.2 tables from
//! ComfyUI's `comfy/latent_formats.py` (Comfy-Org/ComfyUI, GPL-3.0) —
//! reproduced here as plain float constants, not the original source file.
//! They're empirically fit per architecture, not derived math, so treat them
//! as "close enough for a ghost image", not a color-accurate preview.
//!
//! Wiring:
//!   - flux.zig (FLUX.2 klein 4B/9B): the working latent is packed
//!     [1,128,H,W] (32 real channels, 2×2-patchified). The call site first
//!     runs the raw sampling latent through `Vae.bnDenorm` — the SAME
//!     per-channel batchnorm denormalization `Vae.decode` applies before
//!     its own unpatchify — then `unpatchifyFlux2` mirrors flux.zig's
//!     private `unpatchify` (same reshape ComfyUI's
//!     `Flux2.latent_rgb_factors_reshape` lambda describes), then
//!     `flux2_32ch` projects the resulting 32 channels straight to RGB.
//!     (`process_in`/`process_out` being identity only means the sampling
//!     latent and the bn-normalized latent are the same space up to that
//!     affine denorm — it does NOT mean the denorm itself is skippable;
//!     an earlier version of this file got that wrong, which is why the
//!     preview used to render as a flat, unchanging blur.)
//!   - krea.zig: the working latent is already [1,16,lat_h,lat_w] (standard
//!     16-channel flow-matching latent, same shape class as FLUX.1/SD3).
//!     The call site unpatchifies the sampling tokens, then runs the result
//!     through `Vae.denorm` — the SAME per-channel `* STD + MEAN` affine
//!     `Vae.decode` applies before its own post_quant_conv — before handing
//!     it to `flux1_16ch`/the taef1 decoder. (An earlier version skipped
//!     this denorm, same class of bug as the FLUX.2 one described above:
//!     the preview stayed on the raw/normalized sampling scale instead of
//!     the scale the factors table and taef1 weights were fit on.) Krea's
//!     own VAE isn't guaranteed to be numerically identical to FLUX.1's, so
//!     even with the denorm applied this is an approximation of an
//!     approximation — right ballpark colors/composition, not a
//!     color-accurate thumbnail. Good enough for "is this failing".
//!
//! Swap-out path: if we can ever source (or train) channel-matched TAESD/
//! TAEF1-style tiny decoder weights, add a second `Approximator` variant next
//! to this one and switch call sites — the `projectToPng`/`PreviewPng`
//! surface callers use doesn't need to change.

const std = @import("std");
const mlx = @import("mlx.zig");
const png = @import("png.zig");
const taesd = @import("taesd.zig");
const taew = @import("taew.zig");

const S = mlx.mlx_stream;

// ── Low-level mlx helpers (mirrors the private copies in flux.zig/krea.zig) ──

inline fn matmul(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&o, a, b, s));
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
/// Nearest-neighbor downsample of an NCHW array by an integer stride on H
/// and W (factor==1 just returns an owned copy). Cheap (no compute, just a
/// strided view materialized), used to cap what the taesd decoder has to
/// upsample back out of — see `MAX_DECODE_INPUT_SIDE`.
fn strideDownsample(x: mlx.mlx_array, factor: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const start = [_]c_int{ 0, 0, 0, 0 };
    const stop = [_]c_int{ sh[0], sh[1], sh[2], sh[3] };
    const strides = [_]c_int{ 1, 1, factor, factor };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, &start, 4, &stop, 4, &strides, 4, s));
    defer _ = mlx.mlx_array_free(o);
    return contig(o, s);
}
inline fn contig(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&o, x, false, s));
    return o;
}

// ── Per-architecture channel→RGB tables ──

pub const Table = struct {
    /// Row-major [channels][3] (R,G,B weights per latent channel).
    factors: []const [3]f32,
    bias: [3]f32,
};

/// FLUX.1-family 16-channel latent (ComfyUI `Flux` class).
const FLUX1_FACTORS = [16][3]f32{
    .{ -0.0346, 0.0244, 0.0681 },
    .{ 0.0034, 0.0210, 0.0687 },
    .{ 0.0275, -0.0668, -0.0433 },
    .{ -0.0174, 0.0160, 0.0617 },
    .{ 0.0859, 0.0721, 0.0329 },
    .{ 0.0004, 0.0383, 0.0115 },
    .{ 0.0405, 0.0861, 0.0915 },
    .{ -0.0236, -0.0185, -0.0259 },
    .{ -0.0245, 0.0250, 0.1180 },
    .{ 0.1008, 0.0755, -0.0421 },
    .{ -0.0515, 0.0201, 0.0011 },
    .{ 0.0428, -0.0012, -0.0036 },
    .{ 0.0817, 0.0765, 0.0749 },
    .{ -0.1264, -0.0522, -0.1103 },
    .{ -0.0280, -0.0881, -0.0499 },
    .{ -0.1262, -0.0982, -0.0778 },
};

/// FLUX.2 32-channel real latent, post-unpatchify (ComfyUI `Flux2` class).
const FLUX2_FACTORS = [32][3]f32{
    .{ 0.0058, 0.0113, 0.0073 },
    .{ 0.0495, 0.0443, 0.0836 },
    .{ -0.0099, 0.0096, 0.0644 },
    .{ 0.2144, 0.3009, 0.3652 },
    .{ 0.0166, -0.0039, -0.0054 },
    .{ 0.0157, 0.0103, -0.0160 },
    .{ -0.0398, 0.0902, -0.0235 },
    .{ -0.0052, 0.0095, 0.0109 },
    .{ -0.3527, -0.2712, -0.1666 },
    .{ -0.0301, -0.0356, -0.0180 },
    .{ -0.0107, 0.0078, 0.0013 },
    .{ 0.0746, 0.0090, -0.0941 },
    .{ 0.0156, 0.0169, 0.0070 },
    .{ -0.0034, -0.0040, -0.0114 },
    .{ 0.0032, 0.0181, 0.0080 },
    .{ -0.0939, -0.0008, 0.0186 },
    .{ 0.0018, 0.0043, 0.0104 },
    .{ 0.0284, 0.0056, -0.0127 },
    .{ -0.0024, -0.0022, -0.0030 },
    .{ 0.1207, -0.0026, 0.0065 },
    .{ 0.0128, 0.0101, 0.0142 },
    .{ 0.0137, -0.0072, -0.0007 },
    .{ 0.0095, 0.0092, -0.0059 },
    .{ 0.0000, -0.0077, -0.0049 },
    .{ -0.0465, -0.0204, -0.0312 },
    .{ 0.0095, 0.0012, -0.0066 },
    .{ 0.0290, -0.0034, 0.0025 },
    .{ 0.0220, 0.0169, -0.0048 },
    .{ -0.0332, -0.0457, -0.0468 },
    .{ -0.0085, 0.0389, 0.0609 },
    .{ -0.0076, 0.0003, -0.0043 },
    .{ -0.0111, -0.0460, -0.0614 },
};

pub const flux1_16ch: Table = .{ .factors = &FLUX1_FACTORS, .bias = .{ -0.0329, -0.0718, -0.0851 } };
pub const flux2_32ch: Table = .{ .factors = &FLUX2_FACTORS, .bias = .{ -0.0329, -0.0718, -0.0851 } };

/// Largest channel count any table above uses — bounds the stack buffer in
/// `project`, keep this in sync if a wider table is ever added.
const MAX_CHANNELS = 32;

// ── Step throttling ──

/// Only emit a preview every this many steps (plus always the very first
/// and last) — a ghost image doesn't need to update every single denoise
/// step, and skipping most of them cuts the taesd decoder's cost (see
/// `MAX_DECODE_INPUT_SIDE` below) by the same factor. Without this, a
/// 30-step generation ran the full decoder 30 times back to back, which is
/// what was pushing Krea into OOM — the decode itself wasn't leaking, it
/// was just being asked to do 30x more full-resolution work than a live
/// thumbnail needs.
const PREVIEW_EVERY_N_STEPS: u32 = 3;

pub fn shouldPreviewStep(step_no: u32, total_steps: u32) bool {
    if (step_no == 1 or step_no >= total_steps) return true;
    return step_no % PREVIEW_EVERY_N_STEPS == 0;
}

/// Longest side (in LATENT pixels, pre-decode) the taesd decoder is allowed
/// to run on. The decoder upsamples 8x (three nearest-2x stages), so this
/// caps its output to `MAX_DECODE_INPUT_SIDE * 8` pixels on the long side —
/// plenty for a chat-sized thumbnail — instead of silently running a
/// full-image-resolution decode (matching the real VAE's output size) on
/// every previewed step, which is the other half of what was blowing up
/// memory on larger Krea/FLUX.2 generations.
const MAX_DECODE_INPUT_SIDE: u32 = 48;

// ── FLUX.2's 128→32×2×2 unpack, mirroring flux.zig's private `unpatchify` ──

/// packed [1,128,H,W] (NCHW) → real latent [1,32,2H,2W] (NCHW). Caller must
/// apply the VAE's bn-denorm (see `flux.zig`'s `Vae.bnDenorm`) to the raw
/// sampling latent FIRST — this only does the reshape/unpack, same as
/// `flux.zig`'s private `unpatchify`. Feeding this a still-normalized
/// latent puts the channel→RGB factors (and the taef2 decoder) on the
/// wrong numeric scale — see module docs above.
pub fn unpatchifyFlux2(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const gh = sh[2];
    const gw = sh[3];
    const r1 = try reshape(x, &[_]c_int{ 1, 32, 2, 2, gh, gw }, s);
    defer _ = mlx.mlx_array_free(r1);
    const t1 = try transpose(r1, &[_]c_int{ 0, 1, 4, 2, 5, 3 }, s);
    defer _ = mlx.mlx_array_free(t1);
    return reshape(t1, &[_]c_int{ 1, 32, gh * 2, gw * 2 }, s);
}

// ── Projection ──

/// latent_nchw: [1,C,H,W] (any float dtype), C == table.factors.len.
/// Returns owned [1,3,H,W] f32 in [0,1] (caller frees) at LATENT resolution
/// — deliberately not upscaled to the target image size. That's the UI's
/// job (nearest/bilinear on a thumbnail is free); doing it here would just
/// spend real compute blurring noise.
pub fn project(latent_nchw: mlx.mlx_array, table: Table, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(latent_nchw);
    const c: usize = @intCast(sh[1]);
    if (c != table.factors.len or c > MAX_CHANNELS) return error.ChannelMismatch;
    const h = sh[2];
    const w = sh[3];

    const lf = try astype(latent_nchw, .float32, s);
    defer _ = mlx.mlx_array_free(lf);
    const nhwc = try transpose(lf, &[_]c_int{ 0, 2, 3, 1 }, s);
    defer _ = mlx.mlx_array_free(nhwc);
    const flat = try reshape(nhwc, &[_]c_int{ h * w, @intCast(c) }, s);
    defer _ = mlx.mlx_array_free(flat);
    var fc = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&fc, flat, false, s));
    defer _ = mlx.mlx_array_free(fc);

    var fbuf: [MAX_CHANNELS * 3]f32 = undefined;
    for (table.factors, 0..) |row, i| {
        fbuf[i * 3 + 0] = row[0];
        fbuf[i * 3 + 1] = row[1];
        fbuf[i * 3 + 2] = row[2];
    }
    const fsh = [_]c_int{ @intCast(c), 3 };
    const fmat = mlx.mlx_array_new_data(&fbuf, &fsh, 2, .float32);
    defer _ = mlx.mlx_array_free(fmat);

    const proj = try matmul(fc, fmat, s); // [H*W,3]
    defer _ = mlx.mlx_array_free(proj);

    var bbuf = table.bias;
    const bsh = [_]c_int{3};
    const barr = mlx.mlx_array_new_data(&bbuf, &bsh, 1, .float32);
    defer _ = mlx.mlx_array_free(barr);
    const biased = try addA(proj, barr, s);
    defer _ = mlx.mlx_array_free(biased);

    const lo = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(lo);
    const hi = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(hi);
    var clo = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_maximum(&clo, biased, lo, s));
    defer _ = mlx.mlx_array_free(clo);
    var chi = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_minimum(&chi, clo, hi, s));
    defer _ = mlx.mlx_array_free(chi);

    const rr = try reshape(chi, &[_]c_int{ 1, h, w, 3 }, s);
    defer _ = mlx.mlx_array_free(rr);
    return transpose(rr, &[_]c_int{ 0, 3, 1, 2 }, s);
}

pub const PreviewPng = struct {
    bytes: []u8,
    width: u32,
    height: u32,
};

/// [1,3,H,W] f32 [0,1] → RGB8 bytes (caller frees). Mirrors krea.zig's
/// `imageToPng` pixel loop.
fn toRgb8(allocator: std.mem.Allocator, img_nchw: mlx.mlx_array, s: S) ![]u8 {
    var cf = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&cf, img_nchw, false, s));
    defer _ = mlx.mlx_array_free(cf);
    _ = mlx.mlx_array_eval(cf);
    const sh = mlx.getShape(cf);
    const h: usize = @intCast(sh[2]);
    const w: usize = @intCast(sh[3]);
    const d = mlx.mlx_array_data_float32(cf) orelse return error.NoData;
    const rgb = try allocator.alloc(u8, w * h * 3);
    const plane = w * h;
    for (0..h) |y| for (0..w) |x| {
        const o = (y * w + x) * 3;
        for (0..3) |ch| {
            const v = d[ch * plane + y * w + x];
            rgb[o + ch] = @intFromFloat(std.math.clamp(v * 255.0, 0, 255));
        }
    };
    return rgb;
}

/// `project` + PNG-encode in one call — what the denoise loops actually use.
/// Caller frees `.bytes`. Any failure (shape mismatch, OOM) is returned so
/// call sites can drop a single preview frame without aborting generation.
pub fn projectToPng(allocator: std.mem.Allocator, latent_nchw: mlx.mlx_array, table: Table, s: S) !PreviewPng {
    const rgb_img = try project(latent_nchw, table, s);
    defer _ = mlx.mlx_array_free(rgb_img);
    const sh = mlx.getShape(rgb_img);
    const h: u32 = @intCast(sh[2]);
    const w: u32 = @intCast(sh[3]);
    const rgb8 = try toRgb8(allocator, rgb_img, s);
    defer allocator.free(rgb8);
    const bytes = try png.encodeRgb(allocator, rgb8, w, h);
    return .{ .bytes = bytes, .width = w, .height = h };
}

/// A "real decode" preview source — either family. taef1/taef2 (taesd.zig)
/// cover FLUX.1/FLUX.2; taew2_1 (taew.zig) covers Krea 2's Qwen-Image VAE
/// space. Both expose the same `decode(latent) -> RGB [1,3,H,W]` shape, so
/// this just lets `previewToPng`/`Progress`/`StreamCtx` carry either one
/// without needing to know which.
pub const PreviewDecoder = union(enum) {
    taesd: *const taesd.Decoder,
    taew: *const taew.Decoder,

    pub fn decode(self: PreviewDecoder, latent: mlx.mlx_array) !mlx.mlx_array {
        return switch (self) {
            .taesd => |d| d.decode(latent),
            .taew => |d| d.decode(latent),
        };
    }
};

/// `projectToPng`, but prefers a real TAESD-family decode when `decoder` is
/// set (see taesd.zig) — noticeably sharper than the linear projection.
/// The decoder input is downsampled first (see `MAX_DECODE_INPUT_SIDE`) so
/// this stays cheap even at large generation resolutions; it is NOT free
/// like the linear projection, which is also why call sites gate it behind
/// `shouldPreviewStep` rather than running it every denoise step.
///
/// `allow_linear_fallback` exists for testing the two preview paths in
/// isolation (see the Image gen window's "Latent RGB" / "TAESD" checkboxes):
/// when true (the normal, default-on-both-checkboxes case) a decoder
/// failure (shape mismatch, a weight that didn't load right) or a missing
/// decoder falls back to the linear table rather than dropping the frame or
/// aborting generation. When false — TAESD checked, Latent RGB unchecked —
/// a missing/failing decoder returns `error.NoPreview` instead, so a taesd
/// problem shows up as no preview rather than being silently masked by the
/// linear projection.
pub fn previewToPng(allocator: std.mem.Allocator, unpacked_latent: mlx.mlx_array, table: Table, decoder: ?PreviewDecoder, allow_linear_fallback: bool, s: S) !PreviewPng {
    if (decoder) |d| {
        // Cap the decoder's input so it never renders at full image
        // resolution just for a live thumbnail (see MAX_DECODE_INPUT_SIDE).
        const in_sh = mlx.getShape(unpacked_latent);
        const long_side: u32 = @intCast(@max(in_sh[2], in_sh[3]));
        const factor: c_int = @intCast(@max(@as(u32, 1), (long_side + MAX_DECODE_INPUT_SIDE - 1) / MAX_DECODE_INPUT_SIDE));
        var decode_input = unpacked_latent;
        var decode_input_owned = false;
        if (factor > 1) {
            if (strideDownsample(unpacked_latent, factor, s)) |ds| {
                decode_input = ds;
                decode_input_owned = true;
            } else |_| {}
        }
        defer if (decode_input_owned) {
            _ = mlx.mlx_array_free(decode_input);
        };
        if (d.decode(decode_input)) |img| {
            defer _ = mlx.mlx_array_free(img);
            const sh = mlx.getShape(img);
            const h: u32 = @intCast(sh[2]);
            const w: u32 = @intCast(sh[3]);
            if (toRgb8(allocator, img, s)) |rgb8| {
                defer allocator.free(rgb8);
                if (png.encodeRgb(allocator, rgb8, w, h)) |bytes| {
                    return .{ .bytes = bytes, .width = w, .height = h };
                } else |_| {}
            } else |_| {}
        } else |_| {}
    }
    if (!allow_linear_fallback) return error.NoPreview;
    return projectToPng(allocator, unpacked_latent, table, s);
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "project rejects a channel-count mismatch instead of corrupting memory" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var buf: [1 * 4 * 2 * 2]f32 = @splat(0.1); // 4-channel latent
    const sh = [_]c_int{ 1, 4, 2, 2 };
    const lat = mlx.mlx_array_new_data(&buf, &sh, 4, .float32);
    defer _ = mlx.mlx_array_free(lat);
    try testing.expectError(error.ChannelMismatch, project(lat, flux1_16ch, s));
}

test "project on a 16-channel latent returns a clipped [1,3,H,W] RGB image" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const H = 4;
    const W = 3;
    var buf: [16 * H * W]f32 = undefined;
    // Deterministic-but-nontrivial synthetic latent so the projection isn't
    // just testing "zeros in, bias out".
    for (&buf, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.37) * 3.0;
    const sh = [_]c_int{ 1, 16, H, W };
    const lat = mlx.mlx_array_new_data(&buf, &sh, 4, .float32);
    defer _ = mlx.mlx_array_free(lat);

    const out = try project(lat, flux1_16ch, s);
    defer _ = mlx.mlx_array_free(out);
    _ = mlx.mlx_array_eval(out);
    const osh = mlx.getShape(out);
    try testing.expectEqual(@as(c_int, 1), osh[0]);
    try testing.expectEqual(@as(c_int, 3), osh[1]);
    try testing.expectEqual(@as(c_int, H), osh[2]);
    try testing.expectEqual(@as(c_int, W), osh[3]);
    const d = mlx.mlx_array_data_float32(out) orelse return error.NoData;
    for (d[0 .. 3 * H * W]) |v| {
        try testing.expect(v >= 0.0 and v <= 1.0);
    }
}

test "projectToPng round-trips a 32-channel (FLUX.2) latent to real PNG bytes" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const a = testing.allocator;
    const H = 5;
    const W = 5;
    var buf: [32 * H * W]f32 = undefined;
    for (&buf, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i % 7)) - 3.0) * 0.5;
    const sh = [_]c_int{ 1, 32, H, W };
    const lat = mlx.mlx_array_new_data(&buf, &sh, 4, .float32);
    defer _ = mlx.mlx_array_free(lat);

    const pv = try projectToPng(a, lat, flux2_32ch, s);
    defer a.free(pv.bytes);
    try testing.expectEqual(@as(u32, W), pv.width);
    try testing.expectEqual(@as(u32, H), pv.height);
    // A real PNG: magic bytes + nonzero body.
    try testing.expect(pv.bytes.len > 8);
    try testing.expectEqualSlices(u8, &[_]u8{ 0x89, 'P', 'N', 'G' }, pv.bytes[0..4]);
}

test "shouldPreviewStep always fires on step 1 and the last step" {
    try testing.expect(shouldPreviewStep(1, 30));
    try testing.expect(shouldPreviewStep(30, 30));
    try testing.expect(shouldPreviewStep(1, 1)); // single-step run: first == last
}

test "shouldPreviewStep skips most interior steps (throttled to every Nth)" {
    var fired: u32 = 0;
    for (1..31) |step| {
        if (shouldPreviewStep(@intCast(step), 30)) fired += 1;
    }
    // Far fewer than 30 (was every step before this fix) but still more
    // than just the two guaranteed endpoints.
    try testing.expect(fired < 30);
    try testing.expect(fired > 2);
}

test "strideDownsample by 1 is a no-op shape-wise" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var buf: [1 * 2 * 4 * 4]f32 = @splat(1.0);
    const sh = [_]c_int{ 1, 2, 4, 4 };
    const lat = mlx.mlx_array_new_data(&buf, &sh, 4, .float32);
    defer _ = mlx.mlx_array_free(lat);
    const out = try strideDownsample(lat, 1, s);
    defer _ = mlx.mlx_array_free(out);
    _ = mlx.mlx_array_eval(out);
    const osh = mlx.getShape(out);
    try testing.expectEqual(@as(c_int, 4), osh[2]);
    try testing.expectEqual(@as(c_int, 4), osh[3]);
}

test "strideDownsample by 2 halves H and W" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var buf: [1 * 2 * 8 * 8]f32 = undefined;
    for (&buf, 0..) |*v, i| v.* = @floatFromInt(i);
    const sh = [_]c_int{ 1, 2, 8, 8 };
    const lat = mlx.mlx_array_new_data(&buf, &sh, 4, .float32);
    defer _ = mlx.mlx_array_free(lat);
    const out = try strideDownsample(lat, 2, s);
    defer _ = mlx.mlx_array_free(out);
    _ = mlx.mlx_array_eval(out);
    const osh = mlx.getShape(out);
    try testing.expectEqual(@as(c_int, 4), osh[2]);
    try testing.expectEqual(@as(c_int, 4), osh[3]);
}

test "unpatchifyFlux2 turns [1,128,H,W] into [1,32,2H,2W]" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const H = 3;
    const W = 2;
    var buf: [128 * H * W]f32 = undefined;
    for (&buf, 0..) |*v, i| v.* = @floatFromInt(i);
    const sh = [_]c_int{ 1, 128, H, W };
    const packed_lat = mlx.mlx_array_new_data(&buf, &sh, 4, .float32);
    defer _ = mlx.mlx_array_free(packed_lat);

    const out = try unpatchifyFlux2(packed_lat, s);
    defer _ = mlx.mlx_array_free(out);
    _ = mlx.mlx_array_eval(out);
    const osh = mlx.getShape(out);
    try testing.expectEqual(@as(c_int, 1), osh[0]);
    try testing.expectEqual(@as(c_int, 32), osh[1]);
    try testing.expectEqual(@as(c_int, H * 2), osh[2]);
    try testing.expectEqual(@as(c_int, W * 2), osh[3]);
}

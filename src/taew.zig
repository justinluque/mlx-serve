//! TAEW2.1 decoder (madebyollin/taehv, `safetensors/taew2_1.safetensors`) —
//! the tiny decoder for Wan 2.1 / Wan 2.2-14B / Qwen-Image's VAE latent
//! space. Used for Krea 2's live preview: Krea 2's own VAE is explicitly
//! the Qwen-Image 3D causal VAE (see krea.zig's module doc), NOT FLUX.1's —
//! so `taesd.zig`'s taef1 (trained to reproduce FLUX.1's VAE specifically)
//! is architecturally the wrong decoder for it, independent of whether it
//! loads correctly. Confirmed against the real madebyollin/taehv source
//! (`taehv.py`, commit 46e58800) and its own README, which states plainly:
//! "taew2_1.pth serves three models: Wan 2.1, Wan 2.2 14B (which retained
//! the older VAE), and Qwen Image." Ported with the actual source in hand —
//! not reconstructed blind the way taef1/taef2 originally were.
//!
//! DECODE-ONLY, SINGLE-SOURCE-FRAME SPECIALIZATION. TAEHV is fundamentally a
//! *video* autoencoder: `MemBlock` threads a "previous frame" through each
//! block for temporal coherence, and `TGrow` can turn one input frame into
//! several output frames (temporal upsampling). Krea 2 generates a single
//! still image, so there's only ever ONE real source latent — but that does
//! NOT mean every `TGrow` is a stride-1 no-op. An earlier version of this
//! file assumed exactly that (reasoning "we're not asking for temporal
//! upsampling, so treat every stride as 1") and it's wrong: the CHECKPOINT
//! ITSELF bakes its stride into each `TGrow`'s saved conv weight shape
//! (`out_channels = stride * in_channels`) at train/save time — that's not
//! something choosable at inference. `madebyollin/taehv`'s real
//! `taew2_1.pth` has strides `[1, 2, 2]` for its three `TGrow`s (confirmed
//! from a real load's logged weight shapes: `decoder.13.conv.weight` is
//! `[256,128,1,1]`, i.e. stride 2, not `[128,128,1,1]`), so one input
//! latent genuinely produces 1×1×2×2 = 4 candidate output frames — matching
//! taehv.py's own default `decoder_time_upscale=(True, True)` — and
//! `decode_video` keeps only the LAST one (`frames_to_trim = 2**2-1 = 3`),
//! discarding the earlier "startup" frames that have no real temporal
//! context behind them.
//!
//! This port handles that correctly rather than pretending it away:
//!   - The batch axis carries `nt` "virtual frames" (mirroring taehv.py's
//!     `apply_model_with_memblocks`' PARALLEL mode, where `x`'s batch axis
//!     is literally `N*T`) — starts at 1, multiplies by each `TGrow`'s real
//!     stride (read off the loaded weight's own shape, not assumed), ends
//!     at 4 for this checkpoint.
//!   - `MemBlock`'s "past" input is a batch-shifted version of its "current"
//!     input: frame 0's memory is zero, frame i>0's memory is frame i-1's
//!     pre-block value — see `buildPast`. For everything before the first
//!     real stride>1 `TGrow` (nt still 1 there), this correctly degenerates
//!     to "memory is always zero", matching the true single-source-frame
//!     case.
//!   - `TGrow` splits its grown channel axis into `stride` new virtual
//!     frames appended to the batch axis — see `tgrowSplit` for the NHWC
//!     reshape/transpose this requires (channel isn't adjacent to batch in
//!     NHWC the way it is in PyTorch's NCHW, so there's no free-reshape
//!     shortcut).
//!   - Only the LAST virtual frame is kept before the (frame-independent)
//!     final relu/conv_out — see `selectLastFrame` — both for correctness
//!     (matching `decode_video`'s trim) and to skip compute on the 3
//!     frames about to be discarded.
//!
//! Everything else (Clamp, 3x upsample-then-transition-conv stages, 16
//! latent channels, patch_size=1, 8x total spatial upsample) matches
//! taesd.zig's taef1/taef2 shape closely enough to reuse its low-level mlx
//! helpers, but the block structure itself (MemBlock vs taesd.py's Block,
//! TGrow, no midblock Pool) is different enough to warrant its own file
//! rather than shoehorning into `taesd.Decoder`.
//!
//! WEIGHT KEY NAMES: taehv.py loads a plain `nn.Sequential` state dict via
//! `self.load_state_dict(...)` directly from the raw checkpoint — NOT
//! diffusers-wrapped like taef1/taef2 — so every key is
//! `decoder.{i}.{...}` using taehv.py's own `nn.Sequential` index `i`
//! *directly*, no offset. Confirmed against the real source:
//!
//!   decoder Sequential layout (indices 0-22; only some carry weights):
//!     0  Clamp()                                    (no weights)
//!     1  conv(16, 256)                               conv_in (bias)
//!     2  ReLU                                        (no weights)
//!     3,4,5   MemBlock(256,256) x3
//!     6  Upsample(2x, nearest)                       (no weights)
//!     7  TGrow(256, stride=1)                        1x1, no bias
//!     8  conv(256, 128, bias=False)                  transition 1
//!     9,10,11 MemBlock(128,128) x3
//!     12 Upsample(2x, nearest)
//!     13 TGrow(128, stride=1)
//!     14 conv(128, 64, bias=False)                   transition 2
//!     15,16,17 MemBlock(64,64) x3
//!     18 Upsample(2x, nearest)
//!     19 TGrow(64, stride=1)
//!     20 conv(64, 64, bias=False)                    transition 3
//!     21 ReLU                                        (no weights)
//!     22 conv(64, 3)                                 conv_out (bias)
//!
//!   MemBlock(n_in, n_out) at index i (n_in == n_out for every MemBlock in
//!   this decoder, so `self.skip` is `nn.Identity()` — no skip weight to
//!   load, ever, for taew2_1's decoder specifically):
//!     decoder.{i}.conv.0.{weight,bias}   3x3, in=2*n_in (post-concat)
//!     decoder.{i}.conv.2.{weight,bias}   3x3, n_out->n_out
//!     decoder.{i}.conv.4.{weight,bias}   3x3, n_out->n_out
//!
//!   TGrow(n_f, stride=1) at index i: decoder.{i}.conv.weight (1x1, no bias)
//!   Plain conv at index i: decoder.{i}.weight [+ decoder.{i}.bias]

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");

const Weights = model_mod.Weights;
const S = mlx.mlx_stream;

// ── mlx helpers (file-local; mirrors taesd.zig/flux.zig/krea.zig) ──

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
inline fn zerosLike(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&o, sh.ptr, sh.len, .float32, s));
    return o;
}
/// Builds the "previous frame" memory input for `MemBlockW.forward` when
/// the batch axis (axis 0) holds `nt` virtual frames (see module doc — this
/// is taehv.py's `N*T`, N always 1 here). Frame 0's memory is zero; frame
/// i>0's memory is frame i-1's PRE-block value. Mirrors
/// `apply_model_with_memblocks`'s parallel-mode padding exactly:
/// `F.pad(_x, (0,0,0,0,0,0,1,0), value=0)[:, :T]`.
fn buildPast(x: mlx.mlx_array, nt: usize, s: S) !mlx.mlx_array {
    if (nt == 1) return zerosLike(x, s);
    const sh = mlx.getShape(x);
    var zero_frame = mlx.mlx_array_new();
    const zero_sh = [_]c_int{ 1, sh[1], sh[2], sh[3] };
    try mlx.check(mlx.mlx_zeros(&zero_frame, &zero_sh, 4, .float32, s));
    defer _ = mlx.mlx_array_free(zero_frame);
    // rest = x[0 .. nt-1] — every frame except the last, shifted forward by
    // one slot once concatenated after the zero frame below.
    const start = [_]c_int{ 0, 0, 0, 0 };
    const stop = [_]c_int{ @intCast(nt - 1), sh[1], sh[2], sh[3] };
    const strides = [_]c_int{ 1, 1, 1, 1 };
    var rest = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&rest, x, &start, 4, &stop, 4, &strides, 4, s));
    defer _ = mlx.mlx_array_free(rest);
    return concat2(zero_frame, rest, 0, s);
}
inline fn concat2(a: mlx.mlx_array, b: mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    _ = mlx.mlx_vector_array_append_value(vec, a);
    _ = mlx.mlx_vector_array_append_value(vec, b);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, axis, s));
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
/// 1x1 conv, no bias (TGrow's own conv — always bias=False in taehv.py).
fn conv1x1(x: mlx.mlx_array, w: mlx.mlx_array, s: S) !mlx.mlx_array {
    const xc = try contig(x, s);
    defer _ = mlx.mlx_array_free(xc);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv2d(&o, xc, w, 1, 1, 0, 0, 1, 1, 1, s));
    return o;
}
/// nearest 2x upsample, NHWC — NOT fused with a conv (unlike taesd.zig's
/// `upsample2x`): TAEHV puts a `TGrow` and a channel-transition conv
/// between the upsample and the next block, so those have to stay separate
/// steps here.
fn upsampleNearest2x(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var r1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r1);
    try mlx.check(mlx.mlx_repeat_axis(&r1, x, 2, 1, s));
    var r2 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_repeat_axis(&r2, r1, 2, 2, s));
    return r2;
}
/// `TGrow`: 1x1 conv, then — if the checkpoint's own weight shape implies
/// stride > 1 (`out_channels = stride * in_channels`, baked in at
/// train/save time — see module doc, this is NOT a choice made here) —
/// splits the grown channel axis into `stride` new virtual frames appended
/// to the batch axis. Mirrors taehv.py's NCHW `x.reshape(-1, C, H, W)`
/// (which decomposes the channel axis as [stride][C], stride outer, C
/// inner, then merges it with the batch axis) — adapted for NHWC via an
/// explicit reshape+transpose+reshape, since channel isn't adjacent to
/// batch here the way it is in NCHW. Returns the new `nt` (`old nt *
/// stride`).
fn tgrowSplit(x: mlx.mlx_array, w: mlx.mlx_array, nt: usize, s: S) !struct { out: mlx.mlx_array, nt: usize } {
    const wsh = mlx.getShape(w); // OHWI: [out, 1, 1, in]
    const n_f: usize = @intCast(wsh[3]);
    const out_c: usize = @intCast(wsh[0]);
    const stride = out_c / n_f;
    const grown = try conv1x1(x, w, s); // [nt, H, W, n_f*stride]
    if (stride == 1) return .{ .out = grown, .nt = nt };
    defer _ = mlx.mlx_array_free(grown);
    const gsh = mlx.getShape(grown);
    // split trailing channel axis: [nt,H,W,n_f*stride] -> [nt,H,W,stride,n_f]
    const r1 = try reshape(grown, &[_]c_int{ gsh[0], gsh[1], gsh[2], @intCast(stride), @intCast(n_f) }, s);
    defer _ = mlx.mlx_array_free(r1);
    // move `stride` next to the batch axis: [nt,stride,H,W,n_f]
    const t1 = try transpose(r1, &[_]c_int{ 0, 3, 1, 2, 4 }, s);
    defer _ = mlx.mlx_array_free(t1);
    // merge (nt,stride) -> nt*stride, nt outer / stride inner (matches
    // taehv.py's NCHW decomposition — see doc above)
    const merged = try reshape(t1, &[_]c_int{ gsh[0] * @as(c_int, @intCast(stride)), gsh[1], gsh[2], @intCast(n_f) }, s);
    return .{ .out = merged, .nt = nt * stride };
}
/// Keeps only the LAST virtual frame along the batch axis, discarding the
/// rest — mirrors taehv.py's `decode_video`'s final `x[:, frames_to_trim:]`
/// trim. After temporal upsampling, only the last decoded frame corresponds
/// to our single real input latent; the earlier ones are transient
/// "startup" frames with no real temporal context behind them (their
/// `MemBlock` memory chains ultimately bottom out at the zero-padding at
/// frame 0, not at anything resembling a real preceding frame).
fn selectLastFrame(x: mlx.mlx_array, nt: usize, s: S) !mlx.mlx_array {
    if (nt == 1) return contig(x, s);
    const sh = mlx.getShape(x);
    const start = [_]c_int{ @intCast(nt - 1), 0, 0, 0 };
    const stop = [_]c_int{ @intCast(nt), sh[1], sh[2], sh[3] };
    const strides = [_]c_int{ 1, 1, 1, 1 };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, &start, 4, &stop, 4, &strides, 4, s));
    return contig(o, s);
}

// ── Weights ──

const MemBlockW = struct {
    c1w: mlx.mlx_array,
    c1b: mlx.mlx_array,
    c2w: mlx.mlx_array,
    c2b: mlx.mlx_array,
    c3w: mlx.mlx_array,
    c3b: mlx.mlx_array,

    fn deinit(self: *MemBlockW) void {
        inline for (.{ "c1w", "c1b", "c2w", "c2b", "c3w", "c3b" }) |f| _ = mlx.mlx_array_free(@field(self, f));
    }

    /// `x`: NHWC, batch axis = `nt` virtual frames (see module doc). `past`
    /// is built per-frame via `buildPast` — zero for frame 0, the
    /// preceding frame's pre-block value otherwise; the concat-then-full-
    /// width-conv (rather than pre-slicing the weight when past happens to
    /// be zero) matches taehv.py's `MemBlock.forward` exactly. `skip` is
    /// always identity (n_in == n_out for every MemBlock in this decoder),
    /// so no skip conv here.
    fn forward(self: *const MemBlockW, x: mlx.mlx_array, nt: usize, s: S) !mlx.mlx_array {
        const past = try buildPast(x, nt, s);
        defer _ = mlx.mlx_array_free(past);
        const cat = try concat2(x, past, 3, s); // NHWC: channel axis = 3
        defer _ = mlx.mlx_array_free(cat);
        const h1 = try conv3x3(cat, self.c1w, self.c1b, s);
        const a1 = try relu(h1, s);
        _ = mlx.mlx_array_free(h1);
        defer _ = mlx.mlx_array_free(a1);
        const h2 = try conv3x3(a1, self.c2w, self.c2b, s);
        const a2 = try relu(h2, s);
        _ = mlx.mlx_array_free(h2);
        defer _ = mlx.mlx_array_free(a2);
        const h3 = try conv3x3(a2, self.c3w, self.c3b, s);
        defer _ = mlx.mlx_array_free(h3);
        const summed = try addA(h3, x, s); // skip(x) == x (identity)
        defer _ = mlx.mlx_array_free(summed);
        return relu(summed, s);
    }
};

pub const Decoder = struct {
    allocator: std.mem.Allocator,
    s: S,
    conv_in_w: mlx.mlx_array,
    conv_in_b: mlx.mlx_array,
    /// 3 groups of 3 MemBlocks each, at n_f = [256, 128, 64] (the group's
    /// OWN channel count, i.e. taehv.py's n_f[0..2] — n_f[3]=64 has no
    /// MemBlocks, it's just the post-transition-3 channel count before
    /// conv_out).
    blocks: [9]MemBlockW,
    tgrow_w: [3]mlx.mlx_array,
    trans_w: [3]mlx.mlx_array, // 256->128, 128->64, 64->64 (all bias=False)
    conv_out_w: mlx.mlx_array,
    conv_out_b: mlx.mlx_array,

    pub fn deinit(self: *Decoder) void {
        _ = mlx.mlx_array_free(self.conv_in_w);
        _ = mlx.mlx_array_free(self.conv_in_b);
        for (&self.blocks) |*b| b.deinit();
        for (&self.tgrow_w) |w| _ = mlx.mlx_array_free(w);
        for (&self.trans_w) |w| _ = mlx.mlx_array_free(w);
        _ = mlx.mlx_array_free(self.conv_out_w);
        _ = mlx.mlx_array_free(self.conv_out_b);
    }

    /// latent [1,16,H,W] (any float dtype) → RGB [1,3,8H,8W] f32 clipped to
    /// [0,1] (NCHW, caller frees).
    pub fn decode(self: *const Decoder, latent: mlx.mlx_array) !mlx.mlx_array {
        const sh = mlx.getShape(latent);
        log.info("[taew] decode: entered, input shape={any}\n", .{sh});
        if (sh[1] != 16) return error.ChannelMismatch;
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
        _ = mlx.mlx_array_eval(h);
        log.info("[taew] decode: conv_in OK, shape={any}\n", .{mlx.getShape(h)});

        var nt: usize = 1;
        var bi: usize = 0;
        for (0..3) |group| {
            for (0..3) |_| {
                const nh = try self.blocks[bi].forward(h, nt, s);
                _ = mlx.mlx_array_eval(nh);
                log.info("[taew] decode: block[{d}] (group {d}, nt={d}) OK, shape={any}\n", .{ bi, group, nt, mlx.getShape(nh) });
                _ = mlx.mlx_array_free(h);
                h = nh;
                bi += 1;
            }
            const up = try upsampleNearest2x(h, s);
            _ = mlx.mlx_array_eval(up);
            log.info("[taew] decode: upsample (group {d}) OK, shape={any}\n", .{ group, mlx.getShape(up) });
            _ = mlx.mlx_array_free(h);
            h = up;
            const grown = try tgrowSplit(h, self.tgrow_w[group], nt, s);
            _ = mlx.mlx_array_eval(grown.out);
            log.info("[taew] decode: tgrow (group {d}) OK, nt {d}->{d}, shape={any}\n", .{ group, nt, grown.nt, mlx.getShape(grown.out) });
            _ = mlx.mlx_array_free(h);
            h = grown.out;
            nt = grown.nt;
            const transitioned = try conv3x3(h, self.trans_w[group], null, s);
            _ = mlx.mlx_array_eval(transitioned);
            log.info("[taew] decode: transition (group {d}) OK, shape={any}\n", .{ group, mlx.getShape(transitioned) });
            _ = mlx.mlx_array_free(h);
            h = transitioned;
        }
        std.debug.assert(bi == 9);

        // Keep only the final virtual frame (see `selectLastFrame`'s doc)
        // before the frame-independent final relu/conv_out, to skip
        // compute on the frames we're about to discard.
        {
            const last = try selectLastFrame(h, nt, s);
            _ = mlx.mlx_array_free(h);
            h = last;
            log.info("[taew] decode: selected last of {d} frame(s), shape={any}\n", .{ nt, mlx.getShape(h) });
        }

        {
            const a = try relu(h, s);
            _ = mlx.mlx_array_free(h);
            h = a;
        }
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

/// Fetch `decoder.{idx}.{sub}` as a conv weight, OIHW -> OHWI transposed.
/// Unlike taesd.zig's `resolveConvW`, no candidate guessing — this key
/// convention is confirmed against the real taehv.py source (see module
/// doc), not reconstructed blind.
fn convW(w: *const Weights, idx: u32, sub: []const u8, a: std.mem.Allocator, s: S) !mlx.mlx_array {
    const key = try fmtKey(a, "decoder.{d}.{s}", .{ idx, sub });
    defer a.free(key);
    const raw = w.get(key) orelse {
        log.err("[taew] missing key \"{s}\" (have {d} tensors total)\n", .{ key, w.count() });
        return error.MissingTaewWeight;
    };
    var owned = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&owned, raw));
    const rsh = mlx.getShape(owned);
    if (rsh.len != 4) {
        log.err("[taew] decoder[{d}].{s} <- \"{s}\" has rank {d}, expected 4\n", .{ idx, sub, key, rsh.len });
        _ = mlx.mlx_array_free(owned);
        return error.UnexpectedTensorRank;
    }
    log.info("[taew]   decoder[{d}].{s} <- \"{s}\" OIHW={any}\n", .{ idx, sub, key, rsh });
    const t = try transpose(owned, &[_]c_int{ 0, 2, 3, 1 }, s);
    _ = mlx.mlx_array_free(owned);
    const c = try contig(t, s);
    _ = mlx.mlx_array_free(t);
    return c;
}

fn convB(w: *const Weights, idx: u32, sub: []const u8, a: std.mem.Allocator) !mlx.mlx_array {
    const key = try fmtKey(a, "decoder.{d}.{s}", .{ idx, sub });
    defer a.free(key);
    const raw = w.get(key) orelse {
        log.err("[taew] missing key \"{s}\" (have {d} tensors total)\n", .{ key, w.count() });
        return error.MissingTaewWeight;
    };
    var owned = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&owned, raw));
    log.info("[taew]   decoder[{d}].{s} <- \"{s}\" shape={any}\n", .{ idx, sub, key, mlx.getShape(owned) });
    return owned;
}

fn loadMemBlock(w: *const Weights, idx: u32, a: std.mem.Allocator, s: S) !MemBlockW {
    var b: MemBlockW = .{
        .c1w = try convW(w, idx, "conv.0.weight", a, s),
        .c1b = try convB(w, idx, "conv.0.bias", a),
        .c2w = try convW(w, idx, "conv.2.weight", a, s),
        .c2b = try convB(w, idx, "conv.2.bias", a),
        .c3w = try convW(w, idx, "conv.4.weight", a, s),
        .c3b = try convB(w, idx, "conv.4.bias", a),
    };
    errdefer b.deinit();
    return b;
}

/// Load `<model_dir>/taew2_1.safetensors` into a `Decoder`. Fails loudly
/// (missing-key errors name the exact key) rather than loading garbage.
pub fn loadDecoder(allocator: std.mem.Allocator, model_dir: []const u8, s: S) !Decoder {
    const path = try std.fmt.allocPrint(allocator, "{s}/taew2_1.safetensors", .{model_dir});
    defer allocator.free(path);
    var w = try model_mod.loadWeightsSingleFile(allocator, path);
    defer w.deinit();

    var dec: Decoder = .{
        .allocator = allocator,
        .s = s,
        .conv_in_w = try convW(&w, 1, "weight", allocator, s),
        .conv_in_b = try convB(&w, 1, "bias", allocator),
        .blocks = undefined,
        .tgrow_w = undefined,
        .trans_w = undefined,
        .conv_out_w = try convW(&w, 22, "weight", allocator, s),
        .conv_out_b = try convB(&w, 22, "bias", allocator),
    };
    errdefer {
        _ = mlx.mlx_array_free(dec.conv_in_w);
        _ = mlx.mlx_array_free(dec.conv_in_b);
        _ = mlx.mlx_array_free(dec.conv_out_w);
        _ = mlx.mlx_array_free(dec.conv_out_b);
    }

    const block_indices = [9]u32{ 3, 4, 5, 9, 10, 11, 15, 16, 17 };
    var built: usize = 0;
    errdefer for (dec.blocks[0..built]) |*b| b.deinit();
    for (block_indices, 0..) |idx, i| {
        dec.blocks[i] = try loadMemBlock(&w, idx, allocator, s);
        built = i + 1;
    }

    const tgrow_indices = [3]u32{ 7, 13, 19 };
    var tgrow_built: usize = 0;
    errdefer for (dec.tgrow_w[0..tgrow_built]) |tw| {
        _ = mlx.mlx_array_free(tw);
    };
    for (tgrow_indices, 0..) |idx, i| {
        dec.tgrow_w[i] = try convW(&w, idx, "conv.weight", allocator, s);
        tgrow_built = i + 1;
    }

    const trans_indices = [3]u32{ 8, 14, 20 };
    var trans_built: usize = 0;
    errdefer for (dec.trans_w[0..trans_built]) |tw| {
        _ = mlx.mlx_array_free(tw);
    };
    for (trans_indices, 0..) |idx, i| {
        dec.trans_w[i] = try convW(&w, idx, "weight", allocator, s);
        trans_built = i + 1;
    }

    log.info("[taew] loaded taew2_1 decoder from {s} (16 channels, Qwen-Image/Wan-2.1 VAE space)\n", .{model_dir});
    return dec;
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "tgrowSplit assigns channel chunks to the correct output frame" {
    // Verifies the reshape+transpose+reshape doesn't scramble which output
    // frame each channel-chunk lands in — a transpose-axis mistake here
    // would silently mix frames rather than error, so this pins it down
    // with content that's only correct if the assignment is right.
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const C = 2;
    const H = 2;
    const W = 2;
    const stride = 2;
    // OHWI [C*stride,1,1,C]: chunk 0 (out 0..C-1) = identity*1, chunk 1
    // (out C..2C-1) = identity*10 — diagonal within each chunk, zero
    // elsewhere.
    var wbuf: [C * stride * C]f32 = @splat(0);
    for (0..stride) |chunk| {
        for (0..C) |c| {
            const out_idx = chunk * C + c;
            const mult: f32 = if (chunk == 0) 1.0 else 10.0;
            wbuf[out_idx * C + c] = mult;
        }
    }
    const wsh = [_]c_int{ C * stride, 1, 1, C };
    const w = mlx.mlx_array_new_data(&wbuf, &wsh, 4, .float32);
    defer _ = mlx.mlx_array_free(w);

    var xbuf: [1 * H * W * C]f32 = undefined;
    for (&xbuf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) + 1.0;
    const xsh = [_]c_int{ 1, H, W, C };
    const x = mlx.mlx_array_new_data(&xbuf, &xsh, 4, .float32);
    defer _ = mlx.mlx_array_free(x);

    const result = try tgrowSplit(x, w, 1, s);
    defer _ = mlx.mlx_array_free(result.out);
    try testing.expectEqual(@as(usize, 2), result.nt);
    _ = mlx.mlx_array_eval(result.out);
    const d = mlx.mlx_array_data_float32(result.out) orelse return error.NoData;
    const frame_sz = H * W * C;
    for (0..frame_sz) |i| {
        try testing.expectApproxEqAbs(xbuf[i] * 1.0, d[i], 1e-4); // frame 0 = chunk 0
        try testing.expectApproxEqAbs(xbuf[i] * 10.0, d[frame_sz + i], 1e-4); // frame 1 = chunk 1
    }
}

test "Decoder.decode rejects a channel-count mismatch" {
    var dec: Decoder = undefined;
    var buf: [1 * 4 * 2 * 2]f32 = @splat(0);
    const sh = [_]c_int{ 1, 4, 2, 2 };
    const lat = mlx.mlx_array_new_data(&buf, &sh, 4, .float32);
    defer _ = mlx.mlx_array_free(lat);
    try testing.expectError(error.ChannelMismatch, dec.decode(lat));
}

test "MemBlock.forward on a zeroed weight set is a shape no-op" {
    // Hermetic architecture check (no real checkpoint needed): with c1/c2
    // zeroed and c3 zeroed, conv1(cat(x,0)) == 0 regardless of x (since the
    // only nonzero input half is x, but the weight is zero), so the block
    // reduces to relu(0 + x) == relu(x) — exercises the concat-with-zeros
    // wiring and the identity-skip add without needing real weights.
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const C = 4;
    const H = 3;
    const W = 3;
    var zeros_w1: [C * 3 * 3 * (2 * C)]f32 = @splat(0); // in = 2*C (post-concat)
    var zeros_w2: [C * 3 * 3 * C]f32 = @splat(0);
    var zeros_b: [C]f32 = @splat(0);
    const w1sh = [_]c_int{ C, 3, 3, 2 * C };
    const w2sh = [_]c_int{ C, 3, 3, C };
    const bsh = [_]c_int{C};
    const zw1 = mlx.mlx_array_new_data(&zeros_w1, &w1sh, 4, .float32);
    defer _ = mlx.mlx_array_free(zw1);
    const zw2 = mlx.mlx_array_new_data(&zeros_w2, &w2sh, 4, .float32);
    defer _ = mlx.mlx_array_free(zw2);
    const zb = mlx.mlx_array_new_data(&zeros_b, &bsh, 1, .float32);
    defer _ = mlx.mlx_array_free(zb);

    const dupA = struct {
        fn f(a: mlx.mlx_array) mlx.mlx_array {
            var o = mlx.mlx_array_new();
            mlx.check(mlx.mlx_array_set(&o, a)) catch unreachable;
            return o;
        }
    }.f;

    var block: MemBlockW = .{
        .c1w = dupA(zw1), .c1b = dupA(zb),
        .c2w = dupA(zw2), .c2b = dupA(zb),
        .c3w = dupA(zw2), .c3b = dupA(zb),
    };
    defer block.deinit();

    var buf: [1 * H * W * C]f32 = undefined;
    for (&buf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.1 - 1.0; // includes negatives
    const xsh = [_]c_int{ 1, H, W, C };
    const x = mlx.mlx_array_new_data(&buf, &xsh, 4, .float32);
    defer _ = mlx.mlx_array_free(x);

    const out = try block.forward(x, 1, s);
    defer _ = mlx.mlx_array_free(out);
    _ = mlx.mlx_array_eval(out);
    const d = mlx.mlx_array_data_float32(out) orelse return error.NoData;
    for (buf, 0..) |v, i| {
        try testing.expectApproxEqAbs(@max(v, 0.0), d[i], 1e-5);
    }
}

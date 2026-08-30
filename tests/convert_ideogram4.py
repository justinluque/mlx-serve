#!/usr/bin/env python3
"""Convert an upstream Ideogram 4 checkpoint into an mlx-serve pack.

    python tests/convert_ideogram4.py \
        --src ideogram-ai/ideogram-4-fp8 \
        --out ~/.mlx-serve/models/ddalcu/Ideogram-4-MLX-Serve-mixed \
        --precision mixed

Reads either published quantization (weight-only FP8 with per-row scales, or
bitsandbytes NF4), dequantizes to bf16, and re-quantizes with `mx.quantize`
into the affine layout `flux.QLinear` / `ideogram4.IgLinear` read.

`--precision`:

Sizes below are the WHOLE pack — both transformers, the text encoder and the
VAE — from `estimate_pack_bytes`, which the self-test pins in the same order.

A mixed name reads bulk_sensitive; the encoder's width is per policy (below),
and `--bulk-bits` / `--sensitive-bits` / `--te-bits` override any tier.

  3         every projection at 3-bit, ~11 GB. Noticeably softer text
            rendering than 4-bit; `mixed_3_8` is barely larger AND protects
            the modulation tier.
  4         every projection at 4-bit, ~14 GB. Smallest well-tested point.
  8         every projection at 8-bit, ~27 GB. The reference point.
  mixed     the bulk (attention + MLP, ~8.7B of the 9.3B) at 4-bit, the
            modulation and conditioning projections at 8-bit, everything small
            enough not to matter left dense bf16, encoder 8-bit. ~18 GB.
  mixed_3_8 the small one: bulk 3-bit, sensitive still 8-bit, encoder 4-bit.
            ~13 GB, and the floor — see `MIN_BULK_BITS`. A `mixed_2_8` pack
            was published and withdrawn: 2-bit affine on the DiT bulk renders
            a woven grid texture at every prompt, seed and resolution, so the
            policy and the `--bulk-bits 2` override are both refused now.

The mixed policies are not a guess about which tensors "matter" in general —
they are about which ones are cheap. `adaln_modulation` is 320M parameters
whose output is a per-layer SCALE and a tanh GATE, so its error is
multiplicative over the whole residual stream rather than averaged into one
projection (the same reason mlx-serve's MageFlow notes flag `img_mod`/`txt_mod`
at 4-bit); the embedding, timestep and final-layer projections together are
under 30M, so holding them at bf16 costs ~60 MB and removes them as suspects
entirely. `mixed_3_8` takes the BULK down for anyone targeting a smaller
download and leaves that reasoning alone: the sensitive tier is 1.13B
parameters across the two transformers, so holding it at 8-bit instead of 6
costs 0.28 GB — 2% of the smallest pack, and the tier a low-bit bulk most needs
protecting from. The bits worth arguing about are the bulk (17.4B) and the text
encoder (6.6B), which is a third of the download: shrinking only the DiT does
not produce a smaller pack, so the small policies take the encoder to 4-bit,
the width the shipped flat `--precision 4` pack already runs it at. See
`te_bits_for` and `apply_overrides`.

TWO transformers are converted: the conditional one and its own separate
unconditional checkpoint (asymmetric CFG). Both get the same policy.

The VAE stays dense bf16 (0.17 GB total) apart from its mid-block attention
projections, which mlx-serve's shared Flux2 decoder loads as quantized linears.

`unconditional_transformer/config.json` is written LAST, on purpose: it is the
completeness marker `model_discovery.requiredMediaMarker` looks for, so an
interrupted conversion stays invisible to `list` and to the loader instead of
half-registering.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors.torch import load_file

# Every projection in one DiT block, and the two the text conditioning enters
# through. Order matters only for readability.
BULK_SUFFIXES = (
    ".attention.qkv",
    ".attention.o",
    ".feed_forward.w1",
    ".feed_forward.w2",
    ".feed_forward.w3",
)
# Held at 8-bit under `mixed` — see the module docstring.
SENSITIVE_MODULES = (".adaln_modulation", "llm_cond_proj")
# Left dense under `mixed`: small, and every one of them is either an input
# embedding or the output projection.
DENSE_MODULES = (
    "input_proj",
    "t_embedding.mlp_in",
    "t_embedding.mlp_out",
    "adaln_proj",
    "final_layer.linear",
    "final_layer.adaln_modulation",
)

SHARD_BYTES = 4 * 1024 * 1024 * 1024


def log(msg: str) -> None:
    print(msg, flush=True)


# ── source resolution ─────────────────────────────────────────────────────


def resolve_src(src: str) -> Path:
    p = Path(src).expanduser()
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    log(f"[src] downloading {src} (gated — accept the license on the model page first)")
    return Path(
        snapshot_download(
            repo_id=src,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "tokenizer/*",
                "*.jinja",
                "*.txt",
                "*.model",
            ],
        )
    )


def load_component(src: Path, subfolder: str, basename: str) -> dict[str, torch.Tensor]:
    """Load a component's weights, sharded (index) or single-file."""
    base = src / subfolder if subfolder else src
    index = base / f"{basename}.safetensors.index.json"
    out: dict[str, torch.Tensor] = {}
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        for shard in sorted(set(weight_map.values())):
            out.update(load_file(str(base / shard)))
        return out
    single = base / f"{basename}.safetensors"
    if single.is_file():
        return load_file(str(single))
    raise FileNotFoundError(f"no {basename} weights under {base}")


# ── dequantization ────────────────────────────────────────────────────────

FP8_SCALE_SUFFIX = ".weight_scale"


def dequantize(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fold any published quantization back to bf16, in place of the packed keys.

    FP8 is weight-only with a per-OUTPUT-ROW float32 scale: `w.to(dtype) *
    scale.unsqueeze(1)`. NF4 is bitsandbytes' packed format, whose dequant only
    exists inside bitsandbytes — imported lazily so an FP8 conversion needs no
    CUDA-only dependency.

    CONSUMES `state`: each source tensor is popped out and freed the moment
    its bf16 replacement is built, so the two full 9.3B-parameter copies are
    never resident at once — `state` is empty (and `out` is the only live
    copy) by the time this returns. Do not read `state` after calling this.
    """
    if any(k.endswith(FP8_SCALE_SUFFIX) for k in state):
        out: dict[str, torch.Tensor] = {}
        keys = [k for k in state if not k.endswith(FP8_SCALE_SUFFIX)]
        n = 0
        for i, k in enumerate(keys):
            v = state.pop(k)
            scale = state.pop(k + "_scale", None)
            if scale is not None and v.dtype == torch.float8_e4m3fn:
                out[k] = (
                    v.to(torch.float32) * scale.to(torch.float32).unsqueeze(1)
                ).to(torch.bfloat16)
                n += 1
            else:
                out[k] = v.to(torch.bfloat16) if v.is_floating_point() else v
            del v, scale
            if i % 200 == 0:
                gc.collect()
        state.clear()
        log(f"[dequant] FP8 weight-only: {n} tensors restored to bf16")
        return out

    if any(
        k.endswith(".quant_state.bitsandbytes__nf4") or k.endswith(".absmax")
        for k in state
    ):
        try:
            import bitsandbytes.functional as bnbf
        except ImportError:  # pragma: no cover - environment-dependent
            sys.exit(
                "this checkpoint is bitsandbytes NF4; install `bitsandbytes` "
                "(CUDA host) or convert from the FP8 repo instead"
            )
        out = {}
        n = 0
        sidecar_suffixes = (
            ".absmax",
            ".quant_map",
            ".nested_absmax",
            ".nested_quant_map",
            ".quant_state.bitsandbytes__nf4",
        )
        keys = [k for k in state if not any(k.endswith(s) for s in sidecar_suffixes)]
        for i, k in enumerate(keys):
            v = state[k]
            if v.dtype == torch.uint8 and (k + ".absmax") in state:
                sidecar_keys = [kk for kk in state if kk.startswith(k + ".")]
                qs = bnbf.QuantState.from_dict(
                    {kk[len(k) + 1 :]: state[kk] for kk in sidecar_keys},
                    device="cpu",
                )
                out[k] = bnbf.dequantize_4bit(v, qs).to(torch.bfloat16)
                n += 1
                for kk in sidecar_keys:
                    del state[kk]
            else:
                out[k] = v.to(torch.bfloat16) if v.is_floating_point() else v
            del state[k], v
            if i % 200 == 0:
                gc.collect()
        state.clear()
        log(f"[dequant] NF4: {n} tensors restored to bf16")
        return out

    out = {}
    keys = list(state.keys())
    for i, k in enumerate(keys):
        v = state.pop(k)
        out[k] = v.to(torch.bfloat16) if v.is_floating_point() else v
        del v
        if i % 200 == 0:
            gc.collect()
    return out


def to_mx(t: torch.Tensor) -> mx.array:
    """torch → mlx, via float32 numpy (numpy has no bfloat16)."""
    return mx.array(t.to(torch.float32).numpy()).astype(mx.bfloat16)


# ── quantization policy ───────────────────────────────────────────────────


# Per-precision (bulk, sensitive, other) bit widths for the two "mixed"
# policies. `other` covers everything not in BULK_SUFFIXES/SENSITIVE_MODULES/
# DENSE_MODULES — small tensors that aren't worth a dedicated bucket, held at
# the same width as the sensitive one.
# Name reads bulk_sensitive. The sensitive tier stays at 8-bit in EVERY policy:
# it is 1.13B parameters across the two transformers, so holding it two bits
# above the bulk costs 0.28 GB — 2% of the smallest pack. The bits worth
# arguing about are the bulk (17.4B) and the text encoder (6.6B), which is why
# the small policies spend their budget by taking the encoder to 4-bit — the
# width the shipped flat `--precision 4` pack already runs it at — rather than
# by shaving the tier that protects glyph shape and prompt adherence.
BASE_POLICIES: dict[str, dict[str, int]] = {
    "mixed": {"bulk": 4, "sensitive": 8, "other": 8, "te": 8},
    "mixed_3_8": {"bulk": 3, "sensitive": 8, "other": 8, "te": 4},
}
# 3 bits is the FLOOR for the bulk, measured on the shipped packs rather than
# reasoned about: a `mixed_2_8` pack renders a woven grid texture at every
# prompt, seed and resolution, while `mixed_3_8` renders the same prompts
# correctly. It is not a soft quality tier, so it is refused rather than
# documented — see `docs/gotchas/models-media.md`.
MIN_BULK_BITS = 3
# The live table: `apply_overrides` rewrites the selected policy in place from
# `--bulk-bits`/`--sensitive-bits`/`--te-bits`, so any point in the space is
# reachable without minting another named policy for it.
MIXED_POLICIES: dict[str, dict[str, int]] = {
    k: dict(v) for k, v in BASE_POLICIES.items()
}


def apply_overrides(
    precision: str, bulk: int | None, sensitive: int | None, te: int | None
) -> dict[str, int]:
    """Fold per-tier CLI overrides into the named policy. Mixed policies only —
    a flat precision has no tiers to override, and silently ignoring the flags
    would ship a pack that does not match what was asked for."""
    policy = MIXED_POLICIES.get(precision)
    if policy is None:
        if bulk or sensitive or te:
            sys.exit(
                f"--bulk-bits/--sensitive-bits/--te-bits need a mixed policy; "
                f"--precision {precision} quantizes everything at one width"
            )
        return {}
    if bulk is not None:
        if bulk < MIN_BULK_BITS:
            sys.exit(
                f"--bulk-bits {bulk}: a {bulk}-bit DiT bulk renders noise, not a "
                f"softer image (measured; {MIN_BULK_BITS}-bit is the floor)"
            )
        policy["bulk"] = bulk
    if sensitive is not None:
        policy["sensitive"] = sensitive
        policy["other"] = sensitive
    if te is not None:
        policy["te"] = te
    for k, v in policy.items():
        if v not in (2, 3, 4, 5, 6, 8):
            sys.exit(f"{k} width {v} is not an affine width mlx-serve can read back")
    return policy


def bits_for(module: str, precision: str, default_bits: int) -> int | None:
    """Bit width for a module path, or None to keep it dense bf16."""
    policy = MIXED_POLICIES.get(precision)
    if policy is None:
        return default_bits
    if any(module.endswith(s) or module == s for s in DENSE_MODULES):
        return None
    if any(module.endswith(s) or module == s for s in SENSITIVE_MODULES):
        return policy["sensitive"]
    if any(module.endswith(s) for s in BULK_SUFFIXES):
        return policy["bulk"]
    return policy["other"]


def te_bits_for(precision: str, default_bits: int) -> int:
    """Bit width for the Qwen3-VL text encoder.

    A mixed policy's own width, because the encoder is a THIRD of the download
    and shrinking only the DiT does not produce a smaller pack: `mixed_3_6`
    held the encoder at 8-bit and came out LARGER than a flat `--precision 4`
    pack it was supposed to undercut. `mixed_2_6` takes it to 4-bit, which is
    exactly what the shipped flat-4 pack already runs the encoder at.
    """
    policy = MIXED_POLICIES.get(precision)
    return default_bits if policy is None else policy["te"]


# Parameter counts read off the published checkpoint, used ONLY for the size
# estimate printed at plan time (and the ordering assertion in the self-test).
# Per transformer — the pack ships TWO, and they are the same geometry.
DIT_BULK_PARAMS = 8.70e9
DIT_SENSITIVE_PARAMS = 0.565e9
DIT_DENSE_PARAMS = 0.030e9
TE_PARAMS = 6.60e9  # Qwen3-VL-8B minus the dropped vision tower
VAE_BYTES = 0.17e9


def bytes_per_param(bits: int | None, group_size: int = 64) -> float:
    """Affine layout: `bits/8` packed, plus a bf16 scale and bias per group."""
    if bits is None:
        return 2.0
    return bits / 8 + 4 / group_size


def estimate_pack_bytes(precision: str, group_size: int = 64) -> float:
    default_bits = {"3": 3, "4": 4, "8": 8}.get(precision, 8)
    dit = (
        DIT_BULK_PARAMS
        * bytes_per_param(
            bits_for(".attention.qkv", precision, default_bits), group_size
        )
        + DIT_SENSITIVE_PARAMS
        * bytes_per_param(
            bits_for(".adaln_modulation", precision, default_bits), group_size
        )
        + DIT_DENSE_PARAMS
        * bytes_per_param(
            bits_for("final_layer.linear", precision, default_bits), group_size
        )
    )
    te = TE_PARAMS * bytes_per_param(te_bits_for(precision, default_bits), group_size)
    return 2 * dit + te + VAE_BYTES


def emit_linear(
    out: dict[str, mx.array],
    module: str,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    bits: int | None,
    group_size: int,
) -> int:
    """Write one linear as either a quantized triple or a dense weight.

    A width that does not divide the input evenly stays DENSE rather than being
    silently rounded — `IgLinear` solves (bits, group_size) back out of the
    packed geometry, so a mismatched pack would be read as a different width.
    """
    w = to_mx(weight)
    in_features = w.shape[1]
    if bias is not None:
        out[f"{module}.bias"] = to_mx(bias)
    if bits is None or in_features % group_size != 0:
        out[f"{module}.weight"] = w
        return 0
    q, s, b = mx.quantize(w, group_size=group_size, bits=bits)
    out[f"{module}.weight"] = q
    out[f"{module}.scales"] = s
    out[f"{module}.biases"] = b
    return bits


# ── writer ────────────────────────────────────────────────────────────────


def write_shards(
    out_dir: Path, tensors: dict[str, mx.array], basename: str = "model"
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, mx.array]] = [{}]
    sizes = [0]
    for k, v in tensors.items():
        nbytes = v.nbytes
        if sizes[-1] + nbytes > SHARD_BYTES and shards[-1]:
            shards.append({})
            sizes.append(0)
        shards[-1][k] = v
        sizes[-1] += nbytes
    if len(shards) == 1:
        mx.save_safetensors(str(out_dir / basename), shards[0])
        log(
            f"[write] {out_dir.name}/{basename}.safetensors ({sizes[0] / 1e9:.2f} GB, {len(shards[0])} tensors)"
        )
        return
    weight_map: dict[str, str] = {}
    total = 0
    for i, shard in enumerate(shards):
        name = f"{basename}-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        mx.save_safetensors(str(out_dir / name.removesuffix(".safetensors")), shard)
        for k in shard:
            weight_map[k] = name
        total += sizes[i]
    (out_dir / f"{basename}.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2
        )
    )
    log(f"[write] {out_dir.name}: {len(shards)} shards, {total / 1e9:.2f} GB")


# ── components ────────────────────────────────────────────────────────────


def convert_transformer(
    src: Path,
    out: Path,
    subfolder: str,
    precision: str,
    default_bits: int,
    group_size: int,
) -> None:
    """One Ideogram4Transformer. Module paths pass through UNCHANGED — the
    reference loads its own state dict with `load_state_dict`, so upstream key
    names already are the module tree `ideogram4.zig` reads.

    Pops each source tensor out of `state` the moment it's converted, so the
    bf16 source and the growing quantized `tensors` dict are never both fully
    resident — same reason `dequantize` consumes its input. At 9.3B params
    per transformer (x2, converted sequentially) holding both whole was the
    difference between finishing and an OOM kill.
    """
    state = dequantize(load_component(src, subfolder, "diffusion_pytorch_model"))
    tensors: dict[str, mx.array] = {}
    counts: dict[str, int] = {}

    # Group by module: a linear is a `.weight` plus an optional `.bias`.
    modules = sorted({k.rsplit(".", 1)[0] for k in state if k.endswith(".weight")})
    for i, module in enumerate(modules):
        w = state.pop(f"{module}.weight")
        b = state.pop(f"{module}.bias", None)
        if w.ndim != 2:
            # 1-D norms and the [2, emb] indicator table: dense, verbatim.
            tensors[f"{module}.weight"] = to_mx(w)
            if b is not None:
                tensors[f"{module}.bias"] = to_mx(b)
            counts["dense-nd"] = counts.get("dense-nd", 0) + 1
        elif module == "embed_image_indicator":
            # A gather table, not a projection: it is READ by row, and mlx-serve
            # reads it with a dense slice. Quantizing it would need a
            # dequantize-on-gather path that does not exist here.
            tensors[f"{module}.weight"] = to_mx(w)
            counts["dense-table"] = counts.get("dense-table", 0) + 1
        else:
            bits = bits_for(module, precision, default_bits)
            got = emit_linear(tensors, module, w, b, bits, group_size)
            key = f"{got}-bit" if got else "dense"
            counts[key] = counts.get(key, 0) + 1
        del w, b
        if i % 64 == 0:
            gc.collect()

    # Anything left is not `<module>.weight[/.bias]` for a module the loop
    # above visited — there should be nothing, but pass it through rather
    # than drop it silently.
    for k in list(state.keys()):
        v = state.pop(k)
        tensors[k] = to_mx(v)
        log(f"[warn] {subfolder}: passthrough of unrecognized tensor {k}")

    log(f"[{subfolder}] {counts}")
    write_shards(out / subfolder, tensors)
    del state, tensors
    gc.collect()
    try:
        mx.clear_cache()
    except AttributeError:  # pragma: no cover - older mlx
        pass


def convert_text_encoder(src: Path, out: Path, te_bits: int, group_size: int) -> None:
    """Qwen3-VL-8B, TEXT TOWER ONLY, renamed into the flat layout
    `flux.loadTextEncoderWith` reads.

    The vision tower is dropped: Ideogram runs the encoder in text-only mode
    (`_get_qwen3_vl_embeddings` reaches straight for `language_model`), so its
    ~1 GB of ViT weights would be resident and never read.
    """
    state = dequantize(load_component(src, "text_encoder", "model"))

    # The text tower's prefix is a transformers-version detail — probe for it
    # rather than pinning one spelling.
    prefix = None
    for cand in ("language_model.", "model.language_model.", "model.", ""):
        if f"{cand}layers.0.self_attn.q_proj.weight" in state:
            prefix = cand
            break
    if prefix is None:
        sys.exit(
            "text_encoder: could not locate the Qwen3-VL text tower (no layers.0.self_attn.q_proj.weight)"
        )
    log(f"[text_encoder] text tower prefix: {prefix!r}")

    tensors: dict[str, mx.array] = {}
    dropped = 0
    n_layers = 0
    for k, v in state.items():
        if not k.startswith(prefix) or k.startswith("visual."):
            dropped += 1
            continue
        name = k[len(prefix) :]
        if name.startswith("visual."):
            dropped += 1
            continue
        m = re.match(r"^layers\.(\d+)\.", name)
        if m:
            n_layers = max(n_layers, int(m.group(1)) + 1)
        # Projections quantize; norms and everything 1-D stay dense.
        if name.endswith(".weight") and v.ndim == 2:
            module = name[: -len(".weight")]
            if module == "embed_tokens":
                # mlx-serve dequantizes this table once at load, so it MUST
                # ship quantized — the loader reads `.scales`/`.biases`
                # unconditionally.
                emit_linear(tensors, module, v, None, te_bits, group_size)
                continue
            if module == "lm_head":
                dropped += 1  # never run: the encoder stops at the taps
                continue
            emit_linear(
                tensors,
                module,
                v,
                state.get(module + ".bias"),
                te_bits,
                group_size,
            )
        elif name.endswith(".bias"):
            continue  # emitted alongside its weight
        else:
            tensors[name] = to_mx(v)
    log(
        f"[text_encoder] {n_layers} layers kept, {dropped} tensors dropped (vision tower / lm_head)"
    )
    write_shards(out / "text_encoder", tensors)


# Diffusers VAE keys whose weights are 1x1 attention projections that
# mlx-serve's shared Flux2 decoder loads as QUANTIZED linears.
VAE_ATTN_RE = re.compile(
    r"^(en|de)coder\.mid_block\.attentions\.0\.to_(q|k|v|out\.0)\.weight$"
)


def convert_vae(src: Path, out: Path, group_size: int) -> None:
    """Flux2 KL autoencoder, in diffusers naming — which is exactly what
    `flux.loadVae` reads, so nothing is renamed. Convs are transposed from
    torch OIHW into MLX's OHWI; `bn.running_*` rides along even though
    Ideogram's decode path uses its own published latent table instead."""
    state = dequantize(
        load_file(str(src / "vae" / "diffusion_pytorch_model.safetensors"))
    )
    tensors: dict[str, mx.array] = {}
    n_conv = 0
    for k, v in state.items():
        if VAE_ATTN_RE.match(k):
            orig_module = k[: -len(".weight")]
            module = (
                orig_module[:-2] if orig_module.endswith("to_out.0") else orig_module
            )
            w = v
            if w.ndim == 4:  # some exports ship these as 1x1 Conv2d
                w = w.reshape(w.shape[0], w.shape[1])
            emit_linear(
                tensors, module, w, state.get(orig_module + ".bias"), 8, group_size
            )
            continue
        if k.endswith(".bias") and VAE_ATTN_RE.match(k[: -len(".bias")] + ".weight"):
            continue
        if v.ndim == 4:
            tensors[k] = to_mx(v.permute(0, 2, 3, 1).contiguous())
            n_conv += 1
        else:
            tensors[k] = to_mx(v)
    if "bn.running_mean" not in tensors:
        log(
            "[warn] vae: no bn.running_mean — the Flux2 decoder's own entry point needs it"
        )
    log(f"[vae] {len(tensors)} tensors, {n_conv} convs transposed OIHW→OHWI")
    write_shards(out / "vae", tensors)


def copy_tokenizer(src: Path, out: Path) -> None:
    dst = out / "tokenizer"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in (src / "tokenizer").iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
            n += 1
    log(f"[tokenizer] {n} files copied")


def chat_framing(src: Path, out: Path) -> tuple[str, str]:
    """Render the checkpoint's OWN chat template around a sentinel and split it.

    mlx-serve frames the caption with two literal strings rather than running
    Jinja per request, so those strings have to come from the template that
    trained the model — not from a transcription. When transformers is not
    installed the pack simply omits them and the engine falls back to
    Qwen3-VL-Instruct's ChatML rendering, which is what the template produces
    for a single text turn.
    """
    sentinel = "\x00IDEOGRAM_PROMPT\x00"
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(src), subfolder="tokenizer")
        text = tok.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": sentinel}]}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception as e:  # noqa: BLE001 - any failure is a clean fallback
        log(
            f"[chat] template not rendered ({e}); the engine will use its ChatML default"
        )
        return "", ""
    if sentinel not in text:
        log("[chat] rendered template did not contain the prompt; falling back")
        return "", ""
    prefix, suffix = text.split(sentinel, 1)
    log(f"[chat] prefix={prefix!r} suffix={suffix!r}")
    return prefix, suffix


# ── self-test (no torch/mlx/checkpoint needed for the policy half) ─────────


def self_test() -> None:
    """Unit-test the quantization policy and the size estimate."""
    bulk = "blocks.0.attention.qkv"
    mlp = "blocks.0.feed_forward.w2"
    sensitive = "blocks.0.adaln_modulation"
    cond = "llm_cond_proj"
    dense = "final_layer.linear"

    # Flat precisions: one width everywhere, including the sensitive modules.
    for p_name, want in (("3", 3), ("4", 4), ("8", 8)):
        for m in (bulk, mlp, sensitive, cond, dense):
            assert bits_for(m, p_name, want) == want, (p_name, m)
        assert te_bits_for(p_name, want) == want

    # Mixed policies: bulk / sensitive / dense-bf16, plus the text encoder.
    for p_name, b, sens, te in (
        ("mixed", 4, 8, 8),
        ("mixed_3_8", 3, 8, 4),
    ):
        assert bits_for(bulk, p_name, 8) == b, p_name
        assert bits_for(mlp, p_name, 8) == b, p_name
        assert bits_for(sensitive, p_name, 8) == sens, p_name
        assert bits_for(cond, p_name, 8) == sens, p_name
        assert bits_for(dense, p_name, 8) is None, p_name
        assert te_bits_for(p_name, 8) == te, p_name
    # Overrides reach every tier without inventing another named policy.
    try:
        apply_overrides("mixed_3_8", bulk=4, sensitive=None, te=8)
        assert bits_for(bulk, "mixed_3_8", 8) == 4
        assert bits_for(sensitive, "mixed_3_8", 8) == 8
        assert te_bits_for("mixed_3_8", 8) == 8
    finally:
        MIXED_POLICIES["mixed_3_8"] = dict(BASE_POLICIES["mixed_3_8"])
    assert bits_for(bulk, "mixed_3_8", 8) == 3

    # A 2-bit DiT bulk is a MEASURED failure, not a quality tier: the
    # `mixed_2_8` pack renders a woven grid texture at every prompt, seed and
    # resolution, while `mixed_3_8` renders the same prompts correctly. So the
    # policy is gone and the width is unreachable through the override too —
    # `--bulk-bits 2` would otherwise mint the same pack under another name.
    assert "mixed_2_8" not in BASE_POLICIES
    assert "mixed_2_8" not in MIXED_POLICIES
    for bad in (1, 2):
        try:
            apply_overrides("mixed_3_8", bulk=bad, sensitive=None, te=None)
        except SystemExit as e:
            assert "renders noise" in str(e), str(e)
        else:
            raise AssertionError(f"--bulk-bits {bad} was accepted")
        finally:
            MIXED_POLICIES["mixed_3_8"] = dict(BASE_POLICIES["mixed_3_8"])
    # The floor is on the BULK alone — the sensitive tier is small enough that
    # nobody would spend it there, but refusing widths the engine can read is
    # not this function's job.
    assert bits_for(bulk, "mixed_3_8", 8) == 3
    print("[self-test] quantization policy OK")

    # Every width the policies emit must be one the engine can solve back out
    # of the packed geometry — the loader reads GEOMETRY, never the config
    # block (`flux.inferQuantGeometry`, `ideogram4.IgLinear.load`).
    # Every affine width the engine accepts — the overrides can select any of
    # them, not just the ones a named policy happens to use today.
    widths = [2, 3, 4, 5, 6, 8]
    probe = mx.random.normal((128, 512)).astype(mx.bfloat16)
    for bits in widths:
        q, sc, _ = mx.quantize(probe, group_size=64, bits=bits)
        assert 32 * q.shape[1] % probe.shape[1] == 0
        assert 32 * q.shape[1] // probe.shape[1] == bits, (bits, q.shape)
        assert probe.shape[1] // sc.shape[1] == 64, (bits, sc.shape)
    print(f"[self-test] packed geometry solves back at {widths} OK")

    # A smaller policy must produce a smaller PACK. `mixed_3_6` did not, for a
    # while: its bulk shrank but its text encoder stayed at 8-bit, which alone
    # is a third of the download.
    sizes = {
        p: estimate_pack_bytes(p)
        for p in ("8", "4", "3", "mixed", "mixed_3_8")
    }
    assert sizes["mixed"] < sizes["8"]
    assert sizes["mixed_3_8"] < sizes["4"]
    assert sizes["mixed_3_8"] < sizes["3"]

    # Protecting the sensitive tier is nearly free — that is WHY it stays at
    # 8-bit in every policy rather than tracking the bulk down. If this ever
    # stops being true the tier is worth re-thinking, not silently paying for.
    cheap = estimate_pack_bytes("mixed_3_8") - (
        2 * DIT_SENSITIVE_PARAMS * (bytes_per_param(8) - bytes_per_param(6))
    )
    assert estimate_pack_bytes("mixed_3_8") - cheap < 0.4e9
    print(
        "[self-test] pack size ordering OK: "
        + ", ".join(f"{k}={v / 1e9:.1f}GB" for k, v in sizes.items())
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", help="HF repo id or a local checkpoint dir")
    ap.add_argument("--out", help="destination pack directory")
    ap.add_argument(
        "--precision",
        choices=("3", "4", "8", "mixed", "mixed_3_8"),
        default="mixed",
    )
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument(
        "--bulk-bits", type=int, help="override a mixed policy's attention/MLP width"
    )
    ap.add_argument(
        "--sensitive-bits",
        type=int,
        help="override a mixed policy's modulation/conditioning width",
    )
    ap.add_argument(
        "--te-bits", type=int, help="override a mixed policy's text-encoder width"
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="run the policy/size unit tests and exit (no checkpoint needed)",
    )
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.src or not args.out:
        ap.error("--src and --out are required")

    # Only read for a flat (non-mixed) precision — bits_for takes the
    # MIXED_POLICIES branch instead whenever args.precision names one of those.
    default_bits = {"3": 3, "4": 4, "8": 8}.get(args.precision, 8)
    # BEFORE resolve_src: a refused width must not cost a multi-GB download.
    apply_overrides(args.precision, args.bulk_bits, args.sensitive_bits, args.te_bits)

    src = resolve_src(args.src)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    te_bits = te_bits_for(args.precision, default_bits)
    log(
        f"[plan] {src} → {out}  precision={args.precision} group_size={args.group_size} "
        f"text_encoder={te_bits}-bit  (estimated pack ≈ {estimate_pack_bytes(args.precision, args.group_size) / 1e9:.1f} GB)"
    )

    convert_text_encoder(src, out, te_bits, args.group_size)
    convert_vae(src, out, args.group_size)
    copy_tokenizer(src, out)
    convert_transformer(
        src, out, "transformer", args.precision, default_bits, args.group_size
    )
    convert_transformer(
        src,
        out,
        "unconditional_transformer",
        args.precision,
        default_bits,
        args.group_size,
    )

    prefix, suffix = chat_framing(src, out)
    config = {
        "model_type": "ideogram4",
        "_class_name": "Ideogram4Pipeline",
        "source_repo": args.src,
        "precision": args.precision,
        "widths": MIXED_POLICIES.get(args.precision, {"flat": default_bits}),
        "quantization": {"mode": "affine", "group_size": args.group_size},
    }
    if prefix or suffix:
        config["chat_prefix"] = prefix
        config["chat_suffix"] = suffix
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "model_index.json").write_text(
        json.dumps({"_class_name": "Ideogram4Pipeline"}, indent=2)
    )

    # LAST, deliberately: `model_discovery.requiredMediaMarker` treats this file
    # as the pack's completeness marker.
    (out / "unconditional_transformer" / "config.json").write_text(
        json.dumps({"_class_name": "Ideogram4Transformer"}, indent=2)
    )
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    log(f"[done] {out} — {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()

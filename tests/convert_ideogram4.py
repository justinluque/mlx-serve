#!/usr/bin/env python3
"""Convert an upstream Ideogram 4 checkpoint into an mlx-serve pack.

    python tests/convert_ideogram4.py \
        --src ideogram-ai/ideogram-4-fp8 \
        --out ~/.mlx-serve/models/justintime47/Ideogram-4-MLX-Serve-mixed_3_8 \
        --precision mixed_3_8

FP8 is the canonical source. NF4 dequant only exists inside `bitsandbytes`,
which needs a CUDA host — there is no way to unpack a bitsandbytes NF4
checkpoint on Apple Silicon, so `--src ideogram-ai/ideogram-4-nf4` dead-ends
here with a named error pointing at the FP8 repo instead. Both are read the
same way otherwise: weight-only quantization, dequantized to bf16 tensor by
tensor, then re-quantized with `mx.quantize` into the affine layout
`flux.QLinear` / `ideogram4.IgLinear` read.

Sizes below are the WHOLE pack — both transformers, the text encoder and the
VAE — from `estimate_pack_bytes`, which the self-test pins in the same order.

`--precision` (a mixed name reads bulk_sensitive; the text encoder's width is
per policy below, and `--bulk-bits`/`--sensitive-bits`/`--te-bits` override
any tier of a mixed policy):

  3         every projection at 3-bit, ~11 GB. Noticeably softer text
            rendering than 4-bit.
  4         every projection at 4-bit, ~14 GB. Smallest well-tested flat
            point.
  8         every projection at 8-bit, ~27 GB. The reference point.
  mixed     the bulk (attention + MLP, ~8.7B of the 9.3B per transformer) at
            4-bit, the modulation and conditioning projections at 8-bit,
            everything small enough not to matter left dense bf16, text
            encoder 8-bit. ~18 GB.
  mixed_3_8 bulk 3-bit, sensitive tier still 8-bit, text encoder 4-bit.
            ~13 GB — THE PRESET: fits a 24 GB Mac with 6 GB not guaranteed by
            the system (~18 GB usable), with headroom left for activations
            and everything else running.

2-bit on the bulk is NOT offered. A `mixed_2_8` pack was published and
withdrawn: 2-bit affine on the DiT bulk renders a woven grid texture at every
prompt, seed and resolution — indistinguishable from a wrong patch packing or
a scrambled MRoPE table from the outside, and the check that settles it is
the SAME prompt on a pack at a different width (`docs/gotchas/models-media.md`).
`MIN_BULK_BITS` refuses `--bulk-bits 2` by name rather than shipping a pack
that does not render.

The mixed policies are not a guess about which tensors "matter" in general —
they are about which ones are cheap, and it independently matches the
per-layer precision map QuantFunc ships for their CUDA engine
(`Ideogram-4-Series/config.json`: block attention/MLP at 4-bit, non-block
projections at 8-bit, `adaln_modulation` held out of quantization entirely).
`adaln_modulation` is 320M parameters whose output is a per-layer SCALE and a
tanh GATE, so its error is multiplicative over the whole residual stream
rather than averaged into one projection (the same reason mlx-serve's
MageFlow notes flag `img_mod`/`txt_mod` at 4-bit); the embedding, timestep and
final-layer projections together are under 30M, so holding them at bf16 costs
~60 MB and removes them as suspects entirely. The sensitive tier stays at
8-bit in EVERY mixed policy rather than tracking the bulk down: it is 1.13B
parameters across the two transformers, so 8-bit instead of 6 costs 0.28 GB —
2% of the smallest pack, and the tier a low-bit bulk most needs protecting
from. The bits worth arguing about are the bulk (17.4B total) and the text
encoder (6.6B, a third of the download on its own), which is why `mixed_3_8`
spends its budget by taking the encoder to 4-bit — the width the shipped flat
`--precision 4` pack already runs it at — rather than by shaving the tier
that protects glyph shape and prompt adherence. See `te_bits_for` and
`apply_overrides`.

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
import re
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import torch
from safetensors import safe_open
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
# Held at 8-bit under every mixed policy — see the module docstring.
SENSITIVE_MODULES = (".adaln_modulation", "llm_cond_proj")
# Left dense under every mixed policy: small, and every one of them is either
# an input embedding or the output projection.
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


class LazyComponent:
    """Per-tensor lazy access into a (possibly sharded) safetensors component.

    Nothing is loaded until `.get(key)` is called, and each call reads only
    that one tensor via `safe_open`'s mmap-backed access — never the whole
    component. This is what lets a 9.3B-parameter transformer convert on a
    24 GB machine: `load_file`-style whole-dict loading needs the entire
    component resident (and a SECOND whole copy once dequantized to bf16)
    before a single byte can be freed.
    """

    def __init__(self, src: Path, subfolder: str, basename: str):
        base = src / subfolder if subfolder else src
        index = base / f"{basename}.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            files = sorted({base / f for f in weight_map.values()})
        else:
            single = base / f"{basename}.safetensors"
            if not single.is_file():
                raise FileNotFoundError(f"no {basename} weights under {base}")
            files = [single]
        self._handles = {f: safe_open(str(f), framework="pt") for f in files}
        self._key_to_file: dict[str, Path] = {}
        for f, h in self._handles.items():
            for k in h.keys():
                self._key_to_file[k] = f
        self.keys: list[str] = list(self._key_to_file.keys())

    def has(self, key: str) -> bool:
        return key in self._key_to_file

    def get(self, key: str) -> torch.Tensor:
        return self._handles[self._key_to_file[key]].get_tensor(key)


# ── dequantization ────────────────────────────────────────────────────────

FP8_SCALE_SUFFIX = ".weight_scale"
# Packing sidecars: consumed alongside the weight they belong to, never
# treated as a module of their own.
_PACKING_SUFFIXES = (
    FP8_SCALE_SUFFIX,
    ".absmax",
    ".quant_map",
    ".nested_absmax",
    ".nested_quant_map",
    ".quant_state.bitsandbytes__nf4",
)


def _is_packing_sidecar(key: str) -> bool:
    return key.endswith(_PACKING_SUFFIXES)


def quant_kind(keys: list[str]) -> str:
    """Detect the source's packed format from its key names alone — no
    tensor data read."""
    if any(k.endswith(FP8_SCALE_SUFFIX) for k in keys):
        return "fp8 weight-only (per-row scale)"
    if any(k.endswith(".quant_state.bitsandbytes__nf4") or k.endswith(".absmax") for k in keys):
        return "bitsandbytes NF4"
    return "dense (no packed quantization detected)"


def dequant_tensor(comp: LazyComponent, key: str) -> torch.Tensor:
    """Fetch ONE tensor from `comp`, dequantized to bf16 if it's FP8 or
    NF4-packed. Reads only this tensor plus its own scale/absmax sidecars —
    the component's other tensors are never touched, let alone held."""
    v = comp.get(key)
    if v.dtype == torch.float8_e4m3fn and comp.has(key + "_scale"):
        scale = comp.get(key + "_scale")
        return (v.to(torch.float32) * scale.to(torch.float32).unsqueeze(1)).to(torch.bfloat16)
    if v.dtype == torch.uint8 and comp.has(key + ".absmax"):
        import bitsandbytes.functional as bnbf

        side = {kk[len(key) + 1 :]: comp.get(kk) for kk in comp.keys if kk.startswith(key + ".")}
        qs = bnbf.QuantState.from_dict(side, device="cpu")
        return bnbf.dequantize_4bit(v, qs).to(torch.bfloat16)
    return v.to(torch.bfloat16) if v.is_floating_point() else v


def dequantize(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Whole-dict dequant — only used for the VAE now (0.17 GB, harmless to
    hold twice). `dequant_tensor` above is the per-tensor version the
    transformer and text-encoder conversions use instead.

    Fold any published quantization back to bf16, in place of the packed keys.

    FP8 is weight-only with a per-OUTPUT-ROW float32 scale: `w.to(dtype) *
    scale.unsqueeze(1)`. NF4 is bitsandbytes' packed format, whose dequant only
    exists inside bitsandbytes — imported lazily so an FP8 conversion needs no
    CUDA-only dependency.
    """
    if any(k.endswith(FP8_SCALE_SUFFIX) for k in state):
        out: dict[str, torch.Tensor] = {}
        n = 0
        for k, v in state.items():
            if k.endswith(FP8_SCALE_SUFFIX):
                continue
            scale = state.get(k + "_scale")
            if scale is not None and v.dtype == torch.float8_e4m3fn:
                out[k] = (v.to(torch.float32) * scale.to(torch.float32).unsqueeze(1)).to(torch.bfloat16)
                n += 1
            else:
                out[k] = v.to(torch.bfloat16) if v.is_floating_point() else v
        log(f"[dequant] FP8 weight-only: {n} tensors restored to bf16")
        return out

    if any(k.endswith(".quant_state.bitsandbytes__nf4") or k.endswith(".absmax") for k in state):
        try:
            import bitsandbytes.functional as bnbf
        except ImportError:  # pragma: no cover - environment-dependent
            sys.exit(
                "this checkpoint is bitsandbytes NF4; install `bitsandbytes` "
                "(CUDA host) or convert from the FP8 repo instead"
            )
        out = {}
        n = 0
        for k, v in state.items():
            if any(k.endswith(s) for s in (".absmax", ".quant_map", ".nested_absmax", ".nested_quant_map", ".quant_state.bitsandbytes__nf4")):
                continue
            if v.dtype == torch.uint8 and (k + ".absmax") in state:
                qs = bnbf.QuantState.from_dict(
                    {kk[len(k) + 1 :]: state[kk] for kk in state if kk.startswith(k + ".")},
                    device="cpu",
                )
                out[k] = bnbf.dequantize_4bit(v, qs).to(torch.bfloat16)
                n += 1
            else:
                out[k] = v.to(torch.bfloat16) if v.is_floating_point() else v
        log(f"[dequant] NF4: {n} tensors restored to bf16")
        return out

    return {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in state.items()}


def to_mx(t: torch.Tensor) -> mx.array:
    """torch → mlx, via float32 numpy (numpy has no bfloat16)."""
    return mx.array(t.to(torch.float32).numpy()).astype(mx.bfloat16)


# ── quantization policy ───────────────────────────────────────────────────

# 2-bit affine on the DiT bulk does not render — see the module docstring and
# docs/gotchas/models-media.md. `--bulk-bits` below this is a refusal, not a
# documented small option.
MIN_BULK_BITS = 3

# Name reads bulk_sensitive. The sensitive tier stays at 8-bit in EVERY
# policy; `other` covers everything not in BULK_SUFFIXES/SENSITIVE_MODULES/
# DENSE_MODULES — small tensors that aren't worth a dedicated bucket, held at
# the same width as the sensitive one. `te` is the Qwen3-VL text encoder's
# own width, independent of the DiT bulk (see the module docstring).
BASE_POLICIES: dict[str, dict[str, int]] = {
    "mixed": {"bulk": 4, "sensitive": 8, "other": 8, "te": 8},
    "mixed_3_8": {"bulk": 3, "sensitive": 8, "other": 8, "te": 4},
}
# The live table: `apply_overrides` rewrites the selected policy in place from
# `--bulk-bits`/`--sensitive-bits`/`--te-bits`, so any point in the space is
# reachable without minting another named policy for it.
MIXED_POLICIES: dict[str, dict[str, int]] = {k: dict(v) for k, v in BASE_POLICIES.items()}


def apply_overrides(precision: str, bulk: int | None, sensitive: int | None, te: int | None) -> dict[str, int]:
    """Fold per-tier CLI overrides into the named policy IN PLACE. Mixed
    policies only — a flat precision has no tiers to override, and silently
    ignoring the flags would ship a pack that does not match what was asked
    for. Raises ValueError rather than exiting, so the self-test can probe a
    refusal without killing the process."""
    policy = MIXED_POLICIES.get(precision)
    if policy is None:
        if bulk is not None or sensitive is not None or te is not None:
            raise ValueError(
                f"--bulk-bits/--sensitive-bits/--te-bits need a mixed policy; "
                f"--precision {precision} quantizes everything at one width"
            )
        return {}
    if bulk is not None:
        if bulk < MIN_BULK_BITS:
            raise ValueError(
                f"--bulk-bits {bulk}: a {bulk}-bit DiT bulk renders a woven-grid "
                f"artifact, not an image (measured on the withdrawn mixed_2_8 "
                f"pack) — {MIN_BULK_BITS} is the floor"
            )
        policy["bulk"] = bulk
    if sensitive is not None:
        policy["sensitive"] = sensitive
        policy["other"] = sensitive
    if te is not None:
        policy["te"] = te
    for k, v in policy.items():
        if v not in (2, 3, 4, 5, 6, 8):
            raise ValueError(f"{k} width {v} is not an affine width mlx-serve can read back")
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
    """Bit width for the Qwen3-VL text encoder: a mixed policy's own `te`
    width, because the encoder is a THIRD of the download and shrinking only
    the DiT does not produce a smaller pack — see the module docstring."""
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
        DIT_BULK_PARAMS * bytes_per_param(bits_for(".attention.qkv", precision, default_bits), group_size)
        + DIT_SENSITIVE_PARAMS * bytes_per_param(bits_for(".adaln_modulation", precision, default_bits), group_size)
        + DIT_DENSE_PARAMS * bytes_per_param(bits_for("final_layer.linear", precision, default_bits), group_size)
    )
    te = TE_PARAMS * bytes_per_param(te_bits_for(precision, default_bits), group_size)
    return 2 * dit + te + VAE_BYTES


def emit_linear(out: dict[str, mx.array], module: str, weight: torch.Tensor, bias: torch.Tensor | None, bits: int | None, group_size: int) -> int:
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


def write_shards(out_dir: Path, tensors: dict[str, mx.array], basename: str = "model") -> None:
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
        log(f"[write] {out_dir.name}/{basename}.safetensors ({sizes[0] / 1e9:.2f} GB, {len(shards[0])} tensors)")
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
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2)
    )
    log(f"[write] {out_dir.name}: {len(shards)} shards, {total / 1e9:.2f} GB")


# ── components ────────────────────────────────────────────────────────────


def convert_transformer(src: Path, out: Path, subfolder: str, precision: str, default_bits: int, group_size: int) -> None:
    """One Ideogram4Transformer. Module paths pass through UNCHANGED — the
    reference loads its own state dict with `load_state_dict`, so upstream key
    names already are the module tree `ideogram4.zig` reads.

    Streamed module-by-module: only one module's bf16 tensors are ever
    resident at once, and `mx.eval` runs right after each one so MLX's own
    lazy graph doesn't quietly re-accumulate the whole transformer anyway."""
    comp = LazyComponent(src, subfolder, "diffusion_pytorch_model")
    log(f"[{subfolder}] source: {quant_kind(comp.keys)}")
    tensors: dict[str, mx.array] = {}
    counts: dict[str, int] = {}
    consumed: set[str] = set()

    # Group by module: a linear is a `.weight` plus an optional `.bias`.
    modules = sorted({k.rsplit(".", 1)[0] for k in comp.keys if k.endswith(".weight") and not _is_packing_sidecar(k)})
    for module in modules:
        wkey, bkey = f"{module}.weight", f"{module}.bias"
        consumed.add(wkey)
        w = dequant_tensor(comp, wkey)
        b = None
        if comp.has(bkey):
            consumed.add(bkey)
            b = dequant_tensor(comp, bkey)

        if w.ndim != 2:
            # 1-D norms and the [2, emb] indicator table: dense, verbatim.
            tensors[wkey] = to_mx(w)
            if b is not None:
                tensors[bkey] = to_mx(b)
            counts["dense-nd"] = counts.get("dense-nd", 0) + 1
        elif module == "embed_image_indicator":
            # A gather table, not a projection: it is READ by row, and mlx-serve
            # reads it with a dense slice. Quantizing it would need a
            # dequantize-on-gather path that does not exist here.
            tensors[wkey] = to_mx(w)
            counts["dense-table"] = counts.get("dense-table", 0) + 1
        else:
            bits = bits_for(module, precision, default_bits)
            got = emit_linear(tensors, module, w, b, bits, group_size)
            key = f"{got}-bit" if got else "dense"
            counts[key] = counts.get(key, 0) + 1

        # Materialize this module's arrays now, then drop the torch/numpy
        # source: without the eval, MLX's lazy graph keeps every module's
        # dequant chain alive until `write_shards`' save call forces it all
        # at once — the same OOM, just moved from torch to Metal.
        mx.eval(*(v for k, v in tensors.items() if k.startswith(module + ".")))
        del w, b

    # Anything that is not `<module>.weight[/.bias]` and not packing metadata.
    for k in comp.keys:
        if k in consumed or _is_packing_sidecar(k):
            continue
        tensors[k] = to_mx(dequant_tensor(comp, k))
        mx.eval(tensors[k])
        log(f"[warn] {subfolder}: passthrough of unrecognized tensor {k}")

    log(f"[{subfolder}] {counts}")
    write_shards(out / subfolder, tensors)
    del comp, tensors
    gc.collect()


def convert_text_encoder(src: Path, out: Path, te_bits: int, group_size: int) -> None:
    """Qwen3-VL-8B, TEXT TOWER ONLY, renamed into the flat layout
    `flux.loadTextEncoderWith` reads.

    The vision tower is dropped: Ideogram runs the encoder in text-only mode
    (`_get_qwen3_vl_embeddings` reaches straight for `language_model`), so its
    ~1 GB of ViT weights are never even FETCHED here, let alone dequantized —
    `LazyComponent` lets us skip them by key name before reading any data.
    """
    comp = LazyComponent(src, "text_encoder", "model")
    log(f"[text_encoder] source: {quant_kind(comp.keys)}")

    # The text tower's prefix is a transformers-version detail — probe for it
    # rather than pinning one spelling. `comp.keys` is metadata-only, so this
    # costs nothing.
    prefix = None
    for cand in ("language_model.", "model.language_model.", "model.", ""):
        if f"{cand}layers.0.self_attn.q_proj.weight" in comp.keys:
            prefix = cand
            break
    if prefix is None:
        sys.exit("text_encoder: could not locate the Qwen3-VL text tower (no layers.0.self_attn.q_proj.weight)")
    log(f"[text_encoder] text tower prefix: {prefix!r}")

    tensors: dict[str, mx.array] = {}
    consumed: set[str] = set()
    dropped = 0
    n_layers = 0
    for k in comp.keys:
        if _is_packing_sidecar(k):
            continue  # consumed alongside its owning weight, never standalone
        name = k[len(prefix) :] if k.startswith(prefix) else k
        if not k.startswith(prefix) or name.startswith("visual."):
            if k.endswith(".weight"):
                dropped += 1
            continue  # vision tower: never fetched — its bytes stay on disk
        if k in consumed:
            continue
        m = re.match(r"^layers\.(\d+)\.", name)
        if m:
            n_layers = max(n_layers, int(m.group(1)) + 1)
        if name.endswith(".bias"):
            continue  # emitted alongside its weight below
        if not name.endswith(".weight"):
            tensors[name] = to_mx(dequant_tensor(comp, k))
            mx.eval(tensors[name])
            continue

        module = name[: -len(".weight")]
        w = dequant_tensor(comp, k)
        if w.ndim != 2:
            # Norms and other 1-D weights stay dense.
            tensors[name] = to_mx(w)
            mx.eval(tensors[name])
            del w
            continue
        if module == "lm_head":
            dropped += 1  # never run: the encoder stops at the taps
            del w
            continue
        bkey = k[: -len(".weight")] + ".bias"
        b = None
        if comp.has(bkey):
            consumed.add(bkey)
            b = dequant_tensor(comp, bkey)
        if module == "embed_tokens":
            # mlx-serve dequantizes this table once at load, so it MUST
            # ship quantized — the loader reads `.scales`/`.biases`
            # unconditionally.
            emit_linear(tensors, module, w, None, te_bits, group_size)
        else:
            emit_linear(tensors, module, w, b, te_bits, group_size)
        mx.eval(*(v for kk, v in tensors.items() if kk.startswith(module + ".")))
        del w, b

    log(f"[text_encoder] {n_layers} layers kept, {dropped} tensors dropped (vision tower / lm_head, never loaded)")
    write_shards(out / "text_encoder", tensors)
    del comp, tensors
    gc.collect()


# Diffusers VAE keys whose weights are 1x1 attention projections that
# mlx-serve's shared Flux2 decoder loads as QUANTIZED linears.
VAE_ATTN_RE = re.compile(r"^(en|de)coder\.mid_block\.attentions\.0\.to_(q|k|v|out\.0)\.weight$")


def convert_vae(src: Path, out: Path, group_size: int) -> None:
    """Flux2 KL autoencoder, in diffusers naming — which is exactly what
    `flux.loadVae` reads, so nothing is renamed. Convs are transposed from
    torch OIHW into MLX's OHWI; `bn.running_*` rides along even though
    Ideogram's decode path uses its own published latent table instead."""
    state = dequantize(load_file(str(src / "vae" / "diffusion_pytorch_model.safetensors")))
    tensors: dict[str, mx.array] = {}
    n_conv = 0
    for k, v in state.items():
        if VAE_ATTN_RE.match(k):
            orig_module = k[: -len(".weight")]
            # diffusers spells the output projection `to_out.0` (element 0 of
            # a Sequential[Linear, Dropout]); `flux.loadVae` reads a plain
            # `to_out.weight` — strip the Sequential index so the emitted key
            # matches what the engine looks up.
            module = orig_module[:-2] if orig_module.endswith("to_out.0") else orig_module
            w = v
            if w.ndim == 4:  # some exports ship these as 1x1 Conv2d
                w = w.reshape(w.shape[0], w.shape[1])
            emit_linear(tensors, module, w, state.get(orig_module + ".bias"), 8, group_size)
            continue
        if k.endswith(".bias") and VAE_ATTN_RE.match(k[: -len(".bias")] + ".weight"):
            continue
        if v.ndim == 4:
            tensors[k] = to_mx(v.permute(0, 2, 3, 1).contiguous())
            n_conv += 1
        else:
            tensors[k] = to_mx(v)
    if "bn.running_mean" not in tensors:
        log("[warn] vae: no bn.running_mean — the Flux2 decoder's own entry point needs it")
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
        log(f"[chat] template not rendered ({e}); the engine will use its ChatML default")
        return "", ""
    if sentinel not in text:
        log("[chat] rendered template did not contain the prompt; falling back")
        return "", ""
    prefix, suffix = text.split(sentinel, 1)
    log(f"[chat] prefix={prefix!r} suffix={suffix!r}")
    return prefix, suffix


# ── self-test (no checkpoint needed) ────────────────────────────────────────


def self_test() -> None:
    """Unit-test the quantization policy and the size estimate. No torch
    tensors, no checkpoint — `mx.quantize` on a random probe array is the only
    GPU-touching call, to prove every affine width the policies can emit is
    one the engine's loader can solve `(bits, group_size)` back out of."""
    bulk = "blocks.0.attention.qkv"
    mlp = "blocks.0.feed_forward.w2"
    sensitive = "blocks.0.adaln_modulation"
    cond = "llm_cond_proj"
    dense = "final_layer.linear"

    # Flat precisions: one width everywhere, including the sensitive/dense
    # modules — "every projection at N-bit" means every one of them.
    for p_name, want in (("3", 3), ("4", 4), ("8", 8)):
        for m in (bulk, mlp, sensitive, cond, dense):
            assert bits_for(m, p_name, want) == want, (p_name, m)
        assert te_bits_for(p_name, want) == want
    print("[self-test] flat precisions OK")

    # Mixed policies: bulk / sensitive / dense-bf16, plus the text encoder.
    for p_name, b, sens, te in (("mixed", 4, 8, 8), ("mixed_3_8", 3, 8, 4)):
        assert bits_for(bulk, p_name, 8) == b, p_name
        assert bits_for(mlp, p_name, 8) == b, p_name
        assert bits_for(sensitive, p_name, 8) == sens, p_name
        assert bits_for(cond, p_name, 8) == sens, p_name
        assert bits_for(dense, p_name, 8) is None, p_name
        assert te_bits_for(p_name, 8) == te, p_name

    # A 2-bit bulk is refused BY NAME, not silently offered — the withdrawn
    # mixed_2_8 policy is gone; this is the only thing standing in for it now.
    try:
        apply_overrides("mixed_3_8", bulk=2, sensitive=None, te=None)
        raise AssertionError("--bulk-bits 2 was accepted")
    except ValueError as e:
        assert str(MIN_BULK_BITS) in str(e), e
    finally:
        MIXED_POLICIES["mixed_3_8"] = dict(BASE_POLICIES["mixed_3_8"])

    # Overrides reach every tier without inventing another named policy.
    try:
        apply_overrides("mixed_3_8", bulk=4, sensitive=6, te=8)
        assert bits_for(bulk, "mixed_3_8", 8) == 4
        assert bits_for(sensitive, "mixed_3_8", 8) == 6
        assert bits_for(cond, "mixed_3_8", 8) == 6
        assert te_bits_for("mixed_3_8", 8) == 8
    finally:
        MIXED_POLICIES["mixed_3_8"] = dict(BASE_POLICIES["mixed_3_8"])
    assert bits_for(bulk, "mixed_3_8", 8) == 3

    # A flat (non-mixed) precision has no tiers: overriding one is refused
    # rather than silently ignored, since a dropped flag ships a pack that
    # does not match what was asked for.
    try:
        apply_overrides("4", bulk=3, sensitive=None, te=None)
        raise AssertionError("--bulk-bits on a flat precision was accepted")
    except ValueError:
        pass
    print("[self-test] quantization policy OK")

    # Every width the policies emit must be one the engine can solve back out
    # of the packed geometry — the loader reads GEOMETRY, never the config
    # block (`flux.inferQuantGeometry`, `ideogram4.IgLinear.load`).
    widths = [2, 3, 4, 5, 6, 8]
    probe = mx.random.normal((128, 512)).astype(mx.bfloat16)
    for bits in widths:
        q, sc, _ = mx.quantize(probe, group_size=64, bits=bits)
        assert 32 * q.shape[1] % probe.shape[1] == 0
        assert 32 * q.shape[1] // probe.shape[1] == bits, (bits, q.shape)
        assert probe.shape[1] // sc.shape[1] == 64, (bits, sc.shape)
    print(f"[self-test] packed geometry solves back at {widths} OK")

    # A smaller policy must produce a smaller PACK.
    sizes = {p: estimate_pack_bytes(p) for p in ("8", "4", "3", "mixed", "mixed_3_8")}
    assert sizes["mixed"] < sizes["8"]
    assert sizes["mixed_3_8"] < sizes["4"]
    # NOT compared against sizes["3"]: mixed_3_8 keeps the sensitive tier at
    # 8-bit (a flat 3 does not), so it is a few percent LARGER than flat 3 in
    # exchange for protecting that tier — smaller-bulk does not mean smaller
    # pack once a policy holds something back.
    # The preset must fit a 24 GB Mac with 6 GB not guaranteed by the system —
    # the whole reason it exists — with real headroom left over for
    # activations and everything else running.
    assert sizes["mixed_3_8"] < 16e9, sizes["mixed_3_8"]
    print("[self-test] pack size ordering OK: " + ", ".join(f"{k}={v / 1e9:.1f}GB" for k, v in sizes.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="HF repo id or a local checkpoint dir")
    ap.add_argument("--out", help="destination pack directory")
    ap.add_argument("--precision", choices=("3", "4", "8", "mixed", "mixed_3_8"), default="mixed_3_8")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--bulk-bits", type=int, help="override a mixed policy's attention/MLP width")
    ap.add_argument("--sensitive-bits", type=int, help="override a mixed policy's modulation/conditioning width")
    ap.add_argument("--te-bits", type=int, help="override a mixed policy's text-encoder width")
    ap.add_argument("--self-test", action="store_true", help="run the policy/size unit tests and exit (no checkpoint needed)")
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
    try:
        apply_overrides(args.precision, args.bulk_bits, args.sensitive_bits, args.te_bits)
    except ValueError as e:
        ap.error(str(e))
    te_bits = te_bits_for(args.precision, default_bits)

    src = resolve_src(args.src)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    log(
        f"[plan] {src} → {out}  precision={args.precision} group_size={args.group_size} "
        f"text_encoder={te_bits}-bit  (estimated pack ≈ {estimate_pack_bytes(args.precision, args.group_size) / 1e9:.1f} GB)"
    )

    convert_text_encoder(src, out, te_bits, args.group_size)
    convert_vae(src, out, args.group_size)
    copy_tokenizer(src, out)
    convert_transformer(src, out, "transformer", args.precision, default_bits, args.group_size)
    convert_transformer(src, out, "unconditional_transformer", args.precision, default_bits, args.group_size)

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
    (out / "model_index.json").write_text(json.dumps({"_class_name": "Ideogram4Pipeline"}, indent=2))

    # LAST, deliberately: `model_discovery.requiredMediaMarker` treats this file
    # as the pack's completeness marker.
    (out / "unconditional_transformer" / "config.json").write_text(
        json.dumps({"_class_name": "Ideogram4Transformer"}, indent=2)
    )
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    log(f"[done] {out} — {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()

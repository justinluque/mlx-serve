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

  4       every projection at 4-bit, ~7.5 GB. Smallest.
  8       every projection at 8-bit, ~13 GB. The reference point.
  mixed   the bulk (attention + MLP, ~8.7B of the 9.3B) at 4-bit, the
          modulation and conditioning projections at 8-bit, and everything
          small enough not to matter left dense bf16. ~8.5 GB.

The mixed policy is not a guess about which tensors "matter" in general — it
is about which ones are cheap. `adaln_modulation` is 320M parameters whose
output is a per-layer SCALE and a tanh GATE, so its error is multiplicative
over the whole residual stream rather than averaged into one projection (the
same reason mlx-serve's MageFlow notes flag `img_mod`/`txt_mod` at 4-bit); the
embedding, timestep and final-layer projections together are under 30M, so
holding them at bf16 costs ~60 MB and removes them as suspects entirely.

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
                    {kk[len(k) + 1 :]: vv for kk, vv in state.items() if kk.startswith(k + ".")},
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


def bits_for(module: str, precision: str, default_bits: int) -> int | None:
    """Bit width for a module path, or None to keep it dense bf16."""
    if precision != "mixed":
        return default_bits
    if any(module.endswith(s) or module == s for s in DENSE_MODULES):
        return None
    if any(module.endswith(s) or module == s for s in SENSITIVE_MODULES):
        return 8
    if any(module.endswith(s) for s in BULK_SUFFIXES):
        return 4
    return 8


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


def convert_text_encoder(src: Path, out: Path, default_bits: int, group_size: int) -> None:
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
            emit_linear(tensors, module, w, None, default_bits, group_size)
        else:
            emit_linear(tensors, module, w, b, default_bits, group_size)
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
            module = k[: -len(".weight")]
            w = v
            if w.ndim == 4:  # some exports ship these as 1x1 Conv2d
                w = w.reshape(w.shape[0], w.shape[1])
            emit_linear(tensors, module, w, state.get(module + ".bias"), 8, group_size)
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="HF repo id or a local checkpoint dir")
    ap.add_argument("--out", required=True, help="destination pack directory")
    ap.add_argument("--precision", choices=("4", "8", "mixed"), default="mixed")
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()

    src = resolve_src(args.src)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    default_bits = 4 if args.precision == "4" else 8
    log(f"[plan] {src} → {out}  precision={args.precision} group_size={args.group_size}")

    convert_text_encoder(src, out, default_bits, args.group_size)
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

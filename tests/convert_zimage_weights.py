#!/usr/bin/env python3
"""Repackage a released Z-Image / Z-Image-Turbo (Tongyi-MAI) repo as an 8-bit
mlx-serve mirror.

USER-RUN (needs mlx + the ~30 GB HF checkpoint: 24 GB DiT + ~5 GB Qwen3 text
encoder + ~0.3 GB VAE). Not run in CI. Produces the diffusers-shaped dir
`src/z_image.zig` loads, same convention as `convert_mageflow_weights.py`:

  (a) `transformer/` + `text_encoder/` linears affine-quantized (default
      8-bit, group 64). `MfLinear` (shared with MageFlow) picks
      dense-vs-quantized PER TENSOR from the presence of a `.scales`
      sibling, so a partially quantized checkpoint (see --keep-bf16) loads
      with no flag anywhere.
  (b) The VAE stays bf16, UNQUANTIZED — it's small (~340 MB) and its
      precision is load-bearing (the engine runs it in f32).

    <out>/model_index.json                              verbatim
    <out>/scheduler/scheduler_config.json               verbatim
    <out>/transformer/config.json                        verbatim
    <out>/transformer/diffusion_pytorch_model.safetensors 8-bit (one file, no index)
    <out>/text_encoder/{config,generation_config}.json   verbatim
    <out>/text_encoder/model.safetensors                 8-bit (one file, no index)
    <out>/tokenizer/{tokenizer.json,tokenizer_config.json,vocab.json,merges.txt}  verbatim
    <out>/vae/config.json                                 verbatim
    <out>/vae/diffusion_pytorch_model.safetensors          bf16, UNQUANTIZED

WHAT IS NOT QUANTIZED, and why:

  - The whole VAE (see above).
  - `embed_tokens.weight` (Qwen3 TE) — read with `mlx_take_axis`, NOT a
    matmul, so a packed table would gather garbage rows.
  - `t_embedder.*`, `cap_embedder.1.weight`, `all_x_embedder.*`,
    `all_final_layer.*`, `x_pad_token`, `cap_pad_token` — tiny (a few MB
    combined) and precision-sensitive (diffusers marks
    `t_embedder`/`cap_embedder` `_skip_layerwise_casting_patterns`).
  - Anything with min(out_features, in_features) < 512, or whose input dim
    isn't a multiple of the group size, or that isn't rank 2.

The output dir name decides the sampling defaults `z_image.dirLooksTurbo`
resolves (8 steps / CFG 0 vs 50 steps / CFG 5) — keep "turbo" in the name
(case-insensitive) for a Z-Image-Turbo build.

Usage:
    python3 tests/convert_zimage_weights.py --src <repo dir> --out <dir>
    python3 tests/convert_zimage_weights.py --src <dir> --dry-run   # manifest only, no mlx
    python3 tests/convert_zimage_weights.py --self-test             # no ckpt/mlx needed

Source: Apache-2.0 upstream (Tongyi-MAI/Z-Image, Tongyi-MAI/Z-Image-Turbo).
"""

import argparse
import glob
import json
import os
import shutil
import struct
import sys

GROUP_SIZE = 64
DEFAULT_BITS = 8
MIN_DIM = 512

# Read with mlx_take_axis, never a matmul.
NEVER_QUANTIZE = ("embed_tokens.weight",)

# Tiny + precision-sensitive; excluded from quantization regardless of shape.
ALWAYS_DENSE_PREFIXES = (
    "t_embedder.",
    "cap_embedder.",
    "all_x_embedder.",
    "all_final_layer.",
)

# Everything the engine opens, per component dir (see src/z_image.zig,
# src/model_discovery.zig's peekZImageIndex, TextEncoder.load).
KEEP_FILES = {
    "": ["model_index.json"],
    "scheduler": ["scheduler_config.json"],
    "transformer": ["config.json"],
    "vae": ["config.json"],
    "text_encoder": ["config.json", "generation_config.json"],
    "tokenizer": ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
}
LICENSE_NAMES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "NOTICE")


def should_quantize(name, shape, bits, keep_bf16=()):
    if bits == 16:
        return False
    if not name.endswith(".weight"):
        return False
    if len(shape) != 2:
        return False
    if any(t in name for t in NEVER_QUANTIZE):
        return False
    if any(name.startswith(p) for p in ALWAYS_DENSE_PREFIXES):
        return False
    if any(t and t in name for t in keep_bf16):
        return False
    out_f, in_f = shape[0], shape[1]
    if in_f % GROUP_SIZE != 0:
        return False
    return min(out_f, in_f) >= MIN_DIM


def quantized_nbytes(shape, bits):
    out_f, in_f = shape
    return out_f * in_f * bits // 8 + 2 * 2 * out_f * (in_f // GROUP_SIZE)


# ── safetensors header reading (dry-run needs no mlx and no big memory) ──
def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def component_files(src, component):
    d = os.path.join(src, component)
    files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    if not files:
        raise SystemExit(f"[error] no .safetensors under {d}")
    return files


def plan_component(src, component, bits, keep_bf16):
    quantize_here = component in ("transformer", "text_encoder")
    rows = []
    for path in component_files(src, component):
        for name, meta in read_header(path).items():
            shape = meta["shape"]
            nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]
            if quantize_here and should_quantize(name, shape, bits, keep_bf16):
                rows.append((name, shape, nbytes, "quant", quantized_nbytes(shape, bits)))
            else:
                rows.append((name, shape, nbytes, "keep", nbytes))
    return rows


def print_manifest(component, rows):
    kinds = {}
    for _, _, src_b, kind, out_b in rows:
        n, s, o = kinds.get(kind, (0, 0, 0))
        kinds[kind] = (n + 1, s + src_b, o + out_b)
    src_total = sum(v[1] for v in kinds.values())
    out_total = sum(v[2] for v in kinds.values())
    print(f"  {component or '(root)'}:")
    for kind in ("quant", "keep"):
        if kind in kinds:
            n, s, o = kinds[kind]
            print(f"    {kind:<6} {n:5d} tensors  {s/1e9:7.3f} GB -> {o/1e9:7.3f} GB")
    print(f"    {'TOTAL':<6} {len(rows):5d} tensors  {src_total/1e9:7.3f} GB -> {out_total/1e9:7.3f} GB")
    return src_total, out_total


# ── conversion ──
def convert_component(src, out, component, bits, keep_bf16, out_name):
    import mlx.core as mx

    rows = plan_component(src, component, bits, keep_bf16)
    decision = {r[0]: r[3] for r in rows}
    packed = {}
    for path in component_files(src, component):
        loaded = mx.load(path)
        for name, arr in loaded.items():
            kind = decision[name]
            if kind == "quant":
                wq, scales, biases = mx.quantize(arr, group_size=GROUP_SIZE, bits=bits)
                base = name[: -len(".weight")]
                packed[f"{base}.weight"] = wq
                packed[f"{base}.scales"] = scales
                packed[f"{base}.biases"] = biases
            else:
                packed[name] = arr
    mx.eval(*packed.values())
    dst_dir = os.path.join(out, component)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, out_name)
    mx.save_safetensors(dst, packed)
    written = os.path.getsize(dst)
    print(f"  wrote {dst} ({written/1e9:.3f} GB, {len(packed)} tensors)")
    return written


def copy_support_files(src, out, bits):
    """Copy the small config/tokenizer files verbatim, EXCEPT
    `text_encoder/config.json`: `transformer.zig`'s shared Qwen3 loader
    decides dense-vs-quantized from `config.json`'s own `quantization`
    block, not from whether `.scales` tensors are present in the
    safetensors file. Serving a quantized `model.safetensors` behind an
    unmodified upstream config.json makes the loader treat every packed
    `.weight` as a dense matrix of the wrong shape (segfault or a shape-
    mismatch matmul error)."""
    for junk in glob.glob(os.path.join(out, "**", ".DS_Store"), recursive=True):
        os.remove(junk)

    for component, names in KEEP_FILES.items():
        dst_dir = os.path.join(out, component) if component else out
        os.makedirs(dst_dir, exist_ok=True)
        for name in names:
            s = os.path.join(src, component, name) if component else os.path.join(src, name)
            if not os.path.exists(s):
                if name == "generation_config.json":
                    print(f"  [skip] {name} absent in source")
                    continue
                raise SystemExit(f"[error] required file missing from source: {s}")
            dst = os.path.join(dst_dir, name)
            if component == "text_encoder" and name == "config.json":
                with open(s) as f:
                    cfg = json.load(f)
                cfg["quantization"] = {"bits": bits, "group_size": GROUP_SIZE, "mode": "affine"}
                with open(dst, "w") as f:
                    json.dump(cfg, f, indent=2)
            else:
                shutil.copy2(s, dst)
    for name in LICENSE_NAMES:
        s = os.path.join(src, name)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(out, name))


def run_self_test():
    """Pure-Python checks — no mlx, no checkpoint. Exercises the tensor
    classification rules that decide the converter's output shape."""
    cases = [
        ("layers.0.attention.to_q.weight", (3840, 3840), True),
        ("layers.0.feed_forward.w2.weight", (3840, 10240), True),
        ("noise_refiner.0.adaLN_modulation.0.weight", (15360, 256), False),  # in=256 < MIN_DIM
        ("all_x_embedder.2-1.weight", (3840, 64), False),  # always-dense prefix
        ("all_final_layer.2-1.linear.weight", (64, 3840), False),  # always-dense prefix
        ("cap_embedder.1.weight", (3840, 2560), False),  # always-dense prefix
        ("t_embedder.mlp.0.weight", (1024, 256), False),  # always-dense prefix
        ("embed_tokens.weight", (151936, 2560), False),  # NEVER_QUANTIZE (gather table)
        ("x_pad_token", (1, 3840), False),  # not a `.weight`
        ("layers.0.attention.norm_q.weight", (128,), False),  # rank 1
    ]
    failed = 0
    for name, shape, expected in cases:
        got = should_quantize(name, shape, DEFAULT_BITS)
        status = "ok" if got == expected else "FAIL"
        if got != expected:
            failed += 1
        print(f"  [{status}] should_quantize({name!r}, {shape}) = {got} (expected {expected})")
    if failed:
        print(f"\n{failed} self-test case(s) failed")
        sys.exit(1)

    # Class guard: `text_encoder/config.json` MUST carry a `quantization`
    # block matching the bits this run quantized the text encoder at.
    # `transformer.zig`'s shared Qwen3 loader decides dense-vs-quantized
    # from THIS field (`config.quant_bits`), not from whether `.scales`
    # tensors happen to exist in the safetensors file — a blind file copy
    # here shipped a checkpoint that crashed (SIGSEGV, later a shape-
    # mismatch matmul) the moment the text encoder actually ran, because
    # every packed `.weight` was read as a dense matrix of the wrong shape.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        src, out = os.path.join(tmp, "src"), os.path.join(tmp, "out")
        os.makedirs(os.path.join(src, "text_encoder"))
        for component, names in KEEP_FILES.items():
            if component == "text_encoder":
                continue
            d = os.path.join(src, component) if component else src
            os.makedirs(d, exist_ok=True)
            for name in names:
                with open(os.path.join(d, name), "w") as f:
                    f.write("{}")
        with open(os.path.join(src, "text_encoder", "config.json"), "w") as f:
            json.dump({"hidden_size": 2560, "num_hidden_layers": 36}, f)
        copy_support_files(src, out, 6)
        with open(os.path.join(out, "text_encoder", "config.json")) as f:
            written = json.load(f)
        q = written.get("quantization")
        ok = q == {"bits": 6, "group_size": GROUP_SIZE, "mode": "affine"}
        print(f"  [{'ok' if ok else 'FAIL'}] copy_support_files writes text_encoder quantization block: {q}")
        if not ok:
            print("\ncopy_support_files quantization-block regression: FAILED")
            sys.exit(1)
        # The source field the loader needs must survive too.
        if written.get("hidden_size") != 2560:
            print("\ncopy_support_files dropped an existing config field: FAILED")
            sys.exit(1)

    print("\nself-test: all cases passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="source repo dir (diffusers layout)")
    ap.add_argument("--out", help="output dir")
    ap.add_argument("--bits", type=int, default=DEFAULT_BITS, choices=(2, 3, 4, 5, 6, 8))
    ap.add_argument("--keep-bf16", nargs="*", default=(), help="substrings of tensor names to leave dense")
    ap.add_argument("--dry-run", action="store_true", help="print the manifest, write nothing, no mlx import")
    ap.add_argument("--self-test", action="store_true", help="run classification unit checks, no ckpt/mlx needed")
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not args.src:
        ap.error("--src is required (unless --self-test)")

    print(f"Z-Image converter: {args.src} -> bits={args.bits}")
    total_src = 0
    total_out = 0
    for component in ("transformer", "text_encoder", "vae"):
        rows = plan_component(args.src, component, args.bits, args.keep_bf16)
        s, o = print_manifest(component, rows)
        total_src += s
        total_out += o
    print(f"\n  {'GRAND TOTAL':<13} {total_src/1e9:7.3f} GB -> {total_out/1e9:7.3f} GB")

    if args.dry_run:
        return
    if not args.out:
        ap.error("--out is required unless --dry-run")

    for component, out_name in (
        ("transformer", "diffusion_pytorch_model.safetensors"),
        ("text_encoder", "model.safetensors"),
        ("vae", "diffusion_pytorch_model.safetensors"),
    ):
        bits = 16 if component == "vae" else args.bits
        convert_component(args.src, args.out, component, bits, args.keep_bf16, out_name)

    copy_support_files(args.src, args.out, args.bits)
    print(f"\nDone -> {args.out}")


if __name__ == "__main__":
    main()

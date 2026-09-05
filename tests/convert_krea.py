#!/usr/bin/env python3
"""Build a mixed-N/8-bit MLX pack for Krea-2-Turbo straight from the gated
`krea/Krea-2-Turbo` bf16 release on HF -- no local source pack needed.

Recipe (mirrors `ddalcu/Krea-2-Turbo-MLX-Serve-mixed-4-8`, just at a narrower
low-bit width for the "rest" tier -- verified against that pack's own tensor
geometry, not just its README note):

  - `mlp.down` in EVERY one of the 28 DiT blocks: always 8-bit/g64 (the
    sensitivity-selected leaf, regardless of block position).
  - `attn.{wq,wk,wv,wo,gate}` in the first/last 2 blocks (0, 1, 26, 27):
    8-bit/g64.
  - Everything else -- `attn.*` in the interior 24 blocks, plus
    `mlp.{gate,up}` in ALL 28 blocks -- goes to `--low-bits` (default 3).
  - `first`/`last`/`tmlp`/`tproj`/`txtfusion.*`/`txtmlp` (the reference
    recipe's sensitive set) and every qknorm/mod/prenorm/postnorm table stay
    dense, byte-for-byte -- these are never in the block-linear leaf set above.

The text encoder (Qwen3-VL-4B) is quantized uniformly at 8-bit/g64 across
every 2-D attention/MLP projection (embed_tokens and norms stay dense), minus
its `visual.*` vision tower -- `krea.loadConditioner` never reads a `visual.`
key, so shipping it is pure waste.

The VAE and tokenizer are copied verbatim: both already ship under the exact
key names and directory layout `src/krea.zig` reads (`decoder.*`/`encoder.*`,
`vae/`, `tokenizer/`), so nothing is renamed anywhere in this script -- the
upstream release IS our on-disk format for these parts.

The output transformer is named `transformer_mixed_<low>_8.safetensors`, which
is a CONVENTION and not a contract: the engine loads every root `*.safetensors`
whatever it is called, and `MixedLinear` solves (bits, group_size) from tensor
geometry. The app's ready marker matches the pattern for the same reason (see
`MediaBundle.krea`) -- it pinned this exact filename once, and a mixed_3_8 pack
served fine while the Create pane offered Download forever.

Streams one source file in at a time via `hf download` and deletes it right
after it's folded into the output pack: the combined source (~36 GB: 26.3 GB
transformer + 8.9 GB text encoder + 0.5 GB VAE) does not comfortably coexist
on disk with the output pack. Pass --keep-src to disable that cleanup.

CPU-only by construction (`mx.set_default_device(mx.cpu)`) so this can run
beside anything already using the GPU.

Usage:
  python3 tests/convert_krea.py --out ~/.mlx-serve/models/<org>/<name> --low-bits 3
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time

import mlx.core as mx

mx.set_default_device(mx.cpu)

REPO = "krea/Krea-2-Turbo"
NUM_BLOCKS = 28
EDGE_BLOCKS = {0, 1, NUM_BLOCKS - 2, NUM_BLOCKS - 1}
ATTN_LEAVES = ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate")
MLP_LOW = ("mlp.gate", "mlp.up")
MLP_HIGH = ("mlp.down",)
BLOCK_LEAVES = ATTN_LEAVES + MLP_LOW + MLP_HIGH
GROUP_SIZE = 64
TE_BITS = 8

TOKENIZER_FILES = ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json")


def leaf_bits(block, leaf, low_bits):
    if leaf in MLP_HIGH:
        return 8
    if leaf in ATTN_LEAVES and block in EDGE_BLOCKS:
        return 8
    return low_bits  # attn.* in interior blocks; mlp.{gate,up} everywhere


def header_metadata(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)).get("__metadata__")


def fetch(repo_file, tmp_dir):
    print(f"  downloading {repo_file} ...", flush=True)
    t0 = time.time()
    subprocess.run(
        ["hf", "download", REPO, repo_file, "--local-dir", tmp_dir],
        check=True,
    )
    path = os.path.join(tmp_dir, repo_file)
    print(f"    -> {os.path.getsize(path)/1e9:.2f} GB in {time.time()-t0:.0f}s", flush=True)
    return path


def quantize_transformer(src, dst, low_bits):
    weights = mx.load(src)
    meta = header_metadata(src) or {}
    out, n_q = {}, 0
    for key in sorted(weights):
        if not key.startswith("blocks.") or not key.endswith(".weight"):
            out[key] = weights[key]
            continue
        block_s, rest = key[len("blocks."):-len(".weight")].split(".", 1)
        block = int(block_s)
        if rest not in BLOCK_LEAVES:
            out[key] = weights[key]  # qknorm/mod/prenorm/postnorm
            continue
        bits = leaf_bits(block, rest, low_bits)
        w = weights[key]
        wq, sc, bi = mx.quantize(w, group_size=GROUP_SIZE, bits=bits)
        mx.eval(wq, sc, bi)
        base = key[: -len(".weight")]
        out[base + ".weight"] = wq
        out[base + ".scales"] = sc
        out[base + ".biases"] = bi
        n_q += 1
        if n_q % 64 == 0:
            mx.clear_cache()
            print(f"    transformer: {n_q} weights quantized", flush=True)
    del weights
    meta["note"] = (
        f"Krea-2-Turbo transformer, mixed {low_bits}/8-bit MLX (down_proj + "
        f"first/last 2 blocks' attn @8-bit, rest of 28-block attn+mlp "
        f"@{low_bits}-bit, g{GROUP_SIZE}; other paths bf16)."
    )
    meta["recipe"] = f"mixed_{low_bits}_8"
    tmp = dst + ".partial.safetensors"
    mx.save_safetensors(tmp, out, metadata=meta)
    os.replace(tmp, dst)
    del out
    mx.clear_cache()
    print(
        f"  transformer: {n_q} weights quantized -> {os.path.getsize(dst)/1e9:.2f} GB",
        flush=True,
    )


def quantize_text_encoder(src, dst):
    weights = mx.load(src)
    out, n_q, n_dropped = {}, 0, 0
    for key in sorted(weights):
        if key.startswith("visual."):
            n_dropped += 1
            continue
        w = weights[key]
        if not key.endswith(".weight") or w.ndim != 2:
            out[key] = w
            continue
        leaf = key[: -len(".weight")].rsplit(".", 1)[-1]
        if leaf == "embed_tokens" or leaf.endswith("norm") or w.shape[-1] % GROUP_SIZE:
            out[key] = w
            continue
        wq, sc, bi = mx.quantize(w, group_size=GROUP_SIZE, bits=TE_BITS)
        mx.eval(wq, sc, bi)
        base = key[: -len(".weight")]
        out[base + ".weight"] = wq
        out[base + ".scales"] = sc
        out[base + ".biases"] = bi
        n_q += 1
    del weights
    tmp = dst + ".partial.safetensors"
    mx.save_safetensors(tmp, out)
    os.replace(tmp, dst)
    del out
    mx.clear_cache()
    print(
        f"  text encoder: {n_q} weights @{TE_BITS}-bit -> {os.path.getsize(dst)/1e9:.2f} GB "
        f"(dropped {n_dropped} visual.* tensors)",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default=os.path.join(os.path.dirname(__file__), "..", ".krea-src-tmp"))
    ap.add_argument("--low-bits", type=int, default=3, choices=(2, 3, 4, 5, 6))
    ap.add_argument("--keep-src", action="store_true")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    tmp = os.path.abspath(args.tmp)
    os.makedirs(out, exist_ok=True)
    os.makedirs(tmp, exist_ok=True)

    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({"model_type": "krea2_turbo"}, f)

    # ── transformer ──
    dst = os.path.join(out, f"transformer_mixed_{args.low_bits}_8.safetensors")
    if os.path.exists(dst):
        print(f"  transformer: {dst} exists, skipping", flush=True)
    else:
        src = fetch("turbo.safetensors", tmp)
        quantize_transformer(src, dst, args.low_bits)
        if not args.keep_src:
            os.remove(src)

    # ── text encoder ──
    te_dst_dir = os.path.join(out, "text_encoder")
    os.makedirs(te_dst_dir, exist_ok=True)
    dst = os.path.join(te_dst_dir, "model.safetensors")
    if os.path.exists(dst):
        print(f"  text encoder: {dst} exists, skipping", flush=True)
    else:
        src = fetch("text_encoder/model.safetensors", tmp)
        quantize_text_encoder(src, dst)
        if not args.keep_src:
            os.remove(src)
    cfg_src = fetch("text_encoder/config.json", tmp)
    shutil.copyfile(cfg_src, os.path.join(te_dst_dir, "config.json"))

    # ── VAE (verbatim -- already our key layout) ──
    vae_dst_dir = os.path.join(out, "vae")
    os.makedirs(vae_dst_dir, exist_ok=True)
    for fn in ("config.json", "diffusion_pytorch_model.safetensors"):
        dst = os.path.join(vae_dst_dir, fn)
        if os.path.exists(dst):
            continue
        src = fetch(f"vae/{fn}", tmp)
        shutil.copyfile(src, dst)
        if not args.keep_src and fn.endswith(".safetensors"):
            os.remove(src)

    # ── tokenizer (verbatim) ──
    tok_dst_dir = os.path.join(out, "tokenizer")
    os.makedirs(tok_dst_dir, exist_ok=True)
    for fn in TOKENIZER_FILES:
        dst = os.path.join(tok_dst_dir, fn)
        if os.path.exists(dst):
            continue
        src = fetch(f"tokenizer/{fn}", tmp)
        shutil.copyfile(src, dst)

    if not args.keep_src:
        shutil.rmtree(tmp, ignore_errors=True)

    total = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, fs in os.walk(out)
        for f in fs
    )
    print(f"\n  TOTAL {total/1e9:.2f} GB -> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

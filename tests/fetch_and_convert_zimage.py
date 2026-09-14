#!/usr/bin/env python3
"""Download a Z-Image / Z-Image-Turbo repo from HF and convert it to one or
more quantized mlx-serve mirrors in a SINGLE streaming pass.

Why this exists instead of just `hf download` + `convert_zimage_weights.py`:
the released checkpoints ship as fp32 (not bf16 — `transformer/config.json`'s
"6B" claim checks out: 24.6GB / 4 bytes per param), so a full raw download is
~33GB. Producing an 8-bit AND a 4-bit mirror the naive way (download once,
run the converter twice) needs raw + both outputs on disk at once
(33 + 11 + 6.3 ~= 50GB) — more than fits on a disk with less headroom than
that. This script downloads ONE shard at a time, quantizes it into EVERY
requested bit width immediately, deletes the shard, and only ever holds one
raw shard (~8GB) plus the accumulating output arrays (in RAM, not disk) at
once. Peak disk stays around one shard + the final output files.

USER-RUN (needs mlx + huggingface_hub + network + ~25GB free disk for the
default --bits 8,4 pair). Not run in CI.

Usage:
    python3 tests/fetch_and_convert_zimage.py \\
        --repo Tongyi-MAI/Z-Image-Turbo \\
        --out-base ~/.mlx-serve/models/Tongyi-MAI \\
        --bits 8,4
"""

import argparse
import json
import os
import shutil

GROUP_SIZE = 64
MIN_DIM = 512
NEVER_QUANTIZE = ("embed_tokens.weight",)
ALWAYS_DENSE_PREFIXES = ("t_embedder.", "cap_embedder.", "all_x_embedder.", "all_final_layer.")


def should_quantize(name, shape, bits):
    if not name.endswith(".weight") or len(shape) != 2:
        return False
    if any(t in name for t in NEVER_QUANTIZE):
        return False
    if any(name.startswith(p) for p in ALWAYS_DENSE_PREFIXES):
        return False
    out_f, in_f = shape[0], shape[1]
    if in_f % GROUP_SIZE != 0:
        return False
    return min(out_f, in_f) >= MIN_DIM


def out_dir_for(out_base, repo, bits):
    name = os.path.basename(repo.rstrip("/"))
    return os.path.join(out_base, f"{name}-MLX-Serve-{bits}bit")


def fetch(hub, repo, remote_path, workdir):
    """Download one file into a cache dir we fully control (NOT the default
    ~/.cache/huggingface, which would grow permanently) and return the
    SYMLINK path `hf_hub_download` gives back — its filename carries the
    real extension (`...safetensors`, `...json`), which `mx.load` dispatches
    on; the blob it points at is a bare content hash with no extension and
    `mx.load` refuses it as "Unknown file format" even though the bytes are
    fine. Pass THIS path to readers; pass it to `forget()` to delete."""
    cache_dir = os.path.join(workdir, ".hf-cache")
    return hub.hf_hub_download(repo, remote_path, cache_dir=cache_dir)


def forget(path):
    """Delete what `fetch` returned: the resolved blob (so disk is actually
    reclaimed, not just the symlink) plus the symlink itself."""
    real = os.path.realpath(path)
    for p in (real, path):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def process_component(hub, repo, component, index_filename, bits_list, out_dirs, workdir):
    """Stream one component's shards: download -> quantize into every bit
    width -> delete the shard -> next. Saves the accumulated safetensors for
    every bit width at the end."""
    import mlx.core as mx

    print(f"[{component}] fetching shard index...")
    index_path = fetch(hub, repo, f"{component}/{index_filename}", workdir)
    with open(index_path) as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))
    print(f"[{component}] {len(shard_files)} shard(s), {len(index['weight_map'])} tensors")

    packed = {b: {} for b in bits_list}
    for i, shard in enumerate(shard_files, 1):
        print(f"[{component}] downloading shard {i}/{len(shard_files)}: {shard}")
        shard_path = fetch(hub, repo, f"{component}/{shard}", workdir)
        loaded = mx.load(shard_path)
        for name, arr in loaded.items():
            arr_bf16 = None
            for bits in bits_list:
                if should_quantize(name, list(arr.shape), bits):
                    wq, scales, biases = mx.quantize(arr, group_size=GROUP_SIZE, bits=bits)
                    base = name[: -len(".weight")]
                    packed[bits][f"{base}.weight"] = wq
                    packed[bits][f"{base}.scales"] = scales
                    packed[bits][f"{base}.biases"] = biases
                else:
                    if arr_bf16 is None:
                        arr_bf16 = arr.astype(mx.bfloat16)
                    packed[bits][name] = arr_bf16
        # Materialize this shard's contribution now and drop the fp32 source
        # before the next shard downloads, so only ~one shard's worth of raw
        # data is ever resident on disk (and in RAM) at a time.
        mx.eval(*[t for d in packed.values() for t in d.values()])
        del loaded
        forget(shard_path)
        print(f"[{component}] shard {i}/{len(shard_files)} processed, raw deleted")

    for bits, out_dir in zip(bits_list, out_dirs):
        dst_dir = os.path.join(out_dir, component)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, "diffusion_pytorch_model.safetensors" if component != "text_encoder" else "model.safetensors")
        mx.save_safetensors(dst, packed[bits])
        written = os.path.getsize(dst)
        print(f"[{component}] bits={bits}: wrote {dst} ({written/1e9:.3f} GB, {len(packed[bits])} tensors)")
    forget(index_path)


def process_vae(hub, repo, out_dirs, workdir):
    """The VAE is small (~170MB) and stays dense fp32 (load-bearing
    precision) — just download once and copy into every output dir."""
    print("[vae] fetching diffusion_pytorch_model.safetensors...")
    path = fetch(hub, repo, "vae/diffusion_pytorch_model.safetensors", workdir)
    for out_dir in out_dirs:
        dst_dir = os.path.join(out_dir, "vae")
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(path, os.path.join(dst_dir, "diffusion_pytorch_model.safetensors"))
    forget(path)
    print(f"[vae] copied to {len(out_dirs)} output dir(s)")


def copy_support_files(hub, repo, out_dirs, bits_list, workdir):
    """Copy the small config/tokenizer files verbatim, EXCEPT
    `text_encoder/config.json`: `transformer.zig`'s shared Qwen3 loader
    decides dense-vs-quantized from `config.json`'s own `quantization`
    block (`config.quant_bits`), not from whether `.scales` tensors happen
    to be present in the safetensors file — unlike `MfLinear` (the DiT's
    loader), which infers quantization from packed geometry alone. Serving
    an 8-bit `model.safetensors` behind an UNMODIFIED (unquantized) upstream
    config.json makes the loader treat every packed `.weight` as a dense
    matrix of the wrong shape, and `mlx_matmul` segfaults or throws a shape
    mismatch. So this writes a per-bit-width `quantization` block into each
    output's own copy."""
    files = {
        "": ["model_index.json"],
        "scheduler": ["scheduler_config.json"],
        "transformer": ["config.json"],
        "vae": ["config.json"],
        "text_encoder": ["config.json", "generation_config.json"],
        "tokenizer": ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
    }
    for component, names in files.items():
        for name in names:
            remote = f"{component}/{name}" if component else name
            try:
                path = fetch(hub, repo, remote, workdir)
            except Exception as e:  # noqa: BLE001 — generation_config.json is optional upstream
                if name == "generation_config.json":
                    print(f"  [skip] {remote} absent in source")
                    continue
                raise SystemExit(f"[error] required file missing: {remote} ({e})")
            for bits, out_dir in zip(bits_list, out_dirs):
                dst_dir = os.path.join(out_dir, component) if component else out_dir
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, name)
                if component == "text_encoder" and name == "config.json":
                    with open(path) as f:
                        cfg = json.load(f)
                    cfg["quantization"] = {"bits": bits, "group_size": GROUP_SIZE, "mode": "affine"}
                    with open(dst, "w") as f:
                        json.dump(cfg, f, indent=2)
                else:
                    shutil.copy2(path, dst)
    print(f"copied support files to {len(out_dirs)} output dir(s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. Tongyi-MAI/Z-Image-Turbo")
    ap.add_argument("--out-base", required=True, help="parent dir; each bit width gets its own <repo>-MLX-Serve-<bits>bit subdir")
    ap.add_argument("--bits", default="8,4", help="comma-separated bit widths to produce in one pass (default: 8,4)")
    ap.add_argument("--workdir", default=None, help="scratch dir for shard downloads (default: <out-base>/.fetch-tmp)")
    args = ap.parse_args()

    bits_list = [int(b) for b in args.bits.split(",")]
    out_base = os.path.expanduser(args.out_base)
    out_dirs = [out_dir_for(out_base, args.repo, b) for b in bits_list]
    workdir = args.workdir or os.path.join(out_base, ".fetch-tmp")
    os.makedirs(workdir, exist_ok=True)

    import huggingface_hub as hub

    print(f"Z-Image streaming fetch+convert: {args.repo} -> bits={bits_list}")
    for d in out_dirs:
        print(f"  output: {d}")

    process_component(hub, args.repo, "transformer", "diffusion_pytorch_model.safetensors.index.json", bits_list, out_dirs, workdir)
    process_component(hub, args.repo, "text_encoder", "model.safetensors.index.json", bits_list, out_dirs, workdir)
    process_vae(hub, args.repo, out_dirs, workdir)
    copy_support_files(hub, args.repo, out_dirs, bits_list, workdir)

    shutil.rmtree(workdir, ignore_errors=True)
    print("\nDone:")
    for d in out_dirs:
        print(f"  {d}")


if __name__ == "__main__":
    main()

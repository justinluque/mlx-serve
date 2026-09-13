#!/usr/bin/env python3
"""FLUX.2 klein (Black Forest Labs' diffusers release) -> an mlx-serve pack,
imatrix-calibrated and budget-allocated.

    python3 tests/convert_flux2.py --src black-forest-labs/FLUX.2-klein-4B \\
        --out ~/.mlx-serve/models/<org>/FLUX.2-klein-4B-MLX-Serve-iq \\
        --imatrix ~/claude-tmp/klein4b-imatrix.safetensors --dit-bpw 4.0 --te-bpw 4.5

One script for klein 4B, 9B and base-9B: `flux.zig` reads every geometry off the
tensors. `--src` is a local directory or a repo id already in the Hugging Face
cache. The output uses the key layout of the mflux packs `flux.zig` was written
against, which differs from diffusers in two spellings
(`time_guidance_embed.timestep_embedder.linear_N` and `attn.to_out.0`).

- transformer: every linear at the width `image_quant.allocate` picks inside
  `--dit-bpw` — including the embedders and modulation projections the mflux
  packs flatten to 4-bit, which the allocator keeps wide when they are worth it.
  Staged as `transformer.partial/` and renamed last, because `flux.zig`'s
  discovery recognises a pack by its transformer alone.
- text_encoder: Qwen3 through layer 26. The DiT reads hidden states 9/18/27
  (the output of layer 26 is the last), so layers 27-35 and `lm_head` are not
  written. `--te-bpw` allocates it, token table included.
- vae: convs OIHW -> OHWI, `to_out.0` -> `to_out`, everything bf16 — 0.17 GB,
  and the decoder writes the pixels, so nothing in it is quantized.
- tokenizer: copied verbatim.

`--imatrix` is the file the ENGINE writes while generating with
`MLX_SERVE_IMATRIX=<path>` (tests/collect_image_imatrix.sh). `config.json` is
written last.

    python3 tests/convert_flux2.py --self-test
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import image_quant as iq  # noqa: E402

# flux.TE_TAPS: the last tap is the output of layer 26.
TE_LAYERS_RUN = 27


def dit_name(key: str) -> str | None:
    return key.replace("time_guidance_embed.timestep_embedder.", "time_guidance_embed.").replace(
        ".attn.to_out.0.", ".attn.to_out.")


def te_name(key: str) -> str | None:
    if key.startswith("lm_head."):
        return None
    key = key.removeprefix("model.")
    m = re.match(r"layers\.(\d+)\.", key)
    if m and int(m.group(1)) >= TE_LAYERS_RUN:
        return None
    return key


def vae_name(key: str) -> str | None:
    if key.endswith("num_batches_tracked"):
        return None
    return key.replace(".to_out.0.", ".to_out.")


def convert_vae(src: Path, out: Path) -> None:
    source = iq.Source.dir(src)
    writer = iq.ShardWriter(out, "model")
    for key, t in sorted(source.tensors.items()):
        name = vae_name(key)
        if name is None:
            continue
        if t.ndim == 4:
            t = t.transpose(0, 2, 3, 1)
        writer.add(name, t.astype(mx.bfloat16) if mx.issubdtype(t.dtype, mx.floating) else t)
    writer.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="black-forest-labs/FLUX.2-klein-{4B,9B,base-9B} or a local copy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imatrix", help="engine-collected activation statistics (MLX_SERVE_IMATRIX)")
    ap.add_argument("--dit-bpw", type=float, required=True, help="transformer budget, bits per weight incl. scales")
    ap.add_argument("--te-bpw", type=float, required=True, help="text-encoder budget, bits per weight incl. scales")
    ap.add_argument("--work", default="~/claude-tmp/image-quant", help="error tables + allocation reports")
    args = ap.parse_args()

    src = iq.resolve_src(args.src)
    for sub in ("transformer", "text_encoder", "vae", "tokenizer"):
        if not (src / sub).is_dir():
            sys.exit(f"{src}: no {sub}/ — expected the diffusers layout of a FLUX.2 klein release")
    out = Path(args.out).expanduser()
    if (out / "config.json").exists() or (out / "transformer").exists():
        sys.exit(f"{out}: already a pack")
    work = Path(args.work).expanduser() / out.name
    work.mkdir(parents=True, exist_ok=True)

    te = iq.convert("text_encoder", iq.Source.dir(src / "text_encoder"), te_name,
                    iq.ShardWriter(out / "text_encoder", "model"),
                    iq.Imatrix(args.imatrix, "text_encoder"), args.te_bpw, work)
    convert_vae(src / "vae", out / "vae")
    shutil.copytree(src / "tokenizer", out / "tokenizer", dirs_exist_ok=True)

    dit_source = iq.Source.dir(src / "transformer")
    inner = dit_source.tensors["x_embedder.weight"].shape[0]
    staging = out / "transformer.partial"
    dit = iq.convert("transformer", dit_source, dit_name,
                     iq.ShardWriter(staging, "model", index="model.safetensors.index.json"),
                     iq.Imatrix(args.imatrix, "transformer"), args.dit_bpw, work)
    staging.rename(out / "transformer")

    size = {3072: "4b", 4096: "9b"}.get(inner, f"inner{inner}")
    config = {
        "model_type": f"flux2-klein-{size}",
        "original_model": args.src,
        "quantization": {"method": "imatrix-weighted affine, measured bit allocation", "group_size": iq.GROUP_SIZE,
                         "transformer": dit, "text_encoder": te},
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    iq.log(f"\n{out}: {total / 1e9:.2f} GB (transformer {dit['gigabytes']} GB at {dit['bits_per_weight']} bpw, "
           f"text encoder {te['gigabytes']} GB at {te['bits_per_weight']} bpw)")
    return 0


def self_test() -> int:
    failures = []

    def check(cond, label):
        print(("PASS " if cond else "FAIL ") + label)
        if not cond:
            failures.append(label)

    # Pairs read off black-forest-labs/FLUX.2-klein-4B and the mflux 4B pack flux.zig loads.
    for bfl, pack in (
        ("time_guidance_embed.timestep_embedder.linear_1.weight", "time_guidance_embed.linear_1.weight"),
        ("time_guidance_embed.timestep_embedder.linear_2.weight", "time_guidance_embed.linear_2.weight"),
        ("transformer_blocks.3.attn.to_out.0.weight", "transformer_blocks.3.attn.to_out.weight"),
        ("transformer_blocks.3.attn.to_add_out.weight", "transformer_blocks.3.attn.to_add_out.weight"),
        ("single_transformer_blocks.7.attn.to_out.weight", "single_transformer_blocks.7.attn.to_out.weight"),
        ("single_transformer_blocks.7.attn.to_qkv_mlp_proj.weight", "single_transformer_blocks.7.attn.to_qkv_mlp_proj.weight"),
        ("double_stream_modulation_img.linear.weight", "double_stream_modulation_img.linear.weight"),
        ("x_embedder.weight", "x_embedder.weight"),
    ):
        check(dit_name(bfl) == pack, f"transformer {bfl} -> {pack}")
    check(te_name("model.layers.26.mlp.down_proj.weight") == "layers.26.mlp.down_proj.weight", "the last tapped encoder layer is kept")
    check(te_name("model.layers.27.self_attn.q_proj.weight") is None, "encoder layers past the last tap are dropped")
    check(te_name("lm_head.weight") is None, "lm_head is dropped")
    check(te_name("model.embed_tokens.weight") == "embed_tokens.weight", "the token table loses the model. prefix")
    check(te_name("model.norm.weight") == "norm.weight", "the final norm the loader expects is kept")
    check(vae_name("decoder.mid_block.attentions.0.to_out.0.weight") == "decoder.mid_block.attentions.0.to_out.weight",
          "vae to_out.0 -> to_out")
    check(vae_name("bn.num_batches_tracked") is None, "vae batch-norm counter is dropped")
    conv = mx.zeros((512, 32, 3, 3)).transpose(0, 2, 3, 1)
    check(tuple(conv.shape) == (512, 3, 3, 32), "vae conv OIHW -> OHWI matches the mflux pack's decoder.conv_in")
    print(f"\n{'OK' if not failures else 'FAILED'}: {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(self_test() if "--self-test" in sys.argv else main())

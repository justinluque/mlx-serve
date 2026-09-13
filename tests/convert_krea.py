#!/usr/bin/env python3
"""Krea-2-Turbo -> an mlx-serve pack, imatrix-calibrated and budget-allocated.

    python3 tests/convert_krea.py --src krea/Krea-2-Turbo \\
        --out ~/.mlx-serve/models/<org>/Krea-2-Turbo-MLX-Serve-iq \\
        --imatrix ~/claude-tmp/krea-imatrix.safetensors --dit-bpw 4.0 --te-bpw 4.5

`--src` is the upstream release — a local directory, or a repo id already in the
Hugging Face cache — holding `turbo.safetensors`, `text_encoder/`, `vae/` and
`tokenizer/`. Tensors are read lazily, so the 26 GB transformer never has to be
resident; the output streams into 2 GB shards.

- transformer: every linear `krea.MixedLinear` loads (the 28 blocks, text fusion,
  the timestep and text MLPs, `first`, `last.linear`) at the width
  `image_quant.allocate` picks inside `--dit-bpw`. Norms, biases and modulation
  tables are stored bf16. The ddalcu mixed-4-8 pack spends ~5.6 bits per weight
  on hand-picked tiers with plain round-to-nearest; this spends its budget where
  the measured, activation-weighted error is.
- text_encoder: the Qwen3-VL-4B language stack through layer 34. The conditioner
  taps the INPUT of layer 35, so layer 35, the final norm and the vision tower
  never reach the DiT and are not written. `--te-bpw` allocates it the same way,
  token table included (the engine gathers rows from the quantized table).
- vae, tokenizer: copied verbatim; the release IS the engine's layout for both.

`--imatrix` is the file the ENGINE writes while generating with
`MLX_SERVE_IMATRIX=<path>` (tests/collect_image_imatrix.sh). Without it every
tensor gets the weight-only search. `config.json` is written LAST: it is what
makes the directory a model, so an interrupted conversion stays invisible.

    python3 tests/convert_krea.py --self-test
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import image_quant as iq  # noqa: E402

# krea.TE_LAYERS_RUN: the last tap reads the input of layer 35.
TE_LAYERS_RUN = 35
TOKENIZER_FILES = ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json")


def dit_name(key: str) -> str | None:
    return key


def te_name(key: str) -> str | None:
    """Pack key for a text-encoder tensor, or None when the conditioner never reads it."""
    if not key.startswith("language_model.") or key == "language_model.norm.weight":
        return None
    m = re.match(r"language_model\.layers\.(\d+)\.", key)
    if m and int(m.group(1)) >= TE_LAYERS_RUN:
        return None
    return key


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="krea/Krea-2-Turbo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imatrix", help="engine-collected activation statistics (MLX_SERVE_IMATRIX)")
    ap.add_argument("--dit-bpw", type=float, required=True, help="transformer budget, bits per weight incl. scales")
    ap.add_argument("--te-bpw", type=float, required=True, help="text-encoder budget, bits per weight incl. scales")
    ap.add_argument("--work", default="~/claude-tmp/image-quant", help="error tables + allocation reports")
    args = ap.parse_args()

    src = iq.resolve_src(args.src)
    turbo = src / "turbo.safetensors"
    if not turbo.is_file():
        sys.exit(f"{src}: no turbo.safetensors (the original-layout Krea-2 transformer this engine reads)")
    out = Path(args.out).expanduser()
    if (out / "config.json").exists():
        sys.exit(f"{out}: already a pack (config.json present)")
    work = Path(args.work).expanduser() / out.name
    work.mkdir(parents=True, exist_ok=True)

    dit = iq.convert("transformer", iq.Source([turbo]), dit_name, iq.ShardWriter(out, "transformer"),
                     iq.Imatrix(args.imatrix, "transformer"), args.dit_bpw, work)
    te = iq.convert("text_encoder", iq.Source.dir(src / "text_encoder"), te_name,
                    iq.ShardWriter(out / "text_encoder", "model"),
                    iq.Imatrix(args.imatrix, "text_encoder"), args.te_bpw, work)
    shutil.copyfile(src / "text_encoder" / "config.json", out / "text_encoder" / "config.json")
    (out / "vae").mkdir(exist_ok=True)
    for fn in ("config.json", "diffusion_pytorch_model.safetensors"):
        shutil.copyfile(src / "vae" / fn, out / "vae" / fn)
    (out / "tokenizer").mkdir(exist_ok=True)
    for fn in TOKENIZER_FILES:
        if (src / "tokenizer" / fn).is_file():
            shutil.copyfile(src / "tokenizer" / fn, out / "tokenizer" / fn)

    config = {
        "model_type": "krea2_turbo",
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

    check(te_name("language_model.layers.34.mlp.down_proj.weight") == "language_model.layers.34.mlp.down_proj.weight",
          "the last layer the conditioner runs is kept")
    check(te_name("language_model.layers.35.self_attn.q_proj.weight") is None, "layer 35 (read only as input) is dropped")
    check(te_name("language_model.norm.weight") is None, "the final norm is dropped")
    check(te_name("visual.blocks.0.attn.proj.weight") is None, "the vision tower is dropped")
    check(te_name("language_model.embed_tokens.weight") == "language_model.embed_tokens.weight", "the token table is kept")
    check(dit_name("blocks.27.mlp.down.weight") == "blocks.27.mlp.down.weight", "transformer keys pass through unchanged")
    print(f"\n{'OK' if not failures else 'FAILED'}: {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(self_test() if "--self-test" in sys.argv else main())

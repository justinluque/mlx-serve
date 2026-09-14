#!/usr/bin/env python3
"""Z-Image-Turbo -> an mlx-serve pack, imatrix-calibrated and budget-allocated.

    python3 tests/convert_zimage.py --src Tongyi-MAI/Z-Image-Turbo \\
        --out ~/.mlx-serve/models/<org>/Z-Image-Turbo-MLX-Serve-iq \\
        --imatrix ~/claude-tmp/zimage-imatrix.safetensors --dit-bpw 4.0 --te-bpw 4.5

`--src` is the upstream diffusers release — a local directory, or a repo id already
in the Hugging Face cache — holding `transformer/`, `text_encoder/`, `vae/`,
`tokenizer/` and `scheduler/`. The release ships fp32; tensors are read lazily and
the output streams into 2 GB shards.

- transformer: every linear `z_image.Dit` loads through `MfLinear`, at the width
  `image_quant.allocate` picks inside `--dit-bpw`. The conditioning path (timestep
  MLP, caption and patch embedders, the final layer, every block's adaLN
  modulation) is held at bf16: the klein recipe in Z-Image's names, and the set the
  8-bit mirror already keeps dense. Norms, biases and pad tokens are stored bf16.
- text_encoder: the Qwen3 LM the chat `Transformer` loads, allocated inside
  `--te-bpw` with the weight-only search, since the engine collects no
  text-encoder statistics. A quantized config makes that loader demand `.scales`
  on every layer linear, so none is stored dense. The token table is allocated too:
  `rawEmbedding` solves its width from geometry. The
  conditioner captures after layer n-2, so the last layer (and any lm_head) is
  read by nothing and is stored at the narrowest width.
- vae, tokenizer, scheduler: copied verbatim.

The output name must say "turbo" exactly when the source does: `z_image.dirLooksTurbo`
reads the DIRECTORY name to choose 8 steps without CFG over 50 steps at CFG 5.
`model_index.json` is written LAST: it is what makes the directory a model, so an
interrupted conversion stays invisible.

`--imatrix` is the file the ENGINE writes while generating with
`MLX_SERVE_IMATRIX=<path>` (tests/collect_image_imatrix.sh).

    python3 tests/convert_zimage.py --self-test
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import image_quant as iq  # noqa: E402

LICENSE_NAMES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "NOTICE")

CONDITIONING = re.compile(
    r"^(t_embedder\.mlp\.\d+|cap_embedder\.1|all_x_embedder\.2-1|all_final_layer\.2-1\.(linear|adaLN_modulation\.1)"
    r"|(noise_refiner|layers)\.\d+\.adaLN_modulation\.0)$")


def keep(key: str) -> str | None:
    return key


def te_layer(module: str) -> int | None:
    m = re.search(r"(?:^|\.)layers\.(\d+)\.", module)
    return int(m.group(1)) if m else None


def unread(module: str, te_layers: int) -> bool:
    """A text-encoder weight the conditioner never reads: it captures after layer n-2."""
    layer = te_layer(module)
    return module.split(".")[-1] == "lm_head" or (layer is not None and layer >= te_layers - 1)


def floor_for(component: str, module: str, te_layers: int) -> int | None:
    """The narrowest width a linear may be stored at, or None to leave it to the allocator."""
    if component == "transformer":
        return iq.DENSE if CONDITIONING.match(module) else None
    layer = te_layer(module)
    # llama.cpp's Q4_K_M widens ffn_down on these layers.
    if module.endswith("mlp.down_proj") and not unread(module, te_layers) and iq.use_more_bits(layer, te_layers):
        return 6
    return None


def ceil_for(component: str, module: str, te_layers: int) -> int | None:
    """The widest width a linear may be stored at, or None for no ceiling."""
    if component != "text_encoder" or module.endswith("embed_tokens"):
        return None
    return min(iq.WIDTHS) if unread(module, te_layers) else max(iq.WIDTHS)


def turbo_mismatch(src: str, out: str) -> bool:
    """`z_image.dirLooksTurbo` matches "turbo" anywhere in the model PATH."""
    looks = lambda s: "turbo" in s.lower()  # noqa: E731
    return looks(src) != looks(out)


def dominant_bits(record: dict) -> int:
    """The width most quantized linears were stored at: what the config declares."""
    counts = {int(k[: -len("bit")]): n for k, n in record["widths"].items() if k.endswith("bit")}
    return max(counts, key=lambda b: (counts[b], b))


def copy(src: Path, dst_dir: Path, required: bool = True) -> None:
    if not src.is_file():
        if required:
            sys.exit(f"required file missing from the release: {src}")
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst_dir / src.name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="Tongyi-MAI/Z-Image-Turbo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imatrix", help="engine-collected activation statistics (MLX_SERVE_IMATRIX)")
    ap.add_argument("--dit-bpw", type=float, required=True, help="transformer budget, bits per weight incl. scales")
    ap.add_argument("--te-bpw", type=float, required=True, help="text-encoder budget, bits per weight incl. scales")
    ap.add_argument("--work", default="~/claude-tmp/image-quant", help="error tables + allocation reports")
    args = ap.parse_args()

    src = iq.resolve_src(args.src)
    index = src / "model_index.json"
    class_name = json.loads(index.read_text()).get("_class_name") if index.is_file() else None
    if class_name != "ZImagePipeline":
        sys.exit(f"{src}: not a Z-Image release (model_index.json _class_name {class_name!r})")
    out = Path(args.out).expanduser()
    if turbo_mismatch(args.src, str(out)):
        sys.exit(f"{out}: the engine picks Turbo sampling from the model path, and it disagrees "
                 f"with {args.src} about 'turbo'")
    if (out / "model_index.json").exists():
        sys.exit(f"{out}: already a pack (model_index.json present)")
    work = Path(args.work).expanduser() / out.name
    work.mkdir(parents=True, exist_ok=True)

    te_cfg = json.loads((src / "text_encoder" / "config.json").read_text())
    te_layers = te_cfg.get("num_hidden_layers")
    if not te_layers:
        sys.exit(f"{src / 'text_encoder' / 'config.json'}: no num_hidden_layers")
    dit = iq.convert("transformer", iq.Source.dir(src / "transformer"), keep,
                     iq.ShardWriter(out / "transformer", "diffusion_pytorch_model",
                                    index="diffusion_pytorch_model.safetensors.index.json"),
                     iq.Imatrix(args.imatrix, "transformer"), args.dit_bpw, work,
                     lambda m: floor_for("transformer", m, te_layers))
    te = iq.convert("text_encoder", iq.Source.dir(src / "text_encoder"), keep,
                    iq.ShardWriter(out / "text_encoder", "model", index="model.safetensors.index.json"),
                    iq.Imatrix(None, "text_encoder"), args.te_bpw, work,
                    lambda m: floor_for("text_encoder", m, te_layers),
                    lambda m: ceil_for("text_encoder", m, te_layers))

    copy(src / "transformer" / "config.json", out / "transformer")
    copy(src / "text_encoder" / "generation_config.json", out / "text_encoder", required=False)
    # The chat loader reads dense-vs-quantized from this block, never from `.scales` presence.
    te_cfg["quantization"] = {"group_size": iq.GROUP_SIZE, "bits": dominant_bits(te), "mode": "affine"}
    (out / "text_encoder" / "config.json").write_text(json.dumps(te_cfg, indent=2) + "\n")
    copy(src / "scheduler" / "scheduler_config.json", out / "scheduler")
    copy(src / "vae" / "config.json", out / "vae")
    vae_weights = sorted((src / "vae").glob("*.safetensors"))
    if not vae_weights:
        sys.exit(f"{src / 'vae'}: no .safetensors")
    for f in vae_weights:
        copy(f, out / "vae")
    for f in sorted((src / "tokenizer").iterdir()):
        if f.is_file():
            copy(f, out / "tokenizer")
    for name in LICENSE_NAMES:
        copy(src / name, out, required=False)
    (out / "quantization.json").write_text(json.dumps(
        {"method": "imatrix-weighted affine, measured bit allocation", "group_size": iq.GROUP_SIZE,
         "transformer": dit, "text_encoder": te}, indent=2) + "\n")
    copy(index, out)
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

    for module in ("t_embedder.mlp.0", "t_embedder.mlp.2", "cap_embedder.1", "all_x_embedder.2-1",
                   "all_final_layer.2-1.linear", "all_final_layer.2-1.adaLN_modulation.1",
                   "layers.29.adaLN_modulation.0", "noise_refiner.1.adaLN_modulation.0"):
        check(floor_for("transformer", module, 36) == iq.DENSE, f"conditioning tensor {module} is held at bf16")
    for module in ("layers.0.attention.to_q", "layers.29.feed_forward.w2", "noise_refiner.1.attention.to_out.0",
                   "context_refiner.0.feed_forward.w1"):
        check(floor_for("transformer", module, 36) is None, f"block linear {module} is left to the allocator")
        check(ceil_for("transformer", module, 36) is None, f"block linear {module} has no ceiling")

    for prefix in ("model.", ""):
        down = [i for i in range(36) if floor_for("text_encoder", f"{prefix}layers.{i}.mlp.down_proj", 36) == 6]
        check(down == [0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 31, 32, 33, 34],
              f"text-encoder down_proj is floored at 6 bits on llama.cpp's layers, bar the unread last ({prefix or 'bare'})")
        check(ceil_for("text_encoder", f"{prefix}layers.35.self_attn.q_proj", 36) == min(iq.WIDTHS),
              "the layer after the capture is stored at the narrowest width")
        check(ceil_for("text_encoder", f"{prefix}layers.34.self_attn.q_proj", 36) == max(iq.WIDTHS),
              "a read layer may not be stored dense under a quantized config")
        check(floor_for("text_encoder", f"{prefix}embed_tokens", 36) is None
              and ceil_for("text_encoder", f"{prefix}embed_tokens", 36) is None,
              "the token table is left to the allocator (rawEmbedding solves its width from geometry)")
    check(ceil_for("text_encoder", "lm_head", 36) == min(iq.WIDTHS), "an lm_head nothing reads is stored narrowest")
    modules = [f"model.layers.{i}.{p}" for i in range(36) for p in
               ("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj")] + ["model.embed_tokens", "lm_head"]
    check(all((floor_for("text_encoder", m, 36) or 0) <= (ceil_for("text_encoder", m, 36) or iq.DENSE) for m in modules),
          "no text-encoder floor sits above its ceiling")

    check(not turbo_mismatch("Tongyi-MAI/Z-Image-Turbo", "Z-Image-Turbo-MLX-Serve-iq"), "a Turbo pack named turbo converts")
    check(turbo_mismatch("Tongyi-MAI/Z-Image-Turbo", "Z-Image-MLX-Serve-iq"), "a Turbo pack without turbo in its name refuses")
    check(turbo_mismatch("Tongyi-MAI/Z-Image", "z-image-turbo-iq"), "a base pack named turbo refuses")
    check(dominant_bits({"widths": {"3bit": 2, "4bit": 10, "5bit": 10, "bf16": 40}}) == 5,
          "the config declares the most common quantized width, ignoring bf16")
    print(f"\n{'OK' if not failures else 'FAILED'}: {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(self_test() if "--self-test" in sys.argv else main())

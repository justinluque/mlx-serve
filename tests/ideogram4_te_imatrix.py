#!/usr/bin/env python3
"""Per-input-channel activation statistics (an "imatrix") for Ideogram 4's
text encoder, collected on the prompts it actually sees: structured JSON
captions.

Why this file exists: `convert_ideogram4.py` quantizes every linear with a bare
`mx.quantize` — round-to-nearest, min/max per group, activation-BLIND. The
encoder is a third of the pack and runs at 4-bit in every shipped policy, and
it is where the whole prompt -> layout signal lives, so it is the tensor most
worth calibrating. `tests/dsv4_imatrix.py`'s `weighted_affine_quant` consumes
what this writes unchanged (mean-squared activation per INPUT channel: an error
dW_j in channel j contributes dW_j^2 * E[x_j^2] to the output, so the weight IS
the mean square, no square rooting).

An imatrix is only valid for the WEIGHTS it was collected on, so collection
runs against the SOURCE checkpoint's `text_encoder/`, never a converted pack.

The corpus is the distribution that matters. Ideogram 4 was trained
EXCLUSIVELY on structured JSON captions, so calibrating on chat or prose would
weight channels this encoder never excites in service. `--captions FILE` takes
real ones (one JSON object per line — the server logs every rewrite it makes);
absent that, `synthetic_captions` generates schema-valid ones across subjects,
media, palettes, bbox layouts, text elements and aspect ratios. Both are
rendered through the pack's own chat framing, because that framing is part of
every prompt the encoder ever encodes.

  # collect (needs mlx_lm + the source checkpoint)
  python3 tests/ideogram4_te_imatrix.py collect --src <ideogram repo> \
      --out ~/claude-tmp/ideogram-iq/te_imatrix.safetensors

  # then
  python3 tests/convert_ideogram4.py --src <repo> --out <pack> \
      --precision mixed_3_8 --te-imatrix ~/claude-tmp/ideogram-iq/te_imatrix.safetensors

  python3 tests/ideogram4_te_imatrix.py --self-test   # pure, no deps

The DiT is calibrated the other way round, and cannot be done here: its
activations depend on the latents and the timestep, so there is no reference
tree to push text through. mlx-serve collects them itself during real
generations (`ideogram4.Imatrix`, armed by `MLX_SERVE_IDEOGRAM_IMATRIX=<path>`,
keyed `<component>/<module>` so one file covers both transformers) and
`convert_ideogram4.py --dit-imatrix` consumes that. Run a spread of prompts and
sizes at the guidance you actually use — the file is the sum over everything
the engine rendered while it was armed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# The chat framing the pack's config.json carries, and what mlx-serve falls
# back to when it does not (`Ideogram4Impl.default_chat_prefix`/`_suffix`).
DEFAULT_CHAT_PREFIX = "<|im_start|>user\n"
DEFAULT_CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

# Every tenth caption is HELD OUT, the same rule `qwen38_imatrix_collect.py`
# uses: a quality number measured on the captions the imatrix was fit to
# flatters the pack it is judging.
HOLDOUT_EVERY = 10


def split_docs(docs: list[str], holdout: bool) -> list[str]:
    return [d for i, d in enumerate(docs) if (i % HOLDOUT_EVERY == 0) == holdout]


# ============================================================
# Naming
# ============================================================

# The prefixes the text tower can carry in a source checkpoint — the same
# probe `convert_text_encoder` runs, in the same order, because the imatrix is
# keyed by what the CONVERTER calls each module (`layers.0.self_attn.q_proj`)
# rather than by the checkpoint's spelling. One naming on both sides means a
# transformers-version prefix change cannot silently void the file.
TOWER_PREFIXES = ("language_model.", "model.language_model.", "model.", "")


def module_key(path: str) -> str:
    """An mlx_lm module path -> the name `convert_ideogram4.emit_linear` writes.

    Returns "" for anything outside the text tower (the vision tower, and
    `lm_head`, which the encoder never runs because it stops at the taps)."""
    for pfx in TOWER_PREFIXES:
        if pfx and path.startswith(pfx):
            path = path[len(pfx):]
            break
    if path.startswith("visual.") or path == "lm_head":
        return ""
    return path


# ============================================================
# Corpus
# ============================================================

_SUBJECTS = [
    ("a weathered red barn", "obj", "photograph"),
    ("a snow leopard mid-stride on grey rock", "obj", "photograph"),
    ("a chrome espresso machine on a marble counter", "obj", "photograph"),
    ("a young woman in a navy blazer, arms folded", "obj", "photograph"),
    ("a hand-thrown ceramic bowl glazed celadon", "obj", "photograph"),
    ("a paper crane made of sheet music", "obj", "illustration"),
    ("a fox curled asleep under ferns", "obj", "illustration"),
    ("a 1970s transit bus at a rain-slick stop", "obj", "photograph"),
    ("a stack of hardcover books tied with twine", "obj", "photograph"),
    ("a cathedral rose window seen from inside", "obj", "photograph"),
    ("an astronaut's glove holding a seedling", "obj", "3D render"),
    ("a bowl of ramen with a soft-boiled egg", "obj", "photograph"),
]
_SIGNS = [
    "HARVEST", "OPEN\nLATE", "CAFÉ MOKO", "第 三 号", "СЕВЕР",
    "Grand Opening", "PLATFORM 9", "क़िताबें", "NO PARKING\nANY TIME",
]
_BACKGROUNDS = [
    "a wheat field under overcast daylight",
    "a fogged studio backdrop, neutral grey",
    "a wet city street at blue hour, sodium streetlamps",
    "pale birch woodland, diffused daylight",
    "a workshop wall hung with hand tools",
    "an empty gallery, white walls, polished concrete",
]
_STYLES = [
    "35mm film photograph", "iPhone photo", "flat vector illustration",
    "editorial digital painting", "Studio Ghibli animation", "Pixar 3D animation",
]
_PALETTES = [
    ["#C0392B", "#F1C40F", "#2C3E50"], ["#1B4332", "#95D5B2"],
    ["#FFFFFF", "#0A0A0A"], ["#E8D5C4", "#7C6A56", "#3A2E24"],
]
_RATIOS = ["1:1", "16:9", "9:16", "4:5", "3:2", "2:3", "3:1"]


def _bbox(rng: random.Random) -> list[int]:
    y1 = rng.randrange(0, 500)
    x1 = rng.randrange(0, 500)
    return [y1, x1, y1 + rng.randrange(120, 500), x1 + rng.randrange(120, 500)]


def synthetic_captions(n: int, seed: int = 0) -> list[str]:
    """`n` schema-valid captions spanning the shapes the renderer is given.

    Generated, and this file says so in the metadata: it is a stand-in for real
    rewrites (`--captions`), and its job is to excite the channels a caption
    excites — key tokens, bbox digit runs, hex colours, non-ASCII sign text —
    not to be good prompts."""
    rng = random.Random(seed)
    out: list[str] = []
    for i in range(n):
        subject, _, medium = _SUBJECTS[i % len(_SUBJECTS)]
        n_el = 1 + (i % 4)
        elements = []
        for j in range(n_el):
            s, _, _ = _SUBJECTS[(i + j * 5) % len(_SUBJECTS)]
            el: dict = {"type": "obj"}
            if j == 0 or rng.random() < 0.6:
                el["bbox"] = _bbox(rng)
            el["desc"] = s
            if rng.random() < 0.4:
                el["color_palette"] = rng.choice(_PALETTES)[:2]
            elements.append(el)
        if i % 3 == 0:
            text_el: dict = {"type": "text", "bbox": _bbox(rng),
                             "text": _SIGNS[i % len(_SIGNS)],
                             "desc": "painted signage, condensed sans serif"}
            elements.insert(rng.randrange(len(elements) + 1), text_el)
        caption = {
            "aspect_ratio": _RATIOS[i % len(_RATIOS)],
            "high_level_description":
                f"{subject.capitalize()}, a {rng.choice(_STYLES)}, "
                f"with {_BACKGROUNDS[i % len(_BACKGROUNDS)]} behind it.",
            "compositional_deconstruction": {
                "background": _BACKGROUNDS[(i + 2) % len(_BACKGROUNDS)],
                "elements": elements,
            },
        }
        if i % 2 == 0:
            caption["style_description"] = {
                "aesthetics": "restrained, documentary",
                "lighting": "overcast daylight, cool-neutral white balance",
                ("photo" if medium == "photograph" else "art_style"):
                    ("35mm, natural grain" if medium == "photograph" else rng.choice(_STYLES)),
                "medium": medium,
                "color_palette": rng.choice(_PALETTES),
            }
        out.append(json.dumps(caption, ensure_ascii=False, separators=(",", ":")))
    return out


def read_captions(path: Path) -> list[str]:
    """Real captions: one JSON object per line. Anything that does not parse as
    an object is dropped rather than fed in as prose — the corpus is a claim
    about the distribution, so a stray log line has no business in it."""
    docs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            docs.append(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    return docs


def framed(captions: list[str], prefix: str, suffix: str) -> list[str]:
    return [prefix + c + suffix for c in captions]


def chat_framing_from(pack_or_src: Path) -> tuple[str, str]:
    """The framing the pack records, else mlx-serve's defaults. The encoder
    sees the framed string, so calibrating on a bare caption under-weights
    whatever the marker tokens excite."""
    cfg = pack_or_src / "config.json"
    if cfg.exists():
        try:
            obj = json.loads(cfg.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                return (obj.get("chat_prefix") or DEFAULT_CHAT_PREFIX,
                        obj.get("chat_suffix") or DEFAULT_CHAT_SUFFIX)
        except (json.JSONDecodeError, OSError):
            pass
    return DEFAULT_CHAT_PREFIX, DEFAULT_CHAT_SUFFIX


# ============================================================
# Collection (needs mlx_lm + the source checkpoint)
# ============================================================

def collect(args) -> int:
    import mlx.core as mx
    import mlx.nn as nn

    src = Path(args.src).expanduser()
    te_dir = src / "text_encoder"
    if not te_dir.exists():
        sys.exit(f"no text_encoder/ under {src} — collection runs on the SOURCE checkpoint, "
                 "not a converted pack (an imatrix is only valid for the weights it was collected on)")

    try:
        from mlx_lm.utils import load
    except ImportError:
        sys.exit("mlx_lm is required for collection: pip install mlx-lm")

    # The tokenizer lives at the pipeline ROOT, the weights in text_encoder/.
    model, tokenizer = load(str(te_dir), tokenizer_config={}, model_config={})

    acc: dict[str, "mx.array"] = {}
    rows: dict[str, int] = {}
    by_id: dict[int, str] = {}
    for path, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            key = module_key(path)
            if key:
                by_id[id(mod)] = key
                acc[key] = None
                rows[key] = 0

    if not by_id:
        sys.exit("no text-tower linears were wrapped — check the checkpoint layout")

    orig = nn.Linear.__call__

    def patched(self, x):
        key = by_id.get(id(self))
        if key is not None:
            flat = x.reshape(-1, x.shape[-1]).astype(mx.float32)
            s = (flat * flat).sum(axis=0)
            acc[key] = s if acc[key] is None else acc[key] + s
            rows[key] += flat.shape[0]
        return orig(self, x)

    nn.Linear.__call__ = patched
    try:
        if args.captions:
            captions = read_captions(Path(args.captions).expanduser())
            origin = f"captions:{args.captions}"
            if not captions:
                sys.exit(f"{args.captions} held no JSON captions")
        else:
            captions = synthetic_captions(args.count, args.seed)
            origin = "synthetic"
        prefix, suffix = chat_framing_from(src)
        docs = framed(split_docs(captions, holdout=False), prefix, suffix)
        print(f"[imatrix] {len(docs)} captions ({origin}), {len(by_id)} linears wrapped", flush=True)

        total = 0
        for i, doc in enumerate(docs):
            ids = tokenizer.encode(doc)
            # The DiT's text window is 2048; a caption past it is not a prompt
            # this encoder is ever asked for.
            ids = ids[: args.max_tokens]
            model(mx.array([ids]))
            mx.eval([v for v in acc.values() if v is not None])
            total += len(ids)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(docs)} captions, {total} tokens", flush=True)
    finally:
        nn.Linear.__call__ = orig

    missing = [k for k, v in acc.items() if v is None]
    if missing:
        sys.exit(f"{len(missing)} wrapped linears never fired: {missing[:5]}")

    arrays = {k: (v / float(rows[k])).astype(mx.float32) for k, v in acc.items()}
    mx.eval(list(arrays.values()))
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out), arrays, metadata={
        "source": str(src),
        "origin": origin,
        "captions": str(len(docs)),
        "total_tokens": str(total),
        "seed": str(args.seed),
        "values": "mean-squared activation per INPUT channel (sum(x^2)/rows)",
        "keys": "convert_ideogram4 module names (text tower, prefix stripped)",
        "holdout": "calibrated on split_docs(holdout=False); every 10th caption withheld",
    })
    print(f"wrote {out} — {len(arrays)} entries, {total} tokens", flush=True)
    return 0


# ============================================================
# Self-test (pure: no mlx, no torch, no checkpoint)
# ============================================================

def self_test() -> int:
    failures: list[str] = []

    def check(cond, label):
        if not cond:
            failures.append(label)

    # Naming: the imatrix is keyed by what the CONVERTER writes, so every
    # prefix spelling the converter probes must collapse to the same key.
    for pfx in ("language_model.", "model.language_model.", "model.", ""):
        check(module_key(pfx + "layers.7.self_attn.q_proj") == "layers.7.self_attn.q_proj",
              f"prefix {pfx!r} did not strip")
    check(module_key("embed_tokens") == "embed_tokens", "embed_tokens key")
    # Never calibrated: the vision tower is not fetched and lm_head is not run.
    check(module_key("visual.blocks.0.attn.qkv") == "", "vision tower not excluded")
    check(module_key("model.visual.blocks.0.attn.qkv") == "", "prefixed vision tower not excluded")
    check(module_key("lm_head") == "", "lm_head not excluded")

    # Corpus: every caption must satisfy the contract the renderer is given, or
    # the calibration weights channels the model never sees in service.
    caps = synthetic_captions(40, seed=3)
    check(len(caps) == 40, "count")
    check(len(set(caps)) == 40, "captions repeat")
    seen_text = seen_bbox = seen_style = seen_nonascii = 0
    ratios = set()
    for c in caps:
        obj = json.loads(c)
        check(set(obj) <= {"aspect_ratio", "high_level_description", "style_description",
                           "compositional_deconstruction"}, "unknown top-level key")
        check("compositional_deconstruction" in obj, "missing compositional_deconstruction")
        ratios.add(obj["aspect_ratio"])
        cd = obj["compositional_deconstruction"]
        check(isinstance(cd.get("background"), str) and cd["background"], "background")
        els = cd.get("elements")
        check(isinstance(els, list) and els, "elements")
        for el in els:
            check(el.get("type") in ("obj", "text"), "element type")
            check(isinstance(el.get("desc"), str), "element desc")
            if el["type"] == "text":
                seen_text += 1
                check(isinstance(el.get("text"), str) and el["text"], "text element has no text")
            if "bbox" in el:
                seen_bbox += 1
                b = el["bbox"]
                check(len(b) == 4 and all(isinstance(v, int) for v in b), "bbox shape")
                check(b[0] < b[2] and b[1] < b[3], "bbox is not top-left/bottom-right")
        if "style_description" in obj:
            seen_style += 1
            sd = obj["style_description"]
            check(("photo" in sd) != ("art_style" in sd), "exactly one of photo/art_style")
            for hexv in sd.get("color_palette", []):
                check(len(hexv) == 7 and hexv[0] == "#" and hexv[1:].upper() == hexv[1:],
                      f"palette entry {hexv} is not uppercase #RRGGBB")
        if any(ord(ch) > 127 for ch in c):
            seen_nonascii += 1
    check(seen_text > 0, "no text elements — the glyph channels go uncalibrated")
    check(seen_bbox > 0, "no bboxes — the digit-run channels go uncalibrated")
    check(seen_style > 0, "no style_description blocks")
    check(seen_nonascii > 0, "corpus is ASCII-only — the prompt mandates preserving CJK/Cyrillic")
    check(len(ratios) >= 5, "aspect ratios do not span the menu")
    # A caption is what the encoder is GIVEN, but the framing rides with it.
    fr = framed(["{}"], DEFAULT_CHAT_PREFIX, DEFAULT_CHAT_SUFFIX)[0]
    check(fr.startswith(DEFAULT_CHAT_PREFIX) and fr.endswith(DEFAULT_CHAT_SUFFIX), "framing")

    # Holdout: the two halves partition, and neither is empty.
    a, b = split_docs(caps, holdout=False), split_docs(caps, holdout=True)
    check(len(a) + len(b) == len(caps) and not (set(a) & set(b)), "holdout does not partition")
    check(b, "holdout half is empty")

    for f in failures:
        print(f"FAIL: {f}")
    print("ideogram4_te_imatrix self-test:", "PASS" if not failures else f"{len(failures)} FAILED")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", choices=("collect",), help="collect the imatrix")
    ap.add_argument("--src", help="SOURCE checkpoint dir (the one the converter reads)")
    ap.add_argument("--out", help="destination .safetensors")
    ap.add_argument("--captions", help="real captions, one JSON object per line")
    ap.add_argument("--count", type=int, default=256, help="synthetic captions when --captions is absent")
    ap.add_argument("--max-tokens", type=int, default=2048, help="the DiT's own text window")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true", help="pure checks; no mlx, no checkpoint")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.mode != "collect":
        ap.error("nothing to do: pass `collect` or --self-test")
    if not args.src or not args.out:
        ap.error("collect needs --src and --out")
    return collect(args)


if __name__ == "__main__":
    sys.exit(main())

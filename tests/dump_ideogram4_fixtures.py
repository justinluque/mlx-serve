#!/usr/bin/env python3
"""Numeric oracle for the Ideogram 4 DiT.

    pip install torch safetensors 'ideogram4 @ git+https://github.com/ideogram-oss/ideogram4'
    python tests/dump_ideogram4_fixtures.py build --out /tmp/ideo-fix
    IDEOGRAM4_FIXTURE_DIR=/tmp/ideo-fix zig build test -Dtest-filter="ideogram4 fixture"

The published weights are gated and 27 GB, which would make the oracle
something almost nobody could run. So this builds a TINY randomly-initialised
`Ideogram4Transformer` from the reference's own classes and dumps both the
weights and the reference's outputs. That proves the port — the arithmetic,
the MRoPE interleave, the AdaLN gating, the packed-sequence layout — without
proving anything about a checkpoint, which is the right split: a checkpoint
mismatch surfaces as a wrong image, an arithmetic mismatch does not.

The rope tables are dumped SEPARATELY from the forward. They are pure
functions of the position ids, they are the single most transcription-prone
part of this architecture (the reference's interleave leaves `mrope_section[0]`
unread and fills the tail from t), and a mismatch there is otherwise diffuse:
every layer is slightly wrong and the cosine still looks plausible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

# Small enough to run on CPU in seconds, large enough that every structural
# choice is exercised.
#
# head_dim is 192, not something smaller, because the mrope interleave writes
# indices up to 3*max(section) = 60 into a table of head_dim/2 entries — at
# head_dim 96 the reference itself would index past the end, and at anything
# under 120 the tail that section[0] never covers would vanish, which is the
# exact behaviour being pinned. num_heads is 2 so the head reshape is real,
# and intermediate_size is deliberately not a multiple of emb_dim.
TINY = dict(
    emb_dim=384,      # 2 heads x 192
    num_layers=3,
    num_heads=2,
    intermediate_size=320,
    adanln_dim=64,
    in_channels=128,
)


def build_config(llm_dim: int):
    from ideogram4.modeling_ideogram4 import Ideogram4Config

    return Ideogram4Config(
        emb_dim=TINY["emb_dim"],
        num_layers=TINY["num_layers"],
        num_heads=TINY["num_heads"],
        intermediate_size=TINY["intermediate_size"],
        adanln_dim=TINY["adanln_dim"],
        in_channels=TINY["in_channels"],
        llm_features_dim=llm_dim,
        # The real theta and sections: the interleave is what is being pinned,
        # and shrinking either would pin a different table.
        rope_theta=5_000_000,
        mrope_section=(24, 20, 20),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("build",))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text-tokens", type=int, default=7)
    ap.add_argument("--grid-h", type=int, default=4)
    ap.add_argument("--grid-w", type=int, default=5)
    ap.add_argument("--llm-dim", type=int, default=256)
    args = ap.parse_args()

    from ideogram4.constants import (
        IMAGE_POSITION_OFFSET,
        LLM_TOKEN_INDICATOR,
        OUTPUT_IMAGE_INDICATOR,
    )
    from ideogram4.modeling_ideogram4 import Ideogram4Transformer

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = build_config(args.llm_dim)
    model = Ideogram4Transformer(cfg).to(torch.float32).eval()

    n_txt = args.text_tokens
    n_img = args.grid_h * args.grid_w
    total = n_txt + n_img

    # The packed sequence, exactly as `_build_inputs` lays it out for one
    # prompt: [text tokens][image tokens], no left padding (we serve batch 1,
    # and padding only exists to batch several prompts).
    h_idx = torch.arange(args.grid_h).view(-1, 1).expand(args.grid_h, args.grid_w).reshape(-1)
    w_idx = torch.arange(args.grid_w).view(1, -1).expand(args.grid_h, args.grid_w).reshape(-1)
    image_pos = torch.stack([torch.zeros_like(h_idx), h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET
    text_pos = torch.arange(n_txt).view(-1, 1).expand(n_txt, 3)

    position_ids = torch.cat([text_pos, image_pos], dim=0).unsqueeze(0)
    segment_ids = torch.ones(1, total, dtype=torch.long)
    indicator = torch.cat([
        torch.full((n_txt,), LLM_TOKEN_INDICATOR),
        torch.full((n_img,), OUTPUT_IMAGE_INDICATOR),
    ]).unsqueeze(0)

    llm_features = torch.zeros(1, total, cfg.llm_features_dim)
    llm_features[0, :n_txt] = torch.randn(n_txt, cfg.llm_features_dim)
    z = torch.zeros(1, total, cfg.in_channels)
    z_img = torch.randn(1, n_img, cfg.in_channels)
    z[0, n_txt:] = z_img[0]
    t = torch.tensor([0.37])

    with torch.no_grad():
        cos, sin = model.rotary_emb(position_ids)
        velocity = model(
            llm_features=llm_features,
            x=z,
            t=t,
            position_ids=position_ids,
            segment_ids=segment_ids,
            indicator=indicator,
        )

    tensors = {k: v.to(torch.float32).contiguous() for k, v in model.state_dict().items()}
    save_file(tensors, str(out / "model.safetensors"))

    save_file(
        {
            "llm_features_text": llm_features[0, :n_txt].contiguous(),
            "z_image": z_img[0].contiguous(),
            "rope_cos": cos[0].contiguous(),
            "rope_sin": sin[0].contiguous(),
            # Only the image span is meaningful — the reference says so, and
            # the text rows carry whatever the final layer made of them.
            "velocity_image": velocity[0, n_txt:].contiguous(),
        },
        str(out / "reference.safetensors"),
    )

    meta = {
        "emb_dim": cfg.emb_dim,
        "num_layers": cfg.num_layers,
        "num_heads": cfg.num_heads,
        "head_dim": cfg.emb_dim // cfg.num_heads,
        "intermediate_size": cfg.intermediate_size,
        "adaln_dim": cfg.adanln_dim,
        "in_channels": cfg.in_channels,
        "llm_features_dim": cfg.llm_features_dim,
        "rope_theta": cfg.rope_theta,
        "mrope_section": list(cfg.mrope_section),
        "n_text": n_txt,
        "grid_h": args.grid_h,
        "grid_w": args.grid_w,
        "t": float(t.item()),
        "image_position_offset": IMAGE_POSITION_OFFSET,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}: {len(tensors)} weights, seq={total} ({n_txt} text + {n_img} image)")
    print(f"  velocity |mean|={velocity[0, n_txt:].abs().mean():.6f} rms={velocity[0, n_txt:].pow(2).mean().sqrt():.6f}")


if __name__ == "__main__":
    main()

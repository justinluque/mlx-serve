#!/usr/bin/env python3
"""How much of a reference pack's output a quantized image pack keeps.

    python3 tests/image_quant_eval.py --ref <pack> --cand <pack> [--cand <pack> ...] \\
        [--steps 8] [--size 1024x1024] [--seeds 2]

Boots `zig-out/bin/mlx-serve` on one pack at a time (a 24 GB Mac holds one),
renders every held-out prompt (tests/fixtures/image_eval_prompts.txt, never the
calibration corpus) at fixed seeds, and scores each candidate against the
reference render of the same prompt and seed: PSNR over RGB and SSIM over luma.
A few-step sampler turns small weight error into composition drift that pixel
metrics punish hard, so read the numbers as a RANKING of packs at comparable
sizes, and open the PNGs before believing any of them. Beside the scores: bytes
on disk and the server's peak Metal bytes (`/props`).

    python3 tests/image_quant_eval.py --self-test
"""

import argparse
import base64
import json
import os
import signal
import struct
import subprocess
import sys
import time
import urllib.request
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


# ── PNG (8-bit RGB/RGBA, non-interlaced: what the engine writes) ──────────


def decode_png(data: bytes) -> np.ndarray:
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    pos, idat, w = 8, b"", 0
    while pos < len(data):
        n, kind = struct.unpack(">I4s", data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            w, h, depth, color, _, _, interlace = struct.unpack(">IIBBBBB", chunk)
            assert depth == 8 and color in (2, 6) and interlace == 0, "unsupported PNG"
            ch = 3 if color == 2 else 4
        elif kind == b"IDAT":
            idat += chunk
        pos += 12 + n
    raw = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(h, 1 + w * ch)
    out = np.zeros((h, w * ch), dtype=np.int32)
    for y in range(h):
        f, line = raw[y, 0], raw[y, 1:].astype(np.int32)
        up = out[y - 1] if y else np.zeros(w * ch, np.int32)
        if f == 0:
            cur = line
        elif f == 2:
            cur = (line + up) & 0xFF
        else:
            cur = np.zeros(w * ch, np.int32)
            for x in range(w * ch):
                a = cur[x - ch] if x >= ch else 0
                c = up[x - ch] if x >= ch else 0
                if f == 1:
                    pred = a
                elif f == 3:
                    pred = (a + up[x]) >> 1
                else:
                    p = a + up[x] - c
                    pa, pb, pc = abs(p - a), abs(p - up[x]), abs(p - c)
                    pred = a if pa <= pb and pa <= pc else (up[x] if pb <= pc else c)
                cur[x] = (line[x] + pred) & 0xFF
        out[y] = cur
    return out.reshape(h, w, ch)[..., :3].astype(np.uint8)


# ── metrics ───────────────────────────────────────────────────────────────


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def _blur(x: np.ndarray) -> np.ndarray:
    t = np.arange(-5, 6)
    g = np.exp(-(t ** 2) / (2 * 1.5 ** 2))
    g /= g.sum()
    x = np.apply_along_axis(lambda r: np.convolve(r, g, mode="valid"), 1, x)
    return np.apply_along_axis(lambda c: np.convolve(c, g, mode="valid"), 0, x)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Gaussian-window SSIM (11x11, sigma 1.5) over BT.601 luma."""
    wts = np.array([0.299, 0.587, 0.114])
    x, y = a.astype(np.float64) @ wts, b.astype(np.float64) @ wts
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx_, my = _blur(x), _blur(y)
    sx, sy, sxy = _blur(x * x) - mx_ ** 2, _blur(y * y) - my ** 2, _blur(x * y) - mx_ * my
    return float(np.mean(((2 * mx_ * my + c1) * (2 * sxy + c2)) / ((mx_ ** 2 + my ** 2 + c1) * (sx + sy + c2))))


# ── serving ───────────────────────────────────────────────────────────────


def http(url: str, body: dict | None = None, timeout: float = 3600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def render(pack: Path, prompts: list[str], seeds: int, steps: int, size: str, out: Path, port: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "server.log", "w")
    proc = subprocess.Popen([str(ROOT / "zig-out" / "bin" / "mlx-serve"), "--serve", "--model", str(pack),
                             "--host", "127.0.0.1", "--port", str(port)], stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen(base + "/health", timeout=2)
                break
            except Exception:
                if proc.poll() is not None:
                    sys.exit(f"{pack}: server exited during boot (see {out / 'server.log'})")
                time.sleep(1)
        seconds = []
        for i, prompt in enumerate(prompts):
            for seed in range(seeds):
                t0 = time.time()
                res = http(base + "/v1/images/generations", {"prompt": prompt, "size": size, "seed": seed, "steps": steps})
                seconds.append(time.time() - t0)
                (out / f"{i:02d}_{seed}.png").write_bytes(base64.b64decode(res["data"][0]["b64_json"]))
        peak = http(base + "/props")["memory"]["peak_bytes"]
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=60)
        log.close()
    disk = sum(f.stat().st_size for f in pack.rglob("*.safetensors"))
    return {"disk_gb": disk / 1e9, "peak_gb": peak / 1e9, "median_s": float(np.median(seconds))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", action="append", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--port", type=int, default=11430)
    ap.add_argument("--work", default="~/claude-tmp/image-quant-eval")
    args = ap.parse_args()

    prompts = [p for p in (ROOT / "tests" / "fixtures" / "image_eval_prompts.txt").read_text().splitlines() if p.strip()]
    work = Path(args.work).expanduser()
    packs = [Path(args.ref).expanduser()] + [Path(c).expanduser() for c in args.cand]
    stats = {}
    for pack in packs:
        print(f"rendering {pack.name} ...", flush=True)
        stats[pack] = render(pack, prompts, args.seeds, args.steps, args.size, work / pack.name, args.port)

    ref = packs[0]
    print(f"\n| pack | disk GB | peak GB | s/image | PSNR mean | SSIM mean | SSIM min |\n|---|---|---|---|---|---|---|")
    for pack in packs:
        s = stats[pack]
        if pack == ref:
            print(f"| {pack.name} (reference) | {s['disk_gb']:.2f} | {s['peak_gb']:.2f} | {s['median_s']:.1f} | - | - | - |")
            continue
        ps, ss = [], []
        for i in range(len(prompts)):
            for seed in range(args.seeds):
                a = decode_png((work / ref.name / f"{i:02d}_{seed}.png").read_bytes())
                b = decode_png((work / pack.name / f"{i:02d}_{seed}.png").read_bytes())
                ps.append(psnr(a, b))
                ss.append(ssim(a, b))
        print(f"| {pack.name} | {s['disk_gb']:.2f} | {s['peak_gb']:.2f} | {s['median_s']:.1f} | "
              f"{np.mean(ps):.2f} | {np.mean(ss):.4f} | {np.min(ss):.4f} |")
    print(f"\nrenders: {work}")
    return 0


def self_test() -> int:
    failures = []

    def check(cond, label):
        print(("PASS " if cond else "FAIL ") + label)
        if not cond:
            failures.append(label)

    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (9, 7, 3), dtype=np.uint8)

    def encode(pixels, filt):
        h, w, ch = pixels.shape
        rows = []
        prev = np.zeros(w * ch, np.int32)
        for y in range(h):
            cur = pixels[y].reshape(-1).astype(np.int32)
            left = np.concatenate([np.zeros(ch, np.int32), cur[:-ch]])
            ul = np.concatenate([np.zeros(ch, np.int32), prev[:-ch]])
            if filt == 0:
                line = cur
            elif filt == 1:
                line = cur - left
            elif filt == 2:
                line = cur - prev
            elif filt == 3:
                line = cur - ((left + prev) >> 1)
            else:
                p = left + prev - ul
                pa, pb, pc = np.abs(p - left), np.abs(p - prev), np.abs(p - ul)
                line = cur - np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, prev, ul))
            rows.append(bytes([filt]) + (line & 0xFF).astype(np.uint8).tobytes())
            prev = cur

        def chunk(kind, payload):
            return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

        return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))

    for f in range(5):
        check(np.array_equal(decode_png(encode(img, f)), img), f"PNG filter {f} decodes exactly")
    big = rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)
    noisy = np.clip(big.astype(np.int32) + rng.integers(-40, 41, big.shape), 0, 255).astype(np.uint8)
    check(psnr(big, big) == float("inf") and abs(ssim(big, big) - 1.0) < 1e-9, "identical images score perfect")
    check(ssim(big, noisy) < 0.99 and psnr(big, noisy) < 30, "noise lowers both scores")
    print(f"\n{'OK' if not failures else 'FAILED'}: {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(self_test() if "--self-test" in sys.argv else main())

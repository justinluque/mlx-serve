#!/usr/bin/env python3
"""Calibrated, budget-allocated affine quantization shared by the image-pack
converters (tests/convert_krea.py, tests/convert_flux2.py).

Per tensor, `weighted_quant` searches each group's (scale, bias) to minimise
the reconstruction error weighted by the imatrix — the per-input-channel E[x^2]
the engine collects during real generations (MLX_SERVE_IMATRIX). MLX's own
min/max is candidate 0, so the weighted error is never worse than
mx.quantize's. It is the search of dsv4_imatrix.weighted_affine_quant, run on
the GPU at every width the engine reads.

Across tensors, `allocate` spends a byte budget on the largest cut in weighted
relative error per byte, over errors MEASURED at every candidate width. A
converter floors a linear's width (`apply_floors`) only where renders at the same
pack size showed it matters; everything else is read off the checkpoint and its
activations.

The output is mx.quantize's affine layout (U32 weights + BF16 scales/biases,
group 64): `flux.QLinear` and `krea.MixedLinear` solve it from geometry, and a
runtime LoRA still composes because the stored weight is the base weight.

    python3 tests/image_quant.py --self-test
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx

GROUP_SIZE = 64
# 3 is the floor: below it a DiT's output breaks up rather than softening.
WIDTHS = (3, 4, 5, 6, 8)
DENSE = 16
# Elements per search chunk: bounds one chunk's working set near a gigabyte.
CHUNK_ELEMENTS = 1 << 23


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ── sources ───────────────────────────────────────────────────────────────


def hf_snapshot(repo_id: str) -> Path | None:
    """The snapshot of `repo_id` already in the Hugging Face hub cache, or None."""
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = Path(os.environ.get("HF_HUB_CACHE", home / "hub"))
    repo = hub / f"models--{repo_id.replace('/', '--')}"
    ref = repo / "refs" / "main"
    if ref.is_file() and (repo / "snapshots" / ref.read_text().strip()).is_dir():
        return repo / "snapshots" / ref.read_text().strip()
    snaps = sorted((repo / "snapshots").glob("*")) if (repo / "snapshots").is_dir() else []
    return snaps[-1] if snaps else None


def resolve_src(src: str) -> Path:
    p = Path(src).expanduser()
    if p.exists():
        return p
    snap = hf_snapshot(src)
    if snap is None:
        sys.exit(f"{src}: not a local path and not in the Hugging Face cache (hf download {src})")
    return snap


class _Tensors(Mapping):
    """Every lookup is a fresh lazy array, so a tensor its caller evaluated is freed
    when the caller drops it; a dict of `mx.load` results keeps every one resident."""

    def __init__(self, index: dict[str, Path]):
        self.index = index

    def __getitem__(self, key: str) -> mx.array:
        return mx.load(str(self.index[key]))[key]

    def __iter__(self):
        return iter(self.index)

    def __len__(self) -> int:
        return len(self.index)


class Source:
    """Lazy tensors from safetensors files: `mx.load` maps a file without reading
    it, and a tensor's bytes are read only when it is evaluated."""

    def __init__(self, files: list[Path]):
        self.files = [Path(f) for f in files]
        self.tensors = _Tensors({k: f for f in self.files for k in mx.load(str(f))})
        if not self.tensors:
            sys.exit(f"no tensors in {[str(f) for f in self.files]}")

    @classmethod
    def dir(cls, d: Path) -> "Source":
        files = sorted(d.glob("*.safetensors"))
        if not files:
            sys.exit(f"{d}: no .safetensors files")
        return cls(files)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for f in self.files:
            st = f.stat()
            h.update(f"{f.name}:{st.st_size}:{int(st.st_mtime)}".encode())
        return h.hexdigest()[:16]


# ── imatrix ───────────────────────────────────────────────────────────────

_WRAPPERS = ("model.language_model.", "language_model.", "model.")


def module_key(component: str, module: str) -> str:
    """`<component>/<module>`, exactly as imatrix.moduleKey files the engine's
    statistics — a text encoder's wrapper prefix dropped."""
    for p in _WRAPPERS:
        if module.startswith(p):
            module = module[len(p):]
            break
    return f"{component}/{module}"


class Imatrix:
    """One component's channel weights from an engine-collected file, or none.

    A file that matches NOTHING in the component is a named refusal: a stale key
    convention would otherwise ship an uncalibrated pack recorded as calibrated.
    """

    def __init__(self, path: str | None, component: str):
        self.component = component
        self.path = Path(path).expanduser() if path else None
        self.table: dict[str, mx.array] = {}
        self.hit: set[str] = set()
        self.missed: set[str] = set()
        self.declined: dict[str, str] = {}
        self.digest = "none"
        if self.path:
            self.table = {k: v for k, v in mx.load(str(self.path)).items() if k.startswith(component + "/")}
            self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()[:16]
            log(f"[{component}] imatrix: {len(self.table)} entries from {self.path}")

    def weights(self, module: str, in_features: int) -> mx.array | None:
        if not self.path:
            return None
        key = module_key(self.component, module)
        w = self.table.get(key)
        if w is None:
            self.missed.add(key)
            return None
        if tuple(w.shape) != (in_features,):
            # A different checkpoint, not a bad channel.
            self.declined[key] = f"{tuple(w.shape)} vs in_features {in_features}"
            return None
        self.hit.add(key)
        return w.astype(mx.float32)

    def report(self) -> None:
        if not self.path:
            return
        log(f"[{self.component}] calibrated {len(self.hit)} linears, {len(self.missed)} uncalibrated, "
            f"{len(self.declined)} declined")
        for k, why in list(self.declined.items())[:5]:
            log(f"[{self.component}] declined {k}: {why}")
        if not self.hit:
            sys.exit(f"{self.component}: the imatrix matched no linear (first misses: "
                     f"{sorted(self.missed)[:3]}). Re-collect with MLX_SERVE_IMATRIX on this "
                     f"checkpoint rather than ship an uncalibrated pack that reads as calibrated.")


# ── quantization ──────────────────────────────────────────────────────────


def stored_bytes(out_dim: int, in_dim: int, bits: int, group_size: int = GROUP_SIZE) -> int:
    if bits == DENSE:
        return out_dim * in_dim * 2
    return out_dim * (in_dim * bits // 32) * 4 + 2 * out_dim * (in_dim // group_size) * 2


def widths_for(shape, group_size: int = GROUP_SIZE) -> tuple[int, ...]:
    """Candidate stored widths for a 2-D weight; DENSE only when groups cannot tile it."""
    if len(shape) != 2 or shape[1] % group_size:
        return (DENSE,)
    return WIDTHS + (DENSE,)


def pack(q: mx.array, bits: int) -> mx.array:
    """uint q [rows, in] -> mx.quantize's packed words: a dense little-endian bit
    stream, element i at bit i*bits, straddling words at 3/5/6 bits."""
    rows, n = q.shape
    words = n * bits // 32
    q = q.astype(mx.uint32)
    if 32 % bits == 0:
        per = 32 // bits
        shifts = mx.arange(per, dtype=mx.uint32) * bits
        return (q.reshape(rows, words, per) << shifts).sum(-1).astype(mx.uint32)
    bitv = (q[..., None] >> mx.arange(bits, dtype=mx.uint32)) & 1
    return (bitv.reshape(rows, words, 32) << mx.arange(32, dtype=mx.uint32)).sum(-1).astype(mx.uint32)


@dataclass
class Quantized:
    weight: mx.array | None
    scales: mx.array | None
    biases: mx.array | None
    rel_err: float


def weighted_quant(w: mx.array, bits: int, channel_weights: mx.array | None = None,
                   group_size: int = GROUP_SIZE, packed: bool = True,
                   nstep: int = 14, refine: int = 3) -> Quantized:
    """Activation-weighted affine quantization of a 2-D weight. `rel_err` is
    sum_j w_j (What - W)^2 / sum_j w_j W^2 under the ROUNDED bf16 scales — the
    error of exactly what is stored."""
    out_dim, in_dim = w.shape
    G = in_dim // group_size
    n_bins = float((1 << bits) - 1)
    om = mx.ones((in_dim,), mx.float32) if channel_weights is None else channel_weights.astype(mx.float32)
    # A relative floor keeps a never-activated channel from making the refit singular.
    W = (om + (mx.mean(om) * 1e-4 + 1e-30)).reshape(1, G, group_size)
    sw = W.sum(-1)
    step = max(1, CHUNK_ELEMENTS // in_dim)
    ws, ss, bs = [], [], []
    err = norm = 0.0
    for r in range(0, out_dim, step):
        X = w[r:r + step].astype(mx.float32).reshape(-1, G, group_size)
        s, b = _search(X, W, sw, n_bins, bits, group_size, nstep, refine)
        sf = s.astype(mx.bfloat16).astype(mx.float32)
        bf = b.astype(mx.bfloat16).astype(mx.float32)
        dead = (sf == 0) | ~mx.isfinite(sf) | ~mx.isfinite(bf)
        sf = mx.where(dead, 1.0, sf)
        bf = mx.where(dead, ((W * X).sum(-1) / sw).astype(mx.bfloat16).astype(mx.float32), bf)
        q = mx.clip(mx.round((X - bf[..., None]) / sf[..., None]), 0, n_bins)
        resid = sf[..., None] * q + bf[..., None] - X
        e, n = (W * resid * resid).sum(), (W * X * X).sum()
        if packed:
            ws.append(pack(q.reshape(X.shape[0], in_dim), bits))
            ss.append(sf.astype(mx.bfloat16))
            bs.append(bf.astype(mx.bfloat16))
            mx.eval(e, n, ws[-1], ss[-1], bs[-1])
        else:
            mx.eval(e, n)
        err += e.item()
        norm += n.item()
    rel = err / max(norm, 1e-30)
    if not packed:
        return Quantized(None, None, None, rel)
    return Quantized(mx.concatenate(ws), mx.concatenate(ss), mx.concatenate(bs), rel)


def _search(X, W, sw, n_bins, bits, group_size, nstep, refine):
    swx = (W * X).sum(-1)
    swx2 = (W * X * X).sum(-1)

    def sums(q):
        Wq = W * q
        return Wq.sum(-1), (Wq * q).sum(-1), (Wq * X).sum(-1)

    def objective(s, b, sl, sl2, sxl):
        return swx2 + s * s * sl2 + b * b * sw + 2 * s * b * sl - 2 * s * sxl - 2 * b * swx

    def label(s, b):
        return mx.clip(mx.round((X - b[..., None]) / s[..., None]), 0, n_bins)

    # Candidate 0: MLX's own min/max (s, b).
    rows = X.shape[0]
    _, s0, b0 = mx.quantize(X.reshape(rows, -1), group_size=group_size, bits=bits)
    best_s, best_b = s0.astype(mx.float32), b0.astype(mx.float32)
    sl, sl2, sxl = sums(label(best_s, best_b))
    best_e = objective(best_s, best_b, sl, sl2, sxl)

    def refit_take(sl, sl2, sxl):
        nonlocal best_s, best_b, best_e
        D = sw * sl2 - sl * sl
        ok = D > 0
        Ds = mx.where(ok, D, 1.0)
        s = (sw * sxl - swx * sl) / Ds
        b = (sl2 * swx - sl * sxl) / Ds
        e = objective(s, b, sl, sl2, sxl)
        upd = ok & (e < best_e)
        best_e = mx.where(upd, e, best_e)
        best_s = mx.where(upd, s, best_s)
        best_b = mx.where(upd, b, best_b)
        mx.eval(best_e, best_s, best_b)

    refit_take(sl, sl2, sxl)
    # Multi-start: label from a min-anchored grid at a spread of scales, refit each.
    xmin = X.min(-1)
    span = X.max(-1) - xmin
    span = mx.where(span > 0, span, 1.0)
    for k in range(nstep):
        iscale = (n_bins - 1.0 + 0.25 * k) / span
        q = mx.clip(mx.round((X - xmin[..., None]) * iscale[..., None]), 0, n_bins)
        refit_take(*sums(q))
    for _ in range(refine):
        refit_take(*sums(label(best_s, best_b)))
    return best_s, best_b


# ── allocation ────────────────────────────────────────────────────────────


@dataclass
class Plan:
    """One weight to store. `module` is its pack key without `.weight`."""

    module: str
    out_dim: int
    in_dim: int
    errors: dict[int, float] = field(default_factory=dict)
    width: int = 0

    def bytes_at(self, width: int) -> int:
        return stored_bytes(self.out_dim, self.in_dim, width)


SAMPLE_ROWS = 1024


def measure(plans: list[Plan], get, imatrix: Imatrix, cache: Path | None = None, fingerprint: str = "") -> None:
    """Fill every plan's weighted relative error at every candidate width.
    Cached against `fingerprint` (source + imatrix), because it is the slow half."""
    cached: dict = {}
    if cache and cache.is_file():
        blob = json.loads(cache.read_text())
        if blob.get("fingerprint") == fingerprint:
            cached = blob["errors"]
    for i, p in enumerate(plans):
        if p.module in cached:
            p.errors = {int(k): v for k, v in cached[p.module].items()}
            continue
        w = get(p.module)
        if p.out_dim > SAMPLE_ROWS:
            # The error is a mean over rows and concentrates; `emit` applies the
            # chosen width to every row, so only the choice is estimated.
            w = mx.take(w, mx.arange(0, p.out_dim, p.out_dim // SAMPLE_ROWS)[:SAMPLE_ROWS], axis=0)
        mx.eval(w)
        ch = imatrix.weights(p.module, p.in_dim)
        p.errors = {DENSE: 0.0}
        for bits in widths_for(w.shape):
            if bits != DENSE:
                p.errors[bits] = weighted_quant(w, bits, ch, packed=False).rel_err
        cached[p.module] = p.errors
        del w
        mx.clear_cache()
        if cache and (i % 16 == 15 or i == len(plans) - 1):
            cache.write_text(json.dumps({"fingerprint": fingerprint, "errors": cached}))
        log(f"  measured {i + 1}/{len(plans)} {p.module}: "
            + " ".join(f"{b}b={e:.2e}" for b, e in sorted(p.errors.items()) if b != DENSE))


def use_more_bits(layer: int, n_layers: int) -> bool:
    """llama.cpp's layers for a wider ffn_down (llama_tensor_get_type): the first and
    last eighth of the stack and every third layer between."""
    return layer < n_layers // 8 or layer >= 7 * n_layers // 8 or (layer - n_layers // 8) % 3 == 2


def apply_floors(plans: list[Plan], floor_for) -> None:
    """Drop every candidate width below `floor_for(module)` (None = no floor), so
    `allocate` starts the plan there. DENSE is always a candidate, so no plan empties."""
    for p in plans:
        floor = floor_for(p.module)
        if floor is not None:
            p.errors = {b: e for b, e in p.errors.items() if b >= floor}


def apply_ceilings(plans: list[Plan], ceil_for) -> None:
    """Drop every candidate width above `ceil_for(module)` (None = no ceiling): a
    loader that demands `.scales` cannot read DENSE, and a weight nothing reads is
    worth only the narrowest. A ceiling that leaves a plan no candidate refuses."""
    for p in plans:
        ceil = ceil_for(p.module)
        if ceil is None:
            continue
        kept = {b: e for b, e in p.errors.items() if b <= ceil}
        if not kept:
            sys.exit(f"{p.module}: no candidate width at or below {ceil} (candidates {sorted(p.errors)})")
        p.errors = kept


def allocate(plans: list[Plan], budget: int) -> int:
    """Choose every plan's width to minimise summed weighted relative error in
    `budget` bytes: start each at its narrowest candidate, then keep buying the
    upgrade with the largest error cut per byte that still fits. Returns bytes."""
    for p in plans:
        p.width = min(p.errors)
    spent = sum(p.bytes_at(p.width) for p in plans)
    if spent > budget:
        sys.exit(f"budget {budget / 1e9:.3f} GB is below the narrowest this component allows ({spent / 1e9:.3f} GB)")
    heap: list = []

    def offer(i: int) -> None:
        p = plans[i]
        best = None
        for nxt, e in p.errors.items():
            cost = p.bytes_at(nxt) - p.bytes_at(p.width)
            gain = p.errors[p.width] - e
            if nxt <= p.width or gain <= 0 or spent + cost > budget:
                continue
            if best is None or gain / cost > best[0]:
                best = (gain / cost, nxt)
        if best:
            heapq.heappush(heap, (-best[0], i, best[1]))

    for i in range(len(plans)):
        offer(i)
    while heap:
        _, i, nxt = heapq.heappop(heap)
        p = plans[i]
        cost = p.bytes_at(nxt) - p.bytes_at(p.width)
        if nxt > p.width and spent + cost <= budget:
            p.width = nxt
            spent += cost
        offer(i)
    return spent


def budget_for(plans: list[Plan], bits_per_weight: float) -> int:
    return int(sum(p.out_dim * p.in_dim for p in plans) * bits_per_weight / 8)


def summary(plans: list[Plan], spent: int, imatrix: Imatrix) -> dict:
    widths: dict[str, int] = {}
    for p in plans:
        name = "bf16" if p.width == DENSE else f"{p.width}bit"
        widths[name] = widths.get(name, 0) + 1
    params = sum(p.out_dim * p.in_dim for p in plans)
    return {
        "bits_per_weight": round(8 * spent / max(params, 1), 3),
        "gigabytes": round(spent / 1e9, 3),
        "widths": dict(sorted(widths.items())),
        "calibration": imatrix.digest,
        "summed_weighted_rel_err": round(sum(p.errors[p.width] for p in plans), 6),
    }


# ── output ────────────────────────────────────────────────────────────────


class ShardWriter:
    """Streams tensors into ~2 GB safetensors shards, so no conversion holds a
    whole component in memory. `index` names a weight-map json to write last."""

    def __init__(self, out_dir: Path, basename: str, index: str | None = None, shard_bytes: int = 2 << 30):
        self.out_dir = out_dir
        self.basename = basename
        self.index = index
        self.shard_bytes = shard_bytes
        self.pending: dict[str, mx.array] = {}
        self.size = 0
        self.total = 0
        self.weight_map: dict[str, str] = {}
        out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, name: str, arr: mx.array) -> None:
        mx.eval(arr)
        self.pending[name] = arr
        self.size += arr.nbytes
        if self.size >= self.shard_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        fn = f"{self.basename}-{len(set(self.weight_map.values())) + 1:05d}.safetensors"
        mx.save_safetensors(str(self.out_dir / fn), self.pending)
        for k in self.pending:
            self.weight_map[k] = fn
        self.total += self.size
        self.pending, self.size = {}, 0
        mx.clear_cache()

    def close(self) -> int:
        self.flush()
        if self.index:
            (self.out_dir / self.index).write_text(json.dumps(
                {"metadata": {"total_size": self.total}, "weight_map": self.weight_map}, indent=1))
        return self.total


def emit(writer: ShardWriter, plan: Plan, w: mx.array, ch: mx.array | None) -> None:
    if plan.width == DENSE:
        writer.add(f"{plan.module}.weight", w.astype(mx.bfloat16))
        return
    qz = weighted_quant(w, plan.width, ch)
    writer.add(f"{plan.module}.weight", qz.weight)
    writer.add(f"{plan.module}.scales", qz.scales)
    writer.add(f"{plan.module}.biases", qz.biases)


def convert(component: str, source: Source, rename, writer: ShardWriter, imatrix: Imatrix,
            bits_per_weight: float, work: Path, floor_for=None, ceil_for=None) -> dict:
    """One component. `rename(key)` is a tensor's pack key, or None to leave it
    out. Every kept 2-D `.weight` is a linear placed by `allocate`, no narrower than
    `floor_for(module)` and no wider than `ceil_for(module)` when given; everything
    else is stored as-is, floating tensors in bf16. Returns the config.json record."""
    names: dict[str, str] = {}
    for key in sorted(source.tensors):
        name = rename(key)
        if name is not None:
            names[name] = key
    plans = [Plan(n[: -len(".weight")], *source.tensors[k].shape)
             for n, k in names.items() if n.endswith(".weight") and source.tensors[k].ndim == 2]
    for p in plans:
        imatrix.weights(p.module, p.in_dim)
    imatrix.report()  # a file matching nothing refuses before the hours of measuring

    def get(module: str) -> mx.array:
        return source.tensors[names[module + ".weight"]]

    fingerprint = f"{source.fingerprint()}:{imatrix.digest}:{WIDTHS}:{GROUP_SIZE}:{SAMPLE_ROWS}"
    measure(plans, get, imatrix, work / f"{component}.errors.json", fingerprint)
    if floor_for:
        apply_floors(plans, floor_for)
    if ceil_for:
        apply_ceilings(plans, ceil_for)
    spent = allocate(plans, budget_for(plans, bits_per_weight))
    for i, p in enumerate(plans):
        emit(writer, p, get(p.module), imatrix.weights(p.module, p.in_dim))
        if i % 32 == 31:
            log(f"  [{component}] wrote {i + 1}/{len(plans)} linears")
    stored = {p.module + ".weight" for p in plans}
    for name, key in names.items():
        if name not in stored:
            t = source.tensors[key]
            writer.add(name, t.astype(mx.bfloat16) if mx.issubdtype(t.dtype, mx.floating) else t)
    writer.close()
    record = summary(plans, spent, imatrix)
    (work / f"{component}.allocation.json").write_text(json.dumps(
        {"summary": record, "widths": {p.module: p.width for p in plans}}, indent=1))
    log(f"[{component}] {json.dumps(record)}")
    return record


# ── self-test ─────────────────────────────────────────────────────────────


def self_test() -> int:
    failures: list[str] = []

    def check(cond: bool, label: str) -> None:
        print(("PASS " if cond else "FAIL ") + label)
        if not cond:
            failures.append(label)

    mx.random.seed(11)

    def werr(w, deq, om):
        return float((om * (deq - w) ** 2).sum() / (om * w ** 2).sum())

    def deq(qz, bits):
        # f32 scales: the stored VALUES, without bf16 arithmetic's rounding on top.
        return mx.dequantize(qz.weight, qz.scales.astype(mx.float32), qz.biases.astype(mx.float32),
                             group_size=GROUP_SIZE, bits=bits)

    for bits in (2,) + WIDTHS:
        q = mx.random.randint(0, 1 << bits, (4, 256)).astype(mx.uint32)
        ones, zeros = mx.ones((4, 4)), mx.zeros((4, 4))
        back = mx.dequantize(pack(q, bits), ones, zeros, group_size=GROUP_SIZE, bits=bits)
        check(mx.array_equal(back, q.astype(mx.float32)).item(), f"pack round-trips through mx.dequantize at {bits} bits")
        wq, s, b = mx.quantize(mx.random.normal((8, 128)).astype(mx.bfloat16), group_size=GROUP_SIZE, bits=bits)
        check(stored_bytes(8, 128, bits) == wq.nbytes + s.nbytes + b.nbytes, f"stored_bytes is the real triple size at {bits} bits")

    w = mx.random.normal((96, 512))
    om = mx.random.uniform(0.01, 10.0, (512,))
    for bits in WIDTHS:
        qz = weighted_quant(w, bits, om)
        got = werr(w, deq(qz, bits), om)
        check(abs(got - qz.rel_err) <= 1e-3 * got + 1e-8, f"reported error is the error of what is stored at {bits} bits ({got:.3e} vs {qz.rel_err:.3e})")
        check(weighted_quant(w, bits, om, packed=False).rel_err == qz.rel_err, f"the unpacked measurement matches the emitted one at {bits} bits")

    uni = mx.ones((512,))
    for bits in (3, 4):
        ours = werr(w, deq(weighted_quant(w, bits), bits), uni)
        wq, s, b = mx.quantize(w, group_size=GROUP_SIZE, bits=bits)
        theirs = werr(w, mx.dequantize(wq, s, b, group_size=GROUP_SIZE, bits=bits), uni)
        check(ours <= theirs * 1.0001, f"uniform weights: search <= mx.quantize at {bits} bits ({ours:.4e} vs {theirs:.4e})")

    skew = mx.full((512,), 0.01)
    hot = mx.array([int(i) for i in mx.random.permutation(512)[:32].tolist()])
    skew[hot] = 10.0
    for bits in (3, 4):
        aware = weighted_quant(w, bits, skew)
        blind = weighted_quant(w, bits)
        ea, eb = werr(w, deq(aware, bits), skew), werr(w, deq(blind, bits), skew)
        check(ea < eb * 0.98, f"skewed weights: aware beats weight-blind search at {bits} bits ({ea:.4e} vs {eb:.4e})")
        ha = float(((deq(aware, bits) - w)[:, hot] ** 2).sum())
        hb = float(((deq(blind, bits) - w)[:, hot] ** 2).sum())
        check(ha < hb, f"skewed weights: hot channels reconstruct better at {bits} bits")

    def plans():
        return [
            Plan("big", 64, 512, {3: 0.20, 4: 0.05, 5: 0.02, 6: 0.01, 8: 0.001, DENSE: 0.0}),
            # Nothing past 4 bits pays here, so "small" stops at 4 and the rest goes to "big".
            Plan("small", 4, 128, {3: 0.02, 4: 0.01, 5: 0.01, 6: 0.01, 8: 0.01, DENSE: 0.01}),
            Plan("untileable", 4, 100, {DENSE: 0.0}),
        ]

    ps = plans()
    floor = sum(p.bytes_at(min(p.errors)) for p in ps)
    small_up = ps[1].bytes_at(4) - ps[1].bytes_at(3)
    spent = allocate(ps, floor + small_up)
    check(spent <= floor + small_up and ps[1].width == 4 and ps[0].width == 3,
          "allocation buys the cheaper error cut per byte first (small 3->4 before big 3->4)")
    ps = plans()
    allocate(ps, floor + small_up + ps[0].bytes_at(4) - ps[0].bytes_at(3))
    check(ps[0].width == 4, "allocation spends what is left on the next best upgrade")
    ps = plans()
    allocate(ps, 10 ** 9)
    check([p.width for p in ps] == [DENSE, 4, DENSE],
          "an unbounded budget buys every upgrade that cuts error and none that does not")
    try:
        allocate(plans(), floor - 1)
        check(False, "a budget below the floor is refused")
    except SystemExit:
        check(True, "a budget below the floor is refused")

    for comp, mod, want in (
        ("text_encoder", "language_model.layers.3.mlp.down_proj", "text_encoder/layers.3.mlp.down_proj"),
        ("text_encoder", "model.layers.3.mlp.down_proj", "text_encoder/layers.3.mlp.down_proj"),
        ("text_encoder", "layers.3.mlp.down_proj", "text_encoder/layers.3.mlp.down_proj"),
        ("transformer", "blocks.0.attn.wq", "transformer/blocks.0.attn.wq"),
    ):
        check(module_key(comp, mod) == want, f"module_key mirrors imatrix.moduleKey: {mod}")

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t.safetensors"
        mx.save_safetensors(str(f), {f"t{i}": mx.random.normal((2048, 2048)) for i in range(6)})
        src = Source([f])
        mx.clear_cache()
        base = mx.get_active_memory()
        for k in sorted(src.tensors):
            w = src.tensors[k]
            mx.eval(mx.take(w, mx.arange(0, 2048, 16), axis=0))
            del w
            mx.clear_cache()
        held = (mx.get_active_memory() - base) / (2048 * 2048 * 4)
        check(held < 2, f"reading every source tensor keeps at most one resident ({held:.1f} held)")

    check([i for i in range(36) if use_more_bits(i, 36)] == [0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 31, 32, 33, 34, 35],
          "use_more_bits is llama.cpp's first eighth, last eighth and every third layer between")
    ps = plans()
    apply_floors(ps, {"big": DENSE, "small": 6}.get)
    check(sorted(ps[0].errors) == [DENSE] and min(ps[1].errors) == 6 and sorted(ps[2].errors) == [DENSE],
          "a floor drops every narrower candidate and leaves unfloored plans alone")
    allocate(ps, sum(p.bytes_at(min(p.errors)) for p in ps))
    check([p.width for p in ps] == [DENSE, 6, DENSE], "the allocator starts a floored plan at its floor")
    ps = plans()
    apply_ceilings(ps, {"big": 4, "small": min(WIDTHS)}.get)
    check(sorted(ps[0].errors) == [3, 4] and sorted(ps[1].errors) == [3] and sorted(ps[2].errors) == [DENSE],
          "a ceiling drops every wider candidate, DENSE included, and leaves unceilinged plans alone")
    allocate(ps, 10 ** 9)
    check([p.width for p in ps] == [4, 3, DENSE], "an unbounded budget stops a plan at its ceiling")
    try:
        apply_ceilings(plans(), {"untileable": 8}.get)
        check(False, "a ceiling that leaves a plan no candidate is refused")
    except SystemExit:
        check(True, "a ceiling that leaves a plan no candidate is refused")

    print(f"\n{'OK' if not failures else 'FAILED'}: {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    sys.exit(__doc__)

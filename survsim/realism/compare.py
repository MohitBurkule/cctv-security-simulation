"""Compare a synthetic frame set against real CCTV, statistic by statistic.

    python -m survsim.realism compare --real ~/data/real_cctv \
        --synthetic out/render_demo/render/cam_01 [--profiles cctv_day_720p,...] [--fid]

For each statistic prints the real distribution (median, p10-p90) and each
synthetic set's median with a marker: ``ok`` inside the real p10-p90 band,
``LOW``/``HIGH`` outside it. Temporal noise is computed when a directory holds
consecutive frames.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .stats import FrameStats, band, frame_stats, list_images, temporal_noise


def _sample(paths: list[Path], n: int) -> list[Path]:
    if len(paths) <= n:
        return paths
    return [paths[i] for i in np.linspace(0, len(paths) - 1, n).astype(int)]


def _sequences(paths: list[Path], k: int = 6, max_seq: int = 12) -> list[list[Path]]:
    by_dir: dict[Path, list[Path]] = defaultdict(list)
    for p in paths:
        by_dir[p.parent].append(p)
    seqs = []
    for v in by_dir.values():
        v = sorted(v)
        nums = [int(p.stem) for p in v if p.stem.isdigit()]
        if len(nums) >= 2 and len(nums) == len(v):  # numbered, i.e. a real sequence
            start = len(v) // 2  # skip any warm-up at the head
            seqs.append(v[start:start + k])
    return seqs[:max_seq]


def measure_set(paths: list[Path], transform: Callable[[np.ndarray, int], np.ndarray] | None = None,
                n: int = 60, seq_paths: list[Path] | None = None) -> dict:
    """Stats for a sample of ``paths``; ``transform(img, i)`` degrades on the fly."""
    rows = []
    for i, p in enumerate(_sample(paths, n)):
        img = cv2.imread(str(p))
        if img is None:
            continue
        if transform is not None:
            img = transform(img, i)
        rows.append(frame_stats(img).as_dict())
    out = {k: band(r[k] for r in rows) for k in FrameStats.keys()}
    tn = []
    for seq in _sequences(seq_paths if seq_paths is not None else paths):
        frames = [cv2.imread(str(p)) for p in seq]
        if transform is not None:
            frames = [transform(f, j) for j, f in enumerate(frames)]
        tn.append(temporal_noise(frames))
    out["temporal_noise"] = band(tn)
    out["_n"] = len(rows)
    return out


def verdict(real: dict, value: float) -> str:
    if not np.isfinite(value):
        return "  n/a"
    if value < real["p10"]:
        return "  LOW"
    if value > real["p90"]:
        return " HIGH"
    return "   ok"


def format_table(real: dict, synth: dict[str, dict]) -> str:
    keys = [k for k in real if not k.startswith("_")]
    head = f"{'stat':<15} {'real median':>11} {'real p10-p90':>18}"
    for name in synth:
        head += f" | {name[:18]:>18}"
    lines = [head, "-" * len(head)]
    for k in keys:
        r = real[k]
        line = f"{k:<15} {r['median']:>11.3f} {r['p10']:>8.3f}-{r['p90']:<9.3f}"
        for s in synth.values():
            v = s[k]["median"]
            line += f" | {v:>12.3f}{verdict(r, v)}"
        lines.append(line)
    inside = {n: sum(verdict(real[k], s[k]["median"]).strip() == "ok" for k in keys)
              for n, s in synth.items()}
    lines.append("in band: " + ", ".join(f"{n}={c}/{len(keys)}" for n, c in inside.items()))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m survsim.realism")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--real", type=Path, action="append", required=True)
    c.add_argument("--synthetic", type=Path, action="append", required=True)
    c.add_argument("--profiles", default="", help="comma list: degrade synthetic with each")
    c.add_argument("--n", type=int, default=60)
    c.add_argument("--fid", action="store_true", help="also FID via survsim.realism.fid")
    c.add_argument("--json", type=Path)
    a = ap.parse_args(argv)

    real_paths = [p for d in a.real for p in list_images(d.expanduser())]
    real = measure_set(real_paths, n=a.n)
    synth: dict[str, dict] = {}
    names = [d.name for d in a.synthetic]
    for d in a.synthetic:
        paths = list_images(d.expanduser())
        # two sets both called cam_01 (raw vs degraded) must not overwrite each other
        name = d.name if names.count(d.name) == 1 else f"{d.parent.name}/{d.name}"
        if name in synth:
            name = f"{name}#{len(synth)}"
        synth[name] = measure_set(paths, n=a.n)
        for prof in filter(None, a.profiles.split(",")):
            from ..sensors.degrade import apply_profile
            synth[f"{name}+{prof}"] = measure_set(
                paths, n=a.n, transform=lambda img, i, prof=prof: apply_profile(img, prof, seed=i))
    print(f"real n={real['_n']} from {', '.join(map(str, a.real))}")
    print(format_table(real, synth))
    if a.fid:
        from .fid import fid_report
        for d in a.synthetic:
            rep = fid_report(real_paths, list_images(d.expanduser()))
            print(f"FID {d.name}: {json.dumps(rep, default=str)}")
    if a.json:
        a.json.write_text(json.dumps({"real": real, "synthetic": synth}, indent=1))
    return 0

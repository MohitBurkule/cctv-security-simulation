"""Burn camera tampering into rendered frames, from the Director's camera_status.

    .venv/bin/python tools/apply_tamper_frames.py <set>/render <set>/gt/camera_status.jsonl <out_dir> --fps 5

The Unreal render has no tamper model: a covered camera still renders the
scene. GT already follows the live camera (principle 2; mask GT drops blind
frames, survsim.gt.mask_boxes.blind_frames), so the pixels must too before a
video is made. Frame NNNN.png is MOT frame NNNN+1, i.e. t = (NNNN + 1) / fps.
This is a post-process EMULATION: cover -> near-black occluder with a little
light leak, defocus -> Gaussian blur, spray -> milky haze; untampered frames
are copied unchanged.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def tamper_frame(img: np.ndarray, kind: str, seed: int = 0) -> np.ndarray:
    if kind == "cover":
        rng = np.random.default_rng(seed)
        leak = cv2.GaussianBlur(img, (0, 0), 25).astype(np.float32) * 0.08
        return np.clip(leak + rng.normal(6, 2, img.shape), 0, 255).astype(np.uint8)
    if kind == "defocus":
        return cv2.GaussianBlur(img, (0, 0), 7)
    if kind == "spray":
        return cv2.addWeighted(cv2.GaussianBlur(img, (0, 0), 12), 0.4,
                               np.full_like(img, 200), 0.6, 0)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("render_dir", type=Path)
    ap.add_argument("camera_status", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--fps", type=float, default=5.0)
    a = ap.parse_args()
    status: dict[tuple[str, int], str] = {}
    for line in a.camera_status.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("tamper"):
                status[(r["camera"], round(float(r["t"]) * a.fps))] = r["tamper"]
    n = 0
    for cam in sorted(p for p in a.render_dir.iterdir() if p.is_dir()):
        (a.out_dir / cam.name).mkdir(parents=True, exist_ok=True)
        for fp in sorted(cam.glob("*.png")):
            kind = status.get((cam.name, int(fp.stem) + 1))
            if kind:
                cv2.imwrite(str(a.out_dir / cam.name / fp.name),
                            tamper_frame(cv2.imread(str(fp)), kind, int(fp.stem)))
                n += 1
            else:
                shutil.copy2(fp, a.out_dir / cam.name / fp.name)
    print(f"tampered {n} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

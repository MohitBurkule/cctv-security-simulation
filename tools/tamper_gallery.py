#!/usr/bin/env python3
"""Build out/gallery/tamper_gallery.png: before/after stills for the 6 camera
tamper kinds in survsim/director/camera_director.py::TAMPER_KINDS, applied to
one real render frame (out/batch_v8/disguise_and_tamper_day).

`cover`, `spray`, `glare`, `unplug` and `defocus` are pixel effects; `tilt`
changes geometry, not pixels, so a real camera keeps a sharp image of the
wrong scene. Only `defocus` and a lens-dirt approximation of `spray` reuse
`survsim/sensors/degrade.py` primitives directly (marked [code]); `cover`,
`unplug`, `glare` and `tilt` are illustrative emulations built for this
gallery, not the engine's own tamper renderer, which changes camera state/GT
rather than pixels for most kinds (marked [emulated]).

Usage:
    source .venv/bin/activate
    python tools/tamper_gallery.py [--out-dir out/gallery]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from survsim.director.camera_director import TAMPER_KINDS
from survsim.sensors import degrade

FRAME = REPO / "out" / "batch_v8" / "disguise_and_tamper_day" / "render" / "cam_gate" / "0019.png"

NOTE = {
    "cover": "[emulated] black frame — physically obstructed lens",
    "spray": "[code: defocus+lens_dirt] paint smear over the lens",
    "tilt": "[emulated] geometry moves, pixels stay sharp — GT now wrong",
    "unplug": "[emulated] no signal",
    "glare": "[emulated] IR/sun glare blowout",
    "defocus": "[code: degrade.defocus] lens knocked out of focus",
}


def apply_tamper(img: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "cover":
        return np.zeros_like(img)
    if kind == "unplug":
        out = np.zeros_like(img)
        return out
    if kind == "defocus":
        return degrade.defocus(img, radius=3.0)
    if kind == "spray":
        blurred = degrade.defocus(img, radius=2.2)
        return degrade.lens_dirt(blurred, rng, amount=0.85)
    if kind == "glare":
        out = img.astype(np.float32)
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w * 0.35, h * 0.3
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        glare = np.exp(-r2 / (2 * (min(h, w) * 0.35) ** 2)) * 255.0
        out = np.clip(out + glare[..., None], 0, 255)
        return out.astype(np.uint8)
    if kind == "tilt":
        h, w = img.shape[:2]
        m = np.float32([[1, 0.35, -0.15 * w], [0.0, 0.8, 0.1 * h]])
        import cv2
        return cv2.warpAffine(img, m, (w, h), borderValue=(15, 15, 15))
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "out" / "gallery"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import cv2

    if not FRAME.exists():
        raise SystemExit(f"reference frame not found: {FRAME}")
    base = cv2.imread(str(FRAME))
    base = cv2.resize(base, (480, 270))
    rng = np.random.default_rng(0)

    rows = len(TAMPER_KINDS)
    cell_w, cell_h = 480 * 2 + 20, 270 + 40
    sheet = np.full((cell_h * rows, cell_w, 3), 12, dtype=np.uint8)

    for i, kind in enumerate(TAMPER_KINDS):
        after = apply_tamper(base, kind, rng)
        y0 = i * cell_h
        sheet[y0 + 30:y0 + 30 + 270, 0:480] = base
        sheet[y0 + 30:y0 + 30 + 270, 500:980] = after
        cv2.putText(sheet, f"{kind}: before -> after", (4, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(sheet, NOTE[kind], (990, y0 + 150), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (210, 210, 210), 1, cv2.LINE_AA)
        print(f"rendered tamper kind {kind}")

    sheet = sheet[:, :1300]
    png_path = out_dir / "tamper_gallery.png"
    cv2.imwrite(str(png_path), sheet)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()

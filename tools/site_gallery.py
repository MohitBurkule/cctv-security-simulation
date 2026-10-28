#!/usr/bin/env python3
"""Build out/gallery/site_gallery.{mp4,png}: the scene generator's variety —
6 site types x 3 seeds, plan view with buildings, entrances and camera frusta.

Usage:
    source .venv/bin/activate
    python tools/site_gallery.py [--out-dir out/gallery]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from survsim.generator.layout import SITE_TYPES, build_layout

TILE = 480
FPS = 10
HOLD_S = 3
SEEDS = (1, 2, 3)

SOLID_COLOUR = (90, 90, 90)
FLAT_COLOUR = (55, 65, 55)
ENTRANCE_COLOUR = (0, 210, 90)
CAM_COLOUR = (0, 190, 255)


def plot_layout(site_type: str, seed: int) -> np.ndarray:
    import cv2

    L = build_layout(site_type, seed)
    C = L.core_half
    B = C + 12.0
    img = np.full((TILE, TILE, 3), 20, dtype=np.uint8)
    scale = (TILE - 20) / (2 * B)

    def to_px(x, y):
        return int(TILE / 2 + x * scale), int(TILE / 2 - y * scale)

    for it in L.items:
        pts = np.array([to_px(x, y) for x, y in it.corners()], dtype=np.int32)
        colour = FLAT_COLOUR if not it.solid else SOLID_COLOUR
        cv2.fillPoly(img, [pts], colour)

    for e in L.entrances:
        cv2.drawMarker(img, to_px(e["x"], e["y"]), ENTRANCE_COLOUR, cv2.MARKER_TRIANGLE_UP, 10, 2)

    for cam in L.cameras:
        c = to_px(cam["x"], cam["y"])
        yaw = np.radians(cam.get("yaw", 0.0))
        hfov = np.radians(cam.get("hfov", 70.0)) / 2.0
        reach = 16.0 * scale
        for sign in (-1, 1):
            a = yaw + sign * hfov
            cv2.line(img, c, (int(c[0] + reach * np.cos(a)), int(c[1] - reach * np.sin(a))),
                     CAM_COLOUR, 1)
        cv2.circle(img, c, 4, CAM_COLOUR, -1)

    cv2.rectangle(img, (0, 0), (TILE - 1, 24), (0, 0, 0), -1)
    cv2.putText(img, f"{site_type}  seed={seed}", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"{len(L.items)} items  {len(L.cameras)} cams  {len(L.entrances)} entrances",
                (6, TILE - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "out" / "gallery"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = []
    labels = []
    for site in SITE_TYPES:
        for seed in SEEDS:
            tiles.append(plot_layout(site, seed))
            labels.append(f"{site}_s{seed}")
            print(f"plotted {site} seed={seed}")

    import cv2
    import imageio.v2 as imageio

    # video: hold each tile for HOLD_S seconds
    frames = []
    for tile in tiles:
        frames.extend([tile] * (HOLD_S * FPS))
    mp4_path = out_dir / "site_gallery.mp4"
    imageio.mimsave(mp4_path, [f[:, :, ::-1] for f in frames], fps=FPS)
    print(f"wrote {mp4_path} ({len(frames)} frames, {len(tiles)} sites)")

    cols = 6
    rows = -(-len(tiles) // cols)
    sheet = np.full((TILE * rows, TILE * cols, 3), 10, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        sheet[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = tile
    png_path = out_dir / "site_gallery.png"
    cv2.imwrite(str(png_path), sheet)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()

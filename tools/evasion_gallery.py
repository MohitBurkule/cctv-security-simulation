#!/usr/bin/env python3
"""Build out/gallery/evasion_gallery.png: a labelled table of all 52 evasion
techniques (survsim/director/evasion.py::BY_ID), grouped by category, with the
6 techniques actually pixel-rendered in H47 (docs/research/32 H47, figure
h47_examples.png) marked.

Usage:
    source .venv/bin/activate
    python tools/evasion_gallery.py [--out-dir out/gallery]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from survsim.director.evasion import ALL

# H47 pre-registration (docs/research/32_EXPERIMENTS_LOG.md, "H47 pre-registration"):
# renderable subset of 6 techniques actually rendered and measured against a
# zero-shot detector. dark_lowcontrast has no single evasion.py id (a generic
# near-black palette); the closest catalogue id is reversible_jacket (same
# "recolour to dark" mechanism), marked with a note.
RENDERED = {
    "hi_vis_worker": "H47 hivis",
    "hood_up": "H47 hood_up (colour proxy, not geometry)",
    "slow_deliberate": "H47 slow_deliberate",
    "crowd_cover": "H47 crowd_cover",
    "wall_hugging": "H47 wall_hugging",
    "reversible_jacket": "H47 dark_lowcontrast (closest id; generic dark palette, not this exact technique)",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "out" / "gallery"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image, ImageDraw, ImageFont

    def font(size, bold=False):
        names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
        for n in names:
            try:
                return ImageFont.truetype(n, size)
            except OSError:
                continue
        return ImageFont.load_default()

    by_cat: dict[str, list] = {}
    for t in ALL:
        by_cat.setdefault(t.category, []).append(t)

    row_h = 20
    header_h = 40
    n_rows = len(ALL) + len(by_cat)  # + a header row per category
    W = 1500
    H = header_h + n_rows * row_h + 60
    img = Image.new("RGB", (W, H), (14, 16, 20))
    d = ImageDraw.Draw(img)
    f_title = font(22, bold=True)
    f_cat = font(14, bold=True)
    f_row = font(12)
    f_note = font(11)

    d.text((16, 10), f"Evasion technique catalogue — {len(ALL)} techniques "
                      f"({len(RENDERED)} pixel-rendered in H47, green)", font=f_title,
           fill=(235, 235, 245))

    y = header_h
    rendered_count = 0
    for cat, items in by_cat.items():
        d.rectangle([0, y, W, y + row_h], fill=(35, 38, 46))
        d.text((16, y + 3), f"{cat}  ({len(items)})", font=f_cat, fill=(180, 200, 230))
        y += row_h
        for t in items:
            is_rendered = t.id in RENDERED
            bg = (20, 60, 30) if is_rendered else ((26, 26, 30) if items.index(t) % 2 else (18, 18, 22))
            d.rectangle([0, y, W, y + row_h], fill=bg)
            colour = (140, 255, 160) if is_rendered else (215, 215, 220)
            mark = "[RENDERED] " if is_rendered else ""
            protected = " (protected attribute)" if t.protected else ""
            d.text((32, y + 3), f"{mark}{t.id} — {t.label}{protected}", font=f_row, fill=colour)
            d.text((760, y + 3), f"benign_rate={t.benign_rate:g}", font=f_note, fill=(140, 140, 150))
            if is_rendered:
                d.text((900, y + 3), RENDERED[t.id], font=f_note, fill=(140, 255, 160))
                rendered_count += 1
            y += row_h

    d.text((16, y + 10), f"{rendered_count} of {len(ALL)} techniques pixel-rendered to date "
                          "(H47, docs/research/32 + 50; figures/h47_examples.png). The rest are "
                          "testable at cue level in mock (docs/50 §6) — pixel rendering is blocked "
                          "on MetaHuman wardrobe (CLAUDE.md).", font=f_note, fill=(160, 160, 170))

    png_path = out_dir / "evasion_gallery.png"
    img.save(png_path)
    print(f"wrote {png_path} ({len(ALL)} techniques, {rendered_count} marked rendered)")


if __name__ == "__main__":
    main()

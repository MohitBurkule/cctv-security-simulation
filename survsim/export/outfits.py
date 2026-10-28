"""Per-person street-wear outfits for the "painted clothing" MetaHuman material.

The offline MetaHuman bodies ship with at most one garment, so clothing is
painted onto the body by region (scripts/ue/survsim_clothing.py): shoes,
trousers / shorts / skirt, top with short or long sleeves, skin elsewhere. At
CCTV standoff (20-50 m, 40-200 px tall) that reads as clothing. This module
draws the palette - deterministic per person, weighted toward what people
actually wear on a street (dark trousers, jeans, greys, navy, black jackets),
with hi-vis tops available for work sites. Colours are linear RGB.

The result is also the appearance truth for ReID (written next to the render).
"""
from __future__ import annotations

import random

# (name, linear rgb, weight)
TOPS = [("black", (0.012, 0.012, 0.014), 16), ("navy", (0.015, 0.025, 0.08), 10),
        ("charcoal", (0.04, 0.04, 0.045), 10), ("mid_grey", (0.16, 0.16, 0.17), 8),
        ("white", (0.70, 0.70, 0.68), 8), ("olive", (0.06, 0.07, 0.03), 5),
        ("burgundy", (0.12, 0.015, 0.02), 4), ("beige", (0.40, 0.32, 0.22), 5),
        ("denim_light", (0.20, 0.28, 0.42), 3), ("red", (0.45, 0.03, 0.03), 3),
        ("royal_blue", (0.03, 0.08, 0.40), 3), ("green", (0.03, 0.18, 0.06), 2),
        ("mustard", (0.45, 0.30, 0.03), 2), ("pink", (0.60, 0.25, 0.30), 2)]
HIVIS = ("hivis_yellow", (0.75, 0.80, 0.02), 0)
BOTTOMS = [("jeans_blue", (0.04, 0.07, 0.15), 22), ("black", (0.012, 0.012, 0.014), 18),
           ("dark_grey", (0.035, 0.035, 0.04), 12), ("navy", (0.015, 0.02, 0.06), 8),
           ("khaki", (0.28, 0.22, 0.12), 6), ("light_jeans", (0.18, 0.25, 0.38), 5),
           ("grey", (0.14, 0.14, 0.15), 5), ("brown", (0.08, 0.045, 0.025), 3)]
SHOES = [("black", (0.01, 0.01, 0.01), 40), ("white", (0.65, 0.65, 0.63), 25),
         ("brown", (0.06, 0.03, 0.015), 15), ("grey", (0.12, 0.12, 0.13), 12),
         ("tan", (0.30, 0.18, 0.08), 8)]
# Skin tones (linear), light to dark.
SKIN = [(0.62, 0.42, 0.33), (0.55, 0.36, 0.26), (0.45, 0.28, 0.19), (0.33, 0.19, 0.12),
        (0.22, 0.12, 0.07), (0.13, 0.07, 0.04)]
# Hem heights as a fraction of stature: long trousers stop at the shoe,
# shorts above the knee, a skirt at the knee.
LEG_CUTS = [("trousers", 0.055, 80), ("shorts", 0.34, 8), ("skirt", 0.29, 12)]
HAIR = [(0.01, 0.008, 0.006), (0.03, 0.018, 0.01), (0.08, 0.045, 0.02), (0.20, 0.13, 0.06),
        (0.25, 0.25, 0.24)]
SLEEVES = [("long", 0.80, 72), ("short", 0.28, 28)]


def _pick(rng: random.Random, table):
    total = sum(w for *_, w in table)
    x = rng.uniform(0, total)
    for row in table:
        x -= row[-1]
        if x <= 0:
            return row
    return table[-1]


def outfit(uid: str, seed: int = 0, hivis_p: float = 0.0) -> dict:
    """Deterministic outfit for one person (same uid + seed -> same clothes)."""
    rng = random.Random(f"{seed}:outfit:{uid}")
    top = HIVIS if rng.random() < hivis_p else _pick(rng, TOPS)
    bottom = _pick(rng, BOTTOMS)
    shoes = _pick(rng, SHOES)
    leg = _pick(rng, LEG_CUTS)
    sleeve = _pick(rng, SLEEVES)

    def jit(c):
        j = rng.uniform(0.85, 1.15)
        return tuple(round(min(1.0, v * j), 4) for v in c)

    return {"top": top[0], "top_rgb": jit(top[1]), "bottom": bottom[0],
            "bottom_rgb": jit(bottom[1]), "shoes": shoes[0], "shoes_rgb": jit(shoes[1]),
            "leg_cut": leg[0], "hem_h": leg[1], "sleeve": sleeve[0], "sleeve_u": sleeve[1],
            "skin_rgb": jit(SKIN[rng.randrange(len(SKIN))]),
            "hair_rgb": jit(HAIR[rng.randrange(len(HAIR))]),
            "roughness": round(rng.uniform(0.6, 0.9), 3)}


def hivis_probability(site_type: str | None) -> float:
    """Work sites have hi-vis vests; elsewhere they are rare."""
    if site_type == "construction_site":
        return 0.6
    return 0.2 if site_type in ("perimeter_site", "car_park", "loading_dock", "utility_yard",
                                      "construction_site", "farm_yard") else 0.02


# ------------------------------------------------------------ appearance epochs
# A person whose appearance changes mid-run (outfit_change, disguise) gets one
# render palette per epoch (gt/identities.parquet). Epoch 0 is `outfit()`, so
# renders of unchanged people are identical to before; each later epoch starts
# from the previous one and applies what the GT says changed.
COVER_RGB = {"cap": (0.015, 0.02, 0.05), "beanie": (0.012, 0.012, 0.014),
             "hood": (0.03, 0.03, 0.035), "balaclava": (0.01, 0.01, 0.01),
             "mask": (0.01, 0.01, 0.012)}


def srgb_hex_to_linear(h: str) -> tuple[float, float, float] | None:
    """'#d8d8d0' -> linear RGB (the material's colour space); None if unparsable."""
    try:
        h = h.strip().lstrip("#")
        c = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    except (AttributeError, ValueError, IndexError):
        return None
    return tuple(round(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4, 4)
                 for v in c)


def _luma(c) -> float:
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _nearest(table, rgb) -> str:
    return min(table, key=lambda r: sum((a - b) ** 2 for a, b in zip(r[1], rgb)))[0]


def _visible_top(gt: dict | None):
    """The GT colour a camera sees on the torso: the outer layer if worn."""
    if not gt:
        return None
    cols = gt.get("colours") or {}
    return srgb_hex_to_linear(cols.get("outer")) if gt.get("outer") and cols.get("outer") \
        else srgb_hex_to_linear(cols.get("top")) if cols.get("top") else None


def epoch_outfits(uid: str, epochs: list[dict], base: dict, seed: int = 0) -> list[dict]:
    """One render palette per appearance epoch (manifest `appearance_epochs[uid]`).

    Epoch k >= 1 = epoch k-1 with: the top recoloured to the GT's visible top
    colour when the GT changed it (else a contrasting street colour, so the
    change always reaches pixels); the bottom recoloured if the GT bottom
    changed; and, when `face_visible` is false, a face-cover proxy
    (`face_cover`: headwear/hood colour painted over hair and head skin) with
    the hair recoloured to it. Each palette carries `epoch` and `frame_start`.
    """
    out = []
    prev, prev_gt = dict(base), None
    for ep in sorted(epochs, key=lambda e: e["epoch"]):
        gt = ep.get("gt_outfit")
        o = dict(prev)
        o.pop("face_cover", None)
        if ep["epoch"] > 0:
            rng = random.Random(f"{seed}:epoch{ep['epoch']}:{uid}")
            top = _visible_top(gt)
            if top is None or top == _visible_top(prev_gt) or abs(_luma(top) - _luma(prev["top_rgb"])) < 0.05:
                far = [t for t in TOPS if abs(_luma(t[1]) - _luma(prev["top_rgb"])) > 0.1]
                row = _pick(rng, far or TOPS)
                top, o["top_source"] = row[1], "contrast"
            else:
                o["top_source"] = "gt"
            o["top_rgb"], o["top"] = tuple(top), _nearest(TOPS, top)
            gcols, pcols = (gt or {}).get("colours") or {}, (prev_gt or {}).get("colours") or {}
            if gcols.get("bottom") and gcols.get("bottom") != pcols.get("bottom"):
                b = srgb_hex_to_linear(gcols["bottom"])
                if b:
                    o["bottom_rgb"], o["bottom"] = b, _nearest(BOTTOMS, b)
            o["sleeve"], o["sleeve_u"] = "long", 0.80   # a changed top: jacket/hoodie
        if not ep.get("face_visible", True):
            kind = (gt or {}).get("headwear") or "hood"
            if kind not in COVER_RGB:
                kind = "hood"
            o["face_cover"] = {"kind": kind, "rgb": COVER_RGB[kind]}
            o["hair_rgb"] = COVER_RGB[kind]
        o["epoch"], o["frame_start"] = int(ep["epoch"]), int(ep["frame_start"])
        o["face_visible"] = bool(ep.get("face_visible", True))
        out.append(o)
        prev, prev_gt = o, gt
    return out

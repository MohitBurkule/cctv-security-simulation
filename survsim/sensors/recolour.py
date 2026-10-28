"""Give rendered people different clothes, in image space.

The packaged build has no parametric wardrobe, so every spawned actor renders in
the same outfit. A dataset like that teaches a detector "person = this particular
mesh" and makes appearance-based re-identification meaningless, because every
identity looks the same.

Until a build with real wardrobe support exists, the practical substitute is to
recolour garments *after* rendering, using the engine's own instance mask to know
exactly which pixels belong to which person. This is a long-standing augmentation
in the re-identification literature, and it works because appearance models lean
heavily on colour.

**What this is and is not.** Recolouring changes hue and chroma while preserving
luminance, so folds, shadows and specular highlights survive and the garment still
reads as fabric. It does *not* change cut, drape or silhouette — a recoloured
t-shirt is not a recoloured coat. So it buys genuine colour diversity and genuine
appearance *changes* between epochs, but not shape diversity, and anything
studying garment geometry needs a real wardrobe. That limitation travels with the
output rather than being quietly forgotten.

Work is done in CIELAB: L carries the shading and is left alone, while a and b
carry the colour and are steered toward the target. Doing this in RGB or HSV
instead flattens the shading and produces the plastic look that gives synthetic
data away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

# Vertical bands of a standing person, as fractions of bounding-box height.
# Crude but stable: the alternative is a body-part segmentation model, which adds
# a dependency and its own errors for a gain we do not need here.
BODY_BANDS: dict[str, tuple[float, float]] = {
    "headwear": (0.00, 0.14),
    "top": (0.14, 0.52),
    "bottom": (0.52, 0.88),
    "shoes": (0.88, 1.00),
}


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    v = value.lstrip("#")
    r, g, b = int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)
    return (b, g, r)


def _target_ab(bgr: tuple[int, int, int]) -> tuple[float, float]:
    """The a/b chroma of a target colour, which is what we steer toward."""
    patch = np.uint8([[list(bgr)]])
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)[0, 0]
    return float(lab[1]), float(lab[2])


def person_mask(instance_mask: np.ndarray, colour_bgr: tuple[int, int, int],
                tolerance: int = 8) -> np.ndarray:
    """Pixels belonging to one actor, from the engine's instance mask.

    Background is black in UnrealCV masks, so a tolerance window around a dark
    actor colour would otherwise swallow the whole background and recolour the
    scene. Background is therefore excluded explicitly.
    """
    rgb = instance_mask[:, :, :3].astype(int)
    diff = np.abs(rgb - np.array(colour_bgr, dtype=int))
    match = np.all(diff <= tolerance, axis=-1)
    is_background = np.all(rgb == 0, axis=-1)
    return match & ~is_background


def band_mask(mask: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Restrict a person mask to a vertical band of their own bounding box."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros_like(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    height = max(y1 - y0, 1)
    top = y0 + int(band[0] * height)
    # Exclusive upper bound: an inclusive one makes adjacent bands share a row,
    # so a pixel would be recoloured twice by two different garments.
    bottom = y0 + int(band[1] * height)
    if band[1] >= 1.0:
        bottom = y1 + 1
    out = np.zeros_like(mask)
    out[top:bottom, :] = mask[top:bottom, :]
    return out


@dataclass
class RecolourSpec:
    """Target colours per garment slot for one person."""

    uid: str
    mask_colour: tuple[int, int, int]
    slots: dict[str, str] = field(default_factory=dict)   # slot -> "#rrggbb"
    strength: float = 0.85

    @classmethod
    def from_person(cls, person, mask_colour, strength: float = 0.85) -> RecolourSpec:
        """Take the generated wardrobe colours the Director already assigned."""
        colours = dict(getattr(person.outfit, "colours", {}) or {})
        slots = {}
        for slot in ("top", "outer", "bottom", "headwear"):
            value = colours.get(slot)
            if value:
                # The outer layer is what an observer sees, so it wins the torso.
                slots["top" if slot == "outer" else slot] = value
        return cls(uid=person.id, mask_colour=mask_colour, slots=slots, strength=strength)


def recolour_person(
    image: np.ndarray,
    instance_mask: np.ndarray,
    spec: RecolourSpec,
    tolerance: int = 8,
) -> tuple[np.ndarray, int]:
    """Recolour one person's garments. Returns (image, pixels changed)."""
    mask = person_mask(instance_mask, spec.mask_colour, tolerance)
    if not mask.any():
        return image, 0

    lab = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2LAB).astype(np.float32)
    alpha = float(np.clip(spec.strength, 0.0, 1.0))
    touched = np.zeros(mask.shape, dtype=bool)

    for slot, hex_colour in spec.slots.items():
        band = BODY_BANDS.get(slot)
        if band is None:
            continue
        sel = band_mask(mask, band) & ~touched
        if not sel.any():
            continue
        a_t, b_t = _target_ab(hex_to_bgr(hex_colour))
        # Blend chroma toward the target and leave L untouched, so shading,
        # folds and highlights survive and the result still looks like cloth.
        lab[:, :, 1][sel] = lab[:, :, 1][sel] * (1 - alpha) + a_t * alpha
        lab[:, :, 2][sel] = lab[:, :, 2][sel] * (1 - alpha) + b_t * alpha
        touched |= sel

    changed = int(touched.sum())
    if changed == 0:
        return image, 0

    # Composite only the recoloured pixels back. Returning the whole LAB round
    # trip would shift every untouched pixel by a quantisation step, which is a
    # silent, image-wide edit nobody asked for.
    converted = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    out = image.copy()
    out[:, :, :3][touched] = converted[touched]
    return out, changed


def recolour_frame(
    image: np.ndarray,
    instance_mask: np.ndarray,
    specs: Iterable[RecolourSpec],
    tolerance: int = 8,
) -> tuple[np.ndarray, dict[str, int]]:
    """Recolour every person in a frame. Returns (image, pixels changed per uid)."""
    out = image
    report: dict[str, int] = {}
    for spec in specs:
        out, n = recolour_person(out, instance_mask, spec, tolerance)
        report[spec.uid] = n
    return out, report


def diversity_report(specs: Sequence[RecolourSpec]) -> dict[str, Any]:
    """How much apparent variety the recolouring actually introduces.

    Counting distinct hex strings would overstate it — two near-identical greys
    are not two outfits. Distance in CIELAB chroma is closer to what a model and
    an operator can tell apart.
    """
    points: list[tuple[float, float]] = []
    for spec in specs:
        for hex_colour in spec.slots.values():
            points.append(_target_ab(hex_to_bgr(hex_colour)))
    if len(points) < 2:
        return {"n_garments": len(points), "mean_pairwise_chroma_distance": 0.0,
                "distinct_at_10": len(points)}

    arr = np.array(points)
    d = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
    iu = np.triu_indices(len(arr), k=1)
    # Greedy count of colours that are at least 10 LAB units apart — roughly the
    # threshold at which two garments read as different colours.
    kept: list[np.ndarray] = []
    for p in arr:
        if all(np.linalg.norm(p - q) >= 10.0 for q in kept):
            kept.append(p)
    return {
        "n_garments": len(points),
        "mean_pairwise_chroma_distance": round(float(d[iu].mean()), 2),
        "min_pairwise_chroma_distance": round(float(d[iu].min()), 2),
        "distinct_at_10": len(kept),
        "note": "image-space recolouring: hue and chroma only, garment shape unchanged",
    }

"""Per-actor placement checks from world bounds (pure Python, engine-free).

A spawned person must be upright (largest extent on Z), the requested height,
and standing on the ground — checked from the POSED actor's world bounds, never
from an index's height_m (that produced a 10x giant) or a bone name.
"""
from __future__ import annotations

from collections.abc import Sequence

HEIGHT_BAND_CM = (150.0, 200.0)
GROUND_TOL_CM = 2.0


def upright_axis(extent: Sequence[float]) -> str:
    return "xyz"[max(range(3), key=lambda i: extent[i])]


def fit(origin: Sequence[float], extent: Sequence[float], ground_z: float,
        target_height_cm: float, current_scale: float = 1.0) -> dict:
    """Corrections for an actor whose bounds are (origin, half-extent), in cm.

    Returns the new uniform scale and the Z shift that puts the lowest point on
    the ground after rescaling about the actor pivot (assumed at origin.z -
    extent.z + pivot offset; we rescale then re-measure in the engine, so the
    shift here is for the current bounds).
    """
    axis = upright_axis(extent)
    height = 2.0 * float(extent[2])
    scale = current_scale * (target_height_cm / height) if height > 1e-6 else current_scale
    dz = float(ground_z) - (float(origin[2]) - float(extent[2]))
    return {"upright_axis": axis, "height_cm": round(height, 2), "scale": scale,
            "dz": round(dz, 2)}


def violations(origin, extent, ground_z, band=HEIGHT_BAND_CM, tol=GROUND_TOL_CM) -> list[str]:
    out = []
    if upright_axis(extent) != "z":
        out.append(f"not upright (largest extent {upright_axis(extent)})")
    h = 2.0 * float(extent[2])
    if not band[0] <= h <= band[1]:
        out.append(f"height {h:.1f} cm outside {band}")
    gap = (float(origin[2]) - float(extent[2])) - float(ground_z)
    if abs(gap) > tol:
        out.append(f"feet {gap:+.1f} cm from ground")
    return out

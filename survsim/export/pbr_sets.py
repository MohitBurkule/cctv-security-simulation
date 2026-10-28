"""Real PBR texture sets and HDRI skies from the asset index (~/data/assets/index.json).

Host-side and pure: resolves which ambientCG set replaces which procedural
surface (scripts/ue/survsim_materials.py library role), with its tiling in
metres, and where the sun is in a Poly Haven HDRI. The editor job receives
the result as JSON (its Python has no numpy/yaml guarantees). A role with no
set on disk keeps the procedural material: the fallback is never a stub.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

INDEX = Path.home() / "data/assets/index.json"

# role -> (ambientCG set, tile size in metres = one texture repeat, tint)
ROLE_SETS: dict[str, tuple[str, float, tuple[float, float, float]]] = {
    # Tile = the set's physical size (ambientCG 1K sets cover ~1-2 m); an
    # index entry's own "tile_m" wins. [X] 3-4 m tiles showed the texture's
    # own light/dark patches as a checkerboard across the yard.
    "asphalt": ("Asphalt033", 2.0, (0.62, 0.66, 0.74)),   # neutralise the set's brown cast
    "setts": ("PavingStones151", 1.5, (1.0, 1.0, 1.0)),
    "slabs": ("Tiles141", 2.0, (0.75, 0.73, 0.70)),
    "light_slabs": ("Tiles141", 2.4, (0.95, 0.93, 0.9)),
    # [X] The "checker" was never the tiling: yard pads (kerb, concrete) lie on
    # the main ground (proved with a 1 m debug checker on GroundMain: not
    # visible) and got different sets/tints. One set + tint for all three.
    "concrete": ("Concrete048", 2.0, (0.88, 0.8, 0.68)),
    "gravel": ("Concrete048", 2.0, (0.88, 0.8, 0.68)),      # yard ground: weathered concrete
    "kerb": ("Concrete048", 1.5, (0.8, 0.76, 0.68)),
    "wall_brick": ("Bricks105", 1.5, (0.85, 0.8, 0.78)),
    "metal_dark": ("Metal063", 1.0, (0.25, 0.25, 0.25)),
}


def _maps(root: Path, entry: dict[str, Any]) -> dict[str, str] | None:
    d = root / entry["path"]
    name = entry["name"]
    files = {"albedo": f"{name}_1K-JPG_Color.jpg",
             # UE's tangent space is DirectX-handed: use the DX normal map.
             "normal": f"{name}_1K-JPG_NormalDX.jpg",
             "roughness": f"{name}_1K-JPG_Roughness.jpg"}
    out = {k: str(d / v) for k, v in files.items()}
    return out if all(Path(p).is_file() for p in out.values()) else None


def pbr_library(index_path: Path = INDEX) -> dict[str, dict[str, Any]]:
    """{role: {"set", "maps": {albedo, normal, roughness}, "tile_m", "aspect", "tint"}}."""
    if not Path(index_path).is_file():
        return {}
    idx = json.loads(Path(index_path).read_text())
    root = Path(idx.get("root") or Path(index_path).parent)
    sets = {a["name"]: a for a in idx.get("assets", []) if a.get("kind") == "texture"}
    out = {}
    for role, (name, tile, tint) in ROLE_SETS.items():
        entry = sets.get(name)
        maps = _maps(root, entry) if entry else None
        if not maps:
            continue
        w, h = entry.get("resolution", [1024, 1024])
        # A 2:1 texture covers tile x tile/2 metres: without the aspect the
        # bricks would be stretched to double height.
        out[role] = {"set": name, "maps": maps, "tile_m": float(entry.get("tile_m", tile)),
                     "aspect": w / h, "tint": tint}
    return out


def hdri_list(index_path: Path = INDEX) -> list[dict[str, Any]]:
    if not Path(index_path).is_file():
        return []
    idx = json.loads(Path(index_path).read_text())
    root = Path(idx.get("root") or Path(index_path).parent)
    return [dict(a, file=str(root / a["path"])) for a in idx.get("assets", [])
            if a.get("kind") == "hdri" and (root / a["path"]).is_file()]


def pick_hdri(seed: int, index_path: Path = INDEX) -> dict[str, Any] | None:
    hs = sorted(hdri_list(index_path), key=lambda a: a["name"])
    return hs[random.Random(f"hdri:{seed}").randrange(len(hs))] if hs else None


def sun_from_equirect(img) -> dict[str, float]:
    """Brightest direction of an equirectangular HDR (H x W x 3, linear).

    Returns azimuth (deg, 0 at the image centre column, increasing with u -
    the long-lat convention the SkyLight cubemap import uses), elevation
    (deg above the horizon) and the ratio of the peak to the median luminance
    (a clear sun gives >> 100; an overcast sky ~ 3-10, where the sun direction
    means little).
    """
    import numpy as np
    lum = img[..., :3].astype(np.float64).mean(axis=2)
    h, w = lum.shape
    # Smooth a little so one hot pixel does not win.
    k = max(1, w // 256)
    if k > 1:
        lum = lum[: h - h % k, : w - w % k].reshape(h // k, k, w // k, k).mean(axis=(1, 3))
        h, w = lum.shape
    upper = lum[: h // 2]
    v, u = np.unravel_index(int(np.argmax(upper)), upper.shape)
    az = ((u + 0.5) / w - 0.5) * 360.0
    el = 90.0 - (v + 0.5) / h * 180.0
    return {"azimuth_deg": round(float(az), 2), "elevation_deg": round(float(el), 2),
            "peak_ratio": round(float(upper.max() / max(np.median(lum), 1e-9)), 1)}


def sun_for_hdri(path: str) -> dict[str, float]:
    import cv2
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return sun_from_equirect(img[..., ::-1])


def ue_sun_rotation(sun: dict[str, float]) -> tuple[float, float]:
    """(yaw, pitch) of a DirectionalLight shining FROM that direction.

    The light points away from the sun (yaw + 180) and down (pitch = -elev,
    at least 5 deg so the scene is never lit from the horizon).
    [A] The azimuth-to-yaw offset assumes the long-lat cubemap's centre column
    faces +X; not yet checked against a render's shadows.
    """
    yaw = (sun["azimuth_deg"] + 180.0) % 360.0
    return round(yaw, 2), round(-max(5.0, sun["elevation_deg"]), 2)

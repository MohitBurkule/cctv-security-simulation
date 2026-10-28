"""survsim.export.pbr_sets: texture-set resolution and HDRI sun direction."""
import json

import numpy as np

from survsim.export import pbr_sets as P


def _index(tmp_path, names=("Asphalt033", "Bricks105"), res=(1024, 512)):
    root = tmp_path / "assets"
    assets = []
    for n in names:
        d = root / "textures" / n
        d.mkdir(parents=True)
        for m in ("Color", "NormalDX", "Roughness"):
            (d / f"{n}_1K-JPG_{m}.jpg").write_bytes(b"x")
        assets.append({"kind": "texture", "name": n, "path": f"textures/{n}", "resolution": list(res)})
    (root / "sky.hdr").write_bytes(b"x")
    assets.append({"kind": "hdri", "name": "sky", "path": "sky.hdr"})
    idx = root / "index.json"
    idx.write_text(json.dumps({"root": str(root), "assets": assets}))
    return idx


def test_roles_resolve_only_when_maps_exist(tmp_path):
    lib = P.pbr_library(_index(tmp_path))
    assert set(lib) == {"asphalt", "wall_brick"}           # others fall back to procedural
    assert lib["wall_brick"]["aspect"] == 2.0
    assert lib["asphalt"]["maps"]["normal"].endswith("NormalDX.jpg")


def test_missing_index_is_empty(tmp_path):
    assert P.pbr_library(tmp_path / "nope.json") == {} and P.hdri_list(tmp_path / "nope.json") == []


def test_hdri_pick_is_seeded(tmp_path):
    idx = _index(tmp_path)
    assert P.pick_hdri(3, idx)["name"] == "sky"


def test_sun_is_the_brightest_upper_direction():
    img = np.full((256, 512, 3), 0.5, np.float32)
    img[64, 384] = 5000.0                     # u=0.75 -> +90 deg, v=0.25 -> 45 deg up
    img[200, 100] = 9000.0                    # brighter but below the horizon: ignored
    s = P.sun_from_equirect(img)
    assert abs(s["azimuth_deg"] - 90) < 2 and abs(s["elevation_deg"] - 45) < 2
    assert s["peak_ratio"] > 100


def test_sun_rotation_points_away_and_down():
    yaw, pitch = P.ue_sun_rotation({"azimuth_deg": 90.0, "elevation_deg": 2.0})
    assert yaw == 270.0 and pitch == -5.0

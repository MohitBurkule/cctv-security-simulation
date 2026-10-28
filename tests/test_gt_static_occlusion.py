"""Regression tests for docs/BUG_gt_visibility_fences.md: GT visibility ignored
fences (site dressing absent from the Director's occluders; CustomStencil ignores
static geometry), and edge slivers counted as usable targets."""
from pathlib import Path

import numpy as np

from survsim.core.geometry import MIN_USABLE_PX_HEIGHT, box_usable
from survsim.gt.static_occlusion import (
    SS,
    correct_rows,
    depth_map,
    filter_idmap,
    occluder_prims,
    static_cover,
)

RES = (640, 360)
K = [[500.0, 0, 320.0], [0, 500.0, 180.0], [0, 0, 1]]
# Camera at (0, 0, 1.5) m looking along world +x, level.
R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
T = -R @ np.array([0, 0, 1.5])


def _bars(shape, period=4, on=3, depth=2.0):
    z = np.full(shape, np.inf, np.float32)
    for k in range(on):
        z[:, k::period] = depth
    return z


def test_person_behind_bar_pattern_has_low_visibility():
    z = _bars((RES[1] * SS, RES[0] * SS))             # 75 % of columns covered at 2 m
    row = [[5, 1, 300.0, 150.0, 12.0, 34.0, 1, 1, 1.0]]
    rec = correct_rows(row, z, {(5, 1): 6.0}, RES)[0]
    assert rec["vis"] < 0.5
    assert rec["occluder"] == "static"
    assert rec["usable"]


def test_person_in_front_of_bars_is_not_occluded():
    z = _bars((RES[1] * SS, RES[0] * SS), depth=8.0)
    rec = correct_rows([[5, 1, 300.0, 150.0, 12.0, 34.0, 1, 1, 1.0]], z, {(5, 1): 6.0}, RES)[0]
    assert rec["static_cover"] == 0.0 and rec["vis"] == 1.0 and rec["occluder"] == "none"


def test_person_occluder_is_not_relabelled_static():
    z = np.full((RES[1] * SS, RES[0] * SS), np.inf, np.float32)
    rec = correct_rows([[5, 1, 300.0, 150.0, 12.0, 34.0, 1, 1, 0.3]], z, {(5, 1): 6.0}, RES)[0]
    assert rec["occluder"] == "person_or_edge" and rec["vis"] == 0.3


def test_five_px_edge_sliver_is_not_usable():
    z = np.full((RES[1] * SS, RES[0] * SS), np.inf, np.float32)
    # AUDIT_H69 site 05: 34 px tall, 3.5-6 px wide at the right image edge.
    rec = correct_rows([[25, 1, 635.0, 145.0, 12.0, 34.0, 1, 1, 0.5]], z, {}, RES)[0]
    assert rec["usable"] is False
    assert not box_usable(5.0, 34.0)
    assert box_usable(12.0, 34.0)
    # A full-body 20-24 px person is ~6-7 px wide and must stay usable (principle 5
    # keeps the one height threshold; width only rejects slivers).
    assert box_usable(7.1, 24.0)
    assert not box_usable(12.0, MIN_USABLE_PX_HEIGHT - 1)


def test_palisade_projection_occludes_only_people_behind_it():
    # UE cm: palisade 5 m ahead, spanning 4 m across the view (yaw 90), 2.4 m tall.
    items = [{"kind": "fence_run", "style": "palisade", "x": 500.0, "y": 0.0, "yaw": 90.0,
              "size": [400.0, 10.0, 240.0]}]
    z = depth_map(K, R, T, RES, occluder_prims(items))
    assert np.isfinite(z).any()
    bbox = (301.5, 167.5, 37.0, 106.0)                 # 1.7 m person at 8 m
    behind = static_cover(z, bbox, 8.0)
    assert 0.2 < behind < 0.6                          # 7 cm pales every 20 cm + rail
    assert static_cover(z, bbox, 3.0) == 0.0           # same box, in front of the fence


def test_filter_idmap_removes_pixels_behind_static_occluder():
    idmap = np.zeros((RES[1], RES[0]), np.int32)
    idmap[150:184, 300:312] = 7
    z = np.full((RES[1] * SS, RES[0] * SS), np.inf, np.float32)
    z[:, 304 * SS:308 * SS] = 2.0                      # a 4 px post at 2 m
    out = filter_idmap(idmap, z, {7: 6.0})
    assert (out == 7).sum() == 34 * 8
    assert (out[:, 304:308] == 0).all()
    assert (filter_idmap(idmap, z, {7: 1.0}) == 7).sum() == 34 * 12


def test_id_mask_material_depth_tests_static_geometry():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "ue" / "survsim_scene.py").read_text()
    assert "ID_MASK_DEPTH_TEST = True" in src
    assert "PPI_CUSTOM_DEPTH" in src and "PPI_SCENE_DEPTH" in src

"""Static fence cover must not be applied twice to a depth-tested render visibility.

docs/BUG_night_visibility_zero.md: MOT column 9 is now the render visibility
(mask pixels / expected). A depth-tested ID mask already lost the pixels a
fence covers, so vis x (1 - cover) would double count; only masks rendered
without ID_MASK_DEPTH_TEST (key absent -> assume not tested) take the product.
"""
import json

import numpy as np

from survsim.gt.static_occlusion import correct_rows, id_mask_depth_tested


def _half_fenced():
    zbuf = np.full((40, 40), np.inf, np.float32)   # SS=1 test grid
    zbuf[:, :20] = 2.0                             # fence at 2 m covers the left half
    row = [[1, 7, 0.0, 0.0, 40.0, 40.0, 1, 1, 0.8]]
    return row, zbuf, {(1, 7): 5.0}


def test_not_depth_tested_multiplies():
    row, zbuf, dep = _half_fenced()
    r = correct_rows(row, zbuf, dep, (40, 40), ss=1)[0]
    assert 0.3 < r["static_cover"] < 0.7
    assert r["vis"] == round(0.8 * (1 - r["static_cover"]), 3)


def test_depth_tested_keeps_render_visibility():
    row, zbuf, dep = _half_fenced()
    r = correct_rows(row, zbuf, dep, (40, 40), ss=1, depth_tested=True)[0]
    assert 0.3 < r["static_cover"] < 0.7     # still reported
    assert r["vis"] == 0.8                   # not 0.4: no double count
    assert r["occluder"] == "static"


def test_depth_tested_flag_from_render_result(tmp_path):
    assert id_mask_depth_tested(tmp_path) is False                 # no file
    (tmp_path / "render_result.json").write_text(json.dumps({"frames": 3}))
    assert id_mask_depth_tested(tmp_path) is False                 # key absent
    (tmp_path / "render_result.json").write_text(json.dumps({"id_mask_depth_test": True}))
    assert id_mask_depth_tested(tmp_path) is True

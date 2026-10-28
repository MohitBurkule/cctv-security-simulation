"""Regression: MOT column 9 must follow the rendered mask, not the mock painter.

docs/BUG_night_visibility_zero.md: showcase_all_night/dusk cam_01 had 459 rows
at visibility 0.000 (the mock painted a fence as an opaque L x L box) while
456 of them had rendered ID-mask pixels and the frame showed the person."""
from pathlib import Path

import pandas as pd

from survsim.gt.amodal import render_visibility_lines, write_render_visibility


def test_mask_visible_row_is_not_zero():
    lines = ["97,3,16.00,143.00,38.00,56.00,1,1,0.000",   # mock: hidden; mask: 563 px
             "4,10,636.24,98.10,3.76,44.38,0,1,0.000",    # no mask pixels: stays ignore
             "5,6,386.00,68.00,5.00,15.00,1,1,1.000"]     # no render vis: kept
    out, n = render_visibility_lines(lines, {(97, 3): 0.8239, (4, 10): 0.9})
    assert out[0].endswith(",0.824") and n == 1
    assert out[1].endswith(",0.000")      # conf 0 row is not a visible person
    assert out[2] == lines[2]


def test_write_render_visibility(tmp_path: Path):
    (tmp_path / "mot_cam_01.txt").write_text("1,1,0,0,10,30,1,1,0.000\n1,2,20,0,10,30,1,1,0.500\n")
    df = pd.DataFrame({"frame": [1, 1], "track_id": [1, 2], "camera": ["cam_01"] * 2,
                       "mask_bbox": [[0, 0, 10, 30], None], "visibility": [0.95, 0.5]})
    assert write_render_visibility(tmp_path, df) == {"cam_01": 1}
    rows = (tmp_path / "mot_cam_01.txt").read_text().splitlines()
    assert rows == ["1,1,0,0,10,30,1,1,0.950", "1,2,20,0,10,30,1,1,0.500"]

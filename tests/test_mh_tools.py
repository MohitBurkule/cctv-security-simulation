"""MetaHuman route helpers (tools/mh_*.py, scripts/ue/mh_*.py). No engine needed."""
import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cmp_ = _load("mh_detect_compare", ROOT / "tools/mh_detect_compare.py")


def test_iou_basic():
    assert cmp_.iou([0, 0, 10, 10], [0, 0, 10, 10]) > 0.999
    assert cmp_.iou([0, 0, 10, 10], [20, 20, 5, 5]) == 0.0
    assert abs(cmp_.iou([0, 0, 10, 10], [5, 0, 10, 10]) - 50 / 150) < 1e-6


def test_mask_box_measures_the_rendered_person():
    plate = np.full((120, 160, 3), 100, np.uint8)
    frame = plate.copy()
    frame[30:90, 70:85] = 200            # a "person" 60 px tall
    frame[5, 5] = 255                    # a stray pixel must not win
    x, y, w, h = cmp_.mask_box(frame, plate)
    assert (x, y, w, h) == (70, 30, 15, 60)


def test_mask_box_ignores_the_cast_shadow():
    # Regression: the person's long cast shadow joined the diff blob, the box
    # became 3x too wide and real 0.8-confidence detections scored as misses.
    plate = np.zeros((120, 160, 3), np.uint8)
    plate[...] = (90, 120, 150)
    frame = plate.copy()
    frame[80:95, 20:140] = (45, 60, 75)          # shadow: same chroma, half as bright
    frame[30:90, 70:85] = (40, 200, 60)          # person (in front of it): other colour
    x, y, w, h = cmp_.mask_box(frame, plate)
    assert (x, y, w, h) == (70, 30, 15, 60)


def test_mask_box_empty_is_none():
    plate = np.zeros((50, 50, 3), np.uint8)
    assert cmp_.mask_box(plate.copy(), plate) is None


def test_summary_scores_misses_as_zero():
    s = cmp_.summarise([0.0, 0.5, 0.9])
    assert s["n"] == 3 and s["recall@0.25"] == round(2 / 3, 3) and s["min"] == 0.0


def test_mh_runner_rewrites_job_dir():
    src = (ROOT / "scripts/ue/job_runner.py").read_text()
    assert 'JOB_DIR = "/tmp/survsim_jobs"' in src, "mh_job_runner.py substitutes this exact line"


def test_mh_editor_defaults_to_nullrhi():
    # Two GPU editors do not fit the 8 GB card; the sandbox must not render by default.
    src = (ROOT / "tools/mh_editor.py").read_text()
    assert '"-nullrhi"' in src and 'MH_GPU' in src


def test_mh_scripts_parse_and_write_results():
    for p in (ROOT / "scripts/ue").glob("mh_*.py"):
        tree = ast.parse(p.read_text())
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        if p.name != "mh_job_runner.py":
            assert "RESULT" in names or "finish" in names, f"{p.name} must report via RESULT/finish"


def test_bake_refuses_unknown_body_constraint():
    # A mistyped constraint name silently keeps the default in the engine API
    # (docs/12); the bake must raise instead.
    src = (ROOT / "scripts/ue/mh_bake.py").read_text()
    assert "unknown body constraints" in src


lineup = _load("mh_lineup", ROOT / "tools/mh_lineup.py")


def test_hue_error_is_luminance_invariant_and_catches_wrong_hue():
    assert lineup.hue_error((0.4, 0.02, 0.02), (0.08, 0.004, 0.004)) < 0.02   # darker red
    assert lineup.hue_error((0.03, 0.18, 0.06), (0.12, 0.30, 0.15)) < 0.15    # desaturated green
    assert lineup.hue_error((0.4, 0.02, 0.02), (0.02, 0.02, 0.4)) > 0.5        # red vs blue
    assert lineup.hue_error((0.5, 0.5, 0.5), (0.5, 0.1, 0.1)) > 0.5           # grey vs red


def test_project_centre_ray_hits_image_centre():
    cam = {"loc": [0, 0, 300], "pitch": -30.0, "yaw": 45.0, "fov": 60.0, "res": [1280, 720]}
    import math
    d = 1000.0
    p = (d * math.cos(math.radians(-30)) * math.cos(math.radians(45)),
         d * math.cos(math.radians(-30)) * math.sin(math.radians(45)), 300 + d * math.sin(math.radians(-30)))
    u, v = lineup.project(cam, p)
    assert abs(u - 640) < 1e-6 and abs(v - 360) < 1e-6

"""Amodal GT boxes (survsim.gt.amodal) and the rendered-height gate check."""
import json

import numpy as np
import pandas as pd

from survsim.checks.render import check_height_files
from survsim.gt import amodal as A

K = [[800.0, 0, 640], [0, 800.0, 360], [0, 0, 1]]
# Camera at (0, -10, 1.5) m looking along +y (world z up): x_cam = x, y_cam = -z, z_cam = y.
R = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
T = list(-np.asarray(R) @ np.array([0.0, -10.0, 1.5]))


def test_envelope_spans_feet_to_crown():
    b = A.envelope_box(K, R, T, (0.0, 0.0), 1.8)
    x, y, w, h = b
    # crown at z=1.8 -> v = 360 - 800*(0.3)/depth; feet at z=0 -> 360 + 800*1.5/depth
    assert abs((y + h) - (360 + 800 * 1.5 / (10 - A.R_FRAC * 1.8))) < 1.0
    assert abs(x + w / 2 - 640) < 1e-6
    assert A.envelope_box(K, R, T, (0.0, -20.0), 1.8) is None      # behind the camera


def test_offsets_roundtrip_and_calibration():
    env = (100.0, 50.0, 40.0, 120.0)
    off = np.array([0.02, -0.01, -0.03, 0.015])
    mask = A.apply_offsets(env, off)
    assert np.allclose(A.edge_offsets(env, mask), off)
    s = [{"env": env, "mask": mask, "pixels": 0.4 * mask[2] * mask[3]}] * 6
    s.append({"env": env, "mask": (mask[0], mask[1] + 60, mask[2], mask[3] - 60), "pixels": 10})
    c = A.calibrate(s)                  # the half-hidden sample is dropped by the height test
    assert np.allclose(c["offsets"], off) and abs(c["fill"] - 0.4) < 1e-9


def test_containment_and_clean():
    assert A.containment((0, 0, 10, 10), (0, 0, 20, 20)) == 1.0
    assert A.containment((0, 0, 10, 10), (5, 0, 20, 20)) == 0.5
    env = (100, 100, 30, 90)
    assert A.is_clean((100, 100, 30, 90), [], (1280, 720), env)
    assert not A.is_clean((100, 100, 30, 90), [(120, 100, 30, 90)], (1280, 720), env)
    assert not A.is_clean((1, 100, 30, 90), [], (1280, 720), env)   # frame edge


def test_pose_points_follow_clip_phase(tmp_path):
    t = np.linspace(0, 1.0, 101)
    fl = np.c_[np.zeros_like(t), 40 * np.sin(2 * np.pi * t), np.zeros_like(t)]
    cur = tmp_path / "c_walk.json"
    cur.write_text(json.dumps({"len_s": 1.0, "t": t.tolist(), "foot_l": fl.tolist(),
                               "foot_r": (-fl).tolist()}))
    ch = {"height_cm": 175.0, "foot_curves": str(cur), "yaw_offset": 0.0}
    entry = {"height_cm": 175.0, "keys": [{"frame": f, "location": [0, 0, 0]} for f in range(20)],
             "yaws": [0.0] * 20, "segments": [{"clip": "walk", "rate": 1.0, "start": 0, "end": 20}]}
    p = A.pose_points(entry, ch, 5, 20.0, 3)     # t = 0.25 s: left foot fully forward
    # UE yaw 0: local +y -> UE +Y -> SI -y (Y negated)
    assert abs(p[0, 1] - (-0.40)) < 0.01 and abs(p[2, 1] - 0.40) < 0.01


def test_height_check_files(tmp_path):
    gt = tmp_path / "gt"
    gt.mkdir()
    pd.DataFrame({"uid": ["a", "b"], "height_m": [1.80, 1.70]}).to_parquet(gt / "identities.parquet")
    (tmp_path / "render_appearance.json").write_text(json.dumps(
        {"a": {"height_cm": 170.0}, "b": {"height_cm": 170.0}}))
    c = check_height_files(gt, tmp_path / "render")
    assert c.status == "FAIL" and c.value == 1
    (tmp_path / "render_appearance.json").write_text(json.dumps(
        {"a": {"height_cm": 180.0}, "b": {"height_cm": 170.5}}))
    assert check_height_files(gt, tmp_path / "render").status == "PASS"

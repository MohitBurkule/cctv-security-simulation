"""Regression: static-occlusion GT must follow the live PTZ pose (CLAUDE.md
principle 2; docs/BUG_gt_visibility_fences.md "PTZ"). Before the fix the fence
depth map was built from calib/<cam>.json — the configured pose — for every
frame, so a slewed PTZ was scored against fences it was not looking at."""
import numpy as np
import pandas as pd
import pytest

from survsim.gt.static_occlusion import (
    PoseUnknown,
    correct_rows,
    depth_map,
    live_static,
    occluder_prims,
    pose_from_status,
)

RES = (640, 360)
POS = np.array([0.0, 0.0, 1.5])
HFOV = 40.0


def _level_pose(yaw_deg):
    """Hand-built (K, R, t) for a level camera at POS — independent of geometry.py."""
    y = np.radians(yaw_deg)
    fwd, right, down = [np.cos(y), np.sin(y), 0.0], [np.sin(y), -np.cos(y), 0.0], [0, 0, -1.0]
    R = np.array([right, down, fwd])
    fx = (RES[0] / 2) / np.tan(np.radians(HFOV) / 2)
    K = np.array([[fx, 0, RES[0] / 2], [0, fx, RES[1] / 2], [0, 0, 1]])
    return K, R, -R @ POS


def _cal(ptz=True):
    K, R, t = _level_pose(0.0)                      # configured pose: yaw 0
    return {"camera_id": "cam_ptz", "type": "ptz" if ptz else "fixed", "resolution": list(RES),
            "fps": 5.0, "K": K.tolist(), "R": R.tolist(), "t": t.tolist(),
            "position_world": POS.tolist(), "hfov_deg": HFOV}


# A 2 m wall 8 m out on the 30 deg bearing (UE cm, Y negated); the person 12 m
# out on the same bearing, behind it. Outside the yaw-0 view (hfov 40).
B = np.radians(30.0)
ITEMS = [{"kind": "wall_run", "x": 800 * np.cos(B), "y": -800 * np.sin(B), "yaw": -120.0,
          "size": [400, 20, 200]}]
PERSON = np.array([12 * np.cos(B), 12 * np.sin(B)])


def _person_box(K, R, t):
    pts = np.array([[*PERSON, 0.0], [*PERSON, 1.8]])
    cam = pts @ R.T + t
    uv = (cam @ K.T)[:, :2] / cam[:, 2:3]
    h = abs(uv[0, 1] - uv[1, 1])
    return [float(uv[1, 0] - h / 6), float(uv[1, 1]), float(h / 3), float(h)]


def _world(frames):
    return pd.DataFrame({"frame": frames, "uid": "p1", "x": PERSON[0], "y": PERSON[1]})


def test_slewed_ptz_uses_live_pose_and_matches_direct_projection():
    cal = _cal()
    status = [{"t": 0.2, "camera": "cam_ptz", "yaw": 30.0, "pitch": 0.0, "hfov_deg": HFOV}]
    live = live_static(cal, occluder_prims(ITEMS), _world([1]), {"p1": 1}, status, "cam_ptz", [1])
    Kl, Rl, tl = _level_pose(30.0)
    direct = depth_map(Kl, Rl, tl, RES, occluder_prims(ITEMS))
    assert np.array_equal(np.isfinite(live["zbuf"][1]), np.isfinite(direct))
    assert np.allclose(pose_from_status(cal, 30.0, 0.0, HFOV)[1], Rl, atol=1e-9)
    row = [[1, 1, *_person_box(Kl, Rl, tl), 1, 1, 1.0]]
    rec_live = correct_rows(row, live["zbuf"], live["depth"], RES)[0]
    configured = depth_map(cal["K"], cal["R"], cal["t"], RES, occluder_prims(ITEMS))
    rec_cfg = correct_rows(row, configured, live["depth"], RES)[0]
    assert rec_live["static_cover"] > 0.5 and rec_live["occluder"] == "static"
    assert rec_cfg["static_cover"] == 0.0            # the old, configured-pose answer
    assert rec_live["vis"] != rec_cfg["vis"]


def test_ptz_frame_without_recorded_pose_is_flagged_not_calibrated():
    cal = _cal()
    status = [{"t": 0.2, "camera": "cam_ptz", "yaw": 30.0, "pitch": 0.0, "hfov_deg": HFOV}]
    live = live_static(cal, occluder_prims(ITEMS), _world([1, 2]), {"p1": 1}, status, "cam_ptz", [1, 2])
    assert live["pose_unknown"] == [2] and live["zbuf"][2] is None
    rec = correct_rows([[2, 1, 300.0, 150.0, 12.0, 34.0, 1, 1, 0.9]], live["zbuf"], live["depth"], RES)[0]
    assert rec["occluder"] == "pose_unknown" and rec["static_cover"] is None and rec["vis"] == 0.9
    with pytest.raises(PoseUnknown):
        live_static(cal, occluder_prims(ITEMS), _world([2]), {"p1": 1}, [], "cam_ptz", [2], strict=True)


def test_fixed_camera_without_status_falls_back_to_calib():
    cal = _cal(ptz=False)
    live = live_static(cal, occluder_prims(ITEMS), _world([1]), {"p1": 1}, [], "cam_ptz", [1])
    assert live["pose_unknown"] == []
    ref = depth_map(cal["K"], cal["R"], cal["t"], RES, occluder_prims(ITEMS))
    assert np.array_equal(np.isfinite(live["zbuf"][1]), np.isfinite(ref))

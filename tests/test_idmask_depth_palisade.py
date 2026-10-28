"""Regression: the ID mask must not show people through a palisade.

Fixture: 55x40 px crops of site 09 (transit_station s1333, cam_00, frames 1-2)
rendered 2026-09-28 with and without scripts/ue/survsim_scene.ID_MASK_DEPTH_TEST,
plus the projected fence depth map (4x supersampled) and person depths.
docs/BUG_gt_visibility_fences.md "Verification".
"""
from pathlib import Path

import numpy as np

from survsim.gt.static_occlusion import SS, filter_idmap

FX = np.load(Path(__file__).parent / "fixtures" / "idmask_palisade_site09.npz")


def _fence():
    z = FX["zbuf"]
    H, W = z.shape[0] // SS, z.shape[1] // SS
    return (np.isfinite(z).reshape(H, SS, W, SS)).mean(axis=(1, 3)) >= 0.5


def test_depth_test_removes_pixels_only_on_pales():
    fence = _fence()
    for f in (1, 2):
        a, b = FX[f"depth_f{f}"], FX[f"nodepth_f{f}"]
        for k in (2, 3):
            removed = (b == k) & (a != k)
            # Rendered 2026-09-28: 84->40, 62->30, 76->34, 63->23 pixels.
            assert (a == k).sum() < 0.65 * (b == k).sum()
            assert removed.sum() >= 20
            # >= 95 % of removed pixels fall on the projected pales.
            assert (removed & fence).sum() >= 0.95 * removed.sum()
            # The depth test never adds pixels (<= 2 px: the two passes are separate renders).
            assert ((a == k) & (b != k)).sum() <= 2


def test_post_hoc_filter_reduces_legacy_mask_toward_native():
    for f in (1, 2):
        b, a = FX[f"nodepth_f{f}"], FX[f"depth_f{f}"]
        dep = {int(k): float(d) for k, d in FX[f"depth_by_id_f{f}"]}
        out = filter_idmap(b, FX["zbuf"], dep)
        for k in (2, 3):
            assert (out == k).sum() < (b == k).sum()
            # Filter is conservative on the fence: it keeps no more than the native mask + 10 %.
            assert (out == k).sum() <= 1.1 * (a == k).sum() + 2


def test_build_tight_gt_carries_static_occluder(tmp_path):
    """_camera_boxes with static inputs: filters a legacy mask and labels rows."""
    from PIL import Image

    from survsim.gt.mask_boxes import _camera_boxes, encode_ids
    b = FX["nodepth_f1"]
    (tmp_path / "m").mkdir()
    Image.fromarray(encode_ids(b.astype(np.int32))).save(tmp_path / "m" / "0001.png")
    ys, xs = np.nonzero(b == 2)
    mot = tmp_path / "mot_cam_00.txt"
    mot.write_text(f"1,7,{xs.min()},{ys.min()},{xs.ptp() + 1},{ys.ptp() + 1},1,1,1.0\n")
    dep = {int(k): float(d) for k, d in FX["depth_by_id_f1"]}
    static = {"zbuf": FX["zbuf"], "depth": {(1, 7): dep[2]}, "res": (b.shape[1], b.shape[0]),
              "filter": True}
    lines, rows, _ = _camera_boxes((mot, tmp_path / "m", {"p": 2}, {"p": 7}, set(), static))
    r = rows[0]
    assert r["source"] == "mask" and r["mask_pixels"] < (b == 2).sum()
    assert r["static_cover"] >= 0.3 and r["occluder"] == "static"
    assert float(lines[0].split(",")[8]) < 0.7

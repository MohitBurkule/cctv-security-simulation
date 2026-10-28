"""Render gate: each check is fed the failure it guards against, built synthetically.

A gate that cannot tell a black frame, a duplicated camera, a sky-only view or a
missing person from a good render is worth nothing, so each failure is staged.
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from survsim.checks import check_job_result
from survsim.checks.core import FAIL, PASS, format_table
from survsim.checks.render import (
    _boxes_by_id,
    check_animation,
    check_blank,
    check_boxes_in_image,
    check_counts_vs_world,
    check_exposure,
    check_frame_count,
    check_height_files,
    check_horizon,
    check_log_text,
    check_motion,
    check_person_rendered,
    check_rendered_height_matches_gt,
    check_views_differ,
    recall,
    run_all,
)

H, W = 120, 160


def ground(seed=0):
    """Textured grey ground in the bottom 3/4, blue sky strip on top."""
    rng = np.random.default_rng(seed)
    img = rng.integers(60, 140, (H, W, 3), dtype=np.uint8)
    img[: H // 4] = (220, 170, 120)   # BGR sky
    return img


def with_person(img, x, y, w=10, h=30, phase=0):
    out = img.copy()
    out[y:y + h, x:x + w] = (30, 30, 200)
    # legs alternate with phase: the animated part
    leg = (phase % 2) * (w // 2)
    out[y + 2 * h // 3:y + h, x + leg:x + leg + w // 2] = (10, 200, 10)
    return out


def test_blank_frames_fail() -> None:
    assert check_blank([np.zeros((H, W, 3), np.uint8)]).status == FAIL
    assert check_blank([np.full((H, W, 3), 128, np.uint8)]).status == FAIL   # flat grey
    assert check_blank([ground()]).status == PASS


def test_frozen_sequence_fails_and_moving_passes() -> None:
    g = ground()
    assert check_motion([g] * 6).status == FAIL
    assert check_motion([with_person(g, 10 + 5 * i, 50) for i in range(6)]).status == PASS


def test_motion_gt_aware_catches_a_far_dim_person_whole_frame_test_misses() -> None:
    """docs/11 far-night motion gate: a 14 px person walking under a dim lamp moves
    the whole-frame average by less than sensor noise, but clearly moves inside
    their own (dilated) GT box.
    """
    bh, bw = 720, 1280   # a work-resolution-scale frame: the person is a small fraction of it
    rng = np.random.default_rng(0)
    dim = rng.integers(8, 14, (bh, bw, 3), dtype=np.uint8)   # near-black night plate
    idx = list(range(6))
    mot = {}
    frames = {}
    for i in idx:
        img = dim.copy()
        x = 400 + i   # a real, small step every frame
        img[300:314, x:x + 8] = (60, 60, 60)   # ~14 px tall person, barely above noise
        frames[i] = img
        mot[i] = [(0, float(x), 300.0, 8.0, 14.0)]
    by_id = _boxes_by_id(mot, scale=1.0)
    whole_frame = check_motion(frames, name="motion")
    gt_aware = check_motion(frames, name="motion", by_id=by_id, idx=idx)
    assert whole_frame.status == FAIL
    assert gt_aware.status == PASS
    assert "GT-box" in gt_aware.threshold


def test_motion_gt_aware_still_fails_a_genuinely_frozen_small_person() -> None:
    rng = np.random.default_rng(1)
    dim = rng.integers(8, 14, (H, W, 3), dtype=np.uint8)
    idx = list(range(6))
    frames, mot = {}, {}
    for i in idx:
        img = dim.copy()
        img[70:84, 40:48] = (60, 60, 60)   # never moves
        frames[i] = img
        mot[i] = [(0, 40.0, 70.0, 8.0, 14.0)]
    by_id = _boxes_by_id(mot, scale=1.0)
    assert check_motion(frames, name="motion", by_id=by_id, idx=idx).status == FAIL


def test_motion_falls_back_to_whole_frame_without_gt() -> None:
    g = ground()
    assert check_motion({i: g for i in range(6)}, name="motion", by_id=None, idx=[0, 1]).status == FAIL


def test_duplicate_camera_views_fail() -> None:
    """The missing CameraCutTrack bug: views differ only by AA noise."""
    a = ground(0)
    noise = np.clip(a.astype(int) + np.random.default_rng(1).integers(-1, 2, a.shape), 0, 255)
    assert check_views_differ({"c1": a, "c2": noise.astype(np.uint8)}).status == FAIL
    assert check_views_differ({"c1": a, "c2": np.flipud(a).copy()}).status == PASS


def test_exposure_band() -> None:
    assert check_exposure(np.full((H, W, 3), 255, np.uint8)).status == FAIL
    assert check_exposure(np.full((H, W, 3), 3, np.uint8)).status == FAIL
    assert check_exposure(ground()).status == PASS


def test_sky_only_view_fails_horizon() -> None:
    """Inverted pitch: the whole frame is sky."""
    sky = np.zeros((H, W, 3), np.uint8)
    sky[:] = (230, 180, 130)
    assert check_horizon(sky).status == FAIL
    white_sky = np.full((H, W, 3), 235, np.uint8)
    assert check_horizon(white_sky).status == FAIL
    assert check_horizon(ground()).status == PASS


def test_frame_count() -> None:
    assert check_frame_count({"a": 5, "b": 5}).status == PASS
    assert check_frame_count({"a": 5, "b": 4}).status == FAIL
    assert check_frame_count({"a": 5}, expected=6).status == FAIL
    assert check_frame_count({"a": 5}, gaps={"a": 1}).status == FAIL
    assert check_frame_count({}).status == FAIL


def test_box_outside_image_fails() -> None:
    assert check_boxes_in_image({1: [(1, 10, 10, 20, 40)]}, (W, H)).status == PASS
    assert check_boxes_in_image({1: [(1, W - 5, 10, 20, 40)]}, (W, H)).status == FAIL


def test_person_missing_where_gt_says_fails() -> None:
    """GT box over empty ground (people never rendered) must fail."""
    plate = ground()
    mot = {i: [(1, 20 + 10 * i, 50, 10, 30)] for i in range(5)}
    empty = {i: plate.copy() for i in range(5)}
    drawn = {i: with_person(plate, 20 + 10 * i, 50) for i in range(5)}
    assert check_person_rendered(empty, plate, mot, 1.0).status == FAIL
    assert check_person_rendered(drawn, plate, mot, 1.0).status == PASS


def test_unanimated_person_fails_animation() -> None:
    plate = ground()
    mot = {i: [(1, 20 + 3 * i, 50, 10, 30)] for i in range(12)}
    # rigid slide on a flat background: crop content identical frame to frame
    flat = np.full((H, W, 3), 100, np.uint8)
    rigid = {i: with_person(flat, 20 + 3 * i, 50, phase=0) for i in range(12)}
    walking = {i: with_person(plate, 20 + 3 * i, 50, phase=i) for i in range(12)}
    assert check_animation(rigid, mot, 1.0).status == FAIL
    assert check_animation(walking, mot, 1.0).status == PASS


def test_more_boxes_than_people_fails() -> None:
    assert check_counts_vs_world({1: [0, 0]}, {1: 3}).status == PASS
    assert check_counts_vs_world({1: [0, 0, 0, 0]}, {1: 3}).status == FAIL


def test_log_scan() -> None:
    good = "LogInit: ok\nLogPython: hello\n"
    bad = "LogPython: Error: Traceback\nLogWindows: Fatal error!\nLogPython: Warning: Failed to convert type\n"
    assert all(c.status == PASS for c in check_log_text(good))
    st = {c.name: c.status for c in check_log_text(bad)}
    assert st == {"log_python_error": FAIL, "log_fatal": FAIL, "log_type_convert": "WARN"}


def test_recall() -> None:
    assert recall({0: [[0, 0, 10, 10]]}, {0: [[1, 1, 10, 10]]})[0] == 1.0
    assert recall({0: [[0, 0, 10, 10]]}, {0: [[50, 50, 10, 10]]})[0] == 0.0


def test_job_result() -> None:
    assert check_job_result({"ok": True, "seconds": 1})[0].status == PASS
    c = check_job_result({"ok": False, "error": "Traceback...\nValueError: boom"})[0]
    assert c.status == FAIL and "ValueError: boom" in c.detail
    assert check_job_result(None)[0].status == FAIL
    assert check_job_result({"seconds": 1})[0].status == FAIL        # ok missing is not a pass


def _write_run(tmp_path, draw_people=True, same_view=False):
    rd = tmp_path / "run"
    gt = rd / "gt"
    gt.mkdir(parents=True)
    import pandas as pd
    rows, world = {"cam_01": [], "cam_02": []}, []
    for cam, seed in (("cam_01", 0), ("cam_02", 0 if same_view else 7)):
        d = rd / "render" / cam
        d.mkdir(parents=True)
        base = ground(seed)
        for i in range(12):
            x = 20 + 4 * i
            img = with_person(base, x, 50, phase=i) if draw_people else base
            cv2.imwrite(str(d / f"{i:04d}.png"), img)
            rows[cam].append(f"{i},1,{x},50,10,30,1,1,1.0")
    for i in range(12):
        world.append({"frame": i, "uid": "p1"})
    for cam, r in rows.items():
        (gt / f"mot_{cam}.txt").write_text("\n".join(r) + "\n")
    pd.DataFrame(world).to_parquet(gt / "world.parquet")
    return rd


def test_run_all_good_and_bad(tmp_path) -> None:
    good = run_all(_write_run(tmp_path / "g"), tmp_path / "g/run/gt", realism=False)
    assert not [c.name for c in good if c.failed], format_table(good)
    bad = run_all(_write_run(tmp_path / "b", draw_people=False, same_view=True),
                  tmp_path / "b/run/gt", realism=False)
    failed = {c.name for c in bad if c.failed}
    assert {"views_differ", "cam_01:gt_person_rendered", "cam_01:motion"} <= failed


# -------------------------------------------------- rendered height vs GT height


def test_height_check_passes_within_tolerance() -> None:
    gt = {"p1": 1.70, "p2": 1.88}
    rendered = {"p1": 1.71, "p2": 1.87}   # 1 cm off, under the 2 cm tolerance
    c = check_rendered_height_matches_gt(gt, rendered)
    assert c.status == PASS


def test_height_check_flags_a_render_stuck_at_the_default() -> None:
    """The bug this guards: every person rendered at 1.70 m regardless of GT."""
    gt = {"p1": 1.70, "p2": 1.92, "p3": 1.58}
    rendered = {"p1": 1.70, "p2": 1.70, "p3": 1.70}
    c = check_rendered_height_matches_gt(gt, rendered)
    assert c.status == FAIL
    assert c.value == 2   # p2 and p3 exceed the 2 cm band; p1 happens to match


def test_height_check_flags_a_missing_rendered_person() -> None:
    c = check_rendered_height_matches_gt({"p1": 1.70, "p2": 1.80}, {"p1": 1.70})
    assert c.status == FAIL
    assert "missing" in c.detail


def test_height_check_skips_when_nothing_overlaps() -> None:
    assert check_rendered_height_matches_gt({"p1": 1.70}, {"p2": 1.70}).status == "SKIP"


def _write_identities(gt_dir, heights):
    import pandas as pd
    pd.DataFrame({"uid": list(heights), "height_m": list(heights.values())}).to_parquet(
        gt_dir / "identities.parquet")


def test_height_files_finds_render_appearance_next_to_set_dir(tmp_path) -> None:
    """`scripts/check_render.py <set> --gt <set>` passes the set dir itself (not
    its render/ subdir) as `render_dir` - the standalone-CLI layout that used to
    make `gt_height_matches_render` SKIP with "the rendered-identities join is
    missing" (docs/12 Sec.5) because the old code only looked one level above
    `render_dir`, i.e. above the set dir."""
    rd = _write_run(tmp_path, draw_people=False)
    _write_identities(rd / "gt", {"p1": 1.70})
    (rd / "render_appearance.json").write_text(json.dumps({"p1": {"height_cm": 171.0}}))
    c_set_dir = check_height_files(rd / "gt", rd)
    assert c_set_dir.status == PASS
    assert c_set_dir.detail.startswith("0/1")
    # tools/render_run.py's own layout - render_dir is the render/ subdir - must
    # keep working too.
    c_render_subdir = check_height_files(rd / "gt", rd / "render")
    assert c_render_subdir.status == PASS


def test_height_files_flags_a_mismatch_via_render_appearance(tmp_path) -> None:
    rd = _write_run(tmp_path, draw_people=False)
    _write_identities(rd / "gt", {"p1": 1.66})
    (rd / "render_appearance.json").write_text(json.dumps({"p1": {"height_cm": 170.0}}))
    c = check_height_files(rd / "gt", rd)
    assert c.status == FAIL
    assert c.detail.startswith("1/1")


def test_height_files_falls_back_to_identities_rendered_parquet(tmp_path) -> None:
    """No render_appearance.json (e.g. an older set) but identities_rendered.parquet
    is present: join on uid against that instead of SKIPping."""
    rd = _write_run(tmp_path, draw_people=False)
    _write_identities(rd / "gt", {"p1": 1.70, "p2": 1.90})
    import pandas as pd
    pd.DataFrame({"uid": ["p1", "p2"], "height_m": [1.70, 1.70]}).to_parquet(
        rd / "identities_rendered.parquet")
    c = check_height_files(rd / "gt", rd)
    assert c.status == FAIL
    assert c.value == 1   # only p2 (20 cm off) exceeds the 2 cm band


def test_height_files_skips_when_no_rendered_source_exists(tmp_path) -> None:
    rd = _write_run(tmp_path, draw_people=False)
    _write_identities(rd / "gt", {"p1": 1.70})
    assert check_height_files(rd / "gt", rd).status == "SKIP"


def test_cli_exit_code_and_json(tmp_path) -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("cr", "scripts/check_render.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rd = _write_run(tmp_path, draw_people=False)
    out = tmp_path / "r.json"
    assert mod.main([str(rd), "--gt", str(rd), "--json", str(out), "--no-realism"]) == 1
    assert json.loads(out.read_text())["verdict"] == FAIL

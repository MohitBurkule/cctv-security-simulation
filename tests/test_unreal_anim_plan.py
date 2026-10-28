"""The render must bind a walk/idle animation to EVERY person (no engine needed).

A person with a transform track but no animation track renders as a sliding
T-pose — quiet, like the missing camera cut that collapsed a rig to one view.
"""
import itertools
from pathlib import Path

from survsim.export.anim_plan import anim_segments, person_plan, travel_yaws

RENDER = Path(__file__).resolve().parent.parent / "scripts" / "ue" / "survsim_scene.py"


def _track(uid, step_cm, n=40, pause=None):
    keys, x = [], 0.0
    for f in range(n):
        if not (pause and pause[0] <= f < pause[1]):
            x += step_cm
        keys.append({"frame": f, "location": [x, 0.0, 0.0], "yaw": 0.0})
    return {"uid": uid, "keys": keys}


def _manifest():
    return {"fps": 10, "seed": 3, "tracks": [
        _track("walker", 12.0), _track("stander", 0.0),
        _track("pauser", 12.0, pause=(10, 30)), {"uid": "single", "keys": [
            {"frame": 0, "location": [0, 0, 0], "yaw": 0}]}]}


def test_every_person_gets_full_coverage_animation():
    plan = person_plan(_manifest(), 40)
    assert [p["uid"] for p in plan] == ["walker", "stander", "pauser", "single"]
    for p in plan:
        segs = p["segments"]
        assert segs, p["uid"]
        assert segs[0]["start"] == 0 and segs[-1]["end"] >= 40
        for a, b in itertools.pairwise(segs):
            assert a["end"] == b["start"]


def test_rate_tracks_speed_and_idle_when_stationary():
    segs = {p["uid"]: p["segments"] for p in person_plan(_manifest(), 40)}
    assert segs["walker"][0]["clip"] == "walk" and 0.9 <= segs["walker"][0]["rate"] <= 1.3
    assert all(s["clip"] == "idle" for s in segs["stander"])
    assert [s["clip"] for s in segs["pauser"]] == ["walk", "idle", "walk"]


def test_yaw_faces_travel_and_unwraps():
    keys = [{"frame": i, "location": loc, "yaw": 0} for i, loc in
            enumerate([[0, 0, 0], [0, 10, 0], [-10, 10, 0], [-10, 0, 0], [0, 0, 0]])]
    yaws = travel_yaws(keys)
    assert abs(yaws[0] - 90) < 1e-6
    assert all(abs(b - a) <= 180 for a, b in itertools.pairwise(yaws))


def test_segments_never_per_frame_jitter():
    keys = [{"frame": f, "location": [f * (10 + (f % 2) * 3), 0, 0], "yaw": 0}
            for f in range(50)]
    assert len(anim_segments(keys, 10, 50)) <= 50 // 5


def test_render_script_binds_anim_track_and_refuses_partial():
    src = RENDER.read_text()
    assert "MovieSceneSkeletalAnimationTrack" in src
    assert "people animated" in src          # the guard that raises on a gap
    assert "BasicShapes/Cube.Cube\")\n    fps" not in src


def test_person_hidden_outside_key_range():
    """Arrivals/leavers: the plan carries the visible range and the render keys
    a visibility track from it (no frozen figure at spawn or exit)."""
    m = {"fps": 10, "tracks": [{"uid": "late", "keys": [
        {"frame": f, "location": [f * 10.0, 0, 0], "yaw": 0} for f in range(12, 30)]}]}
    plan = person_plan(m, 40)
    assert plan[0]["visible"] == [12, 29]
    src = RENDER.read_text()
    assert "MovieSceneVisibilityTrack" in src and "_key_visibility(b, entry.get(\"visible\")" in src


# --- behaviour -> animation (2026-09-26) ---------------------------------------

def test_every_catalogue_behaviour_maps_to_a_clip_or_fallback():
    import re
    from pathlib import Path

    from survsim.director.behaviours import REGISTRY
    from survsim.export.anim_plan import BEHAVIOUR_ANIM, POSTURE_FALLBACK
    missing = sorted(set(REGISTRY) - set(BEHAVIOUR_ANIM))
    assert not missing, f"behaviours with no animation mapping: {missing}"
    src = (Path(__file__).resolve().parents[1] / "survsim/director/behaviours.py").read_text()
    postures = set(re.findall(r'posture = "([a-z_]+)"', src))
    assert postures <= set(POSTURE_FALLBACK), sorted(postures - set(POSTURE_FALLBACK))


def _keys(speeds_mps, fps=10.0, label="normal.transit", posture="upright"):
    x, keys = 0.0, []
    for f, v in enumerate(speeds_mps):
        keys.append({"frame": f, "location": [x, 0.0, 0.0], "label": label, "posture": posture})
        x += v * 100.0 / fps
    return keys


def test_loiter_stands_and_looks_while_commute_wait_idles():
    from survsim.export.anim_plan import anim_segments
    loiter = anim_segments(_keys([0.0] * 40, label="suspicious.loiter"), 10.0, 40, 1.3)
    wait = anim_segments(_keys([0.0] * 40, label="normal.commute_wait"), 10.0, 40, 1.3)
    assert [s["clip"] for s in loiter] == ["look"] and [s["clip"] for s in wait] == ["idle"]


def test_fast_movement_runs_only_with_a_run_clip():
    from survsim.export.anim_plan import anim_segments
    k = _keys([3.5] * 30, label="suspicious.flee")
    assert anim_segments(k, 10.0, 30, 1.3, run_speed=3.5)[0]["clip"] == "run"
    assert anim_segments(k, 10.0, 30, 1.3, run_speed=None)[0]["clip"] == "walk"


def test_crouch_without_a_clip_holds_the_idle_loop():
    from survsim.export.anim_plan import anim_segments
    segs = anim_segments(_keys([0.5] * 30, posture="crouch"), 10.0, 30, 1.3)
    assert [s["clip"] for s in segs] == ["idle"]


def test_look_turns_the_body_and_idle_phase_differs_between_people():
    from survsim.export.anim_plan import person_plan
    man = {"fps": 10, "seed": 1, "tracks": [
        {"uid": u, "keys": _keys([0.0] * 60, label="suspicious.loiter")} for u in ("a", "b")]}
    plan = person_plan(man, 60, {"X": {"mesh": "m", "walk": "w", "idle": "i"}})
    yaws = plan[0]["yaws"]
    assert max(yaws) - min(yaws) > 30.0                     # scans left and right
    assert plan[0]["segments"][0]["phase_s"] != plan[1]["segments"][0]["phase_s"]

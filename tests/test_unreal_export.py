"""Tests for exporting a run to an Unreal-rebuildable manifest.

Coordinate conversion is the whole risk here. Getting it wrong does not raise —
it produces a scene that renders happily with people mirrored across an axis or
cameras aimed at the sky, and the ground truth still says they were pointed at
the ground. That is the inverted-pitch bug, which survived a self-consistent
test and silently pointed every camera upward.

So these tests pin the conversion against *known geometry* rather than against
itself: a person to the north must end up north, a camera pitched down must stay
pitched down.
"""

from __future__ import annotations

import json
from typing import ClassVar

import pytest

from survsim.export.unreal_scene import (
    M_TO_CM,
    MANIFEST_VERSION,
    WardrobeSpec,
    build_manifest,
    camera_entry,
    to_unreal_location,
    to_unreal_rotation,
    track_entries,
    wardrobe_from_appearance,
    wardrobe_from_identity_row,
    write_manifest,
)


class _Mount:
    def __init__(self, pos, yaw, pitch=0.0, roll=0.0):
        self.pos, self.yaw, self.pitch, self.roll = pos, yaw, pitch, roll


class _Camera:
    def __init__(self, cid="cam_a", pos=(1.0, 2.0, 4.0), yaw=90.0, pitch=-20.0):
        self.id = cid
        self.mount = _Mount(pos, yaw, pitch)
        self.capture = {"res": (1280, 720), "fps": 25.0}


class _Scenario:
    name, seed, fps, duration_s = "s", 7, 25.0, 40.0
    cameras: ClassVar = [_Camera()]


# --------------------------------------------------------------- location


def test_metres_become_centimetres() -> None:
    assert to_unreal_location((1.0, 0.0, 0.0))[0] == 100.0
    assert M_TO_CM == 100.0


def test_the_y_axis_is_negated_for_unreals_handedness() -> None:
    """Miss this and the whole scene is mirrored, silently."""
    assert to_unreal_location((0.0, 3.0, 0.0))[1] == -300.0


def test_height_is_preserved_not_flipped() -> None:
    assert to_unreal_location((0.0, 0.0, 4.0))[2] == 400.0


def test_a_missing_z_defaults_to_the_ground() -> None:
    assert to_unreal_location((1.0, 2.0)) == [100.0, -200.0, 0.0]


# --------------------------------------------------------------- rotation


def test_yaw_handedness_is_flipped() -> None:
    assert to_unreal_rotation(90.0)[1] == -90.0


def test_pitch_is_not_negated() -> None:
    """The inverted-pitch bug, guarded in the new location.

    A camera pitched 20 degrees down must still be pitched down after export.
    Negating it here would aim every camera at the sky while ground truth
    continued to claim it was looking at the floor.
    """
    pitch, _, _ = to_unreal_rotation(0.0, -20.0)
    assert pitch == -20.0, "a downward camera must stay downward"


def test_rotation_is_ordered_pitch_yaw_roll_as_unreal_expects() -> None:
    assert to_unreal_rotation(10.0, 20.0, 30.0) == [20.0, -10.0, 30.0]


# ---------------------------------------------------------------- cameras


def test_camera_entry_carries_pose_and_intrinsics() -> None:
    entry = camera_entry(_Camera(pos=(1.0, 2.0, 4.0), yaw=90.0, pitch=-20.0))

    assert entry["location"] == [100.0, -200.0, 400.0]
    assert entry["rotation"][0] == -20.0 and entry["rotation"][1] == -90.0
    assert entry["resolution"] == [1280, 720]
    assert entry["fps"] == 25.0


# ----------------------------------------------------------------- tracks


def _rows():
    return [
        {"frame": 1, "t": 0.04, "uid": "p1", "x": 1.0, "y": 2.0, "speed": 1.2,
         "heading_deg": 90.0, "posture": "upright"},
        {"frame": 0, "t": 0.0, "uid": "p1", "x": 0.0, "y": 0.0, "speed": 0.0,
         "heading_deg": 0.0, "posture": "upright"},
        {"frame": 0, "t": 0.0, "uid": "p2", "x": 5.0, "y": 0.0, "speed": 1.0,
         "heading_deg": 180.0, "posture": "upright"},
    ]


def test_tracks_are_grouped_by_person_and_sorted_by_frame() -> None:
    """Sequencer binds tracks to actors, so person-major is the useful layout."""
    tracks = track_entries(_rows(), fps=25.0)

    assert [t["uid"] for t in tracks] == ["p1", "p2"]
    assert [k["frame"] for k in tracks[0]["keys"]] == [0, 1]


def test_track_positions_are_converted() -> None:
    tracks = track_entries(_rows(), fps=25.0)
    assert tracks[0]["keys"][1]["location"] == [100.0, -200.0, 0.0]


def test_track_yaw_is_converted() -> None:
    tracks = track_entries(_rows(), fps=25.0)
    assert tracks[0]["keys"][1]["yaw"] == -90.0


def test_altitude_is_carried_so_airborne_actors_are_not_flattened() -> None:
    """A drone on the ground plane would make its ground truth nonsense."""
    rows = [{"frame": 0, "t": 0.0, "uid": "d", "x": 0.0, "y": 0.0,
             "altitude_m": 45.0}]
    assert track_entries(rows, 25.0)[0]["keys"][0]["location"][2] == 4500.0


def test_rows_without_an_identity_are_skipped_not_crashed_on() -> None:
    assert track_entries([{"frame": 0, "x": 1.0, "y": 1.0}], 25.0) == []


def test_missing_time_is_derived_from_the_frame_rate() -> None:
    rows = [{"frame": 50, "uid": "p", "x": 0.0, "y": 0.0}]
    assert track_entries(rows, 25.0)[0]["keys"][0]["t"] == pytest.approx(2.0)


# -------------------------------------------------------------- wardrobe


class _Outfit:
    top_colour, bottom_colour = "#ff0000", "#0000ff"
    outer, headwear, footwear = "jacket", None, "boots"


class _Appearance:
    outfit = _Outfit()
    accessories: ClassVar = ["backpack"]


def test_wardrobe_is_described_not_pointed_at_an_asset() -> None:
    """The simulator says what someone looks like; the engine maps it.

    Keeping these apart is why the simulator was unaffected when the engine's
    wardrobe commands turned out not to exist.
    """
    w = wardrobe_from_appearance(_Appearance())
    assert w.top_colour == "#ff0000"
    assert w.outer == "jacket"
    assert "backpack" in w.accessories
    assert not hasattr(w, "asset_path")


def test_face_covering_is_derived_from_what_is_worn() -> None:
    """A separate flag could disagree with the accessory list; this cannot."""
    plain = wardrobe_from_appearance(_Appearance())
    assert plain.face_covered == 0.0

    class Masked(_Appearance):
        accessories: ClassVar = ["balaclava"]

    assert wardrobe_from_appearance(Masked()).face_covered == 1.0


def test_wardrobe_survives_an_appearance_with_nothing_set() -> None:
    class Empty:
        pass

    w = wardrobe_from_appearance(Empty())
    assert w.top_colour and w.bottom_colour


def test_wardrobe_round_trips_through_json() -> None:
    assert json.loads(json.dumps(WardrobeSpec().as_dict()))["build"] == "average"


# -------------------------------------------------------------- manifest


def test_manifest_carries_everything_needed_to_rebuild_the_scene() -> None:
    m = build_manifest(_Scenario(), _rows(),
                       [{"uid": "p1", "epoch": 0, "gait_style": "normal"}])

    assert m["version"] == MANIFEST_VERSION
    assert m["fps"] == 25.0 and m["seed"] == 7
    assert len(m["cameras"]) == 1
    assert len(m["tracks"]) == 2
    assert m["people"][0]["uid"] == "p1"


# ---------------------------------------------------- Director height -> manifest
#
# The [X] item in docs/12 ("1.70 m render height vs the Director's height"):
# identities.parquet carries a `height_m` column per (uid, epoch)
# (core.appearance.SOFT_BIOMETRIC_KEYS) but no `wardrobe` field, so
# `build_manifest` used to fall through to WardrobeSpec()'s 1.70 m default for
# every single person, regardless of what the Director actually generated.


def test_wardrobe_from_identity_row_uses_the_row_height() -> None:
    row = {"uid": "p1", "height_m": 1.92, "outfit": json.dumps({"top_colour": "#123456"})}
    w = wardrobe_from_identity_row(row)
    assert w["height_m"] == pytest.approx(1.92)
    assert w["top_colour"] == "#123456"


def test_wardrobe_from_identity_row_defaults_when_height_is_absent() -> None:
    assert wardrobe_from_identity_row({"uid": "p1"})["height_m"] == pytest.approx(1.70)


@pytest.mark.parametrize("height_m", [1.55, 1.70, 1.68, 1.94, 2.01])
def test_manifest_height_equals_gt_height_for_every_person(height_m: float) -> None:
    """The manifest's wardrobe height must equal identities.parquet's height_m,
    not the 1.70 m WardrobeSpec default -- for every height, not just the
    default's own value."""
    m = build_manifest(_Scenario(), _rows(),
                       [{"uid": "p1", "epoch": 0, "height_m": height_m,
                         "outfit": json.dumps({})}])
    assert m["people"][0]["wardrobe"]["height_m"] == pytest.approx(height_m)


def test_manifest_height_ignores_nan_height_and_falls_back() -> None:
    """A pandas row for a person with no biometric (shouldn't happen, but a
    missing column reads back as NaN) must not poison the wardrobe with NaN."""
    m = build_manifest(_Scenario(), _rows(),
                       [{"uid": "p1", "epoch": 0, "height_m": float("nan")}])
    assert m["people"][0]["wardrobe"]["height_m"] == pytest.approx(1.70)


def test_manifest_prefers_an_explicit_wardrobe_but_still_applies_gt_height() -> None:
    """A caller-supplied `wardrobe` (e.g. a pre-computed one) keeps its other
    fields, but the Director's own height_m always overrides its height --
    ground truth must win over a stale or default wardrobe height."""
    m = build_manifest(_Scenario(), _rows(),
                       [{"uid": "p1", "epoch": 0, "height_m": 1.88,
                         "wardrobe": {"top_colour": "#abcdef", "height_m": 1.70}}])
    w = m["people"][0]["wardrobe"]
    assert w["top_colour"] == "#abcdef"
    assert w["height_m"] == pytest.approx(1.88)


def test_manifest_states_the_unit_convention_it_used() -> None:
    """So a future reader never has to infer it from the numbers."""
    units = build_manifest(_Scenario(), _rows())["units"]
    assert "negate Y" in units["conversion"]
    assert "centimetres" in units["target"]


def test_manifest_is_versioned_so_a_mismatch_fails_loudly() -> None:
    """An unversioned manifest would build a subtly wrong scene instead."""
    assert build_manifest(_Scenario(), [])["version"] == MANIFEST_VERSION


def test_manifest_is_written_as_readable_json(tmp_path) -> None:
    path = write_manifest(tmp_path / "s" / "manifest.json",
                          build_manifest(_Scenario(), _rows()))
    assert path.exists()
    assert json.loads(path.read_text())["scenario"] == "s"


def test_an_empty_run_still_produces_a_valid_manifest() -> None:
    m = build_manifest(_Scenario(), [])
    assert m["tracks"] == [] and m["version"] == MANIFEST_VERSION


# ------------------------------------------------- parametric body mapping
#
# The constraint names below are the engine's own, read off a real character in
# UE 5.8.2. They are the API: a mistyped name does NOT raise, it leaves that
# measurement at its default while ground truth records the value we asked for.


from survsim.export.unreal_scene import (
    BODY_CONSTRAINT_NAMES,
    BUILD_SHAPES,
    HEIGHT_MAX_CM,
    HEIGHT_MIN_CM,
    body_constraints_for,
)


def test_all_thirty_verified_constraints_are_recorded() -> None:
    assert len(BODY_CONSTRAINT_NAMES) == 30
    for name in ("Height", "Chest", "Waist", "Hip", "Inseam",
                 "Fat", "Muscularity", "Masculine/Feminine"):
        assert name in BODY_CONSTRAINT_NAMES


def test_height_is_converted_to_the_centimetres_the_engine_wants() -> None:
    assert body_constraints_for(1.90)["Height"] == pytest.approx(190.0)


def test_height_is_clamped_to_the_range_the_engine_accepts() -> None:
    """Silently clamping beats being silently ignored.

    Passing 250 cm makes the engine keep the default height while our ground
    truth claims 2.5 m — a mismatch nothing downstream could detect. Clamping is
    wrong by a knowable amount; the alternative is wrong by an unknowable one.
    """
    assert body_constraints_for(3.0)["Height"] == HEIGHT_MAX_CM
    assert body_constraints_for(0.5)["Height"] == HEIGHT_MIN_CM


def test_build_labels_map_to_distinct_shapes() -> None:
    slim = body_constraints_for(1.75, "slim")
    heavy = body_constraints_for(1.75, "heavy")
    athletic = body_constraints_for(1.75, "athletic")

    assert heavy["Fat"] > slim["Fat"]
    assert athletic["Muscularity"] > heavy["Muscularity"]


def test_an_unknown_build_falls_back_to_average_rather_than_failing() -> None:
    assert body_constraints_for(1.75, "nonsense") == body_constraints_for(1.75, "average")


def test_femininity_is_optional_and_only_set_when_asked_for() -> None:
    """Presenting as a different gender defeats appearance ReID far more than a
    change of coat, so it is exposed — but it is not imposed on every person."""
    assert "Masculine/Feminine" not in body_constraints_for(1.75)
    assert body_constraints_for(1.75, femininity=1.0)["Masculine/Feminine"] == 1.0


def test_femininity_is_clamped_to_its_zero_to_one_range() -> None:
    assert body_constraints_for(1.75, femininity=5.0)["Masculine/Feminine"] == 1.0
    assert body_constraints_for(1.75, femininity=-2.0)["Masculine/Feminine"] == 0.0


def test_every_build_shape_is_inside_the_unit_range() -> None:
    for name, (fat, muscle) in BUILD_SHAPES.items():
        assert 0.0 <= fat <= 1.0 and 0.0 <= muscle <= 1.0, name


def test_exported_fov_is_the_one_ground_truth_was_projected_with() -> None:
    """The render and the ground truth must share a lens.

    The exporter once read `capture.fov_deg`, which no camera has, so every
    manifest carried None and Unreal used its default ~40 degree lens while
    ground truth was projected at the scenario's FOV. Nothing failed; every box
    simply landed on the wrong pixels.
    """
    from survsim.core.geometry import intrinsics
    from survsim.core.loader import load_scenario

    scenario = load_scenario("configs/scenarios/smoke_mock.yaml")
    for cam in scenario.cameras:
        entry = camera_entry(cam)
        assert entry["fov_deg"] is not None
        assert entry["fov_deg"] == pytest.approx(cam.hfov_deg)
        # And it is the FOV the projection actually used.
        w = cam.capture.res[0]
        import numpy as np
        fx = intrinsics(cam)[0, 0]
        assert np.degrees(2 * np.arctan(w / 2 / fx)) == pytest.approx(entry["fov_deg"])

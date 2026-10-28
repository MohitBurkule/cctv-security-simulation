"""Tests for parametric gait synthesis.

The reason this module exists rather than a clip library: a mocap clip is one
recording of one actor, so playing it for everyone with a given style makes them
all walk identically — reintroducing, one layer down, the exact defect that
`gait/signature.py` removed. These tests therefore pin *individuality* as hard as
they pin plausibility, because a synthesiser that produces correct-looking but
identical motion would pass a naive review.

The other property under test is agreement between what the signature asks for
and what the animation does. Those can drift apart silently while every joint
curve still looks right, and then the rendered person walks a different stride
from the one ground truth records.
"""

from __future__ import annotations

import numpy as np
import pytest

from survsim.gait.signature import signature_for
from survsim.gait.synthesis import (
    JOINTS,
    NORMAL_HARMONICS,
    measured_step_length_m,
    pose_at,
    stride_duration_s,
    synthesise,
)

# ------------------------------------------------------------- basic shape


def test_every_joint_is_driven() -> None:
    pose = pose_at(signature_for("alice"), 0.25)
    assert set(pose.left) == set(JOINTS)
    assert set(pose.right) == set(JOINTS)


def test_the_two_legs_run_half_a_cycle_apart() -> None:
    """Otherwise it is a hop, not a walk."""
    sig = signature_for("alice", style="normal")
    a = pose_at(sig, 0.0)
    b = pose_at(sig, 0.5)
    # Left at phase 0 should match right at phase 0.5 for a symmetric walker.
    assert a.left["hip_flex"] == pytest.approx(b.right["hip_flex"], abs=1.0)


def test_motion_is_periodic_over_one_stride() -> None:
    sig = signature_for("alice")
    assert pose_at(sig, 0.0).left["hip_flex"] == pytest.approx(
        pose_at(sig, 1.0).left["hip_flex"], abs=1e-6)


def test_joint_angles_stay_in_anatomically_plausible_ranges() -> None:
    """A knee that hyperextends 60 degrees would render as a broken leg."""
    for style in ("normal", "limp_L", "shuffle", "run", "drunk"):
        sig = signature_for("alice", style=style)
        for i in range(60):
            pose = pose_at(sig, i / 60)
            for side in (pose.left, pose.right):
                assert -40 <= side["hip_flex"] <= 80, style
                assert -15 <= side["knee_flex"] <= 90, style


# ------------------------------------------------- signature vs animation


def test_the_animation_walks_the_stride_the_signature_asked_for() -> None:
    """These drift apart silently while every joint curve still looks correct.

    Before calibration the synthesised step was 0.65 m against a requested
    0.75 m — a 15% error that would have put every rendered person out of step
    with their own ground truth.
    """
    for who in ("alice", "bob", "carol"):
        sig = signature_for(who)
        measured = measured_step_length_m(sig)
        assert measured == pytest.approx(sig.step_length_m, rel=0.05), who


@pytest.mark.parametrize("style", ["limp_L", "shuffle", "slow", "brisk"])
def test_step_length_tracks_the_signature_across_styles(style) -> None:
    """Looser tolerance at the extremes: hip angle relates to step by a sine,
    so the linear calibration is exact only near the nominal amplitude."""
    sig = signature_for("alice", style=style)
    assert measured_step_length_m(sig) == pytest.approx(sig.step_length_m, rel=0.12)


def test_a_shuffler_takes_visibly_shorter_steps_than_a_runner() -> None:
    shuffle = measured_step_length_m(signature_for("alice", style="shuffle"))
    run = measured_step_length_m(signature_for("alice", style="run"))
    assert run > shuffle * 2


def test_stride_duration_follows_cadence() -> None:
    slow = signature_for("alice", style="slow")
    brisk = signature_for("alice", style="brisk")
    assert stride_duration_s(slow) > stride_duration_s(brisk)
    # A stride is two steps.
    assert stride_duration_s(brisk) == pytest.approx(
        2.0 / (brisk.cadence_spm / 60.0), rel=1e-6)


# ------------------------------------------------------------ asymmetry


def test_a_limp_makes_the_two_legs_behave_differently() -> None:
    """The whole point of modelling asymmetry: a symmetric limp is not a limp."""
    normal = signature_for("alice", style="normal")
    limp = signature_for("alice", style="limp_L")

    def leg_gap(sig):
        left = [pose_at(sig, i / 60).left["hip_flex"] for i in range(60)]
        right = [pose_at(sig, i / 60).right["hip_flex"] for i in range(60)]
        return abs(np.ptp(left) - np.ptp(right))

    assert leg_gap(limp) > leg_gap(normal) + 2.0


def test_the_limping_side_is_determined_by_the_sign() -> None:
    left_limp = signature_for("alice", style="limp_L")
    right_limp = signature_for("alice", style="limp_R")

    def swing(sig, side):
        return np.ptp([getattr(pose_at(sig, i / 60), side)["hip_flex"]
                       for i in range(60)])

    assert swing(left_limp, "left") > swing(left_limp, "right")
    assert swing(right_limp, "right") > swing(right_limp, "left")


def test_asymmetry_increases_the_pelvic_drop() -> None:
    """Visible from the front, where leg motion is foreshortened — which is
    most camera angles in a real rig."""
    def drop(sig):
        return max(abs(pose_at(sig, i / 60).pelvis_drop_deg) for i in range(60))

    assert drop(signature_for("alice", style="limp_L")) > \
        drop(signature_for("alice", style="normal"))


# ------------------------------------------------------- individuality


def test_two_people_with_the_same_style_do_not_move_identically() -> None:
    """The defect that playing a shared mocap clip would reintroduce."""
    a = [pose_at(signature_for("alice"), i / 40).left["hip_flex"] for i in range(40)]
    b = [pose_at(signature_for("bob"), i / 40).left["hip_flex"] for i in range(40)]

    assert a != b
    assert float(np.abs(np.array(a) - np.array(b)).mean()) > 0.1


def test_the_same_person_moves_the_same_way_every_time() -> None:
    a = [pose_at(signature_for("alice"), i / 40).left for i in range(40)]
    b = [pose_at(signature_for("alice"), i / 40).left for i in range(40)]
    assert a == b


def test_a_crowd_can_be_decorrelated_so_it_does_not_march_in_unison() -> None:
    """Without a phase offset a crowd steps in lockstep, which reads as
    synthetic and gives motion-energy detectors a periodicity no crowd has."""
    sig = signature_for("alice")
    in_step = synthesise(sig, 1.0, 30.0, phase_offset=0.0)
    offset = synthesise(sig, 1.0, 30.0, phase_offset=0.37)

    assert in_step[0].left["hip_flex"] != offset[0].left["hip_flex"]


# -------------------------------------------------------------- sampling


def test_frame_count_follows_duration_and_frame_rate() -> None:
    sig = signature_for("alice")
    assert len(synthesise(sig, 2.0, 30.0)) == 60
    assert len(synthesise(sig, 2.0, 25.0)) == 50


def test_motion_is_continuous_so_any_frame_rate_works() -> None:
    """A clip is locked to its capture rate; this is not."""
    sig = signature_for("alice")
    assert len(synthesise(sig, 1.0, 120.0)) == 120
    assert len(synthesise(sig, 1.0, 12.0)) == 12


def test_zero_or_negative_duration_produces_nothing() -> None:
    sig = signature_for("alice")
    assert synthesise(sig, 0.0, 30.0) == []
    assert synthesise(sig, -1.0, 30.0) == []


def test_poses_serialise_for_ground_truth() -> None:
    pose = pose_at(signature_for("alice"), 0.3)
    payload = pose.as_dict()
    assert set(payload["left"]) == set(JOINTS)
    assert isinstance(payload["pelvis_sway_m"], float)


def test_harmonics_are_stated_rather_than_buried() -> None:
    """They are argued from normative gait data, so they must be inspectable."""
    assert set(NORMAL_HARMONICS) == set(JOINTS)
    for mean, harmonics in NORMAL_HARMONICS.values():
        assert len(harmonics) >= 1

"""Tests for recovering gait cadence from pose.

Cadence is easy to measure wrongly and hard to notice: any periodicity detector
returns *some* frequency for any input, including a standing person, and 110
steps per minute recovered from jitter looks exactly like 110 steps per minute
recovered from walking.

So the tests build keypoint tracks where the cadence is known by construction
and check it is recovered, then check each way of getting a fake answer is
refused: a stationary person, a person walking at the camera, a track too short
to carry cycles, and ankles that are never both visible.
"""

from __future__ import annotations

import numpy as np
import pytest

from survsim.gait.cadence import (
    L_ANKLE,
    L_HIP,
    L_SHOULDER,
    R_ANKLE,
    R_HIP,
    R_SHOULDER,
    ankle_separation_signal,
    count_steps,
    lateral_fraction,
    measure_track,
    metres_per_pixel,
    summarise,
)

FPS = 25.0


def _walker(n_frames: int = 100, cadence_spm: float = 110.0,
            speed_px_per_frame: float = 4.0, ankle_swing_px: float = 25.0,
            shoulder_to_ankle_px: float = 300.0, vertical: bool = False,
            ankle_score: float = 0.9) -> list[np.ndarray]:
    """A walker whose cadence is exactly `cadence_spm` by construction.

    The signed ankle separation completes one full period per **gait cycle**,
    which is two steps — left forward, then right forward, then back to the
    start. That gives two zero-crossings per cycle, i.e. one per step, which is
    what the counter is built to find.

    Getting this wrong is easy and was wrong here first: swinging once per
    *step* produces exactly double the intended cadence, and since the module
    and the fixture then disagreed by a clean factor of two it was obvious
    something was off. Had both shared the same mistaken model, the tests would
    have passed and every measured cadence would have been 2x reality.
    """
    cycles_per_frame = (cadence_spm / 60.0 / 2.0) / FPS
    track = []
    for f in range(n_frames):
        x = 400.0 + (0.0 if vertical else speed_px_per_frame * f)
        y = 300.0 + (speed_px_per_frame * f if vertical else 0.0)
        phase = 2.0 * np.pi * cycles_per_frame * f
        swing = ankle_swing_px * np.sin(phase)

        kp = np.zeros((17, 3))
        kp[:, 2] = 0.9
        kp[L_SHOULDER] = [x - 20, y - shoulder_to_ankle_px, 0.9]
        kp[R_SHOULDER] = [x + 20, y - shoulder_to_ankle_px, 0.9]
        kp[L_HIP] = [x - 12, y - shoulder_to_ankle_px * 0.45, 0.9]
        kp[R_HIP] = [x + 12, y - shoulder_to_ankle_px * 0.45, 0.9]
        kp[L_ANKLE] = [x + swing, y, ankle_score]
        kp[R_ANKLE] = [x - swing, y, ankle_score]
        track.append(kp)
    return track


# ------------------------------------------------------------------- scale


def test_scale_comes_from_the_body_so_perspective_is_absorbed() -> None:
    near = _walker(shoulder_to_ankle_px=400.0)[0]
    far = _walker(shoulder_to_ankle_px=200.0)[0]

    assert metres_per_pixel(far) == pytest.approx(2 * metres_per_pixel(near), rel=1e-6)


def test_scale_is_refused_when_the_joints_are_not_visible() -> None:
    """Guessing here would silently rescale one person's stride."""
    kp = _walker()[0].copy()
    kp[[L_ANKLE, R_ANKLE], 2] = 0.0
    assert metres_per_pixel(kp) is None


def test_scale_is_refused_for_a_degenerate_body_span() -> None:
    kp = _walker(shoulder_to_ankle_px=2.0)[0]
    assert metres_per_pixel(kp) is None


# ------------------------------------------------------------------ signal


def test_ankle_separation_is_signed_so_the_legs_swapping_is_visible() -> None:
    """Taking the magnitude would double the apparent step frequency."""
    signal, usable = ankle_separation_signal(_walker(n_frames=60))
    assert usable.all()
    assert signal.min() < 0 < signal.max(), "the sign must flip as legs swap"


def test_frames_without_both_ankles_are_marked_unusable() -> None:
    track = _walker(n_frames=40)
    track[5][L_ANKLE, 2] = 0.0
    _, usable = ankle_separation_signal(track)
    assert not usable[5] and usable[6]


# ------------------------------------------------------------------- steps


def test_a_known_cadence_is_recovered() -> None:
    result = measure_track(_walker(n_frames=150, cadence_spm=110.0), FPS)
    assert result is not None
    assert result.cadence_spm == pytest.approx(110.0, abs=12.0)


@pytest.mark.parametrize("cadence", [70.0, 100.0, 130.0])
def test_cadence_tracks_the_truth_across_the_walking_range(cadence) -> None:
    result = measure_track(_walker(n_frames=200, cadence_spm=cadence), FPS)
    assert result is not None
    assert result.cadence_spm == pytest.approx(cadence, rel=0.2)


def test_a_standing_person_yields_no_cadence() -> None:
    """The failure that matters: jitter around zero read as a very fast walk."""
    still = _walker(n_frames=150, ankle_swing_px=0.0, speed_px_per_frame=0.0)
    rng = np.random.default_rng(0)
    for kp in still:
        kp[[L_ANKLE, R_ANKLE], 0] += rng.normal(0, 0.8, size=2)

    assert count_steps(ankle_separation_signal(still)[0]) == 0
    assert measure_track(still, FPS) is None


def test_a_track_too_short_to_contain_cycles_is_refused() -> None:
    assert measure_track(_walker(n_frames=6), FPS) is None


def test_a_track_with_no_visible_ankles_is_refused() -> None:
    track = _walker(n_frames=120)
    for kp in track:
        kp[[L_ANKLE, R_ANKLE], 2] = 0.0
    assert measure_track(track, FPS) is None


# -------------------------------------------------------------- direction


def test_motion_across_the_camera_is_recognised_as_lateral() -> None:
    assert lateral_fraction(_walker(n_frames=80)) > 0.9


def test_motion_toward_the_camera_is_recognised_as_not_lateral() -> None:
    assert lateral_fraction(_walker(n_frames=80, vertical=True)) < 0.1


def test_cadence_is_refused_for_someone_walking_at_the_camera() -> None:
    """There is almost no ankle separation to measure; a number here is invented."""
    assert measure_track(_walker(n_frames=150, vertical=True), FPS) is None


# ------------------------------------------------------- self-consistency


def test_speed_cadence_and_stride_satisfy_their_identity() -> None:
    """speed = (cadence/60) x step length. Redundancy is the check.

    Three quantities constrained by one identity cannot all be noise and still
    agree, which is why all three are measured rather than just cadence.
    """
    result = measure_track(_walker(n_frames=200, cadence_spm=110.0,
                                   speed_px_per_frame=4.0), FPS)
    assert result is not None
    predicted = (result.cadence_spm / 60.0) * (result.stride_length_m / 2.0)
    assert predicted == pytest.approx(result.speed_mps, rel=0.3)
    assert result.consistent


def test_stride_is_a_plausible_human_length() -> None:
    result = measure_track(_walker(n_frames=200, cadence_spm=110.0), FPS)
    assert result is not None
    assert 0.5 < result.stride_length_m < 2.5


# ------------------------------------------------------------ reporting


def test_summary_reports_how_many_tracks_survived() -> None:
    cycles = [measure_track(_walker(n_frames=200, cadence_spm=c), FPS)
              for c in (95.0, 105.0, 115.0)]
    cycles = [c for c in cycles if c is not None]
    out = summarise(cycles)

    assert out["n_consistent"] == len(cycles)
    assert 80 < out["cadence_spm_mean"] < 130


def test_summary_is_honest_when_nothing_was_measurable() -> None:
    out = summarise([])
    assert out["n_tracks"] == 0
    assert "no track" in out["note"]


def test_summary_carries_its_caveat() -> None:
    cycles = [c for c in [measure_track(_walker(n_frames=200), FPS)] if c]
    assert "lateral" in summarise(cycles)["note"]

"""Tests for the background crowd.

Crowding is not scene dressing: it causes most real occlusion, it is where
trackers swap identities, and it slows everyone down in a way that changes what a
detector sees. These tests pin the properties that make a crowd worth simulating
rather than merely populous.
"""

from __future__ import annotations

import numpy as np
import pytest

from survsim.director.crowd import (
    DENSITY_BANDS,
    BackgroundCrowd,
    CrowdConfig,
    density_band,
    speed_for_density,
)

ENTRIES = [np.array([-40.0, 0.0]), np.array([0.0, -40.0])]
EXITS = [np.array([40.0, 0.0]), np.array([0.0, 40.0])]


def _crowd(**kw) -> BackgroundCrowd:
    cfg = CrowdConfig(**{"target_density_per_m2": 0.3, "area_m2": 400.0,
                         "seed": 1, **kw})
    return BackgroundCrowd(cfg, ENTRIES, EXITS)


# ------------------------------------------------------------- density


def test_density_bands_match_crowd_safety_language() -> None:
    """These are the levels a venue's safety case is written against."""
    assert density_band(0.1) == "free_flow"
    assert density_band(0.5) == "busy"
    assert density_band(1.5) == "crowded"
    assert density_band(3.0) == "dense"
    assert density_band(4.5) == "critical"


def test_every_band_is_reachable() -> None:
    for name, low, high in DENSITY_BANDS:
        assert density_band((low + min(high, low + 1.0)) / 2) in {n for n, _, _ in DENSITY_BANDS}


def test_target_count_follows_density_and_area() -> None:
    assert CrowdConfig(target_density_per_m2=0.5, area_m2=1000.0).target_count == 500
    assert CrowdConfig(target_density_per_m2=0.0, area_m2=1000.0).target_count == 0


def test_crowd_is_created_at_the_requested_density() -> None:
    crowd = _crowd(target_density_per_m2=0.5, area_m2=200.0)
    assert crowd.report()["n_agents"] == 100
    assert crowd.density == pytest.approx(0.5, abs=0.01)


# ------------------------------------------------- fundamental diagram


def test_speed_falls_as_density_rises() -> None:
    """A crowded scene is not a sparse scene with more people: everyone is slower."""
    free = speed_for_density(1.34, 0.1)
    crowded = speed_for_density(1.34, 2.0)
    jammed = speed_for_density(1.34, 5.0)

    assert free > crowded > jammed
    assert free == pytest.approx(1.31, abs=0.02)
    assert jammed < 0.2


def test_speed_never_goes_negative_at_extreme_density() -> None:
    assert speed_for_density(1.34, 20.0) > 0.0


def test_simulated_crowd_moves_slower_when_dense() -> None:
    sparse = _crowd(target_density_per_m2=0.1, area_m2=400.0)
    dense = _crowd(target_density_per_m2=3.0, area_m2=400.0)
    for _ in range(10):
        sparse.step(0.2)
        dense.step(0.2)
    assert dense.report()["mean_speed"] < sparse.report()["mean_speed"]


# --------------------------------------------------------------- motion


def test_agents_move_toward_their_exit() -> None:
    crowd = _crowd(target_density_per_m2=0.05, area_m2=400.0)
    agent = crowd.agents[0]
    before = float(np.linalg.norm(agent.goal - agent.pos))
    for _ in range(20):
        crowd.step(0.2)
    after = float(np.linalg.norm(agent.goal - agent.pos))
    assert after < before


def test_agents_keep_some_personal_space() -> None:
    """Overlapping bodies would produce impossible occlusion patterns."""
    crowd = _crowd(target_density_per_m2=0.8, area_m2=400.0)
    for _ in range(30):
        crowd.step(0.2)

    positions = np.array([a.pos for a in crowd.agents if a.active])
    if len(positions) > 1:
        d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        # Not a hard constraint — crowds do compress — but wholesale overlap
        # would mean the avoidance term is not working at all.
        assert np.median(d.min(axis=1)) > 0.15


def test_crowd_is_spread_along_its_route_from_the_start() -> None:
    """Otherwise the opening frames are a pile of people at one entrance."""
    crowd = _crowd(target_density_per_m2=0.5, area_m2=400.0)
    positions = np.array([a.pos for a in crowd.agents])
    assert positions.std(axis=0).max() > 5.0


def test_respawn_holds_density_steady_over_a_long_run() -> None:
    crowd = _crowd(target_density_per_m2=0.5, area_m2=400.0, respawn=True)
    start = crowd.report()["n_agents"]
    for _ in range(300):
        crowd.step(0.2)
    assert crowd.report()["n_agents"] == pytest.approx(start, rel=0.2)


def test_without_respawn_the_crowd_drains() -> None:
    crowd = _crowd(target_density_per_m2=0.5, area_m2=400.0, respawn=False)
    start = crowd.report()["n_agents"]
    for _ in range(300):
        crowd.step(0.2)
    assert crowd.report()["n_agents"] < start


# ---------------------------------------------------------- ground truth


def test_background_agents_are_marked_and_carry_no_behaviour_label() -> None:
    """Inventing 'normal' for them would pad any behaviour dataset with
    thousands of people who were never given a behaviour."""
    crowd = _crowd()
    crowd.step(0.2)
    rows = crowd.rows(tick=1, t=0.2)

    assert rows
    assert all(r["background"] is True for r in rows)
    assert all(r["label"] is None for r in rows)


def test_rows_carry_what_a_detector_needs() -> None:
    crowd = _crowd()
    crowd.step(0.2)
    row = crowd.rows(1, 0.2)[0]
    assert {"uid", "x", "y", "speed", "height_m"} <= set(row)
    assert 1.2 <= row["height_m"] <= 2.05


def test_identities_are_unique_including_after_respawn() -> None:
    crowd = _crowd(target_density_per_m2=0.4, area_m2=400.0)
    seen: set[str] = set()
    for tick in range(200):
        crowd.step(0.2)
        for row in crowd.rows(tick, tick * 0.2):
            seen.add(row["uid"])
    assert len(seen) >= crowd.report()["n_agents"]


def test_report_states_why_there_are_no_labels() -> None:
    assert "no behaviour label" in _crowd().report()["note"]


def test_empty_crowd_is_handled() -> None:
    crowd = _crowd(target_density_per_m2=0.0)
    crowd.step(0.2)
    assert crowd.rows(1, 0.2) == []
    assert crowd.report()["n_agents"] == 0


# ------------------------------------------------- cadence versus step length
#
# Measured from pose on real CCTV (docs/14 §10): in a crowd people keep their
# leg rhythm and shorten their stride. A slow crowd is not a slow-motion crowd,
# and these tests exist so nobody re-derives crowded walking as "the same walk,
# played slower".


from survsim.director.crowd import (
    FREE_CADENCE_SPM,
    FREE_STEP_LENGTH_M,
    gait_for_density,
)


def test_cadence_is_preserved_while_step_length_collapses() -> None:
    """The headline measurement: 123 spm with a 0.39 m step in a real crowd."""
    free_cadence, free_step = gait_for_density(0.05)
    crowd_cadence, crowd_step = gait_for_density(2.0)

    assert crowd_step < free_step * 0.7, "steps must shorten substantially"
    assert crowd_cadence >= free_cadence, "leg rhythm is not what slows down"


def test_a_crowd_is_not_a_slow_motion_crowd() -> None:
    """Halving speed by halving cadence would look obviously synthetic."""
    _, _free_step = gait_for_density(0.05)
    dense_cadence, _dense_step = gait_for_density(3.0)

    speed_ratio = speed_for_density(1.34, 3.0) / speed_for_density(1.34, 0.05)
    cadence_ratio = dense_cadence / gait_for_density(0.05)[0]

    assert speed_ratio < 0.6, "speed really should fall a lot"
    assert cadence_ratio > 0.9, "cadence should not fall with it"


def test_cadence_times_step_length_reproduces_the_speed_model() -> None:
    """The identity must hold, or the three quantities disagree with each other."""
    for density in (0.1, 0.5, 1.0, 2.0, 3.5):
        cadence, step = gait_for_density(density)
        implied = cadence / 60.0 * step
        expected = speed_for_density(FREE_CADENCE_SPM / 60.0 * FREE_STEP_LENGTH_M,
                                     density)
        assert implied == pytest.approx(expected, rel=1e-6), f"at {density}/m2"


def test_free_walking_matches_the_literature_figures() -> None:
    cadence, step = gait_for_density(0.0)
    assert cadence == pytest.approx(FREE_CADENCE_SPM, rel=0.01)
    assert step == pytest.approx(FREE_STEP_LENGTH_M, rel=0.02)


def test_cadence_rise_saturates_rather_than_growing_without_bound() -> None:
    """People cannot keep increasing step rate as space runs out."""
    at_two = gait_for_density(2.0)[0]
    at_five = gait_for_density(5.0)[0]
    assert at_five == pytest.approx(at_two, rel=0.1)
    assert at_five < FREE_CADENCE_SPM * 1.3


def test_step_length_never_goes_negative_at_extreme_density() -> None:
    cadence, step = gait_for_density(50.0)
    assert step > 0.0 and cadence > 0.0

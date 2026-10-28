"""Tests for person-on-person occlusion in the mock backend.

Without this, every difficulty figure from a mock run is optimistic: the dataset
claims a clear view of people who are standing directly behind someone else.
Crowding is the commonest source of occlusion in surveillance footage, so getting
it approximately right matters more than getting it pixel-perfect.
"""

from __future__ import annotations

import numpy as np
import pytest

from survsim.backends.mock.backend import MockBackend
from survsim.core.models import (
    Camera,
    Environment,
    Mount,
    Person,
    Population,
    Prop,
    Scenario,
    Zone,
)
from survsim.director.world import World


def _world(positions: dict[str, tuple[float, float]],
           props: list[Prop] | None = None) -> tuple[World, Camera, MockBackend]:
    env = Environment(
        id="e",
        zones=[Zone(id="zone.a", tag="public",
                    polygon=[(-30, -30), (30, -30), (30, 30), (-30, 30)])],
        props=props or [],
    )
    people = [Person(id=uid) for uid in positions]
    cam = Camera(
        id="c", mount=Mount(pos=(0.0, 0.0, 2.0), yaw=0.0, pitch=-3.0),
        optics={"hfov_deg": 70}, capture={"res": (1280, 720)},
    )
    sc = Scenario(name="s", environment=env, cameras=[cam], population=Population(hero=people))
    world = World(sc)
    for uid, (x, y) in positions.items():
        world.agents[uid].pos = np.array([float(x), float(y)])

    backend = MockBackend()
    backend.open(sc)
    return world, cam, backend


def _by_uid(rows):
    return {r["uid"]: r for r in rows}


def test_isolated_person_is_fully_visible() -> None:
    world, cam, backend = _world({"alone": (14.0, 0.0)})
    row = _by_uid(backend.observe(world, cam))["alone"]
    assert row["occlusion"] == pytest.approx(0.0, abs=0.02)
    assert row["visible_frac"] > 0.95
    assert row["fully_occluded"] is False


def test_person_directly_behind_another_is_occluded() -> None:
    """The case that makes naive mock ground truth wrong."""
    world, cam, backend = _world({"front": (8.0, 0.0), "behind": (16.0, 0.0)})
    rows = _by_uid(backend.observe(world, cam))

    assert rows["front"]["occlusion"] == pytest.approx(0.0, abs=0.05)
    assert rows["behind"]["occlusion"] > 0.5
    assert rows["behind"]["visible_frac"] < rows["front"]["visible_frac"]


def test_the_nearer_person_is_the_one_that_occludes() -> None:
    """Depth ordering must decide, not list order."""
    near_first, cam, backend = _world({"near": (7.0, 0.0), "far": (15.0, 0.0)})
    rows = _by_uid(backend.observe(near_first, cam))
    assert rows["near"]["range_m"] < rows["far"]["range_m"]
    assert rows["far"]["occlusion"] > rows["near"]["occlusion"]


def test_people_side_by_side_do_not_occlude_each_other() -> None:
    world, cam, backend = _world({"left": (14.0, -3.0), "right": (14.0, 3.0)})
    rows = _by_uid(backend.observe(world, cam))
    assert rows["left"]["occlusion"] < 0.1
    assert rows["right"]["occlusion"] < 0.1


def test_fully_hidden_person_is_flagged_not_dropped() -> None:
    """A tracker is expected to carry someone through a full occlusion.

    Deleting them from ground truth would make that behaviour untestable and
    would quietly reward a tracker that also loses them.
    """
    world, cam, backend = _world({"screen": (6.0, 0.0), "hidden": (6.6, 0.0)})
    rows = _by_uid(backend.observe(world, cam))

    assert "hidden" in rows, "an occluded person must remain in ground truth"
    assert rows["hidden"]["occlusion"] > 0.9


def test_occlusion_grows_as_someone_steps_in_front() -> None:
    occlusions = []
    for lateral in (6.0, 3.0, 1.0, 0.0):
        world, cam, backend = _world({"screen": (8.0, lateral), "target": (15.0, 0.0)})
        occlusions.append(_by_uid(backend.observe(world, cam))["target"]["occlusion"])
    assert occlusions == sorted(occlusions), f"should increase monotonically: {occlusions}"


def test_truncation_is_reported_separately_from_occlusion() -> None:
    """Walking out of frame is a different problem from being hidden."""
    # At 6 m with a 70 deg lens the half-width is about 4.2 m, so this straddles
    # the frame edge rather than falling entirely outside it.
    world, cam, backend = _world({"edge": (6.0, 4.1)})
    row = _by_uid(backend.observe(world, cam))["edge"]
    assert row["truncation"] > 0.0
    assert row["occlusion"] == pytest.approx(0.0, abs=0.05)


def test_crowd_produces_a_spread_of_occlusion_levels() -> None:
    """A dense group should yield partial occlusions, not all-or-nothing.

    The lateral spread is deliberately tight. Silhouettes overlap less than
    bounding boxes do, so a loosely scattered group produces almost no partial
    occlusion — which is correct, and means testing the partial case requires
    people genuinely standing behind one another.
    """
    rng = np.random.default_rng(0)
    positions = {f"p{i}": (8.0 + rng.uniform(0, 10), rng.uniform(-0.6, 0.6))
                 for i in range(14)}
    world, cam, backend = _world(positions)
    values = [r["occlusion"] for r in backend.observe(world, cam)]

    assert max(values) > 0.2
    assert sum(0.05 < v < 0.95 for v in values) >= 2     # genuinely partial cases


# ------------------------------------------------------------------- scenery


def test_a_pillar_occludes_the_person_behind_it() -> None:
    """A car park is mostly pillars; a rig scored without them looks far too good."""
    clear, cam, backend = _world({"p": (14.0, 0.0)})
    baseline = _by_uid(backend.observe(clear, cam))["p"]["occlusion"]

    blocked, cam2, backend2 = _world(
        {"p": (14.0, 0.0)},
        props=[Prop(id="pillar_1", type="pillar", pos=(9.0, 0.0, 0.0))])
    behind = _by_uid(backend2.observe(blocked, cam2))["p"]["occlusion"]

    assert baseline < 0.05
    assert behind > 0.3


def test_a_person_in_front_of_scenery_is_unaffected() -> None:
    world, cam, backend = _world(
        {"p": (6.0, 0.0)},
        props=[Prop(id="pillar_1", type="pillar", pos=(14.0, 0.0, 0.0))])
    assert _by_uid(backend.observe(world, cam))["p"]["occlusion"] < 0.05


def test_unknown_prop_types_do_not_occlude() -> None:
    """Only things with a declared size can hide someone; a waypoint marker cannot."""
    world, cam, backend = _world(
        {"p": (14.0, 0.0)},
        props=[Prop(id="marker", type="camera_mount", pos=(9.0, 0.0, 0.0))])
    assert _by_uid(backend.observe(world, cam))["p"]["occlusion"] < 0.05


def test_fence_panel_uses_its_declared_length() -> None:
    # The person stands off to the side, so only a panel wide enough to reach
    # them occludes. Head-on, any fence of that height hides them regardless of
    # width, which would not test the length at all.
    narrow, cam, backend = _world(
        {"p": (14.0, 2.0)},
        props=[Prop(id="f", type="fence_panel", pos=(9.0, 0.0, 0.0), length_m=1.0)])
    wide, cam2, backend2 = _world(
        {"p": (14.0, 2.0)},
        props=[Prop(id="f", type="fence_panel", pos=(9.0, 0.0, 0.0), length_m=8.0)])

    assert (_by_uid(backend2.observe(wide, cam2))["p"]["occlusion"]
            > _by_uid(backend.observe(narrow, cam))["p"]["occlusion"])

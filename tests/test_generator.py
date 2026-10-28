"""Tests for the scene generator: valid by construction, diverse by design."""

from __future__ import annotations

import pytest

from survsim.generator import (
    CameraPlan,
    IncidentPlan,
    PopulationPlan,
    SceneSpec,
    SiteSpec,
    compose,
    generate_batch,
)
from survsim.generator.place import verify_blind_spots
from survsim.generator.spec import SiteType


def _spec(site="office_building", **kw) -> SceneSpec:
    return SceneSpec(
        name=f"t_{site}", seed=kw.pop("seed", 1), duration_s=90,
        site=SiteSpec(type=site, entrances=kw.pop("entrances", 3), blind_spots=1),
        cameras=CameraPlan(count=kw.pop("cameras", 6), target_coverage=0.5),
        population=PopulationPlan(count=kw.pop("people", 14)),
        incidents=[IncidentPlan(type="disguise_swap"), IncidentPlan(type="loiter")],
    )


@pytest.mark.slow
def test_every_site_type_generates_a_valid_scene() -> None:
    for i, site in enumerate(SiteType.__args__):  # type: ignore[attr-defined]
        scene = compose(_spec(site, seed=10 + i))
        assert scene.valid, f"{site}: {scene.issues}"


def test_cameras_are_not_pointed_at_nothing() -> None:
    """Greedy coverage selection cannot choose an aim that sees no floor."""
    scene = compose(_spec())
    assert scene.reports["placement"]["area_coverage"] > 0.5
    assert len(scene.scenario.cameras) == 6


def test_declared_blind_spots_are_actually_blind() -> None:
    """The disguise narrative is meaningless if the blind spot is observed."""
    scene = compose(_spec())
    report = verify_blind_spots(scene.scenario.cameras, scene.scenario.environment)
    assert report, "site should declare a blind spot"
    for zid, info in report.items():
        assert info["truly_blind"], f"{zid} observed in {info['observed_cells']} cells"


def test_disguise_scene_has_two_entrances_used() -> None:
    """Entering and leaving by the same door makes the arc trivially linkable."""
    scene = compose(_spec(entrances=4, seed=5))
    swaps = [a for a in scene.arcs if a.narrative == "disguise_swap"]
    assert swaps
    steps = swaps[0].person.intent
    assert any(s.type == "outfit_change" for s in steps)


def test_every_incident_actor_has_a_benign_twin() -> None:
    scene = compose(_spec())
    hostile = [a for a in scene.arcs if a.intent == "hostile"]
    twins = [a for a in scene.arcs if a.person.hard_negative_for]
    assert hostile and len(twins) >= len(hostile)


def test_appearance_does_not_predict_intent() -> None:
    scene = compose(_spec())
    audit = scene.reports.get("confound_audit")
    assert audit is not None
    assert audit["balanced"], audit["worst_imbalance"]
    assert audit["protected_clean"]


def test_all_intent_references_resolve() -> None:
    """A scene referencing a zone that does not exist would crash mid-render."""
    scene = compose(_spec())
    assert not [i for i in scene.issues if "unknown" in i]


@pytest.mark.slow
def test_batch_is_diverse_and_all_valid() -> None:
    result = generate_batch(n=8, seed=3, duration_s=60)
    d = result.diversity()
    assert d["all_valid"], [s.issues for s in result.scenes if not s.valid]
    # Stratification must actually spread the design space, not cluster.
    assert d["distinct_site_types"] >= 4
    assert d["camera_count_range"][1] > d["camera_count_range"][0]
    assert d["people_range"][1] > d["people_range"][0]
    assert d["distinct_narratives"] >= 4


@pytest.mark.slow
def test_batch_is_deterministic() -> None:
    a = generate_batch(n=4, seed=11, duration_s=60)
    b = generate_batch(n=4, seed=11, duration_s=60)
    assert [s.scenario.name for s in a.scenes] == [s.scenario.name for s in b.scenes]
    assert [len(s.scenario.cameras) for s in a.scenes] == [len(s.scenario.cameras) for s in b.scenes]


def test_validation_rejects_a_deliberately_broken_scene() -> None:
    """The gate must catch a camera-starved site, or it is decoration."""
    spec = _spec(cameras=1)
    spec.cameras.target_coverage = 0.99
    scene = compose(spec)
    assert not scene.valid

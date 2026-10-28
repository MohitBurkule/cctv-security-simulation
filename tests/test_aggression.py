"""Tests for aggression and emergency behaviours.

Aggression is the capability the vendor names explicitly, and it is different in kind
from loitering: it is a property of a *pair*, and its strongest cue is often the
reaction of the people around it rather than the act itself. These tests pin the
signatures a detector would key on.
"""

from __future__ import annotations

import numpy as np

from survsim.core.models import (
    BehaviourSpec,
    Environment,
    Person,
    Population,
    Scenario,
    Waypoint,
    Zone,
)
from survsim.director.behaviours import REGISTRY
from survsim.director.world import World


def _scenario(people: list[Person], duration_s: float = 40.0) -> Scenario:
    env = Environment(
        id="e",
        zones=[Zone(id="zone.open", tag="public",
                    polygon=[(-40, -40), (40, -40), (40, 40), (-40, 40)])],
        spawn_points=[Waypoint(id="spawn.a", pos=(-10.0, 0.0, 0.0))],
        exits=[Waypoint(id="exit.a", pos=(20.0, 0.0, 0.0))],
    )
    return Scenario(name="s", seed=1, duration_s=duration_s, fps=5,
                    environment=env, population=Population(hero=people))


def _run(world: World, ticks: int) -> None:
    for _ in range(ticks):
        world.step()


def test_all_aggression_behaviours_are_registered() -> None:
    for name in ("argument", "fight", "theft_snatch", "person_down", "flee",
                 "group_walk", "counter_flow"):
        assert name in REGISTRY


# --------------------------------------------------------------- argument


def test_argument_closes_the_gap_between_two_people() -> None:
    """Escalation shows up as shrinking personal distance, which is detectable."""
    a = Person(id="a", intent=[BehaviourSpec(type="argument", partner="b", dur_s=20.0)])
    b = Person(id="b", intent=[BehaviourSpec(type="argument", partner="a", dur_s=20.0)])
    world = World(_scenario([a, b]))
    world.agents["a"].pos = np.array([0.0, 0.0])
    world.agents["b"].pos = np.array([4.0, 0.0])

    world.step()
    start_gap = float(np.linalg.norm(world.agents["a"].pos - world.agents["b"].pos))
    _run(world, 60)
    end_gap = float(np.linalg.norm(world.agents["a"].pos - world.agents["b"].pos))

    assert end_gap < start_gap
    assert world.agents["a"].label.startswith("hostile.aggression")


def test_argument_records_a_pairwise_interaction() -> None:
    """A per-person record would lose the structure that defines aggression."""
    a = Person(id="a", intent=[BehaviourSpec(type="argument", partner="b", dur_s=20.0)])
    b = Person(id="b", intent=[BehaviourSpec(type="argument", partner="a", dur_s=20.0)])
    world = World(_scenario([a, b]))
    _run(world, 10)

    assert ("a", "b") in world.interactions
    assert world.interactions[("a", "b")]["kind"] == "argument"


# ------------------------------------------------------------------- fight


def test_fight_is_fast_motion_inside_a_small_radius() -> None:
    """The signature no ordinary pedestrian produces."""
    a = Person(id="a", gait={"speed_mps": 1.3},
               intent=[BehaviourSpec(type="fight", partner="b", dur_s=20.0)])
    b = Person(id="b", gait={"speed_mps": 1.3},
               intent=[BehaviourSpec(type="fight", partner="a", dur_s=20.0)])
    world = World(_scenario([a, b]))
    world.agents["a"].pos = np.array([0.0, 0.0])
    world.agents["b"].pos = np.array([1.0, 0.0])

    speeds, positions = [], []
    for _ in range(40):
        world.step()
        speeds.append(float(np.linalg.norm(world.agents["a"].vel)))
        positions.append(world.agents["a"].pos.copy())

    spread = float(np.linalg.norm(np.ptp(np.array(positions), axis=0)))
    assert max(speeds) > 1.3          # faster than a walk
    assert spread < 12.0              # but confined


def test_fight_ends_and_restores_posture() -> None:
    a = Person(id="a", intent=[BehaviourSpec(type="fight", partner="b", dur_s=3.0)])
    b = Person(id="b", intent=[BehaviourSpec(type="fight", partner="a", dur_s=3.0)])
    world = World(_scenario([a, b]))
    _run(world, 40)
    assert world.agents["a"].posture == "upright"


# ------------------------------------------------------------------- theft


def test_snatch_transfers_the_object_and_flees() -> None:
    thief = Person(id="thief", gait={"speed_mps": 1.4}, intent=[
        BehaviourSpec(type="theft_snatch", victim="victim", object="handbag",
                      exit="exit.a")])
    victim = Person(id="victim", intent=[BehaviourSpec(type="commute_wait",
                                                       zone="zone.open", dur_s=60.0)])
    world = World(_scenario([thief, victim]))
    world.agents["thief"].pos = np.array([0.0, 0.0])
    world.agents["victim"].pos = np.array([3.0, 0.0])
    _run(world, 120)

    transfers = [r for r in world.sensor_log if r.get("event") == "transfer"]
    assert transfers, "the handbag should have changed hands"
    assert transfers[0]["from"] == "victim" and transfers[0]["to"] == "thief"

    obj = next(o for o in world.objects.values() if o.get("transferred"))
    assert obj["owner"] == "thief" and obj["original_owner"] == "victim"


# ------------------------------------------------------------ person down


def test_person_down_changes_posture_not_position() -> None:
    """Posture is the entire signal: a prone person breaks upright detectors."""
    p = Person(id="p", intent=[BehaviourSpec(type="person_down", dur_s=20.0,
                                             fall_s=1.0)])
    world = World(_scenario([p]))
    world.agents["p"].pos = np.array([5.0, 5.0])
    start = world.agents["p"].pos.copy()

    world.step()
    assert world.agents["p"].posture == "falling"
    _run(world, 20)
    assert world.agents["p"].posture == "prone"
    assert np.allclose(world.agents["p"].pos, start, atol=0.05)


def test_person_down_recovers_when_the_behaviour_ends() -> None:
    p = Person(id="p", intent=[BehaviourSpec(type="person_down", dur_s=2.0)])
    world = World(_scenario([p]))
    _run(world, 40)
    assert world.agents["p"].posture == "upright"


# -------------------------------------------------------------------- flee


def test_flee_moves_away_from_the_threat() -> None:
    """Bystanders scattering is often the first visible sign of an incident."""
    runner = Person(id="runner", gait={"speed_mps": 1.2},
                    intent=[BehaviourSpec(type="flee", threat="threat", to="exit.a")])
    threat = Person(id="threat", intent=[BehaviourSpec(type="commute_wait",
                                                       zone="zone.open", dur_s=60.0)])
    world = World(_scenario([runner, threat]))
    world.agents["threat"].pos = np.array([0.0, 0.0])
    world.agents["runner"].pos = np.array([2.0, 0.0])

    before = float(np.linalg.norm(world.agents["runner"].pos - world.agents["threat"].pos))
    _run(world, 25)
    after = float(np.linalg.norm(world.agents["runner"].pos - world.agents["threat"].pos))

    assert after > before + 3.0


def test_flee_is_faster_than_walking() -> None:
    runner = Person(id="runner", gait={"speed_mps": 1.2},
                    intent=[BehaviourSpec(type="flee", to="exit.a")])
    world = World(_scenario([runner]))
    world.agents["runner"].pos = np.array([-15.0, 0.0])
    world.step()
    assert float(np.linalg.norm(world.agents["runner"].vel)) > 1.2


# -------------------------------------------------------------- group walk


def test_group_members_stay_near_their_leader() -> None:
    """Groups occlude each other and swap identities — the tracker's hard case."""
    leader = Person(id="leader", intent=[BehaviourSpec(type="transit", to="exit.a")])
    follower = Person(id="follower", intent=[
        BehaviourSpec(type="group_walk", leader="leader", offset=[1.0, 0.8],
                      to="exit.a")])
    world = World(_scenario([leader, follower]))
    world.agents["leader"].pos = np.array([-10.0, 0.0])
    world.agents["follower"].pos = np.array([-10.0, 2.0])
    _run(world, 60)

    gap = float(np.linalg.norm(world.agents["leader"].pos - world.agents["follower"].pos))
    assert gap < 4.0

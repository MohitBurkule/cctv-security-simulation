"""Tests for queue, climb, vandalism, rush, throw, vehicle and drone behaviours.

Each of these breaks an assumption the rest of the pipeline makes, and that is
why they are worth simulating rather than merely listing: queues violate personal
space, climbing is vertical motion with no ground translation, vandalism is a
moving arm on a stationary body, and a drone has no foot contact at all.
"""

from __future__ import annotations

import numpy as np

from survsim.core.models import (
    BehaviourSpec,
    Environment,
    Person,
    Population,
    Prop,
    Scenario,
    Waypoint,
    Zone,
)
from survsim.director.behaviours import REGISTRY
from survsim.director.world import World


def _scenario(people: list[Person]) -> Scenario:
    env = Environment(
        id="e",
        zones=[Zone(id="zone.open", tag="public",
                    polygon=[(-60, -60), (60, -60), (60, 60), (-60, 60)]),
               Zone(id="zone.gate", tag="gate",
                    polygon=[(18, -4), (26, -4), (26, 4), (18, 4)])],
        spawn_points=[Waypoint(id="spawn.a", pos=(-20.0, 0.0, 0.0))],
        exits=[Waypoint(id="exit.a", pos=(40.0, 0.0, 0.0))],
        props=[Prop(id="fence_1", type="fence_panel", pos=(10.0, 0.0, 0.0), length_m=3.0),
               Prop(id="wall_1", type="wall", pos=(6.0, 6.0, 0.0))],
    )
    return Scenario(name="s", seed=1, duration_s=80, fps=5,
                    environment=env, population=Population(hero=people))


def _run(world: World, ticks: int) -> None:
    for _ in range(ticks):
        world.step()


def test_all_new_behaviours_are_registered() -> None:
    for name in ("queue", "climb_attempt", "vandalism", "crowd_rush",
                 "throw_object", "vehicle_move", "drone_overflight"):
        assert name in REGISTRY


# -------------------------------------------------------------------- queue


def test_queue_members_hold_station_behind_the_person_ahead() -> None:
    """Queues violate personal space, which is why they occlude so heavily."""
    lead = Person(id="lead", intent=[BehaviourSpec(type="queue", zone="zone.gate",
                                                    dur_s=60.0)])
    back = Person(id="back", intent=[BehaviourSpec(type="queue", zone="zone.gate",
                                                    ahead="lead", spacing_m=0.7,
                                                    dur_s=60.0)])
    world = World(_scenario([lead, back]))
    world.agents["lead"].pos = np.array([22.0, 0.0])
    world.agents["back"].pos = np.array([26.0, 0.0])
    _run(world, 60)

    gap = float(np.linalg.norm(world.agents["back"].pos - world.agents["lead"].pos))
    assert gap < 2.0, "should have closed up behind the person ahead"


def test_queue_movement_is_slower_than_walking() -> None:
    p = Person(id="p", gait={"speed_mps": 1.3},
               intent=[BehaviourSpec(type="queue", zone="zone.gate", dur_s=60.0)])
    world = World(_scenario([p]))
    world.agents["p"].pos = np.array([40.0, 0.0])
    world.step()
    assert float(np.linalg.norm(world.agents["p"].vel)) < 1.3


# -------------------------------------------------------------------- climb


def test_climbing_is_vertical_motion_with_no_ground_translation() -> None:
    """Invisible to any detector that only tracks position on the ground plane."""
    p = Person(id="p", intent=[BehaviourSpec(type="climb_attempt", barrier="fence_1",
                                             dur_s=3.0, succeeds=False)])
    world = World(_scenario([p]))
    world.agents["p"].pos = np.array([8.0, 0.0])
    _run(world, 8)

    assert world.agents["p"].posture == "climbing"
    assert float(np.linalg.norm(world.agents["p"].vel)) < 0.05


def test_successful_climb_records_a_breach() -> None:
    p = Person(id="p", intent=[BehaviourSpec(type="climb_attempt", barrier="fence_1",
                                             dur_s=1.0, succeeds=True)])
    world = World(_scenario([p]))
    world.agents["p"].pos = np.array([9.5, 0.0])
    _run(world, 40)

    assert any(r.get("event") == "breach" for r in world.sensor_log)
    assert world.agents["p"].posture == "upright"


# ---------------------------------------------------------------- vandalism


def test_vandalism_is_a_moving_arm_on_a_stationary_body() -> None:
    """A trajectory model sees someone standing still; this needs pose."""
    p = Person(id="p", intent=[BehaviourSpec(type="vandalism", target="wall_1",
                                             dur_s=20.0)])
    world = World(_scenario([p]))
    world.agents["p"].pos = np.array([6.0, 5.0])
    _run(world, 30)

    positions = []
    for _ in range(10):
        world.step()
        positions.append(world.agents["p"].pos.copy())
    spread = float(np.ptp(np.array(positions), axis=0).max())
    assert spread < 1.0, "should stay essentially in place"


def test_vandalism_records_damage() -> None:
    p = Person(id="p", intent=[BehaviourSpec(type="vandalism", target="wall_1",
                                             dur_s=2.0)])
    world = World(_scenario([p]))
    world.agents["p"].pos = np.array([6.0, 5.5])
    _run(world, 60)
    assert any(r.get("event") == "damage" for r in world.sensor_log)


# --------------------------------------------------------------- crowd rush


def test_rush_is_much_faster_than_walking() -> None:
    p = Person(id="p", gait={"speed_mps": 1.3},
               intent=[BehaviourSpec(type="crowd_rush", target="zone.gate")])
    world = World(_scenario([p]))
    world.agents["p"].pos = np.array([-20.0, 0.0])
    world.step()
    assert float(np.linalg.norm(world.agents["p"].vel)) > 2.0


def test_rush_reaches_its_target() -> None:
    p = Person(id="p", intent=[BehaviourSpec(type="crowd_rush", target="zone.gate")])
    world = World(_scenario([p]))
    world.agents["p"].pos = np.array([-10.0, 0.0])
    _run(world, 60)
    assert world.agents["p"].pos[0] > 10.0


# ------------------------------------------------------------------- throw


def test_throw_has_a_windup_then_releases_an_object() -> None:
    p = Person(id="p", intent=[BehaviourSpec(type="throw_object", target="zone.gate",
                                             windup_s=1.0, object="brick")])
    world = World(_scenario([p]))
    world.agents["p"].pos = np.array([0.0, 0.0])

    world.step()
    assert world.agents["p"].posture == "windup"
    _run(world, 20)
    assert world.objects, "the object should have been released"


# ----------------------------------------------------------------- vehicle


def test_vehicle_moves_far_faster_than_a_pedestrian() -> None:
    v = Person(id="v", gait={"speed_mps": 1.3},
               intent=[BehaviourSpec(type="vehicle_move", to="exit.a", speed_mps=10.0)])
    world = World(_scenario([v]))
    world.agents["v"].pos = np.array([-30.0, 0.0])
    world.step()
    assert float(np.linalg.norm(world.agents["v"].vel)) > 5.0
    assert world.agents["v"].posture == "vehicle"


# ------------------------------------------------------------------- drone


def test_drone_flies_above_the_ground_plane() -> None:
    """A drone has no foot contact, so ground homography gives nonsense for it.

    Recording altitude explicitly is what lets downstream code notice, instead of
    silently projecting an airborne object onto the floor.
    """
    d = Person(id="d", intent=[BehaviourSpec(type="drone_overflight", to="exit.a",
                                             altitude_m=45.0, speed_mps=12.0)])
    world = World(_scenario([d]))
    world.agents["d"].pos = np.array([-40.0, 0.0])
    world.step()

    agent = world.agents["d"]
    assert agent.altitude_m == 45.0
    assert agent.pos3[2] == 45.0          # not on the ground plane
    assert agent.posture == "airborne"


def test_ground_actors_stay_at_zero_altitude() -> None:
    p = Person(id="p", intent=[BehaviourSpec(type="transit", to="exit.a")])
    world = World(_scenario([p]))
    world.step()
    assert world.agents["p"].altitude_m == 0.0
    assert world.agents["p"].pos3[2] == 0.0

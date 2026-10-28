"""Tests for the rest of the behaviour catalogue.

Most of these behaviours are *benign twins* of hostile ones, so the tests are
written around the pairs rather than around each class alone. Two properties
matter and they pull in opposite directions:

  - where a benign and a hostile behaviour really are mechanically identical,
    the test asserts that they *stay* identical, so nobody quietly adds a motion
    cue that lets a classifier separate a plumber from a saboteur by trajectory;
  - where a genuine discriminator exists (the shape of the gap in a greeting
    versus an argument), the test pins that discriminator, because it is the
    only thing a detector is entitled to use.
"""

from __future__ import annotations

import numpy as np
import pytest

from survsim.core.models import (
    BehaviourSpec,
    Camera,
    Environment,
    Mount,
    Person,
    Population,
    Prop,
    Scenario,
    Waypoint,
    Zone,
)
from survsim.director.behaviours import REGISTRY
from survsim.director.world import World

NEW = ("meet_and_greet", "object_handoff", "camera_avoidance", "maintenance",
       "asset_tamper", "smoke_break", "photograph", "lost_tourist", "child_play",
       "drunk_walk", "disabled_mobility", "sit_rest", "weather_reaction",
       "crowd_form", "vehicle_dropoff_loiter", "vehicle_ram", "night_presence")


def _scenario(people: list[Person], cameras: list[Camera] | None = None) -> Scenario:
    env = Environment(
        id="e",
        zones=[Zone(id="zone.open", tag="public",
                    polygon=[(-60, -60), (60, -60), (60, 60), (-60, 60)]),
               # Unfenced: these tests are about gait/avoidance, and the yard sits
               # on the straight spawn->exit line (fences: tests/test_fences.py).
               Zone(id="zone.yard", tag="restricted", fenced=False,
                    polygon=[(10, -6), (26, -6), (26, 6), (10, 6)])],
        spawn_points=[Waypoint(id="spawn.a", pos=(-20.0, 0.0, 0.0))],
        exits=[Waypoint(id="exit.a", pos=(40.0, 0.0, 0.0))],
        props=[Prop(id="cabinet_1", type="cabinet", pos=(6.0, 6.0, 0.0)),
               Prop(id="bench_1", type="bench", pos=(-6.0, 4.0, 0.0)),
               Prop(id="barrier_1", type="barrier", pos=(20.0, 0.0, 0.0))],
    )
    return Scenario(name="s", seed=3, duration_s=200, fps=5, environment=env,
                    population=Population(hero=people), cameras=cameras or [])


def _run(world: World, ticks: int) -> None:
    for _ in range(ticks):
        world.step()


def _at(world: World, uid: str, xy: tuple[float, float]) -> None:
    world.agents[uid].pos = np.array(list(xy), dtype=float)


def test_every_remaining_catalogue_behaviour_is_registered() -> None:
    for name in NEW:
        assert name in REGISTRY, f"{name} missing from REGISTRY"


def test_the_catalogue_is_complete() -> None:
    """50 distinct classes behind 55 catalogue entries (H40b added fence_climb,
    patrol_guard, maintenance_visit, wildlife; H54 added fence_tailgate,
    staff_escort).

    Fewer classes than entries is correct, not a shortfall: several catalogue
    rows are the same mechanism under a different label (`photograph`/`recon`,
    `maintenance`/`asset_tamper`), and collapsing those is the whole point.
    """
    assert len(set(REGISTRY.values())) == 50


# ------------------------------------------------- greeting versus argument


def test_a_greeting_closes_then_reopens_but_an_argument_does_not() -> None:
    """The one cue that genuinely separates them. Proximity alone does not.

    If this test ever passes trivially it means greetings stopped re-opening,
    and an aggression detector trained on this data would flag every handshake.
    """
    def gaps(kind: str) -> list[float]:
        a = Person(id="a", intent=[BehaviourSpec(type=kind, partner="b",
                                                 greet_s=4.0, dur_s=30.0)])
        b = Person(id="b", intent=[BehaviourSpec(type="loiter", zone="zone.open",
                                                 dur_s=60.0)])
        world = World(_scenario([a, b]))
        _at(world, "a", (-4.0, 0.0))
        _at(world, "b", (0.0, 0.0))
        out = []
        for _ in range(60):
            world.step()
            out.append(float(np.linalg.norm(world.agents["a"].pos
                                            - world.agents["b"].pos)))
        return out

    greet = gaps("meet_and_greet")
    argue = gaps("argument")

    assert min(greet) < 1.5, "the greeting should reach contact range"
    assert greet[-1] > min(greet) + 2.0, "the greeting should separate again"
    assert argue[-1] <= min(argue) + 0.5, "an argument stays closed"


def test_meet_and_greet_is_labelled_normal() -> None:
    assert REGISTRY["meet_and_greet"].label.startswith("normal")


# ----------------------------------------------------------------- handoff


def test_handoff_transfers_the_object_and_the_pair_then_separates() -> None:
    """Left-luggage detection sees nothing here: the bag is never unattended."""
    a = Person(id="a", intent=[BehaviourSpec(type="object_handoff", partner="b",
                                             object="package", to="exit.a")])
    b = Person(id="b", intent=[BehaviourSpec(type="loiter", zone="zone.open",
                                             dur_s=200.0)])
    world = World(_scenario([a, b]))
    _at(world, "a", (-10.0, 0.0))
    _at(world, "b", (0.0, 0.0))
    _run(world, 120)

    assert any(e.get("event") in {"handoff", "transfer"} or "transfer" in str(e)
               for e in world.sensor_log + [o for o in world.objects.values()]) or \
        any(o.get("owner") == "b" for o in world.objects.values())
    assert world.agents["a"].pos[0] > 5.0, "the giver should have moved on"


# ------------------------------------------------------- camera avoidance


def _camera(cid: str, pos: tuple[float, float, float], yaw: float) -> Camera:
    return Camera(id=cid, mount=Mount(pos=pos, yaw=yaw, pitch=-20.0),
                  capture={"res": (1280, 720), "fps": 10.0})


def test_avoidance_keeps_a_person_further_from_cameras_than_plain_transit() -> None:
    cams = [_camera("c1", (0.0, 0.0, 4.0), 0.0)]

    def min_gap(kind: str) -> float:
        p = Person(id="p", intent=[BehaviourSpec(type=kind, to="exit.a",
                                                 awareness=1.0)])
        world = World(_scenario([p], cams))
        _at(world, "p", (-20.0, 0.0))
        gaps = []
        for _ in range(200):
            world.step()
            gaps.append(float(np.linalg.norm(world.agents["p"].pos
                                             - np.array([0.0, 0.0]))))
        return min(gaps)

    assert min_gap("camera_avoidance") > min_gap("transit") + 1.0


def test_avoidance_still_reaches_its_destination() -> None:
    """An evader who never arrives is not modelling evasion, only obstruction."""
    p = Person(id="p", intent=[BehaviourSpec(type="camera_avoidance", to="exit.a")])
    world = World(_scenario([p], [_camera("c1", (0.0, 0.0, 4.0), 0.0)]))
    _at(world, "p", (-20.0, 0.0))
    _run(world, 400)
    # Reaching the exit ends the plan and the person leaves the scene.
    assert world.roster["p"].pos[0] > 30.0


def test_zero_awareness_does_not_avoid() -> None:
    """Awareness is a dial, not a switch: an unaware person walks straight."""
    p = Person(id="p", intent=[BehaviourSpec(type="camera_avoidance", to="exit.a",
                                             awareness=0.0)])
    world = World(_scenario([p], [_camera("c1", (0.0, 0.0, 4.0), 0.0)]))
    _at(world, "p", (-20.0, 0.0))
    _run(world, 60)
    assert abs(float(world.agents["p"].pos[1])) < 1.0


def test_avoidance_head_down_posture_is_recorded() -> None:
    p = Person(id="p", intent=[BehaviourSpec(type="camera_avoidance", to="exit.a")])
    world = World(_scenario([p], [_camera("c1", (0.0, 0.0, 4.0), 0.0)]))
    _at(world, "p", (-20.0, 0.0))
    world.step()
    assert world.agents["p"].posture == "head_down"


# --------------------------------------------- maintenance versus tampering


def test_maintenance_and_asset_tamper_are_mechanically_indistinguishable() -> None:
    """Deliberate. Inventing a motion cue here would be inventing a detector.

    A contractor kneeling at a cabinet and a saboteur opening it produce the
    same trajectory and the same posture. What differs is authorisation, and a
    simulator that leaked a trajectory difference would let a model 'solve' the
    hardest false-positive class in the domain for free.
    """
    def track(kind: str) -> tuple[list, str]:
        p = Person(id="p", intent=[BehaviourSpec(type=kind, asset="cabinet_1",
                                                 dur_s=20.0)])
        world = World(_scenario([p]))
        _at(world, "p", (0.0, 0.0))
        path = []
        for _ in range(60):
            world.step()
            path.append(world.agents["p"].pos.copy())
        return path, world.agents["p"].posture

    maint, maint_posture = track("maintenance")
    tamper, tamper_posture = track("asset_tamper")

    assert maint_posture == tamper_posture
    assert np.allclose(np.array(maint), np.array(tamper), atol=1e-9)


def test_only_the_tamper_records_damage() -> None:
    """The outcome differs even though the behaviour does not."""
    def damage_events(kind: str) -> int:
        p = Person(id="p", intent=[BehaviourSpec(type=kind, asset="cabinet_1",
                                                 dur_s=4.0)])
        world = World(_scenario([p]))
        _at(world, "p", (6.0, 5.0))
        _run(world, 80)
        return sum(1 for r in world.sensor_log if r.get("event") == "damage")

    assert damage_events("maintenance") == 0
    assert damage_events("asset_tamper") >= 1


def test_maintenance_holds_a_working_posture_at_the_asset() -> None:
    p = Person(id="p", intent=[BehaviourSpec(type="maintenance", asset="cabinet_1",
                                             dur_s=60.0)])
    world = World(_scenario([p]))
    _at(world, "p", (6.0, 5.0))
    _run(world, 20)
    assert world.agents["p"].posture == "kneeling"


def test_labels_separate_the_twins_even_though_motion_does_not() -> None:
    assert REGISTRY["maintenance"].label.startswith("normal")
    assert REGISTRY["asset_tamper"].label.startswith("hostile")


# ------------------------------------------------------- benign loiterers


@pytest.mark.parametrize("kind,prefix", [("smoke_break", "normal"),
                                         ("photograph", "normal"),
                                         ("lost_tourist", "normal")])
def test_benign_twins_carry_benign_labels(kind: str, prefix: str) -> None:
    assert REGISTRY[kind].label.startswith(prefix)


def test_smoke_break_dwells_without_leaving_its_zone() -> None:
    """The dwell centre is now a jittered point sampled inside the zone (not the
    exact centroid — see `Loiter.begin`), so the person can walk anywhere in
    `zone.open` to get there; the invariant that still must hold is that the
    behaviour keeps them inside their zone, not near the origin specifically."""
    p = Person(id="p", intent=[BehaviourSpec(type="smoke_break", zone="zone.open",
                                             dur_s=40.0)])
    world = World(_scenario([p]))
    _at(world, "p", (0.0, 0.0))
    _run(world, 100)
    assert world.zone_of(world.agents["p"].pos) == "zone.open"


def test_photograph_uses_the_same_mechanism_as_recon() -> None:
    """Only the target list differs, which is exactly how it is in real footage."""
    from survsim.director.behaviours import Photograph, Recon
    assert issubclass(Photograph, Recon)
    assert Photograph.step is Recon.step


def test_photograph_orients_at_its_target() -> None:
    p = Person(id="p", intent=[BehaviourSpec(type="photograph", targets=["cabinet_1"],
                                             dwell_s=5.0, standoff_m=6.0)])
    world = World(_scenario([p]))
    _at(world, "p", (0.0, 0.0))
    _run(world, 60)
    assert world.agents["p"].look_at is not None


# ------------------------------------------------------------- child play


def test_child_play_stays_in_its_zone_but_moves_faster_than_it_walks() -> None:
    p = Person(id="p", gait={"speed_mps": 1.2},
               intent=[BehaviourSpec(type="child_play", zone="zone.open",
                                     radius_m=6.0, dur_s=60.0)])
    world = World(_scenario([p]))
    _at(world, "p", (0.0, 0.0))
    speeds = []
    for _ in range(120):
        world.step()
        speeds.append(float(np.linalg.norm(world.agents["p"].vel)))

    assert float(np.linalg.norm(world.agents["p"].pos)) < 14.0
    assert max(speeds) > 1.2, "play involves bursts faster than a walk"


# ------------------------------------------------------------ drunk walk


def test_drunk_walk_arrives_but_not_in_a_straight_line() -> None:
    """A gait anomaly, not a trajectory anomaly: the destination is sensible."""
    p = Person(id="p", intent=[BehaviourSpec(type="drunk_walk", to="exit.a",
                                             sway_amplitude_m=1.2,
                                             sway_period_s=4.0)])
    world = World(_scenario([p]))
    _at(world, "p", (-20.0, 0.0))
    lateral = []
    for _ in range(300):
        world.step()
        lateral.append(float(world.agents["p"].pos[1]))

    assert world.agents["p"].pos[0] > 25.0, "should still get there"
    assert np.ptp(np.array(lateral)) > 0.8, "should sway off the straight line"


def test_drunk_walk_posture_is_recorded() -> None:
    p = Person(id="p", intent=[BehaviourSpec(type="drunk_walk", to="exit.a")])
    world = World(_scenario([p]))
    _at(world, "p", (-20.0, 0.0))
    world.step()
    assert world.agents["p"].posture == "unsteady"


# ------------------------------------------------------ mobility and rest


def test_wheelchair_user_is_seated_and_slower_than_walking() -> None:
    """Both halves matter: a seated silhouette is missed, then slowness is flagged."""
    p = Person(id="p", gait={"speed_mps": 1.4},
               intent=[BehaviourSpec(type="disabled_mobility", aid="wheelchair",
                                     to="exit.a")])
    world = World(_scenario([p]))
    _at(world, "p", (-20.0, 0.0))
    world.step()

    assert world.agents["p"].posture == "seated"
    assert float(np.linalg.norm(world.agents["p"].vel)) < 1.0


def test_mobility_aids_differ_in_speed() -> None:
    def speed(aid: str) -> float:
        p = Person(id="p", gait={"speed_mps": 1.4},
                   intent=[BehaviourSpec(type="disabled_mobility", aid=aid,
                                         to="exit.a")])
        world = World(_scenario([p]))
        _at(world, "p", (-20.0, 0.0))
        world.step()
        return float(np.linalg.norm(world.agents["p"].vel))

    assert speed("cane") > speed("wheelchair") > speed("frame")


def test_sit_rest_goes_upright_to_seated_to_prone() -> None:
    """The benign twin of a collapse, separated only by how it gets there."""
    p = Person(id="p", intent=[BehaviourSpec(type="sit_rest", bench="bench_1",
                                             dur_s=200.0, lie_after_s=10.0)])
    world = World(_scenario([p]))
    _at(world, "p", (-6.0, 3.0))

    seen = []
    for _ in range(120):
        world.step()
        seen.append(world.agents["p"].posture)

    assert "seated" in seen
    assert "prone" in seen
    assert seen.index("seated") < seen.index("prone"), "sitting precedes lying"


def test_a_collapse_reaches_prone_without_passing_through_seated() -> None:
    """The discriminating cue against `sit_rest`, pinned so it cannot regress."""
    p = Person(id="p", intent=[BehaviourSpec(type="person_down", dur_s=60.0)])
    world = World(_scenario([p]))
    seen = []
    for _ in range(40):
        world.step()
        seen.append(world.agents["p"].posture)
    assert "prone" in seen
    assert "seated" not in seen


# ------------------------------------------------------- weather reaction


def test_weather_reaction_advances_an_appearance_epoch_just_like_concealment() -> None:
    """Why a 'face suddenly covered' rule storms at the first shower."""
    p = Person(id="p", intent=[BehaviourSpec(type="weather_reaction", to="exit.a",
                                             at_s=1.0)])
    world = World(_scenario([p]))
    _at(world, "p", (-20.0, 0.0))
    before = world.agents["p"].appearance_epoch
    _run(world, 40)
    assert world.agents["p"].appearance_epoch > before


def test_weather_reaction_is_benign() -> None:
    assert REGISTRY["weather_reaction"].label.startswith("normal")


# ----------------------------------------------------------- crowd_form


def test_a_gathering_converges_on_its_centre() -> None:
    people = [Person(id=f"p{i}", intent=[BehaviourSpec(type="crowd_form",
                                                       zone="zone.open",
                                                       radius_m=3.0, dur_s=120.0)])
              for i in range(6)]
    world = World(_scenario(people))
    for i, person in enumerate(people):
        _at(world, person.id, (20.0 * np.cos(i), 20.0 * np.sin(i)))
    _run(world, 200)

    spread = np.array([world.agents[p.id].pos for p in people])
    assert float(np.linalg.norm(spread, axis=1).max()) < 6.0


def test_no_individual_in_a_gathering_looks_anomalous_alone() -> None:
    """Each person just walks, then stands. The anomaly is only in the density."""
    p = Person(id="p", intent=[BehaviourSpec(type="crowd_form", zone="zone.open",
                                             dur_s=120.0)])
    world = World(_scenario([p]))
    _at(world, "p", (20.0, 0.0))
    speeds = []
    for _ in range(200):
        world.step()
        speeds.append(float(np.linalg.norm(world.agents["p"].vel)))
    assert max(speeds) <= world.agents["p"].speed + 1e-6


# -------------------------------------------------------------- vehicles


def test_dropoff_loiter_waits_near_the_vehicle_then_reboards() -> None:
    v = Person(id="v", intent=[BehaviourSpec(type="loiter", zone="zone.open",
                                             dur_s=400.0)])
    p = Person(id="p", intent=[BehaviourSpec(type="vehicle_dropoff_loiter",
                                             vehicle="v", dur_s=20.0,
                                             standoff_m=4.0)])
    world = World(_scenario([v, p]))
    _at(world, "v", (0.0, 0.0))
    _at(world, "p", (1.0, 0.0))

    _run(world, 60)
    waiting = float(np.linalg.norm(world.agents["p"].pos - world.agents["v"].pos))
    assert 2.0 < waiting < 8.0, "should stand off, not wander away"

    _run(world, 200)
    assert float(np.linalg.norm(world.agents["p"].pos
                                - world.agents["v"].pos)) < 2.0


def test_a_ram_accelerates_and_breaches_the_barrier() -> None:
    """Seconds of warning: the speed vector is the only early cue."""
    v = Person(id="v", intent=[BehaviourSpec(type="vehicle_ram", barrier="barrier_1",
                                             speed_mps=20.0, accel_mps2=4.0,
                                             initial_speed_mps=2.0)])
    world = World(_scenario([v]))
    _at(world, "v", (-30.0, 0.0))

    world.step()
    first = float(np.linalg.norm(world.agents["v"].vel))
    _run(world, 15)
    later = float(np.linalg.norm(world.agents["v"].vel))

    assert later > first, "should be accelerating"
    _run(world, 200)
    assert any(r.get("event") == "breach" for r in world.sensor_log)


def test_a_ram_is_far_faster_than_ordinary_vehicle_transit() -> None:
    def top_speed(spec: BehaviourSpec) -> float:
        world = World(_scenario([Person(id="v", intent=[spec])]))
        _at(world, "v", (-40.0, 0.0))
        peak = 0.0
        for _ in range(40):
            world.step()
            peak = max(peak, float(np.linalg.norm(world.agents["v"].vel)))
        return peak

    ram = top_speed(BehaviourSpec(type="vehicle_ram", barrier="barrier_1"))
    normal = top_speed(BehaviourSpec(type="vehicle_move", to="exit.a", speed_mps=8.0))
    assert ram > normal


# --------------------------------------------------------- night presence


def test_night_presence_has_no_cue_in_the_trajectory_at_all() -> None:
    """It is plain transit. The alarm has to come from the schedule."""
    def path(kind: str) -> np.ndarray:
        p = Person(id="p", intent=[BehaviourSpec(type=kind, to="exit.a")])
        world = World(_scenario([p]))
        _at(world, "p", (-20.0, 0.0))
        out = []
        for _ in range(80):
            world.step()
            out.append(world.agents["p"].pos.copy())
        return np.array(out)

    assert np.allclose(path("night_presence"), path("transit"), atol=1e-9)


@pytest.mark.parametrize("hour,expected", [(2.0, True), (23.0, True),
                                           (12.0, False), (7.0, False)])
def test_closed_hours_wrap_around_midnight(hour: float, expected: bool) -> None:
    """22:00–06:00 is the common case and the one an unwrapped compare gets wrong."""
    from survsim.director.behaviours import build
    b = build({"type": "night_presence", "to": "exit.a", "hour": hour,
               "closed_hours": (22.0, 6.0)})
    b._closed = (22.0, 6.0)
    assert b.out_of_hours is expected

"""Tests for the Unreal backend's locomotion and coordinate handling.

Run against a fake UnrealCV client, so they verify the protocol and the decision
logic without an engine. The behaviours that matter:

* people should be *walked*, not teleported, whenever the build allows it;
* truth must be read back from the engine once the engine owns movement;
* unit conversion must not leak centimetres or left-handedness upward.
"""

from __future__ import annotations

import numpy as np
import pytest

from survsim.backends.unreal.backend import (
    M_TO_CM,
    UnrealBackend,
    to_ue_location,
    to_ue_yaw,
)
from survsim.core.models import (
    BehaviourSpec,
    Camera,
    Environment,
    Mount,
    Person,
    Population,
    Scenario,
    Waypoint,
    Zone,
)
from survsim.director.world import World

FULL = (
    "vset /animation/soma_glb/apply [str] [Anything]\n"
    "vset /agent/[str]/nav/goto [float] [float] [float]\n"
    "vset /agent/[str]/nav/stop\n"
    "vget /agent/[str]/nav/status\n"
    "vset /metahuman/parametric/wardrobe [str] [str] [str]\n"
    "vset /metahuman/parametric/body [str] [str] [float] [float] [float]\n"
)
NO_NAV = "vset /animation/soma_glb/apply [str] [Anything]\n"


class FakeClient:
    def __init__(self, listing: str = FULL, animation_works: bool = True) -> None:
        self.listing = listing
        self.animation_works = animation_works
        self.commands: list[str] = []
        self.positions: dict[str, tuple[float, float, float]] = {}

    def connect(self, timeout: float = 0) -> None: ...
    def isconnected(self) -> bool: return True
    def disconnect(self) -> None: ...

    def request(self, cmd: str, timeout: float = 60):
        self.commands.append(cmd)
        if "list_cmd" in cmd:
            return self.listing
        if cmd.startswith("vget /level/name"):
            return "TestLevel"
        if cmd.startswith("vset /objects/spawn_from_path"):
            return cmd.split()[-1]
        if "/color" in cmd:
            return "(R=10,G=20,B=30,A=255)"
        if cmd.startswith("vget /object/") and cmd.endswith("/location"):
            # "vget /object/<handle>/location" -> ['vget ', 'object', '<handle>', 'location']
            handle = cmd.split("/")[2]
            x, y, z = self.positions.get(handle, (150.0, -250.0, 0.0))
            return f"{x} {y} {z}"
        if "soma_glb/status" in cmd:
            return ("attached=true playing=true duration=1.20"
                    if self.animation_works else
                    "attached=false playing=false duration=0.000")
        if "nav/status" in cmd:
            return "moving"
        return "ok"

    def sent(self, needle: str) -> list[str]:
        return [c for c in self.commands if needle in c]


def _scenario() -> Scenario:
    env = Environment(
        id="t",
        zones=[Zone(id="zone.a", tag="public", polygon=[(-20, -20), (20, -20), (20, 20), (-20, 20)])],
        spawn_points=[Waypoint(id="spawn.s", pos=(-10.0, 0.0, 0.0))],
        exits=[Waypoint(id="exit.e", pos=(10.0, 0.0, 0.0))],
    )
    person = Person(
        id="p0", gait={"style": "limp_L", "speed_mps": 1.3},
        intent=[BehaviourSpec(type="transit", **{"from": "spawn.s", "to": "exit.e"})],
    )
    cam = Camera(id="c0", mount=Mount(pos=(0.0, -12.0, 5.0), yaw=90, pitch=-20))
    return Scenario(name="t", seed=1, duration_s=4, fps=5, engine="unreal",
                    environment=env, cameras=[cam], population=Population(hero=[person]))


def _backend(listing: str = FULL, **kw) -> tuple[UnrealBackend, FakeClient, World, Scenario]:
    sc = _scenario()
    be = UnrealBackend(**kw)
    client = FakeClient(listing)
    be.client = client
    be.open(sc)
    return be, client, World(sc), sc


# ------------------------------------------------------------------ conversion


def test_location_conversion_flips_y_and_scales_to_centimetres() -> None:
    x, y, z = to_ue_location(np.array([1.5, 2.0, 0.0]), ground_z_cm=-931.85)
    assert x == 150.0
    assert y == -200.0          # handedness flip
    assert z == pytest.approx(-931.85)


def test_yaw_conversion_flips_sign() -> None:
    assert to_ue_yaw(90.0) == -90.0


# ------------------------------------------------------------------ locomotion


def test_walking_uses_navigation_not_teleporting() -> None:
    """Teleporting slides the character and destroys the gait signal."""
    be, client, world, _ = _backend()
    world.step()
    be.sync(world)

    assert client.sent("nav/goto"), "should have issued a navigation goal"
    assert not [c for c in client.sent("/location") if c.startswith("vset")], \
        "should not be setting the transform directly while navigating"


def test_teleport_fallback_when_build_has_no_navigation() -> None:
    be, client, world, _ = _backend(listing=NO_NAV)
    world.step()
    be.sync(world)

    assert not client.sent("nav/goto")
    assert [c for c in client.sent("/location") if c.startswith("vset")]
    assert any("teleporting" in n for n in be.limitations)


def test_navigation_goals_are_throttled() -> None:
    """Re-sending the same destination every tick makes the character stutter."""
    be, client, world, _ = _backend()
    for _ in range(6):
        world.step()
        be.sync(world)
    goals = client.sent("nav/goto")
    assert 0 < len(goals) < 6


def test_stopping_issues_nav_stop() -> None:
    be, client, world, _ = _backend()
    world.step()
    be.sync(world)
    client.commands.clear()

    for agent in world.agents.values():      # simulate arriving
        agent.vel = np.zeros(2)
    be.sync(world)
    assert client.sent("nav/stop")


def test_truth_is_read_back_from_the_engine() -> None:
    """Once the engine walks people, the engine knows where they really are."""
    be, client, world, _ = _backend()
    client.positions["p0"] = (400.0, -600.0, 0.0)
    world.step()
    be.sync(world)

    agent = world.agents["p0"]
    assert agent.pos[0] == pytest.approx(400.0 / M_TO_CM)
    assert agent.pos[1] == pytest.approx(600.0 / M_TO_CM)   # Y flips back


def test_read_back_survives_a_malformed_reply() -> None:
    be, client, world, _ = _backend()
    world.step()
    before = world.agents["p0"].pos.copy()
    client.request = lambda cmd, timeout=60: "error"   # type: ignore[assignment]
    be.read_back(world)
    assert np.allclose(world.agents["p0"].pos, before)


# ------------------------------------------------------------------ animation


def test_gait_clip_is_applied_on_open() -> None:
    _, client, _, _ = _backend()
    assert client.sent("soma_glb/apply")


def test_idle_clip_replaces_gait_when_stationary() -> None:
    """A loiterer must stand, not freeze mid-stride."""
    be, client, world, _ = _backend()
    world.step()
    be.sync(world)
    client.commands.clear()

    for agent in world.agents.values():
        agent.vel = np.zeros(2)
    be.sync(world)
    assert client.sent("soma_glb/apply"), "should switch to a standing clip"


def test_motion_style_is_not_reapplied_every_tick() -> None:
    be, client, world, _ = _backend()
    world.step()
    be.sync(world)
    client.commands.clear()
    for _ in range(4):
        world.step()
        be.sync(world)
    assert len(client.sent("soma_glb/apply")) <= 1


# ----------------------------------------------------------------- capability


def test_missing_wardrobe_is_recorded_as_a_limitation() -> None:
    be, _, _, _ = _backend(listing=NO_NAV)
    assert any("render identically" in n for n in be.limitations)

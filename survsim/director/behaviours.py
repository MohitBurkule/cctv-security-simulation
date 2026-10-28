"""Behaviour leaves.

Each behaviour is a small state machine stepped once per tick. It moves its
person and reports the label that ground truth records. The catalogue in
docs/03_BEHAVIOUR_CATALOG.md is the specification; this module implements the
first slice of it (transit, loiter, commute_wait, object_abandon, follow).

`commute_wait` exists to be a hard negative for `loiter`: it looks identical to
a dwell-time rule, which is exactly the false positive that makes operators
switch a system off.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from . import fences, planner

if TYPE_CHECKING:
    from .world import Agent, World


@dataclass
class Behaviour:
    # `label` is a ClassVar, not a dataclass field: a field default is captured when
    # Behaviour is created, so a plain subclass attribute would never override it and
    # every behaviour would report the base label.
    label: ClassVar[str] = "normal"
    # True for behaviours that physically get past a fence (climb, cut, walk
    # into a zone they are not allowed in, an animal slipping under). Everyone
    # else is stopped by `World.step` at the fence line (fences.py).
    crosses_fence: ClassVar[bool] = False

    spec: dict[str, Any]
    done: bool = False
    t_start: float | None = None
    _goal: np.ndarray | None = field(default=None, repr=False)

    def begin(self, t: float, agent: Agent, world: World) -> None:
        self.t_start = t

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        raise NotImplementedError

    def elapsed(self, t: float) -> float:
        return 0.0 if self.t_start is None else t - self.t_start


class Transit(Behaviour):
    """Walk from wherever you are to a named point or zone, then finish.

    Destinations are sampled per person (`World.sample_goal`) so a crowd does not
    converge on one coordinate. If progress stalls — the goal is occupied, or
    avoidance has pinned the walker — the walker accepts a nearby arrival or
    re-plans to a fresh destination instead of standing still indefinitely.
    """

    label: ClassVar[str] = "normal.transit"
    STALL_S: ClassVar[float] = 1.5
    ACCEPT_M: ClassVar[float] = 2.5
    MAX_REPLANS: ClassVar[int] = 4

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._goal = world.sample_goal(self.spec.get("to"))
        if (src := self.spec.get("from")) is not None and self.elapsed(t) == 0.0:
            agent.pos = world.sample_goal(src).copy()
        # Waypoints round any fence in the way (fences.py); [] = straight line,
        # which is every leg on a site without fenced zones.
        self._route = world.route(agent.pos, self._goal)[:-1]
        self._best = self._remaining(agent)
        self._best_t = t
        self._replans = 0

    def _remaining(self, agent: Agent) -> float:
        """Path length still to walk (straight-line distance when unrouted)."""
        assert self._goal is not None
        g = self._goal
        if not self._route:
            return math.hypot(float(g[0] - agent.pos[0]), float(g[1] - agent.pos[1]))
        pts = [agent.pos, *self._route, g]
        return float(sum(math.hypot(float(b[0] - a[0]), float(b[1] - a[1]))
                         for a, b in itertools.pairwise(pts)))

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        assert self._goal is not None
        while self._route and planner.arrived(agent.pos, self._route[0], 1.0):
            self._route.pop(0)
        target = self._route[0] if self._route else self._goal
        v = planner.steer(agent.pos, target, agent.speed, world.neighbours(agent), world.rng)
        agent.pos = agent.pos + v * dt
        agent.vel = v
        if planner.arrived(agent.pos, self._goal):
            self.done = True
            return
        dist = self._remaining(agent)
        if dist < self._best - 0.1:
            self._best, self._best_t = dist, t
        elif t - self._best_t >= self.STALL_S:
            if float(np.linalg.norm(self._goal - agent.pos)) <= self.ACCEPT_M \
                    or self._replans >= self.MAX_REPLANS:
                self.done = True              # close enough: the spot is taken
                return
            self._replans += 1
            self._goal = world.sample_goal(self.spec.get("to"))
            self._route = world.route(agent.pos, self._goal)[:-1]
            self._best, self._best_t = self._remaining(agent), t


class Loiter(Behaviour):
    """Stay inside a zone for a set duration, pacing or standing.

    The emitted label carries the dwell seconds so downstream evaluation can test
    a threshold rule rather than only a binary class.
    """

    label: ClassVar[str] = "suspicious.loiter"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._radius = float(self.spec.get("radius_m", 3.0))
        self._activity = self.spec.get("activity", "idle")
        self._dur = float(self.spec.get("dur_s", 120.0))
        # Use `sample_goal`, not `resolve_point`: the latter is the exact zone
        # centroid, which is the same seeded-RNG jitter innocent placements
        # (e.g. Transit's destinations) already get from `world.sample_goal`.
        # Sending every loiterer to the literal centroid while a hard negative
        # like `commute_wait` (which subclasses this same `begin`, so it was
        # never actually spared either) walked there too meant "distance to
        # zone centroid" carried no information either way -- but it is
        # exactly the kind of simulation artefact a detector could latch onto
        # if any other code path ever computed it from the true centroid
        # instead of from what the person actually walked to. Sampling here
        # keeps ground truth (what happened) matching the same jittered
        # distribution every other destination in the simulator uses, per
        # principle #1 in CLAUDE.md (docs/research/58 "Simulator fixes this
        # exposed"). Retry a few times so a corner sample plus the pacing
        # radius doesn't walk the person out of a small zone -- that would
        # turn "pace inside the zone" into "pace outside it", which is a
        # bigger behavioural change than fixing the label leak calls for.
        zone_ref = self.spec.get("zone")
        zid = str(zone_ref) if zone_ref is not None else None
        centre = world.sample_goal(zone_ref)
        if zid is not None:
            probes = [
                np.array([self._radius, 0.0]), np.array([-self._radius, 0.0]),
                np.array([0.0, self._radius]), np.array([0.0, -self._radius]),
            ]
            for _ in range(20):
                if all(world.zone_of(centre + p) == zid for p in probes):
                    break
                centre = world.sample_goal(zone_ref)
        self._centre = centre
        self._goal = self._centre.copy()

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        if self.elapsed(t) >= self._dur:
            self.done = True
            return
        if self._activity == "pace" and planner.arrived(agent.pos, self._goal, 0.4):
            offset = world.rng.uniform(-self._radius, self._radius, size=2)
            self._goal = self._centre + offset
        speed = agent.speed * (0.35 if self._activity == "pace" else 0.0)
        v = planner.steer(agent.pos, self._goal, speed, world.neighbours(agent), world.rng)
        agent.pos = agent.pos + v * dt
        agent.vel = v

    @property
    def dwell_label(self) -> str:
        return f"{self.label}.{self._activity}"


class CommuteWait(Loiter):
    """Waiting for a train. Indistinguishable from loitering by dwell time alone."""

    label: ClassVar[str] = "normal.commute_wait"


class ObjectAbandon(Behaviour):
    """Carry an object into a zone, put it down, then walk away and do not return."""

    label: ClassVar[str] = "suspicious.object.abandon"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._drop = world.resolve_point(self.spec.get("zone"))
        self._leave = float(self.spec.get("leave_dist_m", 20.0))
        self._exit = world.resolve_point(self.spec.get("exit"))
        self._dropped = False

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        goal = self._exit if self._dropped else self._drop
        v = planner.steer(agent.pos, goal, agent.speed, world.neighbours(agent), world.rng)
        agent.pos = agent.pos + v * dt
        agent.vel = v
        if not self._dropped and planner.arrived(agent.pos, self._drop):
            obj = self.spec.get("object", "bag")
            world.drop_object(agent, obj, agent.pos.copy(), t)
            self._dropped = True
        elif self._dropped and np.linalg.norm(agent.pos - self._drop) >= self._leave:
            self.done = True


class Follow(Behaviour):
    """Shadow another person at a held distance, matching their stops."""

    label: ClassVar[str] = "suspicious.follow"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._target = str(self.spec.get("target"))
        self._dist = float(self.spec.get("dist_m", 8.0))
        self._dur = float(self.spec.get("dur_s", 120.0))

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        target = world.agents.get(self._target)
        if target is None or self.elapsed(t) >= self._dur:
            self.done = True
            return
        offset = agent.pos - target.pos
        n = float(np.linalg.norm(offset))
        hold = target.pos + (offset / n * self._dist if n > 1e-6 else np.array([self._dist, 0.0]))
        v = planner.steer(agent.pos, hold, agent.speed, world.neighbours(agent), world.rng)
        agent.pos = agent.pos + v * dt
        agent.vel = v


class OutfitChange(Behaviour):
    """Change clothes, ideally where no camera can see it.

    This is the behaviour that breaks appearance-based re-identification on
    purpose. It advances the person's appearance epoch, so evaluation can score
    matching before and after separately.
    """

    label: ClassVar[str] = "suspicious.appearance.change"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        target = self.spec.get("in_zone") or self.spec.get("zone")
        self._goal = world.resolve_point(target) if target else agent.pos.copy()
        self._delta = dict(self.spec.get("delta", {}))
        self._dwell = float(self.spec.get("dur_s", 4.0))
        self._applied = False

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        assert self._goal is not None
        if not planner.arrived(agent.pos, self._goal, 1.0):
            v = planner.steer(agent.pos, self._goal, agent.speed, world.neighbours(agent))
            agent.pos = agent.pos + v * dt
            agent.vel = v
            return
        agent.vel = np.zeros(2)
        if not self._applied:
            world.change_appearance(agent, self._delta, t)
            self._applied = True
            self._changed_at = t
        elif t - self._changed_at >= self._dwell:
            self.done = True


class Recon(Behaviour):
    """Move between vantage points and dwell while looking at each target.

    Hostile reconnaissance is characterised by *attention* — repeated, oriented
    dwell on cameras, doors and guards — rather than by any single pose. Gaze
    direction is recorded so a classifier has something to learn from.
    """

    label: ClassVar[str] = "suspicious.recon"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._targets = list(self.spec.get("targets", []))
        self._dwell = float(self.spec.get("dwell_s", 5.0))
        self._i = 0
        self._arrived_at: float | None = None
        self._advance(world, agent)

    def _advance(self, world: World, agent: Agent) -> None:
        if self._i >= len(self._targets):
            self.done = True
            return
        target = self._targets[self._i]
        self._look_at = world.resolve_target(target)
        # Stand off from the target rather than walking into it.
        offset = agent.pos - self._look_at
        n = float(np.linalg.norm(offset))
        direction = offset / n if n > 1e-6 else np.array([1.0, 0.0])
        self._goal = self._look_at + direction * float(self.spec.get("standoff_m", 8.0))

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        if self.done or self._goal is None:
            return
        if not planner.arrived(agent.pos, self._goal, 1.2):
            v = planner.steer(agent.pos, self._goal, agent.speed, world.neighbours(agent))
            agent.pos = agent.pos + v * dt
            agent.vel = v
            self._arrived_at = None
            return
        agent.vel = np.zeros(2)
        agent.look_at = self._look_at
        self._arrived_at = self._arrived_at or t
        if t - self._arrived_at >= self._dwell:
            self._i += 1
            self._arrived_at = None
            self._advance(world, agent)


class PerimeterProbe(Behaviour):
    """Approach a fence line, walk along it, touch it, withdraw.

    With fences modelled (fences.py) the probe stands off the fence on the
    outside and walks along the tangent of the nearest fence edge — the path a
    patrolling guard also takes, which is what makes `patrol_guard` its twin.
    """

    label: ClassVar[str] = "suspicious.probe.fence"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._segment = world.resolve_point(self.spec.get("segment"))
        self._walk_m = float(self.spec.get("walk_along_m", 6.0))
        along, stand = np.array([1.0, 0.0]), self._segment.copy()
        comp = world.fences.nearest(self._segment)
        if comp is not None and comp.poly.exterior.distance(
                _pt(self._segment)) <= 3.0:
            q, n = fences.outward_normal(comp.poly, self._segment)
            stand, along = q + n * 0.8, np.array([-n[1], n[0]])
        self._stand, self._along = stand, along
        self._goal = stand.copy()
        self._phase = "approach"

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        assert self._goal is not None
        v = planner.steer(agent.pos, self._goal, agent.speed * 0.7, world.neighbours(agent))
        agent.pos = agent.pos + v * dt
        agent.vel = v
        if planner.arrived(agent.pos, self._goal, 0.8):
            if self._phase == "approach":
                self._goal = self._stand + self._along * self._walk_m
                self._phase = "along"
            else:
                self.done = True
        elif self.elapsed(t) > 180.0:
            self.done = True                  # pinned against something: give up


class CameraTamper(Behaviour):
    """Walk to a camera mount and disable or misalign it."""

    label: ClassVar[str] = "hostile.tamper.camera"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._camera_id = str(self.spec.get("camera"))
        self._kind = str(self.spec.get("kind", "spray"))
        self._goal = world.camera_position(self._camera_id)[:2]
        self._reach_s = float(self.spec.get("dur_s", 5.0))
        self._at: float | None = None

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        assert self._goal is not None
        if not planner.arrived(agent.pos, self._goal, 1.5):
            v = planner.steer(agent.pos, self._goal, agent.speed, world.neighbours(agent))
            agent.pos = agent.pos + v * dt
            agent.vel = v
            return
        agent.vel = np.zeros(2)
        self._at = self._at or t
        if t - self._at >= self._reach_s:
            world.tamper_camera(self._camera_id, self._kind, t, agent)
            self.done = True

    @property
    def dwell_label(self) -> str:
        return f"{self.label}.{self._kind}"


def _pt(p: np.ndarray):
    from shapely.geometry import Point
    return Point(float(p[0]), float(p[1]))


class _Crossing(Behaviour):
    """Shared post-breach arc: get across a boundary and actually go somewhere.

    H40 (docs/research/32) measured the old behaviours stopping *at* the zone
    edge (6/100 intruders 1.5 m deep within 10 s) and never carrying a fence
    cutter across at all. Real intruders cross and go for something. Phases:

    approach -> (subclass breach at the fence) -> cross to a point >= depth_m
    inside (default 3.5 m) -> the asset inside (a cabinet prop, if any) and
    pause dwell_s -> exit back out through the breach point.
    """

    crosses_fence: ClassVar[bool] = True
    SPEED: ClassVar[float] = 1.1
    TIMEOUT_S: ClassVar[float] = 400.0

    def _setup(self, world: World, agent: Agent, poly, hint: np.ndarray) -> None:
        q, n = fences.outward_normal(poly, hint)
        self._poly = poly
        self._q, self._n = q, n
        self._out = q + n * 0.6
        self._in = q - n * 0.6
        self._deep, self._depth = fences.deep_point(poly, q, float(self.spec.get("depth_m", 3.5)))
        self._dwell = float(self.spec.get("dwell_s", self.spec.get("dur_s", 20.0)))
        self._asset = None
        cands = [np.array(p.pos[:2], float) for p in world.scenario.environment.props
                 if p.type in ("cabinet", "asset", "substation") and poly.contains(_pt(p.pos))]
        if self.spec.get("asset") is not None:
            try:
                cands = [world.resolve_target(self.spec.get("asset"))]
            except KeyError:
                pass
        if cands:
            self._asset = min(cands, key=lambda c: float(np.linalg.norm(c - self._deep)))
        self._phase = "approach"
        self._route = world.route(agent.pos, self._out)[:-1]
        self._at: float | None = None

    def _move(self, agent: Agent, world: World, goal: np.ndarray, dt: float, k: float = 1.0) -> None:
        v = planner.steer(agent.pos, goal, agent.speed * k, world.neighbours(agent))
        agent.pos = agent.pos + v * dt
        agent.vel = v

    def _approach(self, dt: float, agent: Agent, world: World) -> bool:
        """Walk to the breach point outside. True once there."""
        while self._route and planner.arrived(agent.pos, self._route[0], 1.0):
            self._route.pop(0)
        goal = self._route[0] if self._route else self._out
        if not self._route and planner.arrived(agent.pos, self._out, 0.5):
            agent.vel = np.zeros(2)
            return True
        self._move(agent, world, goal, dt, 0.9)
        return False

    def _breach(self, dt: float, t: float, agent: Agent, world: World) -> bool:
        """Subclass hook at the fence. True when through (default: walk in)."""
        return True

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        if self.elapsed(t) > self.TIMEOUT_S:
            agent.posture = "upright"
            self.done = True
            return
        if self._phase == "approach":
            if self._approach(dt, agent, world):
                self._phase, self._at = "breach", t
            return
        if self._phase == "breach":
            if self._breach(dt, t, agent, world):
                agent.posture = "upright"
                self._phase = "cross"
            return
        if self._phase == "cross":
            self._move(agent, world, self._deep, dt, self.SPEED)
            if planner.arrived(agent.pos, self._deep, 0.4):
                self._phase = "asset" if self._asset is not None else "pause"
                self._at = t
            return
        if self._phase == "asset":
            assert self._asset is not None
            self._move(agent, world, self._asset, dt, 0.8)
            if planner.arrived(agent.pos, self._asset, 1.2):
                self._phase, self._at = "pause", t
            return
        if self._phase == "pause":
            agent.vel = np.zeros(2)
            agent.look_at = (self._asset if self._asset is not None else self._deep).copy()
            if t - (self._at or t) >= self._dwell:
                self._phase = "exit"
            return
        # exit: back out through the breach
        goal = self._in if self._poly.contains(_pt(agent.pos)) and \
            float(np.linalg.norm(agent.pos - self._in)) > 0.5 else self._out + self._n * 1.0
        self._move(agent, world, goal, dt, self.SPEED)
        if not self._poly.contains(_pt(agent.pos)) and planner.arrived(agent.pos, self._out + self._n, 0.6):
            self.done = True


class FenceCut(_Crossing):
    """Crouch at a fence panel, cut it, pass through, go for the asset, leave.

    Without any fenced zone in the environment it degrades to the original
    cut-in-place (breach event, no crossing), because there is nowhere to go.
    """

    label: ClassVar[str] = "hostile.tamper.fence.cut"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._segment = world.resolve_point(self.spec.get("segment"))
        self._cut_s = float(self.spec.get("cut_s", self.spec.get("dur_s", 45.0)))
        if "dwell_s" not in self.spec:
            self.spec = {**self.spec, "dwell_s": 20.0}
        comp = world.fences.nearest(self._segment)
        self._legacy = comp is None
        if self._legacy:
            self._goal = self._segment.copy()
            self._at = None
            return
        self._setup(world, agent, comp.poly, self._segment)

    def _breach(self, dt: float, t: float, agent: Agent, world: World) -> bool:
        agent.vel = np.zeros(2)
        agent.posture = "crouch"
        if t - (self._at or t) >= self._cut_s:
            world.breach(self.spec.get("segment"), agent, t)
            return True
        return False

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        if not self._legacy:
            super().step(dt, t, agent, world)
            return
        assert self._goal is not None
        if not planner.arrived(agent.pos, self._goal, 0.8):
            self._move(agent, world, self._goal, dt, 0.8)
            return
        agent.vel = np.zeros(2)
        agent.posture = "crouch"
        self._at = self._at or t
        if t - self._at >= self._cut_s:
            world.breach(self.spec.get("segment"), agent, t)
            agent.posture = "upright"
            self.done = True


class FenceClimb(FenceCut):
    """Climb over a fence panel, drop inside, go for the asset, leave.

    The ground position barely moves while climbing (the cue is vertical), then
    the body passes over the line in the last third of the climb — so the
    ground-truth crossing time is when the feet land, not when the climb begins.
    """

    label: ClassVar[str] = "hostile.breach.climb"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        self.spec = {**self.spec, "cut_s": float(self.spec.get("climb_s", self.spec.get("dur_s", 8.0)))}
        super().begin(t, agent, world)

    def _breach(self, dt: float, t: float, agent: Agent, world: World) -> bool:
        agent.vel = np.zeros(2)
        agent.posture = "climbing"
        frac = (t - (self._at or t)) / max(self._cut_s, 1e-6)
        if frac >= 0.67:
            # over the top: the ground position passes the fence line
            k = min(1.0, (frac - 0.67) / 0.33)
            new = self._out + (self._in - self._out) * k
            agent.vel = (new - agent.pos) / dt
            agent.pos = new
        if frac >= 1.0:
            world.breach(self.spec.get("segment"), agent, t)
            return True
        return False


class Tailgate(Behaviour):
    """Follow an authorised person through a controlled door on their credential."""

    label: ClassVar[str] = "suspicious.access.tailgate"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._target = str(self.spec.get("target"))
        self._gap_m = float(self.spec.get("gap_m", 1.2))
        self._door = self.spec.get("door")
        self._through = False

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        target = world.agents.get(self._target)
        if target is None:
            self.done = True
            return
        offset = agent.pos - target.pos
        n = float(np.linalg.norm(offset))
        direction = offset / n if n > 1e-6 else np.array([1.0, 0.0])
        hold = target.pos + direction * self._gap_m
        v = planner.steer(agent.pos, hold, agent.speed * 1.1, world.neighbours(agent))
        agent.pos = agent.pos + v * dt
        agent.vel = v
        if self._door and not self._through:
            door_pos = world.resolve_point(self._door)
            if float(np.linalg.norm(agent.pos - door_pos)) < 1.5:
                world.access_event(str(self._door), target, agent, t)
                self._through = True
                self.done = True


class ErraticMotion(Behaviour):
    """Random-walk inside a zone: sudden direction changes and stops."""

    label: ClassVar[str] = "suspicious.motion.erratic"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._centre = world.resolve_point(self.spec.get("zone"))
        self._radius = float(self.spec.get("radius_m", 6.0))
        self._dur = float(self.spec.get("dur_s", 60.0))
        self._goal = self._centre.copy()

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        if self.elapsed(t) >= self._dur:
            self.done = True
            return
        assert self._goal is not None
        if planner.arrived(agent.pos, self._goal, 0.6) or world.rng.random() < 0.02:
            self._goal = self._centre + world.rng.uniform(-self._radius, self._radius, size=2)
        speed = agent.speed * float(world.rng.uniform(0.2, 1.3))
        v = planner.steer(agent.pos, self._goal, speed, world.neighbours(agent), world.rng)
        agent.pos = agent.pos + v * dt
        agent.vel = v


class ZoneIntrusion(_Crossing):
    """Enter a zone this person's role is not permitted in, go in, act, leave.

    Crosses at the boundary point nearest to where the person is when the
    behaviour starts (not the gate), goes >= 3.5 m inside, visits the asset
    there if any and stays `dur_s`, then leaves the way it came.
    """

    label: ClassVar[str] = "suspicious.access.intrusion"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._zone_id = str(self.spec.get("zone"))
        poly = world._zones.get(self._zone_id)
        if poly is None:
            from shapely.geometry import Point as _P
            poly = _P(*world.resolve_point(self._zone_id)).buffer(4.0)
        self._setup(world, agent, poly, agent.pos.copy())


class Argument(Behaviour):
    """Two people square up: face each other, close distance, gesture.

    Aggression is a *joint* behaviour, and that is what makes it hard. A single
    person's trajectory looks like standing still; the signal is in the pair —
    mutual facing, closing personal distance, and escalating motion. Modelling it
    per-person would make it undetectable for the right reasons.
    """

    label: ClassVar[str] = "hostile.aggression.verbal"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._partner = str(self.spec.get("partner", ""))
        self._dur = float(self.spec.get("dur_s", 25.0))
        self._closing = float(self.spec.get("closing_distance_m", 0.7))

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        other = world.agents.get(self._partner)
        if other is None or self.elapsed(t) >= self._dur:
            self.done = True
            return
        # Escalation: the gap closes as the argument develops, which is the cue a
        # proximity-and-orientation detector can actually pick up.
        progress = min(1.0, self.elapsed(t) / max(self._dur, 1e-6))
        target_gap = 2.2 - (2.2 - self._closing) * progress
        offset = agent.pos - other.pos
        n = float(np.linalg.norm(offset))
        direction = offset / n if n > 1e-6 else np.array([1.0, 0.0])
        hold = other.pos + direction * target_gap

        v = planner.steer(agent.pos, hold, agent.speed * 0.5, world.neighbours(agent))
        agent.pos = agent.pos + v * dt
        agent.vel = v
        agent.look_at = other.pos.copy()
        world.set_interaction(agent, other, "argument", t)


class Fight(Behaviour):
    """Physical altercation: rapid, erratic, close-range motion between two people.

    The motion signature is distinctive — high speed within a very small radius,
    which no ordinary pedestrian behaviour produces. Bystanders react, and that
    secondary signal (a ring forming, or people fleeing) is often what a camera
    sees first when the fight itself is occluded by the crowd around it.
    """

    label: ClassVar[str] = "hostile.aggression.fight"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._partner = str(self.spec.get("partner", ""))
        self._dur = float(self.spec.get("dur_s", 15.0))
        self._intensity = float(self.spec.get("intensity", 0.8))
        self._centre = agent.pos.copy()
        self._radius = float(self.spec.get("radius_m", 1.2))

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        other = world.agents.get(self._partner)
        if other is None or self.elapsed(t) >= self._dur:
            agent.posture = "upright"
            self.done = True
            return
        self._centre = (agent.pos + other.pos) / 2.0
        jitter = world.rng.normal(0.0, self._radius * 0.6, size=2)
        target = self._centre + jitter
        # Fast motion inside a tiny radius: the signature that separates a fight
        # from two people talking.
        speed = agent.speed * (1.4 + self._intensity)
        v = planner.steer(agent.pos, target, speed, None, world.rng)
        agent.pos = agent.pos + v * dt
        agent.vel = v
        agent.look_at = other.pos.copy()
        agent.posture = "brace"
        world.set_interaction(agent, other, "fight", t)


class TheftSnatch(Behaviour):
    """Approach a victim quickly, take something, then run."""

    label: ClassVar[str] = "hostile.theft.snatch"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._victim = str(self.spec.get("victim", ""))
        self._exit = world.resolve_point(self.spec.get("exit"))
        self._taken = False

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        victim = world.agents.get(self._victim)
        if victim is None:
            self.done = True
            return
        if not self._taken:
            v = planner.steer(agent.pos, victim.pos, agent.speed * 1.5,
                              world.neighbours(agent))
            agent.pos = agent.pos + v * dt
            agent.vel = v
            if float(np.linalg.norm(agent.pos - victim.pos)) < 1.0:
                world.transfer_object(victim, agent, self.spec.get("object", "handbag"), t)
                self._taken = True
            return
        # The flight is the loudest cue: a sudden sprint away from a stationary
        # person is rare and easy to detect, unlike the grab itself.
        v = planner.steer(agent.pos, self._exit, agent.speed * 2.0, world.neighbours(agent))
        agent.pos = agent.pos + v * dt
        agent.vel = v
        if planner.arrived(agent.pos, self._exit, 2.0):
            self.done = True


class PersonDown(Behaviour):
    """Someone collapses and stays down — medical emergency or the result of assault."""

    label: ClassVar[str] = "emergency.person_down"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._dur = float(self.spec.get("dur_s", 60.0))
        self._fall_s = float(self.spec.get("fall_s", 1.5))

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        elapsed = self.elapsed(t)
        agent.vel = np.zeros(2)
        # Posture, not position, is the whole signal here: a person lying down
        # occupies a wide, short bounding box that upright-person detectors miss.
        agent.posture = "falling" if elapsed < self._fall_s else "prone"
        if elapsed >= self._dur:
            agent.posture = "upright"
            self.done = True


class Flee(Behaviour):
    """Run away from a threat — the bystander reaction that betrays an incident."""

    label: ClassVar[str] = "emergency.flee"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        # `threat` is the unambiguous name; `from` is accepted for symmetry with
        # the other behaviours but means a *person* here, not a place.
        self._from = self.spec.get("threat", self.spec.get("from"))
        self._goal = world.resolve_point(self.spec.get("to"))

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        assert self._goal is not None
        threat = world.agents.get(str(self._from)) if self._from else None
        target = self._goal
        if threat is not None:
            away = agent.pos - threat.pos
            n = float(np.linalg.norm(away))
            if n > 1e-6:
                target = agent.pos + away / n * 10.0
        v = planner.steer(agent.pos, target, agent.speed * 2.2, world.neighbours(agent))
        agent.pos = agent.pos + v * dt
        agent.vel = v
        if planner.arrived(agent.pos, self._goal, 3.0):
            self.done = True


class GroupWalk(Behaviour):
    """Walk together as a group, keeping formation.

    Groups matter because they break single-person assumptions: they occlude each
    other constantly, and they split and merge, which is where trackers swap
    identities.
    """

    label: ClassVar[str] = "normal.group_walk"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._leader = str(self.spec.get("leader", ""))
        self._offset = np.array(self.spec.get("offset", [1.2, 0.8]), dtype=float)
        self._goal = world.resolve_point(self.spec.get("to"))

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        leader = world.agents.get(self._leader)
        target = (leader.pos + self._offset) if leader is not None else self._goal
        v = planner.steer(agent.pos, target, agent.speed, world.neighbours(agent), world.rng)
        agent.pos = agent.pos + v * dt
        agent.vel = v
        if leader is None and planner.arrived(agent.pos, self._goal):
            self.done = True


class CounterFlow(Behaviour):
    """Move against the dominant direction of the crowd."""

    label: ClassVar[str] = "suspicious.motion.counterflow"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._goal = world.resolve_point(self.spec.get("to"))
        self._dur = float(self.spec.get("dur_s", 40.0))

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        if self.elapsed(t) >= self._dur:
            self.done = True
            return
        assert self._goal is not None
        v = planner.steer(agent.pos, self._goal, agent.speed, world.neighbours(agent))
        agent.pos = agent.pos + v * dt
        agent.vel = v


class Queue(Behaviour):
    """Wait in line, shuffling forward as the person ahead moves.

    Queues break pedestrian models that assume free movement: people stand closer
    than personal space normally allows, move in bursts, and occlude each other
    almost continuously from a shallow camera angle.
    """

    label: ClassVar[str] = "normal.queue"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._head = world.resolve_point(self.spec.get("head") or self.spec.get("zone"))
        self._ahead = self.spec.get("ahead")
        self._spacing = float(self.spec.get("spacing_m", 0.7))
        self._dur = float(self.spec.get("dur_s", 90.0))

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        if self.elapsed(t) >= self._dur:
            self.done = True
            return
        ahead = world.agents.get(str(self._ahead)) if self._ahead else None
        if ahead is not None:
            offset = agent.pos - ahead.pos
            n = float(np.linalg.norm(offset))
            direction = offset / n if n > 1e-6 else np.array([1.0, 0.0])
            target = ahead.pos + direction * self._spacing
        else:
            target = self._head
        # Queue movement is slow and intermittent, not a steady walk.
        speed = agent.speed * 0.25
        v = planner.steer(agent.pos, target, speed, None, world.rng)
        agent.pos = agent.pos + v * dt
        agent.vel = v


class ClimbAttempt(Behaviour):
    """Approach a barrier, climb partway, drop back or cross."""

    label: ClassVar[str] = "hostile.breach.climb"
    crosses_fence: ClassVar[bool] = True

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._barrier = world.resolve_point(self.spec.get("barrier"))
        self._goal = self._barrier.copy()
        self._dur = float(self.spec.get("dur_s", 12.0))
        self._succeeds = bool(self.spec.get("succeeds", True))
        self._at: float | None = None

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        assert self._goal is not None
        if not planner.arrived(agent.pos, self._goal, 0.8):
            v = planner.steer(agent.pos, self._goal, agent.speed, world.neighbours(agent))
            agent.pos = agent.pos + v * dt
            agent.vel = v
            return
        agent.vel = np.zeros(2)
        self._at = self._at or t
        # Vertical motion with no ground translation is the distinguishing cue,
        # and it is invisible to any detector that only tracks ground position.
        agent.posture = "climbing"
        if t - self._at >= self._dur:
            if self._succeeds:
                world.breach(self.spec.get("barrier"), agent, t)
                agent.pos = agent.pos + (self._barrier - agent.pos) * 2.0
            agent.posture = "upright"
            self.done = True


class Vandalism(Behaviour):
    """Repetitive arm motion against a surface — spraying, scratching, breaking."""

    label: ClassVar[str] = "hostile.vandalism"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._surface = world.resolve_point(self.spec.get("target"))
        self._goal = self._surface.copy()
        self._dur = float(self.spec.get("dur_s", 30.0))
        self._at: float | None = None

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        assert self._goal is not None
        if not planner.arrived(agent.pos, self._goal, 1.0):
            v = planner.steer(agent.pos, self._goal, agent.speed, world.neighbours(agent))
            agent.pos = agent.pos + v * dt
            agent.vel = v
            return
        self._at = self._at or t
        # Small oscillation in place: stationary body, moving arm. A trajectory
        # model sees someone standing still, which is why this needs pose.
        sway = np.array([np.sin((t - self._at) * 6.0) * 0.12, 0.0])
        agent.pos = self._goal + sway
        agent.vel = sway / max(dt, 1e-6) * 0.1
        if t - self._at >= self._dur:
            world.damage(self.spec.get("target"), agent, t)
            self.done = True


class CrowdRush(Behaviour):
    """Run toward a point with others — a surge at a gate or away from a threat."""

    label: ClassVar[str] = "hostile.crowd.rush"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._goal = world.resolve_point(self.spec.get("target"))

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        assert self._goal is not None
        # Deliberately weak avoidance: in a surge people do not maintain personal
        # space, which is precisely what makes it dangerous and hard to track.
        v = planner.steer(agent.pos, self._goal, agent.speed * 2.0, None, world.rng)
        agent.pos = agent.pos + v * dt
        agent.vel = v
        if planner.arrived(agent.pos, self._goal, 2.0):
            self.done = True


class ThrowObject(Behaviour):
    """Wind up and throw something toward a target."""

    label: ClassVar[str] = "hostile.throw"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._target = world.resolve_point(self.spec.get("target"))
        self._windup_s = float(self.spec.get("windup_s", 1.2))
        self._thrown = False

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        agent.vel = np.zeros(2)
        agent.look_at = self._target.copy()
        if self.elapsed(t) < self._windup_s:
            agent.posture = "windup"
            return
        if not self._thrown:
            world.drop_object(agent, str(self.spec.get("object", "projectile")),
                              self._target.copy(), t)
            self._thrown = True
            agent.posture = "upright"
        self.done = True


class VehicleMove(Behaviour):
    """A vehicle driving a route. Much faster and larger than a pedestrian."""

    label: ClassVar[str] = "normal.vehicle.transit"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._goal = world.resolve_point(self.spec.get("to"))
        self._speed = float(self.spec.get("speed_mps", 8.0))

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        assert self._goal is not None
        v = planner.steer(agent.pos, self._goal, self._speed, None)
        agent.pos = agent.pos + v * dt
        agent.vel = v
        agent.posture = "vehicle"
        if planner.arrived(agent.pos, self._goal, 3.0):
            self.done = True


class DroneOverflight(Behaviour):
    """A small aerial object crossing the site.

    Included because counter-UAS is a stated capability and because a drone
    breaks the ground-plane assumption every other part of the pipeline makes:
    it has no foot contact, so homography-based localisation gives nonsense.
    """

    label: ClassVar[str] = "hostile.airspace.drone"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._goal = world.resolve_point(self.spec.get("to"))
        self._speed = float(self.spec.get("speed_mps", 12.0))
        self._altitude = float(self.spec.get("altitude_m", 40.0))

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        assert self._goal is not None
        v = planner.steer(agent.pos, self._goal, self._speed, None)
        agent.pos = agent.pos + v * dt
        agent.vel = v
        agent.posture = "airborne"
        agent.altitude_m = self._altitude
        if planner.arrived(agent.pos, self._goal, 5.0):
            self.done = True


# ---------------------------------------------------------------------------
# The rest of the catalogue.
#
# Almost everything below is a *benign twin* of a hostile behaviour already
# implemented above, and that is the point rather than an accident. A plumber
# kneeling at a cabinet for four minutes and a saboteur opening that cabinet
# produce the same trajectory, the same posture and the same dwell; someone
# pulling a hood up in the rain and someone concealing their face before a theft
# both advance an appearance epoch. If the simulator only ever generated the
# hostile member of each pair, a classifier trained on it would learn "kneeling
# near an asset is an attack" and fire on every maintenance visit on site.
#
# So each twin below is deliberately built on the *same mechanism* as its
# hostile counterpart, and differs only in the label and in the context that
# genuinely separates them (which asset, which hours, whether the gap keeps
# closing). Where a benign and hostile behaviour are mechanically identical, the
# class is shared outright and only `label` differs — pretending otherwise would
# manufacture a cue that does not exist in real footage.
# ---------------------------------------------------------------------------


class MeetAndGreet(Behaviour):
    """Two people converge, greet at contact range, then separate.

    The benign twin of `Argument`: both are two people standing unusually close
    and facing each other, which is why proximity-plus-orientation alone is not
    an aggression detector. The separating cue is the *shape of the gap over
    time* — a greeting closes, holds briefly and then re-opens, whereas an
    argument closes monotonically and stays closed.
    """

    label: ClassVar[str] = "normal.meet_and_greet"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._partner = str(self.spec.get("partner", ""))
        self._greet_s = float(self.spec.get("greet_s", 6.0))
        self._dur = float(self.spec.get("dur_s", 20.0))
        self._contact = float(self.spec.get("contact_distance_m", 0.8))
        self._exit = self.spec.get("to")

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        other = world.agents.get(self._partner)
        elapsed = self.elapsed(t)
        if other is None or elapsed >= self._dur:
            self.done = True
            return

        if elapsed < self._greet_s:
            offset = agent.pos - other.pos
            n = float(np.linalg.norm(offset))
            direction = offset / n if n > 1e-6 else np.array([1.0, 0.0])
            hold = other.pos + direction * self._contact
            v = planner.steer(agent.pos, hold, agent.speed * 0.6, world.neighbours(agent))
            agent.look_at = other.pos.copy()
            world.set_interaction(agent, other, "greet", t)
        else:
            # The gap re-opens. This is the whole difference from an argument.
            goal = world.resolve_point(self._exit) if self._exit else agent.pos + \
                (agent.pos - other.pos) * 4.0
            v = planner.steer(agent.pos, goal, agent.speed, world.neighbours(agent))
        agent.pos = agent.pos + v * dt
        agent.vel = v


class ObjectHandoff(Behaviour):
    """Two people meet briefly, pass an object, and leave in different directions.

    Harder than abandonment: the object is never static and never unattended, so
    a left-luggage detector sees nothing at all. The only evidence is a brief
    convergence followed by divergence, plus the object changing owner — which
    is why ground truth records the transfer explicitly.
    """

    label: ClassVar[str] = "suspicious.object.handoff"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._partner = str(self.spec.get("partner", ""))
        self._object = str(self.spec.get("object", "package"))
        self._exit = self.spec.get("to")
        self._range = float(self.spec.get("contact_distance_m", 1.0))
        self._passed = False

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        other = world.agents.get(self._partner)
        if other is None:
            self.done = True
            return
        if not self._passed:
            v = planner.steer(agent.pos, other.pos, agent.speed, world.neighbours(agent))
            agent.pos = agent.pos + v * dt
            agent.vel = v
            agent.look_at = other.pos.copy()
            if float(np.linalg.norm(agent.pos - other.pos)) <= self._range:
                world.transfer_object(agent, other, self._object, t)
                self._passed = True
            return
        goal = world.resolve_point(self._exit) if self._exit else \
            agent.pos + (agent.pos - other.pos)
        v = planner.steer(agent.pos, goal, agent.speed, world.neighbours(agent))
        agent.pos = agent.pos + v * dt
        agent.vel = v
        if (self._exit is not None and planner.arrived(agent.pos, goal, 2.0)) or (
                self._exit is None and float(np.linalg.norm(agent.pos - other.pos)) > 12.0):
            self.done = True


class CameraAvoidance(Behaviour):
    """Walk to a goal while steering away from the cameras that can see you.

    This is the behaviour that turns a rig's coverage gaps into an attack
    surface, so it is worth simulating for what it reveals about *placement*
    rather than about the person: if avoidance succeeds, the rig has a usable
    corridor through it, and the analyser should have found that before an
    intruder did.

    Avoidance is deliberately imperfect. `awareness` scales the repulsion, so
    a person who knows roughly where the cameras are behaves differently from
    one who has surveyed them, and neither is invisible.
    """

    label: ClassVar[str] = "suspicious.evasion.camera"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._goal = world.resolve_point(self.spec.get("to"))
        self._awareness = float(self.spec.get("awareness", 1.0))
        self._radius = float(self.spec.get("avoid_radius_m", 12.0))
        self._cameras = [c.id for c in world.scenario.cameras]
        self.frames_seen = 0
        self.frames_total = 0

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        assert self._goal is not None
        to_goal = self._goal - agent.pos
        distance = float(np.linalg.norm(to_goal))
        if distance < 1.0:
            self.done = True
            return
        desired = to_goal / distance

        # Push away from every nearby camera, strongest when closest. A camera
        # far away is ignored, because a person cannot plausibly route around
        # the entire site.
        #
        # The push is mostly *tangential* rather than straight away from the
        # camera. A purely radial field deadlocks whenever a camera sits between
        # the person and where they are going: repulsion cancels attraction
        # exactly and they stall in the open, which is both a useless evader and
        # a behaviour no real person exhibits. Sliding around the camera is what
        # someone avoiding it actually does.
        for cam_id in self._cameras:
            try:
                cam_pos = world.camera_position(cam_id)[:2]
            except (KeyError, AttributeError):
                continue
            away = agent.pos - cam_pos
            gap = float(np.linalg.norm(away))
            if 1e-6 < gap < self._radius:
                weight = (self._radius - gap) / self._radius * self._awareness
                radial = away / gap
                # Of the two perpendiculars, take the one that still makes
                # progress toward the goal, so detouring never becomes retreat.
                perp = np.array([-radial[1], radial[0]])
                if float(np.dot(perp, to_goal)) < 0.0:
                    perp = -perp
                desired = desired + (radial * 0.6 + perp * 1.4) * weight

        n = float(np.linalg.norm(desired))
        v = desired / n * agent.speed if n > 1e-6 else np.zeros(2)
        agent.pos = agent.pos + v * dt
        agent.vel = v
        # Head down and turned away is the other half of the technique, and it
        # is what defeats face capture even when avoidance of the FOV fails.
        agent.posture = "head_down"

        self.frames_total += 1
        if any(world.sees(c, agent) for c in self._cameras):
            self.frames_seen += 1

    @property
    def exposure(self) -> float:
        """Fraction of ticks this person was usably visible to any camera."""
        return self.frames_seen / self.frames_total if self.frames_total else 0.0


class Maintenance(Behaviour):
    """A worker kneels at an asset and works on it for several minutes.

    The benign twin of `AssetTamper` and of `CameraTamper` — and the single
    biggest source of false alarms in real deployments, because a contractor on
    a ladder under a camera is exactly what tampering looks like. It shares the
    tamper mechanism completely; only the label and the authorisation differ.
    """

    label: ClassVar[str] = "normal.maintenance"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._goal = world.resolve_target(self.spec.get("asset"))
        self._dur = float(self.spec.get("dur_s", 240.0))
        self._posture = str(self.spec.get("posture", "kneeling"))
        self._arrived_at: float | None = None

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        assert self._goal is not None
        if not planner.arrived(agent.pos, self._goal, 1.5):
            v = planner.steer(agent.pos, self._goal, agent.speed, world.neighbours(agent))
            agent.pos = agent.pos + v * dt
            agent.vel = v
            return
        agent.vel = np.zeros(2)
        agent.posture = self._posture
        agent.look_at = self._goal.copy()
        self._arrived_at = self._arrived_at or t
        if t - self._arrived_at >= self._dur:
            agent.posture = "upright"
            self.done = True


class AssetTamper(Maintenance):
    """Interfering with a cabinet, valve or transformer.

    Mechanically identical to `Maintenance` on purpose: same approach, same
    kneel, same dwell. What separates them in the real world is authorisation
    and schedule, not motion — so the simulator refuses to invent a motion cue
    that would let a classifier cheat. It does record the damage, because the
    *outcome* differs even when the behaviour does not.
    """

    label: ClassVar[str] = "hostile.tamper.asset"

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        was_done = self.done
        super().step(dt, t, agent, world)
        if self.done and not was_done:
            world.damage(self.spec.get("asset"), agent, t)


class SmokeBreak(Loiter):
    """A few minutes outside, pacing a little. Dwell-time loitering, benignly."""

    label: ClassVar[str] = "normal.smoke_break"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        self.spec.setdefault("activity", "pace")
        self.spec.setdefault("radius_m", 3.0)
        self.spec.setdefault("dur_s", 300.0)
        super().begin(t, agent, world)


class Photograph(Recon):
    """A tourist photographing a landmark.

    The benign twin of `Recon`, and mechanically the same: stand off, orient at
    a target, dwell. Hostile reconnaissance is separated by *what* is being
    looked at — cameras, doors and guards rather than architecture — which is a
    property of the target list, not of the person's motion.
    """

    label: ClassVar[str] = "normal.photograph"


class LostTourist(ErraticMotion):
    """Stop-start wandering while trying to find the way. Benignly erratic."""

    label: ClassVar[str] = "normal.lost"


class ChildPlay(Behaviour):
    """Running, stopping and changing direction far faster than an adult.

    Worth its own class rather than a fast `ErraticMotion` because the height is
    what breaks things: a small person at range falls below the pixel-height
    floor a detector was tuned on, so children are missed in exactly the scenes
    where missing them matters most.
    """

    label: ClassVar[str] = "normal.child_play"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._centre = world.resolve_point(self.spec.get("zone"))
        self._radius = float(self.spec.get("radius_m", 8.0))
        self._dur = float(self.spec.get("dur_s", 120.0))
        self._goal = self._centre.copy()

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        if self.elapsed(t) >= self._dur:
            self.done = True
            return
        assert self._goal is not None
        if planner.arrived(agent.pos, self._goal, 0.8) or world.rng.random() < 0.06:
            self._goal = self._centre + world.rng.uniform(-self._radius, self._radius, size=2)
        speed = agent.speed * float(world.rng.uniform(0.0, 2.2))
        v = planner.steer(agent.pos, self._goal, speed, world.neighbours(agent), world.rng)
        agent.pos = agent.pos + v * dt
        agent.vel = v


class DrunkWalk(Behaviour):
    """Transit with a lateral sway superimposed on the path.

    A gait anomaly rather than a trajectory anomaly: the person is going
    somewhere perfectly sensible, but not in a straight line. Included as the
    benign twin of `ErraticMotion` and because intoxication is one of the few
    behaviours an operator genuinely wants flagged without it being a crime.
    """

    label: ClassVar[str] = "normal.impaired.drunk"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._goal = world.resolve_point(self.spec.get("to"))
        self._amp = float(self.spec.get("sway_amplitude_m", 0.8))
        self._period = float(self.spec.get("sway_period_s", 4.0))

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        assert self._goal is not None
        to_goal = self._goal - agent.pos
        distance = float(np.linalg.norm(to_goal))
        if distance < 1.0:
            self.done = True
            return
        forward = to_goal / distance
        lateral = np.array([-forward[1], forward[0]])
        phase = 2.0 * np.pi * self.elapsed(t) / max(self._period, 1e-6)
        target = agent.pos + forward * 2.0 + lateral * self._amp * float(np.sin(phase))

        v = planner.steer(agent.pos, target, agent.speed * 0.7, world.neighbours(agent))
        agent.pos = agent.pos + v * dt
        agent.vel = v
        agent.posture = "unsteady"


class DisabledMobility(Behaviour):
    """Travel using a wheelchair, crutches or a walking frame.

    Present for fairness rather than for threat. A wheelchair user is seated, so
    they are shorter and wider than the upright body model every pedestrian
    detector is trained on, and they move at roughly half walking pace — a
    combination that gets them missed by the detector and then flagged as
    anomalous by the behaviour layer for moving too slowly.
    """

    label: ClassVar[str] = "normal.mobility_aid"

    AID_SPEED: ClassVar[dict[str, float]] = {"wheelchair": 0.55, "crutch": 0.45, "frame": 0.35, "cane": 0.7}

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._goal = world.resolve_point(self.spec.get("to"))
        self._aid = str(self.spec.get("aid", "wheelchair"))
        self._factor = float(self.spec.get(
            "speed_factor", self.AID_SPEED.get(self._aid, 0.5)))

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        assert self._goal is not None
        v = planner.steer(agent.pos, self._goal, agent.speed * self._factor,
                          world.neighbours(agent))
        agent.pos = agent.pos + v * dt
        agent.vel = v
        agent.posture = "seated" if self._aid == "wheelchair" else "assisted"
        if planner.arrived(agent.pos, self._goal, 1.0):
            agent.posture = "upright"
            self.done = True


class SitRest(Behaviour):
    """Sit on a bench, possibly lie down and sleep.

    The benign twin of `PersonDown`, and the reason a lying-posture detector
    cannot be deployed unsupervised: a person asleep on a bench at 3 a.m. and a
    person who has collapsed present the same silhouette. Only context — bench
    or bare floor, gradual or sudden — distinguishes them, so both are
    generated and the *transition* is what carries the signal.
    """

    label: ClassVar[str] = "normal.sit_rest"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._goal = world.resolve_target(self.spec.get("bench", self.spec.get("zone")))
        self._dur = float(self.spec.get("dur_s", 180.0))
        self._lie_after = self.spec.get("lie_after_s")
        self._seated_at: float | None = None

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        assert self._goal is not None
        if not planner.arrived(agent.pos, self._goal, 1.0):
            v = planner.steer(agent.pos, self._goal, agent.speed, world.neighbours(agent))
            agent.pos = agent.pos + v * dt
            agent.vel = v
            return
        agent.vel = np.zeros(2)
        self._seated_at = self._seated_at or t
        sitting = t - self._seated_at
        # The descent is gradual, unlike a collapse: the posture passes through
        # seated on the way down rather than going upright → prone in a second.
        if self._lie_after is not None and sitting >= float(self._lie_after):
            agent.posture = "prone"
        else:
            agent.posture = "seated"
        if sitting >= self._dur:
            agent.posture = "upright"
            self.done = True


class WeatherReaction(Behaviour):
    """Put a hood up, open an umbrella, or hurry for cover when it rains.

    The benign twin of `concealment`. Everyone in the scene changes appearance
    at the same moment, which is precisely why a "face suddenly covered" rule
    generates an alarm storm at the first shower — and why the confound audit
    needs benign examples that advance an appearance epoch.
    """

    label: ClassVar[str] = "normal.weather_reaction"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._goal = world.resolve_point(self.spec.get("to")) if self.spec.get("to") \
            else agent.pos.copy()
        self._delta = dict(self.spec.get("delta", {"outer": "hood_up", "face_covered": 0.4}))
        self._at = float(self.spec.get("at_s", 2.0))
        self._hurry = float(self.spec.get("hurry_factor", 1.4))
        self._applied = False

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        assert self._goal is not None
        if not self._applied and self.elapsed(t) >= self._at:
            world.change_appearance(agent, self._delta, t)
            self._applied = True
        speed = agent.speed * (self._hurry if self._applied else 1.0)
        v = planner.steer(agent.pos, self._goal, speed, world.neighbours(agent))
        agent.pos = agent.pos + v * dt
        agent.vel = v
        if self._applied and planner.arrived(agent.pos, self._goal, 1.5):
            self.done = True


class CrowdForm(Behaviour):
    """Converge on a point and stand there with others.

    A density spike with no single-person anomaly in it: every individual is
    just walking then standing. Whether a gathering is a queue, an audience or
    the start of a disturbance is not visible in any one trajectory, which is
    the argument for density-level analytics alongside per-person ones.
    """

    label: ClassVar[str] = "suspicious.crowd.gather"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._centre = world.resolve_point(self.spec.get("zone", self.spec.get("at")))
        self._radius = float(self.spec.get("radius_m", 3.0))
        self._dur = float(self.spec.get("dur_s", 90.0))
        angle = float(world.rng.uniform(0.0, 2.0 * np.pi))
        self._goal = self._centre + np.array([np.cos(angle), np.sin(angle)]) * self._radius

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        assert self._goal is not None
        if self.elapsed(t) >= self._dur:
            self.done = True
            return
        if planner.arrived(agent.pos, self._goal, 0.8):
            agent.vel = np.zeros(2)
            agent.look_at = self._centre.copy()
            return
        v = planner.steer(agent.pos, self._goal, agent.speed, world.neighbours(agent))
        agent.pos = agent.pos + v * dt
        agent.vel = v


class VehicleDropoffLoiter(Behaviour):
    """Get out of a vehicle, wait nearby, then get back in and leave.

    The behaviour that needs person-to-vehicle association to make any sense:
    on its own the person is a short loiter and the vehicle is a parked object,
    and neither is remarkable. Together, a vehicle waiting with its occupant
    outside it near a perimeter is one of the oldest indicators there is.
    """

    label: ClassVar[str] = "suspicious.vehicle.loiter"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._vehicle = str(self.spec.get("vehicle", ""))
        self._dur = float(self.spec.get("dur_s", 60.0))
        self._standoff = float(self.spec.get("standoff_m", 4.0))
        self._exited_at: float | None = None
        self._returning = False

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        vehicle = world.agents.get(self._vehicle)
        if vehicle is None:
            self.done = True
            return
        if self._returning:
            v = planner.steer(agent.pos, vehicle.pos, agent.speed, world.neighbours(agent))
            agent.pos = agent.pos + v * dt
            agent.vel = v
            if float(np.linalg.norm(agent.pos - vehicle.pos)) < 1.5:
                world.set_interaction(agent, vehicle, "board", t)
                self.done = True
            return

        offset = agent.pos - vehicle.pos
        n = float(np.linalg.norm(offset))
        direction = offset / n if n > 1e-6 else np.array([1.0, 0.0])
        station = vehicle.pos + direction * self._standoff
        v = planner.steer(agent.pos, station, agent.speed * 0.5, world.neighbours(agent))
        agent.pos = agent.pos + v * dt
        agent.vel = v
        self._exited_at = self._exited_at or t
        if t - self._exited_at >= self._dur:
            self._returning = True


class VehicleRam(Behaviour):
    """A vehicle accelerating at a barrier.

    Seconds of warning at most, which is the entire design constraint: anything
    that needs a multi-frame track to become confident is already too late. The
    cue available early is the speed *vector* relative to a protected line.
    """

    label: ClassVar[str] = "hostile.vehicle.ram"
    crosses_fence: ClassVar[bool] = True      # through the barrier, by definition

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._goal = world.resolve_target(self.spec.get("barrier", self.spec.get("to")))
        self._top = float(self.spec.get("speed_mps", 20.0))
        self._accel = float(self.spec.get("accel_mps2", 3.0))
        self._speed = float(self.spec.get("initial_speed_mps", 2.0))
        self._hit = False

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        assert self._goal is not None
        if self._hit:
            self.done = True
            return
        self._speed = min(self._top, self._speed + self._accel * dt)
        v = planner.steer(agent.pos, self._goal, self._speed, None)
        agent.pos = agent.pos + v * dt
        agent.vel = v
        agent.posture = "vehicle"
        if planner.arrived(agent.pos, self._goal, 2.5):
            world.breach(self.spec.get("barrier"), agent, t)
            self._hit = True


class _Waypoints(Behaviour):
    """Walk a list of (point, pause_s, posture) legs, routing round fences."""

    def _start(self, world: World, agent: Agent, legs: list[tuple[np.ndarray, float, str]]) -> None:
        self._legs = legs
        self._route: list[np.ndarray] = []
        self._routed = False
        self._paused: float | None = None
        self._speed_k = float(self.spec.get("speed_k", 0.9))

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        if not self._legs or self.elapsed(t) > float(self.spec.get("timeout_s", 1800.0)):
            agent.posture = "upright"
            self.done = True
            return
        goal, pause, posture = self._legs[0]
        if self._paused is not None:
            agent.vel = np.zeros(2)
            agent.posture = posture
            if t - self._paused >= pause:
                agent.posture = "upright"
                self._legs.pop(0)
                self._paused, self._routed = None, False
            return
        if not self._routed:
            self._route = world.route(agent.pos, goal)[:-1]
            self._routed = True
        while self._route and planner.arrived(agent.pos, self._route[0], 1.0):
            self._route.pop(0)
        v = planner.steer(agent.pos, self._route[0] if self._route else goal,
                          agent.speed * self._speed_k, world.neighbours(agent))
        agent.pos = agent.pos + v * dt
        agent.vel = v
        if planner.arrived(agent.pos, goal, 0.6):
            if pause > 0:
                self._paused = t
            else:
                self._legs.pop(0)
                self._routed = False


def _zone_asset(world: World, poly) -> np.ndarray:
    cands = [np.array(p.pos[:2], float) for p in world.scenario.environment.props
             if p.type in ("cabinet", "asset", "substation") and poly.contains(_pt(p.pos))]
    if cands:
        return cands[0]
    return fences.deep_point(poly, np.array(poly.centroid.coords[0]), 2.0)[0]


class PatrolGuard(_Waypoints):
    """A guard walks the fence line on a schedule and checks inside via the gate.

    The hard negative for `perimeter_probe`, `fence_cut`, `fence_climb` (and,
    for a role-blind rule, `zone_intrusion`): the same path along the fence and
    a real crossing into the sterile area, by someone entitled to make it. A
    video-only line-cross rule cannot tell them apart; a role or schedule can.
    """

    label: ClassVar[str] = "normal.patrol"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        off = float(self.spec.get("offset_m", 1.0))
        check = float(self.spec.get("check_s", 15.0))
        zones = self.spec.get("zones")
        comps = [c for c in world.fences.components
                 if not zones or any(z in c.zone_ids for z in zones)]
        legs: list[tuple[np.ndarray, float, str]] = []
        for _ in range(int(self.spec.get("rounds", 1))):
            for c in comps:
                ring = np.asarray(c.poly.buffer(off, join_style=2).exterior.coords)[:-1]
                k = int(np.argmin(np.linalg.norm(ring - c.gate_out, axis=1)))
                loop = [ring[(k + i) % len(ring)] for i in range(len(ring) + 1)]
                legs += [(c.gate_out.copy(), 0.0, "upright")]
                legs += [(p.copy(), 0.0, "upright") for p in loop]
                legs += [(c.gate_out.copy(), 0.0, "upright"), (c.gate_in.copy(), 0.0, "upright"),
                         (_zone_asset(world, c.poly), check, "upright"),
                         (c.gate_in.copy(), 0.0, "upright"), (c.gate_out.copy(), 0.0, "upright")]
        self._start(world, agent, legs)


class MaintenanceVisit(_Waypoints):
    """An authorised worker enters a restricted zone by its gate and works there.

    The hard negative for `zone_intrusion`: a person inside the restricted zone,
    deep, stationary at the asset for minutes — everything an intrusion rule
    keys on except the gate and the authorisation.
    """

    label: ClassVar[str] = "normal.maintenance.visit"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        zid = str(self.spec.get("zone"))
        comp = world.fences.component_of_zone(zid)
        dur = float(self.spec.get("dur_s", 120.0))
        if comp is None:
            poly = world._zones.get(zid)
            legs = [(_zone_asset(world, poly) if poly is not None else world.resolve_point(zid),
                     dur, "kneeling")]
        else:
            poly = world._zones.get(zid, comp.poly)
            legs = [(comp.gate_out.copy(), 0.0, "upright"), (comp.gate_in.copy(), 0.0, "upright"),
                    (_zone_asset(world, poly), dur, "kneeling"),
                    (comp.gate_in.copy(), 0.0, "upright"), (comp.gate_out.copy(), 0.0, "upright")]
        self._start(world, agent, legs)


class FenceTailgate(Behaviour):
    """Follow an authorised person through a *fence gate* on their credential.

    The intruder waits outside the gate of a fenced component (``zone`` picks
    it, else the one nearest), a few metres along the fence. When a person
    whose role the fence admits walks in through that gate, the intruder
    follows through the open gate within seconds — never over the fence, so a
    fence-vibration sensor has nothing to feel. With nobody to follow within
    ``wait_s`` it gives up and leaves without entering. Inside it goes to the
    asset, dwells ``dur_s``, and walks out by the gate.

    Truth (principle 1) is what happened: the label runs from the start of the
    wait, and ``sensor_log`` gets one ``access_control``/``tailgate`` record
    with the leader, the follower and both *actual* gate-passage times.
    The legitimate twin is `StaffEscort` (both people badged).
    """

    label: ClassVar[str] = "suspicious.access.tailgate"
    crosses_fence: ClassVar[bool] = True     # through the gate on someone else's badge
    GATE_R: ClassVar[float] = 3.0

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        zid = self.spec.get("zone")
        comp = world.fences.component_of_zone(str(zid)) if zid else None
        self._comp = comp or world.fences.nearest(agent.pos)
        if self._comp is None:
            self.done = True
            return
        c = self._comp
        n = c.gate_out - c.gate
        n = n / max(float(np.linalg.norm(n)), 1e-9)
        tng = np.array([n[1], -n[0]])
        self._wait_pt = c.gate_out + tng * float(self.spec.get("side_m", 3.0)) + n * 0.5
        self._wait_s = float(self.spec.get("wait_s", 240.0))
        self._dur = float(self.spec.get("dur_s", 45.0))
        self._phase = "approach"
        self._route = world.route(agent.pos, self._wait_pt)[:-1]
        self._near: dict[str, float] = {}       # authorised uid -> last t seen outside near the gate
        self._leader: tuple[str, float] | None = None
        self._legs: list[tuple[np.ndarray, float]] = []
        self._t_wait = t
        self._paused: float | None = None

    def _watch(self, t: float, agent: Agent, world: World) -> None:
        c = self._comp
        for a in world.agents.values():
            if a is agent or a.spec.role not in c.allowed_roles:
                continue
            d = float(np.linalg.norm(a.pos - c.gate))
            inside = c.contains(a.pos)
            if not inside and d <= self.GATE_R:
                self._near[a.uid] = t
            elif inside and d <= self.GATE_R + 0.5 and t - self._near.get(a.uid, -1e9) <= 10.0:
                self._leader = (a.uid, t)
                return

    def _walk(self, dt: float, target: np.ndarray, k: float, agent: Agent, world: World) -> None:
        v = planner.steer(agent.pos, target, agent.speed * k, world.neighbours(agent))
        agent.pos = agent.pos + v * dt
        agent.vel = v

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        c = self._comp
        if self._phase in ("approach", "wait"):
            if self._leader is None:
                self._watch(t, agent, world)
            if self._leader is not None:
                self._phase = "follow"
                self._legs = [(c.gate_out.copy(), 0.0), (c.gate_in.copy(), 0.0)]
            elif self._phase == "approach":
                while self._route and planner.arrived(agent.pos, self._route[0], 1.0):
                    self._route.pop(0)
                self._walk(dt, self._route[0] if self._route else self._wait_pt, 1.0, agent, world)
                if planner.arrived(agent.pos, self._wait_pt, 0.6):
                    self._phase, self._t_wait = "wait", t
                return
            else:
                agent.vel = np.zeros(2)
                if t - self._t_wait >= self._wait_s:
                    self.done = True           # nobody to follow: leave without entering
                return
        was_in = c.contains(agent.pos)
        if self._phase == "follow":
            self._walk(dt, self._legs[0][0], 1.2, agent, world)
            if not was_in and c.contains(agent.pos):
                uid, t_lead = self._leader
                world.sensor_log.append(
                    {"t": round(t, 3), "sensor": "access_control", "door": f"gate:{c.zone_ids[0]}",
                     "event": "tailgate", "uid": uid, "unbadged_uid": agent.uid, "people_through": 2,
                     "t_leader": round(t_lead, 3), "t_follower": round(t, 3),
                     "gap_s": round(t - t_lead, 3)})
            if planner.arrived(agent.pos, self._legs[0][0], 0.6):
                self._legs.pop(0)
                if not self._legs:
                    self._phase = "inside"
                    self._legs = [(_zone_asset(world, c.poly), self._dur),
                                  (c.gate_in.copy(), 0.0), (c.gate_out.copy(), 0.0)]
            return
        # inside: asset, dwell, out by the gate
        if not self._legs:
            self.done = True
            return
        goal, pause = self._legs[0]
        if self._paused is not None:
            agent.vel = np.zeros(2)
            if t - self._paused >= pause:
                self._legs.pop(0)
                self._paused = None
            return
        self._walk(dt, goal, 1.0, agent, world)
        if planner.arrived(agent.pos, goal, 0.6):
            if pause > 0:
                self._paused = t
            else:
                self._legs.pop(0)


class StaffEscort(Behaviour):
    """A badged worker walking in *with* a colleague: the twin of `FenceTailgate`.

    Follows ``target`` at ``gap_m`` through the gate and to the job, and stops
    when the target's gate-using work is over (its maintenance/patrol
    behaviour has finished) or the target has left. Same geometry and timing
    as a tailgater; the only difference is that this person holds a badge
    (role ``worker``, so the fence lets them through).
    """

    label: ClassVar[str] = "normal.maintenance.escort"
    WORK: ClassVar[tuple[str, ...]] = ("normal.maintenance.visit", "normal.patrol")

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        self._target = str(self.spec.get("target"))
        self._gap = float(self.spec.get("gap_m", 1.2))
        self._timeout = float(self.spec.get("timeout_s", 1800.0))
        self._worked = False

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        tg = world.agents.get(self._target)
        if tg is None or self.elapsed(t) > self._timeout:
            if tg is None and self._target in world.roster and not self._worked \
                    and self.elapsed(t) < self._timeout:
                agent.vel = np.zeros(2)       # partner not arrived yet: wait at the spawn
                return
            self.done = True
            return
        working = tg.label in self.WORK
        if working:
            self._worked = True
        elif self._worked:
            self.done = True
            return
        off = agent.pos - tg.pos
        n = float(np.linalg.norm(off))
        hold = tg.pos + (off / n if n > 1e-6 else np.array([1.0, 0.0])) * self._gap
        if n <= self._gap + 0.2 and float(np.linalg.norm(tg.vel)) < 0.1:
            agent.vel = np.zeros(2)
            return
        v = planner.steer(agent.pos, hold, max(agent.speed, 1.1 * float(np.linalg.norm(tg.vel))),
                          world.neighbours(agent))
        agent.pos = agent.pos + v * dt
        agent.vel = v


class Wildlife(Behaviour):
    """A small animal (fox, dog, deer fawn): fast, erratic, ground-level, not a person.

    The i-LIDS sterile-zone distraction set is mostly this. It goes under or
    through fences (`crosses_fence`), darts in bursts up to `top_mps`, stops,
    changes direction, then leaves by the nearest exit. The agent's
    `object_class` is "animal" (Person.identity["class"]), so ground truth
    marks it non-person and it is never a truth event.
    """

    label: ClassVar[str] = "distractor.wildlife"
    crosses_fence: ClassVar[bool] = True

    def begin(self, t: float, agent: Agent, world: World) -> None:
        super().begin(t, agent, world)
        at = self.spec.get("zone", self.spec.get("at"))
        self._centre = world.resolve_point(at) if at is not None else agent.pos.copy()
        self._radius = float(self.spec.get("radius_m", 15.0))
        self._dur = float(self.spec.get("dur_s", 60.0))
        self._top = float(self.spec.get("top_mps", 4.0))
        self._exit = self.spec.get("exit")
        self._leaving = False
        self._stop_until = -1.0
        self._leg_speed = self._top * 0.5
        self._goal = self._centre.copy()

    def step(self, dt: float, t: float, agent: Agent, world: World) -> None:
        assert self._goal is not None
        rng = world.rng
        if not self._leaving and self.elapsed(t) >= self._dur:
            self._leaving = True
            self._goal = world.resolve_point(self._exit) if self._exit is not None else \
                world._exit_pts[int(np.argmin(np.linalg.norm(world._exit_pts - agent.pos, axis=1)))] \
                if len(world._exit_pts) else agent.pos.copy()
            self._leg_speed = self._top * 0.8
        if t < self._stop_until:
            agent.vel = np.zeros(2)
            return
        if self._leaving:
            v = planner.steer(agent.pos, self._goal, self._leg_speed, None)
            agent.pos, agent.vel = agent.pos + v * dt, v
            if planner.arrived(agent.pos, self._goal, 1.0):
                self.done = True
            return
        if planner.arrived(agent.pos, self._goal, 0.5) or rng.random() < 0.04:
            if rng.random() < 0.3:
                self._stop_until = t + float(rng.uniform(0.5, 3.0))
            ang = float(rng.uniform(0, 2 * np.pi))
            r = self._radius * float(np.sqrt(rng.uniform()))
            self._goal = self._centre + r * np.array([np.cos(ang), np.sin(ang)])
            self._leg_speed = self._top * float(rng.uniform(0.3, 1.0))
        v = planner.steer(agent.pos, self._goal, self._leg_speed, None)
        agent.pos, agent.vel = agent.pos + v * dt, v


class NightPresence(Transit):
    """Being somewhere perfectly ordinary, at a time when nobody should be.

    The purest context-only behaviour in the catalogue, and it subclasses
    `Transit` without overriding `step` for exactly that reason: the motion *is*
    plain transit, tick for tick, and there is no cue whatsoever in the
    trajectory. Reimplementing the walk here would have quietly produced a
    slightly different path from an ordinary one, handing a classifier a cue
    that does not exist in real footage.

    It exists to stop the pipeline being evaluated as though every alarm must
    come from motion. A schedule is a sensor too, and a cheap one.
    """

    label: ClassVar[str] = "suspicious.temporal.afterhours"

    def begin(self, t: float, agent: Agent, world: World) -> None:
        self.spec.setdefault("to", self.spec.get("zone"))
        super().begin(t, agent, world)
        self._closed = tuple(self.spec.get("closed_hours", (22.0, 6.0)))

    @property
    def out_of_hours(self) -> bool:
        """Whether the scenario's time of day falls inside the closed window."""
        hour = float(self.spec.get("hour", 2.0))
        start, end = self._closed
        return hour >= start or hour < end if start > end else start <= hour < end


REGISTRY: dict[str, type[Behaviour]] = {
    "meet_and_greet": MeetAndGreet,
    "object_handoff": ObjectHandoff,
    "camera_avoidance": CameraAvoidance,
    "maintenance": Maintenance,
    "asset_tamper": AssetTamper,
    "smoke_break": SmokeBreak,
    "photograph": Photograph,
    "lost_tourist": LostTourist,
    "child_play": ChildPlay,
    "drunk_walk": DrunkWalk,
    "disabled_mobility": DisabledMobility,
    "sit_rest": SitRest,
    "weather_reaction": WeatherReaction,
    "crowd_form": CrowdForm,
    "vehicle_dropoff_loiter": VehicleDropoffLoiter,
    "vehicle_ram": VehicleRam,
    "night_presence": NightPresence,
    "queue": Queue,
    "climb_attempt": ClimbAttempt,
    "vandalism": Vandalism,
    "crowd_rush": CrowdRush,
    "throw_object": ThrowObject,
    "vehicle_move": VehicleMove,
    "drone_overflight": DroneOverflight,
    "argument": Argument,
    "fight": Fight,
    "theft_snatch": TheftSnatch,
    "person_down": PersonDown,
    "flee": Flee,
    "group_walk": GroupWalk,
    "counter_flow": CounterFlow,
    "transit": Transit,
    "loiter": Loiter,
    "loiter_intermittent": Loiter,
    "commute_wait": CommuteWait,
    "object_abandon": ObjectAbandon,
    "follow": Follow,
    "following": Follow,
    "outfit_change": OutfitChange,
    "concealment": OutfitChange,
    "recon": Recon,
    "perimeter_probe": PerimeterProbe,
    "camera_tamper": CameraTamper,
    "fence_cut": FenceCut,
    "fence_climb": FenceClimb,
    "patrol_guard": PatrolGuard,
    "maintenance_visit": MaintenanceVisit,
    "wildlife": Wildlife,
    "tailgating": Tailgate,
    "tailgate": Tailgate,
    "fence_tailgate": FenceTailgate,
    "staff_escort": StaffEscort,
    "erratic_motion": ErraticMotion,
    "wrong_zone_entry": ZoneIntrusion,
    "zone_intrusion": ZoneIntrusion,
}


def build(spec: dict[str, Any]) -> Behaviour:
    kind = spec.get("type", "transit")
    cls = REGISTRY.get(kind)
    if cls is None:
        raise KeyError(
            f"unknown behaviour {kind!r}; implemented: {sorted(REGISTRY)}. "
            "See docs/03_BEHAVIOUR_CATALOG.md for the full catalogue still to build."
        )
    return cls(spec=spec)

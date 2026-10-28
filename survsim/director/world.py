"""World state and the tick loop.

Truth is what happened, not what was planned: ground truth records the positions
the simulation actually produced after avoidance and clamping, which is why the
recorder reads back from agents rather than from behaviour targets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from shapely.geometry import Point, Polygon

from ..core.appearance import AppearanceState
from ..core.models import Person, Scenario
from ..core.rng import RngHub
from . import behaviours
from .behaviours import Behaviour
from .camera_director import CameraDirector


@dataclass
class Agent:
    spec: Person
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    queue: list[Behaviour] = field(default_factory=list)
    current: Behaviour | None = None
    appearance: AppearanceState | None = None
    look_at: np.ndarray | None = None
    posture: str = "upright"
    # Non-zero only for airborne actors. A drone has no foot contact, so any
    # ground-plane homography applied to it returns a meaningless position.
    altitude_m: float = 0.0

    @property
    def appearance_epoch(self) -> int:
        return self.appearance.epoch if self.appearance else 0

    @property
    def uid(self) -> str:
        return self.spec.id

    @property
    def speed(self) -> float:
        return self.spec.gait.speed_mps

    @property
    def height_m(self) -> float:
        return self.spec.body.height_m

    @property
    def pos3(self) -> np.ndarray:
        return np.array([self.pos[0], self.pos[1], self.altitude_m])

    @property
    def object_class(self) -> str:
        """"person" unless the spec says otherwise (wildlife: "animal")."""
        return str(self.spec.identity.get("class", "person"))

    @property
    def label(self) -> str:
        if self.current is None:
            return "normal.idle"
        return getattr(self.current, "dwell_label", self.current.label)


@dataclass
class Event:
    event_id: str
    t_start: float
    t_end: float | None
    label: str
    actors: list[str]
    target: dict[str, Any] | None = None
    zone: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "t_start": round(self.t_start, 3),
            "t_end": None if self.t_end is None else round(self.t_end, 3),
            "label": self.label,
            "actors": self.actors,
            "target": self.target,
            "zone": self.zone,
        }


class World:
    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.rngs = RngHub(scenario.seed)
        self.rng = self.rngs("planner")
        self.t = 0.0
        self.tick = 0
        self.dt = 1.0 / scenario.fps
        # `agents` is who is in the scene *now*. `roster` is everyone who has been
        # or will be, so identities survive a departure. Generated people arrive
        # on a staggered schedule (`_pending`) and leave through exits.
        self.agents: dict[str, Agent] = {}
        self.roster: dict[str, Agent] = {}
        self._pending: list[tuple[float, Agent]] = []
        self._exit_pts = np.array([w.pos[:2] for w in scenario.environment.exits], dtype=float
                                  ).reshape(-1, 2)
        self.objects: dict[str, dict[str, Any]] = {}
        self.events: list[Event] = []
        self.sensor_log: list[dict[str, Any]] = []
        self.interactions: dict[tuple[str, str], dict[str, Any]] = {}
        self.epochs: list[dict[str, Any]] = []
        self._open: dict[str, Event] = {}
        self._zones = {z.id: Polygon(z.polygon) for z in scenario.environment.zones}
        # `zone_of` must resolve overlapping/nested polygons to the *smallest*
        # containing zone, not "first in config order" — a person standing in a
        # small restricted zone nested inside a large public one is standing in
        # the restricted zone, and every consumer of `zone_of`/`world_df["zone"]`
        # (GT writers, `ZoneIntrusion`, `PathLoiterRule`'s role exemption) relies
        # on that being true. Ties (equal area) break on zone id, ascending, so
        # the result is deterministic and independent of declaration order.
        self._zone_order = sorted(
            self._zones.items(), key=lambda kv: (kv[1].area, kv[0])
        )
        self.cameras = CameraDirector(scenario.cameras)
        from .fences import Fences
        self.fences = Fences(scenario.environment.zones, scenario.environment.props)
        self._goal_rng = self.rngs("goals")
        self._spawn_agents()
        self._warm_up()

    # ------------------------------------------------------------------ setup

    def _spawn_agents(self) -> None:
        """Place scripted heroes now; schedule generated people as arrivals.

        Heroes keep their scripted timeline: they are in the world at t=0 exactly
        as the scenario says. Generated people arrive at times drawn uniformly over
        [-warmup, 0.75 * duration) — a Poisson process per entrance conditioned on
        its count — so the ones with negative arrival times are already spread
        along their routes when recording starts, and the rest trickle in.
        """
        rng = self.rngs("spawn")
        arrivals = self.rngs("arrivals")
        pop = self.scenario.population
        for person in pop.hero:
            self._admit(self._make_agent(person, rng), 0.0)
        warm = float(pop.warmup_s)
        horizon = max(0.75 * float(self.scenario.duration_s), self.dt)
        for person in self._generated_people():
            agent = self._make_agent(person, rng)
            if pop.arrivals == "t0":
                t_arrive = 0.0
            elif person.role == "intruder":
                # A generated positive must be on camera, not spent in warm-up.
                t_arrive = float(arrivals.uniform(0.0, 0.25 * horizon))
            else:
                t_arrive = float(arrivals.uniform(-warm, horizon))
            self.roster[person.id] = agent
            self._pending.append((t_arrive, agent))
        self._pending.sort(key=lambda x: x[0])

    def _make_agent(self, person: Person, rng: np.random.Generator) -> Agent:
        start = self._initial_position(person, rng)
        agent = Agent(spec=person, pos=start, appearance=AppearanceState.from_person(person))
        agent.queue = [behaviours.build(b.model_dump()) for b in person.intent]
        return agent

    def _admit(self, agent: Agent, t: float) -> None:
        self.agents[agent.uid] = agent
        self.roster[agent.uid] = agent
        self._open_epoch(agent, t)

    def _admit_arrivals(self) -> None:
        while self._pending and self._pending[0][0] <= self.t + 1e-9:
            _, agent = self._pending.pop(0)
            self._admit(agent, self.t)

    def _warm_up(self) -> None:
        """Run the world from the earliest arrival up to t=0 without recording."""
        if not self._pending or self._pending[0][0] >= 0.0:
            return
        start_tick = int(np.floor(self._pending[0][0] / self.dt))
        self.tick, self.t = start_tick, start_tick * self.dt
        while self.tick < 0:
            self.step()
        self.tick, self.t = 0, 0.0
        # Whatever happened before recording started is not ground truth; an
        # event still running at t=0 is truth from t=0.
        gone = {u for u in self.roster if u not in self.agents
                and u not in {a.uid for _, a in self._pending}}
        for uid in gone:
            self.roster.pop(uid)
        self.events = [e for e in self.events
                       if (e.t_end is None or e.t_end > 0.0) and not set(e.actors) & gone]
        for e in self.events:
            e.t_start = max(e.t_start, 0.0)
        for i, e in enumerate(self.events):
            e.event_id = f"e_{i:05d}"
        self.epochs = [r for r in self.epochs
                       if r["uid"] not in gone and (r["t_to"] is None or r["t_to"] > 0.0)]
        for r in self.epochs:
            r["t_from"] = max(r["t_from"], 0.0)
        self.sensor_log = [r for r in self.sensor_log if r.get("t", 0.0) >= 0.0]

    def _depart(self, agent: Agent) -> None:
        """A person whose plan ended at an exit has left the scene."""
        self.agents.pop(agent.uid, None)
        ev = self._open.pop(agent.uid, None)
        if ev is not None:
            ev.t_end = self.t
        for row in reversed(self.epochs):
            if row["uid"] == agent.uid and row["t_to"] is None:
                row["t_to"] = round(self.t, 3)
                break

    def at_exit(self, pos: np.ndarray, tol: float = 3.0) -> bool:
        if not len(self._exit_pts):
            return False
        return bool(np.linalg.norm(self._exit_pts - pos, axis=1).min() <= tol)

    def _generated_people(self) -> list[Person]:
        """Expand `population.generators` into concrete people."""
        from .population import expand_generator

        entries = self.scenario.population.generators
        if not entries:
            return []
        env = self.scenario.environment
        points = {
            "zones": [z.id for z in env.zones if z.tag in ("gate", "queue", "public")],
            "spawns": [w.id for w in env.spawn_points],
            "exits": [w.id for w in env.exits],
            "blindspots": [z.id for z in env.zones if z.tag == "blindspot"],
        }
        rng = self.rngs("population")
        out: list[Person] = []
        for i, entry in enumerate(entries):
            out.extend(expand_generator(entry, rng, points, index_offset=i * 100))
        return out

    def _initial_position(self, person: Person, rng: np.random.Generator) -> np.ndarray:
        for step in person.intent:
            extra = step.model_extra or {}
            if (src := extra.get("from")) is not None:
                try:
                    return self.resolve_point(src)
                except KeyError:
                    # `from` is not always a place: `flee` uses it for the threat
                    # person. A spawn point should never be the reason a scenario
                    # fails to start, so fall through to the default.
                    continue
        spawns = self.scenario.environment.spawn_points
        if spawns:
            return np.array(spawns[0].pos[:2], dtype=float)
        return rng.uniform(-5.0, 5.0, size=2)

    # ------------------------------------------------------------- geometry

    def resolve_point(self, ref: Any) -> np.ndarray:
        """Turn a zone id, spawn/exit id, or literal [x, y] into a 2D point."""
        if ref is None:
            return np.zeros(2)
        if isinstance(ref, (list, tuple)) and len(ref) >= 2:
            return np.array(ref[:2], dtype=float)
        rid = str(ref)
        if rid in self._zones:
            c = self._zones[rid].centroid
            return np.array([c.x, c.y])
        try:
            return np.array(self.scenario.environment.point(rid)[:2], dtype=float)
        except KeyError:
            pass
        for route in self.scenario.environment.routes:
            if route.id == rid:
                return np.array(route.waypoints[0], dtype=float)
        for prop in self.scenario.environment.props:
            if prop.id == rid:
                return np.array(prop.pos[:2], dtype=float)
        raise KeyError(f"cannot resolve point reference {rid!r}")

    def sample_goal(self, ref: Any) -> np.ndarray:
        """A concrete destination for `ref` that is not the same for everyone.

        Sending every walker to a zone centroid or an exact exit coordinate makes
        them pile up on one point where reciprocal avoidance guarantees only one
        can ever "arrive" — the others stand pinned at the goal forever. A zone
        becomes a random interior point; a spawn/exit gets a small spread.
        """
        rid = None if ref is None or isinstance(ref, (list, tuple)) else str(ref)
        rng = self._goal_rng
        if rid in self._zones:
            poly = self._zones[rid]
            x0, y0, x1, y1 = poly.bounds
            inner = poly.buffer(-0.5)
            test = inner if not inner.is_empty else poly
            # A goal inside a fence is unreachable for anyone the fence stops;
            # only a fenced zone asked for by name may be its own destination.
            fenced_ref = self.fences.component_of_zone(rid) is not None
            for _ in range(50):
                p = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
                if test.contains(Point(p[0], p[1])) and (fenced_ref or not self.fences.inside_any(p)):
                    return p
            return self.resolve_point(ref)
        base = self.resolve_point(ref)
        if rid is not None and (any(w.id == rid for w in self.scenario.environment.spawn_points)
                                or any(w.id == rid for w in self.scenario.environment.exits)):
            r, a = 1.5 * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
            return base + r * np.array([np.cos(a), np.sin(a)])
        return base

    def zone_of(self, pos: np.ndarray) -> str | None:
        """The smallest zone containing `pos`, not the first one declared.

        Overlapping/nested zone polygons are legitimate (e.g. `zone.substation`
        inside `zone.yard`); resolving by first-match-in-config-order silently
        reported the enclosing zone for anyone standing in the nested one.
        """
        p = Point(float(pos[0]), float(pos[1]))
        for zid, poly in self._zone_order:
            if poly.contains(p):
                return zid
        return None

    def route(self, start: np.ndarray, goal: np.ndarray) -> list[np.ndarray]:
        """Waypoints round the fences from `start` to `goal` ([] = straight)."""
        return self.fences.route(start, goal)

    def neighbours(self, agent: Agent, radius: float = 4.0) -> np.ndarray:
        others = [a.pos for a in self.agents.values() if a.uid != agent.uid]
        if not others:
            return np.zeros((0, 2))
        arr = np.asarray(others)
        d = np.linalg.norm(arr - agent.pos, axis=1)
        return arr[d < radius]

    def resolve_target(self, ref: Any) -> np.ndarray:
        """Like `resolve_point`, but also accepts a camera id or another person."""
        rid = str(ref)
        if rid in self.cameras.states:
            return self.camera_position(rid)[:2]
        if rid in self.agents:
            return self.agents[rid].pos.copy()
        return self.resolve_point(ref)

    def camera_position(self, camera_id: str) -> np.ndarray:
        cam = self.cameras.cameras.get(camera_id)
        if cam is None:
            raise KeyError(f"unknown camera {camera_id!r}")
        return np.asarray(cam.mount.pos, dtype=float)

    # --------------------------------------------------------------- objects

    def drop_object(self, agent: Agent, kind: str, pos: np.ndarray, t: float) -> None:
        oid = f"{agent.uid}_{kind}"
        self.objects[oid] = {"kind": kind, "pos": pos, "owner": agent.uid, "t_drop": t}

    def set_interaction(self, a: Agent, b: Agent, kind: str, t: float) -> None:
        """Record that two people are interacting.

        Aggression is a property of a *pair*, not a person: mutual facing and a
        closing gap are the cues, and storing them per-person would lose exactly
        the structure a detector needs.
        """
        key = tuple(sorted([a.uid, b.uid]))
        record = self.interactions.get(key)
        if record is None or record["kind"] != kind:
            self.interactions[key] = {"kind": kind, "t_start": round(t, 3),
                                      "t_last": round(t, 3), "uids": list(key)}
        else:
            record["t_last"] = round(t, 3)

    def transfer_object(self, giver: Agent, taker: Agent, kind: str, t: float) -> None:
        """Move a carried object between people — the core of a snatch or a handover."""
        oid = f"{giver.uid}_{kind}"
        self.objects[oid] = {"kind": kind, "pos": taker.pos.copy(),
                             "owner": taker.uid, "original_owner": giver.uid,
                             "t_drop": t, "transferred": True}
        self.sensor_log.append({"t": round(t, 3), "sensor": "object",
                                "event": "transfer", "object": kind,
                                "from": giver.uid, "to": taker.uid})

    # ------------------------------------------------------------ appearance

    def _open_epoch(self, agent: Agent, t: float) -> None:
        assert agent.appearance is not None
        self.epochs.append(
            {
                "uid": agent.uid,
                "epoch": agent.appearance.epoch,
                "t_from": round(t, 3),
                "t_to": None,
                "appearance_hash": agent.appearance.hash,
                "face_visible": agent.appearance.face_visible,
                "dominant_colour": agent.appearance.dominant_colour,
                "gait_style": agent.appearance.gait_style,
                "outfit": json.dumps(agent.appearance.outfit.model_dump()),
            }
        )

    def change_appearance(self, agent: Agent, delta: dict[str, Any], t: float) -> None:
        """Apply an appearance delta and, if it really changed, start a new epoch."""
        assert agent.appearance is not None
        if not agent.appearance.apply(delta, t):
            return
        for row in reversed(self.epochs):
            if row["uid"] == agent.uid and row["t_to"] is None:
                row["t_to"] = round(t, 3)
                break
        self._open_epoch(agent, t)
        observers = [c for c in self.cameras.states if self.sees(c, agent)]
        self.sensor_log.append(
            {
                "t": round(t, 3),
                "sensor": "appearance",
                "uid": agent.uid,
                "event": "appearance_change",
                "new_epoch": agent.appearance.epoch,
                "observed_by": observers,
                "unobserved": not observers,
            }
        )

    def sees(self, camera_id: str, agent: Agent) -> bool:
        """Whether a camera currently observes this person usefully.

        Uses the same pixel-height standard as coverage analysis, so a person who
        is technically inside the frustum but only a few pixels tall does not count
        as observed. Geometric only: scene occlusion needs the Unreal backend.
        """
        from ..core.geometry import usably_visible

        if self.cameras.states[camera_id].blind:
            return False
        live = self.cameras.live_camera(camera_id)
        return usably_visible(live, agent.pos3, agent.height_m)

    # -------------------------------------------------------- sensors/tamper

    def tamper_camera(self, camera_id: str, kind: str, t: float, agent: Agent) -> None:
        self.cameras.tamper(camera_id, kind, t)
        self.sensor_log.append(
            {"t": round(t, 3), "sensor": "camera_health", "camera": camera_id,
             "event": f"tamper.{kind}", "uid": agent.uid}
        )

    def damage(self, target: Any, agent: Agent, t: float) -> None:
        """Record damage to a surface or asset."""
        self.sensor_log.append(
            {"t": round(t, 3), "sensor": "asset", "target": str(target),
             "event": "damage", "uid": agent.uid})

    def breach(self, segment: Any, agent: Agent, t: float) -> None:
        self.sensor_log.append(
            {"t": round(t, 3), "sensor": "fence_vibration", "segment": str(segment),
             "event": "breach", "uid": agent.uid}
        )

    def access_event(self, door: str, authorised: Agent, follower: Agent, t: float) -> None:
        """One badge presentation, two people through — the tailgating signature."""
        self.sensor_log.append(
            {"t": round(t, 3), "sensor": "access_control", "door": door,
             "event": "badge_granted", "uid": authorised.uid, "people_through": 2,
             "unbadged_uid": follower.uid}
        )

    # ---------------------------------------------------------------- events

    def _sync_events(self) -> None:
        """Open an event when a behaviour starts, close it when the label changes."""
        for agent in self.agents.values():
            label = agent.label
            open_ev = self._open.get(agent.uid)
            if open_ev is not None and open_ev.label != label:
                open_ev.t_end = self.t
                self._open.pop(agent.uid)
                open_ev = None
            if open_ev is None and label != "normal.idle":
                ev = Event(
                    event_id=f"e_{len(self.events):05d}",
                    t_start=self.t,
                    t_end=None,
                    label=label,
                    actors=[agent.uid],
                    zone=self.zone_of(agent.pos),
                )
                self.events.append(ev)
                self._open[agent.uid] = ev

    def finalise(self) -> None:
        for ev in self._open.values():
            ev.t_end = self.t
        self._open.clear()
        for row in self.epochs:
            if row["t_to"] is None:
                row["t_to"] = round(self.t, 3)

    # ------------------------------------------------------------------ tick

    def step(self) -> None:
        self.cameras.step(self.t, self.dt)
        self._admit_arrivals()
        leaving: list[Agent] = []
        for agent in self.agents.values():
            if agent.current is None or agent.current.done:
                agent.current = None
                while agent.queue:
                    nxt = agent.queue.pop(0)
                    nxt.begin(self.t, agent, self)
                    if not nxt.done:
                        agent.current = nxt
                        break
            if agent.current is not None:
                prev = agent.pos.copy()
                agent.current.step(self.dt, self.t, agent, self)
                if self.fences and not getattr(agent.current, "crosses_fence", False) \
                        and agent.altitude_m <= 0.0:
                    # Truth is the position actually reached: a fence stops an
                    # unauthorised walker whatever the behaviour asked for.
                    got = self.fences.clamp(prev, agent.pos, agent.spec.role)
                    if got is not agent.pos:
                        agent.pos = got
                        agent.vel = (got - prev) / self.dt
            else:
                agent.vel = np.zeros(2)
                # Plan finished at an exit: the person has left, rather than
                # standing on the exit for the rest of the run.
                if agent.spec.intent and self.at_exit(agent.pos):
                    leaving.append(agent)
        for agent in leaving:
            self._depart(agent)
        self._sync_events()
        self.tick += 1
        self.t = self.tick * self.dt

"""Inject rare events at a controlled rate.

Hand-placed incidents are fine for a demo and useless for measurement: the base
rate is whatever the author happened to type, so precision and false-alarm figures
describe the scenario file rather than the system. Real sites have a *rate* — a
tamper every few months, a fight on a Saturday night — and evaluation only means
something when that rate is set deliberately.

This injector schedules events as a Poisson process with a configurable rate, so
a scenario can be asked for "0.5 suspicious events per camera-hour" and the
resulting precision figure has a defensible denominator.

Two properties that matter more than the sampling:

* **Rarity is preserved.** The injector will not manufacture a dense incident
  stream just because a short scenario would otherwise contain nothing. If the
  rate implies zero events in the duration, zero is the correct answer, and the
  report says so rather than quietly bumping the rate.
* **Hard negatives scale with positives.** Every injected suspicious event also
  schedules innocent look-alikes at the configured ratio, because a precision
  number computed without them is meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..core.models import BehaviourSpec, Person

# Events the injector can schedule, with the innocent behaviour that resembles
# each one closely enough to be a genuine hard negative.
INJECTABLE: dict[str, dict[str, Any]] = {
    "loiter": {"benign_twin": "commute_wait", "actors": 1, "severity": 0.4},
    "object_abandon": {"benign_twin": "commute_wait", "actors": 1, "severity": 0.7},
    "camera_tamper": {"benign_twin": "transit", "actors": 1, "severity": 0.9},
    "zone_intrusion": {"benign_twin": "transit", "actors": 1, "severity": 0.8},
    "recon": {"benign_twin": "transit", "actors": 1, "severity": 0.6},
    "erratic_motion": {"benign_twin": "transit", "actors": 1, "severity": 0.3},
    "argument": {"benign_twin": "meet", "actors": 2, "severity": 0.6},
    "fight": {"benign_twin": "meet", "actors": 2, "severity": 1.0},
    "theft_snatch": {"benign_twin": "transit", "actors": 2, "severity": 0.8},
    "person_down": {"benign_twin": "commute_wait", "actors": 1, "severity": 0.7},
}


@dataclass
class InjectedEvent:
    t: float
    kind: str
    actors: list[str]
    severity: float
    is_benign_twin: bool = False
    # Which suspicious event this innocent behaviour is the look-alike for. Needed
    # to label it correctly: a waiting commuter is a hard negative *for loitering*,
    # and the label has to say which, or precision cannot be attributed.
    twin_of: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"t": round(self.t, 2), "kind": self.kind, "actors": self.actors,
                "severity": self.severity, "benign_twin": self.is_benign_twin,
                "twin_of": self.twin_of}


@dataclass
class AnomalyInjector:
    """Schedules rare events at a stated rate."""

    rate_per_camera_hour: float = 0.5
    allowed: tuple[str, ...] = tuple(INJECTABLE)
    hard_negative_ratio: float = 3.0
    seed: int = 0
    # Events too close together stop being rare and start being a crowd scene.
    min_separation_s: float = 20.0

    def schedule(
        self,
        duration_s: float,
        n_cameras: int,
        available_uids: Sequence[str],
    ) -> tuple[list[InjectedEvent], dict[str, Any]]:
        """Draw event times and assign actors. Returns (events, report)."""
        rng = np.random.default_rng(self.seed)
        camera_hours = n_cameras * duration_s / 3600.0
        expected = self.rate_per_camera_hour * camera_hours

        n_events = int(rng.poisson(expected))
        uids = list(available_uids)
        events: list[InjectedEvent] = []
        accepted_times: list[float] = []

        for _ in range(n_events):
            if len(uids) < 2:
                break
            kind = str(rng.choice([k for k in self.allowed if k in INJECTABLE]))
            spec = INJECTABLE[kind]
            t = float(rng.uniform(0.0, max(duration_s - 10.0, 1.0)))
            # Compare against every accepted time, not just the previous draw:
            # times arrive unsorted, so checking one neighbour lets clusters
            # through and the events stop being rare.
            if any(abs(t - prev) < self.min_separation_s for prev in accepted_times):
                continue
            accepted_times.append(t)

            n_actors = min(int(spec["actors"]), len(uids))
            actors = [str(u) for u in rng.choice(uids, size=n_actors, replace=False)]
            for a in actors:
                uids.remove(a)
            events.append(InjectedEvent(t=t, kind=kind, actors=actors,
                                        severity=float(spec["severity"])))

        # Hard negatives: innocent people doing the look-alike behaviour. Without
        # them, precision measures nothing.
        twins: list[InjectedEvent] = []
        for event in events:
            for _ in range(int(round(self.hard_negative_ratio))):
                if not uids:
                    break
                actor = str(rng.choice(uids))
                uids.remove(actor)
                twins.append(InjectedEvent(
                    t=float(rng.uniform(0.0, max(duration_s - 10.0, 1.0))),
                    kind=INJECTABLE[event.kind]["benign_twin"],
                    actors=[actor], severity=0.0, is_benign_twin=True,
                    twin_of=event.kind))

        report = {
            "camera_hours": round(camera_hours, 3),
            "rate_per_camera_hour": self.rate_per_camera_hour,
            "expected_events": round(expected, 3),
            "scheduled_events": len(events),
            "hard_negatives": len(twins),
            "actors_remaining": len(uids),
            # Stated plainly: a short scenario at a realistic rate contains
            # nothing, and inflating the rate to avoid an empty run would destroy
            # the only number that makes precision interpretable.
            "note": (
                "no events scheduled: the requested rate implies fewer than one "
                "event in this duration. Lengthen the scenario or raise the rate "
                "deliberately rather than treating this as a failure."
                if not events else
                f"{len(events)} events over {round(camera_hours, 2)} camera-hours"
            ),
        }
        return events + twins, report


def apply_to_people(
    events: Sequence[InjectedEvent],
    people: dict[str, Person],
    env_points: dict[str, list[str]],
    seed: int = 0,
) -> int:
    """Splice scheduled events into the intent timelines of their actors."""
    rng = np.random.default_rng(seed)
    applied = 0

    def pick(options: list[str]) -> str | None:
        return str(rng.choice(options)) if options else None

    for event in events:
        zone = pick(env_points.get("zones", []))
        exit_point = pick(env_points.get("exits", []))
        for i, uid in enumerate(event.actors):
            person = people.get(uid)
            if person is None:
                continue
            partner = event.actors[1 - i] if len(event.actors) > 1 else None
            step = _build_step(event.kind, zone, exit_point, partner,
                               env_points, rng)
            if step is None:
                continue
            # Insert before the final transit so the person still leaves the scene.
            insert_at = max(len(person.intent) - 1, 0)
            person.intent.insert(insert_at, step)
            if event.is_benign_twin and event.twin_of:
                person.hard_negative_for = sorted(
                    {*person.hard_negative_for, event.twin_of})
            applied += 1
    return applied


def _build_step(kind, zone, exit_point, partner, env_points, rng) -> BehaviourSpec | None:
    dur = float(rng.uniform(20.0, 70.0))
    if kind in ("loiter", "commute_wait"):
        return BehaviourSpec(type=kind, zone=zone, dur_s=dur, activity="pace")
    if kind == "object_abandon":
        return BehaviourSpec(type=kind, zone=zone, object="backpack_large",
                             leave_dist_m=15.0, exit=exit_point)
    if kind == "camera_tamper":
        cameras = env_points.get("cameras", [])
        if not cameras:
            return None
        return BehaviourSpec(type=kind, camera=str(rng.choice(cameras)),
                             kind=str(rng.choice(["spray", "cover", "tilt"])))
    if kind == "zone_intrusion":
        restricted = env_points.get("restricted", []) or env_points.get("zones", [])
        return BehaviourSpec(type="wrong_zone_entry",
                             zone=str(rng.choice(restricted)) if restricted else zone,
                             dur_s=dur)
    if kind == "recon":
        return BehaviourSpec(type=kind, targets=env_points.get("cameras", [])[:3],
                             dwell_s=5.0)
    if kind == "erratic_motion":
        return BehaviourSpec(type=kind, zone=zone, dur_s=dur)
    if kind in ("argument", "fight") and partner:
        return BehaviourSpec(type=kind, partner=partner, dur_s=min(dur, 25.0))
    if kind == "theft_snatch" and partner:
        return BehaviourSpec(type=kind, victim=partner, exit=exit_point)
    if kind == "person_down":
        return BehaviourSpec(type=kind, dur_s=dur)
    if kind == "meet":
        return BehaviourSpec(type="commute_wait", zone=zone, dur_s=dur)
    if kind == "transit":
        return BehaviourSpec(type="transit", to=exit_point)
    return None

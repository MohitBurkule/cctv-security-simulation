"""Narratives: multi-phase person arcs, not single behaviours.

A real incident is a story across time and space — someone enters by the north
door dressed as a courier, loiters near the gateline, steps into a stairwell the
cameras miss, and leaves by the south door in a different jacket with a different
gait. Scoring that needs the *whole arc* as ground truth: which entrance, which
appearance epoch, which camera should have seen the hand-off.

Each narrative returns the behaviour timeline plus the evasion persona, so the
arc, the disguise and the matched innocent twin stay coupled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.models import BehaviourSpec, Environment, Person
from ..director import evasion


@dataclass
class Arc:
    """One person's full storyline."""

    person: Person
    intent: str                       # "benign" | "hostile"
    narrative: str
    phases: list[str] = field(default_factory=list)
    persona: evasion.Persona | None = None
    notes: dict[str, Any] = field(default_factory=dict)


def _zones(env: Environment, tag: str) -> list[str]:
    return [z.id for z in env.zones if z.tag == tag]


def _entrances(env: Environment) -> list[str]:
    return [w.id for w in env.spawn_points]


def _exits(env: Environment) -> list[str]:
    return [w.id for w in env.exits]


def _pick(rng: np.random.Generator, options: list[str], fallback: str | None = None) -> str | None:
    return str(rng.choice(options)) if options else fallback


# ------------------------------------------------------------------ narratives


def transit_through(env: Environment, rng: np.random.Generator) -> list[BehaviourSpec]:
    """Ordinary person crossing the site. The bulk of any real scene."""
    entries, leaves = _entrances(env), _exits(env)
    mid = _pick(rng, _zones(env, "public"), "zone.open")
    return [
        BehaviourSpec(type="transit", **{"from": _pick(rng, entries), "to": mid}),
        BehaviourSpec(type="transit", to=_pick(rng, leaves)),
    ]


def wait_then_leave(env: Environment, rng: np.random.Generator, dur: float) -> list[BehaviourSpec]:
    """Innocent waiting — the hard negative for loitering."""
    zone = _pick(rng, _zones(env, "queue") or _zones(env, "gate"), "zone.open")
    return [
        BehaviourSpec(type="transit", **{"from": _pick(rng, _entrances(env)), "to": zone}),
        BehaviourSpec(type="commute_wait", zone=zone, dur_s=dur, activity="idle"),
        BehaviourSpec(type="transit", to=_pick(rng, _exits(env))),
    ]


def loiter_arc(env: Environment, rng: np.random.Generator, dur: float) -> list[BehaviourSpec]:
    zone = _pick(rng, _zones(env, "gate") or _zones(env, "public"), "zone.open")
    return [
        BehaviourSpec(type="transit", **{"from": _pick(rng, _entrances(env)), "to": zone}),
        BehaviourSpec(type="loiter", zone=zone, dur_s=dur, activity="pace", radius_m=3.0),
        BehaviourSpec(type="transit", to=_pick(rng, _exits(env))),
    ]


def disguise_swap_arc(
    env: Environment, rng: np.random.Generator, dur: float, delta: dict[str, Any]
) -> list[BehaviourSpec]:
    """Enter by one door, change appearance unseen, leave by a different door.

    This is the flagship cross-epoch re-identification case: two appearance epochs
    for one identity, with the change deliberately unobserved, and different
    entrance and exit so a naive spatial prior fails too.
    """
    entries, leaves = _entrances(env), _exits(env)
    enter = _pick(rng, entries)
    # Prefer leaving by a *different* door than the one used to enter.
    others = [e for e in leaves if e.split(".")[-1] != str(enter).split(".")[-1]] or leaves
    blind = _pick(rng, _zones(env, "blindspot"))
    mid = _pick(rng, _zones(env, "gate") or _zones(env, "public"), "zone.open")

    phases = [
        BehaviourSpec(type="transit", **{"from": enter, "to": mid}),
        BehaviourSpec(type="loiter", zone=mid, dur_s=dur * 0.4, activity="scan"),
    ]
    if blind:
        phases.append(BehaviourSpec(type="outfit_change", in_zone=blind, dur_s=6.0, delta=delta))
    phases.append(BehaviourSpec(type="transit", to=_pick(rng, others)))
    return phases


def tailgate_arc(env: Environment, rng: np.random.Generator, target: str) -> list[BehaviourSpec]:
    door = next((p.id for p in env.props if p.type == "door"), None)
    return [
        BehaviourSpec(type="transit", **{"from": _pick(rng, _entrances(env)), "to": "zone.open"}),
        BehaviourSpec(type="tailgating", target=target, door=door, gap_m=1.2),
        BehaviourSpec(type="transit", to=_pick(rng, _exits(env))),
    ]


def recon_arc(env: Environment, rng: np.random.Generator, camera_ids: list[str]) -> list[BehaviourSpec]:
    targets = list(rng.choice(camera_ids, size=min(3, len(camera_ids)), replace=False)) \
        if camera_ids else []
    restricted = _zones(env, "restricted")
    if restricted:
        targets.append(restricted[0])
    return [
        BehaviourSpec(type="transit", **{"from": _pick(rng, _entrances(env)), "to": "zone.open"}),
        BehaviourSpec(type="recon", targets=[str(t) for t in targets], dwell_s=5.0, standoff_m=8.0),
        BehaviourSpec(type="transit", to=_pick(rng, _exits(env))),
    ]


def intrusion_arc(env: Environment, rng: np.random.Generator, dur: float) -> list[BehaviourSpec]:
    restricted = _pick(rng, _zones(env, "restricted"))
    if restricted is None:
        return loiter_arc(env, rng, dur)
    return [
        BehaviourSpec(type="transit", **{"from": _pick(rng, _entrances(env)), "to": "zone.open"}),
        BehaviourSpec(type="wrong_zone_entry", zone=restricted, dur_s=dur),
        BehaviourSpec(type="transit", to=_pick(rng, _exits(env)), gait={"style": "brisk"}),
    ]


def uniformed_intruder_arc(env: Environment, rng: np.random.Generator, dur: float) -> list[BehaviourSpec]:
    """H63 infiltration variant: the intrusion arc, walked by someone dressed as a guard.

    Spawn the person with ``role=UNIFORMED_INTRUDER_ROLE`` (appearance "guard") but give them no
    roster entry, badge or registered RF tag. Motion is exactly ``intrusion_arc`` -- a direct,
    non-patrol route into the restricted zone -- so only appearance differs from the intruder.
    """
    return intrusion_arc(env, rng, dur)


UNIFORMED_INTRUDER_ROLE = "guard"


def tamper_arc(env: Environment, rng: np.random.Generator, camera_id: str) -> list[BehaviourSpec]:
    return [
        BehaviourSpec(type="transit", **{"from": _pick(rng, _entrances(env)), "to": "zone.open"}),
        BehaviourSpec(type="camera_tamper", camera=camera_id,
                      kind=str(rng.choice(["spray", "cover", "tilt"])), dur_s=5.0),
        BehaviourSpec(type="transit", to=_pick(rng, _exits(env)), gait={"style": "brisk"}),
    ]


def _gate_of(env: Environment, zone: str | None) -> str | None:
    """The gate prop on a fenced zone's boundary (site.py puts one on each)."""
    if zone is None:
        return None
    stem = zone.split(".", 1)[-1]
    return next((p.id for p in env.props if p.type == "gate" and p.id == f"gate.{stem}"), None) \
        or next((p.id for p in env.props if p.type == "gate"), None)


def _fence_segment(env: Environment, rng: np.random.Generator) -> str | list[float] | None:
    """A fence panel, or — on sites without panels — a point on a restricted
    zone's fence away from its gate. H40 found `perimeter_probe` silently
    becoming `recon` on every site but `perimeter_site` for want of panels."""
    panels = [p.id for p in env.props if p.type == "fence_panel"]
    if panels:
        return str(rng.choice(panels))
    zones = [z for z in env.zones if z.tag == "restricted"]
    if not zones:
        return None
    z = zones[int(rng.integers(len(zones)))]
    xs, ys = zip(*z.polygon)
    gate = _gate_of(env, z.id)
    gpos = next((p.pos[:2] for p in env.props if p.id == gate), (np.mean(xs), np.mean(ys)))
    edges = [((min(xs) + max(xs)) / 2, min(ys)), ((min(xs) + max(xs)) / 2, max(ys)),
             (min(xs), (min(ys) + max(ys)) / 2), (max(xs), (min(ys) + max(ys)) / 2)]
    far = [e for e in edges if np.hypot(e[0] - gpos[0], e[1] - gpos[1]) > 2.0] or edges
    e = far[int(rng.integers(len(far)))]
    return [float(e[0]), float(e[1])]


def probe_arc(env: Environment, rng: np.random.Generator) -> list[BehaviourSpec]:
    """Probe along the fence, cut through, go for the asset inside, run."""
    seg = _fence_segment(env, rng)
    if seg is None:
        return recon_arc(env, rng, [])
    return [
        BehaviourSpec(type="transit", **{"from": _pick(rng, _entrances(env)), "to": "zone.open"}),
        BehaviourSpec(type="perimeter_probe", segment=seg, walk_along_m=8.0),
        BehaviourSpec(type="fence_cut", segment=seg, dur_s=40.0, dwell_s=float(rng.uniform(10, 40))),
        BehaviourSpec(type="transit", to=_pick(rng, _exits(env)), gait={"style": "run"}),
    ]


def climb_arc(env: Environment, rng: np.random.Generator) -> list[BehaviourSpec]:
    """Short probe, climb over the fence, go for the asset inside, run."""
    seg = _fence_segment(env, rng)
    if seg is None:
        return recon_arc(env, rng, [])
    return [
        BehaviourSpec(type="transit", **{"from": _pick(rng, _entrances(env)), "to": "zone.open"}),
        BehaviourSpec(type="perimeter_probe", segment=seg, walk_along_m=4.0),
        BehaviourSpec(type="fence_climb", segment=seg, climb_s=float(rng.uniform(6, 12)),
                      dwell_s=float(rng.uniform(10, 40))),
        BehaviourSpec(type="transit", to=_pick(rng, _exits(env)), gait={"style": "run"}),
    ]


def patrol_guard(env: Environment, rng: np.random.Generator) -> list[BehaviourSpec]:
    """Guard round: along every fence, in through the gate, check, out. Role: guard."""
    fenced = [z.id for z in env.zones if z.tag in ("restricted", "perimeter_fence")]
    if not fenced:
        return transit_through(env, rng)
    return [
        BehaviourSpec(type="transit", **{"from": _pick(rng, _entrances(env)), "to": "zone.open"}),
        BehaviourSpec(type="patrol_guard", check_s=float(rng.uniform(10, 30))),
        BehaviourSpec(type="transit", to=_pick(rng, _exits(env))),
    ]


def maintenance_visit(env: Environment, rng: np.random.Generator) -> list[BehaviourSpec]:
    """Worker enters a restricted zone by its gate, works at the asset, leaves. Role: worker."""
    zone = _pick(rng, _zones(env, "restricted"))
    if zone is None:
        return transit_through(env, rng)
    return [
        BehaviourSpec(type="transit", **{"from": _pick(rng, _entrances(env)), "to": "zone.open"}),
        BehaviourSpec(type="maintenance_visit", zone=zone, dur_s=float(rng.uniform(60, 240))),
        BehaviourSpec(type="transit", to=_pick(rng, _exits(env))),
    ]


def wildlife(env: Environment, rng: np.random.Generator, dur: float) -> list[BehaviourSpec]:
    """An animal crossing the site: enters at an edge, darts about, leaves."""
    at = _pick(rng, [z.id for z in env.zones if z.tag in ("restricted", "perimeter_fence", "public")],
               "zone.open")
    return [
        BehaviourSpec(type="transit", **{"from": _pick(rng, _entrances(env)), "to": at}),
        BehaviourSpec(type="wildlife", zone=at, dur_s=dur, radius_m=float(rng.uniform(8, 20)),
                      top_mps=float(rng.uniform(2.5, 5.0)), exit=_pick(rng, _exits(env))),
    ]


def fence_tailgate_arc(env: Environment, rng: np.random.Generator) -> list[BehaviourSpec]:
    """Wait by a fence gate, follow a badged person through it, go for the asset, leave."""
    zone = _pick(rng, _zones(env, "restricted"))
    if zone is None:
        return recon_arc(env, rng, [])
    return [
        BehaviourSpec(type="transit", **{"from": _pick(rng, _entrances(env)), "to": "zone.open"}),
        BehaviourSpec(type="fence_tailgate", zone=zone, wait_s=float(rng.uniform(150, 280)),
                      dur_s=float(rng.uniform(20, 60))),
        BehaviourSpec(type="transit", to=_pick(rng, _exits(env))),
    ]


def maintenance_visit_at(env: Environment, rng: np.random.Generator, zone: str) -> list[BehaviourSpec]:
    """`maintenance_visit` on a given zone (the fence_tailgate twin uses the tailgater's gate)."""
    return [
        BehaviourSpec(type="transit", **{"from": _pick(rng, _entrances(env)), "to": "zone.open"}),
        BehaviourSpec(type="maintenance_visit", zone=zone, dur_s=float(rng.uniform(60, 240))),
        BehaviourSpec(type="transit", to=_pick(rng, _exits(env))),
    ]


def staff_escort(env: Environment, rng: np.random.Generator, target: str, entrance: str | None
                 ) -> list[BehaviourSpec]:
    """A badged colleague who walks in with `target` (the legitimate twin of fence_tailgate)."""
    return [
        BehaviourSpec(type="staff_escort", target=target, gap_m=1.2,
                      **({"from": entrance} if entrance else {})),
        BehaviourSpec(type="transit", to=_pick(rng, _exits(env))),
    ]


# Twin narratives whose person needs an authorised role to pass the gate.
TWIN_ROLE = {"patrol_guard": "guard", "maintenance_visit": "worker"}


def abandon_arc(env: Environment, rng: np.random.Generator) -> list[BehaviourSpec]:
    zone = _pick(rng, _zones(env, "public") or _zones(env, "queue"), "zone.open")
    return [
        BehaviourSpec(type="transit", **{"from": _pick(rng, _entrances(env)), "to": zone}),
        BehaviourSpec(type="object_abandon", object="backpack_large", zone=zone,
                      leave_dist_m=15.0, exit=_pick(rng, _exits(env))),
        BehaviourSpec(type="transit", to=_pick(rng, _exits(env))),
    ]


def erratic_arc(env: Environment, rng: np.random.Generator, dur: float) -> list[BehaviourSpec]:
    zone = _pick(rng, _zones(env, "public"), "zone.open")
    return [
        BehaviourSpec(type="transit", **{"from": _pick(rng, _entrances(env)), "to": zone}),
        BehaviourSpec(type="erratic_motion", zone=zone, dur_s=dur, radius_m=6.0),
        BehaviourSpec(type="transit", to=_pick(rng, _exits(env))),
    ]


def follow_arc(env: Environment, rng: np.random.Generator, target: str, dur: float) -> list[BehaviourSpec]:
    return [
        BehaviourSpec(type="transit", **{"from": _pick(rng, _entrances(env)), "to": "zone.open"}),
        BehaviourSpec(type="follow", target=target, dist_m=8.0, dur_s=dur),
        BehaviourSpec(type="transit", to=_pick(rng, _exits(env))),
    ]


# Narratives whose *behaviour* is innocent. Pairing a hostile arc with one of
# these, wearing the same disguise, is what stops appearance predicting intent.
BENIGN_TWIN = {
    "loiter": "wait_then_leave",
    "disguise_swap": "wait_then_leave",
    "recon": "transit_through",
    "zone_intrusion": "maintenance_visit",
    "camera_tamper": "transit_through",
    "perimeter_probe": "patrol_guard",
    "fence_cut": "patrol_guard",
    "fence_climb": "patrol_guard",
    "object_abandon": "transit_through",
    "erratic_motion": "transit_through",
    "follow": "transit_through",
    "tailgate": "transit_through",
    "fence_tailgate": "maintenance_visit",
}

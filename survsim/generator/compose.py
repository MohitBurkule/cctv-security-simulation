"""Compose a full `Scenario` from a `SceneSpec`, then prove it is usable.

Generation is deliberately *generate → validate → repair*, never "sample and
hope". A scene is only useful if the cameras can actually see the action, the
blind spots really are blind, every referenced place exists, and appearance does
not give away intent. Each of those is checked here and reported as a concrete
issue rather than left to be discovered after a night of rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.models import Person, Population, Scenario
from ..director import evasion
from ..director.population import random_person
from . import narrative
from .place import place_cameras, verify_blind_spots
from .site import build_site
from .spec import SceneSpec


@dataclass
class GeneratedScene:
    scenario: Scenario
    arcs: list[narrative.Arc]
    reports: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.issues

    def summary(self) -> dict[str, Any]:
        s = self.scenario
        hostile = sum(a.intent == "hostile" for a in self.arcs)
        return {
            "name": s.name,
            "site": s.environment.id,
            "cameras": len(s.cameras),
            "entrances": len(s.environment.spawn_points),
            "people": len(s.population.hero),
            "hostile": hostile,
            "benign": len(self.arcs) - hostile,
            "duration_s": s.duration_s,
            "coverage": self.reports.get("placement", {}).get("area_coverage"),
            "valid": self.valid,
            "issues": self.issues,
        }


def compose(spec: SceneSpec) -> GeneratedScene:
    rng = np.random.default_rng(spec.seed)
    env = build_site(spec.site, seed=spec.seed)
    cameras, placement = place_cameras(env, spec.cameras, seed=spec.seed)
    camera_ids = [c.id for c in cameras]

    arcs: list[narrative.Arc] = []
    uid = 0

    def new_person(intent: str, persona: evasion.Persona | None) -> Person:
        nonlocal uid
        p = random_person(f"p{uid:03d}", rng, role="intruder" if intent == "hostile" else "commuter")
        uid += 1
        if persona is not None:
            delta = persona.delta()
            if isinstance(delta.get("gait"), dict) and delta["gait"].get("style"):
                p.gait.style = delta["gait"]["style"]  # type: ignore[assignment]
            for slot in ("top", "outer", "bottom", "shoes", "headwear"):
                if slot in delta:
                    setattr(p.outfit, slot, None if delta[slot] in (None, "none") else delta[slot])
            if isinstance(delta.get("accessories"), list):
                p.accessories = list(delta["accessories"])
        return p

    # ---- incident actors, each paired with an innocent twin wearing the same thing
    for plan in spec.incidents:
        for _ in range(plan.count):
            personas = evasion.generate_matched_population(
                n_pairs=1, seed=int(rng.integers(0, 2**31)),
                n_techniques=spec.population.evasion_techniques_per_actor,
                allow_protected=spec.population.allow_protected_attributes,
            )
            benign_persona, hostile_persona = personas[0], personas[1]

            hostile = new_person("hostile", hostile_persona)
            hostile.intent = _hostile_phases(plan.type, env, rng, camera_ids, arcs, spec)
            arcs.append(narrative.Arc(person=hostile, intent="hostile", narrative=plan.type,
                                      persona=hostile_persona,
                                      phases=[b.type for b in hostile.intent]))

            twin = new_person("benign", benign_persona)
            twin.hard_negative_for = [plan.type]
            twin.intent = _benign_phases(plan.type, env, rng, spec)
            if plan.type == "fence_tailgate":
                # the twin badges in at the gate the tailgater waits by
                z = next((b.model_extra.get("zone") for b in hostile.intent if b.type == "fence_tailgate"), None)
                if z is not None:
                    twin.intent = narrative.maintenance_visit_at(env, rng, str(z))
            twin_kind = narrative.BENIGN_TWIN.get(plan.type, "transit")
            if twin_kind in narrative.TWIN_ROLE and any(b.type == twin_kind for b in twin.intent):
                twin.role = narrative.TWIN_ROLE[twin_kind]
            arcs.append(narrative.Arc(person=twin, intent="benign",
                                      narrative=narrative.BENIGN_TWIN.get(plan.type, "transit"),
                                      persona=benign_persona,
                                      phases=[b.type for b in twin.intent]))

    # ---- background population, and extra hard negatives
    n_incident_actors = len(arcs)
    n_background = max(0, spec.population.count - n_incident_actors)
    n_negative = round(spec.population.hard_negative_ratio * max(1, n_incident_actors // 2))

    for i in range(n_background):
        p = new_person("benign", None)
        if i < n_negative:
            p.intent = narrative.wait_then_leave(env, rng, dur=float(rng.uniform(30, 90)))
            p.hard_negative_for = ["loiter"]
            tag = "wait_then_leave"
        else:
            p.intent = narrative.transit_through(env, rng)
            tag = "transit_through"
        arcs.append(narrative.Arc(person=p, intent="benign", narrative=tag,
                                 phases=[b.type for b in p.intent]))

    from ..director.population import random_animal
    for j in range(spec.population.wildlife):
        a = random_animal(f"w{j:03d}", rng)
        a.intent = narrative.wildlife(env, rng, dur=float(rng.uniform(30, 120)))
        arcs.append(narrative.Arc(person=a, intent="distractor", narrative="wildlife",
                                 phases=[b.type for b in a.intent]))

    scenario = Scenario(
        name=spec.name, seed=spec.seed, duration_s=spec.duration_s, fps=spec.fps,
        engine=spec.engine, environment=env, cameras=cameras,
        population=Population(hero=[a.person for a in arcs]),
    )

    scene = GeneratedScene(scenario=scenario, arcs=arcs,
                           reports={"placement": placement,
                                    "blind_spots": verify_blind_spots(cameras, env)})
    scene.issues = validate(scene, spec)
    return scene


def _hostile_phases(kind, env, rng, camera_ids, arcs, spec):
    dur = float(rng.uniform(25, 90))
    delta = {"outer": "none", "headwear": "cap"}
    existing = [a.person.id for a in arcs] or ["p000"]
    return {
        "loiter": lambda: narrative.loiter_arc(env, rng, dur),
        "disguise_swap": lambda: narrative.disguise_swap_arc(env, rng, dur, delta),
        "tailgate": lambda: narrative.tailgate_arc(env, rng, str(rng.choice(existing))),
        "fence_tailgate": lambda: narrative.fence_tailgate_arc(env, rng),
        "object_abandon": lambda: narrative.abandon_arc(env, rng),
        "recon": lambda: narrative.recon_arc(env, rng, camera_ids),
        "perimeter_probe": lambda: narrative.probe_arc(env, rng),
        "fence_cut": lambda: narrative.probe_arc(env, rng),
        "fence_climb": lambda: narrative.climb_arc(env, rng),
        "camera_tamper": lambda: narrative.tamper_arc(env, rng, str(rng.choice(camera_ids))),
        "zone_intrusion": lambda: narrative.intrusion_arc(env, rng, dur),
        "erratic_motion": lambda: narrative.erratic_arc(env, rng, dur),
        "follow": lambda: narrative.follow_arc(env, rng, str(rng.choice(existing)), dur),
    }.get(kind, lambda: narrative.loiter_arc(env, rng, dur))()


def _benign_phases(kind, env, rng, spec):
    twin = narrative.BENIGN_TWIN.get(kind, "transit_through")
    if twin == "patrol_guard":
        return narrative.patrol_guard(env, rng)
    if twin == "maintenance_visit":
        return narrative.maintenance_visit(env, rng)
    if twin == "wait_then_leave":
        return narrative.wait_then_leave(env, rng, dur=float(rng.uniform(25, 90)))
    return narrative.transit_through(env, rng)


# --------------------------------------------------------------------- validation


def validate(scene: GeneratedScene, spec: SceneSpec) -> list[str]:
    """Reject scenes that cannot produce meaningful data."""
    issues: list[str] = []
    s = scene.scenario
    placement = scene.reports.get("placement", {})

    if not s.cameras:
        issues.append("no cameras placed")
    if placement.get("area_coverage", 0.0) < 0.25:
        issues.append(f"coverage too low ({placement.get('area_coverage')}) — cameras see nothing")
    if not placement.get("meets_target", True):
        issues.append(
            f"coverage {placement.get('area_coverage')} below target {spec.cameras.target_coverage}"
        )

    # A "blind spot" a camera can see makes the disguise scenario untestable.
    for zid, info in (scene.reports.get("blind_spots") or {}).items():
        if not info["truly_blind"]:
            issues.append(f"{zid} is observed ({info['observed_cells']}/{info['cells']} cells)")

    needs_blindspot = any(a.narrative == "disguise_swap" for a in scene.arcs)
    if needs_blindspot and not any(z.tag == "blindspot" for z in s.environment.zones):
        issues.append("disguise_swap requested but the site has no blind spot")

    known = ({z.id for z in s.environment.zones}
             | {w.id for w in s.environment.spawn_points}
             | {w.id for w in s.environment.exits}
             | {p.id for p in s.environment.props})
    cam_ids = {c.id for c in s.cameras}
    for person in s.population.hero:
        for i, step in enumerate(person.intent):
            extra = step.model_extra or {}
            for key in ("zone", "from", "to", "in_zone", "segment", "door", "exit"):
                ref = extra.get(key)
                if isinstance(ref, str) and "." in ref and ref not in known:
                    issues.append(f"{person.id}.intent[{i}].{key} -> unknown {ref}")
            cam = extra.get("camera")
            if isinstance(cam, str) and cam not in cam_ids:
                issues.append(f"{person.id}.intent[{i}].camera -> unknown {cam}")

    hostile = [a for a in scene.arcs if a.intent == "hostile"]
    if hostile and not any(a.person.hard_negative_for for a in scene.arcs):
        issues.append("incidents present but no hard negatives — precision would be meaningless")

    personas = [a.persona for a in scene.arcs if a.persona is not None]
    if personas:
        audit = evasion.audit_confounds(personas)
        scene.reports["confound_audit"] = audit
        if not audit["balanced"]:
            issues.append(f"appearance predicts intent (ratio {audit['worst_imbalance']})")
        if not audit["protected_clean"]:
            issues.append(f"protected attributes imbalanced: {list(audit['protected_imbalance'])}")

    return issues

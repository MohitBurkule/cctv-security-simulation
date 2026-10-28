"""Generate N diverse *and valid* scenes.

"Diverse" and "random" are not the same thing. Independent random sampling
clusters near the middle of the design space and leaves whole regions unvisited,
so you get twenty scenes that are all six-camera daytime offices. Instead the
design space is **stratified**: each axis is divided into N bins and every bin is
used exactly once, then the axes are shuffled independently (a Latin-hypercube
design). Twenty scenes therefore span twenty different camera counts, twenty
different densities, and all the site types in rotation.

"Valid" is enforced separately, because diversity is worthless if half the scenes
are unusable. Every candidate goes through `compose`'s validation, and a failure
is *repaired* along the axis that caused it — too little coverage adds cameras
rather than rerolling the whole scene, which preserves the stratification.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .compose import GeneratedScene, compose
from .spec import CameraPlan, IncidentPlan, PopulationPlan, SceneSpec, SiteSpec

SITE_TYPES: tuple[str, ...] = (
    "office_building", "transit_station", "perimeter_site",
    "retail_street", "car_park", "event_plaza",
)

# Infinite sites v2. Kept out of SITE_TYPES so experiments that index it
# (`SITE_TYPES[i % len]`) keep drawing exactly the worlds they drew before.
SITE_TYPES_V2: tuple[str, ...] = (
    "loading_dock", "school_gate", "rail_platform", "hospital_entrance",
    "utility_yard", "construction_site", "farm_yard", "residential_close",
)
ALL_SITE_TYPES: tuple[str, ...] = SITE_TYPES + SITE_TYPES_V2

# Per v2 type: the incident it is built for, and the innocent behaviour that
# looks the same to a rule (compose pairs each incident actor with that twin;
# the queue zone in site._V2_ZONES is where the waiting hard negatives stand).
SITE_PROFILES: dict[str, dict[str, str]] = {
    "loading_dock": {"incident": "tailgate", "hard_negative": "driver waiting on the apron"},
    "school_gate": {"incident": "loiter", "hard_negative": "parent waiting at pick-up"},
    "rail_platform": {"incident": "object_abandon", "hard_negative": "passenger waiting with luggage"},
    "hospital_entrance": {"incident": "loiter", "hard_negative": "smoker at the shelter"},
    "utility_yard": {"incident": "fence_climb", "hard_negative": "maintenance visit through the gate"},
    "construction_site": {"incident": "zone_intrusion", "hard_negative": "hi-vis worker in the works"},
    "farm_yard": {"incident": "recon", "hard_negative": "farmer / contractor crossing the yard"},
    "residential_close": {"incident": "follow", "hard_negative": "resident / delivery walking the close"},
}

# Incident mixes, chosen so every scene exercises a different analytic problem.
INCIDENT_MIXES: tuple[tuple[str, ...], ...] = (
    ("loiter",),
    ("disguise_swap",),
    ("loiter", "tailgate"),
    ("recon", "camera_tamper"),
    ("object_abandon", "loiter"),
    ("zone_intrusion", "recon"),
    ("disguise_swap", "loiter", "follow"),
    ("perimeter_probe",),
    ("erratic_motion", "loiter"),
    ("tailgate", "zone_intrusion"),
)


def _latin_hypercube(n: int, rng: np.random.Generator) -> np.ndarray:
    """One stratified sample per bin, per axis, shuffled independently."""
    centres = (np.arange(n) + 0.5) / n
    return centres[rng.permutation(n)]


def _pick_from(values: Sequence, u: float):
    idx = min(int(u * len(values)), len(values) - 1)
    return values[idx]


@dataclass
class BatchResult:
    scenes: list[GeneratedScene]
    attempts: int
    repaired: int

    def diversity(self) -> dict[str, Any]:
        """How much of the design space the batch actually covers."""
        sites = [s.scenario.environment.id.split("_")[0] for s in self.scenes]
        cams = [len(s.scenario.cameras) for s in self.scenes]
        people = [len(s.scenario.population.hero) for s in self.scenes]
        narratives = sorted({a.narrative for s in self.scenes for a in s.arcs})
        return {
            "n_scenes": len(self.scenes),
            "distinct_site_types": len(set(sites)),
            "camera_count_range": [min(cams), max(cams)] if cams else [],
            "people_range": [min(people), max(people)] if people else [],
            "distinct_narratives": len(narratives),
            "narratives": narratives,
            "all_valid": all(s.valid for s in self.scenes),
        }


def generate_batch(
    n: int = 20,
    seed: int = 0,
    duration_s: float = 240.0,
    fps: float = 10.0,
    engine: str = "mock",
    site_types: Iterable[str] | None = None,
    max_repairs: int = 4,
) -> BatchResult:
    """Produce `n` scenes that are individually valid and collectively diverse."""
    rng = np.random.default_rng(seed)
    sites = tuple(site_types) if site_types else SITE_TYPES

    u_site = _latin_hypercube(n, rng)
    u_cams = _latin_hypercube(n, rng)
    u_people = _latin_hypercube(n, rng)
    u_mix = _latin_hypercube(n, rng)
    u_size = _latin_hypercube(n, rng)
    u_entr = _latin_hypercube(n, rng)

    scenes: list[GeneratedScene] = []
    attempts = repaired = 0

    for i in range(n):
        site_type = _pick_from(sites, u_site[i])
        n_cams = int(4 + round(u_cams[i] * 10))          # 4..14
        n_people = int(12 + round(u_people[i] * 38))     # 12..50
        extent = 50.0 + u_size[i] * 80.0                 # 50..130 m
        entrances = int(1 + round(u_entr[i] * 4))        # 1..5
        mix = _pick_from(INCIDENT_MIXES, u_mix[i])

        spec = SceneSpec(
            name=f"scene_{i:02d}_{site_type}",
            seed=seed * 1000 + i,
            duration_s=duration_s, fps=fps, engine=engine,  # type: ignore[arg-type]
            site=SiteSpec(type=site_type, footprint_m=(extent, extent * 0.75),  # type: ignore[arg-type]
                          entrances=entrances,
                          restricted_zones=1 + (i % 3),
                          blind_spots=1 + (i % 2)),
            cameras=CameraPlan(count=n_cams, target_coverage=0.55),
            population=PopulationPlan(count=n_people,
                                      evasion_techniques_per_actor=1 + (i % 3)),
            incidents=[IncidentPlan(type=m) for m in mix],  # type: ignore[arg-type]
        )

        scene = compose(spec)
        attempts += 1
        # Repair along the offending axis instead of rerolling: this keeps the
        # stratified design intact while fixing the specific defect.
        for _ in range(max_repairs):
            if scene.valid:
                break
            repaired += 1
            if any("coverage" in x for x in scene.issues):
                spec.cameras.count = min(spec.cameras.count + 3, 40)
                spec.cameras.target_coverage = max(0.3, spec.cameras.target_coverage - 0.08)
            if any("blind" in x.lower() for x in scene.issues):
                spec.site.blind_spots = max(1, spec.site.blind_spots)
                spec.site.footprint_m = (spec.site.footprint_m[0] * 1.15,
                                         spec.site.footprint_m[1] * 1.15)
            scene = compose(spec)
            attempts += 1

        scenes.append(scene)

    return BatchResult(scenes=scenes, attempts=attempts, repaired=repaired)


def write_batch(result: BatchResult, out_dir: Path) -> list[Path]:
    """Write each scenario as JSON plus a manifest describing the batch."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    manifest = []
    for scene in result.scenes:
        p = out_dir / f"{scene.scenario.name}.json"
        p.write_text(json.dumps(scene.scenario.model_dump(mode="json"), indent=1))
        written.append(p)
        manifest.append({**scene.summary(), "file": p.name,
                         "narratives": sorted({a.narrative for a in scene.arcs})})
    (out_dir / "manifest.json").write_text(json.dumps(
        {"diversity": result.diversity(), "attempts": result.attempts,
         "repairs": result.repaired, "scenes": manifest}, indent=1))
    return written

"""The user-facing scene specification.

A `SceneSpec` is the short, human-authored description of *what kind of scene you
want* — how many entrances, how many cameras, how many people, which incidents,
how many innocent look-alikes. The generator expands it into a complete
`Scenario` with concrete geometry, camera poses, people and behaviour timelines.

The split matters for maintainability: the spec is a stable contract that a user
writes, while the expansion is free to get smarter (better camera placement,
richer narratives) without invalidating anybody's spec files.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SiteType = Literal[
    "office_building",     # lobby, reception, controlled doors, several entrances
    "transit_station",     # concourse, platforms, gatelines, high flow
    "perimeter_site",      # fence line, gatehouse, yard — CNI / border style
    "retail_street",       # open street with shopfronts
    "car_park",            # rows, pillars, poor lighting, many blind spots
    "event_plaza",         # open area, stage, crowd
    # infinite sites v2 (survsim/generator/layout_sites_v2.py)
    "loading_dock",        # warehouse docks, trailers, drivers waiting on the apron
    "school_gate",         # pavement at pick-up, palisade, parents waiting
    "rail_platform",       # platform core, track trespass zone, passengers waiting
    "hospital_entrance",   # drop-off loop, smokers' shelter, ambulance bay
    "utility_yard",        # unmanned palisaded compound (substation / water / telecom)
    "construction_site",   # heras hoarding, cabins, frame going up, crane
    "farm_yard",           # barns, bales, hedges, fields
    "residential_close",   # houses round a turning head, driveways
]

IncidentType = Literal[
    "loiter", "disguise_swap", "tailgate", "object_abandon", "recon",
    "perimeter_probe", "camera_tamper", "fence_cut", "fence_climb", "zone_intrusion",
    "erratic_motion", "follow", "fence_tailgate",
]


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SiteSpec(Base):
    """Physical layout to generate."""

    type: SiteType = "office_building"
    footprint_m: tuple[float, float] = (80.0, 60.0)
    entrances: int = Field(default=2, ge=1, le=8)
    restricted_zones: int = Field(default=1, ge=0, le=6)
    # Blind spots are generated deliberately: an appearance change has to happen
    # somewhere unobserved or the cross-epoch ReID task is untestable.
    blind_spots: int = Field(default=1, ge=0, le=6)
    obstacles: int = Field(default=0, ge=0, le=40)
    seed_offset: int = 0


class CameraPlan(Base):
    """How many cameras of each kind, and how to place them."""

    count: int = Field(default=6, ge=1, le=64)
    placement: Literal["auto", "perimeter", "entrances"] = "auto"
    mix: dict[str, float] = Field(default_factory=lambda: {"fixed_bullet": 0.7, "ptz": 0.15,
                                                           "dome": 0.15})
    height_m: tuple[float, float] = (3.5, 6.0)
    resolution: tuple[int, int] = (1280, 720)
    fps: float = 10.0
    # Leaving coverage deliberately imperfect is realistic and is what makes
    # cross-camera hand-off and blind-spot exploitation meaningful.
    target_coverage: float = Field(default=0.75, ge=0.1, le=1.0)


class PopulationPlan(Base):
    """Who is in the scene."""

    count: int = Field(default=20, ge=1, le=200)
    # Innocent people per incident actor. Precision is meaningless without them.
    hard_negative_ratio: float = Field(default=3.0, ge=0.0, le=20.0)
    evasion_techniques_per_actor: int = Field(default=2, ge=0, le=5)
    allow_protected_attributes: bool = True
    appearance_pool: str | None = None
    # Non-person distractors (fox, dog, deer...) — the i-LIDS sterile-zone
    # negatives. Default 0 so existing specs generate exactly as before.
    wildlife: int = Field(default=0, ge=0, le=50)


class IncidentPlan(Base):
    type: IncidentType
    count: int = Field(default=1, ge=1, le=50)
    # Optional window; the generator schedules inside the scenario duration.
    t_window: tuple[float, float] | None = None


class SceneSpec(Base):
    """Everything needed to generate a scenario."""

    name: str = "generated_scene"
    seed: int = 0
    duration_s: float = 300.0
    fps: float = 10.0
    engine: Literal["mock", "unreal"] = "mock"
    site: SiteSpec = SiteSpec()
    cameras: CameraPlan = CameraPlan()
    population: PopulationPlan = PopulationPlan()
    incidents: list[IncidentPlan] = Field(default_factory=list)
    engine_map: str | None = None
    ground_z_cm: float = 0.0

    @property
    def n_ticks(self) -> int:
        return round(self.duration_s * self.fps)

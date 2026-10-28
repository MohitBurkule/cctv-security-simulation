"""Typed scenario model. YAML and the Python SDK both resolve to these objects.

Units are SI throughout: metres, seconds, degrees for human-facing angles.
Backends convert at their own boundary (UE uses centimetres and a left-handed
frame); engine units never appear above the backend interface.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Vec3 = tuple[float, float, float]


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- world


class Zone(Base):
    id: str
    tag: Literal[
        "public", "restricted", "perimeter_fence", "gate", "queue",
        "parking", "sensitive_asset", "blindspot",
    ] = "public"
    polygon: list[tuple[float, float]] = Field(min_length=3)
    allowed_roles: list[str] = []
    # restricted / perimeter_fence zones are fenced obstacles by default
    # (survsim/director/fences.py); set False for an open, painted-line zone.
    fenced: bool | None = None


class Waypoint(Base):
    id: str
    pos: Vec3


class Prop(Base):
    id: str
    type: str
    pos: Vec3 = (0.0, 0.0, 0.0)
    cuttable: bool = False
    seats: int = 0
    length_m: float | None = None
    reachable_height_m: float | None = None


class Route(Base):
    id: str
    waypoints: list[tuple[float, float]] = Field(min_length=2)


class Environment(Base):
    id: str
    engine_map: str | None = None
    lat_lon: tuple[float, float] | None = None
    units: Literal["m"] = "m"
    nav: dict[str, Any] = {}
    zones: list[Zone] = []
    routes: list[Route] = []
    spawn_points: list[Waypoint] = []
    exits: list[Waypoint] = []
    props: list[Prop] = []
    lighting: dict[str, Any] = {}

    def zone(self, zid: str) -> Zone:
        for z in self.zones:
            if z.id == zid:
                return z
        raise KeyError(f"unknown zone {zid!r}")

    def point(self, pid: str) -> Vec3:
        for w in list(self.spawn_points) + list(self.exits):
            if w.id == pid:
                return w.pos
        raise KeyError(f"unknown spawn/exit {pid!r}")


# -------------------------------------------------------------------------- camera


class Optics(Base):
    hfov_deg: float | list[float] = 78.0
    model: str | None = None
    fov_deg: float | None = None
    distortion: dict[str, Any] = {}
    rolling_shutter_ms: float = 0.0


class Capture(Base):
    res: tuple[int, int] = (1920, 1080)
    fps: float = 25.0
    exposure: str = "auto"
    night_mode: str = "off"
    palette: str | None = None


class Stream(Base):
    codec: str = "h264"
    bitrate_kbps: int = 2500
    gop: int = 50
    clock_offset_ms: float = 0.0
    drift_ppm: float = 0.0
    packet_loss: float = 0.0


class PTZ(Base):
    pan: tuple[float, float] = (-170.0, 170.0)
    tilt: tuple[float, float] = (-90.0, 10.0)
    zoom: tuple[float, float] | None = None
    speed_dps: float = 60.0
    patrol: list[dict[str, Any]] = []
    auto_track: bool = False


class Mount(Base):
    pos: Vec3
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


class Camera(Base):
    id: str
    type: Literal[
        "fixed_bullet", "dome", "ptz", "fisheye", "thermal", "body_cam", "drone"
    ] = "fixed_bullet"
    mount: Mount
    optics: Optics = Optics()
    capture: Capture = Capture()
    stream: Stream = Stream()
    ptz: PTZ | None = None
    presets: dict[str, dict[str, float]] = {}
    degrade: dict[str, Any] = {}
    tamper: list[dict[str, Any]] = []
    overlay: dict[str, Any] = {}

    @property
    def hfov_deg(self) -> float:
        """Widest field of view; a PTZ zoom range reports its wide end."""
        h = self.optics.hfov_deg
        return float(max(h)) if isinstance(h, list) else float(h)


# -------------------------------------------------------------------------- people


class Body(Base):
    height_m: float = 1.70
    shoulder_w: float = 0.45
    bmi_class: Literal["slim", "normal", "heavy", "athletic"] = "normal"
    skin_tone: int = Field(default=5, ge=1, le=10)  # Monk Skin Tone scale
    sex_presentation: Literal["m", "f", "x"] = "x"
    age_band: str = "25-40"


class Gait(Base):
    style: Literal[
        "normal", "brisk", "slow", "limp_L", "limp_R",
        "shuffle", "drunk", "military", "crouch", "run",
    ] = "normal"
    speed_mps: float = 1.35
    arm_swing: float = 1.0
    posture: Literal["upright", "slouched"] = "upright"


class Outfit(Base):
    top: str = "tshirt"
    outer: str | None = None
    bottom: str = "jeans"
    shoes: str = "trainers"
    headwear: str | None = None
    colours: dict[str, str] = {}
    pattern: dict[str, str] = {}


class Adversarial(Base):
    camera_awareness: float = Field(default=0.0, ge=0.0, le=1.0)
    face_hiding: float = Field(default=0.0, ge=0.0, le=1.0)
    role_disguise: str | None = None
    avoid_zones: list[str] = []


class BehaviourSpec(Base):
    """One entry in a person's intent timeline. `type` selects the behaviour class."""

    model_config = ConfigDict(extra="allow")
    type: str


class Person(Base):
    id: str
    role: str = "commuter"
    identity: dict[str, Any] = {}
    body: Body = Body()
    face: dict[str, Any] = {}
    hair: dict[str, Any] = {}
    outfit: Outfit = Outfit()
    accessories: list[str] = []
    gait: Gait = Gait()
    adversarial: Adversarial = Adversarial()
    change_events: list[dict[str, Any]] = []
    rf: list[dict[str, Any]] = []
    intent: list[BehaviourSpec] = []
    hard_negative_for: list[str] = []


# ------------------------------------------------------------------------ scenario


class Weather(Base):
    preset: str = "clear"
    fog_density: float = 0.0
    rain_intensity: float = 0.0
    wind_mps: float = 0.0
    wetness: float = 0.0
    schedule: list[dict[str, Any]] = []


class Population(Base):
    background: dict[str, Any] = {}
    hero: list[Person] = []
    generators: list[dict[str, Any]] = []
    distractors: list[dict[str, Any]] = []
    # Generated people are pre-warmed: the Director runs this long before frame 0
    # so the scene is already populated, then keeps staggering arrivals.
    warmup_s: float = 30.0
    arrivals: Literal["staggered", "t0"] = "staggered"


class Output(Base):
    root: str = "./out/${name}/${seed}"
    modalities: list[str] = ["rgb"]
    gt: list[str] = ["mot", "events", "world_parquet", "calib"]
    video: dict[str, Any] = {}
    degrade_profiles: list[str] = ["clean"]
    save_clean_and_degraded: bool = True


class Scenario(Base):
    schema_version: int = 1
    name: str
    seed: int = 0
    duration_s: float = 60.0
    fps: float = 25.0
    engine: Literal["unreal", "isaac", "mock"] = "mock"
    environment: Environment
    cameras: list[Camera] = []
    time_of_day: dict[str, Any] = {}
    weather: Weather = Weather()
    population: Population = Population()
    anomaly_injector: dict[str, Any] = {}
    sensors_extra: dict[str, Any] = {}
    output: Output = Output()

    @field_validator("cameras")
    @classmethod
    def _unique_camera_ids(cls, v: list[Camera]) -> list[Camera]:
        ids = [c.id for c in v]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate camera ids: {sorted(dupes)}")
        return v

    @property
    def n_ticks(self) -> int:
        return round(self.duration_s * self.fps)

"""Export a completed run as a manifest the Unreal editor can rebuild.

**Why this and not UnrealCV.** The packaged-runtime route drove the engine live
over a socket, and it has cost more than it returned: the server implements no
wardrobe or gait commands, it serves one client at a time, and this build now
hangs at frame 1 (docs/12 §2). More fundamentally, live socket control couples
our simulation clock to the engine's, which is where "ground truth must follow
the live camera" gets hard.

Unreal Editor 5.8 offers a better shape. It ships PythonScriptPlugin,
SequencerScripting and MovieRenderPipeline, so the pipeline becomes:

    Director runs headless  ->  ground truth is final and known
              |
              v
    this manifest           ->  positions, appearance, cameras, timing
              |
              v
    in-editor Python        ->  builds MetaHumans and a Level Sequence
              |
              v
    Movie Render Queue      ->  photoreal frames, offline, deterministic

Ground truth is decided *before* rendering rather than read back from it, so a
dropped frame or a slow render cannot desynchronise anything. Re-rendering a run
at higher quality changes the pixels and not one value of the truth. And because
nothing is live, a render can take as long as it needs — which is the whole
point of Movie Render Queue over a real-time capture.

**Units are converted here and nowhere else.** SurvSim is metres, Z-up,
right-handed, yaw counter-clockwise; Unreal is centimetres, left-handed, yaw
clockwise. The conversion is ×100, negate Y, negate yaw. Scattering that across
an export would reproduce the inverted-pitch bug in a new place, so every
conversion in this file goes through `to_unreal_*`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

M_TO_CM = 100.0

# Manifest format version. The in-editor script checks it, because a silently
# mismatched manifest would build a subtly wrong scene rather than fail.
MANIFEST_VERSION = 1


def to_unreal_location(pos: Sequence[float]) -> list[float]:
    """Metres (x, y, z) right-handed -> Unreal centimetres, Y negated."""
    x, y = float(pos[0]), float(pos[1])
    z = float(pos[2]) if len(pos) > 2 else 0.0
    return [x * M_TO_CM, -y * M_TO_CM, z * M_TO_CM]


def to_unreal_rotation(yaw_deg: float, pitch_deg: float = 0.0,
                       roll_deg: float = 0.0) -> list[float]:
    """Degrees -> Unreal (pitch, yaw, roll) with yaw handedness flipped.

    Pitch is *not* negated. Unreal's pitch sign already matches ours for a
    camera looking down at a negative pitch, and negating it here is precisely
    the bug that once pointed every camera at the sky.
    """
    return [float(pitch_deg), -float(yaw_deg), float(roll_deg)]


# MetaHuman body constraints, verified present on a real character in UE 5.8.2
# (`scripts/ue/build_character.py`). All 30, in the engine's own order and
# spelling — the names are the API, and a typo produces a character that builds
# successfully at the default measurement rather than an error.
#
# Measurements are in CENTIMETRES, except the last three which are 0..1 shape
# parameters.
BODY_CONSTRAINT_NAMES = (
    "Height", "Chest", "Bust Span", "Underbust", "Waist", "Rise", "Inseam",
    "Hip", "High Hip", "Neck Base", "Neck", "Thigh", "Knee", "Calf",
    "Front Interscye Length", "Upper Arm Length", "Lower Arm Length", "Bicep",
    "Elbow", "Wrist", "Neck to Waist", "Across Shoulder", "Shoulder Height",
    "Shoulder to Apex", "Neck Length", "Forearm", "Hand Circumference",
    "Fat", "Masculine/Feminine", "Muscularity",
)

# Verified range for Height on a default character.
HEIGHT_MIN_CM, HEIGHT_MAX_CM = 135.0, 220.0

# Our coarse build labels, as (Fat, Muscularity) in the engine's 0..1 shape
# parameters. Deliberately coarse: these are the levers a scenario author
# actually reaches for, and anyone wanting finer control can set constraints
# directly.
BUILD_SHAPES = {
    "slim": (0.2, 0.35),
    "average": (0.5, 0.5),
    "heavy": (0.8, 0.4),
    "athletic": (0.3, 0.8),
}


def body_constraints_for(height_m: float, build: str = "average",
                         femininity: float | None = None) -> dict[str, float]:
    """Map our appearance fields onto MetaHuman body constraint targets.

    Kept here, in pure Python, rather than inside the editor script: this is the
    part with decisions in it, and decisions deserve tests. The editor half only
    looks these up and assigns them.

    Height is clamped to the range the engine accepts. Silently clamping is the
    right behaviour because the alternative — passing 250 cm and having the
    engine quietly ignore it — produces a person of default height whose ground
    truth claims otherwise, and nothing downstream could detect that.
    """
    fat, muscularity = BUILD_SHAPES.get(build, BUILD_SHAPES["average"])
    targets = {
        "Height": min(max(height_m * M_TO_CM, HEIGHT_MIN_CM), HEIGHT_MAX_CM),
        "Fat": fat,
        "Muscularity": muscularity,
    }
    if femininity is not None:
        # 0 = masculine, 1 = feminine. Exposed because presenting as a different
        # gender is a documented evasion technique, and one that defeats
        # appearance-based ReID far more effectively than a change of coat.
        targets["Masculine/Feminine"] = min(max(float(femininity), 0.0), 1.0)
    return targets


@dataclass
class WardrobeSpec:
    """Appearance in terms an artist — or the MetaHuman API — can act on.

    Deliberately not a MetaHuman asset path. The simulator describes what a
    person looks like; mapping that onto whatever the engine build actually
    offers is the engine layer's job, and keeping the two apart is why the
    simulator survived the wardrobe commands being missing.
    """

    top_colour: str = "#404040"
    bottom_colour: str = "#202020"
    outer: str | None = None
    headwear: str | None = None
    footwear: str | None = None
    accessories: list[str] = field(default_factory=list)
    face_covered: float = 0.0
    height_m: float = 1.70
    build: str = "average"

    def as_dict(self) -> dict[str, Any]:
        return {
            "top_colour": self.top_colour, "bottom_colour": self.bottom_colour,
            "outer": self.outer, "headwear": self.headwear,
            "footwear": self.footwear, "accessories": list(self.accessories),
            "face_covered": round(float(self.face_covered), 3),
            "height_m": round(float(self.height_m), 3), "build": self.build,
        }


def wardrobe_from_appearance(state: Any, height_m: float = 1.70) -> WardrobeSpec:
    """Build a wardrobe spec from an `AppearanceState`."""
    outfit = getattr(state, "outfit", None)

    def attr(name: str, default: Any = None) -> Any:
        return getattr(outfit, name, default) if outfit is not None else default

    accessories = list(getattr(state, "accessories", []) or [])
    # Face covering is a *property of what is worn*, so it is derived here
    # rather than trusted as a separate field that could disagree with the
    # accessory list.
    covering = {"mask", "balaclava", "niqab", "scarf", "face_mask", "covid_mask"}
    covered = 1.0 if any(a in covering for a in accessories) else 0.0

    return WardrobeSpec(
        top_colour=str(attr("top_colour", attr("top", "#404040")) or "#404040"),
        bottom_colour=str(attr("bottom_colour", attr("bottom", "#202020")) or "#202020"),
        outer=attr("outer"),
        headwear=attr("headwear") or ("hat" if "hat" in accessories else None),
        footwear=attr("footwear"),
        accessories=accessories,
        face_covered=covered,
        height_m=float(height_m),
        build=str(getattr(state, "build", "average") or "average"),
    )


def wardrobe_from_identity_row(ident: dict[str, Any]) -> dict[str, Any]:
    """Build a wardrobe dict straight from an identities.parquet row.

    identities.parquet (survsim/gt/writers.py `_write_identities`) carries no
    `wardrobe` field: appearance lives in `outfit` (a JSON string of
    `Outfit.model_dump()`) and biometrics are flat columns
    (`core.appearance.SOFT_BIOMETRIC_KEYS`, which includes `height_m`). Without
    this, `build_manifest` fell through to `WardrobeSpec()`'s 1.70 m default
    for every person — the Director's real per-person height (CLAUDE.md
    principle 1: truth is what happened, and the Director already knows it)
    never reached the render.
    """
    outfit = ident.get("outfit")
    if isinstance(outfit, str):
        try:
            outfit = json.loads(outfit)
        except ValueError:
            outfit = {}
    outfit = outfit if isinstance(outfit, dict) else {}
    accessories = list(ident.get("accessories") or [])
    covering = {"mask", "balaclava", "niqab", "scarf", "face_mask", "covid_mask"}
    covered = 1.0 if any(a in covering for a in accessories) else 0.0
    height_m = _valid_height(ident.get("height_m"))
    return WardrobeSpec(
        top_colour=str(outfit.get("top_colour", outfit.get("top", "#404040")) or "#404040"),
        bottom_colour=str(outfit.get("bottom_colour", outfit.get("bottom", "#202020")) or "#202020"),
        outer=outfit.get("outer"),
        headwear=outfit.get("headwear") or ("hat" if "hat" in accessories else None),
        footwear=outfit.get("footwear"),
        accessories=accessories,
        face_covered=covered,
        height_m=height_m if height_m is not None else 1.70,
        build=str(ident.get("build") or "average"),
    ).as_dict()


def _valid_height(value: Any) -> float | None:
    """`value` as a real height in metres, or None (missing/NaN/non-positive)."""
    try:
        h = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(h) or h <= 0.0:   # pandas turns a missing column into NaN
        return None
    return h


def person_wardrobe(ident: dict[str, Any]) -> dict[str, Any]:
    """The manifest wardrobe for one identities row: a `wardrobe` field if the
    caller already supplied one (e.g. tests), else derived from the row —
    but the Director's own `height_m` always wins, because it is ground truth
    and a stale/default wardrobe height must never override it."""
    wardrobe = ident.get("wardrobe")
    wardrobe = dict(wardrobe) if isinstance(wardrobe, dict) else wardrobe_from_identity_row(ident)
    height_m = _valid_height(ident.get("height_m"))
    if height_m is not None:
        wardrobe["height_m"] = round(height_m, 3)
    return wardrobe


def _hfov(camera: Any) -> float | None:
    """The horizontal FOV the simulator projects with, or None if unknown."""
    value = getattr(camera, "hfov_deg", None)
    if value is None:
        optics = getattr(camera, "optics", None)
        value = getattr(optics, "hfov_deg", None)
    if isinstance(value, (list, tuple)):      # a zoom range: use the wide end
        value = value[0] if value else None
    return float(value) if value else None


def camera_entry(camera: Any) -> dict[str, Any]:
    """One camera, in Unreal's frame, with the intrinsics a render needs."""
    mount = camera.mount
    capture = dict(getattr(camera, "capture", {}) or {})
    res = capture.get("res", (1280, 720))
    return {
        "id": camera.id,
        "location": to_unreal_location(mount.pos),
        "rotation": to_unreal_rotation(mount.yaw, getattr(mount, "pitch", 0.0),
                                       getattr(mount, "roll", 0.0)),
        "resolution": [int(res[0]), int(res[1])],
        "fps": float(capture.get("fps", 25.0)),
        # Horizontal FOV, taken from the SAME property geometry.intrinsics uses
        # to project ground truth. This once read `capture.fov_deg`, which no
        # camera has, so every manifest carried None and Unreal rendered with its
        # default ~40 degree lens while ground truth was projected at, say, 95 —
        # every box on the wrong pixels, silently.
        "fov_deg": _hfov(camera),
        "focal_mm": capture.get("focal_mm"),
        "sensor_mm": capture.get("sensor_mm"),
    }


def track_entries(world_rows: Iterable[dict[str, Any]], fps: float,
                  ) -> list[dict[str, Any]]:
    """Per-person keyframes, in Unreal space, grouped by identity.

    Grouped per person rather than per frame because Sequencer binds transform
    tracks to actors: a frame-major layout would have to be transposed inside
    the editor, where debugging is far more painful.
    """
    by_uid: dict[str, dict[str, Any]] = {}
    for row in world_rows:
        uid = str(row.get("uid", row.get("id", "")))
        if not uid:
            continue
        entry = by_uid.setdefault(uid, {"uid": uid, "keys": []})
        z = float(row.get("altitude_m", 0.0) or 0.0)
        entry["keys"].append({
            "frame": int(row["frame"]),
            "t": round(float(row.get("t", int(row["frame"]) / max(fps, 1e-6))), 4),
            "location": to_unreal_location((row.get("x", 0.0), row.get("y", 0.0), z)),
            "yaw": round(-float(row.get("heading_deg", 0.0) or 0.0), 3),
            "posture": row.get("posture", "upright"),
            "speed": round(float(row.get("speed", 0.0) or 0.0), 3),
        })

    for entry in by_uid.values():
        entry["keys"].sort(key=lambda k: k["frame"])
    return sorted(by_uid.values(), key=lambda e: e["uid"])


def appearance_epochs(identities: Sequence[dict[str, Any]] | None,
                      world_rows: Iterable[dict[str, Any]] | None, fps: float,
                      ) -> dict[str, list[dict[str, Any]]]:
    """Per-person appearance epochs with the RENDER frame each one starts at.

    `identities` rows (gt/identities.parquet) carry one row per (uid, epoch):
    t_from/t_to, appearance_hash, face_visible and the outfit JSON. The frame
    an epoch starts is the first world.parquet frame whose `appearance_epoch`
    is that epoch (truth is what happened: world rows are the per-frame
    record); round(t_from * fps) only when the world has no such row. Only
    people with >= 2 epochs are returned; epoch 0 starts at frame 0.
    """
    rows: dict[str, list[dict[str, Any]]] = {}
    for ident in identities or []:
        uid = str(ident.get("uid", ""))
        if uid:
            rows.setdefault(uid, []).append(ident)
    rows = {u: r for u, r in rows.items() if len({int(x.get("epoch", 0) or 0) for x in r}) > 1}
    if not rows:
        return {}
    first: dict[tuple[str, int], int] = {}
    for w in world_rows or []:
        uid = str(w.get("uid", ""))
        if uid not in rows or w.get("appearance_epoch") is None:
            continue
        key = (uid, int(w["appearance_epoch"]))
        f = int(w["frame"])
        if key not in first or f < first[key]:
            first[key] = f
    out: dict[str, list[dict[str, Any]]] = {}
    for uid, rs in rows.items():
        eps = []
        for r in sorted(rs, key=lambda x: int(x.get("epoch", 0) or 0)):
            e = int(r.get("epoch", 0) or 0)
            t_from = float(r.get("t_from", 0.0) or 0.0)
            outfit = r.get("outfit")
            if isinstance(outfit, str):
                try:
                    outfit = json.loads(outfit)
                except ValueError:
                    outfit = {"raw": outfit}
            fv = r.get("face_visible")
            eps.append({
                "epoch": e,
                "t_from": round(t_from, 4),
                "t_to": round(float(r.get("t_to", 0.0) or 0.0), 4),
                "frame_start": 0 if e == 0 else first.get((uid, e), round(t_from * fps)),
                "frame_source": "world.appearance_epoch" if (uid, e) in first
                                else "identities.t_from",
                "appearance_hash": r.get("appearance_hash"),
                "face_visible": True if fv is None else bool(fv),
                "dominant_colour": r.get("dominant_colour")
                if isinstance(r.get("dominant_colour"), str) else None,
                "gt_outfit": outfit if isinstance(outfit, dict) else None,
            })
        out[uid] = eps
    return out


def build_manifest(scenario: Any, world_rows: Sequence[dict[str, Any]],
                   identities: Sequence[dict[str, Any]] | None = None,
                   ) -> dict[str, Any]:
    """Assemble everything the editor needs to rebuild and render a run."""
    fps = float(getattr(scenario, "fps", 25.0))
    people = []
    for ident in (identities or []):
        people.append({
            "uid": str(ident.get("uid", "")),
            "epoch": int(ident.get("epoch", 0) or 0),
            "wardrobe": person_wardrobe(ident),
            "gait_style": ident.get("gait_style", "normal"),
        })

    return {
        "version": MANIFEST_VERSION,
        "scenario": getattr(scenario, "name", "unnamed"),
        "seed": int(getattr(scenario, "seed", 0) or 0),
        "fps": fps,
        "duration_s": float(getattr(scenario, "duration_s", 0.0) or 0.0),
        "units": {"source": "metres, Z-up, right-handed, yaw CCW",
                  "target": "Unreal centimetres, left-handed, yaw CW",
                  "conversion": "x100, negate Y, negate yaw"},
        "cameras": [camera_entry(c) for c in getattr(scenario, "cameras", [])],
        "people": people,
        "tracks": track_entries(world_rows, fps),
        # {uid: [{epoch, frame_start, t_from, face_visible, gt_outfit, ...}]}
        # for people whose appearance changes (outfit_change, disguise): the
        # renderer switches their clothing at frame_start.
        "appearance_epochs": appearance_epochs(identities, world_rows, fps),
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return path

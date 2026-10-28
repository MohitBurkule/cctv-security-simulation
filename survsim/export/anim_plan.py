"""Pure-Python animation plan for rendered people (stdlib only).

The in-editor renderer imports this so the decisions that decide whether a
person *walks* — which clip, what play rate, which way they face — are testable
without an engine. A person bound to a transform track but no animation track
renders as a sliding T-pose, which a detector still half-recognises; that makes
the failure quiet, the same class as the missing camera cut (docs/12).

Units are the manifest's (Unreal centimetres, frames at manifest fps).
"""

from __future__ import annotations

import math
import random
from typing import Any

from .foot_slide import limit_turn_rate

WALK_CLIP_SPEED_MPS = 1.1   # CesiumMan walk cycle speed at play rate 1.0 (estimate)
IDLE_BELOW_MPS = 0.2
MIN_RATE, MAX_RATE = 0.15, 2.2   # 0.4 clamped slow walkers into sliding
RATE_STEP = 0.02            # was 0.1: up to 5 % rate error = visible slide
MIN_SEGMENT_FRAMES = 5
RUN_ABOVE_MPS = 2.3          # faster than a brisk walk: the run loop, if the character has one

# Behaviour kind (survsim.director.behaviours.REGISTRY) -> how the body moves.
#   stationary: "idle" (idle loop, random phase) | "look" (idle loop + the body
#               turning to scan, +-35 deg over ~6 s: loiter/recon read as
#               looking around) ; fast: True if its fast segments are runs.
# Every catalogue behaviour must be here (tests/test_unreal_anim_plan.py).
_LOOK = {"stationary": "look", "fast": False}
_IDLE = {"stationary": "idle", "fast": False}
_RUN = {"stationary": "idle", "fast": True}
BEHAVIOUR_ANIM: dict[str, dict] = {
    **{k: _LOOK for k in ("loiter", "loiter_intermittent", "recon", "perimeter_probe",
                          "vehicle_dropoff_loiter", "night_presence", "lost_tourist",
                          "photograph", "camera_avoidance", "follow", "following",
                          "tailgating", "tailgate", "fence_tailgate", "patrol_guard")},
    **{k: _IDLE for k in ("commute_wait", "queue", "smoke_break", "meet_and_greet",
                          "argument", "maintenance", "sit_rest", "weather_reaction",
                          "crowd_form", "object_handoff", "object_abandon", "asset_tamper",
                          "camera_tamper", "fence_cut", "vandalism", "climb_attempt",
                          "throw_object", "person_down", "outfit_change", "concealment",
                          "disabled_mobility", "drunk_walk", "child_play", "transit",
                          "group_walk", "counter_flow", "zone_intrusion", "wrong_zone_entry",
                          "erratic_motion", "fence_climb", "maintenance_visit", "staff_escort", "vehicle_move", "vehicle_ram", "drone_overflight",
                          "fight")},
    **{k: _RUN for k in ("flee", "crowd_rush", "theft_snatch", "wildlife")},
}
# Postures with no clip on the offline MetaHuman skeleton (only walk/run/idle
# loops ship): held on the idle loop. Explicit, so a new posture is a decision.
POSTURE_FALLBACK = {"crouch": "idle", "seated": "idle", "prone": "idle", "falling": "idle",
                    "climbing": "idle", "brace": "idle", "head_down": "idle",
                    "windup": "idle", "airborne": "idle", "unsteady": "walk",
                    "vehicle": "idle", "slouched": "walk", "upright": None}


def behaviour_for_label(label: str | None) -> str | None:
    """World label ("normal.transit", "suspicious.access.tailgate") -> behaviour kind."""
    if not label:
        return None
    parts = str(label).replace("-", "_").split(".")
    for part in reversed(parts):
        if part in BEHAVIOUR_ANIM:
            return part
    for kind in sorted(BEHAVIOUR_ANIM, key=len, reverse=True):
        if kind in str(label):
            return kind
    return None


def travel_speeds(keys: list[dict], fps: float) -> list[float]:
    """Speed in m/s at each key from consecutive positions (cm), not the label."""
    speeds = []
    for i, key in enumerate(keys):
        j = i + 1 if i + 1 < len(keys) else i
        k = i - 1 if j == i else i
        a, b = keys[k], keys[j]
        df = max(1, int(b["frame"]) - int(a["frame"]))
        dx = b["location"][0] - a["location"][0]
        dy = b["location"][1] - a["location"][1]
        speeds.append(math.hypot(dx, dy) / 100.0 * fps / df)
    return speeds


def travel_yaws(keys: list[dict], min_step_cm: float = 1.0) -> list[float]:
    """Unreal yaw (deg) facing the direction of travel; holds when stationary.

    Unwrapped so interpolation between keys never spins through 360.
    """
    yaws: list[float] = []
    last = float(keys[0].get("yaw", 0.0)) if keys else 0.0
    for i in range(len(keys)):
        j = min(i + 1, len(keys) - 1)
        k = j - 1 if j == i else i
        dx = keys[j]["location"][0] - keys[k]["location"][0]
        dy = keys[j]["location"][1] - keys[k]["location"][1]
        if math.hypot(dx, dy) >= min_step_cm:
            raw = math.degrees(math.atan2(dy, dx))
            raw += round((last - raw) / 360.0) * 360.0
            last = raw
        yaws.append(last)
    return yaws


def _state(speed: float, clip_speed: float = WALK_CLIP_SPEED_MPS, label: str | None = None,
           posture: str | None = None, run_speed: float | None = None) -> tuple[str, float]:
    kind = behaviour_for_label(label)
    mode = BEHAVIOUR_ANIM.get(kind, _IDLE)
    held = POSTURE_FALLBACK.get(posture or "upright")
    if speed < IDLE_BELOW_MPS or held == "idle":
        return ("look" if mode["stationary"] == "look" and held is None else "idle"), 0.0
    if run_speed and (speed > RUN_ABOVE_MPS or (mode["fast"] and speed > 0.8 * RUN_ABOVE_MPS)):
        rate = min(MAX_RATE, max(MIN_RATE, speed / run_speed))
        return "run", round(round(rate / RATE_STEP) * RATE_STEP, 2)
    rate = min(MAX_RATE, max(MIN_RATE, speed / clip_speed))
    return "walk", round(round(rate / RATE_STEP) * RATE_STEP, 2)


def anim_segments(keys: list[dict], fps: float, n_frames: int,
                  clip_speed: float = WALK_CLIP_SPEED_MPS,
                  run_speed: float | None = None) -> list[dict]:
    """Contiguous [start, end) sections covering the track: walk, run, idle or look.

    Keys may carry "label" (the world behaviour label) and "posture"; speed
    comes from positions, never from the label.
    """
    keys = [k for k in keys if int(k["frame"]) < n_frames]
    if not keys:
        return []
    speeds = travel_speeds(keys, fps)
    segs: list[dict] = []
    for key, speed in zip(keys, speeds):
        clip, rate = _state(speed, clip_speed, key.get("label"), key.get("posture"), run_speed)
        f = int(key["frame"])
        if segs and segs[-1]["clip"] == clip and segs[-1]["rate"] == rate:
            segs[-1]["end"] = f + 1
        else:
            if segs:
                segs[-1]["end"] = f
            segs.append({"clip": clip, "rate": rate, "start": f, "end": f + 1})
    # Absorb tiny segments into the previous one: rate jitter must not produce
    # a section per frame (visible pops in the cycle).
    merged: list[dict] = []
    for seg in segs:
        if merged and (seg["end"] - seg["start"] < MIN_SEGMENT_FRAMES):
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))
    merged[0]["start"] = 0
    merged[-1]["end"] = max(merged[-1]["end"], n_frames)
    # Rate from the section's actual distance / time, not the first key's
    # instantaneous speed: merged sections otherwise keep a stale rate and the
    # planted foot slides (docs/12 "Foot slide").
    fr = [int(k["frame"]) for k in keys]
    for seg in merged:
        if seg["clip"] not in ("walk", "run"):
            continue
        idx = [i for i, f in enumerate(fr) if seg["start"] <= f < seg["end"]]
        if len(idx) < 2:
            continue
        dist = sum(math.hypot(keys[i + 1]["location"][0] - keys[i]["location"][0],
                              keys[i + 1]["location"][1] - keys[i]["location"][1])
                   for i in idx[:-1]) / 100.0
        dur = (fr[idx[-1]] - fr[idx[0]]) / fps
        if dur > 0:
            ref = clip_speed if seg["clip"] == "walk" else (run_speed or clip_speed)
            rate = min(MAX_RATE, max(MIN_RATE, dist / dur / ref))
            seg["rate"] = round(round(rate / RATE_STEP) * RATE_STEP, 2)
    return merged


def clothing_colour(uid: str, seed: int = 0) -> tuple[float, float, float]:
    """Deterministic per-person tint (linear RGB), so people differ."""
    rng = random.Random(f"{seed}:{uid}")
    palette = [(0.05, 0.05, 0.06), (0.6, 0.6, 0.62), (0.08, 0.12, 0.35),
               (0.45, 0.05, 0.05), (0.1, 0.3, 0.1), (0.5, 0.4, 0.25),
               (0.9, 0.9, 0.88), (0.35, 0.2, 0.45), (0.8, 0.55, 0.1)]
    r, g, b = rng.choice(palette)
    j = rng.uniform(0.85, 1.15)
    return (min(r * j, 1.0), min(g * j, 1.0), min(b * j, 1.0))


REF_HEIGHT_M = 1.75


def person_height_m(manifest: dict[str, Any], uid: str, seed: int = 0) -> float:
    """Height from the manifest wardrobe, else a seeded draw; clamped 1.55-1.95 m."""
    for person in manifest.get("people", []):
        if person.get("uid") == uid:
            h = (person.get("wardrobe") or {}).get("height_m")
            if h:
                return min(1.95, max(1.55, float(h)))
    return round(random.Random(f"{seed}:h:{uid}").uniform(1.6, 1.9), 3)


def choose_character(uid: str, characters: dict[str, dict], seed: int = 0) -> str:
    """Deterministic per-track character choice among usable catalogue entries."""
    usable = sorted(k for k, v in characters.items()
                    if v.get("mesh") and v.get("walk") and not v.get("exclude"))
    if not usable:
        raise ValueError("no usable characters in catalogue")
    return usable[random.Random(f"{seed}:char:{uid}").randrange(len(usable))]


LOOK_AMPLITUDE_DEG = 35.0
LOOK_PERIOD_S = 6.0


def look_around(keys: list[dict], yaws: list[float], segs: list[dict], fps: float,
                phase: float = 0.0) -> list[float]:
    """Stationary "look" segments turn the body to scan: yaw += A sin(2 pi t / T + phase).

    The idle clip has no head turn and the offline rig has no control rig to
    drive the neck, so the whole body turns - which is how loitering reads on
    CCTV anyway. Ramps in and out over 1 s so walking yaw is continuous.
    """
    out = list(yaws)
    for seg in segs:
        if seg["clip"] != "look":
            continue
        dur = (seg["end"] - seg["start"]) / fps
        for i, k in enumerate(keys):
            f = int(k["frame"])
            if seg["start"] <= f < seg["end"]:
                t = (f - seg["start"]) / fps
                ramp = min(1.0, t, max(0.0, dur - t))
                out[i] = yaws[i] + ramp * LOOK_AMPLITUDE_DEG * math.sin(
                    2 * math.pi * t / LOOK_PERIOD_S + phase)
    return out


def heading(keys: list[dict], segs: list[dict], fps: float) -> list[float]:
    """Facing per key: path direction, 3-tap smoothed while moving (GT avoidance
    jitter swung the planted foot: median slide 10.7 -> 8.1 cm/s on smoke_mock);
    while stationary a heading change > 45 deg becomes a turn in place at
    <= 120 deg/s instead of a snap. Capping the turn rate while *walking* was
    measured to raise slide (the body stops facing its path), so it is not done.
    """
    raw = travel_yaws(keys)
    n = len(raw)
    sm = [(raw[max(0, i - 1)] + raw[i] + raw[min(n - 1, i + 1)]) / 3.0 for i in range(n)]
    frames = [int(k["frame"]) for k in keys]
    still = {f for sg in segs if sg["clip"] in ("idle", "look") for f in range(sg["start"], sg["end"])}
    turned = limit_turn_rate(raw, frames, fps, max_dps=120.0)
    return [turned[i] if frames[i] in still else sm[i] for i in range(n)]


def person_plan(manifest: dict[str, Any], n_frames: int,
                characters: dict[str, dict] | None = None) -> list[dict]:
    """One entry per track: yaws, animation segments, tint, character, height.

    Never drops a track. `characters` is the in-editor catalogue
    (scripts/ue/import_gltf.py); a character may carry `walk_speed_mps`, the
    speed of its walk clip at rate 1.0 for a 1.75 m person. Cadence scales with
    leg length, so a taller person covers more ground per cycle: the clip's
    ground speed is scaled by height / 1.75 before the rate is computed.
    """
    fps = float(manifest.get("fps", 25))
    seed = int(manifest.get("seed", 0))
    characters = characters or {"CesiumMan": {"mesh": "?", "walk": "?"}}
    plan = []
    for track in manifest.get("tracks", []):
        uid = track["uid"]
        keys = [k for k in track["keys"] if int(k["frame"]) < n_frames] or track["keys"][:1]
        char = track.get("character") if track.get("character") in characters \
            else choose_character(uid, characters, seed)
        height = person_height_m(manifest, uid, seed)
        clip_speed = float(characters[char].get("walk_speed_mps", WALK_CLIP_SPEED_MPS))
        clip_speed *= height / REF_HEIGHT_M
        run_speed = characters[char].get("run_speed_mps")
        run_speed = float(run_speed) * height / REF_HEIGHT_M if run_speed else None
        segs = anim_segments(keys, fps, n_frames, clip_speed, run_speed) or [
            {"clip": "idle", "rate": 0.0, "start": 0, "end": max(1, n_frames)}]
        rng = random.Random(f"{seed}:phase:{uid}")
        for seg in segs:
            # Idle phase per person and segment: a waiting group must not
            # breathe and shift weight in sync.
            if seg["clip"] in ("idle", "look"):
                seg["phase_s"] = round(rng.uniform(0.0, 10.0), 2)
        plan.append({"uid": uid, "keys": keys,
                     "yaws": look_around(keys, heading(keys, segs, fps), segs, fps,
                                         rng.uniform(0, 2 * math.pi)),
                     "segments": segs,
                     "colour": clothing_colour(uid, seed),
                     "colour_bottom": clothing_colour(uid + ":bottom", seed + 1),
                     "character": char, "height_cm": round(height * 100.0, 1),
                     # Frames outside [first, last] have no truth for this person
                     # (not yet arrived / already left): the render hides them.
                     "visible": [int(keys[0]["frame"]), int(keys[-1]["frame"])]})
    return plan


EPOCH_SEP = "#e"


def split_epochs(plan: list[dict], palettes: dict[str, list[dict]]) -> list[dict]:
    """Give each appearance epoch its own actor, shown only during that epoch.

    `palettes[uid]` is survsim.export.outfits.epoch_outfits (each with `epoch`
    and `frame_start`). A person with epochs becomes one entry per epoch that
    overlaps the planned frames: same keys, yaws and animation segments (so the
    body moves identically), its own `outfit`, and `visible` clipped to
    [frame_start, next frame_start - 1]. The epoch-0 entry keeps the uid; later
    ones are `uid#e<k>` with `person_uid` = uid. The actor swap is exact to the
    frame and needs no material track on the sequencer.
    """
    out: list[dict] = []
    for entry in plan:
        pals = sorted(palettes.get(entry["uid"]) or [], key=lambda p: p["frame_start"])
        if len(pals) < 2:
            out.append(entry)
            continue
        a, b = entry["visible"]
        made = []
        for i, pal in enumerate(pals):
            lo = max(a, int(pal["frame_start"]))
            hi = min(b, int(pals[i + 1]["frame_start"]) - 1) if i + 1 < len(pals) else b
            if lo > hi:
                continue
            e = dict(entry, outfit=pal, visible=[lo, hi], person_uid=entry["uid"],
                     epoch=int(pal["epoch"]), epoch_frame_start=int(pal["frame_start"]))
            if pal["epoch"]:
                e["uid"] = f"{entry['uid']}{EPOCH_SEP}{pal['epoch']}"
            made.append(e)
        out.extend(made or [entry])
    return out

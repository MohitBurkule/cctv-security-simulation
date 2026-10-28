"""Measure gait parameters from reference mocap, to check the argued styles.

`gait/signature.py` marks `carrying_bag` and `heavy_coat` as MEASURED (CASIA-B,
docs/14 §13) and everything else as ARGUED. This closes part of that gap using
reference mocap clips of the same rig walking normally and in each style.

**These clips are reference data, never playback assets.** Driving every
simulated limper from one recording would make them all move identically, which
is the defect `gait/synthesis.py` exists to avoid. What the clips are good for
is telling us what a limp actually *does* to cadence, step length and asymmetry
— numbers that then parameterise motion generated per person.

Same-rig comparison is the point: every clip is retargeted onto the same
skeleton, so rig differences cancel and what remains is the style. This is the
same paired design that made the CASIA-B result trustworthy.

    .venv/bin/python tools/measure_mocap_gait.py ~/Downloads

Requires `bpy` (Blender as a Python module) for FBX import. The system Blender
on this machine has a broken library link, which is why the module is used.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# Mixamo bone suffixes. Matched by suffix because the rig prefixes everything
# with "mixamorig:" and the prefix has changed between exports.
BONES = {
    "hip_L": "LeftUpLeg", "hip_R": "RightUpLeg",
    "knee_L": "LeftLeg", "knee_R": "RightLeg",
    "foot_L": "LeftFoot", "foot_R": "RightFoot",
    "root": "Hips",
}

# Which file is the baseline every other clip is compared against.
BASELINE = "walking"

# Map a clip's filename to the style it is evidence for.
CLIP_STYLES = {
    "walking": "normal",
    "injured walk": "limp_L",
    "crouched walking": "crouch",
    "drunk walk": "drunk",
}


def load_clip(path: Path) -> dict[str, Any] | None:
    """Per-frame bone positions and rotations from one FBX."""
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.import_scene.fbx(filepath=str(path))
    except Exception as exc:                                  # noqa: BLE001
        print(f"  could not import {path.name}: {exc}")
        return None

    armature = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    if armature is None:
        return None

    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        return None
    start, end = int(action.frame_range[0]), int(action.frame_range[1])

    resolved = {}
    for key, suffix in BONES.items():
        for bone in armature.pose.bones:
            if bone.name.endswith(suffix):
                resolved[key] = bone.name
                break
    if "hip_L" not in resolved or "hip_R" not in resolved:
        return None

    frames = []
    scene = bpy.context.scene
    for f in range(start, end + 1):
        scene.frame_set(f)
        row = {}
        for key, name in resolved.items():
            pose_bone = armature.pose.bones[name]
            world = armature.matrix_world @ pose_bone.matrix
            euler = pose_bone.matrix_basis.to_euler("XYZ")
            row[key] = {
                "pos": [world.translation.x, world.translation.y,
                        world.translation.z],
                "rot": [math.degrees(euler.x), math.degrees(euler.y),
                        math.degrees(euler.z)],
            }
        frames.append(row)

    return {"name": path.stem, "fps": float(scene.render.fps),
            "n_frames": len(frames), "frames": frames}


def _foot_separation(clip: dict[str, Any]) -> np.ndarray:
    """Forward distance between the feet, per frame.

    Uses foot positions where present and hips otherwise. The walking direction
    is taken as the axis with the largest variance across the clip, so this does
    not depend on which way the actor happened to face.
    """
    key = "foot_L" if "foot_L" in clip["frames"][0] else "hip_L"
    other = "foot_R" if key == "foot_L" else "hip_R"

    left = np.array([f[key]["pos"] for f in clip["frames"]])
    right = np.array([f[other]["pos"] for f in clip["frames"]])
    delta = left - right
    axis = int(np.argmax(delta.std(axis=0)))
    return delta[:, axis]


# A clip needs this many step transitions before cadence means anything. These
# are short library loops — one to three strides — and below this the estimate
# is dominated by where the loop happens to start and end.
MIN_TRANSITIONS = 4


def measure_clip(clip: dict[str, Any]) -> dict[str, Any]:
    """Cadence and timing asymmetry, or an explicit refusal.

    **Asymmetry is measured from step *timing*, not from amplitude.** The first
    attempt compared the positive and negative excursions of foot separation and
    returned exactly 0.0 for an unmistakable limp — which falsifies that metric
    rather than the clip. A limp is a difference in how long each leg bears
    weight, so the quantity to compare is the duration of alternate half-cycles.
    """
    separation = _foot_separation(clip)
    fps = clip["fps"]
    centred = separation - separation.mean()

    # Every zero crossing, in either direction: one per step.
    #
    # A sample landing exactly on zero has sign 0, and a strict sign-change test
    # then misses the crossing entirely. Zeros carry the previous sign forward,
    # so a signal that touches zero at a transition is still counted once.
    signs = np.sign(centred)
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    crossings = np.flatnonzero(signs[:-1] * signs[1:] < 0)
    n_transitions = len(crossings)

    duration_s = len(separation) / fps
    result: dict[str, Any] = {
        "clip": clip["name"],
        "style": CLIP_STYLES.get(clip["name"].lower()),
        "fps": fps, "n_frames": clip["n_frames"],
        "duration_s": round(duration_s, 3),
        "n_transitions": n_transitions,
        "step_amplitude": round(float(np.abs(centred).max()), 4),
    }

    if n_transitions < MIN_TRANSITIONS:
        result.update({
            "cadence_spm": None, "asymmetry": None, "reliable": False,
            "reason": (f"only {n_transitions} step transitions in "
                       f"{duration_s:.2f}s — a library loop this short cannot "
                       "support a cadence or asymmetry estimate, and any "
                       "number here would be an artefact of where the loop "
                       "begins"),
        })
        return result

    # Half-cycle durations, in frames, between successive crossings.
    intervals = np.diff(crossings) / fps
    cadence = 60.0 / float(np.mean(intervals))

    # Alternate intervals are the two legs. A limp makes them unequal.
    odd, even = intervals[0::2], intervals[1::2]
    n = min(len(odd), len(even))
    if n == 0:
        asymmetry = 0.0
    else:
        a, b = float(np.mean(odd[:n])), float(np.mean(even[:n]))
        asymmetry = (a - b) / (a + b) if (a + b) > 0 else 0.0

    result.update({
        "cadence_spm": round(cadence, 2),
        "asymmetry": round(asymmetry, 4),
        "reliable": True, "reason": None,
    })
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path, nargs="?",
                    default=Path.home() / "Downloads")
    ap.add_argument("--out", type=Path, default=Path("out/mocap_gait.json"))
    args = ap.parse_args()

    paths = sorted(p for p in args.directory.glob("*.fbx")
                   if p.stem.lower() in CLIP_STYLES)
    if not paths:
        print(f"no reference clips in {args.directory}; expected one of "
              f"{sorted(CLIP_STYLES)}")
        return 1

    measured = []
    for path in paths:
        print(f"reading {path.name}", flush=True)
        clip = load_clip(path)
        if clip is None:
            continue
        measured.append(measure_clip(clip))

    baseline = next((m for m in measured if m["clip"].lower() == BASELINE), None)
    report: dict[str, Any] = {"clips": measured}

    if baseline is None:
        report["note"] = ("no normal-walk baseline, so only absolute values are "
                          "reported; style effects need a paired baseline")
    else:
        effects = {}
        for m in measured:
            if m is baseline or not m["style"]:
                continue
            if not m.get("reliable") or not baseline.get("reliable"):
                effects[m["style"]] = {"refused": m.get("reason")
                                       or "baseline unreliable"}
                continue
            effects[m["style"]] = {
                "cadence_ratio": round(
                    m["cadence_spm"] / max(baseline["cadence_spm"], 1e-6), 4),
                "step_ratio": round(
                    m["step_amplitude"] / max(baseline["step_amplitude"], 1e-6), 4),
                # Absolute delta: a normal walk's asymmetry is near zero.
                "asymmetry_delta": round(
                    m["asymmetry"] - baseline["asymmetry"], 4),
            }
        report["baseline"] = baseline["clip"]
        report["effects"] = effects
        report["caveat"] = (
            "one actor per style, so these are a single sample and cannot give "
            "a spread — unlike the CASIA-B result, which is paired across 124 "
            "subjects. Treat as a sanity check on the argued values, not as a "
            "replacement for them.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

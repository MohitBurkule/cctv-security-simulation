"""Synthesise joint motion from a gait signature, so everyone walks differently.

**Why not just play mocap clips.** A downloaded "Injured Walk" is one recording
of one actor's limp, at one severity and one speed. Driving every simulated
person who has `limp_L` from that clip makes them all walk *identically* — which
is exactly the defect `gait/signature.py` was written to remove, reintroduced
one layer down. Gait re-identification evaluated on such data would be
meaningless: matching would either be perfect (everyone's clip is a fingerprint
of their style, not of them) or chance.

So clips are treated as **reference data to calibrate against**, never as
playback assets. This module generates the motion instead, from the same six
parameters that define a person's `GaitSignature`. Two people with the same
style get similar but distinct joint curves, and a person's curves are stable
across cameras and runs because their signature is.

**Representation.** Human gait is periodic, so each joint angle is a short
Fourier series over the gait cycle. This is the standard description in the
biomechanics literature, and it has three properties that matter here:

  - a handful of harmonics reproduces a real joint curve closely, so the model
    is small enough to reason about;
  - amplitude and phase are separately controllable, which is what lets a
    parameter like `asymmetry` act on one leg and not the other;
  - it is *continuous* — motion can be sampled at any frame rate or phase,
    rather than being locked to a clip's 30 fps.

The harmonic coefficients for a normal walk come from the classic normative
gait description (Winter): hip flexion swinging roughly +/-25 degrees about a
slight flexion, knee showing the characteristic double bump per cycle, ankle a
smaller dorsi/plantarflexion. They are stated here rather than buried so they
can be replaced by measurements — `tools/measure_mocap_gait.py` fits exactly
these coefficients to the reference clips.

Angles are degrees. The cycle phase runs 0..1 over one *stride* (two steps).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .signature import GaitSignature

# Joints we drive. Enough to define a recognisable walk without pretending to a
# full biomechanical model, and they map onto the bones every humanoid rig has.
JOINTS = ("hip_flex", "knee_flex", "ankle_flex", "shoulder_flex", "elbow_flex")

# Fourier coefficients for a normal walk: (mean, [(amplitude, phase)...]) per
# joint, with phase in radians of the stride cycle. Amplitudes in degrees.
#
# ARGUED from normative gait descriptions, not measured from our own data.
# `tools/measure_mocap_gait.py` refits them against reference mocap, and where
# it has done so the result is recorded in docs/14.
NORMAL_HARMONICS: dict[str, tuple[float, list[tuple[float, float]]]] = {
    # Hip: near-sinusoidal, one cycle per stride, flexed forward in swing.
    "hip_flex":      (5.0,  [(22.0, 0.0), (3.0, 1.2)]),
    # Knee: the double bump — a small flexion at loading, a large one in swing.
    "knee_flex":     (20.0, [(18.0, 2.4), (14.0, 0.5)]),
    # Ankle: smaller range, dominated by push-off.
    "ankle_flex":    (0.0,  [(8.0, 3.0), (5.0, 1.0)]),
    # Arms swing opposite the leg on the same side.
    "shoulder_flex": (0.0,  [(18.0, np.pi), (2.0, 0.4)]),
    "elbow_flex":    (18.0, [(9.0, np.pi), (2.0, 1.0)]),
}

# Which parameter scales which joint's amplitude. Stated as a table so the
# claim each parameter makes is legible.
AMPLITUDE_DRIVERS = {
    "hip_flex": "step",
    "knee_flex": "step",
    "ankle_flex": "step",
    "shoulder_flex": "arm_swing",
    "elbow_flex": "arm_swing",
}

# A stride is two steps, so the leg cycle runs at half the step rate.
STEPS_PER_STRIDE = 2

# Anatomical limits, degrees. Joints have hard stops and a Fourier series does
# not know about them: the normal-walk harmonics drove the knee to -17 degrees
# of hyperextension, which no knee does and which would have rendered as a leg
# bending backwards. Clamping here is the physical behaviour, not a fudge —
# ligaments impose exactly this.
#
# Ranges are the functional ones for locomotion, not the extremes a joint can
# reach passively.
JOINT_LIMITS = {
    "hip_flex": (-30.0, 75.0),
    "knee_flex": (-5.0, 140.0),
    "ankle_flex": (-30.0, 25.0),
    "shoulder_flex": (-60.0, 90.0),
    "elbow_flex": (0.0, 150.0),
}


@dataclass
class GaitPose:
    """Joint angles for one instant, both sides, plus root motion."""

    phase: float
    left: dict[str, float]
    right: dict[str, float]
    pelvis_sway_m: float
    pelvis_drop_deg: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": round(self.phase, 4),
            "left": {k: round(v, 3) for k, v in self.left.items()},
            "right": {k: round(v, 3) for k, v in self.right.items()},
            "pelvis_sway_m": round(self.pelvis_sway_m, 4),
            "pelvis_drop_deg": round(self.pelvis_drop_deg, 3),
        }


def _series(mean: float, harmonics: Sequence[tuple[float, float]],
            phase: float, scale: float = 1.0) -> float:
    """Evaluate a Fourier series at a phase in 0..1 of the stride."""
    total = mean
    for k, (amplitude, offset) in enumerate(harmonics, start=1):
        total += scale * amplitude * np.sin(2.0 * np.pi * k * phase + offset)
    return float(total)


def _reference_step_length_m() -> float:
    from .signature import FREE_STEP_LENGTH_M
    return FREE_STEP_LENGTH_M


# Nominal leg length for an adult of 1.70 m. Used to relate hip angle to how
# far the foot actually travels.
NOMINAL_LEG_LENGTH_M = 0.85

# Hip amplitude and step length are related by a sine, not a ratio, so scaling
# the amplitude by "requested / reference" does NOT produce the requested step.
# Measured directly: at nominal the synthesised step came out 0.65 m against a
# requested 0.75 m — a 15% shortfall that would have had every rendered person
# walking a different stride from the one ground truth recorded, while every
# joint curve still looked correct.
#
# Solved numerically at import rather than hard-coded, so it stays right if the
# harmonics are refitted from mocap.
def _solve_hip_calibration() -> float:
    target = _reference_step_length_m()
    amplitude = NORMAL_HARMONICS["hip_flex"][1][0][0]

    def step_for(scale: float) -> float:
        # Peak foot separation over the cycle, for a symmetric nominal walk.
        best = 0.0
        for i in range(180):
            phase = i / 180.0
            a = np.radians(_series(NORMAL_HARMONICS["hip_flex"][0],
                                   [(amplitude * scale, NORMAL_HARMONICS["hip_flex"][1][0][1]),
                                    NORMAL_HARMONICS["hip_flex"][1][1]], phase))
            b = np.radians(_series(NORMAL_HARMONICS["hip_flex"][0],
                                   [(amplitude * scale, NORMAL_HARMONICS["hip_flex"][1][0][1]),
                                    NORMAL_HARMONICS["hip_flex"][1][1]], phase + 0.5))
            best = max(best, abs(NOMINAL_LEG_LENGTH_M * (np.sin(a) - np.sin(b))))
        return best

    low, high = 0.1, 5.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if step_for(mid) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


HIP_CALIBRATION = _solve_hip_calibration()


def pose_at(signature: GaitSignature, phase: float) -> GaitPose:
    """Joint angles at a given point in the stride cycle.

    The two legs run half a cycle apart, which is what makes it a walk rather
    than a hop. `asymmetry` breaks that symmetry: a positive value lengthens the
    left leg's contribution and shortens the right, which is what a limp does —
    the painful side spends less time bearing weight.
    """
    phase = float(phase) % 1.0

    # Joint amplitudes scale with how far the person is actually stepping,
    # relative to a normal walk. Someone shuffling moves their hips less.
    step_scale = (signature.step_length_m / max(_reference_step_length_m(), 1e-6)
                  * HIP_CALIBRATION)
    scales = {"step": step_scale, "arm_swing": signature.arm_swing}

    asym = float(np.clip(signature.asymmetry, -0.9, 0.9))

    def side(side_phase: float, gain: float) -> dict[str, float]:
        out = {}
        for joint in JOINTS:
            mean, harmonics = NORMAL_HARMONICS[joint]
            driver = AMPLITUDE_DRIVERS[joint]
            angle = _series(mean, harmonics, side_phase,
                            scale=scales[driver] * gain)
            low, high = JOINT_LIMITS[joint]
            out[joint] = float(np.clip(angle, low, high))
        return out

    # +ve asymmetry favours the left leg: it swings further, the right less.
    left = side(phase, 1.0 + asym)
    right = side(phase + 0.5, 1.0 - asym)

    # Pelvis: lateral sway at one cycle per stride, and a contralateral drop
    # (Trendelenburg) at twice that — the drop is what makes a limp visible
    # from the front, where leg motion is foreshortened.
    sway = signature.sway_amplitude_m * np.sin(2.0 * np.pi * phase)
    drop = 4.0 * (1.0 + abs(asym) * 2.0) * np.sin(4.0 * np.pi * phase)

    return GaitPose(phase=phase, left=left, right=right,
                    pelvis_sway_m=float(sway), pelvis_drop_deg=float(drop))


def synthesise(signature: GaitSignature, duration_s: float, fps: float = 30.0,
               phase_offset: float = 0.0) -> list[GaitPose]:
    """A full animation for one person, sampled at `fps`.

    `phase_offset` decorrelates people: without it, a crowd spawned at the same
    instant would step in perfect unison, which reads as obviously synthetic and
    would also give a motion-energy detector a periodicity that no real crowd
    has.
    """
    if duration_s <= 0 or fps <= 0:
        return []

    # Cadence is steps per minute; a stride is two steps.
    strides_per_second = signature.cadence_spm / 60.0 / STEPS_PER_STRIDE
    n_frames = round(duration_s * fps)
    return [pose_at(signature, phase_offset + i / fps * strides_per_second)
            for i in range(n_frames)]


def stride_duration_s(signature: GaitSignature) -> float:
    """Seconds for one full stride — two steps."""
    steps_per_second = signature.cadence_spm / 60.0
    return STEPS_PER_STRIDE / max(steps_per_second, 1e-6)


def foot_positions(signature: GaitSignature, phase: float,
                   leg_length_m: float = 0.85) -> tuple[float, float]:
    """Forward position of each foot relative to the pelvis, in metres.

    A crude forward-kinematic projection from hip angle. Enough to check that
    the synthesised motion produces a plausible step length — the property most
    likely to drift when parameters are changed, and one that a joint-angle
    curve alone does not make obvious.
    """
    pose = pose_at(signature, phase)
    left = leg_length_m * np.sin(np.radians(pose.left["hip_flex"]))
    right = leg_length_m * np.sin(np.radians(pose.right["hip_flex"]))
    return float(left), float(right)


def measured_step_length_m(signature: GaitSignature,
                           leg_length_m: float = 0.85,
                           samples: int = 200) -> float:
    """Step length implied by the synthesised motion.

    Reports what the animation *does*, which is not automatically what the
    signature asked for. If these disagree, the joint amplitudes and the
    signature have drifted apart and the rendered person walks at a different
    stride from the one ground truth records.
    """
    spreads = []
    for i in range(samples):
        left, right = foot_positions(signature, i / samples, leg_length_m)
        spreads.append(abs(left - right))
    return float(max(spreads))

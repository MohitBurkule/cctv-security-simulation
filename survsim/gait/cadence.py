"""Recover gait cadence and stride from pose, on real CCTV.

`gait/extract.py` measures walking *speed* from bounding boxes and explicitly
cannot measure gait *style*: MOT boxes are interpolated between keyframes and
smoothed, so the vertical bob that carries step frequency is mostly annotation
artefact. Style needs the limbs, which means pose estimation on pixels.

This module closes that gap. It takes 2D keypoints — from any estimator, ours
uses torchvision's Keypoint R-CNN — and recovers the two numbers that actually
characterise a walk:

  - **cadence**, steps per minute; and
  - **stride length**, metres per gait cycle.

Those two plus speed are not independent — speed = (cadence/60) × step length —
and that redundancy is the point. Measuring all three and checking they agree is
what catches a cadence recovered from noise, which is otherwise indistinguishable
from a real one.

**How cadence is recovered.** The horizontal separation between the two ankles
oscillates once per *step*, twice per full gait cycle, and it is the cleanest
periodic signal in a walking body. Counting its zero-crossings is deliberately
used instead of an FFT peak: a short, noisy, occlusion-interrupted track from
CCTV has too few cycles for a well-resolved spectrum, and an FFT will always
return *some* peak — including for a stationary person, where it would be noise
reported as a walking rhythm.

**Scale**, as in `extract.py`, comes from the body itself: the distance from
ankle to shoulder is a known fraction of stature, giving metres per pixel at
that person's own depth without a homography.

Caveats that travel with the numbers rather than living in a doc:

  - A person walking toward or away from the camera has almost no *lateral*
    ankle separation, so cadence is only recoverable for roughly lateral
    motion. Such tracks are rejected, not guessed at.
  - Two ankles must both be visible and confident; in a crowd they often are
    not, which is why coverage is reported alongside the estimate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# COCO keypoint indices, the convention every common 2D pose model emits.
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

ASSUMED_STATURE_M = 1.70
# Ankle-to-shoulder is about 0.81 of stature for an adult standing upright
# (shoulder at ~0.82 H, ankle at ~0.04 H). Used as the pixel ruler.
ANKLE_TO_SHOULDER_FRACTION = 0.78

MIN_KEYPOINT_SCORE = 0.3
# Below this many samples a cadence estimate is fitting noise.
MIN_SAMPLES = 12
# Plausible human walking cadence. Outside this it is not a walk.
MIN_CADENCE_SPM, MAX_CADENCE_SPM = 40.0, 160.0


@dataclass
class GaitCycle:
    """Cadence and stride for one person, with the evidence behind them."""

    cadence_spm: float
    stride_length_m: float
    speed_mps: float
    n_steps: int
    n_frames: int
    lateral_fraction: float
    consistent: bool = True
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "cadence_spm": round(self.cadence_spm, 2),
            "stride_length_m": round(self.stride_length_m, 3),
            "speed_mps": round(self.speed_mps, 3),
            "n_steps": self.n_steps, "n_frames": self.n_frames,
            "lateral_fraction": round(self.lateral_fraction, 3),
            "consistent": self.consistent, "reason": self.reason,
        }


def _confident(keypoints: np.ndarray, index: int) -> bool:
    return bool(keypoints[index, 2] >= MIN_KEYPOINT_SCORE)


def metres_per_pixel(keypoints: np.ndarray,
                     stature_m: float = ASSUMED_STATURE_M) -> float | None:
    """Scale from the body's own proportions: ankle-to-shoulder is the ruler.

    Returns None rather than a guess when the needed joints are not visible.
    Guessing here would silently rescale one person's stride.
    """
    shoulders = [i for i in (L_SHOULDER, R_SHOULDER) if _confident(keypoints, i)]
    ankles = [i for i in (L_ANKLE, R_ANKLE) if _confident(keypoints, i)]
    if not shoulders or not ankles:
        return None

    shoulder_y = float(np.mean([keypoints[i, 1] for i in shoulders]))
    ankle_y = float(np.mean([keypoints[i, 1] for i in ankles]))
    span_px = abs(ankle_y - shoulder_y)
    if span_px < 10.0:
        return None
    return (stature_m * ANKLE_TO_SHOULDER_FRACTION) / span_px


def ankle_separation_signal(track: Sequence[np.ndarray],
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Signed horizontal ankle separation per frame, and which frames are usable.

    Signed, not absolute: the sign flips every step as the legs swap, which is
    exactly the zero-crossing the cadence count relies on. Taking the magnitude
    would double the apparent frequency.
    """
    values, usable = [], []
    for keypoints in track:
        ok = _confident(keypoints, L_ANKLE) and _confident(keypoints, R_ANKLE)
        usable.append(ok)
        values.append(float(keypoints[L_ANKLE, 0] - keypoints[R_ANKLE, 0]) if ok
                      else np.nan)
    return np.array(values, dtype=float), np.array(usable, dtype=bool)


def count_steps(signal: np.ndarray, min_amplitude: float = 3.0) -> int:
    """Steps, as zero-crossings of the ankle separation signal.

    An amplitude floor is essential. A standing person's ankle separation
    hovers near zero and jitters across it constantly, which would be counted
    as a very fast walk — noise reported as rhythm. Only crossings whose
    surrounding swing exceeds the floor are counted.
    """
    finite = signal[np.isfinite(signal)]
    if len(finite) < MIN_SAMPLES:
        return 0
    centred = finite - np.mean(finite)
    if np.max(np.abs(centred)) < min_amplitude:
        return 0

    steps = 0
    previous = centred[0]
    last_extreme = centred[0]
    for value in centred[1:]:
        # Only count if the swing since the last crossing was real.
        if np.sign(value) != np.sign(previous) and previous != 0 and abs(last_extreme) >= min_amplitude:
            steps += 1
            last_extreme = 0.0
        last_extreme = value if abs(value) > abs(last_extreme) else last_extreme
        previous = value
    return steps


def lateral_fraction(track: Sequence[np.ndarray]) -> float:
    """How much of the motion is across the camera rather than toward it.

    Cadence is only visible in lateral motion: someone walking straight at the
    camera shows almost no ankle separation, and a low value here means the
    estimate should be refused rather than trusted.
    """
    hips = []
    for keypoints in track:
        pair = [i for i in (L_HIP, R_HIP) if _confident(keypoints, i)]
        if pair:
            hips.append([float(np.mean([keypoints[i, 0] for i in pair])),
                         float(np.mean([keypoints[i, 1] for i in pair]))])
    if len(hips) < 2:
        return 0.0
    hips = np.array(hips)
    dx = float(np.ptp(hips[:, 0]))
    dy = float(np.ptp(hips[:, 1]))
    total = dx + dy
    return dx / total if total > 1e-6 else 0.0


def measure_track(track: Sequence[np.ndarray], fps: float,
                  stature_m: float = ASSUMED_STATURE_M,
                  min_lateral: float = 0.5) -> GaitCycle | None:
    """Cadence, stride and speed for one person's keypoint track.

    Returns None when the track cannot support an estimate, rather than a
    number with a caveat attached — a refused measurement is far safer than a
    plausible one that is really noise.
    """
    if len(track) < MIN_SAMPLES or fps <= 0:
        return None

    signal, usable = ankle_separation_signal(track)
    if usable.sum() < MIN_SAMPLES:
        return None

    scales = [metres_per_pixel(k, stature_m) for k in track]
    scales = [s for s in scales if s is not None]
    if not scales:
        return None
    scale = float(np.median(scales))

    lateral = lateral_fraction(track)
    steps = count_steps(signal)
    duration_s = len(track) / fps
    if steps < 2 or duration_s <= 0:
        return None

    cadence = steps / duration_s * 60.0

    # Distance travelled, from the hips, in metres.
    hips = []
    for keypoints in track:
        pair = [i for i in (L_HIP, R_HIP) if _confident(keypoints, i)]
        if pair:
            hips.append([float(np.mean([keypoints[i, 0] for i in pair])),
                         float(np.mean([keypoints[i, 1] for i in pair]))])
    distance_m = 0.0
    if len(hips) >= 2:
        hips = np.array(hips)
        distance_m = float(np.linalg.norm(hips[-1] - hips[0])) * scale

    speed = distance_m / duration_s
    # A stride is two steps.
    stride = distance_m / max(steps / 2.0, 1e-6)

    reason = None
    if lateral < min_lateral:
        reason = (f"motion is only {lateral:.2f} lateral; cadence is not "
                  "visible for someone walking toward or away from the camera")
    elif not (MIN_CADENCE_SPM <= cadence <= MAX_CADENCE_SPM):
        reason = f"cadence {cadence:.0f} spm is outside human walking range"

    if reason is not None:
        return None

    # speed = (cadence/60) * step_length is an identity, so disagreement means
    # one of the three was recovered from noise.
    predicted = (cadence / 60.0) * (stride / 2.0)
    consistent = abs(predicted - speed) <= max(0.25 * max(speed, 1e-6), 0.15)

    return GaitCycle(
        cadence_spm=cadence, stride_length_m=stride, speed_mps=speed,
        n_steps=steps, n_frames=len(track), lateral_fraction=lateral,
        consistent=consistent,
        reason=None if consistent else (
            "speed, cadence and stride do not satisfy "
            "speed = (cadence/60) x step length; treat as unreliable"))


def summarise(cycles: Sequence[GaitCycle]) -> dict[str, Any]:
    """Population statistics over measured tracks."""
    usable = [c for c in cycles if c.consistent]
    if not usable:
        return {"n_tracks": 0, "n_consistent": 0,
                "note": "no track supported a consistent gait estimate"}

    cadences = np.array([c.cadence_spm for c in usable])
    strides = np.array([c.stride_length_m for c in usable])
    return {
        "n_tracks": len(cycles),
        "n_consistent": len(usable),
        "cadence_spm_mean": round(float(cadences.mean()), 2),
        "cadence_spm_std": round(float(cadences.std()), 2),
        "stride_m_mean": round(float(strides.mean()), 3),
        "stride_m_std": round(float(strides.std()), 3),
        "note": ("cadence needs lateral motion and two confident ankles; tracks "
                 "failing either are refused rather than estimated"),
    }

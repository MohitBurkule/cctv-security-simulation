"""Acoustic sensing: event sound levels, direction of arrival, and nuisance.

Audio is the classic corroborating sensor — a raised voice or breaking glass
carries information a camera cannot see, and microphones are cheap. It is also
the classic source of embarrassing false alarms, because the world is full of
sounds that resemble the ones you care about.

Modelled here:

* **Inverse-square spreading** plus atmospheric absorption, so a sound 40 m away
  is ~32 dB weaker than at 1 m and frequently below the ambient floor.
* **Ambient noise** that rises with crowd density and weather — the reason a
  detector tuned on a quiet test site fails at a busy one.
* **Direction of arrival** with error that grows as SNR falls, which is what makes
  slewing a PTZ to an acoustic cue unreliable at range.
* **Nuisance events** at a configurable rate, because a fusion result computed
  without them is meaningless.

Legal note carried from the research: recording audio in public spaces is far more
tightly restricted than video in the UK and EU, so an audio channel that helps
technically may still be unusable in deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

# Typical source levels at 1 m, in dB SPL.
SOURCE_LEVELS_DB = {
    "footstep": 45.0,
    "conversation": 60.0,
    "raised_voice": 75.0,
    "shout": 88.0,
    "glass_break": 95.0,
    "gunshot": 140.0,
    "door_slam": 85.0,
    "alarm": 100.0,
    "vehicle": 80.0,
}

# Sounds that a naive classifier confuses with the ones above. Each is a real
# reported source of false alarms in deployed acoustic systems.
NUISANCE_SOURCES = {
    "glass_break": ("dropped bottle", "keys on tile", "till drawer"),
    "gunshot": ("firework", "car backfire", "skip delivery", "door slam"),
    "shout": ("laughter", "child playing", "football chant"),
    "alarm": ("reversing beeper", "phone ringtone"),
}

AMBIENT_QUIET_DB = 35.0
ATMOSPHERIC_ABSORPTION_DB_PER_M = 0.05


@dataclass
class Microphone:
    id: str
    pos: tuple[float, float, float]
    noise_floor_db: float = AMBIENT_QUIET_DB
    # A single mic cannot do DOA; an array can, with error that grows as SNR drops.
    array: bool = True
    doa_sigma_deg_at_20db: float = 3.0


@dataclass
class AudioConfig:
    ambient_db: float = AMBIENT_QUIET_DB
    crowd_ambient_gain_db: float = 6.0     # per doubling of crowd size
    weather_ambient_db: float = 0.0
    detection_threshold_snr_db: float = 6.0
    nuisance_rate_per_hour: float = 4.0


def spl_at(distance_m: float, source_db: float) -> float:
    """Sound pressure level at a distance: inverse-square plus absorption."""
    d = max(float(distance_m), 1.0)
    return float(source_db - 20.0 * np.log10(d) - ATMOSPHERIC_ABSORPTION_DB_PER_M * d)


def ambient_level(cfg: AudioConfig, crowd_size: int = 0) -> float:
    """Background level, rising with crowd size and weather."""
    crowd_term = cfg.crowd_ambient_gain_db * np.log2(max(crowd_size, 1))
    return float(cfg.ambient_db + crowd_term + cfg.weather_ambient_db)


def doa_error_deg(snr_db: float, mic: Microphone, rng: np.random.Generator) -> float:
    """Bearing error grows sharply as the event approaches the noise floor."""
    scale = mic.doa_sigma_deg_at_20db * max(1.0, 20.0 / max(snr_db, 1.0))
    return float(rng.normal(0.0, scale))


class AudioSimulator:
    def __init__(self, microphones: list[Microphone], cfg: AudioConfig | None = None,
                 seed: int = 0) -> None:
        self.microphones = microphones
        self.cfg = cfg or AudioConfig()
        self.rng = np.random.default_rng(seed)
        self.detections: list[dict[str, Any]] = []
        self.truth: list[dict[str, Any]] = []

    def emit(self, t: float, kind: str, pos: np.ndarray, uid: str | None = None,
             crowd_size: int = 0) -> list[dict[str, Any]]:
        """A real sound happened; return what each microphone heard."""
        source_db = SOURCE_LEVELS_DB.get(kind, 70.0)
        self.truth.append({"t": round(t, 3), "kind": kind, "uid": uid,
                           "pos": [round(float(v), 2) for v in np.asarray(pos)[:2]]})
        return self._observe(t, kind, pos, source_db, crowd_size, is_nuisance=False, uid=uid)

    def emit_nuisance(self, t: float, confusable_with: str, pos: np.ndarray,
                      crowd_size: int = 0) -> list[dict[str, Any]]:
        """Something harmless that a classifier may report as `confusable_with`."""
        label = str(self.rng.choice(NUISANCE_SOURCES.get(confusable_with, ("unknown",))))
        source_db = SOURCE_LEVELS_DB.get(confusable_with, 70.0) - float(self.rng.uniform(0, 12))
        return self._observe(t, confusable_with, pos, source_db, crowd_size,
                             is_nuisance=True, uid=None, true_source=label)

    def _observe(self, t, kind, pos, source_db, crowd_size, is_nuisance, uid,
                 true_source=None) -> list[dict[str, Any]]:
        ambient = ambient_level(self.cfg, crowd_size)
        out: list[dict[str, Any]] = []
        pos = np.asarray(pos, dtype=float)
        for mic in self.microphones:
            mic_pos = np.asarray(mic.pos, dtype=float)[:len(pos)]
            distance = float(np.linalg.norm(mic_pos - pos))
            level = spl_at(distance, source_db)
            floor = max(ambient, mic.noise_floor_db)
            snr = level - floor
            if snr < self.cfg.detection_threshold_snr_db:
                continue

            true_bearing = float(np.degrees(np.arctan2(pos[1] - mic_pos[1],
                                                       pos[0] - mic_pos[0])))
            bearing = true_bearing + (doa_error_deg(snr, mic, self.rng) if mic.array else 0.0)
            detection = {
                "t": round(t, 3), "mic": mic.id, "kind": kind,
                "spl_db": round(level, 1), "snr_db": round(snr, 1),
                "doa_deg": round(bearing, 1) if mic.array else None,
                "true_doa_deg": round(true_bearing, 1),
                "distance_m": round(distance, 2),
                "is_nuisance": is_nuisance,
                "true_source": true_source or kind,
                "uid_truth": uid,
            }
            out.append(detection)
            self.detections.append(detection)
        return out

    def step_nuisance(self, t: float, dt: float, area_centre: np.ndarray,
                      radius_m: float = 30.0, crowd_size: int = 0) -> list[dict[str, Any]]:
        """Poisson-sample background nuisance events."""
        expected = self.cfg.nuisance_rate_per_hour * dt / 3600.0
        out: list[dict[str, Any]] = []
        for _ in range(int(self.rng.poisson(expected))):
            kind = str(self.rng.choice(list(NUISANCE_SOURCES)))
            offset = self.rng.uniform(-radius_m, radius_m, size=2)
            out += self.emit_nuisance(t, kind, np.asarray(area_centre, dtype=float)[:2] + offset,
                                      crowd_size=crowd_size)
        return out

    # ------------------------------------------------------------- analysis

    def precision_if_all_detections_alarmed(self) -> float:
        """Precision of a system that trusts every acoustic detection."""
        if not self.detections:
            return 0.0
        real = sum(not d["is_nuisance"] for d in self.detections)
        return round(real / len(self.detections), 4)

    def doa_error_stats(self) -> dict[str, float]:
        bearings = [(d["doa_deg"], d["true_doa_deg"]) for d in self.detections
                    if d["doa_deg"] is not None]
        if not bearings:
            return {"mean_abs_deg": 0.0, "n": 0}
        errors = np.array([abs(((b - t) + 180) % 360 - 180) for b, t in bearings])
        return {
            "mean_abs_deg": round(float(errors.mean()), 2),
            "p90_abs_deg": round(float(np.percentile(errors, 90)), 2),
            "n": int(len(errors)),
        }

"""RF sensing of people: Wi-Fi probe requests and BLE advertisements.

Phones announce themselves constantly, which makes RF an attractive corroborating
sensor for perimeter and crowd work — it sees through walls, works in darkness
and costs nothing to deploy next to a camera.

Two effects dominate whether it is actually useful, and both are modelled here
because ignoring either produces wildly optimistic results:

1. **MAC randomisation.** Since roughly iOS 8 / Android 8, handsets rotate a
   random MAC in probe requests. A naive unique-MAC count therefore *over-counts*
   people badly — reported ratios cluster around an order of magnitude (R05). A
   pipeline that assumes one MAC equals one person is measuring its own rotation
   period.
2. **Propagation.** RSSI follows a log-distance path loss with log-normal
   shadowing, so distance estimates from signal strength carry several metres of
   error even in the open, and far more through walls.

Ground truth records the true MAC-to-identity mapping, which real deployments
never have and which is exactly what makes association research possible here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

# Free-space-ish reference: received power at 1 m, and the exponent describing how
# fast it decays. n=2 is free space; 2.7-3.5 is typical outdoors with clutter.
RSSI_AT_1M_DBM = -40.0
PATH_LOSS_EXPONENT = 2.8
SHADOWING_SIGMA_DB = 4.0
NOISE_FLOOR_DBM = -95.0


@dataclass
class Emitter:
    """One radio on one person."""

    uid: str
    proto: str = "wifi_probe"
    interval_s: float = 25.0
    tx_power_dbm: float = 15.0
    randomise_mac: bool = True
    rotate_every_s: float = 300.0
    _mac: str = ""
    _next_rotate_s: float = 0.0

    def mac_at(self, t: float, rng: np.random.Generator) -> str:
        """Current MAC, rotating on schedule when randomisation is on."""
        if not self.randomise_mac:
            if not self._mac:
                self._mac = _stable_mac(self.uid, 0)
            return self._mac
        if not self._mac or t >= self._next_rotate_s:
            epoch = int(t // max(self.rotate_every_s, 1e-6))
            self._mac = _stable_mac(self.uid, epoch, locally_administered=True)
            self._next_rotate_s = (epoch + 1) * self.rotate_every_s
        return self._mac


def _stable_mac(uid: str, epoch: int, locally_administered: bool = False) -> str:
    import hashlib

    digest = hashlib.blake2b(f"{uid}:{epoch}".encode(), digest_size=6).digest()
    octets = list(digest)
    if locally_administered:
        # Randomised MACs set the locally-administered bit and clear multicast —
        # which is itself a detectable signature that the address is not real.
        octets[0] = (octets[0] | 0x02) & 0xFE
    return ":".join(f"{b:02x}" for b in octets)


@dataclass
class Sniffer:
    """A passive receiver at a fixed position."""

    id: str
    pos: tuple[float, float, float]
    sensitivity_dbm: float = -92.0
    wall_attenuation_db: float = 0.0


@dataclass
class RfConfig:
    path_loss_exponent: float = PATH_LOSS_EXPONENT
    shadowing_sigma_db: float = SHADOWING_SIGMA_DB
    rssi_at_1m_dbm: float = RSSI_AT_1M_DBM
    detection_jitter: float = 0.15          # chance a due emission is simply missed


def rssi_at(distance_m: float, tx_power_dbm: float, cfg: RfConfig,
            rng: np.random.Generator, extra_loss_db: float = 0.0) -> float:
    """Log-distance path loss plus log-normal shadowing."""
    d = max(float(distance_m), 0.5)
    loss = 10.0 * cfg.path_loss_exponent * np.log10(d)
    shadow = rng.normal(0.0, cfg.shadowing_sigma_db)
    return float(cfg.rssi_at_1m_dbm + (tx_power_dbm - 15.0) - loss - extra_loss_db + shadow)


def distance_from_rssi(rssi_dbm: float, cfg: RfConfig, tx_power_dbm: float = 15.0) -> float:
    """Invert the path-loss model. Deliberately exposed so its error is visible."""
    exponent = (cfg.rssi_at_1m_dbm + (tx_power_dbm - 15.0) - rssi_dbm) / (
        10.0 * cfg.path_loss_exponent)
    return float(10.0**exponent)


class RfSimulator:
    """Generates RF detections for a population of emitters."""

    def __init__(self, sniffers: list[Sniffer], emitters: list[Emitter],
                 cfg: RfConfig | None = None, seed: int = 0) -> None:
        self.sniffers = sniffers
        self.emitters = {e.uid: e for e in emitters}
        self.cfg = cfg or RfConfig()
        self.rng = np.random.default_rng(seed)
        self._next_emit: dict[str, float] = {e.uid: 0.0 for e in emitters}
        self.detections: list[dict[str, Any]] = []
        self.truth: list[dict[str, Any]] = []

    def step(self, t: float, positions: dict[str, np.ndarray]) -> list[dict[str, Any]]:
        """Emit for anyone whose interval has elapsed; return this tick's detections."""
        out: list[dict[str, Any]] = []
        for uid, emitter in self.emitters.items():
            if uid not in positions or t < self._next_emit.get(uid, 0.0):
                continue
            # Real probe intervals are irregular, driven by screen state and scans.
            self._next_emit[uid] = t + float(self.rng.uniform(0.5, 1.5)) * emitter.interval_s
            mac = emitter.mac_at(t, self.rng)
            self.truth.append({"t": round(t, 3), "uid": uid, "mac": mac, "proto": emitter.proto})

            pos = np.asarray(positions[uid], dtype=float)
            for sniffer in self.sniffers:
                if self.rng.random() < self.cfg.detection_jitter:
                    continue
                distance = float(np.linalg.norm(np.asarray(sniffer.pos, dtype=float)[:len(pos)] - pos))
                rssi = rssi_at(distance, emitter.tx_power_dbm, self.cfg, self.rng,
                               sniffer.wall_attenuation_db)
                if rssi < sniffer.sensitivity_dbm:
                    continue
                detection = {
                    "t": round(t, 3), "sniffer": sniffer.id, "mac": mac,
                    "proto": emitter.proto, "rssi_dbm": round(rssi, 1),
                    "est_distance_m": round(distance_from_rssi(rssi, self.cfg,
                                                               emitter.tx_power_dbm), 2),
                    "true_distance_m": round(distance, 2),
                    "uid_truth": uid,
                }
                out.append(detection)
                self.detections.append(detection)
        return out

    # ------------------------------------------------------------- analysis

    def naive_device_count(self) -> int:
        """Unique MACs — what a deployed counter without de-randomisation reports."""
        return len({d["mac"] for d in self.detections})

    def true_person_count(self) -> int:
        return len({d["uid_truth"] for d in self.detections})

    def overcount_ratio(self) -> float:
        """How badly MAC randomisation inflates a naive headcount."""
        true_count = self.true_person_count()
        return self.naive_device_count() / true_count if true_count else 0.0

    def localisation_error(self) -> dict[str, float]:
        """Error in range estimated from RSSI — the reason RF alone cannot localise."""
        if not self.detections:
            return {"mean_abs_error_m": 0.0, "median_abs_error_m": 0.0, "n": 0}
        errors = np.array([abs(d["est_distance_m"] - d["true_distance_m"])
                           for d in self.detections])
        return {
            "mean_abs_error_m": round(float(errors.mean()), 2),
            "median_abs_error_m": round(float(np.median(errors)), 2),
            "p90_abs_error_m": round(float(np.percentile(errors, 90)), 2),
            "n": int(len(errors)),
        }

"""Thermal (LWIR) imaging.

Thermal is a genuinely different sensor from a night-mode camera, and conflating
them is a common and costly mistake. A night camera lifts its IR-cut filter and
lights the scene with onboard 850 nm LEDs, so it sees only as far as the
illuminator reaches — perhaps 30 m, and nothing beyond. A thermal camera detects
emitted long-wave radiation, needs no illumination at all, and sees a person at
several hundred metres in total darkness. That is why perimeters use thermal and
car parks use IR.

What it images is **temperature and emissivity**, not reflected light, which
inverts several intuitions:

* Clothing is close to ambient on the outside, so a heavily wrapped person is
  much dimmer than a bare face or hands. Thermal detection of people is really
  detection of exposed skin and thin fabric.
* Camouflage that defeats a colour camera does nothing, but a thermal blanket
  defeats thermal entirely while being obvious in the visible band.
* **Thermal crossover**: twice a day, as surfaces pass through air temperature,
  scene contrast collapses and detection ranges fall sharply. A system benchmarked
  at noon does not describe dawn.

Resolution is the other hard constraint: uncooled microbolometers are typically
320x256 or 640x512, so a person at 200 m is a handful of pixels regardless of how
hot they are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

# Apparent temperatures in Celsius. Skin is the reliable signal; clothing is
# barely above ambient, which is why a wrapped-up person nearly disappears.
SURFACE_TEMPS: dict[str, float] = {
    "skin": 33.0,
    "clothing_light": 26.0,
    "clothing_heavy": 21.0,
    "hair": 30.0,
    "vehicle_engine": 65.0,
    "vehicle_body": 24.0,
    "tarmac_sunlit": 45.0,
    "tarmac_shaded": 18.0,
    "vegetation": 19.0,
    "building": 17.0,
    "sky": -20.0,
    "water": 12.0,
}

# Emissivity: how efficiently a surface radiates. Metal and glass are low, which
# makes them look misleadingly cold and is a classic source of thermal confusion.
EMISSIVITY: dict[str, float] = {
    "skin": 0.98, "clothing_light": 0.95, "clothing_heavy": 0.95, "hair": 0.95,
    "vehicle_engine": 0.90, "vehicle_body": 0.60, "tarmac_sunlit": 0.95,
    "tarmac_shaded": 0.95, "vegetation": 0.98, "building": 0.92,
    "sky": 0.90, "water": 0.96,
}

PALETTES = ("white_hot", "black_hot", "ironbow")


# Johnson criteria: pixels across a target needed for each task. These, not
# thermal contrast, are what limit a thermal camera at long range.
JOHNSON_PIXELS = {"detect": 1.5, "recognise": 6.0, "identify": 12.0}


@dataclass
class ThermalConfig:
    """Sensor and scene parameters for a LWIR camera."""

    resolution: tuple[int, int] = (640, 512)
    hfov_deg: float = 24.0
    ambient_c: float = 15.0
    # Noise-equivalent temperature difference: the smallest temperature step the
    # sensor can resolve. 50 mK is a good uncooled core; 100 mK is a cheap one.
    netd_mk: float = 50.0
    palette: str = "white_hot"
    # Span of the display mapping. Narrow span gives punchy contrast but clips.
    span_c: tuple[float, float] = (10.0, 40.0)
    atmospheric_transmission_per_m: float = 0.0008
    fixed_pattern_mk: float = 20.0
    dead_pixel_rate: float = 1e-4


def apparent_temperature(surface: str, config: ThermalConfig,
                         distance_m: float = 0.0, sun_factor: float = 0.0) -> float:
    """Temperature a camera would report for a surface.

    Combines the surface's own temperature with its emissivity — a low-emissivity
    surface reflects ambient and therefore *reads* closer to ambient than it is —
    and then attenuates the difference over distance through the atmosphere.
    """
    true_temp = SURFACE_TEMPS.get(surface, config.ambient_c)
    if surface.startswith("tarmac") or surface == "building":
        true_temp += sun_factor * 12.0     # solar loading on hard surfaces

    emissivity = EMISSIVITY.get(surface, 0.95)
    apparent = emissivity * true_temp + (1.0 - emissivity) * config.ambient_c

    # Atmosphere pulls everything toward ambient with range — the reason a warm
    # target at 400 m has far less contrast than the same target at 40 m.
    attenuation = np.exp(-config.atmospheric_transmission_per_m * max(distance_m, 0.0))
    return float(config.ambient_c + (apparent - config.ambient_c) * attenuation)


def thermal_contrast(config: ThermalConfig, distance_m: float = 0.0,
                     clothing: str = "clothing_light", sun_factor: float = 0.0) -> float:
    """Temperature difference between a person and their background, in kelvin.

    This is the number that decides whether a thermal camera can see someone at
    all, and it is what collapses at thermal crossover.
    """
    person = apparent_temperature(clothing, config, distance_m, sun_factor)
    background = apparent_temperature("tarmac_shaded", config, distance_m, sun_factor)
    return abs(person - background)


def pixels_on_target(config: ThermalConfig, distance_m: float,
                     target_height_m: float = 1.7) -> float:
    """Vertical pixels a person subtends at a given range."""
    vertical_px = config.resolution[1]
    vfov_rad = 2.0 * np.arctan(np.tan(np.radians(config.hfov_deg) / 2.0)
                               * vertical_px / max(config.resolution[0], 1))
    rad_per_px = vfov_rad / max(vertical_px, 1)
    subtended = 2.0 * np.arctan(target_height_m / (2.0 * max(distance_m, 0.1)))
    return float(subtended / max(rad_per_px, 1e-9))


def detectable(config: ThermalConfig, distance_m: float, clothing: str = "clothing_light",
               sun_factor: float = 0.0, snr_required: float = 3.0,
               hour: float | None = None, task: str = "detect") -> bool:
    """Whether a person can be found at this range.

    Two independent limits, and a model with only the first is misleading:

    * **Thermal contrast** must exceed the sensor's noise floor. This is what NETD
      buys, and it degrades with atmospheric attenuation and at crossover.
    * **Spatial resolution** must put enough pixels on the target. At long range
      this dominates completely — a person at 800 m on a 640x512 core with a 24
      degree lens is about two pixels tall, and no amount of thermal contrast
      makes two pixels into a recognisable person.
    """
    contrast = thermal_contrast(config, distance_m, clothing, sun_factor)
    if hour is not None:
        # At crossover the scene approaches uniform temperature.
        contrast *= max(0.05, 1.0 - 0.9 * crossover_factor(hour))
    if contrast < snr_required * config.netd_mk / 1000.0:
        return False
    return pixels_on_target(config, distance_m) >= JOHNSON_PIXELS.get(task, 1.5)


def crossover_factor(hour: float) -> float:
    """How close the scene is to thermal crossover, 1.0 being worst.

    Surfaces track air temperature twice a day — shortly after dawn and again
    after dusk — and scene contrast nearly vanishes. Any thermal performance
    figure quoted without a time of day is incomplete.
    """
    # Peaks at roughly 07:00 and 19:00.
    return float(max(np.exp(-((hour - 7.0) ** 2) / 2.0),
                     np.exp(-((hour - 19.0) ** 2) / 2.0)))


def render(
    temperature_map: np.ndarray,
    config: ThermalConfig | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Turn a temperature field in Celsius into a thermal image."""
    cfg = config or ThermalConfig()
    rng = np.random.default_rng(seed)
    temps = np.asarray(temperature_map, dtype=np.float32)

    if temps.shape[:2] != (cfg.resolution[1], cfg.resolution[0]):
        temps = cv2.resize(temps, cfg.resolution, interpolation=cv2.INTER_LINEAR)

    # Temporal noise at the sensor's NETD, plus the column-wise fixed pattern that
    # uncooled cores show and that survives naive frame averaging.
    temps = temps + rng.normal(0.0, cfg.netd_mk / 1000.0, temps.shape)
    temps = temps + rng.normal(0.0, cfg.fixed_pattern_mk / 1000.0, (1, temps.shape[1]))

    low, high = cfg.span_c
    scaled = np.clip((temps - low) / max(high - low, 1e-6), 0.0, 1.0)
    grey = (scaled * 255).astype(np.uint8)

    if cfg.dead_pixel_rate > 0:
        n = int(grey.size * cfg.dead_pixel_rate)
        if n:
            ys = rng.integers(0, grey.shape[0], n)
            xs = rng.integers(0, grey.shape[1], n)
            grey[ys, xs] = rng.choice([0, 255], n)

    return _apply_palette(grey, cfg.palette)


def _apply_palette(grey: np.ndarray, palette: str) -> np.ndarray:
    if palette == "black_hot":
        return cv2.cvtColor(255 - grey, cv2.COLOR_GRAY2BGR)
    if palette == "ironbow":
        # Ironbow is what operators usually run; INFERNO is the closest stock map.
        return cv2.applyColorMap(grey, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def scene_temperature_map(
    depth_m: np.ndarray,
    person_mask: np.ndarray | None = None,
    config: ThermalConfig | None = None,
    sun_factor: float = 0.0,
    clothing: str = "clothing_light",
) -> np.ndarray:
    """Build a temperature field from a depth map and a person mask.

    This is how a thermal frame is derived from what the simulator already has:
    depth gives range attenuation, and the instance mask says which pixels are
    people. No new rendering pass is required.
    """
    cfg = config or ThermalConfig()
    depth = np.asarray(depth_m, dtype=np.float32)
    temps = np.empty_like(depth)

    # Background: ground nearby, sky far away — a useful approximation because
    # thermal sky is dramatically cold and dominates the top of most frames.
    far = depth > np.percentile(depth, 92)
    for surface, selector in (("sky", far), ("tarmac_shaded", ~far)):
        if not selector.any():
            continue
        d = depth[selector]
        base = apparent_temperature(surface, cfg, 0.0, sun_factor)
        attenuation = np.exp(-cfg.atmospheric_transmission_per_m * d)
        temps[selector] = cfg.ambient_c + (base - cfg.ambient_c) * attenuation

    if person_mask is not None and person_mask.any():
        mask = person_mask.astype(bool)
        if mask.shape != depth.shape:
            mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        d = depth[mask]
        base = apparent_temperature(clothing, cfg, 0.0, sun_factor)
        attenuation = np.exp(-cfg.atmospheric_transmission_per_m * d)
        temps[mask] = cfg.ambient_c + (base - cfg.ambient_c) * attenuation

    return temps


def range_report(config: ThermalConfig | None = None,
                 distances: tuple[float, ...] = (25.0, 100.0, 300.0, 800.0)) -> dict[str, Any]:
    """Contrast and detectability against range — the perimeter design question."""
    cfg = config or ThermalConfig()
    rows = []
    for d in distances:
        rows.append({
            "distance_m": d,
            "contrast_k": round(thermal_contrast(cfg, d), 3),
            "pixels_on_target": round(pixels_on_target(cfg, d), 2),
            "detectable_light_clothing": detectable(cfg, d, "clothing_light"),
            "detectable_heavy_clothing": detectable(cfg, d, "clothing_heavy"),
            "recognisable": detectable(cfg, d, "clothing_light", task="recognise"),
            "detectable_at_crossover": detectable(cfg, d, "clothing_light", hour=7.0),
            # Which limit bites first at this range, which is what decides whether
            # to buy a better core or a longer lens.
            "limited_by": ("resolution"
                           if pixels_on_target(cfg, d) < JOHNSON_PIXELS["detect"]
                           else "contrast"
                           if thermal_contrast(cfg, d) < 3 * cfg.netd_mk / 1000.0
                           else "neither"),
        })
    return {
        "netd_mk": cfg.netd_mk,
        "hfov_deg": cfg.hfov_deg,
        "rows": rows,
        "note": ("thermal needs no illumination, so range is set by contrast and "
                 "optics rather than by an illuminator — the reason perimeters use "
                 "it and car parks use IR"),
    }

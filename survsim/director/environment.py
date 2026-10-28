"""Time of day and weather, driving the renderer and the sensor model together.

The reason this is one module rather than two settings is that lighting is not a
look, it is a *chain*. Low light forces the camera to raise gain, which raises
noise; it lengthens exposure, which adds motion blur; below a threshold the
IR-cut filter lifts and the image goes monochrome, at which point colour-based
re-identification stops working entirely and the far field goes black because the
illuminator cannot reach it.

If the render is darkened without also changing gain, blur and colour, the result
is a dim picture that is still unrealistically clean — and a model trained on it
will look excellent until it meets real night footage.

So `EnvironmentDirector` computes illuminance from sun position and cloud cover,
then derives both the engine parameters and the matching degradation settings
from the same physical state.

Sun position uses the standard solar-position approximation: accurate to a degree
or so, which is far finer than anything downstream needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

# Illuminance in lux for the sky conditions that matter operationally.
LUX_DIRECT_SUN = 100_000.0
LUX_OVERCAST_DAY = 2_000.0
LUX_TWILIGHT = 10.0
LUX_FULL_MOON = 0.3
LUX_OVERCAST_NIGHT = 0.02

# Below this, a camera switches to IR: colour is gone and so is colour-based ReID.
IR_SWITCH_LUX = 5.0


@dataclass
class WeatherPreset:
    name: str
    cloud: float = 0.0            # 0 clear .. 1 overcast
    fog_density: float = 0.0      # extinction coefficient, 1/m-ish
    rain: float = 0.0             # 0 none .. 1 torrential
    snow: float = 0.0
    wind_mps: float = 1.0
    wetness: float = 0.0          # ground reflectivity, drives glare and puddles
    lens_water: float = 0.0       # droplets on the housing


PRESETS: dict[str, WeatherPreset] = {
    "clear": WeatherPreset("clear", cloud=0.05),
    "overcast": WeatherPreset("overcast", cloud=0.9, wind_mps=3.0),
    "rain_light": WeatherPreset("rain_light", cloud=1.0, rain=0.3, wind_mps=4.0,
                                wetness=0.7, lens_water=0.25),
    "rain_heavy": WeatherPreset("rain_heavy", cloud=1.0, rain=1.0, wind_mps=9.0,
                                wetness=1.0, lens_water=0.7, fog_density=0.03),
    "fog": WeatherPreset("fog", cloud=1.0, fog_density=0.35, wind_mps=0.5, wetness=0.4),
    "snow": WeatherPreset("snow", cloud=1.0, snow=0.7, wind_mps=5.0, wetness=0.3,
                          lens_water=0.2),
    "dust": WeatherPreset("dust", cloud=0.3, fog_density=0.18, wind_mps=12.0),
}


def solar_elevation_deg(when: datetime, lat_deg: float, lon_deg: float) -> float:
    """Sun elevation above the horizon, in degrees. Negative means below."""
    day = when.timetuple().tm_yday
    hour = when.hour + when.minute / 60.0 + when.second / 3600.0

    decl = math.radians(23.45) * math.sin(math.radians(360.0 * (284 + day) / 365.0))
    # Equation of time, then true solar time for this longitude.
    b = math.radians(360.0 * (day - 81) / 364.0)
    eot = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)
    solar_time = hour + (4.0 * lon_deg + eot) / 60.0
    hour_angle = math.radians(15.0 * (solar_time - 12.0))

    lat = math.radians(lat_deg)
    sin_elev = (math.sin(lat) * math.sin(decl)
                + math.cos(lat) * math.cos(decl) * math.cos(hour_angle))
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))


def illuminance_lux(elevation_deg: float, cloud: float, moon_fraction: float = 0.5) -> float:
    """Horizontal illuminance from sun elevation and cloud cover.

    The curve is piecewise because daylight, twilight and night are governed by
    different physics, and the twilight band is where cameras actually struggle.
    """
    cloud_factor = 1.0 - 0.95 * float(np.clip(cloud, 0.0, 1.0))

    if elevation_deg > 5.0:
        direct = LUX_DIRECT_SUN * math.sin(math.radians(elevation_deg))
        return max(direct * cloud_factor, LUX_OVERCAST_DAY * cloud_factor)
    if elevation_deg > -0.833:                       # sun on the horizon
        return float(np.interp(elevation_deg, [-0.833, 5.0],
                               [400.0, LUX_OVERCAST_DAY]) * cloud_factor)
    if elevation_deg > -18.0:                        # civil to astronomical twilight
        frac = (elevation_deg + 18.0) / 17.167
        return float(max(LUX_TWILIGHT * (frac**3) * 40.0, LUX_FULL_MOON) * cloud_factor)

    night = LUX_FULL_MOON * moon_fraction
    return float(max(night * cloud_factor, LUX_OVERCAST_NIGHT))


@dataclass
class EnvironmentState:
    """The physical state at one instant, and everything derived from it."""

    t: float
    when: datetime
    sun_elevation_deg: float
    lux: float
    weather: WeatherPreset
    is_night: bool
    ir_mode: bool

    def engine_params(self) -> dict[str, Any]:
        """What to set in the renderer."""
        return {
            "sun_elevation_deg": round(self.sun_elevation_deg, 2),
            "sun_intensity_lux": round(self.lux, 2),
            "cloud": round(self.weather.cloud, 3),
            "fog_density": round(self.weather.fog_density, 4),
            "rain": round(self.weather.rain, 3),
            "snow": round(self.weather.snow, 3),
            "wind_mps": round(self.weather.wind_mps, 2),
            "wetness": round(self.weather.wetness, 3),
            "street_lights_on": self.lux < 50.0,
        }

    def degradation_overrides(self) -> dict[str, Any]:
        """Sensor settings implied by the same physical state.

        Gain rises as light falls, and noise rises with gain; exposure lengthens,
        so motion blur grows; below the IR threshold the image is monochrome with
        illuminator falloff. These are consequences of the lighting, not
        independent style choices, which is why they are computed here.
        """
        # Roughly one stop of gain per halving of light, clamped to sane hardware.
        gain = float(np.clip(3.0 * (2000.0 / max(self.lux, 0.02)) ** 0.35, 1.0, 16.0))
        blur = float(np.clip(0.4 * (500.0 / max(self.lux, 0.02)) ** 0.25, 0.0, 9.0))
        return {
            "iso_gain": round(gain, 2),
            "read_noise": round(float(np.clip(1.5 + gain * 0.35, 1.5, 7.0)), 2),
            "fpn": round(float(np.clip(0.3 + gain * 0.1, 0.3, 2.0)), 2),
            "motion_px": round(blur, 2),
            "night_ir": self.ir_mode,
            "haze": round(min(0.6, self.weather.fog_density * 1.4), 3),
            "dirt": round(min(0.4, 0.08 + self.weather.lens_water * 0.4), 3),
            "ev": round(float(np.clip(math.log2(max(self.lux, 0.02) / 2000.0) * 0.25, -1.5, 0.5)), 2),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "t": round(self.t, 2),
            "time": self.when.isoformat(),
            "sun_elevation_deg": round(self.sun_elevation_deg, 2),
            "lux": round(self.lux, 3),
            "weather": self.weather.name,
            "is_night": self.is_night,
            "ir_mode": self.ir_mode,
        }


class EnvironmentDirector:
    """Advances time of day and weather, including scheduled changes."""

    def __init__(
        self,
        start: datetime | str = "2026-06-14T21:30:00",
        lat_lon: tuple[float, float] = (51.5074, -0.1278),
        weather: str | WeatherPreset = "clear",
        time_scale: float = 1.0,
        schedule: list[dict[str, Any]] | None = None,
        moon_fraction: float = 0.5,
    ) -> None:
        self.start = datetime.fromisoformat(start) if isinstance(start, str) else start
        self.lat, self.lon = lat_lon
        self.time_scale = float(time_scale)
        self.moon_fraction = moon_fraction
        self._base = PRESETS[weather] if isinstance(weather, str) else weather
        self._current = self._base
        # Each entry: {t, preset, ramp_s}. Weather changes take time, and a ramp
        # is where auto-exposure hunts and detectors behave worst.
        self._schedule = sorted(schedule or [], key=lambda s: float(s["t"]))
        self._applied: list[dict[str, Any]] = []
        self.history: list[EnvironmentState] = []

    def weather_at(self, t: float) -> WeatherPreset:
        """Blend toward any scheduled preset that has started."""
        current = self._base
        for entry in self._schedule:
            t0 = float(entry["t"])
            if t < t0:
                break
            target = PRESETS[entry["preset"]] if isinstance(entry["preset"], str) \
                else entry["preset"]
            ramp = float(entry.get("ramp_s", 0.0))
            alpha = 1.0 if ramp <= 0 else float(np.clip((t - t0) / ramp, 0.0, 1.0))
            current = _blend(current, target, alpha)
        return current

    def state_at(self, t: float) -> EnvironmentState:
        when = self.start + timedelta(seconds=t * self.time_scale)
        elevation = solar_elevation_deg(when, self.lat, self.lon)
        weather = self.weather_at(t)
        lux = illuminance_lux(elevation, weather.cloud, self.moon_fraction)
        return EnvironmentState(
            t=t, when=when, sun_elevation_deg=elevation, lux=lux, weather=weather,
            is_night=elevation < -0.833, ir_mode=lux < IR_SWITCH_LUX,
        )

    def step(self, t: float) -> EnvironmentState:
        state = self.state_at(t)
        self.history.append(state)
        self._current = state.weather
        return state

    def summary(self) -> dict[str, Any]:
        if not self.history:
            return {}
        lux = [s.lux for s in self.history]
        return {
            "lux_min": round(min(lux), 3),
            "lux_max": round(max(lux), 3),
            "ir_fraction": round(sum(s.ir_mode for s in self.history) / len(self.history), 3),
            "night_fraction": round(sum(s.is_night for s in self.history) / len(self.history), 3),
            "weathers": sorted({s.weather.name for s in self.history}),
        }


def _blend(a: WeatherPreset, b: WeatherPreset, alpha: float) -> WeatherPreset:
    """Linear interpolation between two weather states."""
    if alpha <= 0:
        return a
    if alpha >= 1:
        return b
    mix = lambda x, y: float(x + (y - x) * alpha)  # noqa: E731
    return WeatherPreset(
        name=f"{a.name}->{b.name}@{alpha:.2f}",
        cloud=mix(a.cloud, b.cloud), fog_density=mix(a.fog_density, b.fog_density),
        rain=mix(a.rain, b.rain), snow=mix(a.snow, b.snow),
        wind_mps=mix(a.wind_mps, b.wind_mps), wetness=mix(a.wetness, b.wetness),
        lens_water=mix(a.lens_water, b.lens_water),
    )

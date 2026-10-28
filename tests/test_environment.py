"""Tests for the time-of-day and weather director.

The properties that matter: the sun has to be in the right place, illuminance has
to span the seven orders of magnitude between noon and a moonless night, and the
sensor consequences of darkness must follow automatically — because a render that
is merely dim, without the gain noise and the loss of colour, trains a model that
fails on real night footage.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from survsim.director.environment import (
    IR_SWITCH_LUX,
    EnvironmentDirector,
    illuminance_lux,
    solar_elevation_deg,
)

LONDON = (51.5074, -0.1278)


# ------------------------------------------------------------- sun position


def test_sun_is_high_at_summer_noon_and_below_horizon_at_midnight() -> None:
    noon = solar_elevation_deg(datetime(2026, 6, 21, 12, 0), *LONDON)  # noqa: DTZ001 - naive local time is the API contract
    midnight = solar_elevation_deg(datetime(2026, 6, 21, 0, 0), *LONDON)  # noqa: DTZ001
    assert 55.0 < noon < 65.0        # London midsummer noon is about 62 degrees
    assert midnight < 0.0


def test_winter_sun_is_lower_than_summer_sun() -> None:
    summer = solar_elevation_deg(datetime(2026, 6, 21, 12, 0), *LONDON)  # noqa: DTZ001
    winter = solar_elevation_deg(datetime(2026, 12, 21, 12, 0), *LONDON)  # noqa: DTZ001
    assert winter < summer - 30.0


def test_equator_noon_is_higher_than_london_noon() -> None:
    equator = solar_elevation_deg(datetime(2026, 3, 21, 12, 0), 0.0, 0.0)  # noqa: DTZ001
    london = solar_elevation_deg(datetime(2026, 3, 21, 12, 0), *LONDON)  # noqa: DTZ001
    assert equator > london


# ------------------------------------------------------------- illuminance


def test_illuminance_spans_day_to_night() -> None:
    day = illuminance_lux(60.0, cloud=0.0)
    overcast = illuminance_lux(60.0, cloud=1.0)
    twilight = illuminance_lux(-6.0, cloud=0.2)
    night = illuminance_lux(-30.0, cloud=0.5)

    assert day > 50_000
    assert overcast < day / 10
    assert twilight < overcast
    assert night < 1.0
    assert night > 0.0          # never absolutely black


def test_cloud_reduces_daylight() -> None:
    assert illuminance_lux(45.0, cloud=1.0) < illuminance_lux(45.0, cloud=0.0)


# ------------------------------------------------- lighting drives the sensor


def test_darkness_raises_gain_and_noise() -> None:
    """Low light is a chain: less light -> more gain -> more noise."""
    day = EnvironmentDirector("2026-06-21T12:00:00", LONDON, "clear").state_at(0)
    night = EnvironmentDirector("2026-06-21T01:00:00", LONDON, "clear").state_at(0)

    d, n = day.degradation_overrides(), night.degradation_overrides()
    assert n["iso_gain"] > d["iso_gain"] * 2
    assert n["read_noise"] > d["read_noise"]
    assert n["motion_px"] > d["motion_px"]


def test_night_switches_the_camera_to_infrared() -> None:
    """Below the IR threshold the image is monochrome, so colour ReID stops working."""
    night = EnvironmentDirector("2026-01-15T23:00:00", LONDON, "clear").state_at(0)
    assert night.lux < IR_SWITCH_LUX
    assert night.ir_mode is True
    assert night.degradation_overrides()["night_ir"] is True


def test_daylight_does_not_use_infrared() -> None:
    day = EnvironmentDirector("2026-06-21T12:00:00", LONDON, "clear").state_at(0)
    assert day.ir_mode is False
    assert day.is_night is False


def test_street_lights_come_on_when_it_gets_dark() -> None:
    night = EnvironmentDirector("2026-01-15T22:00:00", LONDON, "clear").state_at(0)
    day = EnvironmentDirector("2026-06-21T12:00:00", LONDON, "clear").state_at(0)
    assert night.engine_params()["street_lights_on"] is True
    assert day.engine_params()["street_lights_on"] is False


def test_fog_raises_haze_in_both_render_and_sensor() -> None:
    """A weather change must reach the renderer and the degradation together."""
    clear = EnvironmentDirector("2026-06-21T12:00:00", LONDON, "clear").state_at(0)
    fog = EnvironmentDirector("2026-06-21T12:00:00", LONDON, "fog").state_at(0)

    assert fog.engine_params()["fog_density"] > clear.engine_params()["fog_density"]
    assert fog.degradation_overrides()["haze"] > clear.degradation_overrides()["haze"]


def test_rain_puts_water_on_the_lens() -> None:
    rain = EnvironmentDirector("2026-06-21T12:00:00", LONDON, "rain_heavy").state_at(0)
    clear = EnvironmentDirector("2026-06-21T12:00:00", LONDON, "clear").state_at(0)
    assert rain.degradation_overrides()["dirt"] > clear.degradation_overrides()["dirt"]
    assert rain.engine_params()["wetness"] > 0.5


# ------------------------------------------------------------- scheduling


def test_scheduled_weather_ramps_rather_than_snapping() -> None:
    """Fog rolls in; the ramp is where auto-exposure hunts and detectors suffer."""
    d = EnvironmentDirector("2026-06-21T12:00:00", LONDON, "clear",
                            schedule=[{"t": 100, "preset": "fog", "ramp_s": 60}])
    before = d.state_at(50).weather.fog_density
    middle = d.state_at(130).weather.fog_density
    after = d.state_at(200).weather.fog_density

    assert before == pytest.approx(0.0, abs=1e-6)
    assert 0 < middle < after
    assert after == pytest.approx(0.35, abs=1e-6)


def test_time_scale_accelerates_the_day() -> None:
    """A dusk transition in a short run needs time to move faster than real time."""
    # 60 s of run time becomes 2 h of simulated time, taking 20:00 to 22:00.
    # Care is needed not to overshoot into the following dawn, when the sun is up
    # again and the scene is correctly bright once more.
    fast = EnvironmentDirector("2026-06-21T20:00:00", LONDON, "clear", time_scale=120.0)
    start = fast.state_at(0).lux
    later = fast.state_at(60).lux
    assert later < start
    assert fast.state_at(60).when.hour == 22


def test_summary_reports_the_lighting_range_covered() -> None:
    d = EnvironmentDirector("2026-06-21T19:00:00", LONDON, "clear", time_scale=900.0)
    for tick in range(40):
        d.step(tick * 1.0)
    s = d.summary()
    assert s["lux_max"] > s["lux_min"]
    assert 0.0 <= s["ir_fraction"] <= 1.0

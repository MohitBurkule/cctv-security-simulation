"""Tests for thermal (LWIR) imaging.

Thermal is often treated as "night mode but better", and these tests pin the ways
it genuinely differs: it needs no illuminator, it images clothing rather than
people, low-emissivity surfaces lie about their temperature, and contrast
collapses at thermal crossover regardless of how good the sensor is.
"""

from __future__ import annotations

import numpy as np

from survsim.sensors.thermal import (
    EMISSIVITY,
    JOHNSON_PIXELS,
    PALETTES,
    SURFACE_TEMPS,
    ThermalConfig,
    apparent_temperature,
    crossover_factor,
    detectable,
    pixels_on_target,
    range_report,
    render,
    scene_temperature_map,
    thermal_contrast,
)

# ------------------------------------------------------------- physics


def test_skin_is_much_warmer_than_clothing() -> None:
    """Thermal detection of people is really detection of exposed skin."""
    cfg = ThermalConfig()
    assert (apparent_temperature("skin", cfg)
            > apparent_temperature("clothing_light", cfg)
            > apparent_temperature("clothing_heavy", cfg))


def test_heavy_clothing_hides_a_person() -> None:
    """Wrapping up is a genuine, low-tech counter to thermal."""
    cfg = ThermalConfig()
    assert (thermal_contrast(cfg, 50.0, "clothing_heavy")
            < thermal_contrast(cfg, 50.0, "clothing_light"))


def test_low_emissivity_surfaces_read_closer_to_ambient() -> None:
    """A car body reads far cooler than it is, which routinely confuses operators."""
    cfg = ThermalConfig(ambient_c=15.0)
    body = apparent_temperature("vehicle_body", cfg)
    assert EMISSIVITY["vehicle_body"] < 0.7
    assert body < SURFACE_TEMPS["vehicle_body"]


def test_contrast_falls_with_range() -> None:
    cfg = ThermalConfig()
    assert thermal_contrast(cfg, 20.0) > thermal_contrast(cfg, 500.0)


def test_thermal_sees_further_than_an_ir_illuminator_would() -> None:
    """The reason perimeters use thermal: no illumination is required at all."""
    cfg = ThermalConfig()
    assert detectable(cfg, 300.0), "a person at 300 m should still be detectable"


def test_resolution_not_contrast_limits_long_range() -> None:
    """The correction that makes this model usable for perimeter design.

    Modelling thermal contrast alone says a cheap core detects people at 800 m,
    because a warm body stays well above the noise floor. In reality the person
    subtends under one pixel on a wide-lens 320x256 core, and no amount of
    thermal contrast turns that into a detection.
    """
    wide = ThermalConfig(resolution=(320, 256), hfov_deg=50.0)
    assert thermal_contrast(wide, 800.0) > 3 * wide.netd_mk / 1000.0   # contrast is fine
    assert pixels_on_target(wide, 800.0) < JOHNSON_PIXELS["detect"]     # pixels are not
    assert not detectable(wide, 800.0)


def test_a_longer_lens_buys_range_at_the_same_sensor() -> None:
    narrow = ThermalConfig(resolution=(320, 256), hfov_deg=12.0)
    wide = ThermalConfig(resolution=(320, 256), hfov_deg=50.0)
    assert pixels_on_target(narrow, 300.0) > pixels_on_target(wide, 300.0)
    assert detectable(narrow, 800.0) and not detectable(wide, 800.0)


def test_recognition_needs_far_more_pixels_than_detection() -> None:
    """Seeing that something is there is not seeing what it is."""
    cfg = ThermalConfig()
    assert detectable(cfg, 800.0, task="detect")
    assert not detectable(cfg, 800.0, task="recognise")


def test_crossover_reduces_effective_range() -> None:
    cheap = ThermalConfig(netd_mk=200.0)
    assert detectable(cheap, 400.0, hour=13.0)          # midday, good contrast
    assert not detectable(cheap, 400.0, hour=7.0)       # dawn crossover


def test_report_names_the_binding_constraint() -> None:
    """Whether to buy a better core or a longer lens is the actual question."""
    rows = range_report(ThermalConfig(resolution=(320, 256), hfov_deg=50.0),
                        distances=(25.0, 800.0))["rows"]
    assert rows[0]["limited_by"] == "neither"
    assert rows[-1]["limited_by"] == "resolution"


def test_a_better_sensor_detects_further() -> None:
    good = ThermalConfig(netd_mk=30.0)
    cheap = ThermalConfig(netd_mk=150.0)
    far = 900.0
    assert detectable(good, far) or not detectable(cheap, far)
    assert not (detectable(cheap, far) and not detectable(good, far))


# ---------------------------------------------------------- crossover


def test_crossover_peaks_around_dawn_and_dusk() -> None:
    """Twice a day the scene loses contrast, and a noon benchmark hides it."""
    assert crossover_factor(7.0) > 0.9
    assert crossover_factor(19.0) > 0.9
    assert crossover_factor(13.0) < 0.1
    assert crossover_factor(2.0) < 0.1


def test_sun_loading_changes_background_temperature() -> None:
    cfg = ThermalConfig()
    shaded = apparent_temperature("tarmac_sunlit", cfg, sun_factor=0.0)
    sunlit = apparent_temperature("tarmac_sunlit", cfg, sun_factor=1.0)
    assert sunlit > shaded + 5.0


# ------------------------------------------------------------ render


def test_render_produces_an_image_at_the_sensor_resolution() -> None:
    cfg = ThermalConfig(resolution=(320, 256))
    img = render(np.full((100, 120), 20.0), cfg)
    assert img.shape == (256, 320, 3)
    assert img.dtype == np.uint8


def test_warm_regions_are_brighter_in_white_hot() -> None:
    temps = np.full((64, 64), 15.0)
    temps[20:40, 20:40] = 33.0
    img = render(temps, ThermalConfig(resolution=(64, 64), palette="white_hot"), seed=1)

    warm = img[20:40, 20:40].mean()
    cold = img[0:10, 0:10].mean()
    assert warm > cold + 40


def test_black_hot_inverts_the_mapping() -> None:
    temps = np.full((64, 64), 15.0)
    temps[20:40, 20:40] = 33.0
    white = render(temps, ThermalConfig(resolution=(64, 64), palette="white_hot"), seed=1)
    black = render(temps, ThermalConfig(resolution=(64, 64), palette="black_hot"), seed=1)

    assert white[20:40, 20:40].mean() > white[0:10, 0:10].mean()
    assert black[20:40, 20:40].mean() < black[0:10, 0:10].mean()


def test_all_palettes_render() -> None:
    temps = np.full((32, 32), 20.0)
    for palette in PALETTES:
        img = render(temps, ThermalConfig(resolution=(32, 32), palette=palette))
        assert img.shape == (32, 32, 3)


def test_sensor_noise_is_present_and_scales_with_netd() -> None:
    temps = np.full((128, 128), 20.0)
    quiet = render(temps, ThermalConfig(resolution=(128, 128), netd_mk=10.0,
                                        fixed_pattern_mk=1.0, dead_pixel_rate=0.0), seed=2)
    noisy = render(temps, ThermalConfig(resolution=(128, 128), netd_mk=400.0,
                                        fixed_pattern_mk=1.0, dead_pixel_rate=0.0), seed=2)
    assert noisy.std() > quiet.std()


# ------------------------------------------------- scene reconstruction


def test_temperature_map_makes_people_stand_out_from_background() -> None:
    depth = np.full((64, 64), 30.0, dtype=np.float32)
    mask = np.zeros((64, 64), dtype=bool)
    mask[25:40, 25:40] = True

    temps = scene_temperature_map(depth, mask, ThermalConfig())
    assert temps[25:40, 25:40].mean() > temps[0:10, 0:10].mean() + 3.0


def test_distant_people_have_less_contrast_than_near_ones() -> None:
    """Atmospheric attenuation, the reason a long perimeter needs a better core."""
    cfg = ThermalConfig()
    near = np.full((32, 32), 20.0, dtype=np.float32)
    far = np.full((32, 32), 900.0, dtype=np.float32)
    mask = np.ones((32, 32), dtype=bool)

    near_t = scene_temperature_map(near, mask, cfg).mean()
    far_t = scene_temperature_map(far, mask, cfg).mean()
    assert abs(near_t - cfg.ambient_c) > abs(far_t - cfg.ambient_c)


def test_map_without_people_is_background_only() -> None:
    temps = scene_temperature_map(np.full((32, 32), 25.0, dtype=np.float32), None)
    assert np.isfinite(temps).all()


# ------------------------------------------------------------- report


def test_range_report_answers_the_perimeter_design_question() -> None:
    report = range_report(ThermalConfig())
    assert report["rows"][0]["contrast_k"] > report["rows"][-1]["contrast_k"]
    assert "no illumination" in report["note"]


def test_report_distinguishes_clothing_levels() -> None:
    rows = range_report(ThermalConfig(netd_mk=120.0),
                        distances=(50.0, 600.0))["rows"]
    assert any(r["detectable_light_clothing"] != r["detectable_heavy_clothing"]
               for r in rows) or all(r["detectable_light_clothing"] for r in rows)

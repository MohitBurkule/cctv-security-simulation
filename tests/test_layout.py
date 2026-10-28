"""Site layouts (survsim.generator.layout): distinct, deterministic, installable."""
import datetime as dt
import math

import numpy as np
import pytest

from survsim.export.env_plan import layout_camera_manifest, plan_from_layout
from survsim.generator.layout import (
    SITE_TYPES,
    build_layout,
    features,
    solar_position,
    sun_to_unreal,
)

SEEDS = range(1, 7)


@pytest.fixture(scope="module")
def layouts():
    return {(s, k): build_layout(s, k) for s in SITE_TYPES for k in SEEDS}


def _placements(L):
    return {(i.kind, round(i.x / 2.0), round(i.y / 2.0)) for i in L.items
            if i.kind not in ("decal",)}


def shared(a, b):
    pa, pb = _placements(a), _placements(b)
    return len(pa & pb) / max(1, min(len(pa), len(pb)))


def test_deterministic():
    a, b = build_layout("car_park", 5), build_layout("car_park", 5)
    assert a.to_dict() == b.to_dict()
    assert a.to_dict() != build_layout("car_park", 6).to_dict()


def test_two_seeds_of_a_type_share_under_half(layouts):
    worst = 0.0
    for s in SITE_TYPES:
        for k in SEEDS:
            for j in SEEDS:
                if k < j:
                    worst = max(worst, shared(layouts[(s, k)], layouts[(s, j)]))
    assert worst < 0.5, worst


def test_site_types_are_separable_by_a_simple_classifier(layouts):
    """Leave-one-out nearest centroid on the plan features: no confusions."""
    keys = list(layouts)
    X = np.array([features(layouts[k]) for k in keys])
    y = [k[0] for k in keys]
    wrong = 0
    for i in range(len(keys)):
        cents = {}
        for s in SITE_TYPES:
            idx = [j for j in range(len(keys)) if j != i and y[j] == s]
            cents[s] = X[idx].mean(axis=0)
        pred = min(cents, key=lambda s: np.linalg.norm(X[i] - cents[s]))
        wrong += pred != y[i]
    assert wrong == 0, wrong


def test_every_layout_passes_its_gate(layouts):
    for k, L in layouts.items():
        assert L.report["ok"], (k, L.report["failures"])
        assert all(L.report["reachable"].values())
        assert all(L.report["entrance_cameras"].values())


def test_cameras_are_installed_like_an_installer_would(layouts):
    for L in layouts.values():
        assert L.cameras
        for c in L.cameras:
            assert 3.0 <= c["z"] <= 6.0
            assert c["mount"] in ("wall", "pole")
            assert -40.0 <= c["pitch"] <= -12.0
        assert {c["target"] for c in L.cameras} & {e["id"] for e in L.entrances}


def test_scenario_entrances_become_the_gates():
    pts = [(0.0, 16.0), (16.0, 0.0), (-16.0, -3.0)]
    L = build_layout("perimeter_site", 3, core_half=17.0, entrances=pts)
    assert len(L.entrances) == 3
    for (px, py), e in zip(pts, L.entrances):
        assert math.hypot(px - e["x"], py - e["y"]) < 6.0
    assert L.report["ok"], L.report["failures"]


def test_solar_model_known_values():
    # London, 2026-06-21 12:00 UTC: sun ~62 deg up, due south-ish.
    el, az = solar_position(51.51, -0.13, dt.datetime(2026, 6, 21, 12, 0, tzinfo=dt.timezone.utc))
    assert 61.0 < el < 63.0 and 170 < az < 190
    # Midwinter noon is low; a winter evening is below the horizon.
    el, _ = solar_position(51.51, -0.13, dt.datetime(2026, 12, 21, 12, 0, tzinfo=dt.timezone.utc))
    assert 14.0 < el < 16.0
    assert solar_position(51.51, -0.13, dt.datetime(2026, 12, 21, 18, 0, tzinfo=dt.timezone.utc))[0] < 0
    # Sun in the south: light travels north (survsim +y) = Unreal yaw -90.
    yaw, pitch = sun_to_unreal(40.0, 180.0)
    assert pitch == -40.0 and yaw % 360 == 270.0


def test_weather_and_time_vary_and_are_daytime(layouts):
    states = {L.weather["state"] for L in layouts.values()}
    assert states == {"wet", "dry"}
    assert all(L.time["sun_elevation"] >= 12.0 for L in layouts.values())
    assert len({L.time["season"] for L in layouts.values()}) >= 3


def test_plan_v2_keeps_the_near_view_and_walked_cells_clear():
    L = build_layout("retail_street", 2).to_dict()
    cams = layout_camera_manifest(L)
    walked = {(i, 0) for i in range(-15, 15)}
    p = plan_from_layout(L, cams, walked)
    from survsim.export.env_plan import _Placer
    P = _Placer(cams, p["half_cm"], walked)
    for it in p["items"]:
        if it["size"][2] >= 30 and it["kind"] not in ("canopy", "shop_sign", "awning"):
            assert not P.in_near_frustum(it["x"], it["y"], it["yaw"], it["size"][:2]), it
    assert p["version"] == 2 and p["entrances"] and p["lighting"]["sun_pitch"] < 0


def test_wall_cameras_are_outside_every_block(layouts):
    """[X] 2026-09-26: mounts used unrotated w/d and put cameras inside yaw-90 blocks."""
    for L in layouts.values():
        for c in L.cameras:
            for it in L.items:
                if it.kind == "block":
                    assert not it.contains(c["x"], c["y"], pad=-0.05), (L.site_type, L.seed, c)


def test_every_camera_watches_the_walkable_core(layouts):
    """No camera is aimed at nothing: each usably sees >= 30 % of the walk core
    (>= 20 px, the one visibility rule), so people cross every view."""
    from survsim.core.geometry import in_view, pixel_heights
    from survsim.generator.layout import camera_models
    for L in layouts.values():
        C = L.core_half
        g = np.arange(-C, C + 0.1, 1.0)
        gx, gy = np.meshgrid(g, g)
        G = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1)
        for cam in camera_models(L):
            frac = float((in_view(cam, G) & (pixel_heights(cam, G) >= 20)).mean())
            assert frac >= 0.3, (L.site_type, L.seed, cam.id, frac)

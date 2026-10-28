"""Tests that run with no GPU and no engine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from survsim.core.geometry import extrinsics, in_view, intrinsics, pixel_height, project
from survsim.core.loader import load_scenario
from survsim.core.models import Camera, Mount
from survsim.core.rng import RngHub
from survsim.simulator import Simulator

SCENARIO = Path(__file__).parent.parent / "configs" / "scenarios" / "smoke_mock.yaml"


@pytest.fixture
def cam() -> Camera:
    return Camera(
        id="c", mount=Mount(pos=(0.0, 0.0, 3.0), yaw=0.0, pitch=-20.0),
        optics={"hfov_deg": 78}, capture={"res": (1280, 720)},
    )


# ------------------------------------------------------------------- geometry


def test_principal_ray_lands_at_image_centre(cam: Camera) -> None:
    """A point straight down the optical axis must hit the principal point."""
    r, _ = extrinsics(cam)
    forward = r.T @ np.array([0.0, 0.0, 1.0])
    uv, depth = project(cam, np.asarray(cam.mount.pos) + forward * 25.0)
    assert depth[0] > 0
    assert uv[0] == pytest.approx([640.0, 360.0], abs=1e-6)


def test_points_behind_camera_have_negative_depth(cam: Camera) -> None:
    _, depth = project(cam, np.array([[-30.0, 0.0, 1.0]]))
    assert depth[0] < 0


def test_pixel_height_falls_with_range(cam: Camera) -> None:
    heights = [pixel_height(cam, (d, 0.0, 0.0), 1.7) for d in (10.0, 20.0, 40.0)]
    assert heights[0] > heights[1] > heights[2] > 0


def test_pixel_height_matches_closed_form() -> None:
    """At zero tilt the optical axis is horizontal, so px = fx * H / Z exactly.

    The tilted case is deliberately not checked against 1/d: the true range is
    sqrt(d^2 + mount_height^2) and the tilt adds foreshortening, so a naive
    "double the distance, halve the pixels" assertion is wrong by several percent.
    """
    level = Camera(
        id="level", mount=Mount(pos=(0.0, 0.0, 1.7), yaw=0.0, pitch=0.0),
        optics={"hfov_deg": 78}, capture={"res": (1280, 720)},
    )
    fx = intrinsics(level)[0, 0]
    for d in (10.0, 20.0, 40.0):
        assert pixel_height(level, (d, 0.0, 0.0), 1.7) == pytest.approx(fx * 1.7 / d, rel=1e-6)


def test_negative_pitch_tilts_the_camera_downwards() -> None:
    """A `pitch: -20` camera must look at the ground, not the sky.

    Geometry pins the answer exactly. The camera sits 4.5 m up with its axis 20 deg
    below horizontal, so the axis meets the ground at 4.5/tan(20 deg) = 12.4 m.
    A point nearer than that is below the axis (v > centre); one further away is
    above it (v < centre). Getting the sign wrong inverts both, which is how this
    bug survived a self-consistent principal-ray test.
    """
    down = Camera(
        id="down", mount=Mount(pos=(0.0, 0.0, 4.5), yaw=0.0, pitch=-20.0),
        optics={"hfov_deg": 70}, capture={"res": (1920, 1080)},
    )
    crossover = 4.5 / np.tan(np.radians(20.0))
    near_uv, near_d = project(down, np.array([[crossover / 2, 0.0, 0.0]]))
    far_uv, far_d = project(down, np.array([[crossover * 2, 0.0, 0.0]]))

    assert near_d[0] > 0 and far_d[0] > 0
    assert near_uv[0, 1] > 540.0, "near ground point should fall below image centre"
    assert far_uv[0, 1] < 540.0, "far ground point should fall above image centre"


def test_downward_camera_actually_sees_its_scene() -> None:
    """A normally mounted camera must see the ground in front of it."""
    cam = Camera(
        id="c", mount=Mount(pos=(20.0, -10.0, 4.5), yaw=135.0, pitch=-20.0),
        optics={"hfov_deg": 70}, capture={"res": (1920, 1080)},
    )
    yaw = np.radians(cam.mount.yaw)
    pts = np.array([
        [cam.mount.pos[0] + d * np.cos(yaw), cam.mount.pos[1] + d * np.sin(yaw), 0.0]
        for d in (8.0, 15.0, 25.0)
    ])
    assert in_view(cam, pts).all()


def test_intrinsics_match_requested_fov(cam: Camera) -> None:
    k = intrinsics(cam)
    w = cam.capture.res[0]
    recovered = np.degrees(2 * np.arctan((w / 2) / k[0, 0]))
    assert recovered == pytest.approx(cam.hfov_deg, rel=1e-6)


# ------------------------------------------------------------------------ rng


def test_seed_derivation_is_stable_and_independent() -> None:
    a, b = RngHub(42), RngHub(42)
    assert a("spawn").random() == b("spawn").random()
    assert RngHub(42)("spawn").random() != RngHub(43)("spawn").random()
    # Drawing from one subsystem must not disturb another.
    hub = RngHub(42)
    hub("planner").random()
    assert hub("spawn").random() == RngHub(42)("spawn").random()


# ------------------------------------------------------------------- scenario


def test_smoke_scenario_loads() -> None:
    s = load_scenario(SCENARIO)
    assert s.name == "smoke_mock"
    assert len(s.cameras) == 2
    assert {p.id for p in s.population.hero} == {"loiterer_1", "commuter_1", "courier_1"}


def test_duplicate_camera_ids_rejected() -> None:
    s = load_scenario(SCENARIO)
    with pytest.raises(ValueError, match="duplicate camera ids"):
        type(s).model_validate({**s.model_dump(), "cameras": [
            s.cameras[0].model_dump(), s.cameras[0].model_dump()]})


# ----------------------------------------------------------------- end to end


def test_run_produces_ground_truth(tmp_path: Path) -> None:
    sim = Simulator.from_yaml(SCENARIO, engine="mock")
    result = sim.record(tmp_path, topdown_video=False)

    assert result["ticks"] == sim.scenario.n_ticks
    assert result["events"] > 0

    mot = (tmp_path / "gt" / "mot_cam_01.txt").read_text().splitlines()
    assert mot, "camera 1 saw nobody; check camera placement or projection"
    assert len(mot[0].split(",")) == 9  # MOTChallenge column count

    calib = json.loads((tmp_path / "calib" / "cam_01.json").read_text())
    assert np.allclose(np.linalg.det(np.array(calib["R"])), 1.0, atol=1e-6)


def test_loiter_and_hard_negative_both_emitted(tmp_path: Path) -> None:
    """The simulator must produce the false-positive case, not just the positive."""
    sim = Simulator.from_yaml(SCENARIO, engine="mock")
    sim.record(tmp_path, topdown_video=False)

    labels = {
        json.loads(line)["label"]
        for line in (tmp_path / "gt" / "events.jsonl").read_text().splitlines()
        if line.strip()
    }
    assert any(x.startswith("suspicious.loiter") for x in labels)
    assert any(x.startswith("normal.commute_wait") for x in labels)


@pytest.mark.slow
def test_run_is_deterministic(tmp_path: Path) -> None:
    def positions() -> list[tuple[float, float]]:
        sim = Simulator.from_yaml(SCENARIO, seed=7, engine="mock")
        sim.record(tmp_path / "x", topdown_video=False)
        return [(float(a.pos[0]), float(a.pos[1])) for a in sim.world.agents.values()]

    assert positions() == positions()


def test_abandoned_object_is_left_behind(tmp_path: Path) -> None:
    sim = Simulator.from_yaml(SCENARIO, engine="mock")
    sim.record(tmp_path, topdown_video=False)
    assert sim.world.objects, "courier never dropped the bag"
    obj = next(iter(sim.world.objects.values()))
    owner = sim.world.roster[obj["owner"]]  # may have left via the exit
    assert np.linalg.norm(owner.pos - obj["pos"]) > 5.0

"""Lens distortion tests, validated against OpenCV.

Distortion matters here for a specific reason: the degradation pipeline bends the
rendered pixels, so if projection stays pinhole the ground truth ends up in a
different coordinate system from the image. Boxes then drift away from people
toward the frame edge — worst exactly where wide CCTV lenses put most of their
distortion — and any calibration or multi-view result built on that is wrong.

OpenCV is used as the reference implementation because it is what every
downstream consumer of the exported calibration will use.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from survsim.core.geometry import (
    apply_distortion,
    distortion_coeffs,
    extrinsics,
    intrinsics,
    project,
)
from survsim.core.models import Camera, Mount


def _cam(distortion: dict | None = None, model: str | None = None) -> Camera:
    return Camera(
        id="c", mount=Mount(pos=(0.0, 0.0, 4.0), yaw=0.0, pitch=-15.0),
        optics={"hfov_deg": 90, "distortion": distortion or {}, "model": model},
        capture={"res": (1280, 720)},
    )


def _world_points(n: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([
        rng.uniform(5.0, 40.0, n),
        rng.uniform(-18.0, 18.0, n),
        rng.uniform(0.0, 1.8, n),
    ], axis=1)


def test_zero_coefficients_leave_projection_unchanged() -> None:
    cam = _cam()
    pts = _world_points()
    with_d, _ = project(cam, pts, distort=True)
    without, _ = project(cam, pts, distort=False)
    assert np.allclose(with_d, without)


def test_brown_conrady_matches_opencv() -> None:
    """The authoritative check: our maths must agree with what consumers use."""
    coeffs = {"k1": -0.28, "k2": 0.07, "p1": 0.001, "p2": -0.0015, "k3": 0.002}
    cam = _cam(coeffs)
    pts = _world_points()

    ours, depth = project(cam, pts, distort=True)

    r, t = extrinsics(cam)
    _, dist = distortion_coeffs(cam)
    rvec, _ = cv2.Rodrigues(r)
    expected, _ = cv2.projectPoints(pts, rvec, t.reshape(3, 1), intrinsics(cam), dist)
    expected = expected.reshape(-1, 2)

    in_front = depth > 0
    assert in_front.sum() > 10
    assert np.allclose(ours[in_front], expected[in_front], atol=1e-6)


def test_fisheye_matches_opencv() -> None:
    coeffs = {"model": "fisheye", "k1": -0.05, "k2": 0.01, "k3": -0.002, "k4": 0.0005}
    cam = _cam(coeffs)
    pts = _world_points(seed=3)

    ours, depth = project(cam, pts, distort=True)

    r, t = extrinsics(cam)
    _, dist = distortion_coeffs(cam)
    rvec, _ = cv2.Rodrigues(r)
    expected = cv2.fisheye.projectPoints(
        pts.reshape(-1, 1, 3), rvec, t.reshape(3, 1), intrinsics(cam), dist)[0].reshape(-1, 2)

    in_front = depth > 0
    assert np.allclose(ours[in_front], expected[in_front], atol=1e-5)


def test_barrel_distortion_pulls_points_toward_the_centre() -> None:
    """Negative k1 is barrel distortion: the frame edge compresses inward."""
    cam = _cam({"k1": -0.3})
    edge = np.array([[6.0, 14.0, 1.0]])          # well off-axis
    distorted, _ = project(cam, edge, distort=True)
    pinhole, _ = project(cam, edge, distort=False)

    centre = np.array(cam.capture.res) / 2.0
    assert np.linalg.norm(distorted[0] - centre) < np.linalg.norm(pinhole[0] - centre)


def test_distortion_is_negligible_on_axis_and_large_at_the_edge() -> None:
    """Which is why a pinhole approximation looks fine until you check the corners."""
    cam = _cam({"k1": -0.28, "k2": 0.07})
    on_axis = np.array([[30.0, 0.0, 1.2]])
    off_axis = np.array([[8.0, 12.0, 1.2]])

    centre_shift = np.linalg.norm(project(cam, on_axis, True)[0]
                                  - project(cam, on_axis, False)[0])
    edge_shift = np.linalg.norm(project(cam, off_axis, True)[0]
                                - project(cam, off_axis, False)[0])
    assert centre_shift < 1.0
    assert edge_shift > 10 * max(centre_shift, 1e-6)


def test_coefficients_are_read_in_opencv_order() -> None:
    model, c = distortion_coeffs(_cam({"k1": 1, "k2": 2, "p1": 3, "p2": 4, "k3": 5}))
    assert model == "brown"
    assert list(c) == [1, 2, 3, 4, 5]       # k1 k2 p1 p2 k3


def test_fisheye_model_selected_by_either_key() -> None:
    assert distortion_coeffs(_cam({"model": "kannala_brandt", "k1": 0.1}))[0] == "fisheye"
    assert distortion_coeffs(_cam({"k1": 0.1}, model="fisheye"))[0] == "fisheye"


def test_apply_distortion_handles_the_optical_centre() -> None:
    """r = 0 must not divide by zero."""
    x, y = apply_distortion(_cam({"model": "fisheye", "k1": -0.05}),
                            np.array([0.0]), np.array([0.0]))
    assert np.isfinite(x).all() and np.isfinite(y).all()


def test_exported_calibration_states_the_coefficients_actually_used(tmp_path) -> None:
    """Consumers must not be silently handed a pinhole model."""
    import json

    from survsim.core.models import Environment, Population, Scenario, Zone
    from survsim.director.world import World
    from survsim.gt.writers import Recorder

    cam = _cam({"k1": -0.22, "k2": 0.05})
    env = Environment(id="e", zones=[Zone(id="zone.a", tag="public",
                                          polygon=[(-5, -5), (5, -5), (5, 5), (-5, 5)])])
    sc = Scenario(name="s", environment=env, cameras=[cam], population=Population())
    rec = Recorder(sc, tmp_path)
    rec.write(World(sc))

    calib = json.loads((tmp_path / "calib" / "c.json").read_text())
    assert calib["distortion"]["model"] == "brown"
    assert calib["distortion"]["coeffs"][0] == pytest.approx(-0.22)

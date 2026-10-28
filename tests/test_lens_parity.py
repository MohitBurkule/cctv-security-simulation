"""Render/ground-truth lens parity.

Ground truth is projected by geometry.project with the camera's distortion; the
pixels must go through the identical lens or every label drifts from what it
labels. The old degrade stage used its own coefficients and invented focal
length and moved corner pixels 26.6 px against a pinhole ground truth.
Idea from FruitCast fc_synthetic, which holds render-vs-numpy parity to 0.14 px.
"""

import copy

import cv2
import numpy as np
import pytest

from survsim.core.loader import load_scenario
from survsim.sensors.lens import distort_image, distortion_maps, has_distortion, rescale_points


def _camera(k1=-0.25, k2=0.05):
    cam = copy.deepcopy(load_scenario("configs/scenarios/smoke_mock.yaml").cameras[0])
    cam.optics.distortion = {"model": "brown", "k1": k1, "k2": k2}
    return cam


def _dot_centroid(img):
    ys, xs = np.nonzero(img > 32)
    w = img[ys, xs].astype(float)
    return np.array([(xs * w).sum() / w.sum(), (ys * w).sum() / w.sum()])


@pytest.mark.parametrize("uv_pin", [(80, 60), (640, 360), (1180, 650), (300, 600), (1000, 120)])
def test_a_rendered_point_lands_where_ground_truth_projects_it(uv_pin):
    cam = _camera()
    w, h = cam.capture.res
    img = np.zeros((h, w), np.uint8)
    cv2.circle(img, uv_pin, 4, 255, -1, lineType=cv2.LINE_AA)
    warped = distort_image(img, cam)

    # Where geometry says that pinhole ray lands once the lens is applied.
    k = __import__("survsim.core.geometry", fromlist=["intrinsics"]).intrinsics(cam)
    xn = (uv_pin[0] - k[0, 2]) / k[0, 0]
    yn = (uv_pin[1] - k[1, 2]) / k[1, 1]
    from survsim.core.geometry import apply_distortion
    xd, yd = apply_distortion(cam, np.array([xn]), np.array([yn]))
    expected = np.array([xd[0] * k[0, 0] + k[0, 2], yd[0] * k[1, 1] + k[1, 2]])

    got = _dot_centroid(warped)
    assert np.hypot(*(got - expected)) < 0.6, f"{got} vs {expected}"


def test_inversion_converges_to_sub_pixel_everywhere():
    _, _, residual = distortion_maps(_camera())
    assert residual < 1e-3


def test_a_camera_without_distortion_is_never_distorted():
    """The old stage distorted every frame with a profile default."""
    cam = _camera(0.0, 0.0)
    assert not has_distortion(cam)
    img = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    assert np.array_equal(distort_image(img, cam), img)


def test_wrong_resolution_is_refused_not_silently_mismatched():
    with pytest.raises(ValueError):
        distort_image(np.zeros((360, 640), np.uint8), _camera())


def test_points_follow_a_non_uniform_resize():
    pts = rescale_points(np.array([[1279.0, 719.0], [0.0, 0.0]]), (1280, 720), (704, 576))
    assert pts[0] == pytest.approx([703.0, 575.0], abs=0.51)
    assert pts[1] == pytest.approx([-0.5 + 0.5 * 704 / 1280, -0.5 + 0.5 * 576 / 720], abs=1e-9)

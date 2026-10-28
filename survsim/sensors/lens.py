"""Warp rendered pixels through the camera's lens — the same lens ground truth uses.

**Render/ground-truth parity is the whole point of this module.** Boxes, masks
and keypoints are projected by `core.geometry.project`, which applies the
camera's own distortion (`optics.distortion`). Pixels must go through the
identical model, or every label drifts away from what it labels, silently, by an
amount that grows toward the frame edge.

That is not hypothetical. The degradation chain used to apply a *second*,
hand-rolled distortion: coefficients taken from the degradation profile rather
than the camera (default k1 = -0.16), a focal length it invented
(`fx = max(w, h)`) rather than the camera's, and an inverse approximated by
negating the coefficients. With a camera declaring no distortion — so ground
truth was a plain pinhole — `cctv_day_720p` moved corner pixels by **26.6 px**.
A distant person 20 px tall near a corner had a ground-truth box that did not
overlap them at all.

The idea is lifted from FruitCast's `fc_synthetic`, where one calibrated lens
drives both the renderer and the numpy projection and a parity test holds them
to 0.14 px mean. Here the render is pinhole (Unreal's CineCamera), so the lens
is applied in post: for every *output* pixel we invert
`geometry.apply_distortion` to find which pinhole pixel it sees. Using that
function itself — rather than OpenCV's copy of the same maths — is what makes
parity hold by construction instead of by coincidence.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from ..core.geometry import apply_distortion, distortion_coeffs, intrinsics

# Fixed-point iterations for inverting the distortion. Brown-Conrady at CCTV
# strengths converges to sub-1e-6 normalised units in well under this.
_INVERT_ITERATIONS = 40


def has_distortion(camera: Any) -> bool:
    _, coeffs = distortion_coeffs(camera)
    return bool(np.any(coeffs))


def undistort_normalised(camera: Any, xd: np.ndarray, yd: np.ndarray,
                         iterations: int = _INVERT_ITERATIONS,
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Invert `geometry.apply_distortion` by fixed-point iteration.

    Solves distort(xu) = xd for xu. Each step moves the estimate by the current
    residual, which converges for the monotonic radial profiles real lenses
    have within their field of view.
    """
    xu, yu = xd.copy(), yd.copy()
    for _ in range(iterations):
        xdd, ydd = apply_distortion(camera, xu, yu)
        xu = xu - (xdd - xd)
        yu = yu - (ydd - yd)
    return xu, yu


def _camera_key(camera: Any) -> tuple:
    model, coeffs = distortion_coeffs(camera)
    return (camera.id, tuple(camera.capture.res), float(camera.hfov_deg), model,
            tuple(np.round(coeffs, 12)))


_MAP_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray, float]] = {}


def distortion_maps(camera: Any) -> tuple[np.ndarray, np.ndarray, float]:
    """cv2.remap maps: for each output (distorted) pixel, its pinhole source.

    Returns (map_x, map_y, max_residual_px); the residual reports how well the
    inversion converged, so a pathological lens is visible rather than silent.
    """
    key = _camera_key(camera)
    if key in _MAP_CACHE:
        return _MAP_CACHE[key]

    w, h = camera.capture.res
    k = intrinsics(camera)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]

    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    xd = (u - cx) / fx
    yd = (v - cy) / fy
    xu, yu = undistort_normalised(camera, xd, yd)

    # Residual check: pushing the solution forward must land on the pixel we
    # started from.
    xr, yr = apply_distortion(camera, xu, yu)
    residual = float(np.max(np.hypot((xr - xd) * fx, (yr - yd) * fy)))

    map_x = (xu * fx + cx).astype(np.float32)
    map_y = (yu * fy + cy).astype(np.float32)
    _MAP_CACHE[key] = (map_x, map_y, residual)
    return map_x, map_y, residual


def distort_image(img: np.ndarray, camera: Any) -> np.ndarray:
    """Apply the camera's lens to a pinhole render.

    The image must be at the camera's native capture resolution, because that
    is the resolution its intrinsics describe. A camera with no distortion is
    returned unchanged — never distorted by some default, which is exactly the
    mistake this module replaces.
    """
    if not has_distortion(camera):
        return img
    w, h = camera.capture.res
    if img.shape[1] != w or img.shape[0] != h:
        raise ValueError(
            f"image is {img.shape[1]}x{img.shape[0]} but camera {camera.id} "
            f"captures at {w}x{h}; distort at native resolution, then resize")
    map_x, map_y, _ = distortion_maps(camera)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def rescale_points(uv: np.ndarray, src_res: tuple[int, int],
                   dst_res: tuple[int, int]) -> np.ndarray:
    """Map pixel coordinates through a resize, e.g. a profile's sensor output.

    `cctv_legacy_sd` resizes 1280x720 to 704x576 — non-uniformly — so ground
    truth has to follow that too. Pixel centres are respected (the +0.5 terms),
    which matters by a half pixel at low resolution.
    """
    sx = dst_res[0] / float(src_res[0])
    sy = dst_res[1] / float(src_res[1])
    out = np.asarray(uv, dtype=np.float64).copy()
    out[..., 0] = (out[..., 0] + 0.5) * sx - 0.5
    out[..., 1] = (out[..., 1] + 0.5) * sy - 0.5
    return out

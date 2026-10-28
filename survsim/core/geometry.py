"""Camera geometry: intrinsics, extrinsics, projection.

Convention: right-handed world, Z up, metres. Camera looks down its local +X axis
before the mount rotation is applied, matching the yaw/pitch/roll in the YAML.
Image coordinates are pixels with the origin at the top-left.
"""

from __future__ import annotations

import numpy as np

from .models import Camera


def intrinsics(cam: Camera) -> np.ndarray:
    """Pinhole K from horizontal FOV and resolution."""
    w, h = cam.capture.res
    fx = (w / 2.0) / np.tan(np.radians(cam.hfov_deg) / 2.0)
    return np.array([[fx, 0.0, w / 2.0], [0.0, fx, h / 2.0], [0.0, 0.0, 1.0]])


def _rot_zyx(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Mount rotation. Negative pitch tilts the camera *down*, as CCTV specs write it.

    The Y-rotation is negated because a positive right-handed rotation about +Y
    lifts the forward axis; without this a `pitch: -20` camera looks at the sky.
    """
    y, p, r = np.radians([yaw_deg, -pitch_deg, roll_deg])
    cz, sz = np.cos(y), np.sin(y)
    cy, sy = np.cos(p), np.sin(p)
    cx, sx = np.cos(r), np.sin(r)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz @ ry @ rx


def extrinsics(cam: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, t) mapping world points into the camera frame: x_cam = R @ x_w + t."""
    r_mount = _rot_zyx(cam.mount.yaw, cam.mount.pitch, cam.mount.roll)
    # World axes (forward=+X, left=+Y, up=+Z) into optical axes (right, down, forward).
    to_optical = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    r = to_optical @ r_mount.T
    c = np.asarray(cam.mount.pos, dtype=float)
    return r, -r @ c


def distortion_coeffs(cam: Camera) -> tuple[str, np.ndarray]:
    """Return (model, coefficients) for the camera's lens.

    Real CCTV lenses are wide and bend straight lines noticeably. Ignoring that
    while the rendered pixels *are* distorted puts the ground truth in a different
    coordinate system from the image — boxes drift from people toward the frame
    edge, and any calibration or multi-view experiment built on it is wrong.
    """
    d = cam.optics.distortion or {}
    model = str(d.get("model", cam.optics.model or "brown")).lower()
    if model in ("kannala_brandt", "fisheye"):
        return "fisheye", np.array([float(d.get(k, 0.0)) for k in ("k1", "k2", "k3", "k4")])
    return "brown", np.array([float(d.get(k, 0.0)) for k in ("k1", "k2", "p1", "p2", "k3")])


def apply_distortion(cam: Camera, xn: np.ndarray, yn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distort normalised image coordinates in place of the ideal pinhole ones."""
    model, c = distortion_coeffs(cam)
    if not np.any(c):
        return xn, yn

    if model == "fisheye":
        k1, k2, k3, k4 = c
        r = np.sqrt(xn**2 + yn**2)
        theta = np.arctan(r)
        theta_d = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
        scale = np.where(r > 1e-9, theta_d / np.maximum(r, 1e-9), 1.0)
        return xn * scale, yn * scale

    k1, k2, p1, p2, k3 = c
    r2 = xn**2 + yn**2
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    x_t = 2 * p1 * xn * yn + p2 * (r2 + 2 * xn**2)
    y_t = p1 * (r2 + 2 * yn**2) + 2 * p2 * xn * yn
    return xn * radial + x_t, yn * radial + y_t


def project(cam: Camera, points_w: np.ndarray, distort: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Project Nx3 world points.

    Returns (Nx2 pixels, N depths in metres). Depth <= 0 means behind the camera;
    callers must treat those as not visible rather than trusting the pixel.

    Distortion is applied by default so projected ground truth lands where the
    rendered person actually appears. Pass `distort=False` for the ideal pinhole
    projection, which is what a calibration algorithm is trying to recover.
    """
    pts = np.atleast_2d(np.asarray(points_w, dtype=float))
    r, t = extrinsics(cam)
    cam_pts = pts @ r.T + t
    depth = cam_pts[:, 2]
    safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)

    xn = cam_pts[:, 0] / safe
    yn = cam_pts[:, 1] / safe
    if distort:
        xn, yn = apply_distortion(cam, xn, yn)

    k = intrinsics(cam)
    uv = np.stack([k[0, 0] * xn + k[0, 2], k[1, 1] * yn + k[1, 2]], axis=1)
    return uv, depth


def in_view(cam: Camera, points_w: np.ndarray, margin: int = 0) -> np.ndarray:
    """Boolean mask of points that land inside the image and in front of the camera."""
    uv, depth = project(cam, points_w)
    w, h = cam.capture.res
    return (
        (depth > 0)
        & (uv[:, 0] >= -margin)
        & (uv[:, 0] < w + margin)
        & (uv[:, 1] >= -margin)
        & (uv[:, 1] < h + margin)
    )


# Below roughly this pixel height a person is not reliably detectable, so counting
# them as "seen" overstates what a rig achieves. One threshold is used everywhere —
# coverage analysis and the "was this act observed?" test must not disagree.
MIN_USABLE_PX_HEIGHT = 20.0
# The image-space counterpart for a GT *box*: a person cut by the frame edge to a
# 3-6 px sliver keeps a full-height box yet shows nothing detectable
# (docs/BUG_gt_visibility_fences.md, AUDIT_H69 site 05). The width floor is
# capped at MIN_USABLE_WIDTH_FRAC x height, because an untruncated upright person
# only MIN_USABLE_PX_HEIGHT tall is itself ~6-7 px wide and must stay usable.
MIN_USABLE_PX_WIDTH = 8.0
MIN_USABLE_WIDTH_FRAC = 0.25


def box_usable(w: float, h: float) -> bool:
    """In-frame GT box usable: >= MIN_USABLE_PX_HEIGHT tall and at least
    min(MIN_USABLE_PX_WIDTH, MIN_USABLE_WIDTH_FRAC x h) wide."""
    w, h = float(w), float(h)
    return h >= MIN_USABLE_PX_HEIGHT and w >= min(MIN_USABLE_PX_WIDTH, MIN_USABLE_WIDTH_FRAC * h)


def usably_visible(
    cam: Camera,
    pos_w: np.ndarray,
    person_height_m: float = 1.7,
    min_px_height: float = MIN_USABLE_PX_HEIGHT,
) -> bool:
    """In frame, in front of the camera, and large enough in pixels to be usable."""
    pos = np.asarray(pos_w, dtype=float).reshape(1, 3)
    if not bool(in_view(cam, pos)[0]):
        return False
    return pixel_height(cam, pos.ravel(), person_height_m) >= min_px_height


def pixel_heights(cam: Camera, ground_pts: np.ndarray, person_height_m: float = 1.7) -> np.ndarray:
    """Pixel height for many ground points at once.

    The per-point version is fine for a handful of people but hopeless inside the
    camera-placement search, which evaluates ~800 candidate aims per camera over a
    grid of a thousand cells. Projecting all feet and all heads as two batched
    arrays turns minutes into milliseconds.
    """
    pts = np.atleast_2d(np.asarray(ground_pts, dtype=float))
    heads = pts + np.array([0.0, 0.0, person_height_m])
    uv_f, d_f = project(cam, pts)
    uv_h, d_h = project(cam, heads)
    valid = (d_f > 0) & (d_h > 0)
    return np.where(valid, np.abs(uv_f[:, 1] - uv_h[:, 1]), 0.0)


def pixel_height(cam: Camera, pos_w: np.ndarray, person_height_m: float) -> float:
    """Pixel height of an upright person at a world position.

    This is the number that decides whether a detector can see someone at all, and
    the one procurement writes into a spec (DORI). Returns 0.0 if behind the camera.
    """
    foot = np.asarray(pos_w, dtype=float)
    head = foot + np.array([0.0, 0.0, person_height_m])
    uv, depth = project(cam, np.vstack([foot, head]))
    if depth[0] <= 0 or depth[1] <= 0:
        return 0.0
    return float(abs(uv[0, 1] - uv[1, 1]))

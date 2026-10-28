"""Camera-rig analysis: coverage, overlap, blind spots, transit priors.

These are the numbers a site survey argues about and a multi-camera tracker
depends on. Overlap decides whether cross-view association is a geometry problem
or a re-identification problem; blind spots decide where someone can change their
jacket unseen; transit time gates which tracks in camera B could possibly be the
person who just left camera A.

Everything here is computed on the ground plane, which is the assumption real
deployments make and real sites violate — sloped and stepped ground is a known
error source, flagged rather than silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.geometry import MIN_USABLE_PX_HEIGHT, pixel_heights, project
from ..core.models import Camera, Environment

PERSON_H = 1.7
WALK_SPEED_MPS = 1.35


@dataclass
class CoverageGrid:
    """Boolean visibility per camera over a regular ground grid."""

    xs: np.ndarray
    ys: np.ndarray
    cell_m: float
    visible: dict[str, np.ndarray]  # camera_id -> (ny, nx) bool
    px_height: dict[str, np.ndarray]  # camera_id -> (ny, nx) float

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.ys), len(self.xs)

    def count(self) -> np.ndarray:
        """How many cameras see each cell."""
        if not self.visible:
            return np.zeros(self.shape, dtype=int)
        return np.sum(np.stack(list(self.visible.values())), axis=0)

    def blind_mask(self) -> np.ndarray:
        return self.count() == 0

    def blind_fraction(self) -> float:
        return float(self.blind_mask().mean())

    def points_for(self, mask: np.ndarray) -> np.ndarray:
        """World xy of the cells selected by `mask`."""
        gy, gx = np.nonzero(mask)
        return np.stack([self.xs[gx], self.ys[gy]], axis=1)


def bounds_of(env: Environment, pad: float = 5.0) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for z in env.zones:
        xs += [p[0] for p in z.polygon]
        ys += [p[1] for p in z.polygon]
    for w in list(env.spawn_points) + list(env.exits):
        xs.append(w.pos[0])
        ys.append(w.pos[1])
    if not xs:
        return -20.0, -20.0, 20.0, 20.0
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def coverage(
    cameras: list[Camera],
    env: Environment,
    cell_m: float = 1.0,
    min_px_height: float = MIN_USABLE_PX_HEIGHT,
) -> CoverageGrid:
    """Which cameras can see a standing person at each ground cell.

    `min_px_height` is a real threshold, not decoration: a person occupying fewer
    than roughly 20 pixels of height is below what general detectors reliably find,
    so counting those cells as "covered" overstates the rig.
    """
    x0, y0, x1, y1 = bounds_of(env)
    xs = np.arange(x0, x1 + cell_m, cell_m)
    ys = np.arange(y0, y1 + cell_m, cell_m)
    gx, gy = np.meshgrid(xs, ys)
    ground = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)

    visible: dict[str, np.ndarray] = {}
    heights: dict[str, np.ndarray] = {}
    for cam in cameras:
        uv, depth = project(cam, ground)
        w, h = cam.capture.res
        inside = (
            (depth > 0)
            & (uv[:, 0] >= 0) & (uv[:, 0] < w)
            & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        )
        px = pixel_heights(cam, ground, PERSON_H)
        visible[cam.id] = (inside & (px >= min_px_height)).reshape(gx.shape)
        heights[cam.id] = px.reshape(gx.shape)

    return CoverageGrid(xs=xs, ys=ys, cell_m=cell_m, visible=visible, px_height=heights)


def overlap_matrix(grid: CoverageGrid) -> tuple[list[str], np.ndarray]:
    """Fraction of camera i's footprint that camera j also sees (asymmetric)."""
    ids = sorted(grid.visible)
    n = len(ids)
    m = np.zeros((n, n))
    for i, a in enumerate(ids):
        area_a = grid.visible[a].sum()
        for j, b in enumerate(ids):
            if area_a == 0:
                continue
            m[i, j] = float((grid.visible[a] & grid.visible[b]).sum() / area_a)
    return ids, m


def transit_priors(
    cameras: list[Camera], grid: CoverageGrid, speed_mps: float = WALK_SPEED_MPS
) -> dict[tuple[str, str], dict[str, float]]:
    """Expected walking time between camera footprints.

    Used as a gate in non-overlapping association: a track reappearing sooner than
    the minimum, or far later than the maximum, is probably a different person.
    Straight-line distance is a lower bound — real routes bend around obstacles, so
    the window is deliberately generous on the upper side.
    """
    out: dict[tuple[str, str], dict[str, float]] = {}
    centroids: dict[str, np.ndarray | None] = {}
    for cam_id, mask in grid.visible.items():
        pts = grid.points_for(mask)
        centroids[cam_id] = pts.mean(axis=0) if len(pts) else None

    for a in grid.visible:
        for b in grid.visible:
            if a == b:
                continue
            ca, cb = centroids[a], centroids[b]
            if ca is None or cb is None:
                continue
            overlapped = bool((grid.visible[a] & grid.visible[b]).any())
            dist = float(np.linalg.norm(cb - ca))
            nominal = dist / speed_mps
            out[(a, b)] = {
                "distance_m": round(dist, 2),
                "t_min_s": 0.0 if overlapped else round(dist / 1.8, 2),
                "t_nominal_s": round(nominal, 2),
                "t_max_s": round(max(nominal * 3.0, nominal + 30.0), 2),
                "overlapping": overlapped,
            }
    return out


def blind_spots(grid: CoverageGrid, env: Environment) -> list[dict[str, Any]]:
    """Connected unobserved regions, largest first.

    A blind spot big enough to stand in is an appearance-change opportunity; that
    is the scenario the cross-epoch re-identification test is built around.
    """
    from scipy import ndimage

    labelled, n = ndimage.label(grid.blind_mask())
    regions: list[dict[str, Any]] = []
    for idx in range(1, n + 1):
        mask = labelled == idx
        area = float(mask.sum()) * grid.cell_m**2
        pts = grid.points_for(mask)
        centre = pts.mean(axis=0)
        regions.append(
            {
                "id": f"blind_{idx:02d}",
                "area_m2": round(area, 1),
                "centre": [round(float(centre[0]), 2), round(float(centre[1]), 2)],
                "large_enough_to_change_clothes": area >= 4.0,
            }
        )
    regions.sort(key=lambda r: -r["area_m2"])
    return regions


def summary(cameras: list[Camera], env: Environment, cell_m: float = 1.0) -> dict[str, Any]:
    grid = coverage(cameras, env, cell_m=cell_m)
    ids, m = overlap_matrix(grid)
    counts = grid.count()
    return {
        "cameras": ids,
        "cell_m": cell_m,
        "blind_fraction": round(grid.blind_fraction(), 4),
        "mean_cameras_per_cell": round(float(counts.mean()), 3),
        "cells_seen_by_2plus": round(float((counts >= 2).mean()), 4),
        "overlap_matrix": m.round(3).tolist(),
        "blind_spots": blind_spots(grid, env)[:10],
        "transit_priors": {f"{a}->{b}": v for (a, b), v in transit_priors(cameras, grid).items()},
    }

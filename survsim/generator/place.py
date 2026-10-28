"""Automatic camera placement.

Choosing where cameras go is the decision that determines what any downstream
algorithm can possibly achieve, and real installers do it by eye. Here it is an
explicit optimisation: greedily pick the mount and aim that adds the most
*useful* new coverage, where useful means a person there would be at least
`MIN_USABLE_PX_HEIGHT` tall — floor area a camera technically sees but cannot
resolve is not coverage.

Greedy maximum-coverage is deliberate. The problem is a submodular set cover, so
greedy is within (1 - 1/e) of optimal, runs in seconds, and is explainable to
whoever has to sign off the design — all of which matter more than the last few
per cent.
"""

from __future__ import annotations

import numpy as np

from ..core.geometry import MIN_USABLE_PX_HEIGHT, pixel_heights, project
from ..core.models import PTZ, Camera, Environment, Mount
from ..director.rig import bounds_of
from .site import mount_candidates
from .spec import CameraPlan

CANDIDATE_YAWS = np.arange(0, 360, 30.0)
CANDIDATE_PITCHES = (-15.0, -25.0, -35.0)


def _visible_cells(cam: Camera, ground: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    uv, depth = project(cam, ground)
    w, h = cam.capture.res
    inside = (
        (depth > 0)
        & (uv[:, 0] >= 0) & (uv[:, 0] < w)
        & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    )
    # Only count cells where a person would be resolvable.
    px = pixel_heights(cam, ground, 1.7)
    return (inside & (px >= MIN_USABLE_PX_HEIGHT)).reshape(shape)


def _make_camera(cam_id: str, pos, yaw, pitch, kind: str, plan: CameraPlan) -> Camera:
    hfov = {"fixed_bullet": 70.0, "dome": 95.0, "ptz": 60.0, "fisheye": 170.0}.get(kind, 75.0)
    cam = Camera(
        id=cam_id, type=kind,  # type: ignore[arg-type]
        mount=Mount(pos=(float(pos[0]), float(pos[1]), float(pos[2])),
                    yaw=float(yaw), pitch=float(pitch)),
        optics={"hfov_deg": hfov},
        capture={"res": list(plan.resolution), "fps": plan.fps},
    )
    if kind == "ptz":
        # A patrolling head means coverage is time-varying: the gaps are real and
        # an adversary who watches the cycle can walk through them.
        cam.ptz = PTZ(speed_dps=45.0,
                      patrol=[{"preset": "a", "dwell_s": 15.0},
                              {"preset": "b", "dwell_s": 15.0}])
        cam.presets = {"a": {"yaw": float(yaw) - 35, "pitch": float(pitch)},
                       "b": {"yaw": float(yaw) + 35, "pitch": float(pitch)}}
    return cam


def _kind_sequence(plan: CameraPlan) -> list[str]:
    """Expand the requested type mix into a concrete per-camera list."""
    kinds: list[str] = []
    total = sum(plan.mix.values()) or 1.0
    for kind, share in plan.mix.items():
        kinds += [kind] * round(plan.count * share / total)
    while len(kinds) < plan.count:
        kinds.append("fixed_bullet")
    return kinds[:plan.count]


def place_cameras(
    env: Environment,
    plan: CameraPlan,
    cell_m: float = 2.0,
    seed: int = 0,
    priority_tags: tuple[str, ...] = ("restricted", "gate", "perimeter_fence"),
) -> tuple[list[Camera], dict]:
    """Greedily place `plan.count` cameras to maximise weighted useful coverage.

    Cells inside high-value zones are weighted more heavily, so a limited camera
    budget is spent watching the gate and the restricted area rather than empty
    ground.
    """
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = bounds_of(env, pad=2.0)
    xs = np.arange(x0, x1 + cell_m, cell_m)
    ys = np.arange(y0, y1 + cell_m, cell_m)
    gx, gy = np.meshgrid(xs, ys)
    shape = gx.shape
    ground = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)

    weights = np.ones(shape)
    from shapely.geometry import Point, Polygon

    # Blind spots are a *design requirement*, not an accident. Real sites have
    # stairwells, corners and structures that cameras cannot see, and the
    # appearance-change scenario needs somewhere genuinely unobserved. A pure
    # coverage maximiser will happily cover them, so seeing a blind-spot cell
    # carries a penalty and the optimiser routes around it.
    blind = np.zeros(shape, dtype=bool)
    for zone in env.zones:
        poly = Polygon(zone.polygon)
        if zone.tag == "blindspot":
            target, flag = -6.0, True
        elif zone.tag in priority_tags:
            target, flag = 4.0, False
        else:
            continue
        for iy, yv in enumerate(ys):
            for ix, xv in enumerate(xs):
                if poly.contains(Point(float(xv), float(yv))):
                    weights[iy, ix] = target
                    blind[iy, ix] = flag

    # Placement reasons on a coarse grid; verification uses a finer one. Without a
    # margin a camera can clip blind-spot cells that only the finer grid sees, so
    # the constraint is dilated by one cell to stay conservative.
    if blind.any():
        from scipy import ndimage
        blind = ndimage.binary_dilation(blind, iterations=1)

    candidates = mount_candidates(env)
    if not candidates:
        candidates = [(float(v[0]), float(v[1])) for v in ground[:: max(1, len(ground) // 24), :2]]

    kinds = _kind_sequence(plan)
    covered = np.zeros(shape, dtype=bool)
    chosen: list[Camera] = []
    used_mounts: set[int] = set()

    for i, kind in enumerate(kinds):
        best = None          # best aim that respects the blind spots
        fallback = None      # best aim overall, used only if nothing qualifies
        for mi, (mx, my) in enumerate(candidates):
            if mi in used_mounts:
                continue
            height = float(rng.uniform(*plan.height_m))
            for yaw in CANDIDATE_YAWS:
                for pitch in CANDIDATE_PITCHES:
                    cam = _make_camera(f"cam_{i:02d}", (mx, my, height), yaw, pitch, kind, plan)
                    vis = _visible_cells(cam, ground, shape)
                    gain = float((weights.clip(min=0) * (vis & ~covered)).sum())
                    if fallback is None or gain > fallback[0]:
                        fallback = (gain, cam, vis, mi)
                    # Hard constraint, not a penalty: in a real building the
                    # stairwell or the corner is *behind a wall*, so no amount of
                    # coverage elsewhere justifies seeing into it. The mock
                    # coverage model has no wall geometry, so the constraint
                    # stands in for the occluder.
                    if blind.any() and (vis & blind).any():
                        continue
                    if best is None or gain > best[0]:
                        best = (gain, cam, vis, mi)
        best = best or fallback
        if best is None:
            break
        _, cam, vis, mi = best
        chosen.append(cam)
        covered |= vis
        used_mounts.add(mi)

    # Coverage is reported over the area we *intended* to watch: counting the
    # deliberately unobserved zones would understate a good rig.
    watchable = ~blind
    denom = float(weights[watchable].clip(min=0).sum()) or 1.0
    area = float(covered[watchable].mean()) if watchable.any() else 0.0
    report = {
        "n_cameras": len(chosen),
        "weighted_coverage": round(float((weights.clip(min=0) * covered)[watchable].sum() / denom), 4),
        "area_coverage": round(area, 4),
        "blind_fraction": round(float((~covered)[watchable].mean()) if watchable.any() else 0.0, 4),
        "blindspot_leak_cells": int((covered & blind).sum()),
        "meets_target": area >= plan.target_coverage,
    }
    return chosen, report


def verify_blind_spots(cameras: list[Camera], env: Environment, cell_m: float = 1.0) -> dict:
    """Confirm the site's declared blind spots are genuinely unobserved.

    A scenario that stages an appearance change in a "blind spot" that a camera
    can actually see is silently testing nothing — this check caught exactly that
    mistake once already.
    """
    from shapely.geometry import Point, Polygon

    from ..director.rig import coverage

    grid = coverage(cameras, env, cell_m=cell_m)
    counts = grid.count()
    out: dict[str, dict] = {}
    for zone in env.zones:
        if zone.tag != "blindspot":
            continue
        poly = Polygon(zone.polygon)
        seen, total = 0, 0
        for iy, yv in enumerate(grid.ys):
            for ix, xv in enumerate(grid.xs):
                if poly.contains(Point(float(xv), float(yv))):
                    total += 1
                    seen += int(counts[iy, ix] > 0)
        out[zone.id] = {
            "cells": total,
            "observed_cells": seen,
            "truly_blind": total > 0 and seen == 0,
        }
    return out

"""Static-occluder (fence / wall / hedge) visibility for GT, without a re-render.

docs/BUG_gt_visibility_fences.md. Two gaps let a person standing behind a
palisade keep GT visibility 0.8-1.0:

* The Director/mock visibility (`visible_frac`, MOT column 9) paints only the
  scenario's own props. The render's site dressing (`set/env_plan.json`
  items: `fence_run`, `wall_run`, `hedge`) never reaches it.
* The ID-mask pass is CustomStencil, and only people write custom depth, so a
  fence cannot remove person pixels from it [A] (export fix in
  scripts/ue/survsim_scene.py, ID_MASK_DEPTH_TEST).

This module rebuilds those occluders as the scene builder does
(scripts/ue/survsim_scene.py, `fence_run` branch: 7 cm pales every 20 cm plus a
rail at 0.8 h), rasterises them at SS x supersampling into a nearest-depth map
through the live camera pose (per frame, from gt/camera_status.jsonl: a slewed
PTZ or tilt-tampered camera no longer matches calib/; a PTZ frame with no
recorded pose is flagged `pose_unknown`, never given the configured pose), and measures what fraction of each person's
silhouette lies behind a nearer occluder. It also filters an existing ID mask
(`filter_idmap`) so mask-derived boxes lose the pixels a fence covers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..core.geometry import box_usable

SS = 4                          # supersampling: pales are ~1 px wide at 640x360
SLAB_CM = 25.0                  # long panels are cut into slabs so depth stays local
KINDS = ("fence_run", "wall_run", "hedge")
STATIC_MIN = 0.10               # static cover below this is not called an occluder


def _box_prims(cx, cy, zc, w, d, h, yaw) -> list[np.ndarray]:
    """UE-cm box -> list of 8x3 SI corner arrays, cut into <= SLAB_CM slabs along w."""
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    n = max(1, int(np.ceil(w / SLAB_CM)))
    out = []
    for i in range(n):
        u0, u1 = -w / 2 + i * w / n, -w / 2 + (i + 1) * w / n
        pts = []
        for u in (u0, u1):
            for v in (-d / 2, d / 2):
                for z in (zc - h / 2, zc + h / 2):
                    pts.append([cx + c * u - s * v, cy + s * u + c * v, z])
        p = np.asarray(pts, float) / 100.0
        p[:, 1] *= -1.0                         # UE -> SI: cm -> m, negate Y
        out.append(p)
    return out


def occluder_prims(items: list[dict[str, Any]], kinds=KINDS) -> list[np.ndarray]:
    """env_plan items -> SI corner sets, mirroring survsim_scene's geometry."""
    prims: list[np.ndarray] = []
    for it in items:
        k = it.get("kind")
        if k not in kinds:
            continue
        x, y, yaw = float(it["x"]), float(it["y"]), float(it.get("yaw", 0.0))
        w, d, h = (float(v) for v in it["size"])
        c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        if k == "fence_run" and it.get("style", "mesh") == "palisade":
            for j in range(int(w // 20)):
                u = -w / 2 + 10 + j * 20
                prims += _box_prims(x + c * u, y + s * u, h / 2, 7, 3, h, yaw)
            prims += _box_prims(x, y, h * 0.8, w, 6, 6, yaw)
        elif k == "fence_run":
            prims += _box_prims(x, y, h / 2, w, 2, h, yaw)
        elif k == "wall_run":
            prims += _box_prims(x, y, (h + 8) / 2, w, max(d, 20), h + 8, yaw)
        else:
            prims += _box_prims(x, y, h / 2, w, d, h, yaw)
    return prims


class PoseUnknown(ValueError):
    """A PTZ frame has no recorded live pose; the configured one would be wrong."""


_TO_OPTICAL = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def is_ptz(cal: dict[str, Any]) -> bool:
    return str(cal.get("type", "")).lower().startswith("ptz")


def pose_from_status(cal: dict[str, Any], yaw: float, pitch: float, hfov_deg: float):
    """(K, R, t) for the live yaw/pitch/hfov of a camera_status row (CLAUDE.md
    principle 2). Position, resolution and roll come from the calib file; roll is
    recovered from the calibrated R (the camera director never changes it)."""
    from ..core.geometry import _rot_zyx
    w, h = cal["resolution"]
    fx = (w / 2.0) / np.tan(np.radians(float(hfov_deg)) / 2.0)
    K = np.array([[fx, 0.0, w / 2.0], [0.0, fx, h / 2.0], [0.0, 0.0, 1.0]])
    m = (_TO_OPTICAL.T @ np.asarray(cal["R"], float)).T
    roll = float(np.degrees(np.arctan2(m[2, 1], m[2, 2])))
    R = _TO_OPTICAL @ _rot_zyx(float(yaw), float(pitch), roll).T
    return K, R, -R @ np.asarray(cal["position_world"], float)


def frame_poses(cal: dict[str, Any], status_rows: list[dict[str, Any]], camera: str,
                frames, fps: float | None = None) -> dict[int, tuple | None]:
    """frame -> pose key per MOT frame: ("live", yaw, pitch, hfov) from
    camera_status (frame = round(t * fps), as blind_frames), "calib" for a fixed
    camera with no status row, None (pose_unknown) for a PTZ frame with none.
    A fixed camera's status row is used too: a tilt tamper moves it."""
    fps = float(fps or cal.get("fps") or 10.0)
    live = {}
    for r in status_rows:
        if r.get("camera") == camera and r.get("yaw") is not None:
            live[round(float(r["t"]) * fps)] = ("live", float(r["yaw"]), float(r["pitch"]),
                                                float(r.get("hfov_deg", cal["hfov_deg"])))
    ptz = is_ptz(cal)
    return {int(f): live.get(int(f), None if ptz else "calib") for f in frames}


def pose_arrays(cal: dict[str, Any], key) -> tuple:
    if key == "calib":
        return cal["K"], cal["R"], cal["t"]
    return pose_from_status(cal, *key[1:])


def live_static(cal: dict[str, Any], prims: list[np.ndarray], world, tids: dict[str, int],
                status_rows: list[dict[str, Any]], camera: str, frames,
                strict: bool = False) -> dict[str, Any]:
    """Per-frame depth maps + person depths through the live camera pose.

    Returns {"zbuf": {frame: zbuf | None}, "depth": {(frame, tid): m},
    "res", "pose_unknown": [frames]}. strict=True raises PoseUnknown instead of
    flagging a PTZ frame with no recorded pose."""
    res = tuple(cal["resolution"])
    poses = frame_poses(cal, status_rows, camera, frames)
    unknown = sorted(f for f, k in poses.items() if k is None)
    if unknown and strict:
        raise PoseUnknown(f"{camera}: {len(unknown)} PTZ frames without a recorded pose "
                          f"(first {unknown[0]})")
    by_key: dict[Any, np.ndarray] = {}
    zb: dict[int, np.ndarray | None] = {}
    depth: dict[tuple[int, int], float] = {}
    for key in {k for k in poses.values() if k is not None}:
        K, R, t = pose_arrays(cal, key)
        by_key[key] = depth_map(K, R, t, res, prims).astype(np.float32)
        fs = [f for f, k in poses.items() if k == key]
        w = world[world["frame"].isin(fs)]
        depth.update(person_depths(w, tids, R, t))
    for f, k in poses.items():
        zb[f] = by_key.get(k) if k is not None else None
    return {"zbuf": zb, "depth": depth, "res": res, "pose_unknown": unknown,
            "poses": len(by_key)}


def depth_map(K, R, t, res: tuple[int, int], prims: list[np.ndarray], ss: int = SS) -> np.ndarray:
    """Nearest static-occluder camera depth (m) per supersampled pixel; inf = none."""
    import cv2
    W, H = int(res[0]) * ss, int(res[1]) * ss
    K = np.asarray(K, float).copy()
    K[:2] *= ss
    R, t = np.asarray(R, float), np.asarray(t, float)
    zbuf = np.full((H, W), np.inf, np.float32)
    for p in prims:
        cam = p @ R.T + t
        if (cam[:, 2] <= 0.05).any():
            continue
        uv = (cam @ K.T)[:, :2] / cam[:, 2:3]
        if uv[:, 0].max() < 0 or uv[:, 1].max() < 0 or uv[:, 0].min() >= W or uv[:, 1].min() >= H:
            continue
        hull = cv2.convexHull(np.round(uv).astype(np.int32))
        x0, y0 = max(int(uv[:, 0].min()) - 1, 0), max(int(uv[:, 1].min()) - 1, 0)
        x1, y1 = min(int(uv[:, 0].max()) + 2, W), min(int(uv[:, 1].max()) + 2, H)
        if x1 <= x0 or y1 <= y0:
            continue
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(m, hull - np.array([x0, y0], np.int32), 1)
        z = float(cam[:, 2].min())
        roi = zbuf[y0:y1, x0:x1]
        roi[(m > 0) & (roi > z)] = z
    return zbuf


def silhouette(shape: tuple[int, int], bbox, ss: int = SS) -> np.ndarray:
    """The mock backend's body silhouette (head ellipse + trapezium) in the SS grid."""
    from ..backends.mock.backend import MockBackend
    buf = np.full(shape, -1, np.int32)
    x, y, w, h = (float(v) * ss for v in bbox)
    MockBackend._paint_person(buf, x, y, w, h, 1)
    return buf == 1


def static_cover(zbuf: np.ndarray, bbox, person_depth: float, ss: int = SS) -> float:
    """Fraction of the person's in-frame silhouette behind a nearer static occluder."""
    sil = silhouette(zbuf.shape, bbox, ss)
    n = int(sil.sum())
    if n == 0:
        return 0.0
    return float((zbuf[sil] < person_depth - 0.05).sum()) / n


def filter_idmap(idmap: np.ndarray, zbuf: np.ndarray, depth_by_id: dict[int, float],
                 ss: int = SS) -> np.ndarray:
    """Post-hoc mask filter for sets rendered before ID_MASK_DEPTH_TEST: zero a
    person's ID pixels where a static occluder is nearer than that person.
    A native pixel is dropped when most of its SS x SS subsamples are covered."""
    lab = np.asarray(idmap).copy()
    H, W = lab.shape
    z = zbuf[:H * ss, :W * ss].reshape(H, ss, W, ss)
    for k, dep in depth_by_id.items():
        on = lab == k
        if not on.any():
            continue
        cov = (z < dep - 0.05).mean(axis=(1, 3))
        lab[on & (cov >= 0.5)] = 0
    return lab


def id_mask_depth_tested(set_dir: Path) -> bool:
    """True when the set's ID mask was rendered with ID_MASK_DEPTH_TEST, read from
    the `id_mask_depth_test` key the editor records in render_result.json
    (scripts/ue/render_run.py). Absent key or file -> False (assume not tested:
    every set rendered before the key existed predates the depth test)."""
    p = Path(set_dir) / "render_result.json"
    if not p.is_file():
        return False
    try:
        return bool(json.loads(p.read_text()).get("id_mask_depth_test", False))
    except (ValueError, OSError):
        return False


def correct_rows(rows: list[list[float]], zbuf: np.ndarray, depth: dict[tuple[int, int], float],
                 res: tuple[int, int], ss: int = SS, depth_tested: bool = False) -> list[dict[str, Any]]:
    """MOT rows (frame, id, x, y, w, h, conf, cls, vis) -> corrected records.

    vis_new = vis_old x (1 - static_cover) (person and static occlusion treated
    as independent); occluder = static when the static deficit is >= STATIC_MIN
    and larger than the person/edge deficit already in vis_old; usable per
    geometry.box_usable on the in-frame box.

    zbuf may be one array (fixed camera) or {frame: array | None} from
    live_static; a None frame (PTZ, no recorded pose) keeps vis_old and gets
    occluder "pose_unknown" rather than a cover computed from the wrong view.

    depth_tested=True: vis_old is a render visibility from a depth-tested ID
    mask, whose pixel count already lost what the fence covers, so multiplying
    by (1 - static_cover) again would double count (docs/BUG_night_visibility_zero.md).
    static_cover and occluder are still reported; vis stays vis_old."""
    W, H = res
    out = []
    for r in rows:
        f, tid, x, y, w, h = int(r[0]), int(r[1]), *map(float, r[2:6])
        vis = float(r[8]) if len(r) > 8 else 1.0
        zf = zbuf.get(f) if isinstance(zbuf, dict) else zbuf
        if zf is None:
            vx0, vy0, vx1, vy1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
            out.append({"frame": f, "track_id": tid, "bbox": [x, y, w, h], "vis_old": vis,
                        "static_cover": None, "vis": round(vis, 3), "occluder": "pose_unknown",
                        "usable": bool(box_usable(max(vx1 - vx0, 0), max(vy1 - vy0, 0))),
                        "conf": float(r[6]), "cls": int(r[7])})
            continue
        dep = depth.get((f, tid))
        sc = static_cover(zf, (x, y, w, h), dep, ss) if dep is not None else 0.0
        vx0, vy0, vx1, vy1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        usable = box_usable(max(vx1 - vx0, 0), max(vy1 - vy0, 0))
        occ = "static" if sc >= STATIC_MIN and sc > 1.0 - vis else ("person_or_edge" if vis < 0.9 else "none")
        out.append({"frame": f, "track_id": tid, "bbox": [x, y, w, h], "vis_old": vis,
                    "static_cover": round(sc, 3),
                    "vis": round(vis if depth_tested else vis * (1.0 - sc), 3),
                    "occluder": occ, "usable": bool(usable), "conf": float(r[6]), "cls": int(r[7])})
    return out


def person_depths(world, tids: dict[str, int], R, t, z_m: float = 0.9) -> dict[tuple[int, int], float]:
    """(frame, track_id) -> camera depth (m) of the person's mid-body point."""
    R, t = np.asarray(R, float), np.asarray(t, float)
    w = world[world["uid"].isin(list(tids))]
    p = np.c_[w["x"].to_numpy(float), w["y"].to_numpy(float), np.full(len(w), z_m)]
    z = (p @ R.T + t)[:, 2]
    return {(int(f), int(tids[u])): float(d) for f, u, d in zip(w["frame"], w["uid"], z)}


def correct_site(site_dir: Path, out_dir: Path, camera: str = "cam_00") -> dict[str, Any]:
    """Write <out_dir>/mot_<camera>.txt (vis corrected; unusable rows conf 0) and
    static_occlusion.parquet for a rendered showcase site. Inputs are read-only."""
    import pandas as pd
    site_dir, out_dir = Path(site_dir), Path(out_dir)
    cal = json.loads((site_dir / "run" / "calib" / f"{camera}.json").read_text())
    env = json.loads((site_dir / "set" / "env_plan.json").read_text())
    res = tuple(cal["resolution"])
    tids = json.loads((site_dir / "run" / "meta" / "track_ids.json").read_text())
    rows = [ln.split(",") for ln in (site_dir / "run" / "gt" / f"mot_{camera}.txt").read_text().splitlines()
            if ln.strip()]
    rows = [[float(v) for v in r] for r in rows]
    st = site_dir / "run" / "gt" / "camera_status.jsonl"
    status = [json.loads(ln) for ln in st.read_text().splitlines() if ln.strip()] if st.is_file() else []
    live = live_static(cal, occluder_prims(env["items"]),
                       pd.read_parquet(site_dir / "run" / "gt" / "world.parquet"), tids, status,
                       camera, {int(r[0]) for r in rows})
    recs = correct_rows(rows, live["zbuf"], live["depth"], res,
                        depth_tested=id_mask_depth_tested(site_dir / "set"))
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"{r['frame']},{r['track_id']},{r['bbox'][0]:.2f},{r['bbox'][1]:.2f},{r['bbox'][2]:.2f},"
             f"{r['bbox'][3]:.2f},{r['conf'] if r['usable'] else 0.0:g},{r['cls']},{r['vis']:.3f}" for r in recs]
    (out_dir / f"mot_{camera}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    pd.DataFrame(recs).to_parquet(out_dir / "static_occlusion.parquet", index=False)
    return {"rows": len(recs), "static": sum(r["occluder"] == "static" for r in recs),
            "unusable": sum(not r["usable"] for r in recs),
            "pose_unknown": sum(r["occluder"] == "pose_unknown" for r in recs),
            "vis_old_median": float(np.median([r["vis_old"] for r in recs])) if recs else None,
            "vis_median": float(np.median([r["vis"] for r in recs])) if recs else None}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("site_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--camera", default="cam_00")
    a = ap.parse_args(argv)
    print(json.dumps(correct_site(a.site_dir, a.out, a.camera), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

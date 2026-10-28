"""Person boxes from the rendered per-person ID mask (CLAUDE.md principle 1).

The Unreal render writes a second pass per camera: every person's components
carry CustomDepth stencil k (scripts/ue/survsim_scene.tag_people_stencil) and a
post-process material writes k as base-6 digits in R, G, B at level/5
(`ID_MASK_CODE`). This module decodes that pass and derives each person's box
from the pixels that are actually theirs in that frame: tight to the rendered
silhouette, truncated by occluders and by the frame edge.

Why this exists: before 2026-09-26 the rendered sets reused the mock backend's
MOT file, whose box is the projected hull of an upright 3-D box
(height_m x shoulder_w x shoulder_w*2/3) at the Director's position. Seen from
a camera above head height, the far top edge of that box projects above the
head, so every box was loose at the top (docs/research/32 §H6(c): detector
head +7 % of box height below GT); the hull is wider than a body, and the
rendered character's own stature / root offset differs from the Director's.

The projected box is kept as a fallback: gt/projected/mot_<cam>.txt holds the
old file unchanged, and gt/boxes_mask.parquet carries both boxes per row.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

LEVELS = 6
MAX_ID = LEVELS ** 3 - 1          # 215
MIN_PIXELS = 4                    # fewer is decode noise, not a person


def _srgb(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


_LIN = np.round(255.0 * np.arange(LEVELS) / (LEVELS - 1))
_SRGB = np.round(255.0 * _srgb(np.arange(LEVELS) / (LEVELS - 1)))


def encode_ids(ids: np.ndarray, srgb: bool = False) -> np.ndarray:
    """Inverse of decode_ids (tests and synthetic fixtures)."""
    ids = np.asarray(ids, dtype=int)
    lv = _SRGB if srgb else _LIN
    digits = np.stack([ids % LEVELS, (ids // LEVELS) % LEVELS, ids // (LEVELS * LEVELS)], -1)
    return lv[digits].astype(np.uint8)


def _lut(lv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per uint8 value: distance to the nearest level and that level's index."""
    d = np.abs(np.arange(256, dtype=float)[:, None] - lv[None, :])
    return d.min(-1), d.argmin(-1)


_LUTS = [_lut(_LIN), _lut(_SRGB)]


def decode_ids(rgb: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 ID-mask image -> HxW int stencil IDs (0 = background).

    Whether the PNG writer applied an sRGB curve is not assumed: both level
    tables are tried and the one the non-black pixels sit closer to wins.

    Lookup tables over the 256 channel values, not an H x W x 3 x L distance
    array: the per-channel nearest level and its error depend on the value
    only. The errors are integers, so the table choice is exactly the one the
    per-pixel mean made (the first table wins a tie, as before).
    """
    px = np.asarray(rgb)[..., :3]
    if px.dtype != np.uint8:
        px = np.clip(np.round(px), 0, 255).astype(np.uint8)
    packed = (px[..., 0].astype(np.uint32) | (px[..., 1].astype(np.uint32) << 8)
              | (px[..., 2].astype(np.uint32) << 16))
    fg = packed != 0
    out = np.zeros(packed.shape, dtype=np.int64)
    if not fg.any():
        return out
    # Background is most of the frame: decode each distinct foreground colour once.
    codes, inv, cnt = np.unique(packed[fg], return_inverse=True, return_counts=True)
    ch = np.stack([codes & 255, (codes >> 8) & 255, (codes >> 16) & 255], -1)   # U x 3
    hist = np.bincount(ch.ravel(), weights=np.repeat(cnt, 3), minlength=256)
    best = None
    for err_t, arg_t in _LUTS:
        err = float((hist * err_t).sum())
        if best is None or err < best[0]:
            best = (err, arg_t)
    digits = best[1][ch]
    ids = digits[:, 0] + LEVELS * digits[:, 1] + LEVELS * LEVELS * digits[:, 2]
    out[fg] = ids[inv.ravel()]
    return out


def boxes_from_ids(idmap: np.ndarray, known: set[int] | None = None,
                   min_pixels: int = MIN_PIXELS) -> dict[int, dict[str, float]]:
    """stencil id -> {bbox (x, y, w, h), pixels} for every id present in the map.

    One nonzero pass over the label image and a sort by id, not one
    full-image comparison per person."""
    lab = np.asarray(idmap)
    ys, xs = np.nonzero(lab > 0)
    out: dict[int, dict[str, Any]] = {}
    if not len(ys):
        return out
    k = lab[ys, xs]
    order = np.argsort(k, kind="stable")
    k, ys, xs = k[order], ys[order], xs[order]
    starts = np.r_[0, np.nonzero(np.diff(k))[0] + 1]
    counts = np.diff(np.r_[starts, len(k)])
    x0, x1 = np.minimum.reduceat(xs, starts), np.maximum.reduceat(xs, starts) + 1
    y0, y1 = np.minimum.reduceat(ys, starts), np.maximum.reduceat(ys, starts) + 1
    for i, kk in enumerate(k[starts].tolist()):
        n = int(counts[i])
        if n < min_pixels or (known is not None and kk not in known):
            continue
        out[kk] = {"bbox": (float(x0[i]), float(y0[i]), float(x1[i] - x0[i]),
                            float(y1[i] - y0[i])), "pixels": n}
    return out


def _read_mot(path: Path) -> dict[int, list[list[str]]]:
    out: dict[int, list] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip():
                f = line.split(",")
                out.setdefault(int(f[0]), []).append(f)
    return out


def _camera_boxes(job: tuple) -> tuple[list[str], list[dict], dict[str, int]]:
    """Mask boxes for one camera: (MOT lines sorted, parquet rows, counts)."""
    from PIL import Image
    mot, cam_mask, stencil_ids, tids, *rest = job
    blind = rest[0] if rest else set()
    # static: {"zbuf", "depth": {(frame, tid): m}, "res", "filter": bool} or None
    # (docs/BUG_gt_visibility_fences.md). filter=True for masks rendered without
    # ID_MASK_DEPTH_TEST, whose person pixels show through fences.
    static = rest[1] if len(rest) > 1 else None
    cam = mot.stem[len("mot_"):]
    by_k = {int(k): u for u, k in stencil_ids.items()}
    uid_of = {t: u for u, t in reversed(list(tids.items()))}   # first uid wins, as next() did
    proj = _read_mot(mot)
    frames = sorted(Path(cam_mask).glob("*.png"))
    lines, rows_pq, n_mask, n_hidden, n_added = [], [], 0, 0, 0
    known = set(by_k)
    for fp in frames:
        f = int(fp.stem)
        ident = decode_ids(np.asarray(Image.open(fp).convert("RGB")))
        if static and static.get("filter"):
            from .static_occlusion import filter_idmap
            zf = static["zbuf"].get(f) if isinstance(static["zbuf"], dict) else static["zbuf"]
            dep = {k: static["depth"][(f, tids[u])] for k, u in by_k.items()
                   if u in tids and (f, tids[u]) in static["depth"]}
            if zf is not None:      # pose_unknown PTZ frame: no filter from a wrong view
                ident = filter_idmap(ident, zf, dep)
        boxes = boxes_from_ids(ident, known)
        seen = set()
        for r in proj.get(f, []):
            tid = int(r[1])
            uid = uid_of.get(tid)
            k = stencil_ids.get(uid) if uid else None
            pb = tuple(float(v) for v in r[2:6])
            b = boxes.get(k) if k else None
            occ = {}
            if static:
                from .static_occlusion import correct_rows
                occ = correct_rows([[float(v) for v in r]], static["zbuf"], static["depth"],
                                   static["res"],
                                   depth_tested=bool(static.get("depth_tested", False)))[0]
            vis = occ.get("vis", float(r[8]))
            if b:
                x, y, w, h = b["bbox"]
                lines.append(f"{f},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,{vis:.3f}")
                n_mask += 1
                seen.add(k)
            else:
                x, y, w, h = pb
                lines.append(f"{f},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},0,1,0.000")
                n_hidden += 1
            rows_pq.append({"frame": f, "camera": cam, "uid": uid, "track_id": tid,
                            "mask_bbox": list(b["bbox"]) if b else None,
                            "mask_pixels": b["pixels"] if b else 0,
                            "projected_bbox": list(pb), "source": "mask" if b else "projected",
                            "static_cover": occ.get("static_cover"), "occluder": occ.get("occluder")})
        for k, b in boxes.items():
            # A blinded (covered/sprayed) camera emits nothing (principle 2): the
            # render has no tamper, so its mask still shows people there.
            if k in seen or f in blind:
                continue
            uid = by_k[k]
            tid = tids.get(uid)
            if tid is None:
                continue
            x, y, w, h = b["bbox"]
            lines.append(f"{f},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1.000")
            n_added += 1
            rows_pq.append({"frame": f, "camera": cam, "uid": uid, "track_id": tid,
                            "mask_bbox": list(b["bbox"]), "mask_pixels": b["pixels"],
                            "projected_bbox": None, "source": "mask_only",
                            "static_cover": None, "occluder": None})
    lines.sort(key=lambda s: (int(s.split(",")[0]), int(s.split(",")[1])))
    return lines, rows_pq, {"frames": len(frames), "mask_rows": n_mask,
                            "no_pixels": n_hidden, "mask_only": n_added,
                            "pose_unknown": sum(r["occluder"] == "pose_unknown" for r in rows_pq)}


def blind_frames(status: Path, scenario: Path) -> dict[str, set[int]]:
    """MOT frame numbers (round(t * fps)) at which each camera is blind."""
    out: dict[str, set[int]] = {}
    if not status.is_file():
        return out
    fps = float(json.loads(scenario.read_text()).get("fps", 10)) if scenario.is_file() else 10.0
    for line in status.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("blind"):
                out.setdefault(r["camera"], set()).add(round(float(r["t"]) * fps))
    return out


def _static_inputs(set_dir: Path, run_dir: Path, cam: str, tids: dict[str, int],
                   depth_tested: bool) -> dict | None:
    """Fence/wall/hedge depth maps + person depths for one camera, or None when
    the set has no env_plan.json (no site dressing) or the camera no calib.
    Per frame through the live pose in gt/camera_status.jsonl (principle 2);
    PTZ frames with no recorded pose get zbuf None -> occluder "pose_unknown"."""
    env_p, cal_p = set_dir / "env_plan.json", run_dir / "calib" / f"{cam}.json"
    world_p = run_dir / "gt" / "world.parquet"
    if not (env_p.is_file() and cal_p.is_file() and world_p.is_file()):
        return None
    import pandas as pd

    from . import static_occlusion as so
    cal = json.loads(cal_p.read_text())
    prims = so.occluder_prims(json.loads(env_p.read_text()).get("items", []))
    if not prims:
        return None
    st = run_dir / "gt" / "camera_status.jsonl"
    status = [json.loads(ln) for ln in st.read_text().splitlines() if ln.strip()] if st.is_file() else []
    mot = run_dir / "gt" / f"mot_{cam}.txt"
    frames = {int(ln.split(",")[0]) for ln in mot.read_text().splitlines() if ln.strip()} \
        if mot.is_file() else set()
    mask = set_dir / "mask" / cam
    if mask.is_dir():
        frames |= {int(p.stem) for p in mask.glob("*.png") if p.stem.isdigit()}
    out = so.live_static(cal, prims, pd.read_parquet(world_p), tids, status, cam, sorted(frames))
    out["filter"] = not depth_tested
    out["depth_tested"] = bool(depth_tested)
    return out


def build_tight_gt(set_dir: Path, run_dir: Path, stencil_ids: dict[str, int],
                   mask_dir: Path | None = None, workers: int = 4,
                   depth_tested: bool = True) -> dict[str, Any]:
    """Write <set_dir>/gt with mask-derived MOT boxes; returns summary counts.

    Rows: every person the projected GT lists (so ids / visibility columns keep
    their meaning) gets the mask box when it has >= MIN_PIXELS in that frame;
    a person with no rendered pixels keeps the projected box with conf 0 and
    visibility 0 (MOT "ignore"), since nothing of them is on screen. A person
    with pixels but no projected row (the render and the Director disagree on
    the frame edge) is added with visibility 1.

    Static occluders (set_dir/env_plan.json fences, walls, hedges): column 9
    becomes vis x (1 - static_cover) and boxes_mask.parquet gains
    `static_cover` / `occluder` (= "static" as in survsim.gt.static_occlusion).
    depth_tested=False (masks rendered without ID_MASK_DEPTH_TEST) also zeroes
    ID pixels behind a nearer occluder (filter_idmap) before boxing.
    """
    import pandas as pd

    set_dir, run_dir = Path(set_dir), Path(run_dir)
    mask_dir = Path(mask_dir or set_dir / "mask")
    src_gt = (run_dir / "gt").resolve()
    gt = set_dir / "gt"
    if gt.is_symlink():
        gt.unlink()
    gt.mkdir(parents=True, exist_ok=True)
    for p in src_gt.iterdir():
        if p.is_file() and not p.name.startswith("mot_"):
            shutil.copy2(p, gt / p.name)
    (gt / "projected").mkdir(exist_ok=True)
    tids = json.loads((run_dir / "meta" / "track_ids.json").read_text())
    rows_pq = []
    summary: dict[str, Any] = {"cameras": {}}
    mots = sorted(src_gt.glob("mot_*.txt"))
    for mot in mots:
        shutil.copy2(mot, gt / "projected" / mot.name)
    blind = blind_frames(src_gt / "camera_status.jsonl", run_dir / "meta" / "scenario.resolved.json")
    jobs = [(mot, mask_dir / mot.stem[len("mot_"):], stencil_ids, tids,
             blind.get(mot.stem[len("mot_"):], set()),
             _static_inputs(set_dir, run_dir, mot.stem[len("mot_"):], tids, depth_tested))
            for mot in mots]
    # One process per camera: PNG decode dominates and the cameras are independent.
    n_proc = max(1, min(workers, len(jobs)))
    if n_proc > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(n_proc) as pool:
            results = pool.map(_camera_boxes, jobs)
    else:
        results = [_camera_boxes(j) for j in jobs]
    for mot, (lines, rows, counts) in zip(mots, results):
        (gt / mot.name).write_text("\n".join(lines) + ("\n" if lines else ""))
        rows_pq += rows
        summary["cameras"][mot.stem[len("mot_"):]] = counts
    pd.DataFrame(rows_pq).to_parquet(gt / "boxes_mask.parquet", index=False)
    summary["box_source"] = "rendered_id_mask"
    summary["static_occlusion"] = {"env_plan": (set_dir / "env_plan.json").is_file(),
                                   "id_mask_depth_tested": bool(depth_tested)}
    (gt / "boxes_source.json").write_text(json.dumps(summary, indent=1))
    return summary

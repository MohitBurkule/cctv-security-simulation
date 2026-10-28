"""Amodal person boxes: the full extent a person would occupy if nothing
occluded them (the MOT20 hand-label convention), next to the visible
(mask-derived) box from survsim.gt.mask_boxes.

Route (docs/12 "Amodal boxes"): a calibrated projected body envelope.

* A second stencil pass cannot give it: CustomDepth is depth-tested, so an
  occluder hides the occluded person's stencil exactly as it hides colour.
* One render pass per person gives it exactly but costs N x render time.
* This module: per frame, take the actor's *keyed* position (the manifest key
  the LevelSequence transform track was built from, so it is where the body was
  actually drawn) and its rendered stature, build a vertical cylinder
  (radius R_FRAC x height; with <set>/render_plan.json a thin crown cylinder
  plus a yawed torso ellipse and the feet/toes from the clip's FK curves at the
  rendered clip time) from ground to crown, project its points through
  the live calibration and take the 2-D hull. The residual between that hull
  and the rendered silhouette (clip pose, arm swing, mesh proportions, root
  offset) is then calibrated per person from their own *unoccluded* frames as a
  median per-edge offset in units of hull height, then locally over clean
  frames within +-LOCAL_W; on clean frames the amodal box is the mask box. Result: the box the rendered
  silhouette would have had, even on frames where it is cut by a car, another
  person or the image edge. No engine, no re-render: manifest + calib + masks.

Visibility = visible mask pixels / expected pixels, where expected = amodal box
area x the person's median fill ratio (pixels / box area) on unoccluded frames,
clipped to [0, 1]. It is a pixel fraction, not a box-area ratio, because a
silhouette fills ~40 % of its own box.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

R_POSED = 0.04               # with a pose, the cylinder is only the head / crown
R_FRAC = 0.13                 # cylinder radius / stature (shoulder half-width ~0.13 H)
N_RIM = 16
TORSO_LAT, TORSO_FA = 0.10, 0.06   # torso ellipse half-axes / stature (shoulders, chest depth)
HANDS = False                 # [X] modelled arm swing lowered holdout IoU; feet alone track phase
EDGE_PX = 3.0                 # a box this close to the frame edge is truncated
MIN_CLEAN = 5                 # clean frames needed for a per-person calibration
LOCAL_W = 20                  # frames either side for the local residual
HOLDOUT_GAP = 2               # validation leaves out +-this many clean neighbours
PAD = 0.10                    # other people's boxes dilated by this x height count as occluders


def project(K, R, t, pts: np.ndarray) -> np.ndarray:
    c = pts @ np.asarray(R).T + np.asarray(t)
    p = c @ np.asarray(K).T
    ok = c[:, 2] > 1e-3
    uv = p[:, :2] / np.where(ok, p[:, 2], np.nan)[:, None]
    return uv


def envelope_box(K, R, t, xy: tuple[float, float], height_m: float,
                 pose: np.ndarray | None = None) -> tuple | None:
    """(x, y, w, h) hull of the projected body, or None if behind the camera.

    Body = vertical cylinder (R_FRAC x stature) from ground to crown, plus
    `pose` (Nx3 world points, metres: feet, toes, swung hands from the clip's
    FK curves at this frame) when the render plan is available.
    """
    a = np.linspace(0, 2 * np.pi, N_RIM, endpoint=False)
    r = R_FRAC * height_m if pose is None else R_POSED * height_m
    ring = np.c_[xy[0] + r * np.cos(a), xy[1] + r * np.sin(a)]
    pts = np.r_[np.c_[ring, np.zeros(N_RIM)], np.c_[ring, np.full(N_RIM, height_m)]]
    if pose is not None and len(pose):
        pts = np.r_[pts, pose]
    uv = project(K, R, t, pts)
    if np.isnan(uv).any():
        return None
    x0, y0 = uv.min(0)
    x1, y1 = uv.max(0)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


def iou(a, b) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def containment(inner, outer) -> float:
    """Fraction of `inner`'s area inside `outer`."""
    iw = max(0.0, min(inner[0] + inner[2], outer[0] + outer[2]) - max(inner[0], outer[0]))
    ih = max(0.0, min(inner[1] + inner[3], outer[1] + outer[3]) - max(inner[1], outer[1]))
    a = inner[2] * inner[3]
    return iw * ih / a if a > 0 else 0.0


def apply_offsets(env, off) -> tuple:
    x, y, w, h = env
    x0, y0, x1, y1 = x + off[0] * h, y + off[1] * h, x + w + off[2] * h, y + h + off[3] * h
    return x0, y0, x1 - x0, y1 - y0


def edge_offsets(env, mask) -> np.ndarray:
    ex, ey, ew, eh = env
    mx, my, mw, mh = mask
    return np.array([mx - ex, my - ey, (mx + mw) - (ex + ew), (my + mh) - (ey + eh)]) / eh


def is_clean(mask, others, size, env) -> bool:
    """Unoccluded by other people and untruncated by the frame (static occluders
    are rejected afterwards by the height test in calibrate)."""
    W, H = size
    x, y, w, h = mask
    if h < 20 or x < EDGE_PX or y < EDGE_PX or x + w > W - EDGE_PX or y + h > H - EDGE_PX:
        return False
    ex, ey, ew, eh = env
    if ex < 0 or ey < 0 or ex + ew > W or ey + eh > H:
        return False
    for o in others:
        p = PAD * max(h, o[3])
        if containment((o[0] - p, o[1] - p, o[2] + 2 * p, o[3] + 2 * p), (x, y, w, h)) > 0 \
                or containment((x, y, w, h), (o[0] - p, o[1] - p, o[2] + 2 * p, o[3] + 2 * p)) > 0:
            return False
    return True


def calibrate(samples: list[dict]) -> dict[str, Any] | None:
    """samples: clean rows {env, mask, pixels}. Two passes: the second drops
    frames whose mask is shorter than 0.92 x the first-pass amodal height
    (a static occluder the person-overlap test cannot see)."""
    if not samples:
        return None
    off = np.median([edge_offsets(s["env"], s["mask"]) for s in samples], 0)
    keep = [s for s in samples if s["mask"][3] >= 0.92 * apply_offsets(s["env"], off)[3]]
    if len(keep) >= max(3, len(samples) // 2):
        off = np.median([edge_offsets(s["env"], s["mask"]) for s in keep], 0)
    else:
        keep = samples
    fill = float(np.median([s["pixels"] / max(1.0, s["mask"][2] * s["mask"][3]) for s in keep]))
    return {"offsets": off.tolist(), "fill": fill, "n": len(keep)}


_CURVES: dict[str, dict] = {}


def _curve(path: str | None) -> dict | None:
    if not path or not Path(path).is_file():
        return None
    if path not in _CURVES:
        d = json.loads(Path(path).read_text())
        d["_np"] = {k: np.asarray(d[k], float) for k in ("t", "foot_l", "foot_r")}
        _CURVES[path] = d
    return _CURVES[path]


def _clip_sample(cur: dict, tsec: float) -> tuple[np.ndarray, np.ndarray]:
    t = cur["_np"]["t"]
    i = int(np.searchsorted(t, tsec % cur["len_s"]).clip(0, len(t) - 1))
    return cur["_np"]["foot_l"][i], cur["_np"]["foot_r"][i]


def _stance(cur: dict) -> float:
    """Walk-cycle time with the feet closest together (the idle stand-in)."""
    if "_stance" not in cur:
        fl, fr = cur["_np"]["foot_l"], cur["_np"]["foot_r"]
        cur["_stance"] = float(cur["_np"]["t"][int(np.argmin(np.abs(fl[:, 1] - fr[:, 1])))])
    return cur["_stance"]


def pose_points(entry: dict, ch: dict, f: int, fps: float, blend: int) -> np.ndarray | None:
    """World points (metres, SI) of feet, toes and swung hands at render frame f,
    from the render plan entry (segments, yaws, keys) and the clip's FK curves.
    Mirrors scripts/ue/survsim_scene.build_sequences: each section restarts the
    clip at max(0, start - blend) and plays at the segment rate."""
    walk = _curve(ch.get("foot_curves"))
    if walk is None:
        return None
    kidx = entry.get("_kidx")
    if kidx is None:
        kidx = entry["_kidx"] = {int(k["frame"]): i for i, k in enumerate(entry["keys"])}
    i = kidx.get(int(f))
    if i is None:
        return None
    seg, si = None, 0
    for j, s in enumerate(entry["segments"]):
        if s["start"] <= f < s["end"]:
            seg, si = s, j
    if seg is None:
        return None
    cur = walk
    if seg["clip"] == "run":
        cur = _curve(str(ch.get("foot_curves", "")).replace("_walk.json", "_run.json")) or walk
    if seg["clip"] in ("walk", "run"):
        t0 = max(0, seg["start"] - (blend if si else 0))
        fl, fr = _clip_sample(cur, (f - t0) / fps * float(seg["rate"]))
    else:
        fl, fr = _clip_sample(walk, _stance(walk))
    sc = entry["height_cm"] / float(ch["height_cm"])
    H = float(ch["height_cm"])
    toe = np.array([0.0, 0.07 * H, 0.0])
    # Arms counter-swing the legs: a hand is ~0.6 of the opposite foot's
    # fore-aft offset, at hip height, ~0.2 H out from the midline.
    hl = np.array([0.2 * H, 0.6 * fr[1], 0.45 * H])
    hr = np.array([-0.2 * H, 0.6 * fl[1], 0.45 * H])
    a = np.linspace(0, 2 * np.pi, N_RIM, endpoint=False)
    ell = np.c_[TORSO_LAT * H * np.cos(a), TORSO_FA * H * np.sin(a)]
    torso = np.r_[np.c_[ell, np.full(N_RIM, 0.25 * H)], np.c_[ell, np.full(N_RIM, 0.8 * H)]]
    loc = np.r_[np.array([fl, fl + toe, fr, fr + toe] + ([hl, hr] if HANDS else [])), torso] * sc  # local cm
    psi = np.radians(float(entry["yaws"][i]) + float(ch.get("yaw_offset", -90.0)))
    c, s_ = np.cos(psi), np.sin(psi)
    X = loc[:, 0] * c - loc[:, 1] * s_
    Y = loc[:, 0] * s_ + loc[:, 1] * c
    kx, ky, _ = entry["keys"][i]["location"]
    return np.c_[(kx + X) / 100.0, -(ky + Y) / 100.0, np.maximum(0.0, loc[:, 2]) / 100.0]


def _tracks(manifest: dict) -> dict[str, dict[int, tuple[float, float]]]:
    """uid -> frame -> (x, y) metres, SI (manifest is UE cm with Y negated)."""
    out: dict[str, dict] = {}
    for tr in manifest.get("tracks", []):
        out[tr["uid"]] = {int(k["frame"]): (k["location"][0] / 100.0, -k["location"][1] / 100.0)
                          for k in tr["keys"]}
    return out


def build_amodal(set_dir: Path, holdout: bool = True) -> dict[str, Any]:
    """Add amodal boxes + visibility to <set_dir>/gt/boxes_mask.parquet and write
    <set_dir>/gt/amodal/mot_<cam>.txt. Needs the set's mask-derived GT
    (build_tight_gt), unreal/manifest.json, calib/ and render_appearance.json.

    Returns a summary with validation numbers: IoU(amodal, mask) on clean
    frames (with `holdout`, calibrated on even frames and scored on odd ones)
    and containment of the mask box in the amodal box on occluded frames.
    """
    import pandas as pd

    set_dir = Path(set_dir)
    gt = set_dir / "gt"
    df = pd.read_parquet(gt / "boxes_mask.parquet")
    manifest = json.loads((set_dir / "unreal" / "manifest.json").read_text())
    heights = {u: r["height_cm"] / 100.0 for u, r in
               json.loads((set_dir / "render_appearance.json").read_text()).items()}
    tracks = _tracks(manifest)
    plan_p = set_dir / "render_plan.json"
    plan = json.loads(plan_p.read_text()) if plan_p.is_file() else None
    fps = float(manifest.get("fps", 10))
    blend = max(2, round(0.3 * fps))          # survsim_scene.BLEND_S
    entries = {}
    if plan:
        for e in plan["people"]:
            entries.setdefault(e.get("person_uid", e["uid"]), []).append(e)
    calib = {p.stem: json.loads(p.read_text()) for p in (set_dir / "calib").glob("*.json")}

    env_col, clean_col = [], []
    for r in df.itertuples():
        c = calib.get(r.camera)
        xy = tracks.get(r.uid, {}).get(int(r.frame))
        h = heights.get(r.uid)
        pose = None
        for e in entries.get(r.uid, []):
            v = e.get("visible") or [0, 10 ** 9]
            if v[0] <= r.frame <= v[1]:
                pose = pose_points(e, plan["catalog"][e["character"]], int(r.frame), fps, blend)
                break
        env = envelope_box(c["K"], c["R"], c["t"], xy, h, pose) if c and xy and h else None
        env_col.append(env)
    df["env"] = env_col
    for (cam, f), g in df.groupby(["camera", "frame"]):
        size = tuple(calib[cam]["resolution"])
        masks = {i: tuple(b) for i, b in zip(g.index, g["mask_bbox"]) if b is not None}
        for i in g.index:
            m, e = masks.get(i), df.at[i, "env"]
            clean_col.append((i, bool(m and e and is_clean(m, [v for j, v in masks.items() if j != i],
                                                          size, e))))
    clean = dict(clean_col)
    df["clean"] = [clean.get(i, False) for i in df.index]

    def fit(rows):
        cal = {}
        for uid, g in rows.groupby("uid"):
            s = [{"env": e, "mask": tuple(m), "pixels": p}
                 for e, m, p in zip(g["env"], g["mask_bbox"], g["mask_pixels"])]
            if len(s) >= MIN_CLEAN:
                cal[uid] = calibrate(s)
        pooled = calibrate([{"env": e, "mask": tuple(m), "pixels": p} for e, m, p in
                            zip(rows["env"], rows["mask_bbox"], rows["mask_pixels"])])
        return cal, pooled

    cl = df[df["clean"]]
    cal, pooled = fit(cl)
    summary: dict[str, Any] = {"route": "calibrated projected body envelope",
                               "pose": "fk_curves" if plan else "cylinder_only",
                               "r_frac": R_FRAC, "clean_rows": len(cl),
                               "calibrated_people": len(cal), "pooled": pooled}
    base, fill, src = [], [], []
    for r in df.itertuples():
        c = cal.get(r.uid) or pooled
        if r.env is None or c is None:
            base.append(None), fill.append(None), src.append("none")
            continue
        base.append(apply_offsets(r.env, c["offsets"]))
        fill.append(c["fill"])
        src.append("person" if r.uid in cal else "pooled")
    # Local residual: the median (mask - model) edge offset over this person's
    # clean frames on this camera within +-LOCAL_W frames (view angle and
    # clip-vs-mesh proportions drift along a track; the gait phase is modelled).
    frames, uids, cams = df["frame"].to_numpy(), df["uid"].tolist(), df["camera"].tolist()
    cleanv, masks = df["clean"].to_numpy(), df["mask_bbox"].tolist()
    local: dict = {}
    for i in range(len(df)):
        if cleanv[i] and base[i] is not None:
            local.setdefault((uids[i], cams[i]), []).append(
                (frames[i], edge_offsets(base[i], tuple(masks[i])),
                 float(df["mask_pixels"].iat[i]) / max(1.0, masks[i][2] * masks[i][3])))
    local = {k: (np.array([x[0] for x in v]), np.array([x[1] for x in v]),
                 np.array([x[2] for x in v])) for k, v in local.items()}
    cur, loo = list(base), []
    for i in range(len(df)):
        if base[i] is None or (uids[i], cams[i]) not in local:
            continue
        F, res, fr = local[(uids[i], cams[i])]
        dF = np.abs(F - frames[i])
        near = (dF <= LOCAL_W) & (dF > 0)
        if near.sum() >= 3:
            cur[i] = apply_offsets(base[i], np.median(res[near], 0))
            fill[i] = float(np.median(fr[near]))
            src[i] += "+local"
        if cleanv[i]:
            gap = (dF <= LOCAL_W) & (dF > HOLDOUT_GAP)
            bb = apply_offsets(base[i], np.median(res[gap], 0)) if gap.sum() >= 3 else base[i]
            loo.append(iou(bb, tuple(masks[i])))
    if holdout:
        v = loo
        summary["clean_iou_holdout"] = _dist(v)
        summary["clean_iou_ge_0.9"] = round(float(np.mean(np.array(v) >= 0.9)), 4) if v else None
        summary["holdout"] = (f"each clean frame predicted without itself or any clean frame "
                              f"within +-{HOLDOUT_GAP} (as if occluded)")

    am, vis = [], []
    for r, fl, cb in zip(df.itertuples(), fill, cur):
        if cb is None:
            am.append(None), vis.append(None)
            continue
        a = tuple(r.mask_bbox) if r.clean else cb   # fully visible: amodal IS the visible box
        am.append([round(v, 2) for v in a])
        exp = fl * a[2] * a[3]
        vis.append(round(float(min(1.0, r.mask_pixels / exp)) if exp > 0 else 0.0, 4))
    src = ["mask" if (c and s0 != "none") else s0 for c, s0 in zip(df["clean"], src)]
    df["amodal_bbox"], df["visibility"], df["amodal_calib"] = am, vis, src
    df["visible_bbox"] = df["mask_bbox"]

    ok = df[df["amodal_bbox"].notna() & df["mask_bbox"].notna()]
    iou_clean = [iou(cur[j], tuple(masks[j])) for j in range(len(df))
                 if cleanv[j] and cur[j] is not None]
    occ = ok[(~ok["clean"]) & (ok["visibility"] < 0.9)]
    cont = [containment(tuple(m), tuple(a)) for a, m in zip(occ["amodal_bbox"], occ["mask_bbox"])]
    summary["clean_iou_model_insample"] = _dist(iou_clean)
    summary["occluded_rows"] = len(occ)
    summary["occluded_containment"] = _dist(cont)
    summary["occluded_containment_ge_0.95"] = float(np.mean(np.array(cont) >= 0.95)) if cont else None
    v = df["visibility"].dropna().to_numpy(float)
    summary["visibility"] = {"n": len(v),
                             "hist_0.1": np.histogram(v, bins=10, range=(0, 1))[0].tolist(),
                             "frac_lt_0.5": float(np.mean(v < 0.5)) if len(v) else None,
                             "frac_lt_0.9": float(np.mean(v < 0.9)) if len(v) else None,
                             "median": float(np.median(v)) if len(v) else None}
    summary["calibration"] = {u: c for u, c in cal.items()}

    out = gt / "amodal"
    out.mkdir(exist_ok=True)
    tids = json.loads((set_dir / "meta" / "track_ids.json").read_text())
    for cam, g in df.groupby("camera"):
        lines = []
        for r in g.sort_values(["frame", "track_id"]).itertuples():
            if r.amodal_bbox is None:
                continue
            x, y, w, h = r.amodal_bbox
            conf = 1 if r.mask_bbox is not None else 0
            lines.append(f"{int(r.frame)},{int(r.track_id)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                         f"{conf},1,{float(r.visibility or 0):.3f}")
        (out / f"mot_{cam}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    del tids
    summary["mot_visibility_from_render"] = write_render_visibility(gt, df)
    df.drop(columns=["env"]).to_parquet(gt / "boxes_mask.parquet", index=False)
    (out / "amodal_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def render_visibility_lines(lines: list[str], vis: dict[tuple[int, int], float]) -> tuple[list[str], int]:
    """MOT lines with column 9 replaced by the rendered-mask visibility.

    Principle 1 (docs/BUG_night_visibility_zero.md): column 9 used to be the
    mock painter's `visible_frac`, which paints every scenery prop as a solid
    image-space rectangle (a fence of length L as an opaque L x L block), so a
    person standing in front of or beside a fence read 0.000 while the render
    showed them whole. vis maps (frame, track_id) -> mask pixels / expected
    pixels (build_amodal); rows with no entry (no mask pixels, or no amodal
    calibration) keep their value. Returns (lines, n_changed)."""
    out, n = [], 0
    for ln in lines:
        f = ln.split(",")
        v = vis.get((int(f[0]), int(f[1])))
        if v is not None and int(float(f[6])) == 1:
            new = f"{float(v):.3f}"
            if new != f[8]:
                n += 1
            f[8] = new
        out.append(",".join(f))
    return out, n


def write_render_visibility(gt: Path, df) -> dict[str, int]:
    """Rewrite <gt>/mot_<cam>.txt column 9 from df's render visibility."""
    ok = df[df["mask_bbox"].notna() & df["visibility"].notna()]
    changed: dict[str, int] = {}
    for cam, g in ok.groupby("camera"):
        p = Path(gt) / f"mot_{cam}.txt"
        if not p.is_file():
            continue
        vis = {(int(f), int(t)): float(v) for f, t, v in zip(g["frame"], g["track_id"], g["visibility"])}
        lines, n = render_visibility_lines([ln for ln in p.read_text().splitlines() if ln.strip()], vis)
        p.write_text("\n".join(lines) + ("\n" if lines else ""))
        changed[cam] = n
    return changed


def _dist(x) -> dict[str, Any]:
    a = np.asarray(x, float)
    if not len(a):
        return {"n": 0}
    return {"n": len(a), "median": round(float(np.median(a)), 4),
            "p10": round(float(np.percentile(a, 10)), 4), "mean": round(float(a.mean()), 4)}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="amodal boxes for a rendered --id-mask set")
    ap.add_argument("set_dir", type=Path)
    a = ap.parse_args(argv)
    s = build_amodal(a.set_dir)
    print(json.dumps({k: v for k, v in s.items() if k != "calibration"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

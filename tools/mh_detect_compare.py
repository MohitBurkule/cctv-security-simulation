"""Does a detector find the MetaHuman "more real" than the stylised figures?

    .venv/bin/python tools/mh_detect_compare.py \
        --mh out/demo_metahuman/<render_dir> \
        --ref out/batch_v1/render_demo_day --ref-cam cam_01 [--json out.json]

Per person box: yolov8n (COCO) person confidence of the best detection at
IoU >= 0.5, or 0.0 if missed (a miss is the strongest "not a person" signal,
so it is scored, not dropped).

* MetaHuman frames: the box is *measured*, not projected (CLAUDE.md principle
  1): the render dir holds `plate.png` (same camera, person hidden) and the
  box is the bounding box of |frame - plate| > 12 grey levels, largest blob.
* Reference frames: MOT GT boxes from `<ref>/gt/mot_<cam>.txt` with visibility
  >= 0.8, restricted to heights within +-35 % of the MetaHuman box height so
  the two populations sit at the same standoff (pixels on target is what a
  detector's confidence depends on most).

Also reports survsim.realism.stats (median of frame_stats) for both sets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from survsim.realism.stats import band, frame_stats, list_images


def iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def mask_box(frame: np.ndarray, plate: np.ndarray, thr: int = 12):
    """Box of the largest changed blob between a frame and its empty plate.

    Cast shadows are excluded: a pixel that only got darker (luma ratio < 1)
    while keeping its chromaticity is the plate in shadow, not the person.
    Without this the box took in the shadow and scored real hits as misses.
    """
    d = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY))
    f, p = frame.astype(np.float32) + 1.0, plate.astype(np.float32) + 1.0
    lf, lp = f.sum(2), p.sum(2)
    dchroma = np.abs(f / lf[..., None] - p / lp[..., None]).sum(2)
    shadow = (lf < lp) & (lf > 0.15 * lp) & (dchroma < 0.06)
    m = ((d > thr) & ~shadow).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = st[i]
    return [float(x), float(y), float(w), float(h)]


def best_conf(net, img_path, box):
    r = net.predict(str(img_path), classes=[0], conf=0.05, verbose=False)[0]
    best = 0.0
    for (x0, y0, x1, y1), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
        if iou(box, [x0, y0, x1 - x0, y1 - y0]) >= 0.5:
            best = max(best, float(c))
    return best


def summarise(confs):
    v = np.asarray(confs, float)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": round(float(v.mean()), 3), "median": round(float(np.median(v)), 3),
            "recall@0.25": round(float((v >= 0.25).mean()), 3), "min": round(float(v.min()), 3),
            "max": round(float(v.max()), 3)}


def realism(paths):
    rows = [frame_stats(cv2.imread(str(p))).as_dict() for p in paths]
    return {k: round(band([r[k] for r in rows])["median"], 3) for k in rows[0]} if rows else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mh", type=Path)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--ref-cam", default="cam_01")
    ap.add_argument("--ref-step", type=int, default=10)
    ap.add_argument("--height", type=float, help="box height (px) to match if --mh absent")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    from ultralytics import YOLO
    net = YOLO("yolov8n.pt")
    out = {}
    h_target = a.height
    if a.mh:
        plate = cv2.imread(str(a.mh / "plate.png"))
        frames = [p for p in list_images(a.mh) if p.name != "plate.png"]
        rows = []
        for p in frames:
            box = mask_box(cv2.imread(str(p)), plate)
            if box is None:
                rows.append({"frame": p.name, "box": None, "conf": 0.0})
                continue
            rows.append({"frame": p.name, "box": box, "conf": best_conf(net, p, box)})
        out["mh"] = {"per_frame": rows, "summary": summarise([r["conf"] for r in rows]),
                     "realism": realism(frames)}
        hs = [r["box"][3] for r in rows if r["box"]]
        h_target = float(np.median(hs)) if hs else h_target
    mot = np.loadtxt(a.ref / "gt" / f"mot_{a.ref_cam}.txt", delimiter=",", ndmin=2)
    imgs = {int(p.stem): p for p in list_images(a.ref / "render" / a.ref_cam)}
    confs, used = [], []
    for fr in sorted(set(mot[:, 0].astype(int)))[::a.ref_step]:
        idx = fr - 1                                    # MOT frames are 1-based
        if idx not in imgs:
            continue
        for row in mot[mot[:, 0] == fr]:
            box, vis = row[2:6].tolist(), row[8]
            if vis < 0.8 or (h_target and not 0.65 * h_target <= box[3] <= 1.35 * h_target):
                continue
            confs.append(best_conf(net, imgs[idx], box))
            used.append(imgs[idx])
    out["ref"] = {"dir": str(a.ref), "cam": a.ref_cam, "height_px_target": h_target,
                  "summary": summarise(confs), "realism": realism(sorted(set(used))[:20])}
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_frame"} for k, v in out.items()},
                     indent=1))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Person recall of a COCO detector on rendered frames vs the run's GT boxes.

    .venv/bin/python tools/eval_recall.py <render_dir> --gt <run_dir> [--n 5] [--model yolov8n.pt]

For each camera: n evenly spaced frames that contain GT people >= 20 px tall
(geometry.usably_visible), YOLO person detections at conf 0.25, greedy IoU>=0.5
matching **in descending detection-score order** (COCO style: the highest-
confidence detection gets first pick of GT boxes, not the tallest GT box first).
Reports recall, precision, and recall by box-height band. GT frame i pairs with
<cam>/NNNN.png index i (the check_render convention; --offset shifts).

Matching was tallest-GT-first before this file's most recent fix, which let a
low-confidence detection sitting on a tall person steal a match before a
high-confidence detection on a shorter person got a turn — the opposite of what
an operator sees at a real confidence threshold. Any recall numbers recorded
before that fix (see docs/13_BASELINES.md) are on the old matching order.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from survsim.checks.render import read_mot

MIN_H = 20.0


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(1e-9, a[2] * a[3] + b[2] * b[3] - inter)


def match_frame(gts: list, dets: list, scores: list, thr: float = 0.5) -> tuple[int, int, list[bool]]:
    """COCO-style greedy matching: detections in descending score order.

    Each detection claims its best-IoU still-unclaimed GT box; a claim below
    `thr` is a false positive rather than a match. This is what a real
    operating threshold sees. Matching GT boxes tallest-first instead (the
    previous behaviour) let a low-confidence detection on a tall person steal
    a match before a high-confidence detection on a shorter person got a turn.

    Returns (tp, fp, gt_used) where `gt_used[j]` says whether GT box `j` was
    matched (so the caller can derive FN and per-band recall from `gts`).
    """
    order = sorted(range(len(dets)), key=lambda j: -scores[j])
    gt_used = [False] * len(gts)
    tp = fp = 0
    for di in order:
        d = dets[di]
        best, bj = 0.0, None
        for j, g in enumerate(gts):
            if not gt_used[j] and iou(g, d) > best:
                best, bj = iou(g, d), j
        if best >= thr:
            gt_used[bj] = True
            tp += 1
        else:
            fp += 1
    return tp, fp, gt_used


def evaluate(render_dir: Path, gt_dir: Path, n: int, model_path: str, offset: int = 0,
             conf: float = 0.25, thr: float = 0.5) -> dict:
    from ultralytics import YOLO
    model = YOLO(model_path)
    out = {}
    for cam_dir in sorted(p for p in render_dir.iterdir() if p.is_dir()):
        mot_p = gt_dir / f"mot_{cam_dir.name}.txt"
        if not mot_p.exists():
            continue
        mot = read_mot(mot_p)
        pngs = {int(p.stem): p for p in cam_dir.glob("*.png")}
        cand = [i for i in sorted(pngs)
                if any(b[4] >= MIN_H for b in mot.get(i + offset, []))]
        if not cand:
            out[cam_dir.name] = {"frames": 0}
            continue
        pick = [cand[round(k)] for k in np.linspace(0, len(cand) - 1, min(n, len(cand)))]
        tp = fn = fp = 0
        bands = {"20-60": [0, 0], "60-120": [0, 0], "120+": [0, 0]}
        for i in pick:
            gts = [b[1:] for b in mot[i + offset] if b[4] >= MIN_H]
            res = model.predict(str(pngs[i]), conf=conf, classes=[0], verbose=False)[0]
            dets = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in res.boxes.xyxy.tolist()]
            scores = res.boxes.conf.tolist()
            frame_tp, frame_fp, gt_used = match_frame(gts, dets, scores, thr)
            tp += frame_tp
            fp += frame_fp
            for j, g in enumerate(gts):
                band = "20-60" if g[3] < 60 else ("60-120" if g[3] < 120 else "120+")
                bands[band][1] += 1
                if gt_used[j]:
                    bands[band][0] += 1
                else:
                    fn += 1
        out[cam_dir.name] = {
            "frames": len(pick), "gt": tp + fn, "tp": tp, "fp": fp,
            "recall": round(tp / max(1, tp + fn), 3), "precision": round(tp / max(1, tp + fp), 3),
            "recall_by_height": {k: (round(v[0] / v[1], 3) if v[1] else None, v[1])
                                 for k, v in bands.items()}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("render_dir", type=Path)
    ap.add_argument("--gt", type=Path, required=True, help="run dir (with gt/) or gt dir")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--model", default=str(Path(
        "/tmp/claude-1000/-home-mohit-multi-person-camera-real-sim/"
        "5f6f75e9-f418-4115-9d7d-4e03fa563d1f/scratchpad/yolov8n.pt")))
    a = ap.parse_args()
    gt = a.gt / "gt" if (a.gt / "gt").is_dir() else a.gt
    model = a.model if Path(a.model).exists() else "yolov8n.pt"
    print(json.dumps(evaluate(a.render_dir, gt, a.n, model, a.offset), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

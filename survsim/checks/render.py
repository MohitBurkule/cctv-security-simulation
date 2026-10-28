"""Render gate: pure checks over frames + ground truth, and ``run_all`` wiring.

Each check guards a failure this project has already shipped (docs/11 "Bugs
found by building and measuring", docs/12 "Automated render gate"):

* blank          - MRQ wrote correctly-sized, entirely black PNGs with success=True
* views_differ   - no CameraCutTrack: every camera rendered the same view
* horizon        - inverted pitch pointed every camera at the sky
* motion / animation - a person without an animation track slides or freezes
* gt_*           - GT that does not follow the pixels
"""

from __future__ import annotations

import itertools
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .core import FAIL, INFO, PASS, SKIP, WARN, Check, ok

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
WORK_WIDTH = 640          # frames are held at this width; GT boxes are scaled to it
MIN_BOX_H = 20            # geometry.usably_visible height rule, in native pixels

# Real CCTV band, docs/15_REALISM_STATS.md (80 frames of real_cctv + MOT20).
REAL_BAND: dict[str, tuple[float, float]] = {
    "noise_sigma": (0.278, 0.353), "lap_var_log10": (1.831, 2.284),
    "mtf_ratio": (0.0146, 0.0327), "edge_width": (5.00, 6.53),
    "luma_mean": (50.6, 117.5), "rms_contrast": (0.478, 0.635),
    "dynamic_range": (131.8, 234.1), "luma_skew": (0.19, 1.46),
    "hist_entropy": (6.67, 7.72), "chroma_mean": (8.0, 15.1),
    "cast_a": (0.47, 7.94), "cast_b": (-8.79, 9.25), "blockiness": (1.111, 1.197),
}


# ------------------------------------------------------------------ discovery
@dataclass
class Frames:
    """Frames of one render, grouped by camera, keyed by frame index."""
    paths: dict[str, dict[int, Path]] = field(default_factory=dict)
    native_size: tuple[int, int] | None = None   # (w, h)


_NUM = re.compile(r"^(?:(.*?)[._-])?(\d+)$")


def discover(root: Path) -> Frames:
    """``<root>/[render/]<cam>/<n>.png`` or flat ``<cam>_<n>.png``."""
    root = Path(root)
    fr = Frames()
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS or "gt" in p.relative_to(root).parts[:-1]:
            continue
        m = _NUM.match(p.stem)
        if not m:
            continue
        prefix, idx = m.group(1), int(m.group(2))
        cam = p.parent.name if p.parent != root else (prefix or "cam")
        if p.parent != root and prefix and prefix != cam and not prefix.startswith(cam):
            cam = f"{p.parent.name}/{prefix}"
        fr.paths.setdefault(cam, {})[idx] = p
    return fr


def load_small(path: Path, width: int = WORK_WIDTH) -> tuple[np.ndarray, tuple[int, int]]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"unreadable image {path}")
    h, w = img.shape[:2]
    if w != width:
        img = cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    return img, (w, h)


def grey(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) if img.ndim == 3 \
        else img.astype(np.float32)


# ------------------------------------------------------------- pixel checks
def check_frame_count(counts: dict[str, int], expected: int | None = None,
                      gaps: dict[str, int] | None = None) -> Check:
    if not counts or not any(counts.values()):
        return Check("frame_count", FAIL, 0, expected or ">0", "no frames found")
    vals = set(counts.values())
    bad = []
    if expected is not None:
        bad += [f"{c}={n}" for c, n in counts.items() if n != expected]
    elif len(vals) > 1:
        bad += [f"{c}={n}" for c, n in counts.items()]
    bad += [f"{c}:{g} gaps" for c, g in (gaps or {}).items() if g]
    return Check("frame_count", ok(not bad), sum(counts.values()),
                 expected if expected is not None else "equal per camera",
                 ", ".join(bad) or f"{len(counts)} camera(s) x {max(vals)}")


def check_blank(frames: Sequence[np.ndarray], min_mean: float = 8.0, min_std: float = 4.0) -> Check:
    """Black or flat frames (the MRQ 'success=True, all black' failure)."""
    bad = [i for i, f in enumerate(frames)
           if (g := grey(f)).mean() < min_mean or g.std() < min_std]
    return Check("blank_frames", ok(not bad), len(bad), 0,
                 f"first blank idx {bad[:5]}" if bad else f"mean>={min_mean}, std>={min_std}")


def consecutive_diffs(frames: Sequence[np.ndarray]) -> list[float]:
    g = [grey(f) for f in frames]
    return [float(np.abs(a - b).mean()) for a, b in itertools.pairwise(g)]


def _boxes_by_id(mot: dict[int, list], scale: float, min_h: float = 0.0) -> dict[int, dict[int, tuple]]:
    """GT rows -> {track id: {frame index: (x, y, w, h)}}, work-resolution scale."""
    by_id: dict[int, dict[int, tuple]] = defaultdict(dict)
    for f, boxes in mot.items():
        for b in boxes:
            if b[4] >= min_h:
                by_id[b[0]][f] = tuple(v * scale for v in b[1:])
    return by_id


def _dilate_box(box: Sequence[float], factor: float, w: int, h: int) -> tuple[int, int, int, int]:
    """xywh -> (x0, y0, x1, y1) int pixel ROI, grown `factor`x around its centre, clamped to frame."""
    x, y, bw, bh = box
    cx, cy = x + bw / 2, y + bh / 2
    hw, hh = bw * factor / 2, bh * factor / 2
    x0, y0 = max(0, int(cx - hw)), max(0, int(cy - hh))
    x1, y1 = min(w, int(np.ceil(cx + hw))), min(h, int(np.ceil(cy + hh)))
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def check_motion(frames: Sequence[np.ndarray] | dict[int, np.ndarray], frozen: float = 0.05,
                 max_frozen_frac: float = 0.5, name: str = "motion",
                 by_id: dict[int, dict[int, tuple]] | None = None,
                 idx: Sequence[int] | None = None) -> Check:
    """Consecutive frames must not be identical (frozen sequence / no playback).

    Whole-frame frame-to-frame difference is blind to far/dim people: a 13-20 px
    person under a dim lamp moves by less than the frame's own sensor/compression
    noise once averaged over the whole picture (docs/11 "far-night motion gate").
    When GT is available (`by_id`: track id -> {frame index: xywh, work-res scale},
    `idx`: this camera's rendered frame indices, `frames`: dict frame index ->
    image), motion is instead measured only inside each track's own box - dilated
    50% around its centre so limb sway at the edges counts - between consecutive
    rendered frames, and the "frozen" step threshold is scaled down for small
    boxes (fewer, lower-contrast pixels move by less even when the person does).
    Falls back to the whole-frame test when no GT is given.
    """
    if by_id and idx and len(idx) >= 2 and isinstance(frames, dict):
        h0, w0 = next(iter(frames.values())).shape[:2]
        greys: dict[int, np.ndarray] = {}
        samples: list[tuple[float, float]] = []   # (step diff, size-scaled threshold)
        for seen in by_id.values():
            for f0, f1 in itertools.pairwise(idx):
                if f0 not in seen or f1 not in seen:
                    continue
                box = seen[f0]
                if box[3] < 1:
                    continue
                x0, y0, x1, y1 = _dilate_box(box, 1.5, w0, h0)
                g0 = greys.setdefault(f0, grey(frames[f0]))
                g1 = greys.setdefault(f1, grey(frames[f1]))
                d = float(np.abs(g0[y0:y1, x0:x1] - g1[y0:y1, x0:x1]).mean())
                thr = frozen * float(np.clip(box[3] / 40.0, 0.25, 1.0))
                samples.append((d, thr))
        if samples:
            frac = sum(d <= t for d, t in samples) / len(samples)
            fail = frac > max_frozen_frac
            return Check(name, ok(not fail), round(frac, 3),
                         f"frozen frac<={max_frozen_frac} (GT-box, size-scaled)",
                         f"{len(samples)} track-steps, median step {np.median([d for d, _ in samples]):.3f}")
        # no track spanned two consecutive rendered frames: fall through to whole-frame
    seq = list(frames.values()) if isinstance(frames, dict) else frames
    if len(seq) < 2:
        return Check(name, SKIP, detail="<2 frames")
    d = consecutive_diffs(seq)
    frac = sum(x < frozen for x in d) / len(d)
    g0, g1 = grey(seq[0]), grey(seq[-1])
    span = float(np.abs(g0 - g1).mean())
    fail = frac > max_frozen_frac or span < frozen
    return Check(name, ok(not fail), round(frac, 3), f"frozen frac<={max_frozen_frac}",
                 f"first-last diff {span:.2f}, median step {np.median(d):.3f}")


def check_views_differ(cam_means: dict[str, np.ndarray], min_diff: float = 5.0) -> Check:
    """Pairwise mean abs diff between cameras' mean frames (duplicate camera-cut bug)."""
    cams = sorted(cam_means)
    if len(cams) < 2:
        return Check("views_differ", SKIP, detail="single camera")
    worst_pair, worst = None, np.inf
    for i, a in enumerate(cams):
        for b in cams[i + 1:]:
            x, y = grey(cam_means[a]), grey(cam_means[b])
            if x.shape != y.shape:
                y = cv2.resize(y, (x.shape[1], x.shape[0]))
            d = float(np.abs(x - y).mean())
            if d < worst:
                worst, worst_pair = d, (a, b)
    return Check("views_differ", ok(worst > min_diff), round(worst, 2), f">{min_diff}",
                 f"closest pair {worst_pair[0]} vs {worst_pair[1]}")


def check_exposure(frame: np.ndarray, band: tuple[float, float] = (20.0, 235.0),
                   max_clip: float = 0.25, name: str = "exposure") -> Check:
    g = grey(frame)
    m = float(g.mean())
    sat, crush = float((g >= 250).mean()), float((g <= 5).mean())
    good = band[0] <= m <= band[1] and sat <= max_clip and crush <= max_clip
    return Check(name, ok(good), round(m, 1), f"luma {band[0]:.0f}-{band[1]:.0f}, clip<={max_clip}",
                 f"saturated {sat:.1%}, crushed {crush:.1%}")


def sky_fraction(img: np.ndarray) -> float:
    """Fraction of pixels that look like sky: bright, blue-dominant or textureless."""
    b, g, r = [img[:, :, i].astype(np.float32) for i in range(3)]
    luma = grey(img)
    blue = (b > r + 12) & (b >= g - 5) & (luma > 90)
    tex = np.abs(cv2.Laplacian(luma, cv2.CV_32F, ksize=3))
    flat_bright = (cv2.blur(tex, (9, 9)) < 1.0) & (luma > 150)
    return float((blue | flat_bright).mean())


def check_horizon(frame: np.ndarray, max_sky: float = 0.6, name: str = "horizon") -> Check:
    """Bottom quarter must be ground, not sky (the inverted-pitch bug)."""
    h = frame.shape[0]
    frac = sky_fraction(frame[int(h * 0.75):])
    return Check(name, ok(frac <= max_sky), round(frac, 3), f"<={max_sky}",
                 "bottom-quarter sky fraction")


def check_rendered_height_matches_gt(gt_heights: dict[str, float],
                                     rendered_heights: dict[str, float],
                                     tol_cm: float = 2.0,
                                     name: str = "gt_height_matches_render") -> Check:
    """Flag any person whose rendered body height disagrees with the Director's
    GT height (`identities.parquet` `height_m`) by more than `tol_cm`.

    Pure and engine-free: takes plain {uid: height_m} maps (e.g. from
    `identities.parquet` and `render_appearance.json`/`identities_rendered.parquet`),
    so it can run in CI with no frames or editor. Guards CLAUDE.md principle 1
    ("truth is what happened") — a manifest that silently falls back to a
    default height renders a body ground truth does not describe.
    """
    common = sorted(set(gt_heights) & set(rendered_heights))
    missing = sorted(set(gt_heights) - set(rendered_heights))
    bad = [(uid, gt_heights[uid], rendered_heights[uid]) for uid in common
           if abs((gt_heights[uid] - rendered_heights[uid]) * 100.0) > tol_cm]
    if not common:
        return Check(name, SKIP, detail="no uid present in both height maps")
    detail = f"{len(bad)}/{len(common)} beyond {tol_cm} cm"
    if bad:
        uid, gt, rend = bad[0]
        detail += f"; e.g. {uid} gt={gt:.3f}m rendered={rend:.3f}m"
    if missing:
        detail += f"; {len(missing)} uid(s) missing a rendered height"
    return Check(name, ok(not bad and not missing), len(bad), f"<= {tol_cm} cm", detail)


def height_maps(gt_dir: Path, appearance: Path) -> tuple[dict[str, float], dict[str, float]] | None:
    """({uid: GT height_m} from identities.parquet, {uid: rendered height_m} from
    render_appearance.json), or None when either file is missing."""
    ident = Path(gt_dir) / "identities.parquet"
    if not ident.is_file() or not Path(appearance).is_file():
        return None
    import json

    import pandas as pd
    df = pd.read_parquet(ident, columns=["uid", "height_m"])
    gt = {u: float(h) for u, h in zip(df["uid"], df["height_m"])}
    app = json.loads(Path(appearance).read_text())
    rend = {u: float(r["height_cm"]) / 100.0 for u, r in app.items()
            if isinstance(r, dict) and r.get("height_cm") is not None}
    return gt, rend


def height_maps_from_rendered_parquet(gt_dir: Path, rendered_parquet: Path,
                                      ) -> tuple[dict[str, float], dict[str, float]] | None:
    """Fallback for `height_maps`: join `identities.parquet` height_m against
    `identities_rendered.parquet` height_m on uid, for sets that carry the
    rendered manifest but not `render_appearance.json`."""
    ident = Path(gt_dir) / "identities.parquet"
    if not ident.is_file() or not Path(rendered_parquet).is_file():
        return None
    import pandas as pd
    df = pd.read_parquet(ident, columns=["uid", "height_m"])
    gt = {u: float(h) for u, h in zip(df["uid"], df["height_m"])}
    rdf = pd.read_parquet(rendered_parquet, columns=["uid", "height_m"]).dropna()
    rend = {u: float(h) for u, h in zip(rdf["uid"], rdf["height_m"])}
    return gt, rend


def check_height_files(gt_dir: Path, render_dir: Path) -> Check:
    """check_rendered_height_matches_gt on <gt>/identities.parquet vs the set's
    render_appearance.json / identities_rendered.parquet.

    `render_dir` may be either the render set dir itself (as passed to
    ``scripts/check_render.py <set> --gt <set>``) or its `render/` subdir (as
    `tools/render_run.py` passes when it builds the same table live) - try the
    set dir first, then its parent, matching how `tools/render_run.py` locates
    `render_appearance.json` next to `render/`.
    """
    for set_dir in (Path(render_dir), Path(render_dir).parent):
        appearance = set_dir / "render_appearance.json"
        if appearance.is_file():
            maps = height_maps(gt_dir, appearance)
            if maps is not None:
                return check_rendered_height_matches_gt(*maps)
        rendered_parquet = set_dir / "identities_rendered.parquet"
        if rendered_parquet.is_file():
            maps = height_maps_from_rendered_parquet(gt_dir, rendered_parquet)
            if maps is not None:
                return check_rendered_height_matches_gt(*maps)
    return Check("gt_height_matches_render", SKIP,
                 detail="identities.parquet, render_appearance.json or "
                        "identities_rendered.parquet missing")


# --------------------------------------------------------------- GT checks
def read_mot(path: Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    out: dict[int, list] = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        p = line.strip().split(",")
        if len(p) >= 6:
            out[int(float(p[0]))].append((int(float(p[1])), *map(float, p[2:6])))
    return dict(out)


def check_boxes_in_image(mot: dict[int, list], size: tuple[int, int], tol: float = 2.0,
                         name: str = "gt_boxes_in_image") -> Check:
    w, h = size
    n = bad = 0
    ex = ""
    for f, boxes in mot.items():
        for tid, x, y, bw, bh in boxes:
            n += 1
            if x < -tol or y < -tol or x + bw > w + tol or y + bh > h + tol or bw <= 0 or bh <= 0:
                bad += 1
                ex = ex or f"frame {f} id {tid} box {x:.0f},{y:.0f},{bw:.0f},{bh:.0f}"
    return Check(name, ok(bad == 0), bad, 0, ex or f"{n} boxes inside {w}x{h}")


def box_contrast(frame: np.ndarray, plate: np.ndarray, box: Sequence[float]) -> float:
    """Mean abs diff between frame and background plate inside box (work-res coords)."""
    x, y, bw, bh = box
    H, W = frame.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(np.ceil(x + bw))), min(H, int(np.ceil(y + bh)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(np.abs(grey(frame[y0:y1, x0:x1]) - grey(plate[y0:y1, x0:x1])).mean())


def check_person_rendered(frames: dict[int, np.ndarray], plate: np.ndarray, mot: dict[int, list],
                          scale: float, min_diff: float = 6.0, min_frac: float = 0.6,
                          name: str = "gt_person_rendered") -> Check:
    """Boxes >= 20 px tall must sit on pixels that differ from the empty background."""
    vals = [box_contrast(frames[f], plate, [v * scale for v in b[1:]])
            for f, boxes in mot.items() if f in frames
            for b in boxes if b[4] >= MIN_BOX_H]
    if not vals:
        return Check(name, SKIP, detail="no GT box >= 20 px on a rendered frame")
    frac = sum(v > min_diff for v in vals) / len(vals)
    return Check(name, ok(frac >= min_frac), round(frac, 3), f">={min_frac}",
                 f"{len(vals)} boxes, median diff {np.median(vals):.1f} (>{min_diff})")


def check_counts_vs_world(mot: dict[int, list], world_counts: dict[int, int],
                          name: str = "gt_count_vs_world") -> Check:
    """A camera cannot see more people than exist; frames must exist in world truth."""
    bad = [f for f, b in mot.items() if len(b) > world_counts.get(f, 0)]
    return Check(name, ok(not bad), len(bad), 0,
                 f"frames with more boxes than people: {bad[:5]}" if bad
                 else f"{len(mot)} frames, <= {max(world_counts.values(), default=0)} people")


def check_gt_frame_coverage(render_idx: Sequence[int], world_frames: set[int], offset: int,
                            name: str = "gt_frame_coverage") -> Check:
    idx = [i + offset for i in render_idx]
    missing = [i for i in idx if i not in world_frames]
    frac = len(missing) / max(1, len(idx))
    return Check(name, ok(frac <= 0.05), len(missing), "<=5% of frames",
                 f"{len(idx)} rendered, {len(world_frames)} GT frames, offset {offset}")


def _crop(img: np.ndarray, box: Sequence[float], size=(24, 48)) -> np.ndarray | None:
    x, y, bw, bh = box
    H, W = img.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(np.ceil(x + bw))), min(H, int(np.ceil(y + bh)))
    if x1 - x0 < 3 or y1 - y0 < 6:
        return None
    return cv2.resize(grey(img[y0:y1, x0:x1]), size, interpolation=cv2.INTER_AREA)


def check_animation(frames: dict[int, np.ndarray], mot: dict[int, list], scale: float,
                    step: int = 5, min_diff: float = 3.0, min_frac: float = 0.5,
                    name: str = "animation") -> Check:
    """Each person's box content must change between t and t+step (limbs move).

    A person with no animation track slides rigidly or stands frozen: the
    box-normalised crop is then near-identical between frames.
    """
    by_id: dict[int, dict[int, tuple]] = defaultdict(dict)
    for f, boxes in mot.items():
        if f in frames:
            for b in boxes:
                if b[4] >= MIN_BOX_H:
                    by_id[b[0]][f] = b[1:]
    per_id = {}
    for tid, seen in by_id.items():
        ds = []
        for f, b in seen.items():
            if f + step in seen:
                a = _crop(frames[f], [v * scale for v in b])
                c = _crop(frames[f + step], [v * scale for v in seen[f + step]])
                if a is not None and c is not None:
                    ds.append(float(np.abs(a - c).mean()))
        if ds:
            per_id[tid] = float(np.median(ds))
    if not per_id:
        return Check(name, SKIP, detail=f"no person visible (>=20 px) at t and t+{step}")
    moving = sum(v > min_diff for v in per_id.values())
    frac = moving / len(per_id)
    still = [t for t, v in per_id.items() if v <= min_diff]
    return Check(name, ok(frac >= min_frac), round(frac, 3), f">={min_frac} ids, diff>{min_diff}",
                 f"{len(per_id)} ids; static ids {still[:5]}" if still else f"{len(per_id)} ids")


# ------------------------------------------------------------- log checks
def check_log_text(text: str) -> list[Check]:
    lines = text.splitlines()
    py = [ln for ln in lines if "LogPython: Error" in ln]
    fatal = [ln for ln in lines if re.search(r"\bFatal\b", ln)]
    conv = [ln for ln in lines if "Failed to convert type" in ln]
    return [
        Check("log_python_error", ok(not py), len(py), 0, py[0].strip()[:120] if py else ""),
        Check("log_fatal", ok(not fatal), len(fatal), 0, fatal[0].strip()[:120] if fatal else ""),
        Check("log_type_convert", WARN if conv else PASS, len(conv), 0,
              conv[0].strip()[:120] if conv else ""),
    ]


# --------------------------------------------------------- realism / detector
def check_realism(frames: Sequence[np.ndarray]) -> Check:
    """Report-only: how many realism stats land in the real CCTV band (docs/15)."""
    from ..realism.stats import frame_stats
    if not frames:
        return Check("realism_band", SKIP)
    rows = [frame_stats(f).as_dict() for f in frames]
    inside, out = [], []
    for k, (lo, hi) in REAL_BAND.items():
        v = float(np.median([r[k] for r in rows]))
        (inside if lo <= v <= hi else out).append(f"{k}={v:.3g}")
    return Check("realism_band", INFO, f"{len(inside)}/{len(REAL_BAND)}", "report only",
                 "out: " + ", ".join(out) if out else "all in band")


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def recall(gt: dict[Any, list], det: dict[Any, list], thr: float = 0.5) -> tuple[float, int]:
    """Greedy IoU recall of GT xywh boxes by detection xywh boxes."""
    hit = n = 0
    for k, boxes in gt.items():
        pool = list(det.get(k, []))
        for g in boxes:
            n += 1
            best = max(range(len(pool)), key=lambda j: iou(g, pool[j]), default=None)
            if best is not None and iou(g, pool[best]) >= thr:
                hit += 1
                pool.pop(best)
    return (hit / n if n else float("nan")), n


def check_detector(paths: dict[tuple[str, int], Path], gt: dict[tuple[str, int], list],
                   min_recall: float | None, model: str = "yolov8n.pt") -> Check:
    try:
        from ultralytics import YOLO
    except ImportError:
        return Check("detector_recall", SKIP, detail="ultralytics not installed")
    net = YOLO(model)
    det = {}
    for k, p in paths.items():
        r = net.predict(str(p), classes=[0], conf=0.25, verbose=False)[0]
        det[k] = [[x0, y0, x1 - x0, y1 - y0] for x0, y0, x1, y1 in r.boxes.xyxy.tolist()]
    rec, n = recall(gt, det)
    status = INFO if min_recall is None else ok(rec >= min_recall)
    return Check("detector_recall", status, round(rec, 3) if n else None,
                 min_recall if min_recall is not None else "report only",
                 f"{model} on {len(paths)} frames, {n} GT boxes >= {MIN_BOX_H} px")


# ------------------------------------------------------------------ wiring
def _world_counts(gt_dir: Path) -> dict[int, int] | None:
    p = gt_dir / "world.parquet"
    if not p.exists():
        return None
    import pandas as pd
    w = pd.read_parquet(p, columns=["frame", "uid"])
    return {int(k): int(v) for k, v in w.groupby("frame")["uid"].nunique().items()}


def _sample(xs: Sequence, n: int) -> list:
    if len(xs) <= n:
        return list(xs)
    return [xs[round(i)] for i in np.linspace(0, len(xs) - 1, n)]


def run_all(render_dir: Path, gt_dir: Path | None = None, log: Path | None = None,
            expected: int | None = None, frame_offset: int = 0, detector: bool = False,
            min_recall: float | None = None, det_frames: int = 5, realism: bool = True,
            ) -> list[Check]:
    fr = discover(render_dir)
    counts = {c: len(v) for c, v in fr.paths.items()}
    gaps = {c: (max(v) - min(v) + 1 - len(v)) for c, v in fr.paths.items() if v}
    checks = [check_frame_count(counts, expected, gaps)]
    if not counts:
        return checks
    mots: dict[str, dict] = {}
    if gt_dir is not None:
        for cam in fr.paths:
            p = gt_dir / f"mot_{cam}.txt"
            if p.exists():
                mots[cam] = read_mot(p)
    world = _world_counts(gt_dir) if gt_dir is not None else None

    cam_means, real_sample = {}, []
    for cam, idx_paths in sorted(fr.paths.items()):
        idx = sorted(idx_paths)
        frames: dict[int, np.ndarray] = {}
        native = None
        for i in idx:
            frames[i], native = load_small(idx_paths[i])
        seq = [frames[i] for i in idx]
        scale = WORK_WIDTH / native[0]
        pre = f"{cam}:"
        c = check_blank(seq)
        c.name = pre + c.name
        checks.append(c)
        # GT frame f <-> image index f - offset; computed before check_motion so it
        # can measure motion inside GT boxes (far/dim people don't move the whole
        # frame's average pixel).
        mot = {f - frame_offset: b for f, b in mots[cam].items()} if cam in mots else None
        by_id = _boxes_by_id(mot, scale) if mot is not None else None
        checks.append(check_motion(frames, name=pre + "motion", by_id=by_id, idx=idx))
        mid = seq[len(seq) // 2]
        checks.append(check_exposure(np.median(np.stack(_sample(seq, 9)), 0).astype(np.uint8),
                                     name=pre + "exposure"))
        checks.append(check_horizon(mid, name=pre + "horizon"))
        cam_means[cam] = np.mean(np.stack(_sample(seq, 15)), 0).astype(np.uint8)
        if realism:
            real_sample += [cv2.imread(str(idx_paths[i])) for i in _sample(idx, 3)]
        if gt_dir is None:
            continue
        if cam not in mots:
            checks.append(Check(pre + "gt_present", FAIL, detail=f"no mot_{cam}.txt in {gt_dir}"))
            continue
        checks.append(check_boxes_in_image(mots[cam], native, name=pre + "gt_boxes_in_image"))
        plate = np.median(np.stack(_sample(seq, 25)), 0).astype(np.uint8)
        checks.append(check_person_rendered(frames, plate, mot, scale, name=pre + "gt_person_rendered"))
        checks.append(check_animation(frames, mot, scale, name=pre + "animation"))
        if world is not None:
            checks.append(check_counts_vs_world(mots[cam], world, name=pre + "gt_count_vs_world"))
            checks.append(check_gt_frame_coverage(idx, set(world), frame_offset,
                                                  name=pre + "gt_frame_coverage"))
        del frames
    checks.append(check_views_differ(cam_means))
    if gt_dir is not None:
        checks.append(check_height_files(gt_dir, render_dir))
    if realism:
        checks.append(check_realism(real_sample))
    if log is not None:
        try:
            checks += check_log_text(Path(log).read_text(errors="replace"))
        except OSError as e:
            checks.append(Check("log_read", FAIL, detail=str(e)))
    if detector:
        paths, gt = {}, {}
        for cam, mot in mots.items():
            cand = [i for i in sorted(fr.paths[cam])
                    if any(b[4] >= MIN_BOX_H for b in mot.get(i + frame_offset, []))]
            for i in _sample(cand, det_frames):
                paths[(cam, i)] = fr.paths[cam][i]
                gt[(cam, i)] = [list(b[1:]) for b in mot[i + frame_offset] if b[4] >= MIN_BOX_H]
        checks.append(check_detector(paths, gt, min_recall) if paths
                      else Check("detector_recall", SKIP, detail="no GT boxes to score"))
    return checks

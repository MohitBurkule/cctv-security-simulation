"""Rendered frames -> DVR-looking per-camera MP4s, plus a realism report.

Layout in: ``<render_dir>/<cam>/0000.png`` (as written by tools/render_run.py;
``<run>/render`` is accepted as well). Out: ``<out>/<cam>.mp4``,
``<out>/degraded/<cam>/NNNN.png``, ``<out>/realism.md``, ``<out>/contact_sheet.png``.

Degradation is ``apply_sequence`` (camera-level effects coherent across frames,
sensor noise fresh per frame). The OSD is burned here rather than by the
profile so the clock advances with the frame index instead of being frozen.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import io
import multiprocessing
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .sensors.degrade import PROFILES, apply_sequence, osd

FRAME_EXTS = (".png", ".jpg", ".jpeg")


def camera_dirs(render_dir: Path) -> list[Path]:
    render_dir = Path(render_dir)
    if (render_dir / "render").is_dir():
        render_dir = render_dir / "render"
    return sorted(d for d in render_dir.iterdir()
                  if d.is_dir() and any(p.suffix.lower() in FRAME_EXTS for p in d.iterdir()))


def frame_paths(cam_dir: Path) -> list[Path]:
    return sorted(p for p in cam_dir.iterdir() if p.suffix.lower() in FRAME_EXTS)


def encode_mp4(frames_dir: Path, out: Path, fps: float = 12.5, crf: int = 30) -> Path:
    """H.264 yuv420p, the way a DVR stores it."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
           "-i", str(frames_dir / "%04d.png"), "-c:v", "libx264", "-preset", "medium",
           "-crf", str(crf), "-pix_fmt", "yuv420p", "-g", str(round(fps * 2)),
           "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    return out


def count_frames(mp4: Path) -> int:
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(mp4)],
                       check=True, capture_output=True, text=True)
    return int(r.stdout.strip())


def degrade_camera(cam_dir: Path, out_dir: Path, profile: str, seed: int, fps: float,
                   start: dt.datetime) -> list[Path]:
    base = PROFILES[profile]
    p = dataclasses.replace(base, osd=False, fps=fps)
    paths = frame_paths(cam_dir)
    frames = (cv2.imread(str(x)) for x in paths)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    label = cam_dir.name.upper().replace("_", "")
    for i, img in enumerate(apply_sequence(frames, p, sequence_seed=seed)):
        img = np.clip(img, 0, 255).astype(np.uint8)
        if base.osd:
            ts = (start + dt.timedelta(seconds=i / fps)).strftime("%Y-%m-%d %H:%M:%S")
            img = np.clip(osd(img, label, ts), 0, 255).astype(np.uint8)
        dst = out_dir / f"{i:04d}.png"
        cv2.imwrite(str(dst), img)
        written.append(dst)
    return written


def _fit(img: np.ndarray, w: int, h: int, label: str) -> np.ndarray:
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    r = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))), interpolation=cv2.INTER_AREA
                   if s < 1 else cv2.INTER_NEAREST)
    tile = np.zeros((h, w, 3), np.uint8)
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    tile[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    cv2.putText(tile, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(tile, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                cv2.LINE_AA)
    return tile


def _person_crop(img: np.ndarray, gt_boxes: list[tuple] | None) -> np.ndarray:
    """Crop around the largest GT box if we have one, else the centre third."""
    h, w = img.shape[:2]
    if gt_boxes:
        x, y, bw, bh = max(gt_boxes, key=lambda b: b[2] * b[3])
        cx, cy, side = x + bw / 2, y + bh / 2, max(bw, bh) * 1.6
    else:
        cx, cy, side = w / 2, h / 2, h / 3
    x0 = int(np.clip(cx - side * 0.8, 0, w - 1)); x1 = int(np.clip(cx + side * 0.8, x0 + 1, w))
    y0 = int(np.clip(cy - side / 2, 0, h - 1)); y1 = int(np.clip(cy + side / 2, y0 + 1, h))
    return img[y0:y1, x0:x1]


def _mot_boxes(mot: Path, frame: int) -> list[tuple]:
    if not mot.exists():
        return []
    out = []
    for line in mot.read_text().splitlines():
        f = line.split(",")
        if len(f) >= 6 and int(float(f[0])) == frame + 1:
            out.append(tuple(float(v) for v in f[2:6]))
    return out


def contact_sheet(real: Path | None, raw: Path, degraded: Path, out: Path,
                  gt_boxes: list[tuple] | None = None, tile=(640, 360)) -> Path:
    w, h = tile
    real_img = cv2.imread(str(real)) if real else None
    if real_img is None:
        real_img = np.zeros((h, w, 3), np.uint8)
    deg = cv2.imread(str(degraded))
    raw_img = cv2.imread(str(raw))
    # boxes are in raw-frame pixels; scale to the degraded frame
    boxes = gt_boxes
    if gt_boxes and raw_img.shape[:2] != deg.shape[:2]:
        sx, sy = deg.shape[1] / raw_img.shape[1], deg.shape[0] / raw_img.shape[0]
        boxes = [(x * sx, y * sy, bw * sx, bh * sy) for x, y, bw, bh in gt_boxes]
    tiles = [_fit(real_img, w, h, f"real: {real.stem if real else 'n/a'}"),
             _fit(raw_img, w, h, "ours: raw render"),
             _fit(deg, w, h, "ours: degraded"),
             _fit(_person_crop(deg, boxes), w, h, "ours: degraded crop (1:1 up)")]
    sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.imwrite(str(out), sheet)
    return out


def realism_report(real_dir: Path, raw_dir: Path, degraded_dir: Path, out: Path,
                   n: int = 60) -> str:
    from .realism.compare import main as compare_main

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        compare_main(["compare", "--real", str(real_dir), "--synthetic", str(raw_dir),
                      "--synthetic", str(degraded_dir), "--n", str(n)])
    table = buf.getvalue()
    out.write_text(
        "# Realism report\n\n"
        f"- real: `{real_dir}`\n- raw synthetic: `{raw_dir}`\n"
        f"- degraded synthetic: `{degraded_dir}`\n\n"
        "`ok` = synthetic median inside the real p10-p90 band; LOW/HIGH = outside.\n"
        "Generated by `survsim video` via `python -m survsim.realism compare`.\n\n"
        "```\n" + table + "```\n\n![contact sheet](contact_sheet.png)\n")
    return table


def _degrade_and_encode_one(args: tuple) -> tuple[str, Path, int]:
    """Worker body for the per-camera Pool in `make_videos` (module-level so it
    is importable/picklable by the spawned/forked worker process)."""
    cam, out, profile, seed, fps, start, crf = args
    ddir = out / "degraded" / cam.name
    written = degrade_camera(cam, ddir, profile, seed, fps, start)
    mp4 = encode_mp4(ddir, out / f"{cam.name}.mp4", fps, crf)
    return cam.name, mp4, len(written)


def make_videos(render_dir: Path, out: Path, profile: str = "cctv_day_720p", fps: float = 12.5,
                crf: int = 30, seed: int = 0, real_dir: Path | None = None,
                report: bool = True, workers: int | None = None) -> dict:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    start = dt.datetime(2026, 9, 17, 14, 2, 11)  # noqa: DTZ001 - fixed naive on-screen CCTV timestamp, not wall time
    cams = camera_dirs(render_dir)
    if not cams:
        raise FileNotFoundError(f"no <cam>/NNNN.png directories under {render_dir}")
    (out / "degraded").mkdir(parents=True, exist_ok=True)
    res: dict = {"videos": {}, "frames": {}}
    jobs = [(cam, out, profile, seed + k, fps, start, crf) for k, cam in enumerate(cams)]
    # Each camera's degrade+encode is independent (own frames, own tempdir, own
    # RNG streams keyed by its own sequence_seed), so cameras parallelise
    # cleanly across processes; capped at 4 to match the machine's real core
    # budget rather than one worker per camera-set.
    n_workers = max(1, min(workers or os.cpu_count() or 1, 4, len(jobs)))
    if n_workers > 1:
        with multiprocessing.Pool(n_workers) as pool:
            results = pool.map(_degrade_and_encode_one, jobs)
    else:
        results = [_degrade_and_encode_one(j) for j in jobs]
    for name, mp4, n_frames in results:
        res["videos"][name] = mp4
        res["frames"][name] = n_frames
    if report:
        real_dir = Path(real_dir or "~/data/real_cctv").expanduser()
        cam = cams[0]
        raw = frame_paths(cam)
        mid = len(raw) // 2
        reals = sorted(p for p in real_dir.glob("*") if p.suffix.lower() in FRAME_EXTS) \
            if real_dir.is_dir() else []
        run_root = cam.parent.parent if cam.parent.name == "render" else cam.parent
        boxes = _mot_boxes(run_root / "gt" / f"mot_{cam.name}.txt", mid)
        res["contact_sheet"] = contact_sheet(reals[0] if reals else None, raw[mid],
                                             out / "degraded" / cam.name / f"{mid:04d}.png",
                                             out / "contact_sheet.png", boxes)
        if reals:
            res["table"] = realism_report(real_dir, cam, out / "degraded" / cam.name,
                                          out / "realism.md")
            res["realism"] = out / "realism.md"
    return res

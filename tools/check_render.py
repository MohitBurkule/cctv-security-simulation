"""Check that rendered frames contain an image, not just a file.

Movie Render Queue reports `success=True` and writes correctly-named,
correctly-sized PNGs that are **entirely black**. That happened on the first
three attempts here, and every signal short of looking at the pixels said the
render had worked: exit code 0, 50 files, 1280x720, the pipeline log reporting
a resolved camera cut and a finished shot.

So this asks the only question that matters — is there a picture in there — and
it distinguishes the specific failure modes rather than reporting one boolean:

  - **black**: nothing rendered. Usually uncompiled shaders (check the project
    DerivedDataCache is not empty) or a renderer that never wrote the target.
  - **uniform**: rendered, but a single flat colour. Typically a camera inside
    geometry, or a scene with no lights where only the sky clear-colour shows.
  - **static**: rendered and lit, but every frame identical. The sequence has no
    animation bound, or the camera cut is pinned to one time.
  - **ok**: lit, varied, and changing over time.

    .venv/bin/python tools/check_render.py /tmp/survsim_render
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# A frame whose mean is below this is black for any practical purpose; sky and
# lit ground both sit far above it.
BLACK_MEAN = 2.0
# Below this standard deviation the frame carries no structure.
UNIFORM_STD = 1.0
# Mean absolute difference between first and last frame, below which nothing
# moved. Compression noise alone stays well under 1.
STATIC_DIFF = 0.5


def inspect(directory: Path, pattern: str = "*.png") -> dict[str, Any]:
    paths = sorted(directory.glob(pattern))
    if not paths:
        return {"verdict": "missing", "n_frames": 0,
                "reason": f"no files matching {pattern} in {directory}"}

    first = cv2.imread(str(paths[0]))
    last = cv2.imread(str(paths[-1]))
    if first is None or last is None:
        return {"verdict": "unreadable", "n_frames": len(paths),
                "reason": "the files exist but are not decodable images"}

    mean = float(first.mean())
    std = float(first.std())
    unique = int(len(np.unique(first)))
    diff = float(np.abs(first.astype(float) - last.astype(float)).mean())

    if mean < BLACK_MEAN:
        verdict, reason = "black", (
            "nothing was rendered — check the project DerivedDataCache is not "
            "empty (uncompiled shaders render black while still reporting "
            "success)")
    elif std < UNIFORM_STD:
        verdict, reason = "uniform", (
            "a single flat colour — camera may be inside geometry, or the "
            "scene has no lights and only the sky clear-colour is visible")
    elif diff < STATIC_DIFF:
        verdict, reason = "static", (
            "lit but identical every frame — no animation is bound, or the "
            "camera cut is pinned to one time")
    else:
        verdict, reason = "ok", None

    return {
        "verdict": verdict, "reason": reason,
        "n_frames": len(paths),
        "resolution": [int(first.shape[1]), int(first.shape[0])],
        "mean": round(mean, 2), "std": round(std, 2),
        "unique_values": unique,
        "first_last_diff": round(diff, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    ap.add_argument("--pattern", default="*.png")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    report = inspect(args.directory, args.pattern)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for key, value in report.items():
            if value is not None:
                print(f"  {key:<16} {value}")
    # Non-zero for anything but a real render, so a script can gate on it.
    return 0 if report["verdict"] == "ok" else 1



def compare_cameras(root: Path, pattern: str = "*.png") -> dict[str, Any]:
    """Check that the cameras in a rig actually see different things.

    A rig whose cameras all render the same viewpoint is the worst kind of
    broken: every file differs by a few bits of anti-aliasing noise, so
    checksums pass, while the frames are the same picture. It happened here —
    Movie Render Queue ignores the camera on a job unless the sequence carries a
    `MovieSceneCameraCutTrack`, and without one an eight-camera rig silently
    becomes one viewpoint copied eight times. Every multi-camera tracking,
    handoff and overlap number computed from that would be meaningless.

    Per-frame mean and standard deviation are compared rather than checksums,
    because those are what stay identical when the content is identical and the
    noise is not.
    """
    cameras = sorted(p for p in Path(root).iterdir() if p.is_dir())
    if len(cameras) < 2:
        return {"verdict": "single_camera", "n_cameras": len(cameras),
                "reason": "nothing to compare"}

    stats: dict[str, tuple[float, float]] = {}
    for camera in cameras:
        frames = sorted(camera.glob(pattern))
        if not frames:
            continue
        image = cv2.imread(str(frames[len(frames) // 2]))
        if image is None:
            continue
        stats[camera.name] = (round(float(image.mean()), 2),
                              round(float(image.std()), 2))

    duplicates = []
    names = list(stats)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if stats[a] == stats[b]:
                duplicates.append((a, b))

    return {
        "verdict": "duplicate_views" if duplicates else "ok",
        "n_cameras": len(stats),
        "per_camera": stats,
        "duplicates": duplicates,
        "reason": (
            "cameras render identical images — the sequence is probably missing "
            "a MovieSceneCameraCutTrack, so MRQ ignores the job's camera"
            if duplicates else None),
    }

if __name__ == "__main__":
    sys.exit(main())

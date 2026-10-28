"""Build a detection/MOT/ReID dataset from seeded scenario runs.

Layout written under `out`::

    manifest.json
    runs/<scenario>_s<seed>/...            # the full `survsim run` output
    frames/<run>/<camera>/<frame:06d>.jpg  # images the baselines consume
    coco/<run>.json                        # COCO person annotations

Frame numbers are MOT frame numbers (1-based). A render from
`tools/render_run.py` names frames `<frame_number>.png` 0-based per camera
(`render/cam_01/0000.png`), so render index i is GT frame i + 1.

GT convention (MOTChallenge): a row is a scoring *target* when conf == 1 and
class == 1 (pedestrian); classes 2, 7, 8, 12 (person on vehicle, static
person, distractor, reflection) are *ignore* regions, as in MOT20 evaluation.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

IGNORE_CLASSES = {2, 7, 8, 12}


def split_for_seed(index: int, n_seeds: int) -> str:
    """Splits are by seed, never by frame: frames of one run are near-duplicates,
    so a frame-level split leaks the test set into training."""
    if n_seeds <= 1:
        return "test"
    if index == n_seeds - 1:
        return "test"
    if n_seeds >= 3 and index == n_seeds - 2:
        return "val"
    return "train"


def read_gt(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (targets, ignore) rows from a MOT gt.txt."""
    targets: list[dict[str, Any]] = []
    ignore: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        v = [float(x) for x in line.split(",")]
        row = {"frame": int(v[0]), "id": int(v[1]), "bbox": v[2:6],
               "vis": v[8] if len(v) > 8 else 1.0}
        conf = v[6] if len(v) > 6 else 1.0
        cls = int(v[7]) if len(v) > 7 else 1
        if conf >= 1 and cls == 1:
            targets.append(row)
        elif cls in IGNORE_CLASSES:
            ignore.append(row)
    return targets, ignore


def load_manifest(dataset_dir: Path) -> dict[str, Any]:
    m = json.loads((Path(dataset_dir) / "manifest.json").read_text())
    m["_root"] = str(Path(dataset_dir).resolve())
    return m


def _paint_frame(res: tuple[int, int], rows: Sequence[dict[str, Any]]) -> np.ndarray:
    """Mock frame: grey ground with per-identity silhouettes at the GT boxes.

    Not a render. Only exists so the pipeline runs end to end without an engine.
    """
    from ..backends.mock.backend import MockBackend

    w, h = res
    img = np.full((h, w, 3), 110, np.uint8)
    for r in sorted(rows, key=lambda r: r["bbox"][1] + r["bbox"][3]):   # far first
        rng = np.random.default_rng(int(r["id"]))
        colour = tuple(int(c) for c in rng.integers(20, 235, 3))
        MockBackend._paint_person(img, *r["bbox"], colour)
    return img


def _read_licences(out: Path, frames_dir: Path | None, run_dir: Path, run_name: str) -> dict[str, Any]:
    """Per-set licences.json (docs/research/47): rendered sets carry one written by
    tools/render_run.py next to their `render/` dir; mock-only runs carry one
    written by survsim/gt/writers.py under `meta/` (always CC0/procedural, so
    always ``commercial_ml_ok``). Missing entirely -> refuse to assume it is OK.
    """
    candidates = []
    if frames_dir is not None:
        candidates.append(Path(frames_dir) / run_name / "licences.json")
    candidates.append(run_dir / "meta" / "licences.json")
    for p in candidates:
        if p.is_file():
            try:
                return json.loads(p.read_text())
            except (OSError, ValueError):
                continue
    return {"asset_licences": ["LicenseRef-Unknown"], "commercial_ml_ok": False,
            "attributions": [], "blocking_assets": ["<no licences.json found>"]}


_RENDER_RE = re.compile(r"(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)


def _render_frames(frames_dir: Path, run_name: str, cam: str) -> dict[int, Path]:
    """Find rendered frames for a camera; accepts <dir>/<run>/<cam>/ or <dir>/<cam>/."""
    for base in (frames_dir / run_name / cam, frames_dir / run_name / "render" / cam,
                 frames_dir / cam):
        if base.is_dir():
            out = {}
            for p in base.iterdir():
                m = _RENDER_RE.search(p.name)
                if m:
                    out[int(m.group(1)) + 1] = p
            return out
    return {}


def build_dataset(scenarios: Sequence[Path], seeds: int, out: Path,
                  frames_dir: Path | None = None, profile: str | None = None,
                  max_frames: int | None = 300, base_seed: int = 0) -> dict[str, Any]:
    import cv2

    from ..gt.coco import CocoBuilder
    from ..simulator import Simulator

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario = Path(scenario)
        for i in range(seeds):
            seed = base_seed + i
            name = f"{scenario.stem}_s{seed}"
            run_dir = out / "runs" / name
            sim = Simulator.from_yaml(scenario, seed=seed, engine="mock")
            sim.record(run_dir, topdown_video=False)
            cams = {c.id: c for c in sim.scenario.cameras}
            coco = CocoBuilder(dataset_name=name)
            entry: dict[str, Any] = {"name": name, "scenario": str(scenario), "seed": seed,
                                     "split": split_for_seed(i, seeds), "cameras": {}}
            for gt_path in sorted((run_dir / "gt").glob("mot_*.txt")):
                cam_id = gt_path.stem[len("mot_"):]
                calib = json.loads((run_dir / "calib" / f"{cam_id}.json").read_text())
                res = tuple(calib["resolution"])
                targets, _ = read_gt(gt_path)
                by_frame: dict[int, list] = {}
                for r in targets:
                    by_frame.setdefault(r["frame"], []).append(r)
                frames = sorted(by_frame)
                renders = _render_frames(Path(frames_dir), name, cam_id) if frames_dir else {}
                if renders:
                    frames = sorted(set(frames) | set(renders))
                    frames = [f for f in frames if f in renders]
                if max_frames:
                    frames = frames[:max_frames]
                cam_obj = cams.get(cam_id)
                listed = []
                for f in frames:
                    dst = out / "frames" / name / cam_id / f"{f:06d}.jpg"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if renders:
                        img = cv2.imread(str(renders[f]))
                        if profile:
                            from ..sensors.degrade import apply_profile
                            img = apply_profile(img, profile, camera=cam_obj, seed=seed * 100003 + f)
                    else:
                        img = _paint_frame(res, by_frame.get(f, []))
                    cv2.imwrite(str(dst), img)
                    listed.append(str(dst.relative_to(out)))
                    iid = coco.add_image(cam_id, f, *res, file_name=listed[-1])
                    for r in by_frame.get(f, []):
                        coco.add_person(iid, str(r["id"]), tuple(r["bbox"]))
                entry["cameras"][cam_id] = {
                    "gt": str(gt_path.relative_to(out)), "fps": calib.get("fps", 10.0),
                    "resolution": list(res), "frames": dict(zip(map(str, frames), listed)),
                }
            coco_path = coco.write(out / "coco" / f"{name}.json")
            entry["coco"] = str(Path(coco_path).relative_to(out))
            entry["licences"] = _read_licences(out, frames_dir, run_dir, name)
            runs.append(entry)
    all_licences = sorted({i for r in runs for i in r["licences"]["asset_licences"]})
    commercial_ml_ok = bool(runs) and all(r["licences"]["commercial_ml_ok"] for r in runs)
    manifest = {
        "kind": "survsim",
        "synthetic_topdown": not frames_dir,
        "note": ("mock silhouettes, NOT renders: numbers measure the pipeline, not a model"
                 if not frames_dir else f"rendered frames, degrade profile={profile}"),
        "degrade_profile": profile, "runs": runs,
        "asset_licences": all_licences, "commercial_ml_ok": commercial_ml_ok,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def from_motchallenge(root: Path, out: Path, split: str = "train",
                      max_frames: int | None = None) -> dict[str, Any]:
    """Wrap MOTChallenge sequences in a manifest (frames referenced, not copied)."""
    import configparser

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for seq in sorted(p for p in (Path(root) / split).iterdir() if (p / "gt" / "gt.txt").is_file()):
        ini = configparser.ConfigParser()
        ini.read(seq / "seqinfo.ini")
        s = ini["Sequence"]
        n = int(s["seqLength"])
        if max_frames:
            n = min(n, max_frames)
        ext = s.get("imExt", ".jpg")
        frames = {str(f): str((seq / s.get("imDir", "img1") / f"{f:06d}{ext}").resolve())
                  for f in range(1, n + 1)}
        runs.append({"name": seq.name, "scenario": "MOTChallenge", "seed": None, "split": "test",
                     "cameras": {seq.name: {"gt": str((seq / "gt" / "gt.txt").resolve()),
                                            "fps": float(s["frameRate"]),
                                            "resolution": [int(s["imWidth"]), int(s["imHeight"])],
                                            "frames": frames}}})
    manifest = {"kind": "motchallenge", "synthetic_topdown": False, "source": str(root),
                "note": "real footage reference row", "runs": runs,
                # MOTChallenge (MOT20 etc.) ships under CC BY-NC-SA 3.0: fine for internal
                # metric validation, not for training/testing a model that is sold
                # (docs/research/47 "Recommended compliant character stack" table).
                "asset_licences": ["CC-BY-NC-SA-3.0"], "commercial_ml_ok": False}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest

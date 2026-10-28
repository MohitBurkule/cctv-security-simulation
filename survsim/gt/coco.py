"""COCO export, with the fields that make surveillance results interpretable.

Standard COCO gives boxes, masks and keypoints. That is enough to *train*, but
not enough to understand a result: a detector reported at "mAP 0.62" on
surveillance footage is meaningless without knowing how much of the set was
people 18 pixels tall in the dark behind a railing.

So every annotation also carries the difficulty factors the research says
dominate performance — pixel height (R01: detection collapses below ~20-30 px),
occlusion (CityPersons heavy-occlusion LAMR is ~5x the reasonable subset), and
lighting — and `difficulty_buckets()` turns them into strata so results can be
reported per-bucket instead of averaged into a single flattering number.

Two box conventions are emitted deliberately:

* `bbox`      — the *visible* extent, what a detector is scored against.
* `amodal_bbox` — the full extent including the occluded part.

The ratio between them is the occlusion measure, and having both is what lets
you separate "the detector missed them" from "there was nothing to see".
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# COCO's 17-keypoint person skeleton, the de-facto standard.
COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
COCO_SKELETON = (
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13), (6, 7),
    (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7),
)

# Pixel-height strata. The boundaries are the ones the literature actually uses:
# ~20 px is where general detectors start failing, and the DORI bands above that
# are what procurement specifies.
HEIGHT_BUCKETS = (
    ("unusable", 0.0, 20.0),
    ("tiny", 20.0, 40.0),
    ("small", 40.0, 80.0),
    ("medium", 80.0, 160.0),
    ("large", 160.0, 1e9),
)
OCCLUSION_BUCKETS = (
    ("none", 0.0, 0.1),
    ("partial", 0.1, 0.45),
    ("heavy", 0.45, 0.8),
    ("severe", 0.8, 1.01),
)


def bucket_of(value: float, buckets: Iterable[tuple[str, float, float]]) -> str:
    for name, low, high in buckets:
        if low <= value < high:
            return name
    return buckets[-1][0] if buckets else "unknown"  # type: ignore[index]


@dataclass
class CocoBuilder:
    """Accumulates COCO annotations across frames and cameras."""

    dataset_name: str = "survsim"
    keypoint_names: tuple[str, ...] = COCO_KEYPOINTS

    def __post_init__(self) -> None:
        self.images: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []
        self._image_index: dict[tuple[str, int], int] = {}
        self._uid_to_track: dict[str, int] = {}

    # ---------------------------------------------------------------- images

    def add_image(self, camera: str, frame: int, width: int, height: int,
                  file_name: str | None = None, **extra: Any) -> int:
        key = (camera, frame)
        if key in self._image_index:
            return self._image_index[key]
        image_id = len(self.images) + 1
        self._image_index[key] = image_id
        self.images.append({
            "id": image_id,
            "file_name": file_name or f"{camera}/{frame:06d}.jpg",
            "width": int(width), "height": int(height),
            "camera": camera, "frame": int(frame), **extra,
        })
        return image_id

    def track_id(self, uid: str) -> int:
        return self._uid_to_track.setdefault(uid, len(self._uid_to_track) + 1)

    # ----------------------------------------------------------- annotations

    def add_person(
        self,
        image_id: int,
        uid: str,
        bbox: tuple[float, float, float, float],
        *,
        amodal_bbox: tuple[float, float, float, float] | None = None,
        keypoints: np.ndarray | None = None,
        px_height: float | None = None,
        lighting_lux: float | None = None,
        appearance_epoch: int = 0,
        behaviour: str | None = None,
        segmentation: list | None = None,
        area: float | None = None,
    ) -> dict[str, Any]:
        x, y, w, h = (float(v) for v in bbox)
        visible_area = float(area) if area is not None else w * h

        occlusion = 0.0
        if amodal_bbox is not None:
            full = float(amodal_bbox[2]) * float(amodal_bbox[3])
            if full > 0:
                # Occlusion is the share of the person's true extent that is hidden.
                occlusion = float(np.clip(1.0 - (w * h) / full, 0.0, 1.0))

        height_px = float(px_height) if px_height is not None else h
        ann: dict[str, Any] = {
            "id": len(self.annotations) + 1,
            "image_id": image_id,
            "category_id": 1,
            "iscrowd": 0,
            "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
            "area": round(visible_area, 2),
            "track_id": self.track_id(uid),
            "uid": uid,
            "appearance_epoch": int(appearance_epoch),
            "occlusion": round(occlusion, 3),
            "px_height": round(height_px, 1),
            "difficulty": {
                "height_bucket": bucket_of(height_px, HEIGHT_BUCKETS),
                "occlusion_bucket": bucket_of(occlusion, OCCLUSION_BUCKETS),
                "lighting_lux": None if lighting_lux is None else round(float(lighting_lux), 1),
            },
        }
        if amodal_bbox is not None:
            ann["amodal_bbox"] = [round(float(v), 2) for v in amodal_bbox]
        if behaviour:
            ann["behaviour"] = behaviour
        if segmentation is not None:
            ann["segmentation"] = segmentation
        if keypoints is not None:
            flat, n_visible = self._flatten_keypoints(keypoints)
            ann["keypoints"] = flat
            ann["num_keypoints"] = n_visible

        self.annotations.append(ann)
        return ann

    @staticmethod
    def _flatten_keypoints(kp: np.ndarray) -> tuple[list[float], int]:
        """COCO stores keypoints flat as x,y,v with v in {0 absent, 1 occluded, 2 visible}."""
        arr = np.asarray(kp, dtype=float).reshape(-1, 3)
        flat: list[float] = []
        visible = 0
        for x, y, v in arr:
            vis = round(v)
            flat += [round(float(x), 2), round(float(y), 2), vis]
            visible += int(vis > 0)
        return flat, visible

    # -------------------------------------------------------------- analysis

    def difficulty_buckets(self) -> dict[str, dict[str, int]]:
        """Counts per stratum — report results against these, not a single mean."""
        heights: dict[str, int] = {}
        occlusions: dict[str, int] = {}
        for ann in self.annotations:
            d = ann["difficulty"]
            heights[d["height_bucket"]] = heights.get(d["height_bucket"], 0) + 1
            occlusions[d["occlusion_bucket"]] = occlusions.get(d["occlusion_bucket"], 0) + 1
        return {"height": heights, "occlusion": occlusions}

    def usable_fraction(self) -> float:
        """Share of annotations a general detector could plausibly find at all."""
        if not self.annotations:
            return 0.0
        unusable = sum(a["difficulty"]["height_bucket"] == "unusable" for a in self.annotations)
        return round(1.0 - unusable / len(self.annotations), 4)

    # ----------------------------------------------------------------- write

    def to_dict(self) -> dict[str, Any]:
        return {
            "info": {
                "description": f"{self.dataset_name} synthetic surveillance",
                "note": "bbox is the visible extent; amodal_bbox includes occluded parts",
            },
            "licenses": [],
            "images": self.images,
            "annotations": self.annotations,
            "categories": [{
                "id": 1, "name": "person", "supercategory": "person",
                "keypoints": list(self.keypoint_names),
                "skeleton": [list(p) for p in COCO_SKELETON],
            }],
            "survsim": {
                "difficulty_buckets": self.difficulty_buckets(),
                "usable_fraction": self.usable_fraction(),
                "n_identities": len(self._uid_to_track),
            },
        }

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1))
        return path


def from_camera_rows(
    rows: Iterable[dict[str, Any]],
    resolution: tuple[int, int],
    dataset_name: str = "survsim",
    include_non_person: bool = False,
) -> CocoBuilder:
    """Build a COCO set from the per-camera rows a backend's `observe()` returns.

    Rows carry ``object_class`` ("person" unless the agent is a wildlife
    distractor, in which case it is "animal" — see `Agent.object_class`).
    Non-person rows are MOT class 8 ("distractor"), ``consider=0`` — i.e.
    ignore regions, never a scoring target (`survsim dataset` treats them the
    same way; see `IGNORE_CLASSES` in `survsim/datasets/build.py`). Exporting
    them as COCO "person" annotations would silently teach a detector to find
    wildlife, so by default they are skipped. Pass ``include_non_person=True``
    to keep them (still under category "person" — callers doing that must
    filter by their own means; this export has no other category yet).
    """
    builder = CocoBuilder(dataset_name=dataset_name)
    for row in rows:
        if not include_non_person and row.get("object_class", "person") != "person":
            continue
        image_id = builder.add_image(row["camera"], int(row["frame"]), *resolution)
        bbox = tuple(float(v) for v in row["bbox"])
        # `fill_ratio` (mask area over box area) is the backend's occlusion proxy;
        # convert it into an amodal box so the exported occlusion is meaningful.
        amodal = None
        fill = row.get("fill_ratio")
        if fill and 0 < float(fill) < 1:
            scale = 1.0 / float(fill)
            amodal = (bbox[0], bbox[1], bbox[2] * scale**0.5, bbox[3] * scale**0.5)
        builder.add_person(
            image_id, row["uid"], bbox,  # type: ignore[arg-type]
            amodal_bbox=amodal,
            px_height=row.get("px_height"),
            appearance_epoch=int(row.get("appearance_epoch", 0)),
            behaviour=row.get("label"),
            area=row.get("mask_area_px"),
        )
    return builder

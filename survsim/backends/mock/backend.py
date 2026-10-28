"""Mock backend: no engine, no GPU.

It renders a top-down schematic instead of photorealistic pixels, but it produces
the *same* ground truth as the Unreal backend — projected boxes, pixel heights,
visibility. That makes it the right place to develop behaviours, verify the
projection maths, and run CI on a machine with no GPU at all.

Occlusion is resolved with a coarse depth-ordered painter's algorithm covering
both people and scenery props, so `visible_frac` is meaningful in a crowd or
behind a pillar. It works on bounding boxes rather than silhouettes, so a thin
railing occludes more than it really would; anything needing pixel-accurate
occlusion should be validated on the Unreal backend's rendered mask.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ...core.geometry import pixel_height, project
from ...core.models import Camera, Scenario
from ...director.world import World
from ..base import bbox_from_points


class MockBackend:
    name = "mock"

    def __init__(self, canvas_px: int = 900, margin_m: float = 5.0) -> None:
        self.canvas_px = canvas_px
        self.margin_m = margin_m
        self._scenario: Scenario | None = None
        self._bounds: tuple[float, float, float, float] | None = None

    # ------------------------------------------------------------------ setup

    # Occluding scenery, with plausible heights and footprints in metres. A car
    # park is mostly pillars, and a rig evaluated without them looks far better
    # than it is.
    OCCLUDER_SIZES: ClassVar[dict[str, tuple[float, float]]] = {
        "pillar": (0.6, 3.0),
        "fence_panel": (3.0, 2.4),
        "cabinet": (1.2, 1.8),
        "bench": (1.8, 0.9),
        "door": (1.2, 2.1),
        "wall": (4.0, 3.0),
        "vehicle": (4.5, 1.5),
    }

    def open(self, scenario: Scenario) -> None:
        self._scenario = scenario
        xs: list[float] = []
        ys: list[float] = []
        for z in scenario.environment.zones:
            xs += [p[0] for p in z.polygon]
            ys += [p[1] for p in z.polygon]
        for w in list(scenario.environment.spawn_points) + list(scenario.environment.exits):
            xs.append(w.pos[0])
            ys.append(w.pos[1])
        for c in scenario.cameras:
            xs.append(c.mount.pos[0])
            ys.append(c.mount.pos[1])
        if not xs:
            xs, ys = [-10.0, 10.0], [-10.0, 10.0]
        m = self.margin_m
        self._bounds = (min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m)

    def sync(self, world: World) -> None:
        """No engine to push to; the Director's state is already authoritative."""

    def close(self) -> None:
        self._scenario = None

    # --------------------------------------------------------------- capture

    def capture(self, cameras: list[Camera], modalities: set[str]) -> dict[str, dict[str, Any]]:
        """Mock capture returns empty frames; use `render_topdown` for a visual."""
        out: dict[str, dict[str, Any]] = {}
        for cam in cameras:
            w, h = cam.capture.res
            frames: dict[str, Any] = {}
            if "rgb" in modalities:
                frames["rgb"] = np.zeros((h, w, 3), dtype=np.uint8)
            out[cam.id] = frames
        return out

    # ------------------------------------------------------------ ground truth

    def observe(self, world: World, camera: Camera) -> list[dict[str, Any]]:
        """Per-camera ground truth, including person-on-person occlusion.

        People *and* scenery are drawn into one depth-ordered buffer as filled
        boxes, nearest last, and the visible fraction is what survives. Ignoring
        either makes every mock-backend difficulty figure optimistic: crowding is
        the commonest occluder outdoors, pillars are in a car park.
        """
        import cv2

        res = camera.capture.res
        candidates: list[dict[str, Any]] = []

        for agent in world.agents.values():
            hull = self._body_points(agent.pos3, agent.height_m, agent.spec.body.shoulder_w)
            uv, depth = project(camera, hull)
            if depth.min() <= 0:
                continue
            inside = (
                (uv[:, 0] >= 0) & (uv[:, 0] < res[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < res[1])
            )
            if not inside.any():
                continue
            x, y, bw, bh = bbox_from_points(uv, res)
            if bw <= 1 or bh <= 1:
                continue
            candidates.append({
                "agent": agent, "bbox": (x, y, bw, bh),
                "depth": float(depth.mean()), "truncation": float(1.0 - inside.mean()),
            })

        # Scenery is painted alongside people in the same depth order, so a person
        # behind a pillar is occluded by it exactly as they would be by another
        # person. Without this a car park's difficulty is badly understated.
        occluders = self._scenery_boxes(camera)
        painted = ([{"kind": "prop", **o} for o in occluders]
                   + [{"kind": "person", "index": i, **c} for i, c in enumerate(candidates)])
        painted.sort(key=lambda c: -c["depth"])

        # Each person's unoccluded area, measured at their *actual* coordinates.
        # Computing it from a separately-rasterised scratch copy leaves a few
        # percent of rounding error, which then reads as phantom occlusion on
        # someone standing entirely alone.
        own_area: dict[int, int] = {}
        scratch = np.full((res[1], res[0]), -1, dtype=np.int32)
        for index, c in enumerate(candidates):
            scratch[:] = -1
            x, y, bw, bh = c["bbox"]
            self._paint_person(scratch, x, y, bw, bh, index)
            own_area[index] = int((scratch == index).sum())

        buffer = np.full((res[1], res[0]), -1, dtype=np.int32)
        for item in painted:
            x, y, bw, bh = item["bbox"]
            label = item["index"] if item["kind"] == "person" else -2
            if item["kind"] == "person":
                self._paint_person(buffer, x, y, bw, bh, label)
            else:
                cv2.rectangle(buffer, (int(x), int(y)), (int(x + bw), int(y + bh)),
                              label, -1)

        rows: list[dict[str, Any]] = []
        for index, c in enumerate(candidates):
            agent = c["agent"]
            x, y, bw, bh = c["bbox"]
            # The denominator is the person's *own silhouette* area, not their
            # bounding box. Using the box would score an unoccluded person as ~30%
            # occluded simply because a body is not rectangular.
            area = max(own_area.get(index, 0), 1)
            visible_px = int((buffer == index).sum())
            occlusion = float(np.clip(1.0 - visible_px / area, 0.0, 1.0))
            # Fully hidden people are still recorded, flagged rather than dropped:
            # a tracker is expected to carry them through, and silently deleting
            # them would make occlusion handling untestable.
            rows.append(
                {
                    "frame": world.tick,
                    "t": round(world.t, 3),
                    "camera": camera.id,
                    "uid": agent.uid,
                    "bbox": (round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)),
                    "visible_frac": round(float(max(0.0, 1.0 - occlusion - c["truncation"])), 3),
                    "occlusion": round(occlusion, 3),
                    "truncation": round(c["truncation"], 3),
                    "fully_occluded": visible_px == 0,
                    "px_height": round(pixel_height(camera, agent.pos3, agent.height_m), 2),
                    "range_m": round(c["depth"], 2),
                    "label": agent.label,
                    "object_class": agent.object_class,
                }
            )
        return rows

    @staticmethod
    def _paint_person(buffer: np.ndarray, x: float, y: float, w: float, h: float,
                      label: int) -> None:
        """Paint a body-shaped silhouette rather than a filled rectangle.

        A box occludes the empty corners beside someone's head and between their
        legs, which overstates how much they hide. A head ellipse over a tapered
        torso covers roughly the right area — a standing person fills about 55% of
        their bounding box, and a rectangle claims 100%.

        Still an approximation: no arms, no pose. Pixel-accurate occlusion needs
        the Unreal backend's rendered mask.
        """
        import cv2

        x, y, w, h = float(x), float(y), float(w), float(h)
        if w < 2 or h < 2:
            cv2.rectangle(buffer, (int(x), int(y)), (int(x + w), int(y + h)), label, -1)
            return

        cx = x + w / 2.0
        head_h = h * 0.16
        head_r = min(w * 0.30, head_h * 0.62)
        cv2.ellipse(buffer, (int(cx), int(y + head_h * 0.55)),
                    (max(int(head_r), 1), max(int(head_h * 0.62), 1)),
                    0, 0, 360, label, -1)

        # Torso and legs as a trapezium: shoulders wider than the stance.
        top = y + head_h * 0.9
        shoulder = w * 0.46
        foot = w * 0.30
        body = np.array([
            [cx - shoulder, top], [cx + shoulder, top],
            [cx + foot, y + h], [cx - foot, y + h],
        ], dtype=np.int32)
        cv2.fillConvexPoly(buffer, body, label)

    def _scenery_boxes(self, camera: Camera) -> list[dict[str, Any]]:
        """Project props that can hide a person into image-space boxes."""
        if self._scenario is None:
            return []
        out: list[dict[str, Any]] = []
        res = camera.capture.res
        for prop in self._scenario.environment.props:
            size = self.OCCLUDER_SIZES.get(prop.type)
            if size is None:
                continue
            width, height = size
            if prop.length_m:                      # fences declare their own span
                width = float(prop.length_m)
            half = width / 2.0
            corners = np.array([
                [prop.pos[0] + dx, prop.pos[1] + dy, z]
                for dx in (-half, half) for dy in (-half, half) for z in (0.0, height)
            ])
            uv, depth = project(camera, corners)
            if depth.min() <= 0:
                continue
            x, y, bw, bh = bbox_from_points(uv, res)
            if bw <= 1 or bh <= 1:
                continue
            out.append({"bbox": (x, y, bw, bh), "depth": float(depth.mean()), "id": prop.id})
        return out

    @staticmethod
    def _body_points(foot: np.ndarray, height: float, shoulder: float) -> np.ndarray:
        """Eight corners of a person-sized box; enough for an amodal bounding box."""
        hw = shoulder / 2.0
        hd = shoulder / 3.0
        corners = []
        for dx in (-hd, hd):
            for dy in (-hw, hw):
                for dz in (0.0, height):
                    corners.append([foot[0] + dx, foot[1] + dy, dz])
        return np.asarray(corners, dtype=float)

    # ----------------------------------------------------------- top-down view

    def render_topdown(self, world: World, size: int | None = None) -> np.ndarray:
        """Schematic plan view: zones, camera footprints, people, dropped objects."""
        import cv2

        assert self._scenario is not None and self._bounds is not None
        size = size or self.canvas_px
        img = np.full((size, size, 3), 24, dtype=np.uint8)
        x0, y0, x1, y1 = self._bounds
        scale = (size - 20) / max(x1 - x0, y1 - y0, 1e-6)

        def to_px(p) -> tuple[int, int]:
            px = int(10 + (float(p[0]) - x0) * scale)
            py = int(size - 10 - (float(p[1]) - y0) * scale)
            return px, py

        for zone in self._scenario.environment.zones:
            pts = np.array([to_px(p) for p in zone.polygon], dtype=np.int32)
            colour = {
                "restricted": (40, 40, 160),
                "gate": (30, 110, 140),
                "perimeter_fence": (60, 60, 200),
                "blindspot": (70, 40, 90),
            }.get(zone.tag, (55, 55, 55))
            cv2.fillPoly(img, [pts], colour)
            cv2.polylines(img, [pts], True, (110, 110, 110), 1)

        for cam in self._scenario.cameras:
            c = to_px(cam.mount.pos)
            half = np.radians(cam.hfov_deg) / 2.0
            yaw = np.radians(cam.mount.yaw)
            reach = 18.0 * scale
            for sign in (-1, 1):
                a = yaw + sign * half
                cv2.line(img, c, (int(c[0] + reach * np.cos(a)), int(c[1] - reach * np.sin(a))),
                         (0, 170, 255), 1)
            cv2.circle(img, c, 4, (0, 200, 255), -1)
            cv2.putText(img, cam.id, (c[0] + 6, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (0, 200, 255), 1, cv2.LINE_AA)

        for obj in world.objects.values():
            cv2.drawMarker(img, to_px(obj["pos"]), (0, 140, 255), cv2.MARKER_SQUARE, 9, 2)

        for agent in world.agents.values():
            p = to_px(agent.pos)
            hostile = agent.label.startswith(("suspicious", "hostile"))
            cv2.circle(img, p, 5, (70, 90, 255) if hostile else (140, 230, 140), -1)
            cv2.putText(img, agent.uid, (p[0] + 7, p[1] + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (210, 210, 210), 1, cv2.LINE_AA)

        cv2.putText(img, f"t={world.t:6.2f}s  tick={world.tick}", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        return img

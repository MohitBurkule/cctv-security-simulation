"""Backend interface.

Everything above this line is engine-agnostic. UnrealBackend, IsaacBackend and
MockBackend all satisfy this protocol, so a scenario that runs on the mock
backend runs unchanged against Unreal.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..core.models import Camera, Scenario
from ..director.world import World


@runtime_checkable
class Backend(Protocol):
    name: str

    def open(self, scenario: Scenario) -> None:
        """Load the environment and place cameras."""

    def sync(self, world: World) -> None:
        """Push the Director's agent state into the engine for this tick."""

    def capture(self, cameras: list[Camera], modalities: set[str]) -> dict[str, dict[str, Any]]:
        """Return {camera_id: {modality: image}}; images are HxWx3 uint8 or HxW float."""

    def observe(self, world: World, camera: Camera) -> list[dict[str, Any]]:
        """Per-camera ground truth for this tick: boxes, visibility, pixel height."""

    def close(self) -> None: ...


def bbox_from_points(uv: np.ndarray, res: tuple[int, int]) -> tuple[float, float, float, float]:
    """Axis-aligned box in xywh, clipped to the image."""
    w, h = res
    x0 = float(np.clip(uv[:, 0].min(), 0, w))
    x1 = float(np.clip(uv[:, 0].max(), 0, w))
    y0 = float(np.clip(uv[:, 1].min(), 0, h))
    y1 = float(np.clip(uv[:, 1].max(), 0, h))
    return x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)

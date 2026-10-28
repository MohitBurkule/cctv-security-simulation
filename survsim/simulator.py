"""Top-level Simulator: wires scenario, world, backend and recorder together."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .backends.base import Backend
from .backends.mock.backend import MockBackend
from .core.loader import load_scenario
from .core.models import Scenario
from .director.world import World
from .gt.writers import Recorder


def make_backend(engine: str) -> Backend:
    if engine == "mock":
        return MockBackend()
    if engine == "unreal":
        from .backends.unreal.backend import UnrealBackend

        return UnrealBackend()
    raise ValueError(f"unsupported engine {engine!r} (have: mock, unreal)")


class Simulator:
    def __init__(self, scenario: Scenario, backend: Backend | None = None) -> None:
        self.scenario = scenario
        self.backend = backend or make_backend(scenario.engine)
        self.world = World(scenario)
        self._hooks: list[Callable[[int, World], None]] = []
        self.backend.open(scenario)

    @classmethod
    def from_yaml(cls, path: str | Path, seed: int | None = None, engine: str | None = None):
        scenario = load_scenario(path, seed=seed)
        if engine is not None:
            scenario.engine = engine  # type: ignore[assignment]
        return cls(scenario)

    def on_tick(self, fn: Callable[[int, World], None]) -> Callable[[int, World], None]:
        self._hooks.append(fn)
        return fn

    def record(self, out: str | Path, topdown_video: bool = False) -> dict[str, Any]:
        """Run the whole scenario and write ground truth. Returns written paths."""
        out = Path(out)
        rec = Recorder(self.scenario, out)
        frames = []

        for _ in range(self.scenario.n_ticks):
            self.world.step()
            self.backend.sync(self.world)
            for cam in self.scenario.cameras:
                # Observe through the camera as it is *now*: a slewed PTZ head or a
                # knocked camera no longer matches its configured pose.
                live = self.world.cameras.live_camera(cam.id)
                if not self.world.cameras.states[cam.id].blind:
                    rec.add_camera_rows(self.backend.observe(self.world, live))
            rec.add_camera_status(self.world)
            rec.add_world(self.world)
            for hook in self._hooks:
                hook(self.world.tick, self.world)
            if topdown_video and isinstance(self.backend, MockBackend):
                frames.append(self.backend.render_topdown(self.world))

        self.world.finalise()
        written = rec.write(self.world)

        if frames:
            written["topdown"] = self._write_video(frames, out / "topdown.mp4")
        return {"paths": written, "events": len(self.world.events), "ticks": self.world.tick}

    def _write_video(self, frames: list, path: Path) -> Path:
        import imageio.v2 as imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(path, [f[:, :, ::-1] for f in frames], fps=int(self.scenario.fps))
        return path

"""PTZ motion and camera tampering.

A PTZ camera is not a fixed camera with extra parameters: it is a camera that is
pointing somewhere else most of the time. Its patrol schedule creates periodic
blind windows, and an adversary who has watched the site can walk through them.
Modelling the schedule is what lets us measure that.

Tamper states change the pixels rather than the geometry, except `tilt`, which
really does move the camera — so it invalidates the calibration, and any system
that keeps using the stored extrinsics is now silently wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.models import Camera

TAMPER_KINDS = ("cover", "spray", "tilt", "unplug", "glare", "defocus")


@dataclass
class CameraState:
    """Live pose and health of one camera at the current tick."""

    camera_id: str
    yaw: float
    pitch: float
    hfov_deg: float
    tamper: str | None = None
    tamper_since_s: float | None = None
    frames_dropped: int = 0

    @property
    def blind(self) -> bool:
        """True when the camera cannot produce usable imagery at all."""
        return self.tamper in ("cover", "unplug")

    def as_dict(self, t: float) -> dict[str, Any]:
        return {
            "t": round(t, 3),
            "camera": self.camera_id,
            "yaw": round(self.yaw, 2),
            "pitch": round(self.pitch, 2),
            "hfov_deg": round(self.hfov_deg, 2),
            "tamper": self.tamper,
            "blind": self.blind,
        }


class CameraDirector:
    """Advances PTZ patrols and applies scheduled tamper events."""

    def __init__(self, cameras: list[Camera]) -> None:
        self.cameras = {c.id: c for c in cameras}
        self.states: dict[str, CameraState] = {
            c.id: CameraState(c.id, c.mount.yaw, c.mount.pitch, c.hfov_deg) for c in cameras
        }
        self.log: list[dict[str, Any]] = []
        self._leg: dict[str, int] = {c.id: 0 for c in cameras}
        self._leg_started: dict[str, float] = {c.id: 0.0 for c in cameras}
        self._pending: list[tuple[float, str, str]] = []
        for cam in cameras:
            for ev in cam.tamper:
                self._pending.append((float(ev["t"]), cam.id, str(ev.get("kind", "cover"))))
        self._pending.sort()

    # -------------------------------------------------------------------- ptz

    def _step_ptz(self, cam: Camera, state: CameraState, t: float, dt: float) -> None:
        if cam.ptz is None or not cam.ptz.patrol:
            return
        patrol = cam.ptz.patrol
        leg = patrol[self._leg[cam.id] % len(patrol)]
        preset = cam.presets.get(str(leg.get("preset", "")), {})
        target_yaw = float(preset.get("yaw", cam.mount.yaw))
        target_pitch = float(preset.get("pitch", cam.mount.pitch))
        target_fov = float(preset.get("hfov_deg", state.hfov_deg))

        # Slew at the mechanical rate; a real head cannot jump between presets.
        max_step = cam.ptz.speed_dps * dt
        state.yaw += max(-max_step, min(max_step, _wrap180(target_yaw - state.yaw)))
        state.pitch += max(-max_step, min(max_step, target_pitch - state.pitch))
        state.hfov_deg += (target_fov - state.hfov_deg) * min(1.0, 4.0 * dt)

        on_target = (
            abs(_wrap180(target_yaw - state.yaw)) < 1.0 and abs(target_pitch - state.pitch) < 1.0
        )
        dwell = float(leg.get("dwell_s", 10.0))
        if on_target and (t - self._leg_started[cam.id]) >= dwell:
            self._leg[cam.id] += 1
            self._leg_started[cam.id] = t

    # ----------------------------------------------------------------- tamper

    def tamper(self, camera_id: str, kind: str, t: float) -> None:
        if kind not in TAMPER_KINDS:
            raise ValueError(f"unknown tamper kind {kind!r}; have {TAMPER_KINDS}")
        state = self.states[camera_id]
        state.tamper = kind
        state.tamper_since_s = t
        if kind == "tilt":
            # A knocked camera keeps working but no longer matches its calibration.
            state.pitch = max(-89.0, state.pitch - 35.0)
            state.yaw = _wrap180(state.yaw + 25.0)
        self.log.append({"t": round(t, 3), "camera": camera_id, "event": "tamper", "kind": kind})

    # ------------------------------------------------------------------- tick

    def step(self, t: float, dt: float) -> None:
        while self._pending and self._pending[0][0] <= t:
            _, cam_id, kind = self._pending.pop(0)
            self.tamper(cam_id, kind, t)
        for cam_id, cam in self.cameras.items():
            self._step_ptz(cam, self.states[cam_id], t, dt)

    def live_camera(self, camera_id: str) -> Camera:
        """The camera as it is *now*, with PTZ pose and tilt-tamper applied.

        Ground truth must be computed against this, not the configured pose —
        otherwise a tampered or slewed camera reports boxes for a view it is not
        actually showing.
        """
        cam = self.cameras[camera_id]
        state = self.states[camera_id]
        live = cam.model_copy(deep=True)
        live.mount.yaw = state.yaw
        live.mount.pitch = state.pitch
        live.optics.hfov_deg = state.hfov_deg
        return live

    def snapshot(self, t: float) -> list[dict[str, Any]]:
        return [s.as_dict(t) for s in self.states.values()]


def _wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0

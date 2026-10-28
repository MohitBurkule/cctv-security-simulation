"""Unreal backend: drives a packaged UnrealZoo build over UnrealCV.

This is the bridge between the engine-agnostic Director and real pixels. The
Director owns behaviour and truth; this module only translates state into engine
commands and reads observations back.

Coordinate conventions differ and the conversion lives *here* so engine units
never leak upward:

    SurvSim : metres, Z up, right-handed, yaw measured CCW from +X
    Unreal  : centimetres, Z up, left-handed, yaw measured CW from +X

so Y and yaw both flip sign, and lengths scale by 100.

Capabilities are **probed at runtime**, never assumed: a given build may or may
not implement parametric appearance, mocap animation or navigation. Anything
missing is recorded in `limitations` and surfaced with the run, because a dataset
of twelve identical people sliding without leg motion looks superficially fine
and is worthless for training.

Measured on UnrealZoo v3.1.0 (UE 5.7): animation and navigation are present,
parametric wardrobe and body are **not** — see docs/12 for the probe output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ...core.models import Camera, Scenario
from ...director.world import World

M_TO_CM = 100.0
DEFAULT_CHARACTER = "/Game/SmartLocomotion/Blueprints/BP_Character.BP_Character_C"


def to_ue_location(xyz_m: np.ndarray, ground_z_cm: float = 0.0) -> tuple[float, float, float]:
    x, y, z = (float(v) for v in xyz_m)
    return x * M_TO_CM, -y * M_TO_CM, z * M_TO_CM + ground_z_cm


def to_ue_yaw(yaw_deg: float) -> float:
    return -float(yaw_deg)


@dataclass
class UnrealBackend:
    """Backend implementation over the UnrealCV TCP API."""

    port: int = 9900
    ip: str = "127.0.0.1"
    character_assets: list[str] = field(default_factory=lambda: [DEFAULT_CHARACTER])
    ground_z_cm: float = 0.0
    settle_s: float = 0.05
    # "nav" lets the engine walk people (real locomotion); "teleport" sets the
    # transform directly, which slides them and destroys any gait signal.
    locomotion: str = "nav"
    nav_lookahead_s: float = 1.5
    nav_retarget_m: float = 1.0
    name: str = "unreal"

    def __post_init__(self) -> None:
        self.client = None
        self._handles: dict[str, str] = {}      # uid -> engine actor name
        self._colours: dict[str, tuple[int, int, int]] = {}  # uid -> mask colour (BGR)
        self._cam_ready = False
        self.limitations: list[str] = []
        self.driver = None
        self._nav_goal: dict[str, np.ndarray] = {}
        self._motion_style: dict[str, str] = {}

    # ------------------------------------------------------------- connection

    def connect(self) -> None:
        from unrealcv import Client

        self.client = Client((self.ip, self.port))
        self.client.connect(timeout=30)
        if not self.client.isconnected():
            raise ConnectionError(
                f"no UnrealCV server on {self.ip}:{self.port}. Start the packaged build first."
            )

    def _req(self, cmd: str, timeout: float = 60) -> str:
        assert self.client is not None, "call connect() first"
        return str(self.client.request(cmd, timeout=timeout))

    # ------------------------------------------------------------------ setup

    def open(self, scenario: Scenario) -> None:
        if self.client is None:
            self.connect()
        from .appearance import AppearanceDriver

        self.level = self._req("vget /level/name", timeout=60).strip()
        self.driver = AppearanceDriver(self._req)
        # Spawn first so the probe has a real actor to test animation against:
        # checking that a command exists proves nothing about whether it works.
        self._spawn_people(scenario)
        sample = next(iter(self._handles.values()), None)
        caps = self.driver.probe(sample_actor=sample)
        self.limitations = list(caps.notes)
        if not caps.navigation and self.locomotion == "nav":
            self.limitations.append(
                "no navigation: falling back to teleporting actors, so legs will not "
                "animate and the footage is unsuitable for gait work"
            )
        self._apply_initial_appearance(scenario)

    def _apply_initial_appearance(self, scenario: Scenario) -> None:
        """Push each person's body, clothing and gait to their actor."""
        if self.driver is None:
            return
        from ...core.appearance import AppearanceState

        for person in scenario.population.hero:
            handle = self._handles.get(person.id)
            if handle is None:
                continue
            self.driver.apply_appearance(handle, person, AppearanceState.from_person(person))
            self.driver.apply_gait(handle, person.gait.style)

    def _spawn_people(self, scenario: Scenario) -> None:
        """One engine actor per Director agent, cycling the available character assets."""
        people = list(scenario.population.hero)
        for i, person in enumerate(people):
            asset = self.character_assets[i % len(self.character_assets)]
            res = self._req(f"vset /objects/spawn_from_path {asset} {person.id}", timeout=90).strip()
            if "error" in res.lower():
                continue
            handle = res or person.id
            self._handles[person.id] = handle
            self._colours[person.id] = self._object_colour(handle)

    def _object_colour(self, handle: str) -> tuple[int, int, int]:
        """Mask colour the engine assigned to this actor, as BGR to match OpenCV."""
        res = self._req(f"vget /object/{handle}/color", timeout=30)
        nums = [int(s) for s in "".join(c if c.isdigit() else " " for c in res).split()[:3]]
        if len(nums) < 3:
            return (0, 0, 0)
        r, g, b = nums
        return (b, g, r)

    # -------------------------------------------------------------- per tick

    def sync(self, world: World) -> None:
        """Drive the engine from Director state.

        Two locomotion modes, and the difference is visible in the footage:

        * **nav** — hand the engine a destination and let it walk there. Legs move,
          feet plant, turns are banked, and the gait clip is actually expressed.
          The engine then owns position, so truth is read back afterwards.
        * **teleport** — set the transform every tick. The character slides with no
          leg motion. Only used when the build has no navigation, and recorded as
          a limitation because the footage is then useless for gait work.
        """
        use_nav = self.driver is not None and self.driver.caps.navigation and self.locomotion == "nav"

        for uid, agent in world.agents.items():
            handle = self._handles.get(uid)
            if handle is None:
                continue

            moving = float(np.linalg.norm(agent.vel)) > 1e-3
            if use_nav:
                self._drive_nav(handle, uid, agent, moving)
            else:
                x, y, z = to_ue_location(agent.pos3, self.ground_z_cm)
                self._req(f"vset /object/{handle}/location {x:.1f} {y:.1f} {z:.1f}", timeout=30)
                if moving:
                    yaw = to_ue_yaw(float(np.degrees(np.arctan2(agent.vel[1], agent.vel[0]))))
                    self._req(f"vset /object/{handle}/rotation 0 {yaw:.1f} 0", timeout=30)

            self._apply_motion_style(handle, uid, agent, moving)

        if self.settle_s:
            time.sleep(self.settle_s)

        if use_nav:
            # The engine moved them, so the engine knows where they are. Ground
            # truth must be what happened, not what was planned.
            self.read_back(world)

    def _drive_nav(self, handle: str, uid: str, agent, moving: bool) -> None:
        """Issue a navigation goal, but only when it has meaningfully changed.

        Re-sending the same destination every tick makes the engine restart its
        path and the character stutters, so goals are throttled by distance.
        """
        assert self.driver is not None
        if not moving:
            if self._nav_goal.pop(uid, None) is not None:
                self.driver.stop_walking(handle)
            return

        goal = agent.pos + agent.vel * self.nav_lookahead_s
        previous = self._nav_goal.get(uid)
        if previous is not None and float(np.linalg.norm(goal - previous)) < self.nav_retarget_m:
            return
        self._nav_goal[uid] = goal
        x, y, z = to_ue_location(np.array([goal[0], goal[1], 0.0]), self.ground_z_cm)
        self.driver.walk_to(handle, x, y, z)

    def _apply_motion_style(self, handle: str, uid: str, agent, moving: bool) -> None:
        """Gait clip while walking, idle clip while stationary.

        A loiterer who stops must keep a *standing* animation rather than freezing
        mid-stride — the stationary case is exactly what motion-based detection
        loses, so it has to look right.
        """
        if self.driver is None or not self.driver.caps.animation:
            return
        wanted = agent.spec.gait.style if moving else self._idle_activity(agent)
        if self._motion_style.get(uid) == wanted:
            return
        self._motion_style[uid] = wanted
        if moving:
            self.driver.apply_gait(handle, agent.spec.gait.style)
        else:
            self.driver.apply_idle(handle, wanted)

    @staticmethod
    def _idle_activity(agent) -> str:
        label = agent.label
        if "recon" in label or "scan" in label:
            return "scan"
        if "commute_wait" in label:
            return "wait"
        if "loiter" in label:
            return "phone"
        return "idle"

    def read_back(self, world: World) -> int:
        """Update Director agents from the engine's actual actor positions."""
        updated = 0
        for uid, agent in world.agents.items():
            handle = self._handles.get(uid)
            if handle is None:
                continue
            raw = self._req(f"vget /object/{handle}/location", timeout=30)
            parts = str(raw).replace(",", " ").split()
            try:
                x_cm, y_cm = float(parts[0]), float(parts[1])
            except (ValueError, IndexError):
                continue
            agent.pos = np.array([x_cm / M_TO_CM, -y_cm / M_TO_CM])
            updated += 1
        return updated

    def step(self, dt: float) -> None:
        """The packaged build ticks in real time; the Director owns simulated time."""

    # --------------------------------------------------------------- cameras

    def place_camera(self, cam: Camera) -> None:
        x, y, z = to_ue_location(np.asarray(cam.mount.pos, dtype=float), self.ground_z_cm)
        self._req(f"vset /camera/0/location {x:.1f} {y:.1f} {z:.1f}", timeout=45)
        self._req(
            f"vset /camera/0/rotation {cam.mount.pitch:.1f} {to_ue_yaw(cam.mount.yaw):.1f} "
            f"{cam.mount.roll:.1f}", timeout=45)
        self._req(f"vset /camera/0/fov {cam.hfov_deg:.1f}", timeout=45)
        time.sleep(0.25)

    def capture(self, cameras: list[Camera], modalities: set[str]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for cam in cameras:
            self.place_camera(cam)
            frames: dict[str, Any] = {}
            if "rgb" in modalities:
                frames["rgb"] = self._grab("lit")
            if "instance_mask" in modalities or "mask" in modalities:
                frames["instance_mask"] = self._grab("object_mask")
            if "depth" in modalities:
                frames["depth"] = self._grab_depth()
            out[cam.id] = frames
        return out

    def _grab(self, viewmode: str) -> np.ndarray:
        assert self.client is not None
        data = self.client.request(f"vget /camera/0/{viewmode} png", timeout=180)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        return img[:, :, :3] if img is not None and img.ndim == 3 else img

    def _grab_depth(self) -> np.ndarray:
        import io

        assert self.client is not None
        data = self.client.request("vget /camera/0/depth npy", timeout=180)
        return np.load(io.BytesIO(data))

    # ---------------------------------------------------------- ground truth

    def observe(self, world: World, camera: Camera) -> list[dict[str, Any]]:
        """Per-camera ground truth read from the engine's own instance mask.

        Boxes come from the rendered mask rather than from projecting a box around
        the agent, so occlusion by scenery and by other people is handled by the
        renderer — the thing the mock backend cannot do.
        """
        self.place_camera(camera)
        mask = self._grab("object_mask")
        if mask is None:
            return []

        rows: list[dict[str, Any]] = []
        for uid, agent in world.agents.items():
            colour = self._colours.get(uid)
            if colour is None:
                continue
            sel = np.all(np.abs(mask.astype(int) - np.array(colour, dtype=int)) <= 8, axis=-1)
            area = int(sel.sum())
            if area < 12:                      # too few pixels to be a usable label
                continue
            ys, xs = np.nonzero(sel)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            rows.append({
                "frame": world.tick,
                "t": round(world.t, 3),
                "camera": camera.id,
                "uid": uid,
                "bbox": (float(x0), float(y0), float(x1 - x0), float(y1 - y0)),
                "mask_area_px": area,
                "px_height": float(y1 - y0),
                # Fill factor: how much of the box the person actually occupies. Low
                # values mean heavy occlusion or an odd pose, which is a useful
                # difficulty signal that projected boxes cannot provide.
                "fill_ratio": round(area / max((x1 - x0) * (y1 - y0), 1), 3),
                "label": agent.label,
                "object_class": agent.object_class,
            })
        return rows

    def bones(self, uid: str) -> dict[str, Any] | None:
        """Skeletal ground truth for 3D pose tasks."""
        handle = self._handles.get(uid)
        if handle is None:
            return None
        import json

        raw = self._req(f"vget /object/{handle}/bones", timeout=90)
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return {"raw": raw[:2000]}

    def close(self) -> None:
        if self.client is not None:
            try:
                self.client.disconnect()
            finally:
                self.client = None

"""Ground-truth writers: MOTChallenge per camera, world parquet, events JSONL, calib."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..core.geometry import extrinsics, intrinsics
from ..core.models import Camera, Scenario
from ..director.world import World


class Recorder:
    def __init__(self, scenario: Scenario, out_dir: Path) -> None:
        self.scenario = scenario
        self.out = Path(out_dir)
        self.rows: list[dict[str, Any]] = []
        self.world_rows: list[dict[str, Any]] = []
        self.camera_status: list[dict[str, Any]] = []
        self._uid_index: dict[str, int] = {}
        self._class: dict[str, str] = {}

    def track_id(self, uid: str) -> int:
        """MOTChallenge needs integer ids; keep a stable mapping back to the uid."""
        return self._uid_index.setdefault(uid, len(self._uid_index) + 1)

    def add_camera_rows(self, rows: list[dict[str, Any]]) -> None:
        self.rows.extend(rows)

    def add_camera_status(self, world: World) -> None:
        self.camera_status.extend(world.cameras.snapshot(world.t))

    def add_world(self, world: World) -> None:
        for agent in world.agents.values():
            self._class[agent.uid] = agent.object_class
            self.world_rows.append(
                {
                    "frame": world.tick,
                    "t": round(world.t, 3),
                    "uid": agent.uid,
                    "object_class": agent.object_class,
                    "x": round(float(agent.pos[0]), 4),
                    "y": round(float(agent.pos[1]), 4),
                    "vx": round(float(agent.vel[0]), 4),
                    "vy": round(float(agent.vel[1]), 4),
                    "speed": round(float(np.linalg.norm(agent.vel)), 4),
                    "zone": world.zone_of(agent.pos),
                    "label": agent.label,
                    "posture": agent.posture,
                    "appearance_epoch": agent.appearance_epoch,
                    "appearance_hash": agent.appearance.hash if agent.appearance else None,
                    "face_visible": agent.appearance.face_visible if agent.appearance else True,
                }
            )

    # ------------------------------------------------------------------ flush

    def write(self, world: World) -> dict[str, Path]:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "gt").mkdir(exist_ok=True)
        (self.out / "calib").mkdir(exist_ok=True)
        written: dict[str, Path] = {}

        for cam in self.scenario.cameras:
            lines = [
                f"{r['frame']},{self.track_id(r['uid'])},"
                f"{r['bbox'][0]:.2f},{r['bbox'][1]:.2f},{r['bbox'][2]:.2f},{r['bbox'][3]:.2f},"
                f"{self._mot_class(r['uid'])},{r['visible_frac']:.3f}"
                for r in self.rows
                if r["camera"] == cam.id
            ]
            p = self.out / "gt" / f"mot_{cam.id}.txt"
            p.write_text("\n".join(lines) + ("\n" if lines else ""))
            written[f"mot_{cam.id}"] = p
            written[f"calib_{cam.id}"] = self._write_calib(cam)

        ev = self.out / "gt" / "events.jsonl"
        ev.write_text("".join(json.dumps(e.as_dict()) + "\n" for e in world.events))
        written["events"] = ev

        sensors = self.out / "gt" / "sensors.jsonl"
        sensors.write_text("".join(json.dumps(r) + "\n" for r in world.sensor_log))
        written["sensors"] = sensors

        status = self.out / "gt" / "camera_status.jsonl"
        status.write_text("".join(json.dumps(r) + "\n" for r in self.camera_status))
        written["camera_status"] = status

        written["world"] = self._write_world()
        written["identities"] = self._write_identities(world)

        meta = self.out / "meta"
        meta.mkdir(exist_ok=True)
        (meta / "scenario.resolved.json").write_text(
            json.dumps(self.scenario.model_dump(mode="json"), indent=2)
        )
        (meta / "track_ids.json").write_text(json.dumps(self._uid_index, indent=2))
        written["scenario"] = meta / "scenario.resolved.json"
        return written

    def _mot_class(self, uid: str) -> str:
        """MOTChallenge "consider,class": a person is 1,1 (pedestrian); anything
        else (wildlife) is 0,8 — MOT's "distractor", ignored by the metrics."""
        return "1,1" if self._class.get(uid, "person") == "person" else "0,8"

    def _write_calib(self, cam: Camera) -> Path:
        from ..core.geometry import distortion_coeffs

        r, t = extrinsics(cam)
        model, coeffs = distortion_coeffs(cam)
        payload = {
            "camera_id": cam.id,
            "type": cam.type,
            "resolution": list(cam.capture.res),
            "fps": cam.capture.fps,
            "K": intrinsics(cam).tolist(),
            "R": r.tolist(),
            "t": t.tolist(),
            "position_world": list(cam.mount.pos),
            "hfov_deg": cam.hfov_deg,
            # The coefficients actually used when projecting, so calibration work
            # is not silently comparing against a pinhole model.
            "distortion": {"model": model, "coeffs": [round(float(c), 8) for c in coeffs],
                           "raw": cam.optics.distortion},
            "clock_offset_ms": cam.stream.clock_offset_ms,
            "drift_ppm": cam.stream.drift_ppm,
        }
        p = self.out / "calib" / f"{cam.id}.json"
        p.write_text(json.dumps(payload, indent=2))
        return p

    def _write_world(self) -> Path:
        import pandas as pd

        p = self.out / "gt" / "world.parquet"
        pd.DataFrame(self.world_rows).to_parquet(p, index=False)
        return p

    def _write_identities(self, world: World) -> Path:
        """One row per appearance epoch.

        Evaluation joins on (uid, epoch) to score re-identification *within* an
        appearance and *across* a change separately. Soft biometrics are carried on
        every row because they are the only cues that survive a change of clothes.
        """
        import pandas as pd

        from ..core.appearance import soft_biometrics
        from .licences import PROCEDURAL, dataset_licences, licence_column

        rows = []
        for row in world.epochs:
            agent = world.roster.get(row["uid"]) or world.agents.get(row["uid"])
            person = agent.spec if agent else None
            rows.append(
                {
                    **row,
                    "role": person.role if person else None,
                    "object_class": agent.object_class if agent else "person",
                    "hard_negative_for": ",".join(person.hard_negative_for) if person else "",
                    **(soft_biometrics(person) if person else {}),
                    # Provenance (docs/research/47): the director's own bodies are
                    # procedural; a render replaces this per person with the
                    # character asset it used (tools/render_run.py).
                    "asset_licences": licence_column([PROCEDURAL]),
                }
            )
        p = self.out / "gt" / "identities.parquet"
        pd.DataFrame(rows).to_parquet(p, index=False)
        meta = self.out / "meta"
        meta.mkdir(exist_ok=True)
        (meta / "licences.json").write_text(json.dumps(
            dataset_licences({r["uid"]: [PROCEDURAL] for r in rows}), indent=1))
        return p

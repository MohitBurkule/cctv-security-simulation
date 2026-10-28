#!/usr/bin/env python3
"""Build out/gallery/behaviour_gallery.{mp4,png} — one short mock-backend
top-down render per catalogue behaviour (CLAUDE.md breadth-of-scenarios ask).

For every name in survsim.director.behaviours.REGISTRY we build a small,
self-contained scenario (one shared environment: a gate, a restricted zone
with an asset, a fenced perimeter with a cuttable panel, a blindspot, a
bench, a wall, a door) and run the mock backend for a few seconds, capturing
`Simulator.record(..., topdown_video=True)` frames. Where the catalogue
names a hard-negative twin, both agents run in the same tile.

Usage:
    source .venv/bin/activate
    python tools/scenario_gallery.py [--out-dir out/gallery] [--behaviours-only]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from survsim.core.loader import load_scenario
from survsim.director.behaviours import REGISTRY
from survsim.simulator import Simulator

TILE_W, TILE_H = 640, 360
FPS = 10
SIM_S = 18  # short sim per behaviour

OUT_DIR = REPO / "out" / "gallery"

# Catalogue: name -> (role_hint, suspicious/innocent, twin name or None)
CATALOGUE = {
    # normal
    "transit": ("normal", None),
    "commute_wait": ("normal", "loiter"),
    "queue": ("normal", None),
    "group_walk": ("normal", None),
    "meet_and_greet": ("normal", None),
    "patrol_guard": ("normal", "fence_climb"),
    "maintenance_visit": ("normal", "zone_intrusion"),
    "staff_escort": ("normal", "fence_tailgate"),
    "vehicle_move": ("normal", "vehicle_ram"),
    "wildlife": ("normal", None),
    "maintenance": ("normal", "asset_tamper"),
    "photograph": ("normal", "recon"),
    "smoke_break": ("normal", "loiter"),
    "lost_tourist": ("normal", None),
    "child_play": ("normal", None),
    "drunk_walk": ("normal", None),
    "disabled_mobility": ("normal", None),
    "weather_reaction": ("normal", None),
    "sit_rest": ("normal", "person_down"),
    # suspicious
    "loiter": ("suspicious", "commute_wait"),
    "loiter_intermittent": ("suspicious", "commute_wait"),
    "perimeter_probe": ("suspicious", "patrol_guard"),
    "recon": ("suspicious", "photograph"),
    "tailgating": ("suspicious", "staff_escort"),
    "tailgate": ("suspicious", "staff_escort"),
    "fence_tailgate": ("suspicious", "staff_escort"),
    "following": ("suspicious", None),
    "follow": ("suspicious", None),
    "object_abandon": ("suspicious", None),
    "object_handoff": ("suspicious", None),
    "concealment": ("suspicious", None),
    "outfit_change": ("suspicious", None),
    "camera_avoidance": ("suspicious", None),
    "counter_flow": ("suspicious", None),
    "erratic_motion": ("suspicious", None),
    "wrong_zone_entry": ("suspicious", "maintenance_visit"),
    "zone_intrusion": ("suspicious", "maintenance_visit"),
    "climb_attempt": ("suspicious", "patrol_guard"),
    "vehicle_dropoff_loiter": ("suspicious", None),
    "crowd_form": ("suspicious", None),
    "night_presence": ("suspicious", None),
    # hostile
    "camera_tamper": ("hostile", None),
    "fence_cut": ("hostile", "patrol_guard"),
    "fence_climb": ("hostile", "patrol_guard"),
    "asset_tamper": ("hostile", "maintenance"),
    "vandalism": ("hostile", None),
    "theft_snatch": ("hostile", None),
    "argument": ("hostile", None),
    "fight": ("hostile", None),
    "crowd_rush": ("hostile", None),
    "person_down": ("hostile", "sit_rest"),
    "throw_object": ("hostile", None),
    "vehicle_ram": ("hostile", "vehicle_move"),
    "flee": ("hostile", None),
    "drone_overflight": ("hostile", None),
}

BASE_ENV = """
schema_version: 1
name: gallery_{name}
seed: 3
duration_s: {dur}
fps: {fps}
engine: mock

environment:
  id: gallery_site
  zones:
    - {{id: zone.gate_A, tag: gate, polygon: [[8, -4], [16, -4], [16, 4], [8, 4]]}}
    - {{id: zone.yard, tag: public, polygon: [[-30, -30], [30, -30], [30, 30], [-30, 30]]}}
    - {{id: zone.restricted, tag: restricted, polygon: [[-28, 10], [-16, 10], [-16, 24], [-28, 24]],
       allowed_roles: [worker, guard]}}
    - {{id: zone.fence_north, tag: perimeter_fence, polygon: [[-30, 27], [30, 27], [30, 29], [-30, 29]]}}
    - {{id: zone.blindspot_A, tag: blindspot, polygon: [[23, 18], [30, 18], [30, 25], [23, 25]]}}
  spawn_points:
    - {{id: spawn.north, pos: [0.0, 26.0, 0.0]}}
    - {{id: spawn.east, pos: [26.0, 0.0, 0.0]}}
    - {{id: spawn.south, pos: [0.0, -26.0, 0.0]}}
  exits:
    - {{id: exit.south, pos: [0.0, -28.0, 0.0]}}
    - {{id: exit.west, pos: [-28.0, 0.0, 0.0]}}
  props:
    - {{id: fence_seg_north, type: fence_panel, pos: [-6.0, 28.0, 0.0], length_m: 3, cuttable: true}}
    - {{id: door.restricted, type: door, pos: [-22.0, 10.0, 0.0]}}
    - {{id: asset.cabinet, type: cabinet, pos: [-22.0, 18.0, 0.0]}}
    - {{id: bench.park, type: bench, pos: [4.0, 10.0, 0.0]}}
    - {{id: wall.plain, type: wall, pos: [10.0, 20.0, 0.0]}}

cameras:
  - id: cam_gate
    type: fixed_bullet
    mount: {{pos: [22.0, -10.0, 4.5], yaw: 135, pitch: -20}}
    optics: {{hfov_deg: 70}}
    capture: {{res: [640, 360], fps: {fps}}}
  - id: cam_yard
    type: dome
    mount: {{pos: [-20.0, -18.0, 5.0], yaw: 60, pitch: -18}}
    optics: {{hfov_deg: 95}}
    capture: {{res: [640, 360], fps: {fps}}}

population:
  hero:
{people}

output:
  modalities: [rgb]
  gt: [mot, events]
"""

PERSON_TMPL = """    - id: {pid}
      role: {role}
      body: {{height_m: 1.75, skin_tone: 4, sex_presentation: m, age_band: 25-40}}
      intent:
{intent}
"""


def _yaml_intent(steps: list[dict]) -> str:
    lines = []
    for step in steps:
        parts = []
        for k, v in step.items():
            if isinstance(v, str):
                parts.append(f"{k}: {v}")
            else:
                parts.append(f"{k}: {v}")
        lines.append("        - {" + ", ".join(parts) + "}")
    return "\n".join(lines)


def person_block(pid: str, role: str, steps: list[dict]) -> str:
    return PERSON_TMPL.format(pid=pid, role=role, intent=_yaml_intent(steps))


def intent_for(kind: str, pid: str, partner: str | None) -> list[dict]:
    """Best-effort minimal intent list exercising `kind` in the shared env."""
    if kind in ("transit",):
        return [{"type": "transit", "from": "spawn.north", "to": "exit.south"}]
    if kind in ("loiter", "loiter_intermittent"):
        return [{"type": "transit", "to": "zone.gate_A"},
                {"type": kind, "zone": "zone.gate_A", "dur_s": 10, "activity": "pace"}]
    if kind == "commute_wait":
        return [{"type": "transit", "to": "zone.gate_A"},
                {"type": "commute_wait", "zone": "zone.gate_A", "dur_s": 10, "activity": "idle"}]
    if kind == "queue":
        return [{"type": "transit", "to": "zone.gate_A"},
                {"type": "queue", "zone": "zone.gate_A", "dur_s": 10}]
    if kind == "group_walk":
        return [{"type": "group_walk", "leader": pid, "to": "exit.south"}]
    if kind == "patrol_guard":
        return [{"type": "patrol_guard", "zones": ["zone.restricted"], "rounds": 1, "check_s": 5}]
    if kind == "maintenance_visit":
        return [{"type": "maintenance_visit", "zone": "zone.restricted", "dur_s": 10}]
    if kind == "staff_escort":
        return [{"type": "staff_escort", "target": partner or pid, "timeout_s": 15}]
    if kind in ("vehicle_move",):
        return [{"type": "vehicle_move", "to": "exit.south", "speed_mps": 6}]
    if kind == "vehicle_ram":
        return [{"type": "vehicle_ram", "barrier": "door.restricted", "speed_mps": 15}]
    if kind == "wildlife":
        return [{"type": "wildlife", "zone": "zone.yard", "dur_s": 10}]
    if kind in ("maintenance",):
        return [{"type": "transit", "to": "zone.restricted"},
                {"type": "maintenance", "asset": "asset.cabinet", "dur_s": 10}]
    if kind == "asset_tamper":
        return [{"type": "transit", "to": "zone.restricted"},
                {"type": "asset_tamper", "prop": "asset.cabinet", "dur_s": 10}]
    if kind in ("photograph", "recon"):
        return [{"type": "transit", "to": "zone.gate_A"},
                {"type": kind, "targets": ["cam_gate"], "dwell_s": 8}]
    if kind == "smoke_break":
        return [{"type": "transit", "to": "zone.gate_A"},
                {"type": "smoke_break", "zone": "zone.gate_A", "dur_s": 8}]
    if kind == "lost_tourist":
        return [{"type": "lost_tourist", "dur_s": 12}]
    if kind == "child_play":
        return [{"type": "child_play", "zone": "zone.yard", "dur_s": 12}]
    if kind == "drunk_walk":
        return [{"type": "drunk_walk", "to": "exit.south"}]
    if kind == "disabled_mobility":
        return [{"type": "disabled_mobility", "to": "exit.south"}]
    if kind == "weather_reaction":
        return [{"type": "weather_reaction", "to": "exit.south"}]
    if kind == "sit_rest":
        return [{"type": "transit", "to": "bench.park"},
                {"type": "sit_rest", "bench": "bench.park", "dur_s": 10}]
    if kind == "person_down":
        return [{"type": "transit", "to": "zone.gate_A"},
                {"type": "person_down", "cause": "fall", "dur_s": 10}]
    if kind in ("perimeter_probe",):
        return [{"type": "perimeter_probe", "segment": "fence_seg_north", "walk_along_m": 5}]
    if kind in ("tailgating", "tailgate"):
        return [{"type": "tailgate", "target": partner or pid, "door": "door.restricted"}]
    if kind == "fence_tailgate":
        return [{"type": "fence_tailgate", "zone": "zone.fence_north", "wait_s": 5, "dur_s": 10}]
    if kind in ("following", "follow"):
        return [{"type": "follow", "target": partner or pid, "dist_m": 6, "dur_s": 12}]
    if kind == "object_abandon":
        return [{"type": "transit", "to": "zone.yard"},
                {"type": "object_abandon", "zone": "zone.yard", "leave_dist_m": 12, "exit": "exit.south"}]
    if kind == "object_handoff":
        return [{"type": "object_handoff", "partner": partner or pid, "object": "bag", "to": "exit.south"}]
    if kind in ("concealment", "outfit_change"):
        return [{"type": "transit", "to": "zone.blindspot_A"},
                {"type": "outfit_change", "zone": "zone.blindspot_A", "dur_s": 6},
                {"type": "transit", "to": "exit.west"}]
    if kind == "camera_avoidance":
        return [{"type": "camera_avoidance", "to": "exit.south", "awareness": 1.0}]
    if kind == "counter_flow":
        return [{"type": "counter_flow", "to": "zone.gate_A", "dur_s": 10}]
    if kind == "erratic_motion":
        return [{"type": "erratic_motion", "zone": "zone.yard", "dur_s": 12}]
    if kind in ("wrong_zone_entry", "zone_intrusion"):
        return [{"type": "zone_intrusion", "zone": "zone.restricted", "depth_m": 3.5, "dur_s": 10}]
    if kind == "climb_attempt":
        return [{"type": "climb_attempt", "barrier": "fence_seg_north", "dur_s": 8}]
    if kind == "vehicle_dropoff_loiter":
        return [{"type": "vehicle_dropoff_loiter", "vehicle": "vehicle_1", "dur_s": 10}]
    if kind == "crowd_form":
        return [{"type": "crowd_form", "zone": "zone.gate_A", "dur_s": 10}]
    if kind == "night_presence":
        return [{"type": "night_presence", "zone": "zone.yard", "hour": 2.0}]
    if kind == "camera_tamper":
        return [{"type": "camera_tamper", "camera": "cam_gate", "kind": "spray", "dur_s": 6}]
    if kind == "fence_cut":
        return [{"type": "fence_cut", "segment": "fence_seg_north", "cut_s": 8, "dwell_s": 4}]
    if kind == "fence_climb":
        return [{"type": "fence_climb", "segment": "fence_seg_north", "climb_s": 6, "dwell_s": 4}]
    if kind == "vandalism":
        return [{"type": "vandalism", "target": "wall.plain", "dur_s": 10}]
    if kind == "theft_snatch":
        return [{"type": "theft_snatch", "victim": partner or pid, "exit": "exit.south"}]
    if kind == "argument":
        return [{"type": "argument", "partner": partner or pid, "dur_s": 10}]
    if kind == "fight":
        return [{"type": "fight", "partner": partner or pid, "dur_s": 8}]
    if kind == "crowd_rush":
        return [{"type": "crowd_rush", "target": "zone.gate_A"}]
    if kind == "throw_object":
        return [{"type": "throw_object", "target": "zone.yard", "windup_s": 1}]
    if kind == "flee":
        return [{"type": "flee", "to": "exit.south"}]
    if kind == "drone_overflight":
        return [{"type": "drone_overflight", "to": "exit.south", "altitude_m": 20}]
    if kind == "meet_and_greet":
        return [{"type": "meet_and_greet", "partner": partner or pid, "dur_s": 8}]
    return [{"type": "transit", "from": "spawn.north", "to": "exit.south"}]


def build_scenario_yaml(name: str, role: str, twin: str | None) -> str:
    pid_a = "actor_a"
    pid_b = "actor_b"
    people = person_block(pid_a, "intruder" if role == "hostile" else
                           ("suspicious" if role == "suspicious" else "worker"),
                           intent_for(name, pid_a, pid_b))
    if twin is not None:
        people += person_block(pid_b, "worker", intent_for(twin, pid_b, pid_a))
    return BASE_ENV.format(name=name, dur=SIM_S, fps=FPS, people=people)


def run_and_capture(name: str, role: str, twin: str | None, work_dir: Path) -> list[np.ndarray]:
    """Run the scenario, return the last ~3s of topdown frames (BGR, square)."""
    yaml_text = build_scenario_yaml(name, role, twin)
    ypath = work_dir / f"{name}.yaml"
    ypath.write_text(yaml_text)
    scenario = load_scenario(ypath, seed=3)
    sim = Simulator(scenario)
    frames: list[np.ndarray] = []
    for _ in range(scenario.n_ticks):
        sim.world.step()
        sim.backend.sync(sim.world)
        for cam in scenario.cameras:
            live = sim.world.cameras.live_camera(cam.id)
            if not sim.world.cameras.states[cam.id].blind:
                sim.backend.observe(sim.world, live)
        frames.append(sim.backend.render_topdown(sim.world))
    sim.world.finalise()
    tail = int(3 * FPS)
    return frames[-tail:] if len(frames) >= tail else frames


def make_tile(frames: list[np.ndarray], name: str, role: str, twin: str | None) -> list[np.ndarray]:
    import cv2

    out = []
    for f in frames:
        square = cv2.resize(f, (TILE_H, TILE_H))
        canvas = np.full((TILE_H, TILE_W, 3), 18, dtype=np.uint8)
        canvas[:, : TILE_H] = square
        panel_x = TILE_H + 6
        cv2.putText(canvas, name, (panel_x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        colour = {"normal": (140, 230, 140), "suspicious": (60, 170, 255), "hostile": (70, 90, 255)}[role]
        cv2.putText(canvas, role, (panel_x, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        if twin:
            cv2.putText(canvas, f"twin: {twin}", (panel_x, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (200, 200, 200), 1, cv2.LINE_AA)
        out.append(canvas)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--behaviours-only", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # unique catalogue names actually implemented in REGISTRY
    names = sorted(REGISTRY.keys())
    tiles: dict[str, list[np.ndarray]] = {}
    failures: dict[str, str] = {}

    for name in names:
        role, twin = CATALOGUE.get(name, ("suspicious", None))
        if twin is not None and twin not in REGISTRY:
            twin = None
        try:
            frames = run_and_capture(name, role, twin, work_dir)
            if not frames:
                raise RuntimeError("no frames produced")
            tiles[name] = make_tile(frames, name, role, twin)
            print(f"OK   {name:24s} role={role:10s} twin={twin}")
        except Exception as exc:  # noqa: BLE001
            failures[name] = f"{type(exc).__name__}: {exc}"
            print(f"FAIL {name:24s} {failures[name]}")

    print(f"\n{len(tiles)}/{len(names)} behaviours rendered, {len(failures)} failed")

    if not tiles:
        print("nothing rendered; aborting montage build")
        return


    write_behaviour_gallery(tiles, out_dir)
    write_readme(tiles, failures, out_dir)
    shutil.rmtree(work_dir, ignore_errors=True)


def write_behaviour_gallery(tiles: dict[str, list[np.ndarray]], out_dir: Path) -> None:
    import cv2

    names = list(tiles.keys())
    n = len(names)
    cols, rows = 4, 3
    per_grid = cols * rows
    n_grids = (n + per_grid - 1) // per_grid
    grid_w, grid_h = TILE_W * cols, TILE_H * rows

    all_frames = []
    n_frames = min(len(f) for f in tiles.values())
    for grid_i in range(n_grids):
        chunk = names[grid_i * per_grid: (grid_i + 1) * per_grid]
        for t in range(n_frames):
            canvas = np.full((grid_h, grid_w, 3), 8, dtype=np.uint8)
            for idx, name in enumerate(chunk):
                r, c = divmod(idx, cols)
                tile = tiles[name][t]
                canvas[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = tile
            all_frames.append(canvas)

    import imageio.v2 as imageio

    mp4_path = out_dir / "behaviour_gallery.mp4"
    imageio.mimsave(mp4_path, [f[:, :, ::-1] for f in all_frames], fps=FPS)
    print(f"wrote {mp4_path} ({len(all_frames)} frames)")

    # still contact sheet: first grid's last frame, one big montage of ALL tiles' final frame
    cols2 = 6
    rows2 = (n + cols2 - 1) // cols2
    sheet = np.full((TILE_H * rows2, TILE_W * cols2, 3), 8, dtype=np.uint8)
    for idx, name in enumerate(names):
        r, c = divmod(idx, cols2)
        sheet[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = tiles[name][-1]
    png_path = out_dir / "behaviour_gallery.png"
    cv2.imwrite(str(png_path), sheet)
    print(f"wrote {png_path}")


def write_readme(tiles: dict, failures: dict, out_dir: Path) -> None:
    lines = [
        "# Scenario gallery",
        "",
        (f"`behaviour_gallery.mp4` / `.png`: {len(tiles)} of {len(REGISTRY)} catalogue behaviours "
         f"rendered via the mock backend (top-down), {len(failures)} not renderable without more "
         "scenario-specific wiring than this generic environment provides."),
        "",
        "## Rendered",
    ]
    for name in sorted(tiles):
        role, twin = CATALOGUE.get(name, ("suspicious", None))
        lines.append(f"- `{name}` ({role}" + (f", twin `{twin}`" if twin else "") + ")")
    if failures:
        lines.append("")
        lines.append("## Not rendered (generic env mismatch)")
        for name, err in sorted(failures.items()):
            lines.append(f"- `{name}`: {err}")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()

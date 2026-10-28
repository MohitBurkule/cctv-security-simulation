"""Scenario -> environment plan for the Unreal render (pure Python, engine-free).

The editor used to build one hard-coded street for every scenario. This turns a
scenario's site type, extent, cameras, weather and time of day into a plan of
primitives the in-editor builder places (`scripts/ue/survsim_scene.build_from_plan`):

    {"site_type", "layout", "seed", "half_cm", "ground": {...}, "items": [...],
     "lighting": {...}, "weather": {...}}

Each item is {"kind", "x", "y", "yaw", ...}; kinds are the builder's vocabulary
(building, road, kerb, lane_line, bay_line, car, lamp, tree, bench, bin,
bollard, pillar, canopy, fence, gatehouse, container, stage, kiosk, planter,
gateline, platform_edge, board, wall, hedge). Units: Unreal cm, Unreal axes.

Deterministic from (site_type, seed). Nothing static is placed inside the walk
envelope (track keys + margin) except low, sparse street furniture on its rim,
and nothing within 8 m in front of a camera.
"""
from __future__ import annotations

import itertools
import math
import random
from typing import Any

import numpy as np

SITE_TYPES = ("office_building", "transit_station", "perimeter_site",
              "retail_street", "car_park", "event_plaza")

LAYOUT = {"office_building": "office_forecourt", "transit_station": "station_forecourt",
          "perimeter_site": "warehouse_yard", "retail_street": "retail_frontage",
          "car_park": "car_park", "event_plaza": "plaza"}

# Ground material set per layout: (main, rim, far) names from survsim_materials.
GROUND = {"office_forecourt": ("setts", "granite", "asphalt"),
          "station_forecourt": ("concrete", "slabs", "asphalt"),
          "warehouse_yard": ("gravel", "concrete", "grass"),
          "retail_frontage": ("slabs", "setts", "asphalt"),
          "car_park": ("asphalt", "concrete", "asphalt"),
          "plaza": ("light_slabs", "setts", "asphalt")}

from survsim.generator.layout import SITE_TYPES as ALL_SITE_TYPES
from survsim.generator.layout import SITE_TYPES_V2

_KEYWORDS = [*((t, (t,)) for t in SITE_TYPES_V2),
             ("perimeter_site", ("fence", "compound", "border", "perimeter", "yard", "substation",
                                 "restricted")),
             ("transit_station", ("station", "platform", "concourse", "transit", "gateline")),
             ("car_park", ("car_park", "carpark", "parking", "garage")),
             ("office_building", ("office", "lobby", "reception")),
             ("retail_street", ("retail", "street", "shop", "high_street")),
             ("event_plaza", ("plaza", "event", "square", "stage"))]


def infer_site_type(environment: dict[str, Any]) -> str:
    """Explicit `site_type`, else keywords in env id / zone ids+tags / props."""
    if environment.get("site_type") in ALL_SITE_TYPES:
        return environment["site_type"]
    words = [str(environment.get("id", ""))]
    for z in environment.get("zones", []):
        words += [str(z.get("id", "")), str(z.get("tag", ""))]
    words += [str(p.get("type", "")) for p in environment.get("props", [])]
    text = " ".join(words).lower()
    for site, keys in _KEYWORDS:
        if any(k in text for k in keys):
            return site
    return "event_plaza"


def lighting_for(scenario: dict[str, Any], preset: str | None) -> dict[str, Any]:
    """Preset + sun direction from time_of_day (hour) unless forced."""
    tod = scenario.get("time_of_day") or {}
    hour = None
    start = str(tod.get("start", ""))
    if "T" in start:
        try:
            hour = int(start.split("T")[1][:2])
        except ValueError:
            hour = None
    if preset is None:
        preset = "night" if hour is not None and (hour >= 20 or hour < 6) else "day"
    sun_yaw = 35.0 if hour is None else (hour - 6) * 15.0 - 90.0
    sun_pitch = -38.0 if hour is None else -max(8.0, 60.0 - abs(hour - 13) * 8.0)
    return {"preset": preset, "hour": hour, "sun_yaw": sun_yaw, "sun_pitch": sun_pitch}


def weather_for(scenario: dict[str, Any]) -> dict[str, Any]:
    w = scenario.get("weather") or {}
    wet = float(w.get("wetness") or 0.0)
    if str(w.get("preset", "")).startswith("rain") or float(w.get("rain_intensity") or 0) > 0:
        wet = max(wet, 0.7)
    return {"wet": round(min(1.0, wet), 2), "fog": float(w.get("fog_density") or 0.0)}


# Intrinsic footprints (cm) of kinds whose size the builder fixes, so the
# frustum rule tests the whole object, not just its centre point.
FOOTPRINT = {"container": (1220, 244), "gatehouse": (440, 340), "car": (470, 190),
             "kiosk": (360, 310), "bench": (180, 45), "planter": (160, 160),
             "ticket_machine": (60, 50), "barrier": (40, 520), "pillar": (62, 62),
             "tree": (500, 500), "lamp": (260, 60), "floodlight": (80, 40),
             "fence": (300, 10), "bollards": (600, 20), "bollard_row": (600, 20),
             "gateline": (30, 150), "board": (300, 20), "bin": (55, 55)}
FLAT = {"road", "lane_line", "bay_line", "platform_edge", "kerb"}
CELL = 100.0     # cm; occupancy grid of walked positions


class _Placer:
    def __init__(self, cameras, half, walked=()):
        self.cams = [(c["location"][0], c["location"][1], c["rotation"][1]) for c in cameras]
        self.half = half
        self.items: list[dict] = []
        self.walked = set(walked)
        self.skipped = 0

    def on_path(self, x, y, radius=150.0):
        r = int(radius // CELL) + 1
        cx, cy = int(x // CELL), int(y // CELL)
        return any((cx + i, cy + j) in self.walked
                   for i in range(-r, r + 1) for j in range(-r, r + 1))

    def clear(self, x, y, radius=800.0):
        for cx, cy, yaw in self.cams:
            dx, dy = x - cx, y - cy
            d = math.hypot(dx, dy)
            if d < 250.0:
                return False
            if d < radius and dx * math.cos(math.radians(yaw)) + dy * math.sin(math.radians(yaw)) > 0:
                return False
        return True

    def in_near_frustum(self, x, y, yaw, size, near=1000.0, half_fov=60.0):
        """Any footprint sample within `near` cm in front of a camera's view."""
        pts = [(x, y)]
        if size:
            sx, sy = size[0] / 2.0, size[1] / 2.0
            c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            n = 6
            for i in range(n + 1):
                for j in range(n + 1):
                    u, v = -sx + 2 * sx * i / n, -sy + 2 * sy * j / n
                    pts.append((x + c * u - s * v, y + s * u + c * v))
        for cx, cy, cyaw in self.cams:
            fx, fy = math.cos(math.radians(cyaw)), math.sin(math.radians(cyaw))
            for px, py in pts:
                dx, dy = px - cx, py - cy
                d = math.hypot(dx, dy)
                if d < 1e-6:
                    return True
                if d < near and (dx * fx + dy * fy) / d > math.cos(math.radians(half_fov)):
                    return True
        return False

    def add(self, kind, x, y, yaw=0.0, check=True, **kw):
        if check and not self.clear(x, y):
            return False
        size = kw.get("size") or FOOTPRINT.get(kind)
        if kind in ("fence",) and kw.get("length"):
            size = (kw["length"], 10)
        if kind not in FLAT and self.in_near_frustum(x, y, yaw, size):
            # No static geometry in the first 10 m of any camera's view.
            self.skipped_frustum = getattr(self, "skipped_frustum", 0) + 1
            return False
        if kind not in FLAT and self.walked and self.on_path(x, y, 250.0 if kind in (
                "car", "container", "kiosk", "stage", "canopy", "gatehouse") else 120.0):
            # Solid geometry never sits where a person's truth says they walked.
            self.skipped += 1
            return False
        self.items.append({"kind": kind, "x": round(x, 1), "y": round(y, 1),
                           "yaw": round(yaw, 1), **kw})
        return True


def _buildings(P, rng, sides, edge, facade_n, heights=(3, 6), depth=(900, 1400), gap_p=0.15,
               style="mixed"):
    """Rows of buildings just outside `edge` on the given sides ('N','S','E','W')."""
    for side in sides:
        pos = -edge - 2000
        while pos < edge + 2000:
            w = rng.uniform(900, 1800)
            d = rng.uniform(*depth)
            h = rng.randint(*heights) * 320.0 + rng.uniform(0, 60)
            if rng.random() >= gap_p:
                c = pos + w / 2
                if side in "NS":
                    y = (edge + d / 2) * (1 if side == "N" else -1)
                    P.add("building", c, y, 0, check=False, size=[w, d, h],
                          facade=rng.randrange(facade_n), style=style)
                else:
                    x = (edge + d / 2) * (1 if side == "E" else -1)
                    P.add("building", x, c, 0, check=False, size=[d, w, h],
                          facade=rng.randrange(facade_n), style=style)
            pos += w + rng.uniform(0, 60)


def _far_ring(P, rng, facade_n, n=24):
    for i in range(n):
        ang = 2 * math.pi * i / n
        r = rng.uniform(9000, 12000)
        P.add("building", r * math.cos(ang), r * math.sin(ang), math.degrees(ang) + 90,
              check=False, size=[rng.uniform(2000, 3500), rng.uniform(1500, 2500),
                                 rng.uniform(1500, 4500)], facade=rng.randrange(facade_n))


def _rim_furniture(P, rng, half, kinds, n):
    for _ in range(n):
        side = rng.choice([-1, 1])
        along = rng.uniform(-half + 200, half - 200)
        x, y = (along, side * (half - 180)) if rng.random() < 0.5 else (side * (half - 180), along)
        P.add(rng.choice(kinds), x, y, 0 if abs(y) > abs(x) else 90)


def _lamps(P, half, step=1200.0, inset=80.0):
    for s in (-1, 1):
        for i in range(-3, 4):
            v = i * step + 300
            if abs(v) > half:
                continue
            P.add("lamp", v, s * (half - inset), 90 if s < 0 else -90)
            P.add("lamp", s * (half - inset), v - 600, 0 if s < 0 else 180)


def walked_cells(manifest: dict[str, Any]) -> set[tuple[int, int]]:
    return {(int(k["location"][0] // CELL), int(k["location"][1] // CELL))
            for t in manifest.get("tracks", []) for k in t["keys"]}


def faced_sides(cameras: list[dict]) -> list[str]:
    """Square sides ('N','S','E','W' in Unreal axes) the cameras look toward."""
    out = []
    for c in cameras:
        yaw = math.radians(c["rotation"][1])
        fx, fy = math.cos(yaw), math.sin(yaw)
        side = ("E" if fx > 0 else "W") if abs(fx) >= abs(fy) else ("N" if fy > 0 else "S")
        if side not in out:
            out.append(side)
    return out or ["N"]


def _side_xy(side, along, offset):
    """Point `along` a side at distance `offset` from the centre line."""
    return {"N": (along, offset), "S": (along, -offset),
            "E": (offset, along), "W": (-offset, along)}[side]


def _side_yaw(side):
    return {"N": 0.0, "S": 0.0, "E": 90.0, "W": 90.0}[side]


def _far(P, rng, facade_n, kind, n=24, heights=(1500, 4500)):
    for i in range(n):
        ang = 2 * math.pi * (i + rng.uniform(-0.3, 0.3)) / n
        r = rng.uniform(8000, 12000)
        x, y = r * math.cos(ang), r * math.sin(ang)
        if kind == "trees":
            P.add("tree", x, y, 0, check=False, scale=rng.uniform(1.5, 2.5))
        else:
            P.add("building", x, y, math.degrees(ang) + 90, check=False,
                  size=[rng.uniform(2000, 3500), rng.uniform(1500, 2500),
                        rng.uniform(*heights)], facade=rng.randrange(facade_n))


def plan(site_type: str, seed: int, half_cm: float, cameras: list[dict],
         lighting: dict | None = None, weather: dict | None = None,
         facade_n: int = 6, walked=()) -> dict[str, Any]:
    if site_type not in SITE_TYPES:
        raise ValueError(f"unknown site type {site_type!r}")
    rng = random.Random(f"env:{site_type}:{seed}")
    layout = LAYOUT[site_type]
    half = float(half_cm)
    P = _Placer(cameras, half, walked)
    edge = half + 700.0
    front = faced_sides(cameras)
    back = [s for s in "NSEW" if s not in front]
    span = 2 * edge

    if layout == "retail_frontage":
        # A high street: shopfronts close on every side, a road in front of
        # the cameras, parked cars, dense furniture.
        side = front[0]
        roff = half + 500
        P.add("road", *_side_xy(side, 0, roff), _side_yaw(side), check=False, width=700.0)
        for k in range(-12, 13):
            P.add("lane_line", *_side_xy(side, k * 900.0, roff), _side_yaw(side), check=False,
                  length=450.0)
            if rng.random() < 0.7:
                P.add("car", *_side_xy(side, k * 600.0 + rng.uniform(-80, 80),
                                       roff + rng.choice([-240, 240])),
                      _side_yaw(side) + rng.choice([0, 180]), check=False, paint=rng.randrange(7))
        for sd in "NSEW":
            _buildings(P, rng, sd, (roff + 800) if sd == side else edge, facade_n,
                       heights=(2, 4), depth=(800, 1100), gap_p=0.05, style="shopfront")
        _lamps(P, half, step=900.0)
        _rim_furniture(P, rng, half, ["bench", "bin", "bollards", "planter", "tree"], 20)
        _far(P, rng, facade_n, "buildings", heights=(1200, 2500))
    elif layout == "car_park":
        # Open lot: bays + cars across the view, ticket machine, barrier, lamp
        # columns, low perimeter wall and only a few distant buildings.
        for row in range(-4, 5):
            ry = row * 700.0
            for b in range(-14, 15):
                bx = b * 250.0
                if abs(bx) > half + 400 or abs(ry) > half + 400:
                    continue
                P.add("bay_line", bx, ry, 90, check=False, length=480.0)
                if rng.random() < 0.6:
                    P.add("car", bx + 125, ry, 90 + rng.choice([0, 180]), paint=rng.randrange(7))
        for sd in "NSEW":
            P.add("wall", *_side_xy(sd, 0, edge), 0, check=False,
                  size=[span, 40, 100] if sd in "NS" else [40, span, 100])
        P.add("barrier", *_side_xy(front[0], 300, edge - 200), _side_yaw(front[0]) + 90)
        P.add("ticket_machine", *_side_xy(front[0], -300, edge - 250), _side_yaw(front[0]))
        _lamps(P, half + 300, step=1600.0)
        _far(P, rng, facade_n, "buildings", n=8, heights=(1500, 3000))
    elif layout == "plaza":
        # Event plaza: stage in view, kiosks, circle of planters, tall city.
        s0 = front[0]
        P.add("stage", *_side_xy(s0, 0, half * 0.8), _side_yaw(s0) + 180, size=[1400, 700, 140])
        for k in range(5):
            P.add("kiosk", *_side_xy(front[-1], -half * 0.6 + k * half * 0.3, half * 0.55), 0)
        for k in range(10):
            ang = 2 * math.pi * k / 10
            P.add("planter", 0.35 * half * math.cos(ang), 0.35 * half * math.sin(ang), 0)
        _lamps(P, half)
        _rim_furniture(P, rng, half, ["bench", "bin", "tree", "tree"], 16)
        _buildings(P, rng, "NSEW", edge + 600, facade_n, heights=(6, 12), style="mixed")
        _far(P, rng, facade_n, "buildings", heights=(3000, 6000))
    elif layout == "station_forecourt":
        # A long, low station building with a canopy facing the cameras, a
        # ticket-gate line and departure boards under it.
        s0 = front[0]
        P.add("building", *_side_xy(s0, 0, edge + 900), 0, check=False,
              size=[span + 2000, 1800, 900] if s0 in "NS" else [1800, span + 2000, 900],
              facade=rng.randrange(facade_n), style="station")
        P.add("canopy", *_side_xy(s0, 0, edge - 250), 0, check=False,
              size=[span, 800, 550] if s0 in "NS" else [800, span, 550])
        for k in range(-7, 8):
            P.add("gateline", *_side_xy(s0, k * 160.0, edge - 450), _side_yaw(s0), check=False)
        for k in range(4):
            P.add("board", *_side_xy(s0, -1200 + k * 800, edge - 300), _side_yaw(s0), check=False)
        _lamps(P, half, step=1000.0)
        _rim_furniture(P, rng, half, ["bench", "bench", "bin"], 14)
        for sd in back:
            _buildings(P, rng, sd, edge + 1500, facade_n, heights=(3, 5), style="mixed")
        _far(P, rng, facade_n, "buildings", n=16, heights=(1000, 2500))
    elif layout == "warehouse_yard":
        # Perimeter: fence line, a road outside it, gatehouse, containers and
        # floodlights; no buildings near, open grass beyond.
        for sd in "NSEW":
            n = int(span // 300)
            for k in range(n):
                v = -edge + 150 + k * 300
                if sd == front[0] and abs(v) < 500:
                    continue                                   # the gate
                P.add("fence", *_side_xy(sd, v, edge), _side_yaw(sd), check=False, length=300.0)
        P.add("road", *_side_xy(front[0], 0, edge + 900), _side_yaw(front[0]), check=False,
              width=650.0)
        P.add("gatehouse", *_side_xy(front[0], 900, edge - 300), 0, check=False)
        for k in range(8):
            P.add("container", *_side_xy(front[-1], -half + 300 + k * 650, half - 100),
                  _side_yaw(front[-1]) + 90, check=False, paint=rng.randrange(7))
        for sx in (-1, 1):
            for sy in (-1, 1):
                P.add("floodlight", sx * (edge - 150), sy * (edge - 150), 0, check=False)
        _lamps(P, half, step=1200.0)
        _far(P, rng, facade_n, "trees", n=30)
    elif layout == "office_forecourt":
        # One or two large glazed blocks with a setback forecourt, bollards,
        # planters and a row of visitor car spaces.
        s0 = front[0]
        for k, w in enumerate([span * 0.55, span * 0.35] if rng.random() < 0.6 else [span * 0.8]):
            along = (-span * 0.2 if k == 0 else span * 0.3) if w < span * 0.7 else 0
            P.add("building", *_side_xy(s0, along, edge + 1300), 0, check=False,
                  size=[w, 2200, rng.choice([4200, 5400, 6600])] if s0 in "NS"
                  else [2200, w, rng.choice([4200, 5400, 6600])],
                  facade=0, style="glass")
        for k in range(-4, 5):
            P.add("bollard_row", *_side_xy(s0, k * 450.0, half - 120), _side_yaw(s0))
        for k in range(7):
            P.add("planter", *_side_xy(s0, -half * 0.8 + k * half * 0.27, half * 0.45), 0)
        side2 = front[-1] if len(front) > 1 else back[0]
        for k in range(10):
            P.add("bay_line", *_side_xy(side2, -1100 + k * 250, edge - 100),
                  _side_yaw(side2) + 90, check=False, length=480.0)
            if rng.random() < 0.7:
                P.add("car", *_side_xy(side2, -975 + k * 250, edge - 100),
                      _side_yaw(side2) + 90, check=False, paint=rng.randrange(7))
        _lamps(P, half, step=1600.0)
        _rim_furniture(P, rng, half, ["bench", "tree"], 8)
        for sd in back:
            _buildings(P, rng, sd, edge + 1200, facade_n, heights=(8, 14), style="office")
        _far(P, rng, facade_n, "buildings", heights=(4000, 8000))

    # Interior lamp grid (every ~11 m) so night scenes have no black gaps;
    # skipped where people walk or in a camera's near view, like any prop.
    g = 1100.0
    n = int(half // g)
    for ix in range(-n, n + 1):
        for iy in range(-n, n + 1):
            P.add("lamp", ix * g + 180.0, iy * g + 180.0, 45.0)

    main, rim, far = GROUND[layout]
    light = dict(lighting or {"preset": "day"})
    if light.get("hour") is None:
        # No time of day in the scenario: sun from the seed, so shadows differ.
        light["sun_yaw"] = round(rng.uniform(0, 360), 1)
        light["sun_pitch"] = round(-rng.uniform(25, 60), 1)
    return {"site_type": site_type, "layout": layout, "seed": seed, "half_cm": half,
            "ground": {"main": main, "rim": rim, "far": far}, "faced_sides": front,
            "items": P.items, "skipped_on_paths": P.skipped,
            "lighting": light, "weather": weather or {"wet": 0.0, "fog": 0.0}}


def plan_for_run(scenario: dict[str, Any], manifest: dict[str, Any], half_cm: float,
                 preset: str | None = None, site_type: str | None = None) -> dict[str, Any]:
    env = scenario.get("environment") or {}
    site = site_type or infer_site_type(env)
    return plan(site, int(manifest.get("seed", 0)), half_cm, manifest.get("cameras", []),
                lighting_for(scenario, preset), weather_for(scenario),
                walked=walked_cells(manifest))


def walk_envelope(manifest: dict[str, Any], margin_cm: float = 300.0,
                  minimum: float = 2000.0) -> float:
    """Half-size (cm) of the square holding every track key and camera, + margin."""
    ext = minimum - margin_cm
    for t in manifest.get("tracks", []):
        for k in t["keys"]:
            ext = max(ext, abs(k["location"][0]), abs(k["location"][1]))
    for c in manifest.get("cameras", []):
        ext = max(ext, abs(c["location"][0]), abs(c["location"][1]))
    return ext + margin_cm


# --------------------------------------------------------------------------
# v2: plans from survsim.generator.layout (real site structure, installer rig)
# --------------------------------------------------------------------------

PLAN_VERSION = 2


def _to_ue(x: float, y: float) -> tuple[float, float]:
    return round(x * 100.0, 1), round(-y * 100.0, 1)


def layout_camera_manifest(layout: dict) -> list[dict]:
    """The layout's installer cameras in manifest form (Unreal cm / degrees)."""
    out = []
    for c in layout["cameras"]:
        ux, uy = _to_ue(c["x"], c["y"])
        out.append({"id": c["id"], "location": [ux, uy, round(c["z"] * 100.0, 1)],
                    "rotation": [c["pitch"], round(-c["yaw"], 1), 0.0],
                    "resolution": list(c.get("res", [1280, 720])), "fps": 10.0,
                    "fov_deg": c["hfov_deg"], "focal_mm": None, "sensor_mm": None,
                    "mount": c["mount"], "target": c["target"]})
    return out


def manifest_camera_models(cameras: list[dict]):
    """Manifest cameras back into survsim Camera models (for usably_visible)."""
    from survsim.core.models import Camera, Mount
    out = []
    for c in cameras:
        x, y, z = c["location"]
        pitch, yaw, _ = c["rotation"]
        out.append(Camera(id=c["id"], mount=Mount(pos=(x / 100.0, -y / 100.0, z / 100.0),
                                                  yaw=-yaw, pitch=pitch),
                          optics={"hfov_deg": float(c.get("fov_deg") or 78.0)},
                          capture={"res": list(c.get("resolution", [1280, 720])), "fps": 10.0}))
    return out


def _footprint_on_path(P: _Placer, it: dict, pad_cm: float) -> bool:
    if not P.walked:
        return False
    c, s = math.cos(math.radians(it["yaw"])), math.sin(math.radians(it["yaw"]))
    sx, sy = it["size"][0] / 2 + pad_cm, it["size"][1] / 2 + pad_cm
    nx, ny = max(1, int(2 * sx // CELL)), max(1, int(2 * sy // CELL))
    for i in range(nx + 1):
        for j in range(ny + 1):
            u, v = -sx + 2 * sx * i / nx, -sy + 2 * sy * j / ny
            px, py = it["x"] + c * u - s * v, it["y"] + s * u + c * v
            if (int(px // CELL), int(py // CELL)) in P.walked:
                return True
    return False


def plan_from_layout(layout: dict, cameras: list[dict] | None = None, walked=(),
                     lighting: dict | None = None) -> dict[str, Any]:
    """Layout (SI) -> editor plan v2 (Unreal cm). Cameras default to the layout's rig.

    Enforces the same two rules as v1 against the cameras actually rendered:
    no solid within a camera's first 10 m (+-60 deg), nothing solid on a walked
    1 m cell (with a footprint test, so a block cannot straddle a path).
    """
    from survsim.generator.layout import FLAT as LFLAT
    from survsim.generator.layout import sun_to_unreal
    cams = cameras if cameras is not None else layout_camera_manifest(layout)
    half = (layout["core_half"] + 3.0) * 100.0
    P = _Placer(cams, half, walked)
    items, dropped = [], {"near_view": 0, "on_path": 0}
    for it in layout["items"]:
        ux, uy = _to_ue(it["x"], it["y"])
        out = {k: v for k, v in it.items() if k not in ("x", "y", "yaw", "w", "d", "h")}
        out.update(kind=it["kind"], x=ux, y=uy, yaw=round(-it["yaw"] % 360.0, 1),
                   size=[round(it["w"] * 100, 1), round(it["d"] * 100, 1), round(it["h"] * 100, 1)])
        flat = it["kind"] in LFLAT or it["h"] < 0.3
        # Built meshes overhang the planned footprint (roofs, tree crowns): pad it.
        fp = ([max(out["size"][0], 60 * it["h"]), max(out["size"][1], 60 * it["h"])]
              if it["kind"] == "tree" else [out["size"][0] + 100, out["size"][1] + 100])
        if not flat and it["kind"] not in ("canopy", "shop_sign", "awning") and \
                P.in_near_frustum(ux, uy, out["yaw"], fp):
            dropped["near_view"] += 1
            continue
        if not flat and it["kind"] not in ("canopy", "shop_sign", "awning", "lamp") and \
                _footprint_on_path(P, out, 60.0 if it["kind"] != "block" else 100.0):
            dropped["on_path"] += 1
            continue
        items.append(out)
    light = dict(lighting or {"preset": "day"})
    t = layout.get("time") or {}
    if light.get("hour") is None and t:
        light["sun_yaw"], light["sun_pitch"] = sun_to_unreal(t["sun_elevation"], t["sun_azimuth"])
        light["solar"] = t
    ents = []
    for e in layout["entrances"]:
        ux, uy = _to_ue(e["x"], e["y"])
        ents.append({"id": e["id"], "x": ux, "y": uy, "kind": e["kind"]})
    return {"version": PLAN_VERSION, "site_type": layout["site_type"],
            "layout": f"{layout['site_type']}_v2", "variant": layout["variant"],
            "seed": layout["seed"], "half_cm": half,
            "ground": layout["ground"], "items": items, "dropped": dropped,
            "skipped_on_paths": dropped["on_path"], "faced_sides": faced_sides(cams),
            "lighting": light, "weather": dict(layout["weather"]), "entrances": ents,
            "cameras": cams, "layout_report": layout["report"]}


def entrance_camera_candidates(plan_v2: dict, cameras: list[dict]) -> dict[str, list[str]]:
    """Per entrance, the cameras that usably see it (frustum + >= 20 px), host side.

    The editor then confirms each by a line trace against the built Env
    (occlusion), so an entrance passes only if frame, size and line of sight agree.
    """
    import numpy as np

    from survsim.core.geometry import usably_visible
    models = manifest_camera_models(cameras)
    out = {}
    for e in plan_v2["entrances"]:
        p = np.array([e["x"] / 100.0, -e["y"] / 100.0, 0.0])
        out[e["id"]] = [m.id for m in models if usably_visible(m, p)]
    return out


def plan_v2_for_run(scenario: dict[str, Any], manifest: dict[str, Any], half_cm: float,
                    preset: str | None = None, site_type: str | None = None,
                    seed: int | None = None) -> dict[str, Any]:
    """v2 plan around a recorded run: its entrances, its cameras, its walked cells."""
    from survsim.generator.layout import build_layout
    env = scenario.get("environment") or {}
    site = site_type or infer_site_type(env)
    ents = [(float(w["pos"][0]), float(w["pos"][1])) for w in env.get("spawn_points", [])
            if "pos" in w]
    lay = build_layout(site, int(manifest.get("seed", 0) if seed is None else seed),
                       core_half=max(10.0, half_cm / 100.0 - 3.0), entrances=ents or None)
    lay_d = lay.to_dict()
    light = lighting_for(scenario, preset)
    p = plan_from_layout(lay_d, manifest.get("cameras", []), walked_cells(manifest), light)
    w = weather_for(scenario)
    if w["wet"] > 0 or w["fog"] > 0:          # the scenario's weather wins over the seed's
        p["weather"] = {"state": "wet" if w["wet"] else p["weather"]["state"],
                        "wet": max(w["wet"], p["weather"]["wet"]), "fog": w["fog"]}
    if ents:
        # Gate the scenario's own entrances (its spawn points), not their
        # projection onto the layout boundary: the projection can sit metres
        # outside a far rig's view ([X] 2026-09-26 batch_v5 street_crowd_far).
        p["entrances"] = [{"id": f"entrance_{i}", "x": round(x * 100.0, 1),
                           "y": round(-y * 100.0, 1), "kind": "scenario_spawn"}
                          for i, (x, y) in enumerate(ents)]
    p["items"] += scenario_fence_items(env)
    p["entrance_cameras"] = entrance_camera_candidates(p, manifest.get("cameras", []))
    return p


FENCED_TAGS = ("restricted", "perimeter_fence")


def scenario_fence_items(env: dict[str, Any], piece_m: float = 3.0, height_m: float = 2.4,
                         gate_gap_m: float = 1.6) -> list[dict[str, Any]]:
    """Palisade `fence_run` pieces along the scenario's own fenced zones (UE cm).

    The v2 layout dresses a generic site; without this the compound the
    Director fences (survsim/director/fences.py) - and that fence_climb /
    fence_cut / fence_tailgate act on - is invisible in the render. Pieces
    within `gate_gap_m` of a ``gate`` prop are left open.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    polys = [Polygon(z["polygon"]) for z in env.get("zones", [])
             if z.get("tag") in FENCED_TAGS and z.get("fenced", True) is not False
             and len(z.get("polygon", [])) >= 3]
    if not polys:
        return []
    u = unary_union(polys)
    gates = [np.array(g["pos"][:2], float) for g in env.get("props", []) if g.get("type") == "gate"]
    items = []
    for geom in getattr(u, "geoms", [u]):
        xy = np.asarray(geom.exterior.coords)
        for a, b in itertools.pairwise(xy):
            L = float(np.hypot(*(b - a)))
            n = max(1, round(L / piece_m))
            yaw = float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))
            for i in range(n):
                c = a + (b - a) * (i + 0.5) / n
                if any(np.hypot(*(c - g)) < gate_gap_m for g in gates):
                    continue
                ux, uy = _to_ue(float(c[0]), float(c[1]))
                items.append({"kind": "fence_run", "style": "palisade", "x": ux, "y": uy,
                              "yaw": round(-yaw % 360.0, 1),
                              "size": [round(L / n * 100, 1), 10.0, height_m * 100]})
    return items

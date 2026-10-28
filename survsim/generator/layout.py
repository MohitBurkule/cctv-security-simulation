"""Site layouts that look like different real places (pure Python, SI units).

`site.build_site` gives the *analytic* topology (zones, entrances, blind
spots). This module gives the *physical* site a camera actually sees: building
footprints, a road network with kerbs and crossings, parking rows, gates that
match the scenario's entrances, zone furniture, vegetation, boundary, signage,
wear decals, weather, a real-date sun, and cameras mounted where an installer
would put them. `survsim.export.env_plan.plan_from_layout` turns it into the
editor's item vocabulary.

Frame: metres, x east, y north, z up, yaw degrees counter-clockwise from +x
(the survsim convention; the Unreal conversion negates y and yaw).

Everything is deterministic from (site_type, seed). The walk *core*
(|x|, |y| <= core_half) is where people are; solid site structure sits outside
it, and only low furniture is allowed on its rim. Each site type is built in a
canonical frame (street/front to the south) and then rotated by a seeded
multiple of 90 degrees, so two seeds of one type differ in orientation as well
as in shape, while two types differ in what they are made of.

Checks carried by the layout itself (`gate`):
* every entrance is reachable on foot from the core centre (grid BFS around
  every solid footprint);
* every entrance is usably visible (`geometry.usably_visible`, the one
  definition) and not occluded by a solid, from at least one camera;
* no solid footprint enters any camera's first 10 m (+-60 deg), the same rule
  the editor's frustum gate enforces.
"""
from __future__ import annotations

import datetime as _dt
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SITE_TYPES_V1 = ("office_building", "transit_station", "perimeter_site",
                 "retail_street", "car_park", "event_plaza")
SITE_TYPES_V2 = ("loading_dock", "school_gate", "rail_platform", "hospital_entrance",
                 "utility_yard", "construction_site", "farm_yard", "residential_close")
SITE_TYPES = SITE_TYPES_V1 + SITE_TYPES_V2

# Kinds that stop a pedestrian / a line of sight. Heights in metres.
SOLID_H = {"block": None, "bus_shelter": 2.6, "bin_store": 1.9, "container": 2.6,
           "gatehouse": 3.0, "kiosk": 2.6, "food_truck": 3.0, "stage": 1.4,
           "wall_run": None, "fence_run": None, "hedge": 1.4, "trailer": 3.8,
           "portable_toilet": 2.3, "speaker_tower": 6.0, "canopy_col": 5.5,
           "skip": 1.6, "pallets": 1.2, "ticket_gates": 1.1, "fountain": 1.2,
           "cycle_rack": 0.9, "bench": 0.9, "bin": 1.0, "planter": 1.0, "tree": 6.0,
           "lamp": 6.0, "floodlight": 12.0, "camera_pole": 5.0, "sign": 2.6,
           "barrier_row": 1.1, "ticket_machine": 1.8, "car": 1.5, "bollards": 0.9,
           "flagpole": 8.0, "phone_box": 2.5, "trolley_bay": 1.2, "asset": 1.5,
           "barrier": 1.0, "monument": 3.5, "tent": 3.0, "dock_door": 4.0}
# Flat things: never block a path or a view, may sit anywhere.
FLAT = {"road", "kerb", "lane_line", "crossing", "bay_line", "pavement", "grass",
        "hatch", "decal", "tactile", "yard_line"}
# Overhead or passable: never refused for standing on an entrance's throat.
PASSABLE = {"lamp", "camera_pole", "floodlight", "canopy", "shop_sign", "awning",
            "ticket_gates", "barrier", "sign"}
# What blocks a camera's line of sight (tall and opaque).
OCCLUDERS = {"block", "container", "gatehouse", "kiosk", "food_truck", "trailer",
             "wall_run", "bus_shelter", "bin_store", "portable_toilet", "stage", "tent"}

CITIES = [("London", 51.51, -0.13), ("Manchester", 53.48, -2.24), ("Leeds", 53.80, -1.55),
          ("Glasgow", 55.86, -4.25), ("Bristol", 51.45, -2.59), ("Birmingham", 52.49, -1.89),
          ("Cardiff", 51.48, -3.18), ("Belfast", 54.60, -5.93)]


# --------------------------------------------------------------------------- sun

def solar_position(lat: float, lon: float, when: _dt.datetime) -> tuple[float, float]:
    """(elevation, azimuth) in degrees for a UTC datetime; azimuth from north, clockwise.

    NOAA general solar position equations (fractional-year form); good to
    ~0.5 degrees, which is far below what a shadow in a CCTV frame resolves.
    """
    doy = when.timetuple().tm_yday
    hour = when.hour + when.minute / 60.0 + when.second / 3600.0
    g = 2 * math.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    tst = hour * 60.0 + eqtime + 4.0 * lon                  # true solar time, minutes
    ha = math.radians(tst / 4.0 - 180.0)
    phi = math.radians(lat)
    cos_z = math.sin(phi) * math.sin(decl) + math.cos(phi) * math.cos(decl) * math.cos(ha)
    zen = math.acos(max(-1.0, min(1.0, cos_z)))
    elev = 90.0 - math.degrees(zen)
    az = math.degrees(math.atan2(math.sin(ha),
                                 math.cos(ha) * math.sin(phi) - math.tan(decl) * math.cos(phi)))
    return elev, (az + 180.0) % 360.0


def season_of(month: int) -> str:
    return {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer"}.get(month, "autumn")


def pick_time(rng: random.Random, min_elev: float = 12.0) -> dict[str, Any]:
    """A real date + local daytime with the sun at least `min_elev` up."""
    city, lat, lon = rng.choice(CITIES)
    for _ in range(200):
        date = _dt.date(2026, 1, 1) + _dt.timedelta(days=rng.randrange(365))
        local_h = rng.uniform(7.0, 18.5)
        bst = 1 if 3 < date.month < 11 else 0          # close enough for UK summer time
        utc = _dt.datetime(date.year, date.month, date.day, tzinfo=_dt.timezone.utc) + _dt.timedelta(hours=local_h - bst)
        elev, az = solar_position(lat, lon, utc)
        if elev >= min_elev:
            break
    return {"city": city, "lat": lat, "lon": lon, "date": date.isoformat(),
            "local_hour": round(local_h, 2), "season": season_of(date.month),
            "sun_elevation": round(elev, 1), "sun_azimuth": round(az, 1)}


def sun_to_unreal(elev: float, az: float) -> tuple[float, float]:
    """Directional-light (yaw, pitch) in Unreal axes for a sun at (elev, az).

    Site north is +y (survsim) = -Y (Unreal). The light travels *away* from the
    sun: survsim yaw = 90 - az + 180; Unreal yaw is its negation.
    """
    yaw_s = (90.0 - az + 180.0) % 360.0
    return round(-yaw_s, 1), round(-elev, 1)


# --------------------------------------------------------------------- geometry

@dataclass
class Item:
    kind: str
    x: float
    y: float
    yaw: float = 0.0
    w: float = 0.5            # extent along the item's local x (m)
    d: float = 0.5            # extent along local y (m)
    h: float = 1.0
    props: dict = field(default_factory=dict)

    def corners(self) -> list[tuple[float, float]]:
        c, s = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        out = []
        for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            du, dv = u * self.w / 2, v * self.d / 2
            out.append((self.x + c * du - s * dv, self.y + s * du + c * dv))
        return out

    def samples(self, n: int = 5) -> list[tuple[float, float]]:
        c, s = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        pts = []
        for i in range(n + 1):
            for j in range(n + 1):
                du, dv = -self.w / 2 + self.w * i / n, -self.d / 2 + self.d * j / n
                pts.append((self.x + c * du - s * dv, self.y + s * du + c * dv))
        return pts

    def contains(self, px: float, py: float, pad: float = 0.0) -> bool:
        c, s = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        dx, dy = px - self.x, py - self.y
        u, v = c * dx + s * dy, -s * dx + c * dy
        return abs(u) <= self.w / 2 + pad and abs(v) <= self.d / 2 + pad

    @property
    def solid(self) -> bool:
        return self.kind not in FLAT

    def as_dict(self) -> dict:
        return {"kind": self.kind, "x": round(self.x, 2), "y": round(self.y, 2),
                "yaw": round(self.yaw % 360.0, 1), "w": round(self.w, 2), "d": round(self.d, 2),
                "h": round(self.h, 2), **self.props}


def _rot(x, y, k):
    """Rotate a point by k*90 degrees counter-clockwise."""
    for _ in range(k % 4):
        x, y = -y, x
    return x, y


def _seg_hits_rect(ax, ay, bx, by, it: Item, pad=0.0) -> bool:
    """Does the open segment a->b cross the item's (padded) footprint? (Liang-Barsky)"""
    return _seg_interval(ax, ay, bx, by, it, pad) is not None


def _seg_interval(ax, ay, bx, by, it: Item, pad=0.0):
    """Parameter interval (t0, t1) of the open segment a->b inside the footprint, or None."""
    r = math.hypot(it.w, it.d) / 2 + pad
    # bounding-circle reject
    vx, vy = bx - ax, by - ay
    ll = vx * vx + vy * vy or 1e-9
    t = max(0.0, min(1.0, ((it.x - ax) * vx + (it.y - ay) * vy) / ll))
    if math.hypot(ax + vx * t - it.x, ay + vy * t - it.y) > r:
        return None
    c, s = math.cos(math.radians(it.yaw)), math.sin(math.radians(it.yaw))
    def loc(px, py):
        dx, dy = px - it.x, py - it.y
        return c * dx + s * dy, -s * dx + c * dy
    (u0, v0), (u1, v1) = loc(ax, ay), loc(bx, by)
    du, dv = u1 - u0, v1 - v0
    lo, hi = 1e-3, 1.0 - 1e-3
    for p0, dp, half in ((u0, du, it.w / 2 + pad), (v0, dv, it.d / 2 + pad)):
        if abs(dp) < 1e-12:
            if abs(p0) > half:
                return None
            continue
        t0, t1 = (-half - p0) / dp, (half - p0) / dp
        if t0 > t1:
            t0, t1 = t1, t0
        lo, hi = max(lo, t0), min(hi, t1)
        if lo > hi:
            return None
    return lo, hi


# ----------------------------------------------------------------------- layout

@dataclass
class Layout:
    site_type: str
    seed: int
    core_half: float
    rotation: int
    variant: dict
    items: list[Item] = field(default_factory=list)
    entrances: list[dict] = field(default_factory=list)
    cameras: list[dict] = field(default_factory=list)
    time: dict = field(default_factory=dict)
    weather: dict = field(default_factory=dict)
    ground: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)

    def add(self, kind, x, y, yaw=0.0, w=0.5, d=0.5, h=None, **props) -> Item | None:
        it = Item(kind, float(x), float(y), float(yaw), float(w), float(d),
                  float(h if h is not None else (SOLID_H.get(kind) or 1.0)), props)
        if it.solid and kind not in PASSABLE and self._blocks_entrance(it):
            return None
        self.items.append(it)
        return it

    def _blocks_entrance(self, it: Item) -> bool:
        """Solids never sit on an entrance's 3 m throat (inside or outside)."""
        for e in self.entrances:
            nx, ny = math.cos(math.radians(e["yaw"])), math.sin(math.radians(e["yaw"]))
            for t in (-3.0, -1.5, 0.0):
                if it.contains(e["x"] + nx * t, e["y"] + ny * t, pad=0.6):
                    return True
        return False

    def to_dict(self) -> dict:
        return {"site_type": self.site_type, "seed": self.seed, "core_half": self.core_half,
                "rotation": self.rotation * 90, "variant": self.variant,
                "items": [i.as_dict() for i in self.items], "entrances": self.entrances,
                "cameras": self.cameras, "time": self.time, "weather": self.weather,
                "ground": self.ground, "report": self.report}


# Per-type builders work in the canonical frame: front (street) to the south,
# back to the north. B = boundary half-size (core + margin).

def _row(L, kind, x0, x1, y, step, yaw=0.0, jitter=0.0, rng=None, p=1.0, **kw):
    x = x0
    while x <= x1 + 1e-6:
        if rng is None or rng.random() < p:
            L.add(kind, x + (rng.uniform(-jitter, jitter) if rng else 0), y, yaw, **kw)
        x += step


def _col(L, kind, y0, y1, x, step, yaw=90.0, rng=None, p=1.0, **kw):
    y = y0
    while y <= y1 + 1e-6:
        if rng is None or rng.random() < p:
            L.add(kind, x, y, yaw, **kw)
        y += step


def _road_x(L, y, x0, x1, width, kerb=True, lanes=True, name="road"):
    """A road along x at y, with kerbs on both edges and a dashed centre line."""
    L.add("road", (x0 + x1) / 2, y, 0.0, w=x1 - x0, d=width, h=0.01, name=name)
    if kerb:
        for s in (-1, 1):
            L.add("kerb", (x0 + x1) / 2, y + s * (width / 2 + 0.15), 0.0, w=x1 - x0, d=0.3, h=0.14)
    if lanes:
        x = x0 + 2.0 + (L.seed * 2.37) % 7.5
        while x < x1 - 2.0:
            L.add("lane_line", x, y, 0.0, w=3.0, d=0.12, h=0.005)
            x += 7.5


def _road_y(L, x, y0, y1, width, kerb=True, lanes=True):
    L.add("road", x, (y0 + y1) / 2, 90.0, w=y1 - y0, d=width, h=0.01)
    if kerb:
        for s in (-1, 1):
            L.add("kerb", x + s * (width / 2 + 0.15), (y0 + y1) / 2, 90.0, w=y1 - y0, d=0.3, h=0.14)
    if lanes:
        y = y0 + 2.0 + (L.seed * 3.11) % 7.5
        while y < y1 - 2.0:
            L.add("lane_line", x, y, 90.0, w=3.0, d=0.12, h=0.005)
            y += 7.5


def _crossing(L, x, y, road_yaw, road_w):
    """Zebra: stripes across the road, parallel to its direction."""
    L.add("crossing", x, y, road_yaw, w=4.0, d=road_w, h=0.005)
    c, s = math.cos(math.radians(road_yaw + 90)), math.sin(math.radians(road_yaw + 90))
    for side in (-1, 1):              # tactile paving both kerbs
        L.add("tactile", x + c * side * (road_w / 2 + 0.9), y + s * side * (road_w / 2 + 0.9),
              road_yaw, w=3.0, d=1.2, h=0.005)


def _footprint(L, rng, shape, cx, cy, w, d, h, style, wing=None, **props):
    """Building blocks for a footprint shape centred at (cx, cy), front face south.

    rect | L | U | courtyard | twin — all axis-aligned rectangles, so the
    editor's axis-aligned bounds are the true footprint.
    """
    t = wing or max(8.0, min(w, d) * 0.35)
    blocks = []
    if shape == "rect":
        blocks.append((cx, cy, w, d))
    elif shape == "L":
        s = rng.choice([-1, 1])
        blocks += [(cx, cy + d / 2 - t / 2, w, t), (cx + s * (w / 2 - t / 2), cy - t / 2, t, d - t)]
    elif shape == "U":
        blocks += [(cx, cy + d / 2 - t / 2, w, t), (cx - w / 2 + t / 2, cy - t / 2, t, d - t),
                   (cx + w / 2 - t / 2, cy - t / 2, t, d - t)]
    elif shape == "courtyard":
        blocks += [(cx, cy + d / 2 - t / 2, w, t), (cx, cy - d / 2 + t / 2, w, t),
                   (cx - w / 2 + t / 2, cy, t, d - 2 * t), (cx + w / 2 - t / 2, cy, t, d - 2 * t)]
    else:  # twin
        g = rng.uniform(6.0, 10.0)
        bw = (w - g) / 2
        blocks += [(cx - (bw + g) / 2, cy, bw, d), (cx + (bw + g) / 2, cy, bw, d)]
    out = []
    for i, (bx, by, bw, bd) in enumerate(blocks):
        hh = h * (1.0 if i == 0 else rng.choice([0.6, 0.8, 1.0]))
        it = L.add("block", bx, by, 0.0, w=bw, d=bd, h=hh, style=style,
                   facade=rng.randrange(6), **props)
        if it:
            out.append(it)
    return out


def _entrance_positions(n, B, rng, sides, spread=0.55):
    """`n` entrances on the boundary sides given (canonical frame), spaced along them."""
    out = []
    for i in range(n):
        side = sides[i % len(sides)]
        k = i // len(sides)
        along = (rng.uniform(-0.2, 0.2) + (k - (n // len(sides) - 1) / 2) * spread) * B
        along = max(-0.8 * B, min(0.8 * B, along))
        out.append((side, along))
    return out


_SIDE = {"S": (0.0, -1.0), "N": (0.0, 1.0), "E": (1.0, 0.0), "W": (-1.0, 0.0)}


def _side_point(side, along, off):
    nx, ny = _SIDE[side]
    return (along if nx == 0 else nx * off, along if ny == 0 else ny * off)


def _yaw_out(side):
    return {"S": -90.0, "N": 90.0, "E": 0.0, "W": 180.0}[side]


def _boundary(L, kind, B, gaps, rng, height, sides="NSEW", step=3.0, **props):
    """Boundary runs (fence/wall/hedge) on the square of half B with gaps at entrances."""
    for side in sides:
        n = int(2 * B / step)
        ph = rng.uniform(0.0, step)
        for k in range(n):
            a = -B + ph + k * step
            if a > B:
                break
            if any(g[0] == side and abs(g[1] - a) < 2.6 for g in gaps):
                continue
            x, y = _side_point(side, a, B)
            yaw = 0.0 if side in "NS" else 90.0
            L.add(kind, x, y, yaw, w=step, d=0.25 if kind != "hedge" else 1.0, h=height, **props)


def _decals(L, rng, B, n, kinds):
    for _ in range(n):
        k = rng.choice(kinds)
        x, y = rng.uniform(-B, B), rng.uniform(-B, B)
        size = {"oil": (1.2, 0.8), "dirt": (3.0, 2.0), "gum": (0.4, 0.4), "crack": (3.0, 0.3),
                "tyre": (4.0, 0.5), "puddle": (2.2, 1.4), "leaves": (1.5, 1.5)}[k]
        sc = rng.uniform(0.6, 1.6)
        L.add("decal", x, y, rng.uniform(0, 180), w=size[0] * sc, d=size[1] * sc, h=0.0, decal=k)


# ---- office -------------------------------------------------------------------

def _office(L, rng, B, n_ent):
    C = L.core_half
    shape = rng.choice(["rect", "L", "U", "courtyard", "twin"])
    floors = rng.randint(3, 11)
    depth = rng.uniform(18, 30)
    width = rng.uniform(1.4, 2.4) * B
    face_y = B + rng.uniform(0.0, 2.0)               # facade line (north)
    _footprint(L, rng, shape, rng.uniform(-6, 6), face_y + depth / 2, width, depth,
               floors * 3.6 + 1.5, rng.choice(["glass", "glass", "office"]))
    L.variant.update(shape=shape, floors=floors)
    # Entrance canopy + revolving door zone over the doors on the facade.
    for e in L.entrances:
        if e["side"] == "N":
            L.add("canopy", e["x"], face_y - 1.5, 0.0, w=7.0, d=3.0, h=3.4, flat_roof=True)
    # Street to the south: pavement + road + crossing on the entrance axis.
    ry = -(B + rng.uniform(5.0, 8.0))
    rw = rng.choice([6.5, 7.3, 9.0])
    _road_x(L, ry, -3 * B, 3 * B, rw)
    L.add("pavement", 0.0, -(B + (abs(ry) - B - rw / 2) / 2), 0.0, w=6 * B,
          d=abs(ry) - B - rw / 2, h=0.14)
    _crossing(L, rng.uniform(-0.4, 0.4) * B, ry, 0.0, rw)
    # Visitor parking on the east or west side: perpendicular bays.
    ps = rng.choice([-1, 1])
    px = ps * (B + 4.0)
    y0 = -0.8 * B + rng.uniform(0.0, 5.0)
    for k in range(int(1.4 * B / 2.5)):
        y = y0 + k * 2.5
        L.add("bay_line", px, y, 0.0, w=5.0, d=0.1, h=0.005)
        if rng.random() < 0.65:
            L.add("car", px + ps * 0.2, y + 1.25, 0.0 if ps > 0 else 180.0, w=4.5, d=1.85,
                  paint=rng.randrange(7))
    L.add("sign", px - ps * 1.0, -0.85 * B, 90.0, w=0.8, d=0.1, text="VISITORS", colour="blue")
    boundary = rng.choice(["open", "hedge", "low_wall"])
    if boundary != "open":
        gaps = [(e["side"], e["along"]) for e in L.entrances] + [("S", 0.0)]
        _boundary(L, "hedge" if boundary == "hedge" else "wall_run", B + 1.0, gaps, rng,
                  1.2 if boundary == "hedge" else 0.9, sides="SEW" if ps else "SE")
    L.variant["boundary"] = boundary
    # Furniture near the doors: cycle racks, smoking shelter, bins, benches, planters.
    for e in L.entrances[:2]:
        L.add("cycle_rack", e["x"] + rng.choice([-6, 6]), e["y"] - 3.0, 0.0, w=5.0, d=1.0)
    L.add("bus_shelter", -ps * (B - 2.0), B - 5.0, 90.0, w=3.2, d=1.6, style="smoking")
    L.add("bin_store", ps * (B + 2.5), B + 1.5, 0.0, w=4.0, d=3.0)
    for k in range(3):
        L.add("flagpole", -B * 0.6 + k * 1.8, -B - 1.0, 0.0, w=0.2, d=0.2, h=8.0,
              colour=["red", "blue", "white"][k])
    trees = rng.choice([4, 8, 14])
    _row(L, "tree", -1.4 * B, 1.4 * B, -(B + 1.2), 2.8 * B / max(1, trees), rng=rng, p=0.8,
         w=1.0, d=1.0, h=rng.uniform(5, 8))
    L.variant["trees"] = trees
    _row(L, "planter", -0.8 * C, 0.8 * C, 0.45 * C, 0.4 * C, rng=rng, p=0.6, w=1.6, d=1.6)
    _row(L, "bollards", -0.6 * B, 0.6 * B, -(B - 0.4), 3.5, rng=rng, p=0.7, w=3.0, d=0.2)


# ---- transit -------------------------------------------------------------------

def _transit(L, rng, B, n_ent):
    shape = rng.choice(["rect", "L", "rect"])
    face_y = B + rng.uniform(3.0, 6.0)
    depth = rng.uniform(14, 22)
    _footprint(L, rng, shape, 0.0, face_y + depth / 2, rng.uniform(2.2, 3.0) * B, depth,
               rng.uniform(8, 13), "station")
    if rng.random() < 0.5:     # clock tower
        L.add("block", rng.choice([-1, 1]) * B * 0.7, face_y + 3.0, 0.0, w=5.0, d=5.0,
              h=rng.uniform(18, 26), style="station", facade=rng.randrange(6))
    # Long canopy over the frontage on columns, gatelines under it.
    cw = rng.uniform(1.4, 2.0) * B
    cy = face_y - 3.0
    L.add("canopy", 0.0, cy, 0.0, w=cw, d=6.0, h=5.5, flat_roof=True)
    for e in L.entrances:
        if e["side"] == "N":
            L.add("ticket_gates", e["x"], face_y - 1.0, 0.0, w=6.0, d=1.2, n=6)
            L.add("sign", e["x"], face_y - 0.3, 0.0, w=4.0, d=0.2, h=4.4, text="WAY IN",
                  colour="red", mount="facade")
    for k in range(rng.randint(2, 4)):
        L.add("board", -cw / 3 + k * cw / 4, face_y - 0.4, 0.0, w=2.5, d=0.2, h=3.2)
    L.add("ticket_machine", -cw / 2 + 2.0, face_y - 1.0, 0.0, w=0.8, d=0.6)
    L.add("ticket_machine", cw / 2 - 2.0, face_y - 1.0, 0.0, w=0.8, d=0.6)
    # Bus interchange: service road with bus bays and shelters, taxi rank.
    ry = -(B + rng.uniform(6.0, 9.0))
    _road_x(L, ry, -3 * B, 3 * B, 7.3)
    L.add("pavement", 0.0, -(B + (abs(ry) - B - 3.65) / 2), 0.0, w=6 * B,
          d=abs(ry) - B - 3.65, h=0.14)
    n_bus = rng.randint(2, 4)
    for k in range(n_bus):
        x = -B + k * (2 * B / max(1, n_bus - 1)) if n_bus > 1 else 0.0
        L.add("bus_shelter", x, ry + 3.65 + 2.2, 0.0, w=4.0, d=1.6, style="bus")
        L.add("sign", x + 2.8, ry + 3.65 + 1.0, 0.0, w=0.5, d=0.1, h=3.0, text="BUS", colour="red")
        L.add("yard_line", x, ry + 2.2, 0.0, w=12.0, d=0.15, h=0.005, colour="yellow")
    for k in range(rng.randint(3, 6)):
        L.add("car", -2.5 * B + k * 5.2, ry - 2.0, 0.0, w=4.6, d=1.85, paint=rng.choice([1, 1, 2]),
              taxi=True)
    _crossing(L, rng.uniform(-0.3, 0.3) * B, ry, 0.0, 7.3)
    # Side: cycle parking bank.
    side = rng.choice([-1, 1])
    _col(L, "cycle_rack", -0.6 * B, 0.6 * B, side * (B + 1.5), 2.5, yaw=90.0, w=2.2, d=0.8)
    _row(L, "bollards", -B, B, -(B + 0.5), 3.2, w=3.0, d=0.2)
    L.add("tactile", 0.0, face_y - 4.0, 0.0, w=cw, d=0.8, h=0.005)
    for s in (-1, 1):
        _col(L, "sign", -B * 0.5, B * 0.5, s * (B - 0.5), B, yaw=90.0, rng=rng, p=0.7,
             w=0.6, d=0.1, h=2.4, text="TOTEM", colour="navy")
    L.variant.update(shape=shape, bus_bays=n_bus)


# ---- perimeter -----------------------------------------------------------------

def _perimeter(L, rng, B, n_ent):
    # Warehouse(s) north and possibly east, loading docks facing the yard.
    shape = rng.choice(["rect", "L", "twin"])
    depth = rng.uniform(25, 40)
    face_y = B + rng.uniform(4.0, 10.0)
    blocks = _footprint(L, rng, shape, rng.uniform(-8, 8), face_y + depth / 2,
                        rng.uniform(1.6, 2.4) * B, depth, rng.uniform(8, 12), "warehouse",
                        wing=rng.uniform(15, 22))
    n_dock = rng.randint(3, 7)
    main = blocks[0] if blocks else None
    if main:
        x0 = main.x - main.w / 2 + 4.0
        for k in range(n_dock):
            x = x0 + k * 4.5
            if x > main.x + main.w / 2 - 3:
                break
            L.add("dock_door", x, main.y - main.d / 2 - 0.15, 0.0, w=3.2, d=0.3, h=4.0)
            L.add("yard_line", x, main.y - main.d / 2 - 8.0, 90.0, w=14.0, d=0.15, h=0.005,
                  colour="yellow")
            if rng.random() < 0.5:
                L.add("trailer", x, main.y - main.d / 2 - 8.5, 90.0, w=13.6, d=2.5, h=3.9,
                      paint=rng.randrange(7))
    # Containers stacked along one side.
    cs = rng.choice(["E", "W"])
    n_c = rng.randint(3, 8)
    y = -0.8 * B
    for k in range(n_c):
        # yaw 90: the container's length runs along y, so step by its length
        # ([X] 2026-09-26: a 3 m step overlapped 12 m boxes into one striped slab).
        ln = 6.1 if rng.random() < 0.5 else 12.2
        yc = y + ln / 2
        if yc + ln / 2 > 1.2 * B:
            break
        x = (B + 2.0) * (1 if cs == "E" else -1)
        L.add("container", x, yc, 90.0, w=ln, d=2.44, h=2.6, paint=rng.randrange(7))
        if ln == 6.1 and rng.random() < 0.35:
            L.add("container", x, yc, 90.0, w=6.1, d=2.44, h=5.2, paint=rng.randrange(7),
                  stacked=True)
        y += ln + rng.uniform(0.4, 1.5)
    L.add("skip", -(B + 1.0) * (1 if cs == "E" else -1), B - 3.0, 0.0, w=3.7, d=1.8)
    L.add("asset", -(B + 1.0) * (1 if cs == "E" else -1), B - 7.0, 0.0, w=1.8, d=1.2, h=1.4,
          asset="kenney_dumpster")
    _row(L, "pallets", -0.4 * B, 0.2 * B, B + 1.5, 1.6, rng=rng, p=0.6, w=1.2, d=1.0)
    # Boundary: palisade or mesh fence, or a wall, gaps at the gates.
    btype = rng.choice(["palisade", "mesh", "wall"])
    gaps = [(e["side"], e["along"]) for e in L.entrances]
    _boundary(L, "wall_run" if btype == "wall" else "fence_run", B + 6.0 + depth * 0.0, gaps, rng,
              2.6 if btype == "wall" else 2.4, style=btype)
    L.variant.update(shape=shape, docks=n_dock, boundary=btype, containers=n_c)
    # Gatehouse + barrier + access road at each gate.
    for e in L.entrances:
        nx, ny = math.cos(math.radians(e["yaw"])), math.sin(math.radians(e["yaw"]))
        tx, ty = -ny, nx
        gx, gy = e["x"], e["y"]
        L.add("gatehouse", gx + tx * 5.5 - nx * 2.5, gy + ty * 5.5 - ny * 2.5, 0.0, w=4.0, d=3.0)
        L.add("barrier", gx - tx * 3.0, gy - ty * 3.0, e["yaw"], w=0.4, d=0.4, arm=5.0)
        L.add("sign", gx + tx * 3.2 + nx * 1.0, gy + ty * 3.2 + ny * 1.0, e["yaw"] + 90,
              w=1.2, d=0.1, h=2.2, text="SECURITY", colour="yellow")
        for c in range(rng.randint(2, 4)):          # CC0 props from the asset index
            L.add("asset", gx - tx * (4.2 + 0.9 * c) - nx * 1.5, gy - ty * (4.2 + 0.9 * c) - ny * 1.5,
                  0.0, w=0.4, d=0.4, h=0.7, asset="traffic_cone")
        if e["side"] in "SN":
            _road_y(L, gx, min(gy, gy + ny * 30), max(gy, gy + ny * 30), 6.5, lanes=False)
        else:
            L.add("road", gx + nx * 15, gy, 0.0, w=30.0, d=6.5, h=0.01)
    ry = -(B + 6.0 + 12.0)
    _road_x(L, ry, -3 * B, 3 * B, 7.3)
    L.add("grass", 0.0, ry + 8.0, 0.0, w=6 * B, d=8.0, h=0.01)
    for sx in (-1, 1):
        for sy in (-1, 1):
            L.add("floodlight", sx * (B + 4.5), sy * (B + 4.5), 0.0, w=0.4, d=0.4, h=12.0)
    _row(L, "tree", -2.5 * B, 2.5 * B, ry - 6.0, 7.0, rng=rng, p=0.5, w=1.0, d=1.0, h=7.0)


# ---- retail --------------------------------------------------------------------

def _terrace(L, rng, x0, x1, y_face, north, depth=(10, 14), floors=(2, 4)):
    """Shopfront terrace along x with its front face at y_face; alleys between."""
    x = x0
    while x < x1:
        w = rng.uniform(5.0, 9.5)
        if rng.random() < 0.1:                 # alley
            x += rng.uniform(2.0, 3.5)
            continue
        d = rng.uniform(*depth)
        h = rng.randint(*floors) * 3.3 + rng.uniform(0, 1.0)
        yc = y_face + (d / 2 if north else -d / 2)
        b = L.add("block", x + w / 2, yc, 0.0, w=w, d=d, h=h, style="shopfront",
                  facade=rng.randrange(6))
        if b:
            fy = y_face + (-0.5 if north else 0.5)
            L.add("shop_sign", x + w / 2, y_face + (-0.1 if north else 0.1), 0.0, w=w * 0.85,
                  d=0.15, h=0.8, z=3.4, colour=rng.randrange(8))
            if rng.random() < 0.45:
                L.add("awning", x + w / 2, fy - (0.8 if north else -0.8), 0.0, w=w * 0.8,
                      d=1.6, h=0.3, z=2.8, colour=rng.randrange(8))
        x += w


def _retail(L, rng, B, n_ent):
    variant = rng.choice(["high_street", "pedestrianised", "junction"])
    L.variant["street"] = variant
    face_n = B + rng.uniform(1.0, 3.0)
    _terrace(L, rng, -3 * B, 3 * B, face_n, north=True)
    if variant == "pedestrianised":
        face_s = -(B + rng.uniform(1.0, 3.0))
        _terrace(L, rng, -3 * B, 3 * B, face_s, north=False)
        _row(L, "tree", -B, B, 0.0, rng.uniform(8, 14), rng=rng, p=0.8, w=1.0, d=1.0, h=6.0)
        _row(L, "bench", -B, B, 2.0, 7.0, rng=rng, p=0.6, w=1.8, d=0.5)
    else:
        rw = rng.choice([6.5, 7.3])
        ry = -(B + 2.5 + rw / 2)
        _road_x(L, ry, -3 * B, 3 * B, rw)
        L.add("pavement", 0.0, -(B + 1.25), 0.0, w=6 * B, d=2.5, h=0.14)
        _terrace(L, rng, -3 * B, 3 * B, ry - rw / 2 - 3.0, north=False)
        L.add("pavement", 0.0, ry - rw / 2 - 1.5, 0.0, w=6 * B, d=3.0, h=0.14)
        _crossing(L, rng.uniform(-0.5, 0.5) * B, ry, 0.0, rw)
        for k in range(-6, 7):
            if rng.random() < 0.6:
                L.add("car", k * 6.0 + rng.uniform(-0.5, 0.5), ry + rw / 2 - 1.1,
                      rng.choice([0.0, 180.0]), w=4.4, d=1.8, paint=rng.randrange(7))
        L.add("bus_shelter", rng.uniform(-0.6, 0.6) * B, -(B + 0.9), 0.0, w=4.0, d=1.4,
              style="bus")
        if variant == "junction":
            jx = rng.choice([-1, 1]) * (B + 4.0)
            _road_y(L, jx, ry, 3 * B, rw)
            _crossing(L, jx, ry + rw / 2 + 4.0, 90.0, rw)
            L.add("asset", jx - 4.5, ry + rw / 2 + 1.0, 0.0, w=0.5, d=0.5, h=3.5,
                  asset="traffic_light")
        _row(L, "tree", -1.5 * B, 1.5 * B, -(B + 0.4), rng.uniform(9, 15), rng=rng, p=0.7,
             w=1.0, d=1.0, h=6.0)
    _row(L, "bin", -B, B, B - 0.6, rng.uniform(8, 14), rng=rng, p=0.8, w=0.55, d=0.55)
    _row(L, "cycle_rack", -0.5 * B, 0.5 * B, B - 1.2, 6.0, rng=rng, p=0.5, w=2.2, d=0.8)
    L.add("phone_box", rng.uniform(-0.8, 0.8) * B, B - 1.0, 0.0, w=1.0, d=1.0)
    for _ in range(rng.randint(1, 3)):
        L.add("asset", rng.uniform(-B, B), B - 0.5, 0.0, w=0.8, d=0.8, h=0.8, asset="trash_bags")
    for _ in range(rng.randint(2, 5)):
        L.add("sign", rng.uniform(-B, B), B - 1.8, 0.0, w=0.6, d=0.4, h=1.0, text="A-BOARD",
              colour=rng.choice(["white", "black"]))


# ---- car park ------------------------------------------------------------------

def _car_park(L, rng, B, n_ent):
    angled = rng.random() < 0.35
    across = rng.random() < 0.5                    # rows along x or along y
    occupancy = rng.uniform(0.35, 0.9)
    L.variant.update(angled=angled, rows_along="y" if across else "x",
                     occupancy=round(occupancy, 2))
    bay_w, bay_d, aisle = 2.5, 5.0, 6.0
    pitch = 2 * bay_d + aisle
    n_rows = int((2 * B + 12) // pitch) + 1
    lo = -(n_rows - 1) * pitch / 2 + rng.uniform(-pitch / 2, pitch / 2)
    for r in range(n_rows):
        c = lo + r * pitch
        for side in (-1, 1):
            rc = c + side * bay_d / 2
            k = -B - 4.0 - rng.uniform(0.0, 4 * bay_w)
            while k <= B + 4.0:
                x, y = (k, rc) if not across else (rc, k)
                line_yaw = (0.0 if across else 90.0) + (30.0 * side if angled else 0.0)
                L.add("bay_line", x, y, line_yaw, w=bay_d, d=0.1, h=0.005)
                if rng.random() < occupancy:
                    cx, cy = (k + bay_w / 2, rc) if not across else (rc, k + bay_w / 2)
                    L.add("car", cx, cy, line_yaw + rng.choice([0, 180]),
                          w=4.5, d=1.85, paint=rng.randrange(7))
                k += bay_w
        # Planted island at each row end.
        if rng.random() < 0.7:
            for end in (-1, 1):
                x, y = (end * (B + 7.0), c) if not across else (c, end * (B + 7.0))
                L.add("grass", x, y, 0.0, w=3.0 if not across else 2 * bay_d,
                      d=2 * bay_d if not across else 3.0, h=0.15)
                L.add("tree", x, y, 0.0, w=1.0, d=1.0, h=rng.uniform(4, 7))
    # Pedestrian walkway hatched through the middle.
    L.add("hatch", 0.0, 0.0, 90.0 if not across else 0.0, w=2 * B, d=2.0, h=0.005)
    # Entry/exit barriers with ticket machines at the entrances.
    for e in L.entrances:
        nx, ny = math.cos(math.radians(e["yaw"])), math.sin(math.radians(e["yaw"]))
        tx, ty = -ny, nx
        L.add("barrier", e["x"] + tx * 3.5 - nx * 1.0, e["y"] + ty * 3.5 - ny * 1.0, e["yaw"] + 90,
              w=0.4, d=0.4, arm=4.0)
        L.add("ticket_machine", e["x"] + tx * 4.5 - nx * 1.5, e["y"] + ty * 4.5 - ny * 1.5,
              e["yaw"], w=0.8, d=0.6)
        L.add("sign", e["x"] - tx * 3.5, e["y"] - ty * 3.5, e["yaw"], w=1.0, d=0.1, h=2.5,
              text="P", colour="blue")
    for _ in range(rng.randint(1, 3)):
        L.add("ticket_machine", rng.uniform(-B, B), rng.choice([-1, 1]) * (B + 1.0), 0.0,
              w=0.8, d=0.6, label="PAY")
    for _ in range(rng.randint(0, 4)):
        L.add("asset", rng.uniform(-B, B), rng.uniform(-B, B), 0.0, w=0.4, d=0.4, h=0.7,
              asset="traffic_cone")
    btype = rng.choice(["wall", "hedge", "fence", "open"])
    if btype != "open":
        gaps = [(e["side"], e["along"]) for e in L.entrances]
        _boundary(L, {"wall": "wall_run", "hedge": "hedge", "fence": "fence_run"}[btype],
                  B + 13.0, gaps, rng, {"wall": 1.2, "hedge": 1.6, "fence": 1.8}[btype],
                  style="mesh")
    L.variant["boundary"] = btype
    # A neighbour: supermarket shed or multi-storey deck on one side.
    nb = rng.choice(["supermarket", "multistorey", "none"])
    if nb != "none":
        y = B + 16.0 + 12.0
        L.add("block", 0.0, y, 0.0, w=3 * B, d=24.0, h=7.5 if nb == "supermarket" else 12.0,
              style="warehouse" if nb == "supermarket" else "deck", facade=rng.randrange(6))
        if nb == "supermarket":
            _row(L, "trolley_bay", -B, B, B + 14.0, B, w=4.0, d=1.2)
    L.variant["neighbour"] = nb
    _row(L, "lamp", -B, B, 0.0 if not across else B + 10, 12.0, w=0.3, d=0.3, h=8.0)


# ---- plaza ---------------------------------------------------------------------

def _plaza(L, rng, B, n_ent):
    C = L.core_half
    # Buildings on three sides (U around the square), tall.
    for side in ("N", "E", "W"):
        off = B + rng.uniform(4.0, 8.0)
        x = -2 * B
        while x < 2 * B:
            w = rng.uniform(10, 20)
            h = rng.uniform(14, 30)
            d = rng.uniform(12, 18)
            px, py = _side_point(side, x + w / 2, off + d / 2)
            sw, sd = (w, d) if side in "NS" else (d, w)
            if rng.random() > 0.12:
                L.add("block", px, py, 0.0, w=sw, d=sd, h=h, style="mixed", facade=rng.randrange(6))
            x += w + (rng.uniform(4, 7) if rng.random() < 0.2 else 0.0)
    event = rng.choice(["concert", "market", "none"])
    L.variant["event"] = event
    centre = rng.choice(["fountain", "monument", "open"])
    L.variant["centrepiece"] = centre
    if centre != "open":
        L.add(centre, 0.0, 0.0, 0.0, w=6.0 if centre == "fountain" else 2.5,
              d=6.0 if centre == "fountain" else 2.5)
    if event == "concert":
        L.add("stage", 0.0, B + 2.0, 0.0, w=14.0, d=7.0, h=1.4)
        for s in (-1, 1):
            L.add("speaker_tower", s * 9.0, B + 1.0, 0.0, w=1.2, d=1.2, h=6.0)
        _row(L, "barrier_row", -8.0, 8.0, B - 3.0, 2.2, w=2.2, d=0.5)
        L.add("tent", -B + 3, -B + 3, 0.0, w=3.0, d=3.0, label="SEARCH")
    elif event == "market":
        for k in range(rng.randint(4, 8)):
            L.add("kiosk", -B + 3 + k * 4.2, B - 2.0, 0.0, w=3.0, d=2.5)
        for k in range(rng.randint(1, 3)):
            L.add("food_truck", -B + 6 + k * 9.0, -(B - 1.5), 0.0, w=6.0, d=2.3,
                  paint=rng.randrange(7))
    for _ in range(rng.randint(0, 4)):
        L.add("portable_toilet", B - 1.5, rng.uniform(-B, 0), 90.0, w=1.2, d=1.2)
    if event != "none":
        for e in L.entrances:
            nx, ny = math.cos(math.radians(e["yaw"])), math.sin(math.radians(e["yaw"]))
            for sgn in (-1, 1):
                L.add("asset", e["x"] - ny * sgn * 3.5 - nx * 1.0, e["y"] + nx * sgn * 3.5 - ny * 1.0,
                      e["yaw"] + 90, w=1.5, d=0.5, h=1.0, asset="kenney_construction_barrier")
        gaps = [(e["side"], e["along"]) for e in L.entrances]
        _boundary(L, "fence_run", B + 1.0, gaps, rng, 2.0, sides="SEW", style="heras")
    ring = rng.choice([0.55, 0.75, 0.9])
    n_t = rng.choice([6, 10, 16])
    for k in range(n_t):
        a = 2 * math.pi * k / n_t
        L.add("tree", ring * C * math.cos(a) * 1.3, ring * C * math.sin(a) * 1.3, 0.0,
              w=1.0, d=1.0, h=rng.uniform(5, 8))
        if rng.random() < 0.5:
            L.add("bench", ring * C * math.cos(a) * 1.15, ring * C * math.sin(a) * 1.15,
                  math.degrees(a) + 90, w=1.8, d=0.5)
    L.variant["trees"] = n_t
    ry = -(B + 6.0)
    _road_x(L, ry, -3 * B, 3 * B, 7.3)
    L.add("pavement", 0.0, -(B + 2.8), 0.0, w=6 * B, d=5.6, h=0.14)
    _crossing(L, 0.0, ry, 0.0, 7.3)


def _core_fill(L, rng, B):
    """Site structure *inside* the walk core, so the camera's foreground is the
    site and not an empty square (a people run drops what its walkers cross)."""
    C, t = L.core_half, L.site_type
    if t == "office_building":
        for s in (-1, 1):
            if rng.random() < 0.8:
                L.add("grass", s * 0.5 * C, rng.uniform(-0.3, 0.3) * C, 0.0, w=0.55 * C,
                      d=rng.uniform(0.3, 0.6) * C, h=0.12)
                L.add("tree", s * 0.5 * C, 0.0, 0.0, w=1.0, d=1.0, h=rng.uniform(5, 7))
        L.add("pavement", 0.0, 0.0, 90.0, w=2 * C, d=3.0, h=0.02, material="granite")
    elif t == "transit_station":
        L.add("tactile", 0.0, 0.3 * C, 0.0, w=1.6 * C, d=0.8, h=0.005)
        for k in range(rng.randint(1, 3)):
            L.add("bench", rng.uniform(-0.6, 0.6) * C, rng.uniform(-0.2, 0.5) * C,
                  rng.choice([0.0, 90.0]), w=1.8, d=0.5)
        L.add("sign", rng.uniform(-0.5, 0.5) * C, 0.55 * C, 0.0, w=0.6, d=0.1, h=2.4,
              text="TOTEM", colour="navy")
    elif t == "perimeter_site":
        yaw = rng.choice([0.0, 90.0])
        for k in range(rng.randint(2, 4)):
            u = -0.6 * C + k * 4.0
            x, y = (u, -0.2 * C) if yaw == 0.0 else (-0.2 * C, u)
            L.add("yard_line", x, y, yaw + 90.0, w=16.0, d=0.15, h=0.005, colour="yellow")
            if rng.random() < 0.4:
                L.add("trailer", x, y, yaw + 90.0, w=13.6, d=2.5, h=3.9, paint=rng.randrange(7))
        for _ in range(rng.randint(1, 3)):
            L.add("pallets", rng.uniform(-C, C) * 0.7, rng.uniform(-C, C) * 0.7, 0.0, w=1.2, d=1.0)
    elif t == "event_plaza":
        for k in range(-2, 3):          # granite banding in the paving
            L.add("pavement", k * C * 0.4, 0.0, 90.0, w=2 * C, d=0.6, h=0.02, material="granite")
    elif t == "retail_street":
        for _ in range(rng.randint(1, 3)):
            L.add("planter", rng.uniform(-0.7, 0.7) * C, rng.uniform(-0.3, 0.5) * C, 0.0,
                  w=1.6, d=1.6)
        L.add("pavement", 0.0, 0.1 * C, 0.0, w=2 * C, d=1.2, h=0.02, material="granite")


def _backdrop(L, rng, B):
    """What the camera sees over the site: city blocks, sheds or a tree line."""
    if L.site_type in _v2.BACKDROP:
        kind, hmin, hmax, n, style = _v2.BACKDROP[L.site_type]
        for i in range(n):
            a = 2 * math.pi * (i + rng.uniform(-0.3, 0.3)) / n
            r = rng.uniform(3.2, 5.0) * B + 30.0
            x, y = r * math.cos(a), r * math.sin(a)
            if kind == "tree":
                L.add("tree", x, y, 0.0, w=1.0, d=1.0, h=rng.uniform(hmin, hmax), backdrop=True)
            else:
                L.add("block", x, y, 0.0, w=rng.uniform(15, 30), d=rng.uniform(12, 25),
                      h=rng.uniform(hmin, hmax), style=style, facade=rng.randrange(6), backdrop=True)
        return
    kind, hmin, hmax, n = {
        "office_building": ("block", 25, 70, 22), "transit_station": ("block", 12, 30, 20),
        "perimeter_site": ("tree", 7, 12, 40), "retail_street": ("block", 9, 16, 26),
        "car_park": ("block", 6, 12, 14), "event_plaza": ("block", 18, 45, 24)}[L.site_type]
    for i in range(n):
        a = 2 * math.pi * (i + rng.uniform(-0.3, 0.3)) / n
        r = rng.uniform(3.2, 5.0) * B + 30.0
        x, y = r * math.cos(a), r * math.sin(a)
        if kind == "tree":
            L.add("tree", x, y, 0.0, w=1.0, d=1.0, h=rng.uniform(hmin, hmax), backdrop=True)
        else:
            L.add("block", x, y, 0.0, w=rng.uniform(15, 30), d=rng.uniform(12, 25),
                  h=rng.uniform(hmin, hmax), style="mixed" if L.site_type != "office_building"
                  else rng.choice(["glass", "office"]), facade=rng.randrange(6), backdrop=True)
    if L.site_type == "perimeter_site":       # a distant industrial estate beyond the trees
        for i in range(6):
            a = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(7, 9) * B + 40
            L.add("block", r * math.cos(a), r * math.sin(a), 0.0, w=40, d=25, h=rng.uniform(8, 14),
                  style="warehouse", facade=rng.randrange(6), backdrop=True)


_BUILDERS = {"office_building": _office, "transit_station": _transit,
             "perimeter_site": _perimeter, "retail_street": _retail,
             "car_park": _car_park, "event_plaza": _plaza}
# Sides where entrances go by default (canonical frame), per type.
_ENTRANCE_SIDES = {"office_building": "NSE", "transit_station": "NS",
                   "perimeter_site": "SE", "retail_street": "EWN",
                   "car_park": "SW", "event_plaza": "SEW"}
_ENTRANCE_KIND = {"office_building": "door", "transit_station": "ticket_gates",
                  "perimeter_site": "vehicle_gate", "retail_street": "street_end",
                  "car_park": "barrier", "event_plaza": "event_gate"}
_GROUND = {"office_building": [("granite", "setts"), ("light_slabs", "granite"), ("setts", "slabs")],
           "transit_station": [("concrete", "slabs"), ("slabs", "granite"), ("light_slabs", "concrete")],
           "perimeter_site": [("concrete", "gravel"), ("gravel", "concrete"), ("asphalt", "gravel")],
           "retail_street": [("slabs", "setts"), ("setts", "granite"), ("light_slabs", "slabs")],
           "car_park": [("asphalt", "concrete"), ("asphalt", "slabs")],
           "event_plaza": [("light_slabs", "setts"), ("granite", "light_slabs"), ("setts", "granite")]}
_WEAR = {"office_building": (3, 8, ["dirt", "gum", "leaves"]),
         "transit_station": (10, 24, ["gum", "dirt", "crack"]),
         "perimeter_site": (14, 30, ["oil", "dirt", "tyre", "crack"]),
         "retail_street": (10, 24, ["gum", "dirt", "crack", "leaves"]),
         "car_park": (16, 34, ["oil", "oil", "tyre", "crack"]),
         "event_plaza": (6, 16, ["gum", "dirt", "leaves"])}


from . import layout_sites_v2 as _v2

_BUILDERS.update(_v2.BUILDERS)
_ENTRANCE_SIDES.update(_v2.ENTRANCE_SIDES)
_ENTRANCE_KIND.update(_v2.ENTRANCE_KIND)
_GROUND.update(_v2.GROUND)
_WEAR.update(_v2.WEAR)


def _rotate_layout(L: Layout, k: int) -> None:
    for it in L.items:
        it.x, it.y = _rot(it.x, it.y, k)
        it.yaw = (it.yaw + 90.0 * k) % 360.0
    for e in L.entrances:
        e["x"], e["y"] = (round(v, 2) for v in _rot(e["x"], e["y"], k))
        e["yaw"] = (e["yaw"] + 90.0 * k) % 360.0
        e["side"] = _rot_side(e["side"], k)


def _rot_side(side: str, k: int) -> str:
    order = "ENWS"                      # counter-clockwise
    return order[(order.index(side) + k) % 4]


def _canonical_side(x: float, y: float) -> str:
    if abs(x) >= abs(y):
        return "E" if x > 0 else "W"
    return "N" if y > 0 else "S"


def build_layout(site_type: str, seed: int, core_half: float | None = None,
                 entrances: list[tuple[float, float]] | int | None = None,
                 n_cameras: int | None = None, wet: float | None = None,
                 time: dict | None = None, camera_res: tuple[int, int] = (1280, 720)) -> Layout:
    """A full site layout for (site_type, seed).

    `entrances`: scenario entrance points (metres, survsim frame) — the
    layout puts its gates/doors *there*; or a count; or None (seeded 1-3).
    """
    if site_type not in SITE_TYPES:
        raise ValueError(f"unknown site type {site_type!r}")
    rng = random.Random(f"layout:{site_type}:{seed}")
    # Layout-only scenes use a compact core (a real forecourt/yard is 15-30 m
    # across); a people run passes its walk envelope instead.
    C = float(core_half if core_half is not None else rng.uniform(9.0, 14.0))
    B = C + rng.uniform(1.5, 4.0)
    k = rng.randrange(4)
    L = Layout(site_type, seed, round(C, 2), k, {})

    # Entrances: canonical frame positions on the boundary square.
    if isinstance(entrances, list) and entrances:
        pts = []
        for ex, ey in entrances:
            cx, cy = _rot(ex, ey, -k)             # into the canonical frame
            side = _canonical_side(cx, cy)
            along = cy if side in "EW" else cx
            pts.append((side, max(-0.85 * B, min(0.85 * B, along))))
        L.variant["entrances_from"] = "scenario"
    else:
        n = entrances if isinstance(entrances, int) else rng.randint(1, 3)
        pts = _entrance_positions(n, B, rng, _ENTRANCE_SIDES[site_type])
        L.variant["entrances_from"] = "seed"
    for i, (side, along) in enumerate(pts):
        x, y = _side_point(side, along, B)
        L.entrances.append({"id": f"entrance_{i}", "x": round(x, 2), "y": round(y, 2),
                            "side": side, "along": round(along, 2), "yaw": _yaw_out(side),
                            "kind": _ENTRANCE_KIND[site_type]})

    _BUILDERS[site_type](L, rng, B, len(pts))
    _core_fill(L, rng, B)
    _backdrop(L, rng, B)

    # Lamp columns round the rim (night scenes need them); the car park has its own.
    if site_type != "car_park":
        step = rng.choice([10.0, 12.0, 15.0])
        for side in "NSEW":
            a = -B + rng.uniform(0.5, step)
            while a < B:
                x, y = _side_point(side, a, B - 0.6)
                L.add("lamp", x, y, _yaw_out(side) + 180.0, w=0.3, d=0.3, h=6.0)
                a += step
    lo, hi, kinds = _WEAR[site_type]
    L.time = dict(time) if time else pick_time(rng)
    season = L.time.get("season", "summer")
    wet_p = {"winter": 0.6, "autumn": 0.5, "spring": 0.35, "summer": 0.2}[season]
    # Damp, not flooded: at >0.6 the procedural ground turns into a mirror of
    # the sky and trips the horizon check ([X] 2026-09-26 event_plaza s3).
    wetness = float(wet) if wet is not None else (round(rng.uniform(0.3, 0.55), 2)
                                                  if rng.random() < wet_p else 0.0)
    L.weather = {"state": "wet" if wetness > 0 else "dry", "wet": wetness, "fog": 0.0}
    dk = kinds + (["puddle"] * 3 if wetness else []) + (["leaves"] * 2 if season == "autumn" else [])
    _decals(L, rng, B + 4.0, rng.randint(lo, hi), dk)
    # Decals only on open ground: one straddling a solid smears up its sides.
    solids = [i for i in L.items if i.solid and i.h >= 0.3]
    L.items = [i for i in L.items if i.kind != "decal" or not any(
        o.contains(i.x, i.y, pad=max(i.w, i.d) / 2 + 0.3) for o in solids)]
    main, rim = rng.choice(_GROUND[site_type])
    L.ground = {"main": main, "rim": rim,
                "far": "grass" if site_type == "perimeter_site" or site_type in _v2.FAR_GRASS else "asphalt"}
    _rotate_layout(L, k)
    L.cameras = place_installer_cameras(L, rng, n_cameras or rng.randint(3, 5), camera_res)
    L.report = gate(L)
    return L


# ------------------------------------------------------------------- cameras

# Site structure a camera must be sited around; everything else solid (cars,
# trees, benches ...) is moved out of a chosen camera's near view instead.
STRUCTURAL = {"block", "gatehouse", "container", "trailer", "wall_run", "fence_run", "hedge",
              "canopy", "stage", "bus_shelter", "food_truck", "kiosk", "dock_door",
              "speaker_tower", "floodlight", "bin_store"}


def _structural(it: Item) -> bool:
    """Low boundary runs (< 1.3 m: hedges, dwarf walls) are trimmed, not designed round."""
    if it.kind in ("wall_run", "fence_run", "hedge") and it.h < 1.3:
        return False
    if it.props.get("style") == "heras" or it.kind in ("kiosk", "food_truck", "trailer"):
        return False                  # temporary kit and vehicles are moved, not designed round
    return it.kind in STRUCTURAL


def _near_frustum_hits(L: Layout, cx, cy, yaw, near=10.0, half_fov=60.0,
                       only=None, items=False, pad=0.5, skip=None) -> list:
    """Solids in the first `near` m of view (+-half_fov). Footprints are padded
    (`pad`, and trees by their crown) because the built meshes overhang them:
    roofs by 15 cm, tree crowns by up to 1.5 m ([X] 2026-09-26, both failed the
    editor's frustum gate before the padding)."""
    fx, fy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cosl = math.cos(math.radians(half_fov))
    out = []
    for it in L.items:
        if not it.solid or it.h < 0.3 or (only is not None and it.kind not in only):
            continue
        if it is skip or (skip is None and it.kind in ("block", "gatehouse")
                          and it.contains(cx, cy, pad=0.5)):
            continue                          # the wall the camera is fixed to
        if math.hypot(it.x - cx, it.y - cy) - math.hypot(it.w, it.d) / 2 - it.h > near:
            continue
        big = Item(it.kind, it.x, it.y, it.yaw,
                   (max(it.w, 0.6 * it.h) if it.kind == "tree" else it.w) + 2 * pad,
                   (max(it.d, 0.6 * it.h) if it.kind == "tree" else it.d) + 2 * pad, it.h)
        for px, py in big.samples(6):
            dx, dy = px - cx, py - cy
            d = math.hypot(dx, dy)
            if d < 1e-6 or (d < near and (dx * fx + dy * fy) / d > cosl):
                out.append(it if items else it.kind)
                break
    return out


# Opaque things a sight line can hit, with the height band (m) they occupy.
# Trees block only through the crown; fences are modelled as opaque panels.
_OCC_BAND = {"tree": (0.55, 1.05), "hedge": (0.0, 1.0), "fence_run": (0.0, 1.0),
             "bus_shelter": (0.0, 1.0), "canopy": (0.93, 1.0)}


def _occluded(L: Layout, cam, pt, skip: Item | None = None, target_z: float = 1.5) -> bool:
    """Sight line camera (x, y, z) -> (pt, target_z) blocked by any solid? (2.5-D)

    For each footprint the line crosses, the line's height inside it is
    compared with the item's height band — the same question the editor's
    line trace answers against the built meshes.
    """
    cx, cy = cam[0], cam[1]
    cz = cam[2] if len(cam) > 2 else 5.0
    for it in L.items:
        if it is skip or not it.solid or it.kind in ("lamp", "camera_pole", "flagpole", "sign",
                                                    "floodlight", "shop_sign", "awning",
                                                    "bollards", "cycle_rack", "decal"):
            continue
        if it.contains(cx, cy, pad=0.05):
            continue                          # the wall a camera is mounted on
        pad = 0.6 if it.kind == "tree" else 0.15
        w, d = (max(it.w, 0.55 * it.h), max(it.d, 0.55 * it.h)) if it.kind == "tree" else (it.w, it.d)
        probe = Item(it.kind, it.x, it.y, it.yaw, w, d, it.h)
        iv = _seg_interval(cx, cy, pt[0], pt[1], probe, pad)
        if iv is None:
            continue
        lo_f, hi_f = _OCC_BAND.get(it.kind, (0.0, 1.0))
        bot, top = it.h * lo_f, it.h * hi_f
        z0, z1 = cz + (target_z - cz) * iv[0], cz + (target_z - cz) * iv[1]
        if min(z0, z1) < top and max(z0, z1) > bot:
            return True
    return False


def _mount_candidates(L: Layout, rng) -> list[dict]:
    """Where an installer can fix a camera: facades (3-6 m) and poles.

    Facade mounts sit 0.4 m proud of a block face that looks toward the core;
    poles are dedicated 5 m columns on the boundary rim (plus lamp columns).
    """
    C = L.core_half
    out = []
    for it in L.items:
        if it.kind in ("block", "gatehouse") and it.h >= 3.0 and not it.props.get("backdrop"):
            # world half-extents: blocks are rotated by multiples of 90 degrees
            # ([X] 2026-09-26: using w/d unrotated put cameras inside yaw-90 blocks)
            q = round((it.yaw % 180.0) / 90.0) % 2
            hx, hy = (it.d / 2, it.w / 2) if q else (it.w / 2, it.d / 2)
            for (nx, ny) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                fx = it.x + nx * (hx + 0.4)
                fy = it.y + ny * (hy + 0.4)
                # face must look toward the core
                if (-fx * nx - fy * ny) <= 0:
                    continue
                span = 2 * hx if nx == 0 else 2 * hy
                for t in (-0.35, 0.0, 0.35):
                    px = fx + (t * span if nx == 0 else 0.0)
                    py = fy + (t * span if ny == 0 else 0.0)
                    if max(abs(px), abs(py)) > 2.6 * C:
                        continue
                    z = min(it.h - 0.5, rng.uniform(3.5, 6.0))
                    if z < 3.0:
                        continue
                    out.append({"x": px, "y": py, "z": round(z, 2), "mount": "wall",
                                "normal": math.degrees(math.atan2(ny, nx)), "on": it})
    B = C + 2.0
    for side in "NSEW":
        for a in (-0.7, -0.35, 0.0, 0.35, 0.7):
            x, y = _side_point(side, a * B, B + 0.8)
            out.append({"x": x, "y": y, "z": round(rng.uniform(4.5, 6.0), 2), "mount": "pole",
                        "normal": _yaw_out(side) + 180.0, "on": None})
    return out


def _cam_model(cid, x, y, z, yaw, pitch, hfov, res):
    from ..core.models import Camera, Mount
    return Camera(id=cid, type="fixed_bullet",
                  mount=Mount(pos=(float(x), float(y), float(z)), yaw=float(yaw), pitch=float(pitch)),
                  optics={"hfov_deg": float(hfov)}, capture={"res": list(res), "fps": 10.0})


def _aim(m, tx, ty, max_off):
    """Yaw toward a target, clamped to +-max_off of the mount's outward normal."""
    yaw = math.degrees(math.atan2(ty - m["y"], tx - m["x"]))
    off = (yaw - m["normal"] + 180.0) % 360.0 - 180.0
    off = max(-max_off, min(max_off, off))
    return m["normal"] + off


def place_installer_cameras(L: Layout, rng, n: int, res=(1280, 720)) -> list[dict]:
    """Cameras as an installer places them: one per entrance first, then coverage.

    Each camera is on a wall (<= 25 deg off the facade normal, so the facade
    stays out of the near view) or a pole, 3-6 m up, pitched so its aim point
    sits in the lower-middle of the frame. Coverage uses
    `geometry.pixel_heights` (>= 20 px: the one visibility threshold) on the
    core grid; entrances use `usably_visible` plus a solid-occlusion test.
    """
    from ..core.geometry import MIN_USABLE_PX_HEIGHT, pixel_heights, usably_visible

    C = L.core_half
    mounts = _mount_candidates(L, rng)
    g = np.arange(-C, C + 0.1, 2.0)
    gx, gy = np.meshgrid(g, g)
    ground = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    covered = np.zeros(len(ground), bool)
    chosen: list[dict] = []
    used = set()

    def candidates(tx, ty):
        for mi, m in enumerate(mounts):
            if mi in used:
                continue
            d = math.hypot(tx - m["x"], ty - m["y"])
            if not 8.0 <= d <= 38.0:
                continue
            yaw = _aim(m, tx, ty, 25.0 if m["mount"] == "wall" else 70.0)
            for dyaw in (0.0, -12.0, 12.0):
                y2 = yaw + dyaw
                if m["mount"] == "wall" and abs((y2 - m["normal"] + 180) % 360 - 180) > 25:
                    continue
                if [it for it in _near_frustum_hits(L, m["x"], m["y"], y2, only=STRUCTURAL,
                                                    items=True, skip=m.get("on"))
                        if _structural(it)]:
                    continue
                pitch = -max(12.0, min(40.0, math.degrees(math.atan2(m["z"] - 0.5, d)) + 4.0))
                hfov = 70.0 if d > 22 else 85.0
                yield mi, m, y2, pitch, hfov

    def vis_cells(cam, m):
        px = pixel_heights(cam, ground, 1.7)
        from ..core.geometry import in_view
        ok = in_view(cam, ground) & (px >= MIN_USABLE_PX_HEIGHT)
        return ok       # occlusion is applied once, to the chosen rig (report)

    def ent_seen(cam, m, e):
        p = np.array([e["x"], e["y"], 0.0])
        return usably_visible(cam, p) and not _occluded(L, (m["x"], m["y"], m["z"]),
                                                       (e["x"], e["y"]), m.get("on"))

    # 1) entrances, in order: the best mount that sees it and most of the core.
    for e in L.entrances:
        if any(ent_seen(c["_model"], c["_mount"], e) for c in chosen):
            continue
        best = None
        nx, ny = math.cos(math.radians(e["yaw"])), math.sin(math.radians(e["yaw"]))
        for mi, m, yaw, pitch, hfov in list(candidates(e["x"] * 0.6, e["y"] * 0.6)) + list(
                candidates(e["x"] - nx * 4.0, e["y"] - ny * 4.0)):
            cam = _cam_model("x", m["x"], m["y"], m["z"], yaw, pitch, hfov, res)
            if not ent_seen(cam, m, e):
                continue
            v = vis_cells(cam, m)
            score = (v & ~covered).sum() + (5 if m["mount"] == "wall" else 0)
            if best is None or score > best[0]:
                best = (score, mi, m, yaw, pitch, hfov, v)
        if best is None:
            continue
        _, mi, m, yaw, pitch, hfov, v = best
        used.add(mi)
        covered |= v
        chosen.append({"_model": _cam_model(f"cam_{len(chosen):02d}", m["x"], m["y"], m["z"], yaw,
                                            pitch, hfov, res), "_mount": m, "target": e["id"]})
    # 2) coverage of the core, greedily.
    while len(chosen) < n:
        best = None
        tx, ty = rng.uniform(-0.4, 0.4) * C, rng.uniform(-0.4, 0.4) * C
        for mi, m, yaw, pitch, hfov in candidates(tx, ty):
            cam = _cam_model("x", m["x"], m["y"], m["z"], yaw, pitch, hfov, res)
            v = vis_cells(cam, m)
            gain = (v & ~covered).sum()
            if best is None or gain > best[0]:
                best = (gain, mi, m, yaw, pitch, hfov, v)
        if best is None or best[0] <= 0:
            break
        _, mi, m, yaw, pitch, hfov, v = best
        used.add(mi)
        covered |= v
        chosen.append({"_model": _cam_model(f"cam_{len(chosen):02d}", m["x"], m["y"], m["z"], yaw,
                                            pitch, hfov, res), "_mount": m, "target": "core"})
    out = []
    for c in chosen:
        cam, m = c["_model"], c["_mount"]
        out.append({"id": cam.id, "x": round(m["x"], 2), "y": round(m["y"], 2), "z": m["z"],
                    "yaw": round(cam.mount.yaw, 1), "pitch": round(cam.mount.pitch, 1),
                    "hfov_deg": cam.hfov_deg, "mount": m["mount"], "target": c["target"],
                    "res": list(res)})
    # Furniture, cars and trees yield to the cameras (installers trim trees and
    # move bins; the editor's frustum gate would fail the build otherwise).
    for c in chosen:
        m = c["_mount"]
        for it in _near_frustum_hits(L, m["x"], m["y"], c["_model"].mount.yaw, items=True):
            if it in L.items and not _structural(it):
                L.items.remove(it)
                L.variant["pruned_for_cameras"] = L.variant.get("pruned_for_cameras", 0) + 1
    # Coverage of the chosen rig with solid occlusion (a 4 m grid keeps it cheap).
    g4 = np.arange(-C, C + 0.1, 4.0)
    cells = [(x, y) for x in g4 for y in g4]
    seen = 0
    for x, y in cells:
        for c in chosen:
            cam, m = c["_model"], c["_mount"]
            if usably_visible(cam, np.array([x, y, 0.0])) and \
                    not _occluded(L, (m["x"], m["y"], m["z"]), (x, y), m.get("on")):
                seen += 1
                break
    L.variant["core_coverage"] = round(seen / max(1, len(cells)), 3)
    return out


def camera_models(L: Layout):
    return [_cam_model(c["id"], c["x"], c["y"], c["z"], c["yaw"], c["pitch"], c["hfov_deg"],
                       tuple(c["res"])) for c in L.cameras]


# ----------------------------------------------------------------------- gate

def occupancy(L: Layout, cell: float = 0.5, extent: float | None = None):
    """Boolean grid of cells a pedestrian cannot stand in (solid footprints)."""
    ext = extent or (L.core_half + 12.0)
    n = int(2 * ext / cell) + 1
    occ = np.zeros((n, n), bool)
    xs = -ext + np.arange(n) * cell
    for it in L.items:
        if not it.solid or it.kind in ("lamp", "camera_pole", "flagpole", "sign", "tree",
                                       "floodlight", "canopy", "shop_sign", "awning"):
            continue        # thin columns: walk round them at this cell size
        r = math.hypot(it.w, it.d) / 2 + cell
        i0, i1 = np.searchsorted(xs, [it.x - r, it.x + r])
        j0, j1 = np.searchsorted(xs, [it.y - r, it.y + r])
        for i in range(max(0, i0), min(n, i1 + 1)):
            for j in range(max(0, j0), min(n, j1 + 1)):
                if it.contains(xs[i], xs[j], pad=0.2):
                    occ[j, i] = True
    return occ, xs


def reachable(L: Layout, cell: float = 0.5) -> dict[str, bool]:
    occ, xs = occupancy(L, cell)
    n = len(xs)

    def idx(v):
        return round((v + xs[-1]) / cell)

    start = (idx(0.0), idx(0.0))
    seen = np.zeros_like(occ)
    if occ[start[1], start[0]]:
        # the centre may hold a fountain/monument: start from the nearest free cell
        free = np.argwhere(~occ)
        d = (free[:, 0] - start[1]) ** 2 + (free[:, 1] - start[0]) ** 2
        j, i = free[int(np.argmin(d))]
        start = (int(i), int(j))
    q = deque([start])
    seen[start[1], start[0]] = True
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < n and 0 <= b < n and not seen[b, a] and not occ[b, a]:
                seen[b, a] = True
                q.append((a, b))
    out = {}
    for e in L.entrances:
        i, j = idx(e["x"]), idx(e["y"])
        ok = 0 <= i < n and 0 <= j < n and bool(seen[j, i])
        out[e["id"]] = ok
    return out


def entrance_visibility(L: Layout, cameras=None) -> dict[str, list[str]]:
    from ..core.geometry import usably_visible
    cams = cameras if cameras is not None else camera_models(L)
    out = {}
    for e in L.entrances:
        p = np.array([e["x"], e["y"], 0.0])
        seen = []
        for cam in cams:
            mx, my = cam.mount.pos[0], cam.mount.pos[1]
            if usably_visible(cam, p) and not _occluded(L, cam.mount.pos, (e["x"], e["y"]),
                                                          _mount_item(L, mx, my)):
                seen.append(cam.id)
        out[e["id"]] = seen
    return out


def _mount_item(L, x, y):
    for it in L.items:
        if it.kind in ("block", "gatehouse") and it.contains(x, y, pad=0.5):
            return it
    return None


def gate(L: Layout) -> dict:
    reach = reachable(L)
    vis = entrance_visibility(L)
    frustum = {c["id"]: _near_frustum_hits(L, c["x"], c["y"], c["yaw"]) for c in L.cameras}
    fails = ([f"{k} unreachable" for k, v in reach.items() if not v]
             + [f"{k} seen by no camera" for k, v in vis.items() if not v]
             + [f"{k}: {v[:3]} in near view" for k, v in frustum.items() if v])
    return {"reachable": reach, "entrance_cameras": vis,
            "near_view_violations": {k: v for k, v in frustum.items() if v},
            "ok": not fails, "failures": fails}


# ------------------------------------------------------------------- features

FEATURE_KINDS = ("block", "road", "kerb", "crossing", "bay_line", "car", "tree", "lamp",
                 "bus_shelter", "cycle_rack", "container", "fence_run", "wall_run", "hedge",
                 "ticket_gates", "canopy", "dock_door", "trailer", "shop_sign", "awning",
                 "stage", "kiosk", "barrier", "ticket_machine", "gatehouse", "floodlight",
                 "sign", "bench", "bin", "planter", "grass", "decal", "pallets", "skip",
                 "barrier_row", "food_truck", "fountain", "monument", "board", "flagpole",
                 "bin_store", "trolley_bay", "tactile", "hatch", "yard_line", "pavement",
                 "asset", "portable_toilet", "speaker_tower", "phone_box", "tent")


def features(L: Layout | dict) -> np.ndarray:
    """A fixed-length numeric description of a layout for the separability test."""
    d = L.to_dict() if isinstance(L, Layout) else L
    items = d["items"]
    n = max(1, len(items))
    counts = {k: 0 for k in FEATURE_KINDS}
    for it in items:
        if it["kind"] in counts:
            counts[it["kind"]] += 1
    blocks = [it for it in items if it["kind"] == "block"]
    area = sum(b["w"] * b["d"] for b in blocks)
    h = [b["h"] for b in blocks] or [0.0]
    road = sum(it["w"] for it in items if it["kind"] == "road")
    vec = [math.log1p(counts[k]) for k in FEATURE_KINDS] + [
        math.log1p(area) / 10.0, float(np.mean(h)) / 30.0, float(np.max(h)) / 30.0,
        math.log1p(road) / 10.0, len(blocks) / n]
    return np.array(vec, float)

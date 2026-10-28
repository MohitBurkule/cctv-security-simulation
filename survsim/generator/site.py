"""Procedural site layout.

Turns a `SiteSpec` into an `Environment`: zones, entrances, exits, camera mount
candidates and props. Each site type has a different topology, because the
analytics problem is different in each — a transit station is a high-flow funnel
through a gateline, a perimeter site is a long thin boundary with one legitimate
crossing, a car park is mostly occlusion.

Layouts are deterministic given the seed so a spec always reproduces the same
site, and every site is guaranteed to contain at least the named zones the
narrative generator relies on.
"""

from __future__ import annotations

import numpy as np

from ..core.models import Environment, Prop, Waypoint, Zone
from .spec import SiteSpec


def _rect(cx: float, cy: float, w: float, h: float) -> list[tuple[float, float]]:
    return [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
            (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]


def _perimeter_points(w: float, h: float, n: int, inset: float = 0.0) -> list[tuple[float, float]]:
    """`n` points spaced evenly around the footprint boundary."""
    half_w, half_h = w / 2 - inset, h / 2 - inset
    per = 2 * (w + h)
    out = []
    for i in range(n):
        d = (i + 0.5) / n * per
        if d < w:
            out.append((-half_w + d, -half_h))
        elif d < w + h:
            out.append((half_w, -half_h + (d - w)))
        elif d < 2 * w + h:
            out.append((half_w - (d - w - h), half_h))
        else:
            out.append((-half_w, half_h - (d - 2 * w - h)))
    return out


_W = ["worker", "guard"]
# Per v2 type: (id, tag, cx, cy, w, h as fractions of the footprint, allowed roles).
# Each has a "queue" zone -- where innocent waiting happens (the loiter hard
# negative) -- and a restricted zone the intrusion narratives cross.
_V2_ZONES = {
    "loading_dock": [("dock_apron", "queue", 0.0, 0.25, 0.6, 0.18, None),
                     ("dock_edge", "restricted", 0.0, 0.40, 0.7, 0.10, _W),
                     ("yard", "public", 0.0, -0.1, 0.8, 0.4, None)],
    "school_gate": [("pickup", "queue", 0.0, 0.25, 0.5, 0.15, None),
                    ("school_grounds", "restricted", 0.0, 0.40, 0.6, 0.12, _W),
                    ("pavement", "public", 0.0, -0.1, 0.9, 0.3, None)],
    "rail_platform": [("platform", "queue", 0.0, 0.0, 0.8, 0.35, None),
                      ("track", "restricted", 0.0, -0.40, 0.9, 0.12, _W),
                      ("concourse", "public", 0.0, 0.3, 0.5, 0.2, None)],
    "hospital_entrance": [("smoking", "queue", -0.3, 0.0, 0.15, 0.2, None),
                          ("dropoff", "gate", 0.0, -0.3, 0.6, 0.12, None),
                          ("ambulance_bay", "restricted", 0.35, 0.25, 0.2, 0.2, _W),
                          ("forecourt", "public", 0.0, 0.05, 0.6, 0.35, None)],
    "utility_yard": [("compound", "restricted", 0.0, 0.275, 0.6, 0.25, _W),
                     ("apron", "public", 0.0, -0.1, 0.8, 0.45, None),
                     ("layby", "queue", -0.3, -0.35, 0.2, 0.12, None)],
    "construction_site": [("works", "restricted", 0.0, 0.3, 0.6, 0.25, _W),
                          ("site_gate", "queue", -0.3, -0.3, 0.2, 0.15, None),
                          ("footpath", "public", 0.0, -0.1, 0.9, 0.3, None)],
    "farm_yard": [("barn", "restricted", 0.25, 0.3, 0.3, 0.2, _W),
                  ("yard", "public", 0.0, -0.05, 0.7, 0.45, None),
                  ("gateway", "queue", -0.3, -0.35, 0.2, 0.12, None)],
    "residential_close": [("turning_head", "public", 0.0, 0.1, 0.4, 0.4, None),
                          ("driveways", "restricted", 0.3, 0.35, 0.25, 0.15, _W),
                          ("bus_stop", "queue", -0.3, -0.35, 0.2, 0.12, None)],
}


def build_site(spec: SiteSpec, seed: int = 0) -> Environment:
    rng = np.random.default_rng(seed + spec.seed_offset)
    w, h = spec.footprint_m
    zones: list[Zone] = [Zone(id="zone.open", tag="public", polygon=_rect(0, 0, w * 0.96, h * 0.96))]
    spawns: list[Waypoint] = []
    exits: list[Waypoint] = []
    props: list[Prop] = []

    # Entrances sit on the boundary; each is both a spawn and an exit, which is
    # what lets a narrative enter by one door and leave by another.
    for i, (ex, ey) in enumerate(_perimeter_points(w, h, spec.entrances)):
        zones.append(Zone(id=f"zone.entrance_{i}", tag="gate",
                          polygon=_rect(ex * 0.92, ey * 0.92, 6.0, 6.0)))
        spawns.append(Waypoint(id=f"spawn.entrance_{i}", pos=(float(ex), float(ey), 0.0)))
        exits.append(Waypoint(id=f"exit.entrance_{i}", pos=(float(ex), float(ey), 0.0)))
        props.append(Prop(id=f"door.entrance_{i}", type="door",
                          pos=(float(ex) * 0.92, float(ey) * 0.92, 0.0)))

    if spec.type == "office_building":
        zones.append(Zone(id="zone.lobby", tag="public", polygon=_rect(0, 0, w * 0.35, h * 0.35)))
        zones.append(Zone(id="zone.reception", tag="queue",
                          polygon=_rect(0, h * 0.22, w * 0.25, 6.0)))
    elif spec.type == "transit_station":
        zones.append(Zone(id="zone.concourse", tag="public",
                          polygon=_rect(0, 0, w * 0.6, h * 0.4)))
        zones.append(Zone(id="zone.gateline", tag="gate",
                          polygon=_rect(0, -h * 0.18, w * 0.5, 5.0)))
        zones.append(Zone(id="zone.platform", tag="queue",
                          polygon=_rect(0, h * 0.3, w * 0.8, 8.0)))
    elif spec.type == "perimeter_site":
        # A fenced compound with a 3 m sterile strip inside its outer fence
        # (the strip is nested in the compound, so depth is measured from the
        # outer fence), a yard outside it, and open ground all round so
        # walkers can go round. Panels sit on the outer fence line.
        yf = h * 0.15
        top = h * 0.40
        cw = w * 0.70
        zones.append(Zone(id="zone.compound", tag="restricted",
                          polygon=_rect(0, (yf + top) / 2, cw, top - yf),
                          allowed_roles=["worker", "guard"]))
        zones.append(Zone(id="zone.fence_line", tag="perimeter_fence",
                          polygon=_rect(0, yf + 1.5, cw, 3.0), allowed_roles=["worker", "guard"]))
        zones.append(Zone(id="zone.yard", tag="public", polygon=_rect(0, -h * 0.1, w * 0.8, h * 0.45)))
        props.append(Prop(id="gate.compound", type="gate", pos=(0.0, float(yf), 0.0)))
        props.append(Prop(id="asset_compound", type="cabinet", pos=(0.0, float((yf + top) / 2 + 1.0), 0.0)))
        for i in range(6):
            props.append(Prop(id=f"fence_seg_{i}", type="fence_panel", cuttable=True,
                              pos=(float(-cw * 0.45 + i * cw * 0.18), float(yf), 0.0),
                              length_m=3.0))
    elif spec.type == "car_park":
        zones.append(Zone(id="zone.bays", tag="parking", polygon=_rect(0, 0, w * 0.8, h * 0.7)))
        for i in range(max(spec.obstacles, 8)):     # pillars: the occlusion source
            props.append(Prop(id=f"pillar_{i}", type="pillar",
                              pos=(float(rng.uniform(-w * 0.4, w * 0.4)),
                                   float(rng.uniform(-h * 0.35, h * 0.35)), 0.0)))
    elif spec.type == "event_plaza":
        zones.append(Zone(id="zone.stage", tag="restricted",
                          polygon=_rect(0, h * 0.35, w * 0.3, 8.0), allowed_roles=["worker", "guard"]))
        zones.append(Zone(id="zone.crowd", tag="public", polygon=_rect(0, 0, w * 0.7, h * 0.5)))
    elif spec.type in _V2_ZONES:
        for zid, tag, cx, cy, fw, fh, roles in _V2_ZONES[spec.type]:
            zones.append(Zone(id=f"zone.{zid}", tag=tag,
                              polygon=_rect(cx * w, cy * h, fw * w, fh * h),
                              **({"allowed_roles": roles} if roles else {})))
        if spec.type == "utility_yard":          # fenced compound, as perimeter_site
            yf, top, cw = h * 0.15, h * 0.40, w * 0.60
            zones.append(Zone(id="zone.fence_line", tag="perimeter_fence",
                              polygon=_rect(0, yf + 1.5, cw, 3.0), allowed_roles=["worker", "guard"]))
            props.append(Prop(id="gate.compound", type="gate", pos=(0.0, float(yf), 0.0)))
            for i in range(5):
                props.append(Prop(id=f"fence_seg_{i}", type="fence_panel", cuttable=True,
                                  pos=(float(-cw * 0.4 + i * cw * 0.2), float(yf), 0.0),
                                  length_m=3.0))
    else:  # retail_street
        zones.append(Zone(id="zone.street", tag="public", polygon=_rect(0, 0, w * 0.9, h * 0.3)))

    for i in range(spec.restricted_zones):
        cx = float(rng.uniform(-w * 0.35, w * 0.35))
        cy = float(rng.uniform(-h * 0.35, h * 0.35))
        zones.append(Zone(id=f"zone.restricted_{i}", tag="restricted",
                          polygon=_rect(cx, cy, 10.0, 8.0), allowed_roles=["worker", "guard"]))
        props.append(Prop(id=f"asset_{i}", type="cabinet", pos=(cx, cy, 0.0)))
        # The one legitimate way in: a gate on the edge facing the site centre.
        gx, gy = (cx, cy - 4.0 * np.sign(cy or 1.0)) if abs(cy) / 4.0 >= abs(cx) / 5.0 \
            else (cx - 5.0 * np.sign(cx or 1.0), cy)
        props.append(Prop(id=f"gate.restricted_{i}", type="gate", pos=(float(gx), float(gy), 0.0)))

    # Blind spots are placed in corners, the part of a site cameras least often
    # cover. `verify_blind_spots` later confirms they really are unobserved.
    for i, (bx, by) in enumerate(_perimeter_points(w * 0.8, h * 0.8, max(spec.blind_spots, 1) * 4)):
        if i % 4 or i // 4 >= spec.blind_spots:
            continue
        zones.append(Zone(id=f"zone.blindspot_{i // 4}", tag="blindspot",
                          polygon=_rect(float(bx), float(by), 7.0, 7.0)))

    # Camera mount candidates: boundary posts plus a few interior masts.
    mounts = _perimeter_points(w * 0.9, h * 0.9, 16) + [
        (float(rng.uniform(-w * 0.3, w * 0.3)), float(rng.uniform(-h * 0.3, h * 0.3)))
        for _ in range(6)
    ]
    for i, (mx, my) in enumerate(mounts):
        props.append(Prop(id=f"mount_{i}", type="camera_mount", pos=(float(mx), float(my), 0.0),
                          reachable_height_m=2.6))

    return Environment(
        id=f"{spec.type}_{w:.0f}x{h:.0f}",
        zones=zones, spawn_points=spawns, exits=exits, props=props,
    )


def mount_candidates(env: Environment) -> list[tuple[float, float]]:
    return [(p.pos[0], p.pos[1]) for p in env.props if p.type == "camera_mount"]

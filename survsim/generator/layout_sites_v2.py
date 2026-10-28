"""Eight more site types for `layout.build_layout` (infinite sites v2).

Built only from the primitives the editor already builds
(`scripts/ue/survsim_scene.build_from_plan_v2`): blocks, roads, fences, walls,
hedges, containers, trailers, dock doors, canopies, pavements (with a
material), yard lines, signs, benches, cars ... Each type is a different
analytic problem and carries its own hard negative (see
`survsim.generator.batch.SITE_PROFILES`); each has 2-3 layout variants drawn
from the seed, in the same canonical frame as the originals (front to the
south, the walk core |x|, |y| <= C kept clear of solid structure).
"""
from __future__ import annotations

import math

from .layout import _boundary, _crossing, _footprint, _road_x, _road_y, _row, _side_point


def _gaps(L):
    return [(e["side"], e["along"]) for e in L.entrances]


# ---- loading dock ------------------------------------------------------------
# Distribution-centre yard: warehouse faces with dock doors and trailers backed
# on, drivers waiting on the apron (hard negative for loiter), tailgating
# through the pedestrian door.

def _loading_dock(L, rng, B, n_ent):
    variant = rng.choice(["flush_docks", "cross_dock", "yard_apron"])
    L.variant["dock"] = variant
    faces = [1, -1] if variant == "cross_dock" else [1]
    n_dock = 0
    for sgn in faces:
        depth = rng.uniform(22, 34)
        fy = sgn * (B + rng.uniform(15.0, 18.0))
        wh = L.add("block", rng.uniform(-5, 5), fy + sgn * depth / 2, 0.0,
                   w=rng.uniform(2.2, 3.0) * B, d=depth, h=rng.uniform(9, 13), style="warehouse",
                   facade=rng.randrange(6))
        if not wh:
            continue
        pitch = rng.choice([4.0, 4.5, 5.0])
        x = wh.x - wh.w / 2 + 3.0
        while x < wh.x + wh.w / 2 - 3.0:
            L.add("dock_door", x, fy - sgn * 0.15, 0.0, w=3.0, d=0.3, h=3.6)
            L.add("yard_line", x + pitch / 2, fy - sgn * 7.5, 90.0, w=15.0, d=0.15, h=0.005,
                  colour="yellow")
            if rng.random() < (0.7 if variant == "yard_apron" else 0.45):
                L.add("trailer", x, fy - sgn * 7.6, 90.0, w=13.6, d=2.5, h=3.9, paint=rng.randrange(7))
            n_dock += 1
            x += pitch
    L.variant["docks"] = n_dock
    if variant == "yard_apron":          # a marshalling row of drop trailers along the south
        for k in range(rng.randint(3, 6)):
            L.add("trailer", -B + k * 4.2, -(B + 10.0), 90.0, w=13.6, d=2.5, h=3.9,
                  paint=rng.randrange(7))
    btype = rng.choice(["mesh", "palisade"])
    _boundary(L, "fence_run", B + 4.0, _gaps(L), rng, 2.4, sides="EW", style=btype)
    L.variant["boundary"] = btype
    for e in L.entrances[:1]:
        nx, ny = math.cos(math.radians(e["yaw"])), math.sin(math.radians(e["yaw"]))
        tx, ty = -ny, nx
        L.add("gatehouse", e["x"] + tx * 6.0 + nx * 3.0, e["y"] + ty * 6.0 + ny * 3.0, 0.0, w=4.0, d=3.0)
        L.add("barrier", e["x"] - tx * 3.0, e["y"] - ty * 3.0, e["yaw"], w=0.4, d=0.4, arm=5.0)
    _row(L, "pallets", -0.8 * B, 0.8 * B, B + 1.2, 1.5, rng=rng, p=0.5, w=1.2, d=1.0)
    L.add("skip", rng.choice([-1, 1]) * (B + 1.5), -(B - 3.0), 90.0, w=3.7, d=1.8)
    L.add("sign", rng.uniform(-0.5, 0.5) * B, B - 0.6, 0.0, w=1.2, d=0.1, h=2.4,
          text="DRIVERS WAIT HERE", colour="yellow")
    for sx in (-1, 1):
        L.add("floodlight", sx * (B + 3.0), B + 3.0, 0.0, w=0.4, d=0.4, h=12.0)


# ---- school gate ---------------------------------------------------------------
# The pavement outside a school at pick-up: parents waiting (the hard negative
# for loiter), a zig-zag keep-clear, a palisade fence and one pupil gate.

def _school_gate(L, rng, B, n_ent):
    variant = rng.choice(["primary", "secondary", "academy"])
    L.variant["school"] = variant
    floors = {"primary": (1, 2), "secondary": (2, 3), "academy": (3, 5)}[variant]
    shape = rng.choice(["rect", "L"]) if variant == "primary" else rng.choice(["rect", "L", "U"])
    depth = rng.uniform(14, 22)
    face_y = B + rng.uniform(9.0, 14.0)
    _footprint(L, rng, shape, rng.uniform(-6, 6), face_y + depth / 2, rng.uniform(1.6, 2.4) * B,
               depth, rng.randint(*floors) * 3.6 + 1.2, "office")
    L.variant["shape"] = shape
    # Playground: markings between fence and building.
    L.add("pavement", 0.0, (B + 1.5 + face_y) / 2, 0.0, w=2.2 * B, d=face_y - B - 2.0, h=0.02,
          material="asphalt")
    for k in range(rng.randint(3, 6)):
        L.add("yard_line", rng.uniform(-B, B), rng.uniform(B + 3.0, face_y - 2.0),
              rng.choice([0.0, 90.0]), w=rng.uniform(3, 6), d=0.12, h=0.005,
              colour=rng.choice(["yellow", "white"]))
    # Palisade along the north with the pupil gate.
    gx = rng.uniform(-0.4, 0.4) * B
    x = -2.0 * B
    while x < 2.0 * B:
        if abs(x + 1.5 - gx) > 2.5:
            L.add("fence_run", x + 1.5, B + 1.2, 0.0, w=3.0, d=0.25, h=2.2, style="palisade")
        x += 3.0
    L.add("sign", gx + 3.2, B + 0.6, 0.0, w=1.2, d=0.1, h=2.2, text="SCHOOL", colour="yellow")
    L.variant["gate_x"] = round(gx, 1)
    # Street with zig-zag keep-clear marking on the school side.
    rw = rng.choice([6.5, 7.3])
    ry = -(B + 3.0 + rw / 2)
    _road_x(L, ry, -3 * B, 3 * B, rw)
    L.add("pavement", 0.0, -(B + 1.5), 0.0, w=6 * B, d=3.0, h=0.14)
    zy = ry + rw / 2 - 0.35
    for k in range(-5, 6):
        L.add("yard_line", k * 1.6, zy, 30.0 if k % 2 else -30.0, w=1.8, d=0.12, h=0.005,
              colour="yellow")
    _crossing(L, rng.uniform(-0.6, 0.6) * B, ry, 0.0, rw)
    for k in range(rng.randint(2, 5)):
        L.add("car", -2.4 * B + k * 5.5, ry - rw / 2 + 1.1, 0.0, w=4.4, d=1.8, paint=rng.randrange(7))
    # Guard rail at the kerb, scooter racks and benches by the gate.
    _row(L, "barrier_row", -0.9 * B, 0.9 * B, -(B + 0.4), 2.2, rng=rng, p=0.7, w=2.2, d=0.4)
    _row(L, "cycle_rack", gx - 8.0, gx + 8.0, B - 0.5, 4.0, rng=rng, p=0.6, w=2.2, d=0.8)
    for _ in range(rng.randint(1, 3)):
        L.add("bench", rng.choice([-1, 1]) * rng.uniform(0.5, 0.9) * B, B - 1.5, 0.0, w=1.8, d=0.5)
    _row(L, "tree", -1.6 * B, 1.6 * B, B + 3.5, rng.uniform(8, 13), rng=rng, p=0.7,
         w=1.0, d=1.0, h=rng.uniform(5, 8))


# ---- rail platform -------------------------------------------------------------
# The platform is the walk core; ballast and rails beyond its edge; passengers
# waiting (hard negative for loiter and for unattended-bag), trespass onto the
# track is the restricted zone.

def _rail_platform(L, rng, B, n_ent):
    variant = rng.choice(["island", "side", "bay"])
    L.variant["platform"] = variant
    tracks = [-1, 1] if variant == "island" else [-1]
    x_end = 0.6 * B if variant == "bay" else 3 * B
    for s in tracks:
        ty = s * (B + 2.6)
        L.add("pavement", (-3 * B + x_end) / 2, ty, 0.0, w=x_end + 3 * B, d=4.4, h=0.02,
              material="gravel")
        for r in (-0.72, 0.72):
            L.add("yard_line", (-3 * B + x_end) / 2, ty + r, 0.0, w=x_end + 3 * B, d=0.09, h=0.005,
                  colour="grey")
        x = -3 * B + 0.8
        while x < x_end:
            L.add("yard_line", x, ty, 90.0, w=2.6, d=0.25, h=0.004, colour="black")
            x += 2.4
        L.add("tactile", 0.0, s * (B - 0.4), 0.0, w=2 * B + 4.0, d=0.6, h=0.005)
        L.add("yard_line", 0.0, s * (B - 1.1), 0.0, w=2 * B + 4.0, d=0.1, h=0.005, colour="yellow")
        if variant == "bay":
            L.add("barrier_row", x_end, ty, 90.0, w=3.0, d=0.8)        # buffer stop
        if rng.random() < 0.55:                     # a train standing at the platform
            n_car = rng.randint(2, 4)
            x0 = rng.uniform(-2.5 * B, -1.2 * B)
            for k in range(n_car):
                cx = x0 + k * 20.0
                if cx + 10.0 > x_end:
                    break
                L.add("trailer", cx, ty, 0.0, w=19.0, d=2.8, h=3.9, paint=rng.choice([2, 4, 5]))
        L.add("fence_run", 0.0, s * (B + 5.6), 0.0, w=6 * B, d=0.2, h=1.8, style="mesh")
    if variant != "island":                          # station building on the north
        face_y = B + rng.uniform(3.0, 6.0)
        depth = rng.uniform(10, 16)
        _footprint(L, rng, rng.choice(["rect", "L"]), rng.uniform(-6, 6), face_y + depth / 2,
                   rng.uniform(1.6, 2.4) * B, depth, rng.uniform(6.0, 10.0), "mixed")
        L.add("canopy", 0.0, face_y - 2.0, 0.0, w=1.4 * B, d=4.0, h=3.8, flat_roof=True)
    for s in (-1, 1):
        L.add("board", s * rng.uniform(0.3, 0.6) * B, 0.55 * L.core_half, 0.0, w=2.4, d=0.2)
    for _ in range(rng.randint(2, 4)):
        L.add("bench", rng.uniform(-0.8, 0.8) * B, rng.choice([-1, 1]) * (B - 2.3), 0.0, w=1.8, d=0.5)
    _row(L, "bin", -0.8 * B, 0.8 * B, 0.0, rng.uniform(10, 16), rng=rng, p=0.7, w=0.55, d=0.55)
    L.add("ticket_machine", rng.choice([-1, 1]) * (B - 1.0), 0.4 * B, 90.0, w=0.8, d=0.6)
    L.add("sign", -0.2 * B, 0.2 * B, 0.0, w=0.8, d=0.1, h=2.6, text="PLATFORM", colour="navy")


# ---- hospital entrance ---------------------------------------------------------
# Drop-off loop outside a hospital: smokers by the shelter (hard negative for
# loiter), ambulance bay (restricted), taxis, tailgating the staff door.

def _hospital_entrance(L, rng, B, n_ent):
    variant = rng.choice(["main_entrance", "emergency", "outpatients"])
    L.variant["hospital"] = variant
    depth = rng.uniform(20, 32)
    face_y = B + rng.uniform(2.0, 5.0)
    shape = rng.choice(["rect", "U", "courtyard", "L"])
    _footprint(L, rng, shape, rng.uniform(-5, 5), face_y + depth / 2, rng.uniform(2.0, 2.8) * B,
               depth, rng.randint(3, 7) * 3.8 + 1.5, "office" if variant == "emergency" else "glass")
    L.variant["shape"] = shape
    for e in L.entrances:
        if e["side"] == "N":
            L.add("canopy", e["x"], face_y - 3.0, 0.0, w=12.0, d=6.0, h=4.2, flat_roof=True)
            L.add("sign", e["x"] + 7.0, face_y - 0.6, 0.0, w=1.6, d=0.1, h=2.8,
                  text="A&E" if variant == "emergency" else "MAIN ENTRANCE",
                  colour="red" if variant == "emergency" else "blue")
    ry = -(B + 4.0)
    _road_x(L, ry, -3 * B, 3 * B, 6.5)
    for k in range(3):                    # drop-off bays: hatched, kerb-side
        L.add("hatch", (k - 1) * 7.0, ry + 2.0, 0.0, w=6.0, d=2.2, h=0.005)
        L.add("yard_line", (k - 1) * 7.0 + 3.0, ry + 2.0, 90.0, w=2.4, d=0.12, h=0.005,
              colour="white")
    L.add("tactile", 0.0, -(B - 0.2), 0.0, w=2 * B, d=0.6, h=0.005)
    L.add("pavement", 0.0, -(B + 0.4), 0.0, w=6 * B, d=0.8, h=0.14)
    side = rng.choice([-1, 1])
    if variant == "emergency":            # ambulance bays with hatching
        for k in range(rng.randint(2, 3)):
            x = side * (B + 3.0)
            y = -0.5 * B + k * 4.0
            L.add("hatch", x, y, 0.0, w=6.0, d=3.2, h=0.005)
            L.add("car", x, y, 0.0, w=5.8, d=2.2, paint=5)
        L.add("sign", side * (B + 0.6), 0.6 * B, 90.0, w=1.4, d=0.1, h=2.4, text="AMBULANCES ONLY",
              colour="red")
    elif variant == "outpatients":        # bus stop on the loop
        L.add("bus_shelter", rng.uniform(-0.6, 0.6) * B, -(B + 1.6), 0.0, w=4.0, d=1.4, style="bus")
        L.add("yard_line", 0.0, ry + 2.0, 0.0, w=14.0, d=0.15, h=0.005, colour="yellow")
    else:                                 # taxi rank
        for k in range(rng.randint(2, 4)):
            L.add("car", -1.8 * B + k * 5.2, ry + 1.6, 0.0, w=4.6, d=1.85, paint=1, taxi=True)
    L.add("bus_shelter", -side * (B + 2.0), 0.3 * B, 90.0, w=3.2, d=1.6, style="smoking")
    L.add("sign", -side * (B - 0.6), -0.3 * B, 90.0, w=0.8, d=0.1, h=2.2, text="NO SMOKING",
          colour="white")
    _row(L, "bollards", -0.7 * B, 0.7 * B, B - 0.6, 3.5, rng=rng, p=0.7, w=3.0, d=0.2)
    _row(L, "bench", -0.7 * B, 0.7 * B, 0.55 * L.core_half, 0.45 * B, rng=rng, p=0.5, w=1.8, d=0.5)
    L.add("ticket_machine", side * (B - 1.0), -0.6 * B, 90.0, w=0.8, d=0.6, label="PAY")
    _row(L, "tree", -1.5 * B, 1.5 * B, -(B + 1.2), rng.uniform(8, 13), rng=rng, p=0.6,
         w=1.0, d=1.0, h=rng.uniform(5, 7))


# ---- utility yard (substation / water works / telecom) --------------------------
# An unmanned compound behind a palisade: sterile, so every person inside is an
# intrusion; the maintenance visit is the hard negative.

def _utility_yard(L, rng, B, n_ent):
    variant = rng.choice(["substation", "water_works", "telecom"])
    L.variant["utility"] = variant
    cy0 = B + 3.0
    cd = rng.uniform(18, 28)
    cw = rng.uniform(1.6, 2.4) * B
    gx = rng.uniform(-0.3, 0.3) * cw
    x = -cw / 2
    while x < cw / 2 - 0.1:
        if abs(x + 1.5 - gx) > 2.6:
            L.add("fence_run", x + 1.5, cy0, 0.0, w=3.0, d=0.25, h=2.4, style="palisade")
        x += 3.0
    for s in (-1, 1):
        y = cy0
        while y < cy0 + cd:
            L.add("fence_run", s * cw / 2, y + 1.5, 90.0, w=3.0, d=0.25, h=2.4, style="palisade")
            y += 3.0
    L.add("sign", gx + 3.2, cy0 - 0.5, 0.0, w=0.9, d=0.1, h=2.0, text="DANGER", colour="yellow")
    L.variant["gate_x"] = round(gx, 1)
    inner = cy0 + 4.0
    if variant == "substation":
        for i in range(rng.randint(2, 4)):
            for j in range(2):
                L.add("container", -cw / 2 + 6.0 + i * 7.0, inner + 2.0 + j * 7.0, 0.0,
                      w=4.0, d=3.0, h=2.6, paint=2)
        L.add("block", cw / 2 - 6.0, inner + cd - 8.0, 0.0, w=8.0, d=5.0, h=4.0, style="warehouse",
              facade=rng.randrange(6))
    elif variant == "water_works":
        for i in range(rng.randint(2, 3)):
            L.add("fountain", -cw / 2 + 7.0 + i * 10.0, inner + 6.0, 0.0, w=8.0, d=8.0, h=1.2)
        L.add("block", cw / 2 - 7.0, inner + 4.0, 0.0, w=10.0, d=6.0, h=5.0, style="warehouse",
              facade=rng.randrange(6))
    else:
        L.add("speaker_tower", 0.0, inner + cd / 2, 0.0, w=1.6, d=1.6, h=24.0)
        for i in range(rng.randint(3, 5)):
            L.add("container", -cw / 2 + 4.0 + i * 3.5, inner + 1.5, 0.0, w=2.5, d=1.6, h=2.2, paint=2)
    for sx in (-1, 1):
        L.add("floodlight", sx * (cw / 2 - 1.0), cy0 + 1.0, 0.0, w=0.4, d=0.4, h=10.0)
    # Outer mesh fence round the apron, a van parked, cones.
    btype = rng.choice(["mesh", "palisade"])
    _boundary(L, "fence_run", B + 5.0, _gaps(L), rng, 2.4, sides="SEW", style=btype)
    L.variant["boundary"] = btype
    for e in L.entrances[:1]:
        nx, ny = math.cos(math.radians(e["yaw"])), math.sin(math.radians(e["yaw"]))
        L.add("barrier", e["x"] + ny * 3.0, e["y"] - nx * 3.0, e["yaw"], w=0.4, d=0.4, arm=5.0)
        L.add("sign", e["x"] - ny * 3.4, e["y"] + nx * 3.4, e["yaw"] + 90, w=1.2, d=0.1, h=2.2,
              text="KEEP OUT", colour="yellow")
        if e["side"] in "SN":
            _road_y(L, e["x"], min(e["y"], e["y"] + ny * 30), max(e["y"], e["y"] + ny * 30), 5.5,
                    lanes=False)
    L.add("car", rng.choice([-1, 1]) * 0.8 * B, B - 1.5, 0.0, w=5.2, d=2.0, paint=5)
    for _ in range(rng.randint(1, 4)):
        L.add("asset", rng.uniform(-B, B), rng.choice([-1, 1]) * (B - 0.8), 0.0, w=0.4, d=0.4,
              h=0.7, asset="traffic_cone")


# ---- construction site ---------------------------------------------------------
# Heras hoarding, welfare cabins, a frame going up and a tower crane: hi-vis
# workers are the hard negative for zone intrusion.

def _construction_site(L, rng, B, n_ent):
    variant = rng.choice(["groundworks", "frame", "fit_out"])
    L.variant["stage"] = variant
    _boundary(L, "fence_run", B + 2.0, _gaps(L), rng, 2.0, step=3.5, style="heras")
    face_y = B + rng.uniform(6.0, 10.0)
    depth = rng.uniform(16, 26)
    width = rng.uniform(1.6, 2.4) * B
    if variant == "groundworks":
        L.add("pavement", 0.0, face_y + depth / 2, 0.0, w=width, d=depth, h=0.02, material="gravel")
        for k in range(rng.randint(2, 4)):
            L.add("skip", -width / 2 + 4.0 + k * 6.0, face_y + 2.0, 0.0, w=3.7, d=1.8)
        L.add("trailer", rng.uniform(-0.5, 0.5) * width, face_y + depth / 2, 0.0, w=8.0, d=2.5,
              h=3.2, paint=7)
    else:
        L.add("block", rng.uniform(-4, 4), face_y + depth / 2, 0.0, w=width, d=depth,
              h=rng.choice([9.0, 12.0, 15.0, 18.0]),
              style="deck" if variant == "frame" else "glass", facade=rng.randrange(6))
    # Tower crane: mast + jib.
    cx = rng.choice([-1, 1]) * (width / 2 + 4.0)
    ch = rng.uniform(24, 34)
    L.add("speaker_tower", cx, face_y + depth / 2, 0.0, w=1.6, d=1.6, h=ch)
    L.add("shop_sign", cx - math.copysign(12.0, cx), face_y + depth / 2, 0.0, w=36.0, d=1.2, h=1.2,
          z=ch - 1.0, colour=7)
    # Welfare cabins stacked two high on one side.
    ws = rng.choice([-1, 1])
    for k in range(rng.randint(2, 4)):
        y = -0.7 * B + k * 3.2
        L.add("container", ws * (B + 5.5), y, 0.0, w=6.1, d=2.44, h=2.6, paint=rng.choice([2, 4, 7]))
        if rng.random() < 0.6:
            L.add("container", ws * (B + 5.5), y, 0.0, w=6.1, d=2.44, h=5.2, paint=rng.choice([2, 4, 7]),
                  stacked=True)
    for k in range(rng.randint(1, 3)):
        L.add("portable_toilet", -ws * (B + 3.8), -0.5 * B + k * 1.6, 90.0, w=1.2, d=1.2)
    _row(L, "pallets", -0.7 * B, 0.7 * B, B + 3.5, 1.6, rng=rng, p=0.55, w=1.2, d=1.0)
    for _ in range(rng.randint(2, 5)):
        L.add("asset", rng.uniform(-0.9, 0.9) * B, rng.uniform(-0.9, 0.9) * B, rng.uniform(0, 180),
              w=1.5, d=0.5, h=1.0, asset="kenney_construction_barrier")
    for _ in range(rng.randint(2, 6)):
        L.add("asset", rng.uniform(-B, B), rng.uniform(-B, B), 0.0, w=0.4, d=0.4, h=0.7,
              asset="traffic_cone")
    L.add("sign", rng.uniform(-0.6, 0.6) * B, -(B + 1.4), 0.0, w=1.6, d=0.1, h=2.2,
          text="HARD HATS", colour="yellow")
    for sx in (-1, 1):
        L.add("floodlight", sx * (B + 3.0), -(B + 3.0), 0.0, w=0.4, d=0.4, h=10.0)
    ry = -(B + 8.0)
    _road_x(L, ry, -3 * B, 3 * B, 7.3)
    L.add("hatch", rng.uniform(-0.5, 0.5) * B, ry + 5.0, 0.0, w=8.0, d=4.0, h=0.005)


# ---- farm yard -----------------------------------------------------------------
# A working yard: barns, a farmhouse, bales, a trailer; the farmer and
# contractors are the hard negatives for barn intrusion and recon (hare
# coursing / plant theft is the real-world rural crime problem).

def _farm_yard(L, rng, B, n_ent):
    variant = rng.choice(["dairy", "arable", "stables"])
    L.variant["farm"] = variant
    if variant == "stables":
        x = -B
        while x < B:                          # a row of low loose boxes on the north
            L.add("block", x + 2.0, B + 4.5, 0.0, w=4.0, d=5.0, h=3.4, style="mixed",
                  facade=rng.randrange(6))
            x += 4.0
        L.add("block", rng.choice([-1, 1]) * (B + 7.0), 0.0, 0.0, w=9.0, d=14.0, h=6.0,
              style="warehouse", facade=rng.randrange(6))
    else:
        n_barn = rng.randint(1, 2) if variant == "arable" else rng.randint(2, 3)
        for k in range(n_barn):
            w = rng.uniform(18, 30) if variant == "arable" else rng.uniform(24, 36)
            h = rng.uniform(7, 10) if variant == "arable" else rng.uniform(5, 7)
            side = ("N", "E", "W")[k]
            off = B + rng.uniform(4.0, 8.0) + 8.0
            px, py = _side_point(side, rng.uniform(-0.3, 0.3) * B, off)
            sw, sd = (w, 16.0) if side == "N" else (16.0, w)
            L.add("block", px, py, 0.0, w=sw, d=sd, h=h, style="warehouse", facade=rng.randrange(6))
    # Farmhouse on a corner.
    hs = rng.choice([-1, 1])
    L.add("block", hs * (B + 9.0), -(B + 6.0), 0.0, w=10.0, d=8.0, h=7.5, style="mixed",
          facade=rng.randrange(6))
    # Bales, a trailer, a pickup, muck heap.
    bx, by = -hs * 0.8 * B, B - 1.5
    for k in range(rng.randint(3, 7)):
        L.add("pallets", bx + (k % 4) * 1.3 * -hs, by - (k // 4) * 1.3, 0.0, w=1.2, d=1.2, h=1.1)
    if rng.random() < 0.7:
        L.add("trailer", hs * 0.7 * B, B + 2.5, 0.0, w=7.0, d=2.4, h=2.6, paint=rng.choice([3, 4]))
    L.add("car", -hs * (B - 2.0), -0.4 * B, 90.0, w=5.2, d=1.9, paint=rng.randrange(7))
    L.add("pavement", hs * 0.6 * B, -(B - 2.0), 0.0, w=5.0, d=4.0, h=0.4, material="gravel")
    # Boundary: hedges and a dry-stone wall, fields beyond.
    btype = rng.choice(["hedge", "wall"])
    _boundary(L, "hedge" if btype == "hedge" else "wall_run", B + 1.5, _gaps(L), rng,
              1.6 if btype == "hedge" else 1.1, sides="SW" if hs > 0 else "SE")
    L.variant["boundary"] = btype
    for sx in (-1, 1):
        L.add("grass", sx * 3.2 * B, 0.0, 0.0, w=3 * B, d=6 * B, h=0.02)
    L.add("grass", 0.0, -3.2 * B, 0.0, w=3 * B, d=2.5 * B, h=0.02)
    _row(L, "tree", -2.5 * B, 2.5 * B, -(B + 4.0), rng.uniform(6, 11), rng=rng, p=0.6,
         w=1.0, d=1.0, h=rng.uniform(6, 10))
    for e in L.entrances[:1]:
        nx, ny = math.cos(math.radians(e["yaw"])), math.sin(math.radians(e["yaw"]))
        L.add("sign", e["x"] - ny * 3.4, e["y"] + nx * 3.4, e["yaw"] + 90, w=1.0, d=0.1, h=1.6,
              text="PRIVATE", colour="white")


# ---- residential close ---------------------------------------------------------
# Houses round a turning head: residents, deliveries and dog walkers are the
# hard negatives for recon (someone casing driveways) and follow.

def _residential_close(L, rng, B, n_ent):
    variant = rng.choice(["cul_de_sac", "terrace_row", "semis"])
    L.variant["street"] = variant
    rw = 5.5
    _road_y(L, 0.0, -3 * B, 0.2 * L.core_half, rw, lanes=False)
    L.add("road", 0.0, 0.25 * L.core_half, 0.0, w=14.0, d=12.0, h=0.01)      # turning head
    L.add("sign", -rw / 2 - 1.5, -(B - 0.5), 0.0, w=0.8, d=0.1, h=2.2, text="NO THROUGH ROAD",
          colour="white")
    pw, gap = {"cul_de_sac": (9.0, 5.0), "terrace_row": (5.0, 0.0), "semis": (11.0, 3.5)}[variant]
    for side in ("N", "E", "W"):
        off = B + rng.uniform(5.0, 8.0)
        a = -1.6 * B + rng.uniform(0.0, 3.0)
        while a < 1.6 * B:
            d = rng.uniform(8.0, 10.0)
            px, py = _side_point(side, a + pw / 2, off + d / 2)
            sw, sd = (pw, d) if side == "N" else (d, pw)
            L.add("block", px, py, 0.0, w=sw, d=sd, h=rng.uniform(6.5, 8.5), style="mixed",
                  facade=rng.randrange(6))
            # front garden, driveway car, wheelie bins
            gx, gy = _side_point(side, a + pw / 2, off - 2.5)
            L.add("grass", gx, gy, 0.0 if side == "N" else 90.0, w=pw * 0.6, d=3.5, h=0.03)
            if gap > 0 and rng.random() < 0.6:
                cx, cy = _side_point(side, a + pw + gap / 2, off - 1.0)
                L.add("car", cx, cy, 90.0 if side == "N" else 0.0, w=4.4, d=1.8, paint=rng.randrange(7))
            if rng.random() < 0.5:
                bx, by = _side_point(side, a + 0.6, off - 0.8)
                L.add("bin", bx, by, 0.0, w=0.6, d=0.6, h=1.1)
            a += pw + gap
        btype = rng.choice(["hedge", "low_wall"])
        L.variant[f"front_{side}"] = btype
    _boundary(L, "hedge", B + 1.5, _gaps(L), rng, 1.0, sides="NEW", step=3.0)
    _row(L, "tree", -0.8 * B, 0.8 * B, -(B - 0.5), rng.uniform(9, 14), rng=rng, p=0.6,
         w=1.0, d=1.0, h=rng.uniform(5, 7))
    for _ in range(rng.randint(0, 2)):
        L.add("car", rw / 2 - 1.0, rng.uniform(-B, -0.3 * B), 90.0, w=4.4, d=1.8, paint=rng.randrange(7))


BUILDERS = {"loading_dock": _loading_dock, "school_gate": _school_gate,
            "rail_platform": _rail_platform, "hospital_entrance": _hospital_entrance,
            "utility_yard": _utility_yard, "construction_site": _construction_site,
            "farm_yard": _farm_yard, "residential_close": _residential_close}
ENTRANCE_SIDES = {"loading_dock": "EW", "school_gate": "SEW", "rail_platform": "EW",
                  "hospital_entrance": "NSW", "utility_yard": "SEW", "construction_site": "SW",
                  "farm_yard": "SE", "residential_close": "SW"}
ENTRANCE_KIND = {"loading_dock": "vehicle_gate", "school_gate": "street_end",
                 "rail_platform": "ticket_gates", "hospital_entrance": "door",
                 "utility_yard": "vehicle_gate", "construction_site": "site_gate",
                 "farm_yard": "field_gate", "residential_close": "street_end"}
GROUND = {"loading_dock": [("concrete", "asphalt"), ("asphalt", "concrete")],
          "school_gate": [("slabs", "asphalt"), ("light_slabs", "slabs"), ("asphalt", "slabs")],
          "rail_platform": [("light_slabs", "concrete"), ("concrete", "slabs"), ("slabs", "light_slabs")],
          "hospital_entrance": [("light_slabs", "granite"), ("slabs", "concrete"), ("granite", "slabs")],
          "utility_yard": [("gravel", "concrete"), ("concrete", "gravel")],
          "construction_site": [("gravel", "concrete"), ("concrete", "gravel"), ("asphalt", "gravel")],
          "farm_yard": [("gravel", "concrete"), ("concrete", "gravel")],
          "residential_close": [("asphalt", "slabs"), ("asphalt", "light_slabs")]}
WEAR = {"loading_dock": (16, 32, ["oil", "tyre", "tyre", "crack"]),
        "school_gate": (8, 18, ["gum", "dirt", "leaves"]),
        "rail_platform": (10, 22, ["gum", "dirt", "crack"]),
        "hospital_entrance": (4, 10, ["gum", "dirt"]),
        "utility_yard": (10, 24, ["oil", "dirt", "crack"]),
        "construction_site": (18, 36, ["dirt", "dirt", "tyre", "oil"]),
        "farm_yard": (18, 36, ["dirt", "dirt", "tyre", "leaves"]),
        "residential_close": (6, 16, ["oil", "leaves", "crack"])}
# (kind, h min, h max, n, style) for the ring beyond the site.
BACKDROP = {"loading_dock": ("block", 8, 14, 16, "warehouse"),
            "school_gate": ("block", 7, 14, 24, "mixed"),
            "rail_platform": ("block", 10, 28, 22, "mixed"),
            "hospital_entrance": ("block", 14, 40, 20, "mixed"),
            "utility_yard": ("tree", 7, 12, 40, None),
            "construction_site": ("block", 12, 45, 20, "mixed"),
            "farm_yard": ("tree", 7, 14, 36, None),
            "residential_close": ("block", 6, 9, 28, "mixed")}
FAR_GRASS = {"utility_yard", "farm_yard"}

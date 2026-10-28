"""Incremental scene building for the persistent editor (scripts/ue/job_runner.py).

Every actor lives in an outliner folder ("Env", "Lights", "People", "Cams");
`clear(group)` removes one group so a job can rebuild the street without
touching people, relight without touching the street, and so on. Nothing here
destroys the whole level unless a job asks for it.

Coordinates are Unreal centimetres (the manifest's). Rotators ALWAYS use
keywords: `unreal.Rotator(a, b, c)` is (roll, pitch, yaw) positionally.
"""
from __future__ import annotations

import json
import math
import os
import random

import survsim_materials as M
import unreal

MAP = "/Game/SurvSim/RunMap_v2"  # RunMap.umap left corrupt by the 2026-09-27 OOM kill (bad PACKAGE_FILE_TAG); kept on disk, not reused
SEQ_PACKAGE = "/Game/SurvSim"
SEQ_NAME = "RunSeq"
ASSET_INDEX = os.path.expanduser("~/data/assets/index.json")

_eal = unreal.EditorAssetLibrary


def actors():
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


def ensure_level():
    world = unreal.EditorLevelLibrary.get_editor_world()
    if world is not None and world.get_path_name().startswith(MAP):
        return False
    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if _eal.does_asset_exist(MAP):
        levels.load_level(MAP)
    else:
        levels.new_level(MAP)
    return True


def _folder(a):
    f = str(a.get_folder_path())
    return "" if f == "None" else f


def group_actors(group):
    return [a for a in (actors().get_all_level_actors() or [])
            if _folder(a).split("/")[0] == group]


def clear(group):
    victims = group_actors(group)
    for a in victims:
        actors().destroy_actor(a)
    return len(victims)


def clear_unfoldered():
    """Remove actors from older builds (pre-folder) — only on explicit request."""
    n = 0
    for a in (actors().get_all_level_actors() or []):
        if not _folder(a):
            if isinstance(a, (unreal.WorldSettings,)) or a.get_class().get_name() in (
                    "WorldSettings", "Brush", "AbstractNavData", "RecastNavMesh"):
                continue
            actors().destroy_actor(a)
            n += 1
    return n


_SHAPES = {}


def _shape(name):
    if name not in _SHAPES:
        _SHAPES[name] = _eal.load_asset(f"/Engine/BasicShapes/{name}.{name}")
    return _SHAPES[name]


def box(group, label, center, size, material, yaw=0.0, shape="Cube"):
    """A basic shape with its size in cm (Cube/Cylinder/Sphere are 100 cm units)."""
    a = actors().spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(*center),
                                        unreal.Rotator(pitch=0, yaw=yaw, roll=0))
    comp = a.static_mesh_component
    comp.set_static_mesh(_shape(shape))
    a.set_actor_scale3d(unreal.Vector(size[0] / 100.0, size[1] / 100.0, size[2] / 100.0))
    if material is not None:
        comp.set_material(0, material)
    a.set_actor_label(label)
    a.set_folder_path(group)
    return a


# --- street layout -------------------------------------------------------------

def load_index():
    try:
        with open(ASSET_INDEX) as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else data.get("assets", [])
    except (OSError, ValueError):
        return []


_CAMS = []


def _clear(x, y, radius=800.0):
    """False if (x, y) is in a camera's near field (within `radius`, in front).

    Static clutter right in front of a lens hides the scene; real installs are
    sited to avoid it, so the generator does too.
    """
    for cx, cy, yaw in _CAMS:
        dx, dy = x - cx, y - cy
        d = math.hypot(dx, dy)
        if d < 250.0:
            return False
        if d < radius and dx * math.cos(math.radians(yaw)) + dy * math.sin(math.radians(yaw)) > 0:
            return False
    return True


def walk_envelope(manifest, margin_cm=300.0, minimum=2000.0):
    """Half-size of the square that holds every track key and camera, + margin."""
    ext = minimum - margin_cm
    for t in manifest.get("tracks", []):
        for k in t["keys"]:
            ext = max(ext, abs(k["location"][0]), abs(k["location"][1]))
    for c in manifest.get("cameras", []):
        ext = max(ext, abs(c["location"][0]), abs(c["location"][1]))
    return ext + margin_cm


def build_street(seed=0, half=2000.0, mats=None, cameras=()):
    """A 60x60 m urban block around a paved plaza of +-`half` cm.

    Layout (Unreal axes): plaza slabs in the middle; a two-lane road with kerbs,
    painted lines and parked cars along +Y; pavements, buildings and a wall on
    the other three sides; lamp posts, bins, benches, bollards and trees on the
    plaza edge. Deterministic from `seed`.
    """
    rng = random.Random(f"street:{seed}")
    L = mats or M.library(seed)
    _CAMS[:] = [(c["location"][0], c["location"][1], c["rotation"][1]) for c in cameras]
    g = "Env"
    out = {"counts": {}}

    def count(k, n=1):
        out["counts"][k] = out["counts"].get(k, 0) + n

    far = 20000.0
    box(g, "GroundFar", (0, 0, -2), (far * 2, far * 2, 4), L["asphalt"])
    box(g, "Plaza", (0, 0, 0.5), (2 * half, 2 * half, 1), L["slabs"])
    # Setts band framing the plaza.
    for i, (cx, cy, sx, sy) in enumerate([(0, half + 60, 2 * half + 240, 120),
                                          (0, -half - 60, 2 * half + 240, 120),
                                          (half + 60, 0, 120, 2 * half),
                                          (-half - 60, 0, 120, 2 * half)]):
        box(g, f"Setts_{i}", (cx, cy, 0.6), (sx, sy, 1.2), L["setts"])

    # Road along +Y: kerb, 2 lanes of 350 cm, kerb, pavement.
    y0 = half + 120
    road_w = 700.0
    box(g, "KerbS", (0, y0 + 15, 7), (far, 30, 14), L["kerb"])
    box(g, "Road", (0, y0 + 30 + road_w / 2, 0.3), (far, road_w, 0.6), L["asphalt"])
    box(g, "KerbN", (0, y0 + 30 + road_w + 15, 7), (far, 30, 14), L["kerb"])
    y_pave = y0 + 60 + road_w
    box(g, "PavementN", (0, y_pave + 200, 7), (far, 400, 14), L["slabs"])
    # Centre dashes and edge lines.
    yc = y0 + 30 + road_w / 2
    for i in range(-40, 41):
        box(g, f"Dash_{i}", (i * 900.0, yc, 0.7), (450, 12, 0.4), L["paint_white"])
        count("road_dashes")
    for side, y in (("S", y0 + 30 + 25), ("N", y0 + 30 + road_w - 25)):
        box(g, f"EdgeLine_{side}", (0, y, 0.7), (far, 10, 0.4), L["paint_yellow"])
    # Zebra crossing.
    for k in range(7):
        box(g, f"Zebra_{k}", (-600 + k * 100.0 - 300, yc, 0.8), (50, road_w - 60, 0.4),
            L["paint_white"])

    # Parked cars on the far lane (kerb side N) and a couple on the near lane.
    cars = [a for a in load_index() if a.get("kind") in ("car", "vehicle")]
    x = -half - 400.0
    while x < half + 400.0:
        x += rng.uniform(500, 900)
        if abs(x - (-900)) < 500:            # keep the zebra clear
            continue
        near = rng.random() < 0.3
        yy = y0 + 30 + (110 if near else road_w - 110)
        if not _clear(x, yy):
            continue
        _car(g, f"Car_{int(x)}", (x, yy), rng.choice([0.0, 180.0]) + rng.uniform(-3, 3),
             rng.choice(L["car_paints"]), L, cars, rng)
        count("cars")

    # Buildings: north side beyond the pavement, and on S/E/W beyond a pavement.
    def row(prefix, axis_fixed, fixed, lo, hi, facing_yaw):
        pos = lo
        while pos < hi:
            w = rng.uniform(900, 1800)
            d = rng.uniform(900, 1400)
            h = rng.choice([3, 4, 4, 5, 6]) * 320.0 + rng.uniform(0, 60)
            gap = rng.random() < 0.15
            c = pos + w / 2
            if not gap:
                if axis_fixed == "y":
                    ctr = (c, fixed + math.copysign(d / 2, fixed), h / 2)
                    size = (w, d, h)
                else:
                    ctr = (fixed + math.copysign(d / 2, fixed), c, h / 2)
                    size = (d, w, h)
                mat = rng.choice(L["facades"])
                box(g, f"{prefix}_{int(pos)}", ctr, size, mat)
                # Parapet / roof edge.
                box(g, f"{prefix}_{int(pos)}_roof",
                    (ctr[0], ctr[1], h + 30), (size[0] + 20, size[1] + 20, 60), L["concrete"])
                count("buildings")
            pos += w + (rng.uniform(300, 700) if gap else rng.uniform(0, 40))

    y_bld = y_pave + 400
    row("BldN", "y", y_bld, -6000, 6000, 0)
    edge = half + 700.0
    box(g, "PavementS", (0, -half - 400, 7), (2 * half + 1400, 560, 14), L["slabs"])
    box(g, "PavementE", (half + 400, -200, 7), (560, 2 * half + 800, 14), L["slabs"])
    box(g, "PavementW", (-half - 400, -200, 7), (560, 2 * half + 800, 14), L["slabs"])
    row("BldS", "y", -edge, -6000, 6000, 0)
    row("BldE", "x", edge, -edge, y0 - 50, 0)
    row("BldW", "x", -edge, -edge, y0 - 50, 0)
    # A distant ring so the horizon is never empty.
    for i in range(24):
        ang = 2 * math.pi * i / 24
        r = rng.uniform(9000, 12000)
        w, d, h = rng.uniform(2000, 3500), rng.uniform(1500, 2500), rng.uniform(1500, 4500)
        box(g, f"BldFar_{i}", (r * math.cos(ang), r * math.sin(ang), h / 2), (w, d, h),
            rng.choice(L["facades"]), yaw=math.degrees(ang) + 90)
        count("far_buildings")

    # Plaza furniture: lamps every 12 m on the edge, bins, benches, bollards, trees.
    lamps = []
    step = 1200.0
    for s in (-1, 1):
        for i in range(-2, 3):
            for (lx, ly, yaw) in ((i * step + 300, s * (half - 80), 90 if s < 0 else -90),
                                  (s * (half - 80), i * step - 300, 0 if s < 0 else 180)):
                if not _clear(lx, ly):
                    count("skipped_near_camera")
                    continue
                lamps.append(_lamp(g, f"Lamp_{len(lamps)}", lx, ly, yaw, L))
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            lx, ly = ix * half * 0.45 + 150.0, iy * half * 0.45 + 150.0
            if _clear(lx, ly):
                lamps.append(_lamp(g, f"Lamp_{len(lamps)}", lx, ly, 0, L))
    count("lamp_posts", len(lamps))
    for k in range(10):
        side = rng.choice([-1, 1])
        along = rng.uniform(-half + 200, half - 200)
        px, py = (along, side * (half - 180)) if rng.random() < 0.5 else (side * (half - 180), along)
        kind = rng.choice(["bin", "bench", "bench", "bollards"])
        if not _clear(px, py):
            count("skipped_near_camera")
            continue
        if kind == "bin":
            box(g, f"Bin_{k}", (px, py, 50), (55, 55, 100), L["metal_green"], shape="Cylinder")
        elif kind == "bench":
            yaw = 0 if abs(py) > abs(px) else 90
            box(g, f"Bench_{k}_seat", (px, py, 45), (180, 45, 6), L["wood"], yaw=yaw)
            box(g, f"Bench_{k}_back", (px, py, 75), (180, 6, 40) if yaw == 0 else (6, 180, 40),
                L["wood"])
            for dx in (-75, 75):
                off = (dx, 0) if yaw == 0 else (0, dx)
                box(g, f"Bench_{k}_leg{dx}", (px + off[0], py + off[1], 22), (8, 40, 44),
                    L["metal_dark"], yaw=yaw)
        else:
            for j in range(4):
                off = (j * 150 - 225, 0) if abs(py) > abs(px) else (0, j * 150 - 225)
                box(g, f"Bollard_{k}_{j}", (px + off[0], py + off[1], 45), (20, 20, 90),
                    L["metal_dark"], shape="Cylinder")
        count(kind)
    for k in range(6):
        tx = -half + 400 + k * (2 * half - 800) / 5.0
        if _clear(tx, -half - 350):
            _tree(g, f"Tree_S{k}", tx, -half - 350, L, rng)
        count("trees")
    for k in range(4):
        tx = -1600 + k * 1100 + rng.uniform(-100, 100)
        if _clear(tx, y_pave + 250):
            _tree(g, f"Tree_N{k}", tx, y_pave + 250, L, rng)
        count("trees")
    # Low wall + hedge to the south-west, railings to the east.
    box(g, "WallSW", (-half - 150, -half + 400, 60), (40, 1600, 120), L["walls"][0])
    box(g, "HedgeSW", (-half - 150, 700, 55), (70, 1400, 110), L["hedge"])
    for j in range(24):
        box(g, f"Rail_{j}", (half + 150, -half + 400 + j * 60, 55), (5, 5, 110), L["metal_dark"])
    box(g, "RailTop", (half + 150, -half + 400 + 11.5 * 60, 108), (6, 24 * 60, 6), L["metal_dark"])
    out["lamps"] = [(lx, ly) for lx, ly in lamps]
    return out


def _lamp(g, label, x, y, yaw, L):
    box(g, f"{label}_pole", (x, y, 300), (16, 16, 600), L["metal_dark"], shape="Cylinder")
    box(g, f"{label}_base", (x, y, 40), (26, 26, 80), L["metal_dark"], shape="Cylinder")
    ax, ay = math.cos(math.radians(yaw)) * 60, math.sin(math.radians(yaw)) * 60
    box(g, f"{label}_arm", (x + ax, y + ay, 595), (120, 8, 8), L["metal_dark"], yaw=yaw)
    box(g, f"{label}_head", (x + 2 * ax, y + 2 * ay, 588), (60, 26, 10), L["lamp_lens"], yaw=yaw)
    return (x + 2 * ax, y + 2 * ay)


def _tree(g, label, x, y, L, rng, scale=1.0):
    h = rng.uniform(350, 520) * scale
    box(g, f"{label}_trunk", (x, y, h / 2), (28 * scale, 28 * scale, h), L["bark"], shape="Cylinder")
    for j in range(3):
        r = rng.uniform(180, 260) * scale
        box(g, f"{label}_crown{j}", (x + rng.uniform(-60, 60), y + rng.uniform(-60, 60),
                                     h + rng.uniform(-40, 80)), (r, r, r * 0.85), L["hedge"],
            shape="Sphere")
    box(g, f"{label}_pit", (x, y, 1.5), (120, 120, 2), L["concrete"])


def _car(g, label, xy, yaw, paint, L, gltf_cars, rng):
    x, y = xy
    if gltf_cars:
        spec = gltf_cars[rng.randrange(len(gltf_cars))]
        mesh = import_static(spec)
        if mesh is not None:
            a = actors().spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(x, y, 0),
                                                unreal.Rotator(pitch=0, yaw=yaw, roll=0))
            a.static_mesh_component.set_static_mesh(mesh)
            a.set_actor_label(label)
            a.set_folder_path(g)
            return a
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    L_, W_ = rng.uniform(400, 470), rng.uniform(175, 190)
    box(g, f"{label}_body", (x, y, 60), (L_, W_, 70), paint, yaw=yaw)
    off = rng.uniform(-40, 10)
    box(g, f"{label}_cabin", (x + c * off, y + s * off, 118), (L_ * 0.55, W_ * 0.9, 50),
        L["glass"], yaw=yaw)
    box(g, f"{label}_roof", (x + c * off, y + s * off, 145), (L_ * 0.5, W_ * 0.86, 6), paint, yaw=yaw)
    for fx in (-0.32, 0.32):
        for fy in (-0.5, 0.5):
            wx = x + c * fx * L_ - s * fy * W_
            wy = y + s * fx * L_ + c * fy * W_
            w = box(g, f"{label}_wheel{fx}{fy}", (wx, wy, 33), (66, 66, 22), L["tyre"],
                    shape="Cylinder")
            w.set_actor_rotation(unreal.Rotator(pitch=0, yaw=yaw, roll=90), False)


_STATIC_CACHE = {}


def import_asset(src, dest):
    """Import a file once (Interchange); returns loaded assets under `dest`."""
    if not _eal.does_directory_exist(dest) or not _eal.list_assets(dest, True):
        task = unreal.AssetImportTask()
        task.filename = src
        task.destination_path = dest
        task.automated = True
        task.replace_existing = True
        task.save = True
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    found = []
    for p in _eal.list_assets(dest, True):
        a = _eal.load_asset(str(p))
        if a is not None:
            found.append(a)
    return found


def import_static(spec):
    key = spec["path"]
    if key not in _STATIC_CACHE:
        name = os.path.splitext(os.path.basename(key))[0].replace(" ", "_").replace(".", "_")
        assets = import_asset(os.path.expanduser(key), f"/Game/SurvSim/Props/{name}")
        meshes = [a for a in assets if isinstance(a, unreal.StaticMesh)]
        _STATIC_CACHE[key] = meshes[0] if meshes else None
    return _STATIC_CACHE[key]


# --- lighting --------------------------------------------------------------------

PRESETS = {
    # Physical units: sun lux, sky-light scale, fixed exposure EV100.
    "day": {"sun_lux": 40000.0, "sun_pitch": -38.0, "sun_yaw": 35.0, "sky": 1.0,
            "ev100": 13.0, "fog": 0.012, "night": 0.0},
    "overcast": {"sun_lux": 8000.0, "sun_pitch": -55.0, "sun_yaw": 35.0, "sky": 1.3,
                 "ev100": 11.5, "fog": 0.03, "night": 0.0},
    "night": {"sun_lux": 0.5, "sun_pitch": -30.0, "sun_yaw": 200.0, "sky": 10.0,
              "ev100": 1.0, "fog": 0.03, "night": 1.0, "lamp_lm": 9000.0},
}


def light(preset="day", lamps=(), hdri=None, ev100=None, sun_lux=None, sun_yaw=None,
          sun_pitch=None):
    clear("Lights")
    p = dict(PRESETS[preset])
    if sun_yaw is not None and preset != "night":
        p["sun_yaw"] = float(sun_yaw)
    if sun_pitch is not None and preset != "night":
        p["sun_pitch"] = float(sun_pitch)
    if ev100 is not None:
        p["ev100"] = ev100
    if sun_lux is not None:
        p["sun_lux"] = sun_lux
    a = actors()
    sun = a.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 1000),
                                   unreal.Rotator(pitch=p["sun_pitch"], yaw=p["sun_yaw"], roll=0))
    sun.set_actor_label("Sun")
    lc = sun.light_component
    lc.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    lc.set_editor_property("intensity", p["sun_lux"])
    lc.set_editor_property("atmosphere_sun_light", True)
    if preset == "night":
        lc.set_editor_property("light_color", unreal.Color(r=150, g=170, b=255, a=255))
    sun.set_folder_path("Lights")

    sky = a.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 800))
    sky.set_actor_label("SkyLight")
    sc = sky.light_component
    sc.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    sc.set_editor_property("intensity", p["sky"])
    if hdri:
        cube = import_hdri(hdri)
        sc.set_editor_property("source_type", unreal.SkyLightSourceType.SLS_SPECIFIED_CUBEMAP)
        sc.set_editor_property("cubemap", cube)
        sc.set_editor_property("real_time_capture", False)
        sc.set_editor_property("intensity", p["sky"] * (2500.0 if preset != "night" else 0.5))
    else:
        # [X] real_time_capture leaves shadows pitch black in editor scene
        # captures; a captured (static-source) sky light re-captured below works.
        sc.set_editor_property("real_time_capture", False)
    sky.set_folder_path("Lights")
    atm = a.spawn_actor_from_class(unreal.SkyAtmosphere, unreal.Vector(0, 0, 0))
    atm.set_folder_path("Lights")
    fog = a.spawn_actor_from_class(unreal.ExponentialHeightFog, unreal.Vector(0, 0, 0))
    fog.component.set_editor_property("fog_density", p["fog"])
    fog.component.set_editor_property("fog_height_falloff", 0.2)
    if preset == "night":
        # Urban night sky is not black: sodium/LED light pollution scattered by
        # haze. Without it the upper third of every frame is crushed to 0.
        try:
            fog.component.set_editor_property(
                "fog_inscattering_luminance", unreal.LinearColor(0.10, 0.12, 0.16, 1.0))
        except Exception:                                       # noqa: BLE001
            fog.component.set_editor_property(
                "fog_inscattering_color", unreal.LinearColor(0.10, 0.12, 0.16, 1.0))
        # Blue-grey ambient (moonlit / light-polluted sky) under the lamps.
        sc.set_editor_property("light_color", unreal.Color(r=170, g=190, b=235, a=255))
    fog.set_folder_path("Lights")

    sc.recapture_sky()
    ppv = a.spawn_actor_from_class(unreal.PostProcessVolume, unreal.Vector(0, 0, 0))
    ppv.set_actor_label("CCTVExposure")
    ppv.set_editor_property("unbound", True)
    s = ppv.get_editor_property("settings")
    # Fixed exposure. [V] SceneCapture ignores histogram min=max; manual mode
    # with physical-camera exposure OFF applies exposure = 2^bias at EV100 0, so
    # bias = -EV100 gives a true fixed EV in both MRQ and scene captures.
    for k, v in (("auto_exposure_method", unreal.AutoExposureMethod.AEM_MANUAL),
                 ("auto_exposure_min_brightness", p["ev100"]),
                 ("auto_exposure_max_brightness", p["ev100"]),
                 ("auto_exposure_bias", -p["ev100"]),
                 ("auto_exposure_apply_physical_camera_exposure", False),
                 ("motion_blur_amount", 0.0),
                 ("bloom_intensity", 0.2),
                 ("lens_flare_intensity", 0.0),
                 ("vignette_intensity", 0.25)):
        s.set_editor_property(k, v)
        s.set_editor_property("override_" + k, True)
    ppv.set_editor_property("settings", s)
    ppv.set_folder_path("Lights")

    for mic in _all_facades():
        unreal.MaterialEditingLibrary.set_material_instance_scalar_parameter_value(
            mic, "Night", p["night"])
        unreal.MaterialEditingLibrary.update_material_instance(mic)

    n_spots = 0
    if preset == "night":
        for (x, y) in lamps:
            spot = a.spawn_actor_from_class(unreal.SpotLight, unreal.Vector(x, y, 580),
                                            unreal.Rotator(pitch=-90, yaw=0, roll=0))
            c = spot.spot_light_component
            c.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
            c.set_editor_property("intensity_units", unreal.LightUnits.LUMENS)
            c.set_editor_property("intensity", p["lamp_lm"])
            c.set_editor_property("attenuation_radius", 3000.0)
            c.set_editor_property("outer_cone_angle", 78.0)
            c.set_editor_property("inner_cone_angle", 35.0)
            # 4000 K LED, 9 000 lm, 30 m reach: v5's 3400 K / 25 000 lm / 50 m
            # pools read amber and blown out (2026-09-26).
            c.set_editor_property("temperature", 4000.0)
            c.set_editor_property("use_temperature", True)
            spot.set_folder_path("Lights")
            n_spots += 1
    return {"preset": preset, "ev100": p["ev100"], "spots": n_spots, "hdri": bool(hdri)}


def _all_facades():
    out = []
    for p in _eal.list_assets(M.MAT_DIR, False):
        if "/MI_Facade" in str(p):
            out.append(_eal.load_asset(str(p)))
    return out


def import_hdri(path):
    name = "HDRI_" + os.path.splitext(os.path.basename(path))[0].replace("-", "_")
    dest = "/Game/SurvSim/HDRI"
    full = f"{dest}/{name}"
    if not _eal.does_asset_exist(full):
        task = unreal.AssetImportTask()
        task.filename = os.path.expanduser(path)
        task.destination_path = dest
        task.destination_name = name
        task.automated = True
        task.replace_existing = True
        task.save = True
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    return _eal.load_asset(full)


# --- cameras -----------------------------------------------------------------

def apply_lens(cam, spec):
    hfov = spec.get("fov_deg")
    comp = cam.get_cine_camera_component()
    width, height = spec.get("resolution", [1280, 720])
    sensor_w = 36.0
    fb = comp.get_editor_property("filmback")
    fb.set_editor_property("sensor_width", sensor_w)
    fb.set_editor_property("sensor_height", sensor_w * float(height) / float(width))
    comp.set_editor_property("filmback", fb)
    if hfov:
        focal = (sensor_w / 2.0) / math.tan(math.radians(float(hfov)) / 2.0)
        lens = comp.get_editor_property("lens_settings")
        lens.set_editor_property("min_focal_length", min(4.0, focal))
        comp.set_editor_property("lens_settings", lens)
        comp.set_editor_property("current_focal_length", focal)
    # Everything in focus, like a CCTV lens at hyperfocal.
    fs = comp.get_editor_property("focus_settings")
    fs.set_editor_property("focus_method", unreal.CameraFocusMethod.DISABLE)
    comp.set_editor_property("focus_settings", fs)


def set_camera_ev(cam, ev100):
    """Per-camera fixed exposure on the CineCamera's own post-process (same
    manual/bias scheme as the CCTVExposure volume, which it overrides)."""
    comp = cam.get_cine_camera_component()
    s = comp.get_editor_property("post_process_settings")
    for k, v in (("auto_exposure_method", unreal.AutoExposureMethod.AEM_MANUAL),
                 ("auto_exposure_min_brightness", ev100),
                 ("auto_exposure_max_brightness", ev100),
                 ("auto_exposure_bias", -ev100),
                 ("auto_exposure_apply_physical_camera_exposure", False)):
        s.set_editor_property(k, v)
        s.set_editor_property("override_" + k, True)
    comp.set_editor_property("post_process_settings", s)
    comp.set_editor_property("post_process_blend_weight", 1.0)


def cameras(manifest):
    clear("Cams")
    out = {}
    for spec in manifest.get("cameras", []):
        pitch, yaw, roll = spec["rotation"]
        cam = actors().spawn_actor_from_class(
            unreal.CineCameraActor, unreal.Vector(*spec["location"]),
            unreal.Rotator(pitch=pitch, yaw=yaw, roll=roll))
        cam.set_actor_label(spec["id"])
        cam.set_folder_path("Cams")
        apply_lens(cam, spec)
        out[spec["id"]] = cam
    return out


def find(group, label):
    for a in group_actors(group):
        if a.get_actor_label() == label:
            return a
    return None


# --- snapshot (fast look check, no sequencer) ----------------------------------------

def snapshot(camera, out_dir, name, spec, finish, delay_ticks=60, result=None):
    """SceneCapture2D from a camera's pose after `delay_ticks` (shader warm-up).

    Async: returns immediately; calls finish() from a tick when written.
    """
    a = actors()
    os.makedirs(out_dir, exist_ok=True)
    world = unreal.EditorLevelLibrary.get_editor_world()
    width, height = spec.get("resolution", [1280, 720])
    rt = unreal.RenderingLibrary.create_render_target2d(
        world, width, height, unreal.TextureRenderTargetFormat.RTF_RGBA8)
    cap = a.spawn_actor_from_class(unreal.SceneCapture2D, camera.get_actor_location(),
                                   camera.get_actor_rotation())
    cap.set_folder_path("Tmp")
    comp = cap.get_component_by_class(unreal.SceneCaptureComponent2D)
    comp.set_editor_property("texture_target", rt)
    comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
    comp.set_editor_property("capture_every_frame", True)
    comp.set_editor_property("fov_angle", float(spec.get("fov_deg") or 78.0))
    ppv = find("Lights", "CCTVExposure")
    if ppv is not None:
        # A scene capture ignores unbound volumes' exposure unless told; copy it.
        comp.set_editor_property("post_process_settings", ppv.get_editor_property("settings"))
        comp.set_editor_property("post_process_blend_weight", 1.0)
    state = {"t": 0}

    def tick(_dt):
        state["t"] += 1
        if state["t"] == delay_ticks // 2:
            sky = find("Lights", "SkyLight")
            if sky is not None:
                sky.light_component.recapture_sky()
        if state["t"] < delay_ticks:
            return
        unreal.unregister_slate_post_tick_callback(state["h"])
        comp.capture_scene()
        ok = unreal.RenderingLibrary.export_render_target(cap, rt, out_dir, f"{name}.png")
        a.destroy_actor(cap)
        finish({"snapshot": f"{out_dir}/{name}.png", "exported": bool(ok)})

    state["h"] = unreal.register_slate_post_tick_callback(tick)


# --- people ----------------------------------------------------------------------

def spawn_people(plan, chars):
    """plan: host-computed person plan; chars: {name: {mesh, height_cm, ...}}."""
    clear("People")
    for a in actors().get_all_level_actors() or []:        # strays from probes
        if isinstance(a, unreal.SkeletalMeshActor) and not _folder(a):
            actors().destroy_actor(a)
    out = {}
    for entry in plan:
        ch = chars[entry["character"]]
        first = entry["keys"][0]
        actor = actors().spawn_actor_from_class(unreal.SkeletalMeshActor,
                                                unreal.Vector(*first["location"]))
        comp = actor.skeletal_mesh_component
        comp.set_skeletal_mesh_asset(_eal.load_asset(ch["mesh"]))
        comp.set_editor_property("animation_mode", unreal.AnimationMode.ANIMATION_SINGLE_NODE)
        scale = entry["height_cm"] / ch["height_cm"]
        actor.set_actor_scale3d(unreal.Vector(scale, scale, scale))
        actor.set_actor_label(entry["uid"])
        actor.set_folder_path("People")
        if ch.get("kind") == "metahuman":
            entry["_parts"] = dress_metahuman(actor, ch)
            if entry.get("outfit"):
                import survsim_clothing
                survsim_clothing.dress_actor(actor, entry["outfit"], ch["height_cm"],
                                             ch.get("xmax_cm", 46.0))
        elif ch.get("kind") == "cc0" and entry.get("outfit"):
            import survsim_clothing
            entry["_slots"] = {"cc0_regions": survsim_clothing.dress_cc0(actor, entry["outfit"])}
        else:
            entry["_slots"] = tint_person(comp, entry, ch)
        out[entry["uid"]] = actor
    return out


def dress_metahuman(actor, ch):
    """Face mesh + outfit (ChaosCloth) on a baked MetaHuman body, leader-posed.

    Components are added as instance subobjects (SubobjectDataSubsystem) so
    they are saved with the level MRQ renders from. [V] docs/12 MetaHuman route.
    """
    body = actor.skeletal_mesh_component
    sds = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
    root = sds.k2_gather_subobject_data_for_instance(actor)[0]
    parts = []
    for cls, path in ((unreal.SkeletalMeshComponent, ch.get("face")),
                      (unreal.ChaosClothComponent, ch.get("outfit"))):
        if not path or not _eal.does_asset_exist(path):
            continue
        before = {c.get_name() for c in actor.get_components_by_class(cls)}
        sds.add_new_subobject(unreal.AddNewSubobjectParams(parent_handle=root, new_class=cls))
        comp = next(c for c in actor.get_components_by_class(cls) if c.get_name() not in before)
        comp.attach_to_component(body, "", unreal.AttachmentRule.SNAP_TO_TARGET,
                                 unreal.AttachmentRule.SNAP_TO_TARGET,
                                 unreal.AttachmentRule.SNAP_TO_TARGET, False)
        asset = _eal.load_asset(path)
        if cls is unreal.ChaosClothComponent:
            comp.set_editor_property("asset", asset)
        else:
            comp.set_skeletal_mesh_asset(asset)
        comp.set_leader_pose_component(body)
        parts.append(f"{comp.get_name()}:{cls.__name__}")
    return parts


_TOP_WORDS = ("shirt", "top", "jacket", "hoodie", "coat", "cloth", "dress", "sweater",
              "vest", "tank", "body", "torso", "main")
_BOTTOM_WORDS = ("pant", "trouser", "jean", "leg", "short", "skirt", "bottom")
_SKIP_WORDS = ("skin", "hair", "eye", "face", "teeth", "brow", "mouth", "lash", "head")


def tint_person(comp, entry, ch):
    """Clothing colour per slot: tops get `colour`, bottoms `colour_bottom`.

    Slots are classified by name; skin/hair/eyes are never tinted. A mesh with a
    single slot (CesiumMan) is tinted whole. Returns {slot: role}.
    """
    top = entry.get("colour")
    bottom = entry.get("colour_bottom") or top
    if not top:
        return {}
    names = [str(n).lower() for n in comp.get_material_slot_names()]
    roles = {}
    for i, m in enumerate(comp.get_materials()):
        if m is None:
            continue
        name = names[i] if i < len(names) else ""
        if len(names) <= 1:
            role = "top"
        elif any(w in name for w in _SKIP_WORDS):
            continue
        elif any(w in name for w in _BOTTOM_WORDS):
            role = "bottom"
        elif any(w in name for w in _TOP_WORDS):
            role = "top"
        else:
            continue
        mid = comp.create_dynamic_material_instance(i, m)
        if mid is None:
            continue
        colour = top if role == "top" else bottom
        for pname in ch.get("tint_params", ("BaseColorFactor",)):
            mid.set_vector_parameter_value(pname, unreal.LinearColor(*colour, 1.0))
        roles[name] = role
    return roles


BLEND_FRAMES = 2            # default; build_sequences uses round(BLEND_S * fps)
BLEND_S = 0.3               # idle<->walk crossfade


def _key_visibility(binding, visible, n_frames):
    """Hide a person before their first and after their last key.

    Without this, arrivals stand at the spawn point and leavers stand frozen at
    the exit for the rest of the render — people the GT says are not there.
    """
    if not visible:
        return
    first, last = int(visible[0]), int(visible[1])
    if first <= 0 and last >= n_frames - 1:
        return
    vt = binding.add_track(unreal.MovieSceneVisibilityTrack)
    vt.set_property_name_and_path("bHiddenInGame", "bHiddenInGame")
    sec = vt.add_section()
    sec.set_range(0, n_frames)
    ch = sec.get_all_channels()[0]
    # [V] The visibility track's bool channel is VISIBLE (true = shown), even
    # though it is bound to bHiddenInGame: keying "hidden=True" showed arrivals
    # clustered at spawn and hid everyone the GT had in view.
    ch.add_key(unreal.FrameNumber(0), first <= 0)
    if first > 0:
        ch.add_key(unreal.FrameNumber(first), True)
    if last < n_frames - 1:
        ch.add_key(unreal.FrameNumber(last + 1), False)


def _ease_in(sec, ticks):
    try:
        e = sec.get_editor_property("easing")
        e.set_editor_property("manual_ease_in", True)
        e.set_editor_property("manual_ease_in_duration", int(ticks))
        sec.set_editor_property("easing", e)
    except Exception as exc:                                    # noqa: BLE001
        unreal.log_warning(f"[survsim] ease-in not set: {exc}")


def set_play_rate(params, rate):
    """[V] 5.8: PlayRate is FMovieSceneTimeWarpVariant; use set_fixed_play_rate."""
    variant = params.get_editor_property("play_rate")
    variant.set_fixed_play_rate(float(rate))
    params.set_editor_property("play_rate", variant)
    return params.get_editor_property("play_rate").to_fixed_play_rate()


def key_camera(seq, cam, cam_binding, keys, spec):
    """Key a camera's live pose + zoom (PTZ) so pixels follow the GT camera."""
    tr = cam_binding.add_track(unreal.MovieScene3DTransformTrack)
    sec = tr.add_section()
    sec.set_range(0, max(k["frame"] for k in keys) + 1)
    ch = sec.get_all_channels()
    loc = cam.get_actor_location()
    comp = cam.get_cine_camera_component()
    fb = seq.add_possessable(comp)
    ftrack = fb.add_track(unreal.MovieSceneFloatTrack)
    ftrack.set_property_name_and_path("CurrentFocalLength", "CurrentFocalLength")
    fsec = ftrack.add_section()
    fsec.set_range(0, max(k["frame"] for k in keys) + 1)
    fch = fsec.get_all_channels()[0]
    lens = comp.get_editor_property("lens_settings")
    lens.set_editor_property("min_focal_length", 1.0)
    lens.set_editor_property("max_focal_length", 2000.0)
    comp.set_editor_property("lens_settings", lens)
    for k in keys:
        f = unreal.FrameNumber(int(k["frame"]))
        ch[0].add_key(f, loc.x)
        ch[1].add_key(f, loc.y)
        ch[2].add_key(f, loc.z)
        ch[4].add_key(f, float(k["pitch"]))
        ch[5].add_key(f, float(k["yaw"]))
        if k.get("hfov_deg"):
            fch.add_key(f, (36.0 / 2.0) / math.tan(math.radians(k["hfov_deg"]) / 2.0))
    for c in ch[6:9]:
        c.add_key(unreal.FrameNumber(0), 1.0)
    return len(keys)


def build_sequences(manifest, plan, people, cams, chars, n_frames, camera_ids=None,
                    camera_keys=None):
    """(Re)build one LevelSequence per camera; keeps assets, replaces tracks."""
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    fps = round(float(manifest.get("fps", 25)))
    global BLEND_FRAMES
    BLEND_FRAMES = max(2, round(BLEND_S * fps))
    seqs = {}
    rates = []
    for cid, cam in cams.items():
        if camera_ids and cid not in camera_ids:
            continue
        name = f"{SEQ_NAME}_{cid}"
        path = f"{SEQ_PACKAGE}/{name}"
        if _eal.does_asset_exist(path):
            _eal.delete_asset(path)
        seq = tools.create_asset(asset_name=name, package_path=SEQ_PACKAGE,
                                 asset_class=unreal.LevelSequence,
                                 factory=unreal.LevelSequenceFactoryNew())
        seq.set_display_rate(unreal.FrameRate(fps, 1))
        seq.set_playback_start(0)
        seq.set_playback_end(n_frames)
        animated = 0
        for entry in plan:
            ch = chars[entry["character"]]
            actor = people[entry["uid"]]
            b = seq.add_possessable(actor)
            track = b.add_track(unreal.MovieSceneSkeletalAnimationTrack)
            tick = seq.get_tick_resolution()
            tps = float(tick.numerator) / float(tick.denominator)
            for si, seg in enumerate(entry["segments"]):
                moving = seg["clip"] in ("walk", "run")
                walk = moving            # (kept: the stance/ease logic below keys off it)
                idle_clip = ch.get("idle")
                if seg["clip"] == "run" and ch.get("run"):
                    clip = ch["run"]
                elif moving:
                    clip = ch["walk"]
                else:                    # "idle" and "look" (the turn is on the yaw keys)
                    clip = idle_clip or ch["walk"]
                rate = seg["rate"] if moving else (1.0 if idle_clip else 0.001)
                sec = track.add_section()
                # Overlap 2 frames with the previous section and ease in, so
                # walk<->idle blends instead of popping.
                start = max(0, seg["start"] - (BLEND_FRAMES if si else 0))
                sec.set_range(start, seg["end"])
                params = sec.get_editor_property("params")
                params.set_editor_property("animation", _eal.load_asset(clip))
                if not walk and not idle_clip and ch.get("stance_s") is not None:
                    # No idle clip: hold the walk cycle at double-stance (feet
                    # together), measured by FK — never a frozen mid-stride.
                    params.set_editor_property("first_loop_start_frame_offset",
                                               unreal.FrameNumber(int(ch["stance_s"] * tps)))
                if not moving and idle_clip and seg.get("phase_s"):
                    # Per-person idle phase: a waiting group must not sway in sync.
                    params.set_editor_property("first_loop_start_frame_offset",
                                               unreal.FrameNumber(int(seg["phase_s"] * tps)))
                got = set_play_rate(params, rate)
                sec.set_editor_property("params", params)
                if si:
                    _ease_in(sec, int(BLEND_FRAMES * tps / fps))
                rates.append((rate, got))
            animated += 1 if track.get_sections() else 0
            tr = b.add_track(unreal.MovieScene3DTransformTrack)
            s = tr.add_section()
            s.set_range(0, n_frames)
            ch_ = s.get_all_channels()
            scale = entry["height_cm"] / ch["height_cm"]
            for key, yaw in zip(entry["keys"], entry["yaws"]):
                f = unreal.FrameNumber(int(key["frame"]))
                x, y, z = key["location"]
                ch_[0].add_key(f, float(x))
                ch_[1].add_key(f, float(y))
                ch_[2].add_key(f, float(z) + ch.get("z_offset_cm", 0.0) * scale)
                ch_[3].add_key(f, float(ch.get("roll_offset", 0.0)))
                ch_[4].add_key(f, float(ch.get("pitch_offset", 0.0)))
                ch_[5].add_key(f, float(yaw) + ch.get("yaw_offset", -90.0))
            for c in ch_[6:9]:
                c.add_key(unreal.FrameNumber(0), float(scale))
            _key_visibility(b, entry.get("visible"), n_frames)
        if animated != len(people):
            raise RuntimeError(f"{cid}: {animated}/{len(people)} people animated")
        cb = seq.add_possessable(cam)
        if camera_keys and camera_keys.get(cid):
            spec = next(c for c in manifest["cameras"] if c["id"] == cid)
            key_camera(seq, cam, cb, camera_keys[cid], spec)
        cut = seq.add_track(unreal.MovieSceneCameraCutTrack)
        cs = cut.add_section()
        cs.set_range(0, n_frames)
        bid = unreal.MovieSceneObjectBindingID()
        bid.set_editor_property("guid", cb.get_id())
        cs.set_camera_binding_id(bid)
        _eal.save_asset(seq.get_path_name())
        seqs[cid] = seq
    bad = [r for r in rates if abs(r[0] - r[1]) > 1e-3]
    return seqs, {"rate_sections": len(rates), "rate_mismatch": bad[:5]}


ID_MASK_MAT = "/Game/SurvSim/M_SurvIdMask"
# Per-person ID mask (CLAUDE.md principle 1: boxes from rendered pixels). Each
# person's components get CustomDepth stencil k = 1..215; a post-process
# material replacing the tonemapper writes k as base-6 digits in R, G, B
# (level/5), so the ID survives an sRGB or a linear PNG write alike. Decoded on
# the host by survsim.gt.mask_boxes.
ID_MASK_CODE = ("float s = round(S.r);"
                "return float3(fmod(s, 6.0), fmod(floor(s / 6.0), 6.0), floor(s / 36.0)) / 5.0;")
# [V] 2026-09-28 site 09 A/B (docs/BUG_gt_visibility_fences.md "Verification"). The
# CustomDepth pass only rasterises primitives with render_custom_depth set - the
# people - so a fence, wall or hedge (which never writes custom depth) cannot
# hide a person's stencil: the pass shows people *through* static geometry.
# Person-on-person occlusion is correct (both write custom depth). Zero the ID
# where the person's CustomDepth lies behind the scene depth, i.e. where
# something that is not a person was drawn in front of them.
ID_MASK_DEPTH_TEST = True
ID_MASK_CODE_DEPTH = ("float s = round(S.r);"
                      "if (CD.r > SD.r * 1.005 + 2.0) { s = 0.0; }"
                      "return float3(fmod(s, 6.0), fmod(floor(s / 6.0), 6.0), floor(s / 36.0)) / 5.0;")


def ensure_id_mask_material():
    """The post-process material that renders CustomStencil as an ID colour."""
    mel = unreal.MaterialEditingLibrary
    if _eal.does_asset_exist(ID_MASK_MAT):
        _eal.delete_asset(ID_MASK_MAT)
    pkg, name = ID_MASK_MAT.rsplit("/", 1)
    mat = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        name, pkg, unreal.Material, unreal.MaterialFactoryNew())
    mat.set_editor_property("material_domain", unreal.MaterialDomain.MD_POST_PROCESS)
    mat.set_editor_property("blendable_location", unreal.BlendableLocation.BL_REPLACING_TONEMAPPER)
    st = mel.create_material_expression(mat, unreal.MaterialExpressionSceneTexture, -600, 0)
    st.set_editor_property("scene_texture_id", unreal.SceneTextureId.PPI_CUSTOM_STENCIL)
    cu = mel.create_material_expression(mat, unreal.MaterialExpressionCustom, -300, 0)
    cu.set_editor_property("code", ID_MASK_CODE_DEPTH if ID_MASK_DEPTH_TEST else ID_MASK_CODE)
    cu.set_editor_property("output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT3)
    names = ["S", "CD", "SD"] if ID_MASK_DEPTH_TEST else ["S"]
    inps = []
    for nm in names:
        inp = unreal.CustomInput()
        inp.set_editor_property("input_name", nm)
        inps.append(inp)
    cu.set_editor_property("inputs", inps)
    mel.connect_material_expressions(st, "Color", cu, "S")
    if ID_MASK_DEPTH_TEST:
        for i, (nm, tex) in enumerate((("CD", unreal.SceneTextureId.PPI_CUSTOM_DEPTH),
                                       ("SD", unreal.SceneTextureId.PPI_SCENE_DEPTH))):
            e = mel.create_material_expression(mat, unreal.MaterialExpressionSceneTexture, -600, 200 + 200 * i)
            e.set_editor_property("scene_texture_id", tex)
            mel.connect_material_expressions(e, "Color", cu, nm)
    mel.connect_material_property(cu, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    mel.recompile_material(mat)
    _eal.save_asset(ID_MASK_MAT)
    return mat


def tag_people_stencil(people):
    """people: {uid: actor} in plan order -> {uid: stencil id}; every primitive
    component (body, face, garments) of a person carries that person's ID."""
    unreal.SystemLibrary.execute_console_command(None, "r.CustomDepth 3")
    ids = {}
    for k, (uid, actor) in enumerate(people.items(), start=1):
        if k > 215:
            raise RuntimeError("id mask encodes at most 215 people")
        for comp in actor.get_components_by_class(unreal.PrimitiveComponent):
            comp.set_editor_property("render_custom_depth", True)
            comp.set_editor_property("custom_depth_stencil_value", k)
            comp.set_editor_property("custom_depth_stencil_write_mask",
                                     unreal.RendererStencilMask.ERSM_DEFAULT)
        ids[uid] = k
    return ids


def _queue(seqs, out_root, resolution, warmup, resolutions):
    sub = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    q = sub.get_queue()
    for job in list(q.get_jobs()):
        q.delete_job(job)
    for cid, seq in seqs.items():
        job = q.allocate_new_job(unreal.MoviePipelineExecutorJob)
        job.job_name = cid
        job.sequence = unreal.SoftObjectPath(seq.get_path_name())
        job.map = unreal.SoftObjectPath(MAP)
        cfg = job.get_configuration()
        cfg.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
        cfg.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
        cfg.find_or_add_setting_by_class(unreal.MoviePipelineCameraSetting)
        o = cfg.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
        o.output_directory = unreal.DirectoryPath(f"{out_root}/{cid}")
        w, h = (resolutions or {}).get(cid, resolution)
        o.output_resolution = unreal.IntPoint(int(w), int(h))
        o.file_name_format = "{frame_number}"
        o.override_existing_output = True
        aa = cfg.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
        aa.spatial_sample_count = 1
        aa.temporal_sample_count = 1
        aa.engine_warm_up_count = warmup
        aa.render_warm_up_frames = True
        cfg.find_or_add_setting_by_class(unreal.MoviePipelineGameOverrideSetting)
        cfg.initialize_transient_settings()
    return sub


def _run_queue(sub, done):
    ex = unreal.MoviePipelinePIEExecutor()
    ex.on_executor_finished_delegate.add_callable_unique(done)
    # Keep the executor alive until it finishes.
    _KEEP.append((ex, done))
    sub.render_queue_with_executor_instance(ex)


def render(seqs, out_root, finish, resolution=(1280, 720), warmup=32, resolutions=None,
           id_mask=False, mask_root=None):
    """MRQ one job per camera (camera-cut per sequence). Async -> finish().

    resolutions: {camera_id: (w, h)} — MUST match the GT's image size.
    id_mask: after the beauty pass, render the same sequences again with an
    unbound post-process volume carrying ID_MASK_MAT into <mask_root>/<cam>/.
    [X] 2026-09-26: the ID material as an MRQ `additional_post_process_materials`
    pass left "Not all frames were fully submitted" and wrote NO frames at all
    (beauty included); a second plain render is reliable and cheap.
    """
    unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level()
    sub = _queue(seqs, out_root, resolution, warmup, resolutions)

    def done(executor, success):
        if not id_mask:
            finish({"render_success": bool(success), "out": out_root})
            return
        mat = ensure_id_mask_material()
        ppv = actors().spawn_actor_from_class(unreal.PostProcessVolume, unreal.Vector(0, 0, 0))
        ppv.set_actor_label("IdMaskPPV")
        ppv.set_folder_path("Tmp")
        ppv.set_editor_property("unbound", True)
        ppv.set_editor_property("priority", 1000.0)
        ps = ppv.get_editor_property("settings")
        ps.set_editor_property("weighted_blendables", unreal.WeightedBlendables(
            array=[unreal.WeightedBlendable(weight=1.0, object=mat)]))
        ppv.set_editor_property("settings", ps)
        unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level()
        sub2 = _queue(seqs, mask_root, resolution, warmup, resolutions)

        def done_mask(executor2, success2):
            actors().destroy_actor(ppv)
            unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level()
            finish({"render_success": bool(success) and bool(success2), "out": out_root,
                    "mask_out": mask_root})

        _run_queue(sub2, done_mask)

    _run_queue(sub, done)


_KEEP = []


# --- plan-driven environments (survsim/export/env_plan.py) -----------------------

def _facade(L, item):
    style = item.get("style", "mixed")
    pool = {"glass": L["facade_glass"], "warehouse": L["facade_warehouse"]}.get(style, L["facades"])
    return pool[int(item.get("facade", 0)) % len(pool)]


def build_from_plan(p, mats=None):
    """Place every item of an environment plan into the "Env" group.

    The vocabulary is primitives + the existing lamp/tree/car builders, so a
    new layout is a change to the pure-Python planner, not to this file.
    """
    L = mats or M.library(int(p.get("seed", 0)))
    g = "Env"
    half = float(p["half_cm"])
    gr = p["ground"]
    far = 20000.0
    box(g, "GroundFar", (0, 0, -2), (far * 2, far * 2, 4), L[gr["far"]])
    box(g, "GroundMain", (0, 0, 0.5), (2 * half + 1400, 2 * half + 1400, 1), L[gr["main"]])
    for i, (cx, cy, sx, sy) in enumerate([(0, half + 60, 2 * half + 240, 120),
                                          (0, -half - 60, 2 * half + 240, 120),
                                          (half + 60, 0, 120, 2 * half),
                                          (-half - 60, 0, 120, 2 * half)]):
        box(g, f"Rim_{i}", (cx, cy, 0.8), (sx, sy, 1.2), L[gr["rim"]])
    lamps, counts = [], {}
    for n, it in enumerate(p["items"]):
        k, x, y, yaw = it["kind"], it["x"], it["y"], it.get("yaw", 0.0)
        lab = f"{k}_{n}"
        counts[k] = counts.get(k, 0) + 1
        if k == "building":
            sx, sy, sz = it["size"]
            box(g, lab, (x, y, sz / 2), (sx, sy, sz), _facade(L, it), yaw=yaw)
            box(g, lab + "_roof", (x, y, sz + 30), (sx + 20, sy + 20, 60), L["concrete"], yaw=yaw)
        elif k == "road":
            # Road runs along the item's yaw (0: along X, 90: along Y).
            w = it.get("width", 700.0)
            c, sn = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            box(g, lab, (x, y, 0.6), (far, w, 0.6), L["asphalt"], yaw=yaw)
            for tag, off, width, mat in (("_kerbA", -w / 2 - 15, 30, "kerb"),
                                         ("_kerbB", w / 2 + 15, 30, "kerb")):
                box(g, lab + tag, (x - sn * off, y + c * off, 7), (far, width, 14), L[mat], yaw=yaw)
        elif k == "lane_line":
            box(g, lab, (x, y, 1.0), (it.get("length", 450.0), 12, 0.4), L["paint_white"], yaw=yaw)
        elif k == "bay_line":
            box(g, lab, (x, y, 1.0), (it.get("length", 500.0), 10, 0.4), L["paint_white"], yaw=yaw)
        elif k == "platform_edge":
            box(g, lab, (x, y, 50), (it.get("length", 4000.0), 300, 100), L["concrete"])
            box(g, lab + "_line", (x, y - 120, 100.5), (it.get("length", 4000.0), 30, 0.4),
                L["paint_yellow"])
        elif k == "car":
            _car(g, lab, (x, y), yaw, L["car_paints"][int(it.get("paint", 0)) % len(L["car_paints"])],
                 L, [], random.Random(lab))
        elif k == "lamp":
            lamps.append(_lamp(g, lab, x, y, yaw, L))
        elif k == "floodlight":
            box(g, lab, (x, y, 450), (20, 20, 900), L["metal_dark"], shape="Cylinder")
            box(g, lab + "_head", (x, y, 900), (80, 40, 30), L["lamp_lens"])
            lamps.append((x, y))
        elif k == "tree":
            _tree(g, lab, x, y, L, random.Random(lab), scale=float(it.get("scale", 1.0)))
        elif k == "ticket_machine":
            box(g, lab, (x, y, 90), (60, 50, 180), L["paint_yellow"], yaw=yaw)
            box(g, lab + "_screen", (x, y, 140), (62, 20, 30), L["glass"], yaw=yaw)
        elif k == "bench":
            box(g, lab, (x, y, 45), (180, 45, 6), L["wood"], yaw=yaw)
            box(g, lab + "_back", (x, y, 75), (180, 6, 40), L["wood"], yaw=yaw)
            box(g, lab + "_legs", (x, y, 22), (150, 40, 44), L["metal_dark"], yaw=yaw)
        elif k == "bin":
            box(g, lab, (x, y, 50), (55, 55, 100), L["metal_green"], shape="Cylinder")
        elif k in ("bollards", "bollard_row"):
            for j in range(4):
                off = j * 150 - 225
                c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
                box(g, f"{lab}_{j}", (x + c * off, y + s * off, 45), (20, 20, 90),
                    L["metal_dark"], shape="Cylinder")
        elif k == "planter":
            box(g, lab, (x, y, 40), (160, 160, 80), L["concrete"])
            box(g, lab + "_hedge", (x, y, 105), (140, 140, 60), L["hedge"], shape="Sphere")
        elif k == "pillar":
            box(g, lab, (x, y, 200), (60, 60, 400), L["concrete"])
            box(g, lab + "_band", (x, y, 60), (62, 62, 40), L["paint_yellow"])
        elif k == "wall":
            sx, sy, sz = it["size"]
            box(g, lab, (x, y, sz / 2), (sx, sy, sz), L["walls"][0], yaw=yaw)
        elif k == "barrier":
            box(g, lab, (x, y, 100), (30, 30, 100), L["metal_dark"])
            box(g, lab + "_arm", (x, y + 250, 95), (8, 500, 8), L["paint_white"], yaw=yaw)
        elif k == "fence":
            ln = it.get("length", 300.0)
            box(g, lab, (x, y, 125), (ln, 4, 250), L["metal_dark"], yaw=yaw)
            box(g, lab + "_post", (x - ln / 2 * math.cos(math.radians(yaw)),
                                   y - ln / 2 * math.sin(math.radians(yaw)), 130),
                (10, 10, 260), L["metal_green"])
        elif k == "gatehouse":
            box(g, lab, (x, y, 150), (400, 300, 300), L["facades"][4])
            box(g, lab + "_roof", (x, y, 310), (440, 340, 20), L["metal_dark"])
        elif k == "container":
            paint = L["car_paints"][int(it.get("paint", 0)) % len(L["car_paints"])]
            box(g, lab, (x, y, 130), (1220, 244, 260), paint, yaw=yaw)
        elif k == "stage":
            sx, sy, sz = it.get("size", [1200, 600, 120])
            box(g, lab, (x, y, sz / 2), (sx, sy, sz), L["metal_dark"], yaw=yaw)
            for dx in (-1, 1):
                box(g, f"{lab}_truss{dx}", (x + dx * sx / 2, y, 350), (30, 30, 700), L["metal_dark"])
            box(g, lab + "_top", (x, y, 700), (sx, 30, 30), L["metal_dark"])
        elif k == "kiosk":
            box(g, lab, (x, y, 125), (300, 250, 250), L["car_paints"][n % 7])
            box(g, lab + "_roof", (x, y, 265), (360, 310, 30), L["paint_white"])
        elif k == "canopy":
            sx, sy, sz = it["size"]
            box(g, lab, (x, y, sz), (sx, sy, 40), L["concrete"])
            for px in range(-3, 4):
                box(g, f"{lab}_col{px}", (x + px * sx / 7, y, sz / 2), (40, 40, sz), L["metal_dark"])
        elif k == "gateline":
            box(g, lab, (x, y, 50), (30, 150, 100), L["metal_dark"])
        elif k == "board":
            box(g, lab, (x, y, 300), (300, 20, 120), L["glass"])
        counts[k] = counts.get(k, 0)
    M.set_wet(float(p.get("weather", {}).get("wet", 0.0)))
    return {"counts": counts, "lamps": lamps}


def frustum_violations(manifest, near=1000.0, half_fov=60.0, min_height=30.0):
    """Static Env actors whose bounds enter any camera's first `near` cm of view.

    Box test: sample the actor's world-bounds footprint on a grid and check each
    sample against the camera's horizontal view cone. Flat actors (ground,
    markings: bounds height < min_height) are ignored.
    """
    bad = []
    cams = [(c["id"], c["location"], c["rotation"][1]) for c in manifest.get("cameras", [])]
    for a in group_actors("Env"):
        o, e = a.get_actor_bounds(False)
        if 2 * e.z < min_height or e.x > 5000 or e.y > 5000:
            continue
        pts = [(o.x + e.x * i / 3.0, o.y + e.y * j / 3.0)
               for i in range(-3, 4) for j in range(-3, 4)]
        for cid, loc, yaw in cams:
            fx, fy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            for px, py in pts:
                dx, dy = px - loc[0], py - loc[1]
                d = math.hypot(dx, dy)
                if d < 1e-6 or (d < near and (dx * fx + dy * fy) / d > math.cos(math.radians(half_fov))):
                    bad.append({"camera": cid, "actor": a.get_actor_label(),
                                "dist_cm": round(d, 1)})
                    break
    return bad


# --- plan v2 (survsim/generator/layout.py via env_plan.plan_from_layout) ---------

DECAL_MASTER = "M_SurvDecal_v2"
_DECAL = {  # tint (linear), opacity, roughness: darker and higher-contrast than
    # v1, whose marks all but vanished on the textured asphalt (batch_v5 [X])
    "oil": ((0.004, 0.004, 0.005), 0.95, 0.12), "dirt": ((0.03, 0.025, 0.018), 0.8, 1.0),
    "gum": ((0.012, 0.012, 0.014), 0.9, 0.7), "crack": ((0.004, 0.004, 0.004), 0.95, 1.0),
    "tyre": ((0.006, 0.006, 0.006), 0.75, 0.55), "puddle": ((0.004, 0.005, 0.007), 0.85, 0.02),
    "leaves": ((0.12, 0.055, 0.018), 0.9, 0.9)}


def ensure_decal_master():
    """Deferred-decal master: soft noisy blob, Tint/Opacity/Rough parameters.

    Refuses to return a material that compiled to 0 pixel instructions (the
    failure mode that made the PBR master render as the default grid).
    """
    mel, path = unreal.MaterialEditingLibrary, f"{M.MAT_DIR}/{DECAL_MASTER}"
    if _eal.does_asset_exist(path):
        mat = _eal.load_asset(path)
    else:
        mat = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            DECAL_MASTER, M.MAT_DIR, unreal.Material, unreal.MaterialFactoryNew())
        mat.set_editor_property("material_domain", unreal.MaterialDomain.MD_DEFERRED_DECAL)
        mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_TRANSLUCENT)
        E = unreal
        tc = mel.create_material_expression(mat, E.MaterialExpressionTextureCoordinate, -900, 0)
        ctr = mel.create_material_expression(mat, E.MaterialExpressionConstant2Vector, -900, 120)
        ctr.set_editor_property("r", 0.5)
        ctr.set_editor_property("g", 0.5)
        sm = mel.create_material_expression(mat, E.MaterialExpressionSphereMask, -650, 40)
        sm.set_editor_property("attenuation_radius", 0.5)
        sm.set_editor_property("hardness_percent", 35.0)
        mel.connect_material_expressions(tc, "", sm, "A")
        mel.connect_material_expressions(ctr, "", sm, "B")
        wp = mel.create_material_expression(mat, E.MaterialExpressionWorldPosition, -900, 260)
        nz = mel.create_material_expression(mat, E.MaterialExpressionNoise, -650, 260)
        nz.set_editor_property("scale", 0.012)
        try:     # [X] v1: the default noise function reads as square blotches
            nz.set_editor_property("noise_function",
                                   unreal.NoiseFunction.NOISEFUNCTION_GRADIENT_ALU)
            nz.set_editor_property("levels", 4)
        except Exception:                                     # noqa: BLE001, S110
            pass
        nz.set_editor_property("output_min", 0.35)
        nz.set_editor_property("output_max", 1.0)
        mel.connect_material_expressions(wp, "", nz, "Position")
        op = mel.create_material_expression(mat, E.MaterialExpressionScalarParameter, -650, 420)
        op.set_editor_property("parameter_name", "Opacity")
        op.set_editor_property("default_value", 0.6)
        m1 = mel.create_material_expression(mat, E.MaterialExpressionMultiply, -400, 100)
        mel.connect_material_expressions(sm, "", m1, "A")
        mel.connect_material_expressions(nz, "", m1, "B")
        m2 = mel.create_material_expression(mat, E.MaterialExpressionMultiply, -250, 160)
        mel.connect_material_expressions(m1, "", m2, "A")
        mel.connect_material_expressions(op, "", m2, "B")
        mel.connect_material_property(m2, "", unreal.MaterialProperty.MP_OPACITY)
        tint = mel.create_material_expression(mat, E.MaterialExpressionVectorParameter, -400, -150)
        tint.set_editor_property("parameter_name", "Tint")
        tint.set_editor_property("default_value", unreal.LinearColor(0.02, 0.02, 0.02, 1.0))
        mel.connect_material_property(tint, "", unreal.MaterialProperty.MP_BASE_COLOR)
        rough = mel.create_material_expression(mat, E.MaterialExpressionScalarParameter, -400, 300)
        rough.set_editor_property("parameter_name", "Rough")
        rough.set_editor_property("default_value", 0.8)
        mel.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)
        mel.recompile_material(mat)
        _eal.save_asset(mat.get_path_name())
    n = int(mel.get_statistics(mat).get_editor_property("num_pixel_shader_instructions"))
    if n <= 0:
        raise RuntimeError(f"{DECAL_MASTER} compiled to {n} pixel instructions")
    return mat


def decal_instances():
    mel = unreal.MaterialEditingLibrary
    master = ensure_decal_master()
    out = {}
    for name, (tint, op, rough) in _DECAL.items():
        path = f"{M.MAT_DIR}/MI_Decal_{name}"
        mic = _eal.load_asset(path) if _eal.does_asset_exist(path) else None
        if mic is None:
            mic = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
                f"MI_Decal_{name}", M.MAT_DIR, unreal.MaterialInstanceConstant,
                unreal.MaterialInstanceConstantFactoryNew())
        if mic.get_editor_property("parent") != master:
            mel.set_material_instance_parent(mic, master)
        mel.set_material_instance_vector_parameter_value(mic, "Tint", unreal.LinearColor(*tint, 1.0))
        mel.set_material_instance_scalar_parameter_value(mic, "Opacity", op)
        mel.set_material_instance_scalar_parameter_value(mic, "Rough", rough)
        mel.update_material_instance(mic)
        out[name] = mic
    return out


def _decal(label, x, y, yaw, w, d, mic):
    a = actors().spawn_actor_from_class(unreal.DecalActor, unreal.Vector(x, y, 5.0),
                                        unreal.Rotator(pitch=-90.0, yaw=yaw, roll=0.0))
    comp = a.get_component_by_class(unreal.DecalComponent)
    comp.set_editor_property("decal_size", unreal.Vector(12.0, max(10.0, d / 2), max(10.0, w / 2)))
    comp.set_decal_material(mic)
    a.set_actor_label(label)
    a.set_folder_path("Decals")
    return a


_COLOUR = {"red": 3, "blue": 4, "navy": 4, "white": 5, "black": 1, "grey": 2}


def _paint(L, key):
    if key == "yellow":
        return L["paint_yellow"]
    if key == "green":
        return L["metal_green"]
    if isinstance(key, int):
        pool = L["car_paints"] + [L["paint_yellow"]]
        return pool[key % len(pool)]
    return L["car_paints"][_COLOUR.get(key, 5)]


def _asset_spec(name):
    for a in load_index():
        if a.get("name") == name:
            return {"path": os.path.join(os.path.expanduser("~/data/assets"), a["path"])}
    return None


def _place_asset(g, lab, it, x, y, yaw):
    """A CC0 glb from the asset index, uniformly scaled to the item's height."""
    spec = _asset_spec(it.get("asset", ""))
    mesh = import_static(spec) if spec else None
    if mesh is None:
        return None
    a = actors().spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(x, y, 0),
                                        unreal.Rotator(pitch=0, yaw=yaw, roll=0))
    a.static_mesh_component.set_static_mesh(mesh)
    if it.get("tint_mat") is not None:
        # [X] 2026-09-26: Kenney kits share one colormap texture sampled by
        # UV; on these meshes it rendered as a red/white/blue flag pattern.
        # A plain painted material per asset instead.
        for i in range(a.static_mesh_component.get_num_materials()):
            a.static_mesh_component.set_material(i, it["tint_mat"])
    o, e = a.get_actor_bounds(False)
    want = float(it["size"][2])
    if e.z > 0.1:
        s = want / (2 * e.z)
        a.set_actor_scale3d(unreal.Vector(s, s, s))
        o, e = a.get_actor_bounds(False)
        a.set_actor_location(unreal.Vector(x, y, -(o.z - e.z)), False, False)
    a.set_actor_label(lab)
    a.set_folder_path(g)
    return a


def build_from_plan_v2(p, mats=None):
    """Build a v2 plan: site structure, kerbs/road paint from geometry, decals, wet."""
    L = mats or M.library(int(p.get("seed", 0)))
    g = "Env"
    clear("Decals")
    half = float(p["half_cm"])
    gr = p["ground"]
    far = 25000.0
    box(g, "GroundFar", (0, 0, -2), (far * 2, far * 2, 4), L[gr["far"]])
    box(g, "GroundMain", (0, 0, 0.5), (2 * half + 2400, 2 * half + 2400, 1), L[gr["main"]])
    for i, (cx, cy, sx, sy) in enumerate([(0, half + 60, 2 * half + 240, 120),
                                          (0, -half - 60, 2 * half + 240, 120),
                                          (half + 60, 0, 120, 2 * half),
                                          (-half - 60, 0, 120, 2 * half)]):
        box(g, f"Rim_{i}", (cx, cy, 0.8), (sx, sy, 1.2), L[gr["rim"]])
    decals = {}
    try:
        decals = decal_instances()
    except Exception as exc:                                  # noqa: BLE001
        unreal.log_warning(f"[survsim] decals unavailable: {exc}")
    lamps, counts, failed = [], {}, {}
    for n, it in enumerate(p["items"]):
        k, x, y, yaw = it["kind"], float(it["x"]), float(it["y"]), float(it.get("yaw", 0.0))
        w, d, h = (float(v) for v in it["size"])
        lab = f"{k}_{n}"
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))

        def at(u, v, _x=x, _y=y, _c=c, _s=s):
            return (_x + _c * u - _s * v, _y + _s * u + _c * v)
        counts[k] = counts.get(k, 0) + 1
        if k == "block":
            st = it.get("style", "mixed")
            mat = (L["facade_glass"][0] if st == "glass" else
                   L["facade_warehouse"][int(it.get("facade", 0)) % 2] if st == "warehouse" else
                   L["concrete"] if st == "deck" else
                   L["facades"][int(it.get("facade", 0)) % len(L["facades"])])
            box(g, lab, (x, y, h / 2), (w, d, h), mat, yaw=yaw)
            box(g, lab + "_roof", (x, y, h + 25), (w + 30, d + 30, 50), L["concrete"], yaw=yaw)
            if st == "deck":
                for f in range(1, int(h // 300) + 1):
                    box(g, f"{lab}_deck{f}", (x, y, f * 300), (w + 40, d + 40, 35), L["concrete"], yaw=yaw)
        elif k == "road":
            box(g, lab, (x, y, 0.8), (w, d, 0.6), L["asphalt"], yaw=yaw)
        elif k == "kerb":
            box(g, lab, (x, y, h / 2), (w, d, h), L["kerb"], yaw=yaw)
        elif k == "pavement":
            box(g, lab, (x, y, max(h, 2.0) / 2), (w, d, max(h, 2.0)),
                L.get(it.get("material", "slabs"), L["slabs"]), yaw=yaw)
        elif k == "grass":
            box(g, lab, (x, y, max(1.0, h / 2)), (w, d, max(2.0, h)), L["grass"], yaw=yaw)
        elif k in ("lane_line", "bay_line"):
            box(g, lab, (x, y, 1.3), (w, max(d, 10.0), 0.4), L["paint_white"], yaw=yaw)
        elif k == "yard_line":
            box(g, lab, (x, y, 1.3), (w, max(d, 10.0), 0.4), _paint(L, it.get("colour", "yellow")), yaw=yaw)
        elif k == "tactile":
            box(g, lab, (x, y, 1.2), (w, d, 0.6), L["paint_yellow"], yaw=yaw)
        elif k == "crossing":
            # Zebra: 50 cm bars along the road direction, repeated across its width.
            nb = int(d // 100)
            for b in range(nb):
                v = -d / 2 + 50 + b * 100
                box(g, f"{lab}_{b}", (*at(0, v), 1.4), (w, 50, 0.4), L["paint_white"], yaw=yaw)
        elif k == "hatch":
            nb = int(w // 120)
            for b in range(nb):
                box(g, f"{lab}_{b}", (*at(-w / 2 + 60 + b * 120, 0), 1.3), (30, d * 1.2, 0.4),
                    L["paint_yellow"], yaw=yaw + 45)
        elif k == "decal":
            mic = decals.get(it.get("decal", "dirt"))
            if mic is not None:
                _decal(lab, x, y, yaw, w, d, mic)
        elif k == "car":
            paint = L["car_paints"][1] if it.get("taxi") else \
                L["car_paints"][int(it.get("paint", 0)) % len(L["car_paints"])]
            _car(g, lab, (x, y), yaw, paint, L, [], random.Random(lab))
        elif k == "lamp":
            lamps.append(_lamp(g, lab, x, y, yaw, L))
        elif k == "floodlight":
            box(g, lab, (x, y, h / 2), (24, 24, h), L["metal_dark"], shape="Cylinder")
            box(g, lab + "_head", (x, y, h), (120, 50, 40), L["lamp_lens"])
            lamps.append((x, y))
        elif k == "tree":
            _tree(g, lab, x, y, L, random.Random(lab), scale=max(0.6, h / 600.0))
        elif k == "bench":
            box(g, lab, (x, y, 45), (w, d, 6), L["wood"], yaw=yaw)
            box(g, lab + "_back", (*at(0, d / 2), 75), (w, 6, 40), L["wood"], yaw=yaw)
            box(g, lab + "_legs", (x, y, 22), (w * 0.8, d * 0.8, 44), L["metal_dark"], yaw=yaw)
        elif k == "bin":
            box(g, lab, (x, y, 50), (55, 55, 100), L["metal_green"], shape="Cylinder")
        elif k == "bollards":
            for j in range(4):
                box(g, f"{lab}_{j}", (*at(-w / 2 + w * (j + 0.5) / 4, 0), 45), (20, 20, 90),
                    L["metal_dark"], shape="Cylinder")
        elif k == "planter":
            box(g, lab, (x, y, 40), (w, d, 80), L["concrete"], yaw=yaw)
            box(g, lab + "_hedge", (x, y, 105), (w * 0.88, d * 0.88, 60), L["hedge"], shape="Sphere")
        elif k == "cycle_rack":
            nh = max(2, int(w // 80))
            for j in range(nh):
                box(g, f"{lab}_{j}", (*at(-w / 2 + (j + 0.5) * w / nh, 0), 45), (6, 70, 90),
                    L["metal_dark"], yaw=yaw)
        elif k == "bus_shelter":
            box(g, lab + "_roof", (x, y, 250), (w + 20, d + 20, 10), L["metal_dark"], yaw=yaw)
            box(g, lab + "_back", (*at(0, d / 2), 130), (w, 4, 200), L["glass"], yaw=yaw)
            for u in (-w / 2, w / 2):
                box(g, f"{lab}_side{u}", (*at(u, 0), 130), (4, d, 200), L["glass"], yaw=yaw)
                box(g, f"{lab}_post{u}", (*at(u, d / 2), 125), (8, 8, 250), L["metal_dark"], yaw=yaw)
            box(g, lab + "_ad", (*at(w / 2, -d / 4), 120), (10, d / 2, 170),
                _paint(L, "white"), yaw=yaw)
        elif k == "bin_store":
            for u, v, ww, dd in ((0, d / 2, w, 20), (-w / 2, 0, 20, d), (w / 2, 0, 20, d)):
                box(g, f"{lab}_{u}_{v}", (*at(u, v), h / 2), (ww, dd, h), L["walls"][0], yaw=yaw)
            box(g, lab + "_gate", (*at(0, -d / 2), h / 2 - 10), (w * 0.9, 6, h - 20), L["wood"], yaw=yaw)
            box(g, lab + "_wheelie", (*at(-w / 4, 0), 55), (70, 80, 110), L["metal_green"], yaw=yaw)
        elif k == "flagpole":
            box(g, lab, (x, y, h / 2), (12, 12, h), L["paint_white"], shape="Cylinder")
            box(g, lab + "_flag", (*at(60, 0), h - 60), (120, 3, 80), _paint(L, it.get("colour", "blue")), yaw=yaw)
        elif k in ("sign", "board"):
            ph = h if k == "sign" else 320.0
            if it.get("mount") != "facade":
                box(g, lab + "_pole", (x, y, ph / 2), (8, 8, ph), L["metal_dark"], shape="Cylinder")
            box(g, lab, (x, y, ph - 40), (max(w, 40), max(d, 6), 70 if k == "sign" else 110),
                _paint(L, it.get("colour", "white")) if k == "sign" else L["glass"], yaw=yaw)
        elif k == "shop_sign":
            box(g, lab, (x, y, float(it.get("z", 3.4)) * 100), (w, max(d, 12), h),
                _paint(L, int(it.get("colour", 0))), yaw=yaw)
        elif k == "awning":
            box(g, lab, (x, y, float(it.get("z", 2.8)) * 100), (w, d, 12),
                _paint(L, int(it.get("colour", 0))), yaw=yaw)
        elif k == "canopy":
            box(g, lab, (x, y, h), (w, d, 30), L["metal_dark"] if it.get("flat_roof") else L["concrete"], yaw=yaw)
            ncol = max(2, int(w // 600) + 1)
            for j in range(ncol):
                u = -w / 2 + 20 + j * (w - 40) / (ncol - 1)
                box(g, f"{lab}_col{j}", (*at(u, -d / 2 + 20), h / 2), (25, 25, h), L["metal_dark"], yaw=yaw)
        elif k == "ticket_gates":
            ng = int(it.get("n", 6))
            for j in range(ng + 1):
                box(g, f"{lab}_{j}", (*at(-w / 2 + j * w / ng, 0), 55), (22, d, 110), L["metal_dark"], yaw=yaw)
                if j < ng:
                    box(g, f"{lab}_p{j}", (*at(-w / 2 + (j + 0.5) * w / ng, 0), 80), (w / ng - 30, 3, 50),
                        L["glass"], yaw=yaw)
        elif k == "ticket_machine":
            box(g, lab, (x, y, 90), (w, d, 180), L["paint_yellow"], yaw=yaw)
            box(g, lab + "_screen", (*at(0, -d / 2), 140), (w * 0.8, 4, 30), L["glass"], yaw=yaw)
        elif k == "barrier":
            box(g, lab, (x, y, 55), (30, 30, 110), L["metal_dark"])
            arm = float(it.get("arm", 4.0)) * 100
            box(g, lab + "_arm", (*at(0, arm / 2), 100), (8, arm, 8), L["paint_white"], yaw=yaw)
        elif k == "fence_run":
            st = it.get("style", "mesh")
            if st == "palisade":
                for j in range(int(w // 20)):
                    box(g, f"{lab}_{j}", (*at(-w / 2 + 10 + j * 20, 0), h / 2), (7, 3, h), L["metal_green"], yaw=yaw)
                box(g, lab + "_rail", (x, y, h * 0.8), (w, 6, 6), L["metal_green"], yaw=yaw)
            else:
                box(g, lab, (x, y, h / 2), (w, 2, h), L["metal_dark"] if st == "mesh" else L["paint_white"], yaw=yaw)
                box(g, lab + "_post", (*at(-w / 2, 0), h / 2 + 5), (8, 8, h + 10), L["metal_dark"])
        elif k == "wall_run":
            box(g, lab, (x, y, h / 2), (w, max(d, 20), h), L["walls"][0], yaw=yaw)
            box(g, lab + "_cope", (x, y, h + 4), (w, max(d, 20) + 6, 8), L["concrete"], yaw=yaw)
        elif k == "hedge":
            box(g, lab, (x, y, h / 2), (w, d, h), L["hedge"], yaw=yaw)
        elif k == "gatehouse":
            box(g, lab, (x, y, h / 2), (w, d, h), L["facades"][4], yaw=yaw)
            box(g, lab + "_roof", (x, y, h + 10), (w + 40, d + 40, 20), L["metal_dark"], yaw=yaw)
            box(g, lab + "_win", (*at(0, -d / 2), 160), (w * 0.7, 4, 90), L["glass"], yaw=yaw)
        elif k in ("container", "trailer"):
            paint = L["car_paints"][int(it.get("paint", 0)) % len(L["car_paints"])]
            z0 = 260 if it.get("stacked") else (110 if k == "trailer" else 0)
            hh = 260 if k == "container" else h - 110
            box(g, lab, (x, y, z0 + hh / 2), (w, d, hh), paint, yaw=yaw)
            if k == "trailer":
                for u in (-w / 2 + 150, -w / 2 + 280):
                    box(g, f"{lab}_axle{u}", (*at(u, 0), 50), (100, d, 100), L["tyre"], yaw=yaw)
        elif k == "dock_door":
            box(g, lab, (x, y, h / 2), (w, d, h), L["metal_green"] if n % 2 else L["paint_white"], yaw=yaw)
            for u in (-w / 2 - 20, w / 2 + 20):
                box(g, f"{lab}_bump{u}", (*at(u, -d), 90), (25, 20, 60), L["paint_yellow"], yaw=yaw)
        elif k == "skip":
            box(g, lab, (x, y, h / 2), (w, d, h), L["paint_yellow"], yaw=yaw)
        elif k == "pallets":
            box(g, lab, (x, y, h / 2), (w, d, h), L["wood"], yaw=yaw)
        elif k == "stage":
            box(g, lab, (x, y, h / 2), (w, d, h), L["metal_dark"], yaw=yaw)
            for u in (-w / 2, w / 2):
                box(g, f"{lab}_truss{u}", (*at(u, 0), 400), (40, 40, 800), L["metal_dark"], yaw=yaw)
            box(g, lab + "_top", (x, y, 800), (w, 40, 40), L["metal_dark"], yaw=yaw)
            box(g, lab + "_screen", (*at(0, d / 2 - 20), 520), (w * 0.5, 10, 280), L["glass"], yaw=yaw)
        elif k == "speaker_tower":
            box(g, lab, (x, y, h / 2), (w, d, h), L["metal_dark"], yaw=yaw)
        elif k == "barrier_row":
            box(g, lab, (x, y, 55), (w, 5, 110), L["metal_dark"], yaw=yaw)
            box(g, lab + "_feet", (x, y, 3), (w, d, 6), L["metal_dark"], yaw=yaw)
        elif k == "kiosk":
            box(g, lab, (x, y, 125), (w, d, 250), L["car_paints"][n % 7], yaw=yaw)
            box(g, lab + "_roof", (x, y, 265), (w + 60, d + 60, 30), L["paint_white"], yaw=yaw)
        elif k == "food_truck":
            paint = L["car_paints"][int(it.get("paint", 0)) % len(L["car_paints"])]
            box(g, lab, (x, y, 160), (w, d, 240), paint, yaw=yaw)
            box(g, lab + "_hatch", (*at(0, -d / 2), 170), (w * 0.5, 4, 90), L["glass"], yaw=yaw)
            box(g, lab + "_cab", (*at(w / 2 + 70, 0), 110), (140, d, 180), paint, yaw=yaw)
        elif k == "portable_toilet":
            box(g, lab, (x, y, 115), (w, d, 230), L["car_paints"][4], yaw=yaw)
        elif k == "tent":
            box(g, lab, (x, y, 110), (w, d, 220), L["paint_white"], yaw=yaw)
            box(g, lab + "_roof", (x, y, 240), (w + 20, d + 20, 40), L["paint_white"], yaw=yaw)
        elif k == "fountain":
            box(g, lab, (x, y, 30), (w, d, 60), L["granite"], shape="Cylinder")
            box(g, lab + "_water", (x, y, 58), (w - 40, d - 40, 4), L["glass"], shape="Cylinder")
            box(g, lab + "_jet", (x, y, 120), (60, 60, 160), L["granite"], shape="Cylinder")
        elif k == "monument":
            box(g, lab, (x, y, 60), (w, d, 120), L["granite"], yaw=yaw)
            box(g, lab + "_col", (x, y, 120 + (h - 120) / 2), (w * 0.4, d * 0.4, h - 120), L["concrete"], yaw=yaw)
        elif k == "trolley_bay":
            box(g, lab, (x, y, 110), (w, d, 8), L["metal_dark"], yaw=yaw)
            for u in (-w / 2, w / 2):
                box(g, f"{lab}_p{u}", (*at(u, 0), 55), (8, 8, 110), L["metal_dark"], yaw=yaw)
        elif k == "phone_box":
            box(g, lab, (x, y, 125), (w, d, 250), L["car_paints"][3], yaw=yaw)
        elif k == "asset":
            name = it.get("asset", "")
            it = dict(it, tint_mat={"traffic_cone": L["car_paints"][3],
                                    "kenney_dumpster": L["metal_green"],
                                    "kenney_construction_barrier": L["paint_white"],
                                    "trash_bags": L["car_paints"][1]}.get(name, L["metal_dark"]))
            if _place_asset(g, lab, it, x, y, yaw) is None:
                failed[it.get("asset", "?")] = failed.get(it.get("asset", "?"), 0) + 1
                box(g, lab, (x, y, h / 2), (w, d, h), L["paint_yellow"], yaw=yaw)
        else:
            failed[k] = failed.get(k, 0) + 1
    M.set_wet(float(p.get("weather", {}).get("wet", 0.0)))
    return {"counts": counts, "lamps": lamps, "unbuilt": failed,
            "decals": len(group_actors("Decals"))}


def _trace_blocked(world, a, b):
    """Line trace a->b; returns the label of the first Env actor hit, or None."""
    hit = unreal.SystemLibrary.line_trace_single(
        world, unreal.Vector(*a), unreal.Vector(*b), unreal.TraceTypeQuery.TRACE_TYPE_QUERY1,
        True, [], unreal.DrawDebugTrace.NONE, True)
    if isinstance(hit, tuple):
        hit = hit[-1] if not isinstance(hit[0], unreal.HitResult) else hit[0]
    if hit is None:
        return None
    t = hit.to_tuple()
    blocking = bool(t[0])
    actor = next((v for v in t if isinstance(v, unreal.Actor)), None)
    if not blocking or actor is None:
        return None
    if _folder(actor).split("/")[0] != "Env":
        return None
    return actor.get_actor_label()


def entrance_gate(plan, cameras, candidates):
    """Each entrance must have >= 1 camera that frames it (host: usably_visible)
    AND an unoccluded line of sight to it in the built world (here: line trace
    camera -> entrance at 1.5 m). A self-test first traces through the largest
    Env block; if that does not hit, occlusion cannot be verified and the gate fails.
    """
    world = unreal.EditorLevelLibrary.get_editor_world()
    blocks = sorted((a for a in group_actors("Env") if a.get_actor_label().startswith("block_")
                     and not a.get_actor_label().endswith("_roof")),
                    key=lambda a: -a.get_actor_bounds(False)[1].x * a.get_actor_bounds(False)[1].y)
    selftest = None
    if blocks:
        o, e = blocks[0].get_actor_bounds(False)
        selftest = _trace_blocked(world, (o.x - e.x - 200, o.y, 100), (o.x + e.x + 200, o.y, 100))
    cams = {c["id"]: c for c in cameras}
    out, fails, blockers = {}, [], {}
    for ent in plan.get("entrances", []):
        seen = []
        for cid in candidates.get(ent["id"], []):
            c = cams[cid]
            # Chest height: a 1 m target is hidden by a barrier arm a real
            # camera sees a person over ([X] 2026-09-26, perimeter_site s1).
            blocker = _trace_blocked(world, tuple(c["location"]), (ent["x"], ent["y"], 150.0))
            if blocker is None:
                seen.append(cid)
            else:
                blockers[f"{ent['id']}<-{cid}"] = blocker
        out[ent["id"]] = seen
        if not seen:
            fails.append(ent["id"])
    ok = not fails and (selftest is not None or not blocks)
    return {"ok": ok, "entrances": out, "unseen": fails, "blocked_by": blockers,
            "trace_selftest_hit": selftest}

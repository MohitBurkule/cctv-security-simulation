"""Environment plans: one per site type, deterministic, nothing on the walkers."""
import math

from survsim.export.env_plan import SITE_TYPES, infer_site_type, plan, weather_for

CAMS = [{"location": [1600.0, 1200.0, 400.0], "rotation": [-18.0, -120.0, 0.0]}]
TALL = {"building", "container", "gatehouse", "canopy", "stage", "wall", "fence"}


def test_every_site_type_gives_a_distinct_layout():
    plans = [plan(s, 1, 2000.0, CAMS) for s in SITE_TYPES]
    assert len({p["layout"] for p in plans}) == len(SITE_TYPES)
    kinds = [frozenset(i["kind"] for i in p["items"]) for p in plans]
    assert len(set(kinds)) == len(SITE_TYPES)


def test_deterministic():
    assert plan("car_park", 3, 2500.0, CAMS) == plan("car_park", 3, 2500.0, CAMS)
    assert plan("car_park", 3, 2500.0, CAMS) != plan("car_park", 4, 2500.0, CAMS)


def test_no_buildings_inside_envelope_and_camera_near_field_clear():
    for s in SITE_TYPES:
        p = plan(s, 1, 2000.0, CAMS)
        for it in p["items"]:
            if it["kind"] == "building":
                sx, sy, _ = it["size"]
                assert abs(it["x"]) + 1 >= 2000.0 + sx / 2 - 1 or abs(it["y"]) + 1 >= 2000.0 + sy / 2 - 1
            if it["kind"] in ("lamp", "tree", "bench", "bin", "pillar", "planter", "kiosk"):
                cx, cy, yaw = 1600.0, 1200.0, -120.0
                d = math.hypot(it["x"] - cx, it["y"] - cy)
                front = (it["x"] - cx) * math.cos(math.radians(yaw)) + (it["y"] - cy) * math.sin(math.radians(yaw))
                assert not (d < 800 and front > 0), (s, it)


def test_site_inference_and_weather():
    assert infer_site_type({"id": "compound", "zones": [{"tag": "restricted"}]}) == "perimeter_site"
    assert infer_site_type({"id": "street_plaza"}) in ("retail_street", "event_plaza")
    assert infer_site_type({"id": "x", "site_type": "car_park"}) == "car_park"
    assert weather_for({"weather": {"preset": "rain_light"}})["wet"] >= 0.7


def test_nothing_solid_on_a_walked_path():
    manifest = {"seed": 1, "cameras": CAMS, "tracks": [{"keys": [
        {"location": [x * 50.0, 0.0, 0.0]} for x in range(-60, 61)]}]}
    from survsim.export.env_plan import walked_cells
    p = plan("car_park", 1, 3000.0, CAMS, walked=walked_cells(manifest))
    for it in p["items"]:
        if it["kind"] in ("car", "pillar", "planter", "bench"):
            assert abs(it["y"]) > 150 or abs(it["x"]) > 3000, it
    assert p["skipped_on_paths"] > 0


def _placements(p):
    return {(i["kind"], round(i["x"] / 200), round(i["y"] / 200)) for i in p["items"]}


def test_site_types_share_under_half_their_placements():
    plans = {s: plan(s, 1, 2000.0, CAMS) for s in SITE_TYPES}
    for a in SITE_TYPES:
        for b in SITE_TYPES:
            if a < b:
                pa, pb = _placements(plans[a]), _placements(plans[b])
                share = len(pa & pb) / max(1, min(len(pa), len(pb)))
                assert share < 0.5, (a, b, share)


def test_sun_differs_by_seed_without_time_of_day():
    a = plan("event_plaza", 1, 2000.0, CAMS)["lighting"]
    b = plan("event_plaza", 2, 2000.0, CAMS)["lighting"]
    assert (a["sun_yaw"], a["sun_pitch"]) != (b["sun_yaw"], b["sun_pitch"])

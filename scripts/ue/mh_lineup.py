# ruff: noqa: F821  (unreal, RESULT, finish, guard are injected by the job runner)
"""Line up N MetaHumans in painted clothing in front of a level camera; 1 frame.

Params: /tmp/survsim_mh_lineup.json {"people": [{"character", "outfit", "height_cm"}],
"catalog": {name: entry (tools/render_run.metahuman_catalog)}, "camera": "cam_01",
"distance_m": 7, "spacing_m": 0.9, "out": dir, "name": "lineup.png"}.
Checks the painted-clothing material (scripts/ue/survsim_clothing.py) and
body variety in one SceneCapture frame; cleans up after itself.
"""
import importlib
import json
import math
import os
import sys

sys.path.insert(0, "/home/mohit/multi-person-camera-real-sim/scripts/ue")
import survsim_clothing as C
import survsim_scene as S

importlib.reload(C)
importlib.reload(S)
with open("/tmp/survsim_mh_lineup.json") as _fh:
    P = json.load(_fh)
if not P.get("licence_restricted"):
    raise RuntimeError(
        "mh_lineup.py renders MetaHuman characters (LicenseRef-Epic-MetaHuman, "
        "docs/research/47): Epic's terms forbid using them to build a database or "
        "train/test AI. Pass {'licence_restricted': true} in the lineup params to "
        "confirm this run is for internal, non-commercial evidence only "
        "(see survsim/gt/licences.py).")
RESULT["loaded_level"] = S.ensure_level()
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
world = unreal.EditorLevelLibrary.get_editor_world()
for a in eas.get_all_level_actors():
    if str(a.get_folder_path()) == "MHLineup":
        eas.destroy_actor(a)
hidden = [a for a in eas.get_all_level_actors() if isinstance(a, unreal.SkeletalMeshActor)
          and not a.is_hidden_ed()]
for a in hidden:
    a.set_actor_hidden_in_game(True)
    a.set_is_temporarily_hidden_in_editor(True)
cam = S.find("Cams", P.get("camera", "cam_01"))
cloc, crot = cam.get_actor_location(), cam.get_actor_rotation()
yaw = math.radians(crot.yaw)
d = P.get("distance_m", 7.0) * 100
cx, cy = cloc.x + d * math.cos(yaw), cloc.y + d * math.sin(yaw)
side = (-math.sin(yaw), math.cos(yaw))
n = len(P["people"])
idle = unreal.load_asset(next(iter(P["catalog"].values()))["idle"])
spawned = []
for i, e in enumerate(P["people"]):
    ch = P["catalog"][e["character"]]
    off = (i - (n - 1) / 2) * P.get("spacing_m", 0.9) * 100
    loc = unreal.Vector(cx + side[0] * off, cy + side[1] * off, 0.0)
    a = eas.spawn_actor_from_class(unreal.SkeletalMeshActor, loc,
                                   unreal.Rotator(pitch=0.0, yaw=crot.yaw + 180 - 90 + (i % 3 - 1) * 20,
                                                  roll=0.0))
    a.set_folder_path("MHLineup")
    comp = a.skeletal_mesh_component
    comp.set_skeletal_mesh_asset(unreal.load_asset(ch["mesh"]))
    s = e.get("height_cm", ch["height_cm"]) / ch["height_cm"]
    a.set_actor_scale3d(unreal.Vector(s, s, s))
    S.dress_metahuman(a, ch)
    C.dress_actor(a, e["outfit"], ch["height_cm"], ch.get("xmax_cm", 46.0))
    comp.set_update_animation_in_editor(True)
    comp.play_animation(idle, True)
    comp.set_position(1.3 * i, False)
    spawned.append(a)
    RESULT.setdefault("people_cm", []).append([loc.x, loc.y, loc.z, e.get("height_cm", ch["height_cm"])])
RESULT["camera"] = {"loc": [cloc.x, cloc.y, cloc.z], "pitch": crot.pitch, "yaw": crot.yaw,
                    "fov": float(P.get("fov", 40.0)), "res": P.get("resolution", [1280, 720])}
w, h = P.get("resolution", [1280, 720])
rt = unreal.RenderingLibrary.create_render_target2d(world, w, h, unreal.TextureRenderTargetFormat.RTF_RGBA8)
cap = eas.spawn_actor_from_class(unreal.SceneCapture2D, cloc, crot)
cap.set_folder_path("MHLineup")
cc = cap.get_component_by_class(unreal.SceneCaptureComponent2D)
cc.set_editor_property("texture_target", rt)
cc.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
cc.set_editor_property("capture_every_frame", True)  # [X] False: single capture is white in day (no exposure history)
cc.set_editor_property("fov_angle", float(P.get("fov", 40.0)))
ppv = next((a for a in eas.get_all_level_actors() if a.get_actor_label() == "CCTVExposure"), None)
RESULT["exposure_found"] = ppv is not None
if ppv is not None:
    pps = ppv.get_editor_property("settings")
    pps.set_editor_property("auto_exposure_bias",
                            pps.get_editor_property("auto_exposure_bias") + float(P.get("ev_offset", 0.0)))
    cc.set_editor_property("post_process_settings", pps)
    cc.set_editor_property("post_process_blend_weight", 1.0)
os.makedirs(P["out"], exist_ok=True)
st = {"t": 0}
RESULT["async"] = True


@guard
def tick(_dt):
    st["t"] += 1
    if st["t"] < P.get("warmup_ticks", 120):
        return
    unreal.unregister_slate_post_tick_callback(st["h"])
    cc.capture_scene()
    unreal.RenderingLibrary.export_render_target(cap, rt, P["out"], P.get("name", "lineup.png"))
    if not P.get("keep"):
        for a in eas.get_all_level_actors():
            if str(a.get_folder_path()) == "MHLineup":
                eas.destroy_actor(a)
    for a in hidden:
        a.set_actor_hidden_in_game(False)
        a.set_is_temporarily_hidden_in_editor(False)
    finish({"out": os.path.join(P["out"], P.get("name", "lineup.png")), "n": n})


st["h"] = unreal.register_slate_post_tick_callback(tick)
RESULT["_handles"] = [st["h"]]

# ruff: noqa: F821  (unreal, RESULT, finish, guard are injected by the job runner)
"""Render a baked offline MetaHuman walking in the *current* level (one camera).

A job for either runner (production /tmp/survsim_jobs or the MH sandbox).
Parameters are the PARAMS dict below (edit a copy before submitting; the
production runner's job dir is not ours to put side files in).

Needs the baked asset set from scripts/ue/mh_bake.py present in the project's
Content at the same /Game path (copy Content/SurvSim/MetaHumans/Baked/<name>).

What it does, in <= ~1 min of editor time:
  1. finds the camera (label PARAMS["camera"], else the first actor in folder
     Cams), puts the person PARAMS["distance_m"] ahead of it on the ground
     (line trace), walking across the view;
  2. body SkeletalMeshActor + face (leader pose) + outfit ChaosClothComponent
     (leader pose, added through SubobjectDataSubsystem);
  3. captures plate.png with the person hidden, then N frames with the walk
     clip seeked to i/fps and the actor advanced at speed_mps;
  4. destroys everything it spawned (folder MHDemo) — the level is left as found;
  5. RESULT: camera/actor transforms, the light list, per-frame bone heights
     (evidence limbs move), file list. Run scripts/check_render.py on the dir.
"""
import math
import os

PARAMS = globals().get("PARAMS_OVERRIDE") or {
    "name": "MHB02", "camera": None, "distance_m": 9.0, "frames": 3, "fps": 10.0,
    "speed_mps": 1.35, "walk_yaw_rel": 90.0, "resolution": [1280, 720], "warmup_ticks": 60,
    "out": "/home/mohit/multi-person-camera-real-sim/out/demo_metahuman/prod_MHB02",
    "anim": "/MetaHumanCharacter/Optional/Animation/UEFNAnimPreset/Locomotion/AS_MH_Neutral_Walk_Loop_F",
}
if not PARAMS.get("licence_restricted"):
    raise RuntimeError(
        "mh_prod_render.py renders MetaHuman characters (LicenseRef-Epic-MetaHuman, "
        "docs/research/47): Epic's terms forbid using them to build a database or "
        "train/test AI. Pass PARAMS['licence_restricted'] = True to confirm this run "
        "is for internal, non-commercial evidence only (see survsim/gt/licences.py).")

EAL = unreal.EditorAssetLibrary
# The persistent editor is shared (e.g. character imports switch to TestMap):
# make sure the render map is loaded before looking for its cameras.
import sys

sys.path.insert(0, "/home/mohit/multi-person-camera-real-sim/scripts/ue")
import survsim_scene as _S

RESULT["loaded_level"] = _S.ensure_level()
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
world = unreal.EditorLevelLibrary.get_editor_world()
base = f"/Game/SurvSim/MetaHumans/Baked/{PARAMS['name']}/"
unreal.AssetRegistryHelpers.get_asset_registry().scan_paths_synchronous([base[:-1]], True)
body_mesh = unreal.load_asset(base + f"SKM_{PARAMS['name']}_Body")
face_mesh = unreal.load_asset(base + f"SKM_{PARAMS['name']}_Face")
outfit = unreal.load_asset(base + f"OA_{PARAMS['name']}_0") if EAL.does_asset_exist(
    base + f"OA_{PARAMS['name']}_0") and PARAMS.get("garment", True) else None
anim = unreal.load_asset(PARAMS["anim"])
if body_mesh is None or anim is None:
    raise RuntimeError(f"missing assets under {base} (copied?) or anim {PARAMS['anim']}")

actors = eas.get_all_level_actors()
for a in actors:
    if str(a.get_folder_path()) == "MHDemo":
        eas.destroy_actor(a)
# Other people (and imported characters left unfoldered at the origin) would
# share the plate diff and merge with the MetaHuman's detection: hide them for
# the capture, restore after (the level is left as found).
_hidden = [a for a in actors if isinstance(a, unreal.SkeletalMeshActor)
           and str(a.get_folder_path()).split("/")[0] in ("People", "None")
           and not a.is_hidden_ed()]
for a in _hidden:
    a.set_is_temporarily_hidden_in_editor(True)
    a.set_actor_hidden_in_game(True)
RESULT["hidden_others"] = len(_hidden)
cams = [a for a in actors if (PARAMS["camera"] and a.get_actor_label() == PARAMS["camera"])
        or (not PARAMS["camera"] and str(a.get_folder_path()) == "Cams")]
if not cams:
    raise RuntimeError("no camera found")
cam = cams[0]
cloc, crot = cam.get_actor_location(), cam.get_actor_rotation()
RESULT["camera"] = {"label": cam.get_actor_label(), "loc": [cloc.x, cloc.y, cloc.z],
                    "rot_pitch_yaw_roll": [crot.pitch, crot.yaw, crot.roll]}
RESULT["lights"] = sorted(f"{a.get_actor_label()}:{a.get_class().get_name()}" for a in actors
                          if isinstance(a, (unreal.Light, unreal.SkyLight, unreal.PostProcessVolume)))
yaw = math.radians(crot.yaw)
d = PARAMS["distance_m"] * 100.0
tx, ty = cloc.x + d * math.cos(yaw), cloc.y + d * math.sin(yaw)
hit = unreal.SystemLibrary.line_trace_single(world, unreal.Vector(tx, ty, cloc.z + 500),
                                             unreal.Vector(tx, ty, -5000),
                                             unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, False, [],
                                             unreal.DrawDebugTrace.NONE, True)
gz = hit.to_tuple()[4].z if hit and hit.to_tuple()[0] else 0.0
target = unreal.Vector(tx, ty, gz)
walk_yaw = crot.yaw + PARAMS["walk_yaw_rel"]
fwd = unreal.Vector(math.cos(math.radians(walk_yaw)), math.sin(math.radians(walk_yaw)), 0.0)

person = eas.spawn_actor_from_class(unreal.SkeletalMeshActor, target,
                                    unreal.Rotator(pitch=0.0, yaw=walk_yaw - 90.0, roll=0.0))
person.set_folder_path("MHDemo")
person.set_actor_label(f"MH_{PARAMS['name']}")
body = person.skeletal_mesh_component
body.set_skeletal_mesh_asset(body_mesh)
parts = [body]
sds = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
root = sds.k2_gather_subobject_data_for_instance(person)[0]
for cls, asset in ((unreal.SkeletalMeshComponent, face_mesh), (unreal.ChaosClothComponent, outfit)):
    if asset is None:
        continue
    before = {c.get_name() for c in person.get_components_by_class(cls)}
    sds.add_new_subobject(unreal.AddNewSubobjectParams(parent_handle=root, new_class=cls))
    comp = next(c for c in person.get_components_by_class(cls) if c.get_name() not in before)
    comp.attach_to_component(body, "", unreal.AttachmentRule.SNAP_TO_TARGET,
                             unreal.AttachmentRule.SNAP_TO_TARGET, unreal.AttachmentRule.SNAP_TO_TARGET, False)
    if cls is unreal.ChaosClothComponent:
        comp.set_editor_property("asset", asset)
    else:
        comp.set_skeletal_mesh_asset(asset)
    comp.set_leader_pose_component(body)
    parts.append(comp)
RESULT["parts"] = [f"{c.get_name()}:{c.get_class().get_name()}" for c in parts]
if PARAMS.get("clothes"):
    # Painted clothing, as production renders dress every MetaHuman.
    import survsim_clothing as _C
    RESULT["dressed_slots"] = _C.dress_actor(person, PARAMS["clothes"], PARAMS["stature_cm"],
                                             PARAMS.get("xmax_cm", 46.0))
body.set_update_animation_in_editor(True)
body.play_animation(anim, True)
RESULT["person"] = {"loc": [target.x, target.y, target.z], "walk_yaw": walk_yaw}

w, h = PARAMS["resolution"]
rt = unreal.RenderingLibrary.create_render_target2d(world, w, h, unreal.TextureRenderTargetFormat.RTF_RGBA8)
cap = eas.spawn_actor_from_class(unreal.SceneCapture2D, cloc, crot)
cap.set_folder_path("MHDemo")
cc = cap.get_component_by_class(unreal.SceneCaptureComponent2D)
cc.set_editor_property("texture_target", rt)
cc.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
# [X] False: a lone capture has no exposure history and comes out white in day light
cc.set_editor_property("capture_every_frame", True)
try:
    fov = cam.get_cine_camera_component().get_horizontal_field_of_view()
except Exception:                                           # noqa: BLE001
    fov = 78.0
cc.set_editor_property("fov_angle", float(fov))
RESULT["fov"] = fov
for a in actors:
    if a.get_actor_label() == "CCTVExposure":
        cc.set_editor_property("post_process_settings", a.get_editor_property("settings"))
        cc.set_editor_property("post_process_blend_weight", 1.0)
out = PARAMS["out"]
os.makedirs(out, exist_ok=True)
n, fps = PARAMS["frames"], PARAMS["fps"]
st = {"t": 0, "i": -1, "files": [], "bones": []}
RESULT["async"] = True


def _pose(i):
    t = max(i, 0) / fps
    body.set_position(t % anim.get_play_length(), False)
    person.set_actor_location(target + fwd * (PARAMS["speed_mps"] * 100.0 * (t - n / (2 * fps))),
                              False, False)
    person.set_actor_hidden_in_game(i < 0)          # i = -1: the empty plate
    for c in parts:
        c.set_visibility(i >= 0, True)


def _done(extra=None):
    unreal.unregister_slate_post_tick_callback(st["h"])
    for a in _hidden:
        a.set_is_temporarily_hidden_in_editor(False)
        a.set_actor_hidden_in_game(False)
    for a in eas.get_all_level_actors():
        if str(a.get_folder_path()) == "MHDemo":
            eas.destroy_actor(a)
    finish(dict({"frames": st["files"], "bones_hand_r_foot_l_foot_r_z": st["bones"], "out": out},
                **(extra or {})))


@guard
def tick(_dt):
    st["t"] += 1
    if st["t"] == 1:
        _pose(-1)
    if st["t"] < PARAMS["warmup_ticks"]:
        return
    k = (st["t"] - PARAMS["warmup_ticks"]) % 3
    if k == 0:
        _pose(st["i"])
        return
    if k != 2:
        return
    i = st["i"]
    if i >= 0:
        loc = person.get_actor_location()
        st["bones"].append([round(body.get_socket_location(b).z - loc.z, 1)
                            for b in ("hand_r", "foot_l", "foot_r")])
    cc.capture_scene()
    name = "plate.png" if i < 0 else f"{i:04d}.png"
    unreal.RenderingLibrary.export_render_target(cap, rt, out, name)
    st["files"].append(name)
    st["i"] += 1
    if st["i"] >= n:
        _done()


st["h"] = unreal.register_slate_post_tick_callback(tick)
RESULT["_handles"] = [st["h"]]

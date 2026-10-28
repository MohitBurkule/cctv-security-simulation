"""Render job: a SurvSim run -> frames, in the PERSISTENT editor (job_runner.py).

Submitted by `tools/render_run.py`; reads its parameters from REQUEST (JSON),
never from text templating, so this file is plain valid Python.

Incremental: the street ("Env") is built only if absent or `rebuild_env` is
set; lights only when the preset changes; people, cameras and the per-camera
sequences are rebuilt each render (cheap). MRQ runs one job per camera, each
sequence with its own camera cut (docs/12: without it every camera renders the
same view). Async: the job returns at once and `finish()` writes the result
when MRQ reports completion.
"""
import importlib
import json
import sys

REQUEST = "/tmp/survsim_render_request.json"
sys.path.insert(0, "/home/mohit/multi-person-camera-real-sim/scripts/ue")
import survsim_clothing as C
import survsim_materials as M
import survsim_scene as S

importlib.reload(C)
importlib.reload(M)
importlib.reload(S)

with open(REQUEST) as fh:
    req = json.load(fh)
# A/B switch for the ID-mask depth test (docs/BUG_gt_visibility_fences.md).
if "id_mask_depth_test" in req:
    S.ID_MASK_DEPTH_TEST = bool(req["id_mask_depth_test"])
with open(req["manifest"]) as fh:
    manifest = json.load(fh)
with open(req["plan"]) as fh:
    plan = json.load(fh)["people"]
with open(req["catalog"]) as fh:
    chars = json.load(fh)

res = RESULT  # noqa: F821  (injected by job_runner)
res["loaded_level"] = S.ensure_level()
# [X] 2026-09-26: 18 MetaHumans + textures + HDRI ran the 8 GB card out of
# memory inside FGPUSkinCache during MRQ (editor crash). The skin cache is
# only needed for ray tracing / mesh deformers: compute skinning in the vertex
# shader instead, and cap the texture pool.
for _cmd in ("r.SkinCache.Mode 0", "r.Streaming.PoolSize 1000", "r.SkeletalMeshLODBias 1"):
    unreal.SystemLibrary.execute_console_command(None, _cmd)  # noqa: F821
M.PBR = req.get("pbr") or {}
S.clear("Tmp")
if req.get("rebuild_env"):
    # Free the previous set's people and street before building the next one.
    S.clear("People")
    S.clear("Env")
    S.clear("Decals")
    unreal.SystemLibrary.collect_garbage()  # noqa: F821
if req.get("rebuild_env") or not S.group_actors("Env"):
    S.clear_unfoldered()
    S.clear("Env")
    if req.get("env_plan"):
        import time as _time
        _t0 = _time.time()
        if req["env_plan"].get("version") == 2:
            street = S.build_from_plan_v2(req["env_plan"])
            res["unbuilt"] = street["unbuilt"]
            res["decals"] = street["decals"]
        else:
            street = S.build_from_plan(req["env_plan"])
        res["env_build_s"] = round(_time.time() - _t0, 2)
        res["layout"] = req["env_plan"]["layout"]
    else:
        half = S.walk_envelope(manifest)
        street = S.build_street(seed=int(manifest.get("seed", 0)), half=half,
                                cameras=manifest.get("cameras", []))
        res["half_cm"] = half
    res["street"] = street["counts"]
    with open("/tmp/survsim_lamps.json", "w") as fh:
        json.dump(street["lamps"], fh)
res["frustum_violations"] = S.frustum_violations(manifest)
if res["frustum_violations"]:
    raise RuntimeError(f"static geometry in a camera's first 10 m: {res['frustum_violations'][:10]}")
if (req.get("env_plan") or {}).get("version") == 2 and req["env_plan"].get("entrance_cameras") is not None:
    res["entrance_gate"] = S.entrance_gate(req["env_plan"], manifest["cameras"],
                                           req["env_plan"]["entrance_cameras"])
    if not res["entrance_gate"]["ok"] and req.get("entrance_gate", "fail") == "fail":
        raise RuntimeError(f"entrance gate failed: {res['entrance_gate']}")
lighting = req.get("lighting", "day")
sun = S.find("Lights", "Sun")
if req.get("relight", True) or sun is None:
    with open("/tmp/survsim_lamps.json") as fh:
        lamps = json.load(fh)
    sun = (req.get("env_plan") or {}).get("lighting", {})
    res["light"] = S.light(lighting, lamps, hdri=req.get("hdri"), ev100=req.get("ev100"),
                           sun_yaw=sun.get("sun_yaw"), sun_pitch=sun.get("sun_pitch"))
cams = S.cameras(manifest)
for _cid, _ev in (req.get("camera_ev") or {}).items():
    if _cid in cams:
        S.set_camera_ev(cams[_cid], float(_ev))
res["camera_ev"] = req.get("camera_ev")
if req.get("cameras"):
    cams = {k: v for k, v in cams.items() if k in req["cameras"]}
people = S.spawn_people(plan, chars)
seqs, rate_info = S.build_sequences(manifest, plan, people, cams, chars, int(req["n_frames"]),
                                    camera_keys=req.get("camera_keys"))
# --window: the sequences are keyed from frame 0 (so animation phase and
# positions are identical to a full render) but play from frame_start;
# MRQ's {frame_number} keeps absolute GT frame numbers in the file names.
_f0 = int(req.get("frame_start") or 0)
if _f0:
    for _seq in seqs.values():
        _seq.set_playback_start(_f0)
        unreal.EditorAssetLibrary.save_asset(_seq.get_path_name())  # noqa: F821
res["frame_start"] = _f0
res["epoch_actors"] = {e["uid"]: [e.get("person_uid"), e.get("epoch"), e.get("visible")]
                       for e in plan if e.get("person_uid")}
res["keyed_cameras"] = sorted((req.get("camera_keys") or {}).keys())
res.update(rate_info)
res["people"] = len(people)
if req.get("id_mask"):
    res["stencil_ids"] = S.tag_people_stencil(people)
    res["id_mask_depth_test"] = bool(S.ID_MASK_DEPTH_TEST)
    # An appearance epoch is the same person: its actor carries the person's ID.
    for _e in plan:
        _pu = _e.get("person_uid")
        if _pu and _e["uid"] != _pu and _pu in res["stencil_ids"]:
            for _c in people[_e["uid"]].get_components_by_class(unreal.PrimitiveComponent):  # noqa: F821
                _c.set_editor_property("custom_depth_stencil_value", res["stencil_ids"][_pu])
            res["stencil_ids"].pop(_e["uid"], None)
res["cameras"] = list(seqs)
res["async"] = True
res["resolutions"] = {c["id"]: req.get("probe_res") or c.get("resolution", [1280, 720])
                      for c in manifest["cameras"]}
S.render(seqs, req["out"], finish,  # noqa: F821 warmup=int(req.get("warmup", 32)),
         resolutions=res["resolutions"], id_mask=bool(req.get("id_mask")), mask_root=req.get("mask_out"))

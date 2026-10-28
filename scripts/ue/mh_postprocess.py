# ruff: noqa: F821  (unreal, RESULT are injected by the job runner)
"""Give each baked MetaHuman body its RigLogic post-process AnimBP (one-time).

[V] 2026-09-26: without it the body's corrective/helper joints stay at the
reference pose while the clavicles animate, so the shoulders stretch into
"bat wings" in every walking or idle frame. The engine ships
/MetaHumanCharacter/Body/ABP_Body_PostProcess (and ABP_Face_PostProcess);
setting them as the meshes' post_process_anim_blueprint runs them after any
animation mode, including the sequencer's single-node tracks.
"""
if not globals().get("LICENCE_RESTRICTED"):
    raise RuntimeError(
        "mh_postprocess.py modifies baked MetaHuman bodies (LicenseRef-Epic-MetaHuman, "
        "docs/research/47): Epic's terms forbid using them to build a database or "
        "train/test AI. Set the job global LICENCE_RESTRICTED = True to confirm this "
        "run is for internal, non-commercial evidence only (see survsim/gt/licences.py).")

EAL = unreal.EditorAssetLibrary
ABP = {"Body": "/MetaHumanCharacter/Body/ABP_Body_PostProcess",
       "Face": "/MetaHumanCharacter/Face/ABP_Face_PostProcess"}
out = {}
for part, path in ABP.items():
    bp = unreal.load_asset(path)
    cls = bp.generated_class() if bp is not None else None
    for name in ("MHB01", "MHB02", "MHB03", *globals().get("EXTRA_BODIES", [])):
        mp = f"/Game/SurvSim/MetaHumans/Baked/{name}/SKM_{name}_{part}"
        if not EAL.does_asset_exist(mp):
            continue
        mesh = unreal.load_asset(mp)
        mesh.set_editor_property("post_process_anim_blueprint", cls)
        EAL.save_asset(mp)
        got = mesh.get_editor_property("post_process_anim_blueprint")
        out[f"{name}_{part}"] = got.get_name() if got else None
RESULT["set"] = out

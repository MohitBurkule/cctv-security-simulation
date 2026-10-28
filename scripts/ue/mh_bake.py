# ruff: noqa: F821  (unreal, RESULT, finish, guard are injected by the job runner)
"""MetaHuman sandbox job: bake an offline MetaHuman preview into real assets.

    params: /tmp/survsim_jobs_mh/mh_bake.json
    {"character": "/Game/SurvSim/MetaHumans/MH_Blank01", "name": "MHB01",
     "body": {"Height": 178.0, "Masculine/Feminine": 0.2}, "preview_material": "EDITABLE",
     "settle_ticks": 60}

build_meta_human is gated on Epic-login source textures (docs/12). The editor
*preview* of the same character is complete offline: rigged body + face,
synthesised skin, grooms. But it lives in /Engine/Transient — the mesh is
garbage-collected as soon as its preview actor goes, and saving anything that
references it raises "Insufficient writing permission to save
/Engine/Transient/BodyMesh_N". This job copies each preview SkeletalMesh, each
of its material instances (as MaterialInstanceConstant, parameters copied) and
every transient texture those reference into /Game/SurvSim/MetaHumans/Baked/<name>,
then saves them. The result is an ordinary SkeletalMesh any render path can use.

Async: the preview pipeline assembles over the next ticks, so the copy happens
after `settle_ticks`.
"""
import json
import time

with open("/tmp/survsim_jobs_mh/mh_bake.json") as _fh:
    P = json.load(_fh)
if not P.get("licence_restricted"):
    raise RuntimeError(
        "mh_bake.py bakes MetaHuman characters (LicenseRef-Epic-MetaHuman, "
        "docs/research/47): Epic's terms forbid using them to build a database or "
        "train/test AI. Pass {'licence_restricted': true} in mh_bake.json to confirm "
        "this run is for internal, non-commercial evidence only (see survsim/gt/licences.py).")
EAL = unreal.EditorAssetLibrary
MEL = unreal.MaterialEditingLibrary
AT = unreal.AssetToolsHelpers.get_asset_tools()
sub = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
DEST = f"/Game/SurvSim/MetaHumans/Baked/{P['name']}"
t0 = time.time()

ch = unreal.load_asset(P["character"])
if not sub.is_object_added_for_editing(ch):
    sub.try_add_object_to_edit(ch)
if P.get("body"):
    cons = list(sub.get_body_constraints(ch))
    by = {str(c.get_editor_property("name")): c for c in cons}
    missing = sorted(set(P["body"]) - set(by))
    if missing:
        raise KeyError(f"unknown body constraints {missing}")   # never silently default
    for k, v in P["body"].items():
        by[k].set_editor_property("target_measurement", float(v))
        by[k].set_editor_property("is_active", True)
    sub.set_body_constraints(ch, cons)
    sub.commit_body_state(ch)
    RESULT["body_readback"] = {n: round(float(by_c.get_editor_property("target_measurement")), 3)
                               for by_c in sub.get_body_constraints(ch)
                               for n in [str(by_c.get_editor_property("name"))] if n in P["body"]}
if P.get("preview_material"):
    ch.set_editor_property("preview_material_type",
                           getattr(unreal.MetaHumanCharacterSkinPreviewMaterial, P["preview_material"]))
    sub.commit_skin_settings(ch, ch.get_editor_property("skin_settings"))
if P.get("skin"):   # e.g. {"u": 0.3, "v": 0.7} skin-tone coordinates
    ss = ch.get_editor_property("skin_settings")
    sk = ss.get_editor_property("skin")
    for k, v in P["skin"].items():
        sk.set_editor_property(k, v)
    ss.set_editor_property("skin", sk)
    sub.commit_skin_settings(ch, ss)
RESULT["t_body_skin_s"] = round(time.time() - t0, 2)

for a in eas.get_all_level_actors():
    if a.get_class().get_name() == "MetaHumanDefaultEditorPipelineActor":
        eas.destroy_actor(a)
preview = sub.spawn_meta_human_actor(ch, True)
preview.set_actor_location(unreal.Vector(0, 0, -100000.0), False, False)
_tex_cache = {}


def _persist_texture(tex, tag):
    if tex is None or not tex.get_path_name().startswith("/Engine/Transient"):
        return tex
    key = tex.get_path_name()
    if key not in _tex_cache:
        _tex_cache[key] = AT.duplicate_asset(f"T_{tag}_{len(_tex_cache)}", DEST, tex)
    return _tex_cache[key]


def _persist_material(mat, tag):
    if mat is None or not mat.get_path_name().startswith("/Engine/Transient"):
        return mat
    parent = mat.get_editor_property("parent")
    while parent is not None and parent.get_path_name().startswith("/Engine/Transient"):
        parent = parent.get_editor_property("parent")
    mic = AT.create_asset(f"MI_{tag}", DEST, unreal.MaterialInstanceConstant,
                          unreal.MaterialInstanceConstantFactoryNew())
    MEL.set_material_instance_parent(mic, parent)
    for pv in mat.get_editor_property("scalar_parameter_values"):
        MEL.set_material_instance_scalar_parameter_value(
            mic, pv.get_editor_property("parameter_info").get_editor_property("name"),
            pv.get_editor_property("parameter_value"))
    for pv in mat.get_editor_property("vector_parameter_values"):
        MEL.set_material_instance_vector_parameter_value(
            mic, pv.get_editor_property("parameter_info").get_editor_property("name"),
            pv.get_editor_property("parameter_value"))
    for pv in mat.get_editor_property("texture_parameter_values"):
        MEL.set_material_instance_texture_parameter_value(
            mic, pv.get_editor_property("parameter_info").get_editor_property("name"),
            _persist_texture(pv.get_editor_property("parameter_value"), tag))
    return mic


state = {"n": 0}


@guard
def tick(_dt):
    state["n"] += 1
    if state["n"] < P.get("settle_ticks", 60):
        return
    unreal.unregister_slate_post_tick_callback(state["h"])
    t1 = time.time()
    out = {}
    comps = {c.get_name(): c for c in preview.get_components_by_class(unreal.SkinnedMeshComponent)}
    for part in ("Body", "Face"):
        if part not in comps:
            continue
        src = comps[part].get_skinned_asset()
        path = f"{DEST}/SKM_{P['name']}_{part}"
        if EAL.does_asset_exist(path):
            EAL.delete_asset(path)
        new = AT.duplicate_asset(f"SKM_{P['name']}_{part}", DEST, src)
        mats = list(new.get_editor_property("materials"))
        for i, sm in enumerate(mats):
            mi = sm.get_editor_property("material_interface")
            sm.set_editor_property("material_interface",
                                   _persist_material(mi, f"{P['name']}_{part}_{i}"))
        new.set_editor_property("materials", mats)
        try:
            pa = new.get_editor_property("physics_asset")
            if pa is not None and pa.get_path_name().startswith("/Engine/Transient"):
                new.set_editor_property("physics_asset", None)
        except Exception as exc:                            # noqa: BLE001
            unreal.log_warning(f"[mh_bake] physics_asset: {exc}")
        out[part] = {"mesh": new.get_path_name(),
                     "skeleton": new.get_editor_property("skeleton").get_path_name(),
                     "materials": [str(m.get_editor_property("material_interface").get_path_name())
                                   if m.get_editor_property("material_interface") else None
                                   for m in mats]}
    grooms = {}
    for g in preview.get_components_by_class(unreal.GroomComponent):
        asset = g.get_editor_property("groom_asset")
        grooms[g.get_name()] = asset.get_path_name() if asset else None
    cloth = []
    for i, c in enumerate(preview.get_components_by_class(unreal.ChaosClothComponent)):
        ca = c.get_editor_property("asset")
        if ca is None:
            continue
        cpath = f"{DEST}/OA_{P['name']}_{i}"
        if EAL.does_asset_exist(cpath):
            EAL.delete_asset(cpath)
        try:
            baked = AT.duplicate_asset(f"OA_{P['name']}_{i}", DEST, ca)
            cloth.append({"src": ca.get_path_name(), "baked": baked.get_path_name() if baked else None,
                          "class": ca.get_class().get_name()})
        except Exception as exc:                            # noqa: BLE001
            cloth.append({"src": ca.get_path_name(), "error": repr(exc)[:300]})
    if "Body" in out:
        bm = unreal.load_asset(out["Body"]["mesh"])
        ext = bm.get_bounds().get_editor_property("box_extent")
        out["Body"]["bounds_height_cm"] = round(2 * ext.z, 1)
    saved = EAL.save_directory(DEST, only_if_is_dirty=False, recursive=True)
    finish({"parts": out, "cloth_assets_in_preview": cloth, "grooms_in_preview": grooms, "saved": bool(saved),
            "n_textures_persisted": len(_tex_cache), "t_bake_s": round(time.time() - t1, 2),
            "t_total_s": round(time.time() - t0, 2)})


state["h"] = unreal.register_slate_post_tick_callback(tick)
RESULT["async"] = True
RESULT["_handles"] = [state["h"]]

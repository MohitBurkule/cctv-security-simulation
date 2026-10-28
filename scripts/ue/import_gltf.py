"""Import rigged glTF humans (mesh + skeleton + clips) and write a catalogue.

Runs as a job in the persistent editor (tools/ue_job.py) or standalone with
-run=pythonscript. Sources: every `kind: human` entry of ~/data/assets/index.json
(schema {kind, path, format, license, anim_clips, height_m}) plus the Khronos
CesiumMan fallback. Already-imported characters are skipped (cached by name).

Writes CATALOG (JSON): {name: {mesh, walk, idle, height_cm, yaw_offset,
tint_params, license}} — the host reads it to assign characters to tracks.
Mesh height is MEASURED from the imported bounds, not trusted from the index.
"""
from __future__ import annotations

import json
import os
import re

import unreal

INDEX = os.path.expanduser("~/data/assets/index.json")
CATALOG = os.path.expanduser("~/data/ue-projects/SurvSim/Saved/survsim_characters.json")
FALLBACK = [{"kind": "human", "path": "~/data/glb/CesiumMan.glb", "format": "glb",
             "license": "CC-BY-4.0 (Khronos sample)", "anim_clips": ["walk"],
             "height_m": 1.5, "name": "CesiumMan"}]
NORM_DIR = os.path.expanduser("~/data/assets/characters_norm")
HUMAN_KINDS = ("human", "character", "person")
_eal = unreal.EditorAssetLibrary


def _entries():
    try:
        with open(INDEX) as fh:
            data = json.load(fh)
        root = "" if isinstance(data, list) else data.get("root", "")
        data = data if isinstance(data, list) else data.get("assets", [])
    except (OSError, ValueError):
        data, root = [], ""
    for d in data:
        if root and not os.path.isabs(os.path.expanduser(d.get("path", ""))):
            d.setdefault("_abs", os.path.join(root, d["path"]))
        # Prefer the armature-baked copy (survsim/export/gltf_normalize.py).
        norm = os.path.join(NORM_DIR, os.path.basename(d.get("path", "")))
        if os.path.isfile(norm):
            d["_abs"] = norm
            d["name"] = d.get("name", os.path.splitext(os.path.basename(norm))[0]) + "_m"
    humans = [d for d in data if str(d.get("kind", "")).lower() in HUMAN_KINDS
              and str(d.get("format", d.get("path", ""))).lower().rsplit(".", 1)[-1]
              in ("glb", "gltf")]
    return humans + FALLBACK


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def _pick(anims, words):
    for w in words:
        hits = [a for a in anims if w in a.get_name().lower()]
        # Prefer the plain clip ("Walk") over variants ("Walk_Carry", "WalkRun").
        hits.sort(key=lambda a: len(a.get_name()))
        if hits:
            return hits[0]
    return None


def _head_top_z(mesh):
    sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor = sub.spawn_actor_from_class(unreal.SkeletalMeshActor, unreal.Vector(0, 0, 0))
    try:
        comp = actor.skeletal_mesh_component
        comp.set_skeletal_mesh_asset(mesh)
        best = None
        names = list(comp.get_all_socket_names())
        # [V] Quaternius armatures carry a x100 root-bone scale: bone positions
        # read 476 cm while the skinned mesh renders 100x that. Multiply it in.
        root_scale = comp.get_socket_transform(
            names[0], unreal.RelativeTransformSpace.RTS_WORLD).scale3d.z if names else 1.0
        for n in names:
            low = str(n).lower()
            if "headtop" in low or low in ("head_end", "head_top", "headtop_end"):
                z = comp.get_socket_location(n).z
                best = z if best is None else max(best, z)
        return None if best is None else best * root_scale
    finally:
        sub.destroy_actor(actor)


def import_one(entry):
    src = entry.get("_abs") or os.path.expanduser(entry["path"])
    name = _safe(entry.get("name") or os.path.splitext(os.path.basename(src))[0])
    dest = f"/Game/SurvSim/Chars/{name}"
    if not _eal.list_assets(dest, True):
        task = unreal.AssetImportTask()
        task.filename = src
        task.destination_path = dest
        task.automated = True
        task.replace_existing = True
        task.save = True
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    loaded = [_eal.load_asset(str(p)) for p in _eal.list_assets(dest, True)]
    meshes = [a for a in loaded if isinstance(a, unreal.SkeletalMesh)]
    anims = [a for a in loaded if isinstance(a, unreal.AnimSequence)]
    if not meshes or not anims:
        return name, {"error": f"meshes={len(meshes)} anims={len(anims)}"}
    mesh = meshes[0]
    walk = _pick(anims, ["walk"]) or anims[0]
    idle = _pick(anims, ["idle", "stand"])
    box = mesh.get_bounds().box_extent
    origin = mesh.get_bounds().origin
    height = 2.0 * box.z
    # [X] Quaternius meshes report bounds of ~100 m (physics/bone bounds), so a
    # head-end bone in the reference pose is the height authority when present.
    head_z = _head_top_z(mesh)
    if head_z:
        height = head_z
    return name, {
        "mesh": mesh.get_path_name(),
        "walk": walk.get_path_name(),
        "walk_len_s": round(walk.get_play_length(), 3),
        "idle": idle.get_path_name() if idle else None,
        "anims": [a.get_name() for a in anims][:40],
        "height_cm": round(height, 2),
        "z_min_cm": round(origin.z - box.z, 2),
        "yaw_offset": float(entry.get("yaw_offset", -90.0)),
        "tint_params": ["BaseColorFactor"],
        "license": entry.get("license"),
        "source": src,
    }


def _name(entry):
    src = entry.get("_abs") or os.path.expanduser(entry["path"])
    return _safe(entry.get("name") or os.path.splitext(os.path.basename(src))[0])


def run(only=None):
    """Import all (or only the named) characters; MERGES into the catalogue.

    One character per job is the safe mode: an Interchange crash on one glTF
    then costs one entry, and the catalogue already holds the others.
    """
    try:
        with open(CATALOG) as fh:
            cat = json.load(fh)
    except (OSError, ValueError):
        cat = {}
    for entry in _entries():
        if only is not None and _name(entry) != only:
            continue
        try:
            name, info = import_one(entry)
        except Exception as exc:                                # noqa: BLE001
            name, info = entry.get("path"), {"error": f"{type(exc).__name__}: {exc}"[:300]}
        cat[name] = info
    os.makedirs(os.path.dirname(CATALOG), exist_ok=True)
    with open(CATALOG, "w") as fh:
        json.dump(cat, fh, indent=2)
    return cat


if __name__ in ("__main__", "__survsim_job__"):
    _cat = run(globals().get("ONLY"))
    try:
        RESULT["characters"] = {k: v.get("error") or v.get("height_cm") for k, v in _cat.items()}
    except NameError:
        with open("/tmp/survsim_gltf_import.json", "w") as _fh:
            json.dump(_cat, _fh, indent=2)

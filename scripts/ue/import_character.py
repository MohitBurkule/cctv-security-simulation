"""Import a rigged humanoid FBX so renders contain people, not cubes.

Until now the render path used scaled cubes as person stand-ins. That is fine
for proving geometry, timing and multi-camera views, and useless for the thing
the renders exist for: a COCO-pretrained person detector scores 0 AP on a cube,
so no detection, tracking or re-identification baseline computed from such
frames would mean anything.

A Mixamo character is a rigged humanoid at correct human scale, which is all
that is needed to make the frames consumable. It is *not* a claim about
photorealism — it is a low-poly game character, and any realism statistic
measured from it should say so.

Run inside the editor:

    UnrealEditor SurvSim.uproject -run=pythonscript \
        -script=scripts/ue/import_character.py -unattended -nosplash

Writes /tmp/survsim_import_result.json, because the editor reports success for
scripts that did nothing.
"""

from __future__ import annotations

import json

try:
    import unreal
except ImportError:  # pragma: no cover - editor only
    raise SystemExit("must run inside Unreal Editor")

SOURCE = "@SOURCE@"
DESTINATION = "/Game/SurvSim/Chars"
ASSET_NAME = "@NAME@"
RESULT = "/tmp/survsim_import_result.json"


def main() -> int:
    out: dict = {}
    try:
        task = unreal.AssetImportTask()
        task.filename = SOURCE
        task.destination_path = DESTINATION
        task.destination_name = ASSET_NAME
        task.automated = True          # no modal dialog in an unattended run
        task.replace_existing = True
        task.save = True

        options = unreal.FbxImportUI()
        # A Mixamo character is a skeletal mesh; importing it as a static mesh
        # silently produces a T-posed prop with no skeleton, which looks right
        # in a thumbnail and cannot be animated.
        options.set_editor_property("import_mesh", True)
        options.set_editor_property("import_as_skeletal", True)
        options.set_editor_property("import_materials", True)
        options.set_editor_property("import_textures", True)
        options.set_editor_property("import_animations", False)
        options.set_editor_property(
            "mesh_type_to_import", unreal.FBXImportType.FBXIT_SKELETAL_MESH)

        skeletal = options.skeletal_mesh_import_data
        skeletal.set_editor_property("import_morph_targets", False)
        skeletal.set_editor_property("update_skeleton_reference_pose", False)
        skeletal.set_editor_property(
            "normal_import_method",
            unreal.FBXNormalImportMethod.FBXNIM_IMPORT_NORMALS_AND_TANGENTS)
        task.options = options

        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
        imported = list(task.get_editor_property("imported_object_paths") or [])
        out["imported"] = imported
        out["ok"] = bool(imported)

        # Report what actually landed, by class — "imported" alone has been
        # true for assets of the wrong type before.
        details = []
        for path in imported:
            asset = unreal.EditorAssetLibrary.load_asset(path)
            if asset is not None:
                details.append({"path": path, "class": type(asset).__name__})
        out["assets"] = details
        out["skeletal_meshes"] = [d["path"] for d in details
                                  if d["class"] == "SkeletalMesh"]
    except Exception as exc:                                   # noqa: BLE001
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"[:400]

    with open(RESULT, "w") as handle:
        json.dump(out, handle, indent=2)
    return 0


main()

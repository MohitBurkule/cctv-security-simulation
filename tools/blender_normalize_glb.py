"""Re-export a rigged glTF through Blender with armature transforms applied.

    blender -b -P tools/blender_normalize_glb.py -- in.glb out.glb

[X] Quaternius glTFs put rotation(-90 X) and scale 100 on the armature NODE,
above the skin's skeleton root. UE 5.8 Interchange applies that node to the
mesh but not to the animation, so imported walkers render face-down with
stretched limbs. Applying rotation+scale to the armature in Blender (and
rescaling bone-location f-curves by the same factor) bakes it into the rest
pose, which every importer then agrees on.
"""
import sys

import bpy

src, dst = sys.argv[sys.argv.index("--") + 1:][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
for arm in arms:
    s = arm.scale[0]
    for o in bpy.context.scene.objects:
        o.select_set(False)
    arm.select_set(True)
    for child in arm.children:
        child.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    if abs(s - 1.0) > 1e-6:
        for action in bpy.data.actions:
            for fc in action.fcurves:
                if fc.data_path.endswith(".location"):
                    for kp in fc.keyframe_points:
                        kp.co[1] *= s
                        kp.handle_left[1] *= s
                        kp.handle_right[1] *= s
bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB", export_animations=True,
                          export_apply=False, export_yup=True)
print("NORMALIZED", src, "->", dst, "armatures", len(arms))

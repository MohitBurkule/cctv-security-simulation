"""Painted clothing for baked MetaHuman bodies (runs inside the editor).

The offline bodies have no garments (docs/12: one default garment ships, and
only some bakes carry it), so this master material paints clothing onto the
body by region, from the PRE-SKINNED local position (the reference pose, so a
region sticks to its limb while the body animates):

    h = z / stature, ax = |x| (arms spread along X in the A-pose reference)
    shoes      h < 0.055
    bottom     HemH <= h < 0.575, not arm       (trousers / shorts / skirt)
    top        0.575 <= h < 0.83, not arm
    sleeve     arm and u < SleeveU, u = (ax - shoulder) / (XMax - shoulder)
    skin       everything else (hands, lower legs under shorts, neck)

Parameters per person come from survsim/export/outfits.py (host side).
Instances are MaterialInstanceDynamic on the body slot (same mechanism as the
CesiumMan tint, which MRQ renders).
"""
from __future__ import annotations

import unreal

MAT_DIR = "/Game/SurvSim/Mats"
VERSION = 5
MASTER = f"M_SurvMHCloth_v{VERSION}"
_mel = unreal.MaterialEditingLibrary
_eal = unreal.EditorAssetLibrary

SCALARS = {"Stature": 170.0, "XMax": 46.0, "HemH": 0.055, "SleeveU": 0.8, "Rough": 0.8,
           "IsFace": 0.0, "FaceZ0": 140.0, "FaceZ1": 175.0}
VECTORS = {"TopCol": (0.02, 0.02, 0.02), "BottomCol": (0.04, 0.07, 0.15),
           "ShoeCol": (0.01, 0.01, 0.01), "SkinCol": (0.45, 0.28, 0.19),
           "HairCol": (0.02, 0.012, 0.008)}

_CODE = r"""
float h = P.z / max(Stature, 1.0);
float ax = abs(P.x);
float shoulder = 0.105 * Stature;
bool arm = (ax > shoulder) && (h > 0.42);
float u = saturate((ax - shoulder) / max(XMax - shoulder, 1.0));
// cheap fabric shading: fine weave + a few soft folds, never flat colour
float n = frac(sin(dot(floor(P.xyz * 1.5), float3(12.9898, 78.233, 37.719))) * 43758.5453);
float fold = 0.92 + 0.08 * sin(P.z * 0.9 + 1.7 * sin(P.x * 0.35));
float3 c = SkinCol;
float r = 0.55;
if (h < 0.055) { c = ShoeCol; r = 0.45; }
else if (!arm && h >= HemH && h < 0.575) { c = BottomCol * fold * (0.95 + 0.1 * n); r = Rough; }
else if (!arm && h >= 0.575 && h < 0.83) { c = TopCol * fold * (0.95 + 0.1 * n); r = Rough; }
else if (arm && u < SleeveU) { c = TopCol * fold * (0.95 + 0.1 * n); r = Rough; }
// the face mesh (own local space, its bounds passed in): skin, hair on the
// crown and the back of the head
if (IsFace > 0.5) {
    float fz = saturate((P.z - FaceZ0) / max(FaceZ1 - FaceZ0, 1.0));
    c = SkinCol; r = 0.5;
    if (fz > 0.82 || (fz > 0.55 && P.y < -4.0)) { c = HairCol * (0.9 + 0.2 * n); r = 0.45; }
}
// Shoulder "wings" were missing RigLogic correctives, not geometry to mask:
// scripts/ue/mh_postprocess.py sets ABP_Body_PostProcess on the meshes.
float o = 1.0;
"""


def _custom(mat, out_type, ret, names, x, y):
    node = _mel.create_material_expression(mat, unreal.MaterialExpressionCustom, x, y)
    node.set_editor_property("code", _CODE + ret)
    node.set_editor_property("output_type", out_type)
    cins = []
    for nm in names:
        ci = unreal.CustomInput()
        ci.set_editor_property("input_name", nm)
        cins.append(ci)
    node.set_editor_property("inputs", cins)
    return node


def ensure_master():
    path = f"{MAT_DIR}/{MASTER}"
    if _eal.does_asset_exist(path):
        return _eal.load_asset(path)
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    mat = tools.create_asset(MASTER, MAT_DIR, unreal.Material, unreal.MaterialFactoryNew())
    try:
        return _build(mat)
    except Exception:
        # [X] a half-built master (exception mid-graph) was saved empty and
        # rendered every body solid black on the next run: never keep one.
        _eal.delete_asset(mat.get_path_name())
        raise


def _build(mat):
    pos = _mel.create_material_expression(mat, unreal.MaterialExpressionPreSkinnedPosition,
                                          -900, 0)
    # [X] PreSkinnedPosition is vertex-shader only: wired straight into a
    # pixel Custom node the material fails to compile (0 instructions) and
    # every body renders in the grey default material. Interpolate it.
    vi = _mel.create_material_expression(mat, unreal.MaterialExpressionVertexInterpolator, -700, 0)
    _mel.connect_material_expressions(pos, "", vi, "")
    sources = {"P": vi}
    y = 100
    for nm, val in SCALARS.items():
        p = _mel.create_material_expression(mat, unreal.MaterialExpressionScalarParameter, -900, y)
        p.set_editor_property("parameter_name", nm)
        p.set_editor_property("default_value", val)
        sources[nm] = p
        y += 60
    for nm, (r, g, b) in VECTORS.items():
        p = _mel.create_material_expression(mat, unreal.MaterialExpressionVectorParameter, -900, y)
        p.set_editor_property("parameter_name", nm)
        p.set_editor_property("default_value", unreal.LinearColor(r, g, b, 1.0))
        sources[nm] = p
        y += 120
    names = list(sources)
    f3, f1 = unreal.CustomMaterialOutputType.CMOT_FLOAT3, unreal.CustomMaterialOutputType.CMOT_FLOAT1
    mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_MASKED)
    for ret, ot, prop, yy in (("return c;", f3, unreal.MaterialProperty.MP_BASE_COLOR, 0),
                              ("return r;", f1, unreal.MaterialProperty.MP_ROUGHNESS, 300),
                              ("return o;", f1, unreal.MaterialProperty.MP_OPACITY_MASK, 600)):
        node = _custom(mat, ot, ret, names, -300, yy)
        for nm in names:
            _mel.connect_material_expressions(sources[nm], "", node, nm)
        _mel.connect_material_property(node, "", prop)
    mat.set_editor_property("used_with_skeletal_mesh", True)
    _mel.recompile_material(mat)
    _eal.save_asset(mat.get_path_name())
    if not _mel.get_vector_parameter_names(mat):
        raise RuntimeError(f"{MASTER}: graph is empty after build")
    st = _mel.get_statistics(mat)
    if int(st.get_editor_property("num_pixel_shader_instructions")) <= 0:
        raise RuntimeError(f"{MASTER}: failed to compile (0 pixel instructions)")
    return mat


def dress(comp, outfit: dict, stature_cm: float, xmax_cm: float, face: bool = False):
    """Paint `outfit` (survsim.export.outfits.outfit) on every slot of `comp`."""
    master = ensure_master()
    mids = []
    fz = None
    if face:
        b = comp.get_skeletal_mesh_asset().get_bounds()
        fz = (b.origin.z - b.box_extent.z, b.origin.z + b.box_extent.z)
    for i in range(len(comp.get_materials())):
        mid = comp.create_dynamic_material_instance(i, master)
        if mid is None:
            continue
        if fz:
            for k, v in (("IsFace", 1.0), ("FaceZ0", fz[0]), ("FaceZ1", fz[1])):
                mid.set_scalar_parameter_value(k, float(v))
        for k, v in (("Stature", stature_cm), ("XMax", xmax_cm), ("HemH", outfit["hem_h"]),
                     ("SleeveU", outfit["sleeve_u"]), ("Rough", outfit["roughness"])):
            mid.set_scalar_parameter_value(k, float(v))
        for k, key in (("TopCol", "top_rgb"), ("BottomCol", "bottom_rgb"),
                       ("ShoeCol", "shoes_rgb"), ("SkinCol", "skin_rgb"), ("HairCol", "hair_rgb")):
            if key in outfit:
                mid.set_vector_parameter_value(k, unreal.LinearColor(*outfit[key], 1.0))
        mids.append(mid)
    return len(mids)


# ---------------------------------------------------------------- CC0 bodies
# The rebaked Quaternius bodies (docs/research/47 option A) are arms-down,
# partly chibi and often UV-less, so regions come from the skin weights instead:
# survsim/export/cc0_regions.py bakes a region code into COLOR_0.r
# (0 skin, 1 top, 2 forearm, 3 thigh, 4 shin, 5 shoes, 6 hair, 7 eyes).
# v2 (2026-09-26): + a face-cover proxy for face_visible=false appearance
# epochs (hood / cap / balaclava): with CoverOn, hair and head skin above the
# local height CoverZ take CoverCol. CoverOn=0 renders exactly as v1.
CC0_VERSION = 2
CC0_MASTER = f"M_SurvCC0Cloth_v{CC0_VERSION}"
CC0_SCALARS = {"LongSleeve": 1.0, "LongLeg": 1.0, "Rough": 0.8, "CoverOn": 0.0, "CoverZ": 1e6}
CC0_VECTORS = dict(VECTORS, CoverCol=(0.02, 0.02, 0.025))
COVER_FRAC = 0.84   # chin/neck line as a fraction of the mesh's height
_CC0_CODE = r"""
int code = (int)round(saturate(VC.r) * 7.0);
float n = frac(sin(dot(floor(P.xyz * 1.5), float3(12.9898, 78.233, 37.719))) * 43758.5453);
float fold = 0.92 + 0.08 * sin(P.z * 0.9 + 1.7 * sin(P.x * 0.35));
float3 cloth = fold * (0.95 + 0.1 * n);
float3 c = SkinCol;
float r = 0.55;
if (code == 1 || (code == 2 && LongSleeve > 0.5)) { c = TopCol * cloth; r = Rough; }
else if (code == 3 || (code == 4 && LongLeg > 0.5)) { c = BottomCol * cloth; r = Rough; }
else if (code == 5) { c = ShoeCol; r = 0.45; }
else if (code == 6) { c = HairCol * (0.9 + 0.2 * n); r = 0.45; }
else if (code == 7) { c = float3(0.02, 0.02, 0.02); r = 0.2; }
if (CoverOn > 0.5 && (code == 6 || code == 7 || (code == 0 && P.z > CoverZ))) {
    c = CoverCol * cloth; r = 0.75;
}
"""


def ensure_cc0_master():
    path = f"{MAT_DIR}/{CC0_MASTER}"
    if _eal.does_asset_exist(path):
        return _eal.load_asset(path)
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    mat = tools.create_asset(CC0_MASTER, MAT_DIR, unreal.Material, unreal.MaterialFactoryNew())
    try:
        pos = _mel.create_material_expression(mat, unreal.MaterialExpressionPreSkinnedPosition,
                                              -900, 0)
        vi = _mel.create_material_expression(mat, unreal.MaterialExpressionVertexInterpolator,
                                             -700, 0)
        _mel.connect_material_expressions(pos, "", vi, "")
        vc = _mel.create_material_expression(mat, unreal.MaterialExpressionVertexColor, -900, -150)
        sources = {"P": vi, "VC": vc}
        y = 100
        for nm, val in CC0_SCALARS.items():
            p = _mel.create_material_expression(mat, unreal.MaterialExpressionScalarParameter, -900, y)
            p.set_editor_property("parameter_name", nm)
            p.set_editor_property("default_value", val)
            sources[nm] = p
            y += 60
        for nm, (r, g, b) in CC0_VECTORS.items():
            p = _mel.create_material_expression(mat, unreal.MaterialExpressionVectorParameter, -900, y)
            p.set_editor_property("parameter_name", nm)
            p.set_editor_property("default_value", unreal.LinearColor(r, g, b, 1.0))
            sources[nm] = p
            y += 120
        names = list(sources)
        f3 = unreal.CustomMaterialOutputType.CMOT_FLOAT3
        f1 = unreal.CustomMaterialOutputType.CMOT_FLOAT1
        for ret, ot, prop, yy in (("return c;", f3, unreal.MaterialProperty.MP_BASE_COLOR, 0),
                                  ("return r;", f1, unreal.MaterialProperty.MP_ROUGHNESS, 300)):
            node = _mel.create_material_expression(mat, unreal.MaterialExpressionCustom, -300, yy)
            node.set_editor_property("code", _CC0_CODE + ret)
            node.set_editor_property("output_type", ot)
            cins = []
            for nm in names:
                ci = unreal.CustomInput()
                ci.set_editor_property("input_name", nm)
                cins.append(ci)
            node.set_editor_property("inputs", cins)
            for nm in names:
                # VertexColor's first output is RGB; the RGBA pin is unnamed too.
                _mel.connect_material_expressions(sources[nm], "", node, nm)
            _mel.connect_material_property(node, "", prop)
        mat.set_editor_property("used_with_skeletal_mesh", True)
        _mel.recompile_material(mat)
        _eal.save_asset(mat.get_path_name())
        st = _mel.get_statistics(mat)
        if int(st.get_editor_property("num_pixel_shader_instructions")) <= 0:
            raise RuntimeError(f"{CC0_MASTER}: failed to compile (0 pixel instructions)")
        return mat
    except Exception:
        _eal.delete_asset(mat.get_path_name())
        raise


def dress_cc0(actor, outfit: dict) -> int:
    """Paint `outfit` on a region-coded CC0 body (every material slot)."""
    master = ensure_cc0_master()
    comp = actor.skeletal_mesh_component
    cover = outfit.get("face_cover")
    cover_z = None
    if cover:
        b = comp.get_skeletal_mesh_asset().get_bounds()
        z0, z1 = b.origin.z - b.box_extent.z, b.origin.z + b.box_extent.z
        cover_z = z0 + COVER_FRAC * (z1 - z0)
    n = 0
    for i in range(len(comp.get_materials())):
        mid = comp.create_dynamic_material_instance(i, master)
        if mid is None:
            continue
        mid.set_scalar_parameter_value("LongSleeve", 1.0 if outfit["sleeve_u"] > 0.5 else 0.0)
        mid.set_scalar_parameter_value("LongLeg", 1.0 if outfit["hem_h"] < 0.1 else 0.0)
        mid.set_scalar_parameter_value("Rough", float(outfit["roughness"]))
        for k, key in (("TopCol", "top_rgb"), ("BottomCol", "bottom_rgb"),
                       ("ShoeCol", "shoes_rgb"), ("SkinCol", "skin_rgb"), ("HairCol", "hair_rgb")):
            if key in outfit:
                mid.set_vector_parameter_value(k, unreal.LinearColor(*outfit[key], 1.0))
        if cover:
            mid.set_scalar_parameter_value("CoverOn", 1.0)
            mid.set_scalar_parameter_value("CoverZ", float(cover_z))
            mid.set_vector_parameter_value("CoverCol", unreal.LinearColor(*cover["rgb"], 1.0))
        n += 1
    return n


def dress_actor(actor, outfit: dict, stature_cm: float, xmax_cm: float):
    """Body and face (every SkeletalMeshComponent) get the outfit material."""
    body = actor.skeletal_mesh_component
    return sum(dress(c, outfit, stature_cm, xmax_cm, face=c != body)
               for c in actor.get_components_by_class(unreal.SkeletalMeshComponent))

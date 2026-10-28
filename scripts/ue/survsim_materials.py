"""Procedural surface materials built in Python, cached by versioned name.

One master material (`M_SurvProc_v<N>`) computes asphalt, paving slabs, brick,
concrete, grass and painted metal from WORLD position with a Custom HLSL node,
plus a facade window grid whose panes glow at night. Everything is keyed to
world metres, so a 40 m slab and a 1 m bin get the same texel density and no
UVs are needed on the engine's basic shapes.

Material instances (`MI_<name>`) are created once and only have their
parameters updated on later calls, so an edit costs no shader recompile.

When real ambientCG PBR sets exist (~/data/assets/index.json), `pbr_instance`
builds a texture-sampled instance of `M_SurvPBR_v<N>` instead; the procedural
set is the fallback, never a stub.
"""
from __future__ import annotations

import unreal

MAT_DIR = "/Game/SurvSim/Mats"
VERSION = 5
MASTER = f"M_SurvProc_v{VERSION}"
PBR_VERSION = 9
PBR_MASTER = f"M_SurvPBR_v{PBR_VERSION}"
# role -> {"maps", "tile_m", "aspect", "tint", "set"} from survsim/export/pbr_sets.py,
# set by the render job before library(); empty = all procedural.
PBR: dict = {}

# Pattern ids (the `Pattern` scalar).
ASPHALT, SLABS, BRICK, CONCRETE, GRASS, PAINT = 0, 1, 2, 3, 4, 5

_BODY = r"""
#ifndef SURVSIM_H2
#define SURVSIM_H2 1
#define H2(q) frac(sin(dot((q), float2(127.1, 311.7))) * 43758.5453)
#endif
float3 wp = P / 100.0;
float3 an = abs(N);
float2 uv;
if (an.z > 0.6) uv = wp.xy; else if (an.x > an.y) uv = float2(wp.y, wp.z); else uv = float2(wp.x, wp.z);
float n = 0.0; float amp = 0.5; float2 q = uv * NoiseScale + Seed;
for (int o = 0; o < 5; o++) {
  float2 i0 = floor(q); float2 f0 = frac(q); f0 = f0 * f0 * (3.0 - 2.0 * f0);
  float a = H2(i0); float b = H2(i0 + float2(1, 0));
  float c = H2(i0 + float2(0, 1)); float d = H2(i0 + float2(1, 1));
  n += amp * lerp(lerp(a, b, f0.x), lerp(c, d, f0.x), f0.y);
  q *= 2.07; amp *= 0.5;
}
float fine = H2(floor(uv * 90.0));
float3 col = lerp(ColA.rgb, ColB.rgb, saturate(n * 1.6 - 0.3));
float rough = Rough;
float3 emis = float3(0, 0, 0);
float tile = max(Tile, 0.01);
if (Pattern < 0.5) {
  col *= 0.82 + 0.36 * fine;
  col = lerp(col, col * 0.55, smoothstep(0.66, 0.74, n));
  rough = saturate(Rough + 0.08 * fine - 0.25 * smoothstep(0.66, 0.74, n));
} else if (Pattern < 1.5) {
  float2 g = uv / tile; float2 id = floor(g); float2 f = frac(g);
  float e = min(min(f.x, 1 - f.x), min(f.y, 1 - f.y)) * tile;
  float h = H2(id + 7.3);
  col *= (0.9 + 0.2 * h) * (0.9 + 0.2 * fine);
  float gr = 1.0 - smoothstep(0.004, 0.010, e);
  col = lerp(col, ColC.rgb, gr);
  rough = lerp(Rough, 0.97, gr);
} else if (Pattern < 2.5) {
  float bw = 0.225; float bh = 0.075;
  float2 g = float2(uv.x / bw, uv.y / bh);
  g.x += 0.5 * fmod(floor(g.y), 2.0);
  float2 id = floor(g); float2 f = frac(g);
  float m = 1.0 - smoothstep(0.004, 0.008, min(min(f.x, 1 - f.x) * bw, min(f.y, 1 - f.y) * bh));
  float h = H2(id);
  col = lerp(ColA.rgb, ColB.rgb, h) * (0.8 + 0.4 * n) * (0.9 + 0.2 * fine);
  col = lerp(col, ColC.rgb, m);
} else if (Pattern < 3.5) {
  col *= 0.9 + 0.2 * fine;
  float2 f = frac(uv / tile);
  float e = min(min(f.x, 1 - f.x), min(f.y, 1 - f.y)) * tile;
  col *= lerp(0.72, 1.0, smoothstep(0.003, 0.012, e));
} else if (Pattern < 4.5) {
  col *= 0.65 + 0.7 * fine;
} else {
  col *= 0.94 + 0.12 * fine;
}
if (WinSpacing > 0.01 && an.z < 0.5) {
  float2 g = float2(uv.x / WinSpacing, uv.y / FloorH);
  float2 id = floor(g); float2 f = frac(g);
  float groundf = step(id.y, 0.5);
  float2 lo = lerp(float2(0.24, 0.30), float2(0.08, 0.06), groundf);
  float2 hi = lerp(float2(0.76, 0.82), float2(0.92, 0.78), groundf);
  float inw = step(lo.x, f.x) * step(f.x, hi.x) * step(lo.y, f.y) * step(f.y, hi.y);
  float fr = step(lo.x - 0.03, f.x) * step(f.x, hi.x + 0.03) * step(lo.y - 0.04, f.y) * step(f.y, hi.y + 0.03) - inw;
  float h = H2(id + Seed * 3.1);
  float3 glass = float3(0.025, 0.03, 0.04) + 0.05 * h + 0.10 * saturate((f.y - lo.y) / (hi.y - lo.y) - 0.4);
  col = lerp(col, float3(0.55, 0.55, 0.52) * (0.6 + 0.4 * h), fr);
  col = lerp(col, glass, inw);
  rough = lerp(rough, 0.06, inw);
  float lit = step(0.42, h);
  emis = inw * lit * float3(1.0, 0.8, 0.55) * (0.4 + 1.2 * H2(id + 5.7)) * Night * WinEmissive;
}
if (an.z < 0.5 && Pattern < 4.5) col *= lerp(0.7, 1.0, saturate(wp.z / 1.0 + 0.25 * n));
float wet = Wet * step(0.6, an.z) * saturate(1.2 - 0.6 * n);
col *= lerp(1.0, 0.55, wet);
rough = lerp(rough, 0.12, wet);
"""

_SCALARS = {"Pattern": 0.0, "Tile": 0.6, "Rough": 0.85, "NoiseScale": 0.8,
            "Seed": 0.0, "WinSpacing": 0.0, "FloorH": 3.2, "Night": 0.0,
            "WinEmissive": 0.5, "Wet": 0.0}
_VECTORS = {"ColA": (0.1, 0.1, 0.1), "ColB": (0.2, 0.2, 0.2), "ColC": (0.3, 0.3, 0.3)}

_mel = unreal.MaterialEditingLibrary
_eal = unreal.EditorAssetLibrary


def _tools():
    return unreal.AssetToolsHelpers.get_asset_tools()


def _custom(mat, out_type, ret, inputs, x, y):
    node = _mel.create_material_expression(mat, unreal.MaterialExpressionCustom, x, y)
    node.set_editor_property("code", _BODY + "\n" + ret)
    node.set_editor_property("output_type", out_type)
    cins = []
    for name in inputs:
        ci = unreal.CustomInput()
        ci.set_editor_property("input_name", name)
        cins.append(ci)
    node.set_editor_property("inputs", cins)
    return node


def ensure_master():
    path = f"{MAT_DIR}/{MASTER}"
    if _eal.does_asset_exist(path):
        return _eal.load_asset(path)
    mat = _tools().create_asset(MASTER, MAT_DIR, unreal.Material, unreal.MaterialFactoryNew())
    wp = _mel.create_material_expression(mat, unreal.MaterialExpressionWorldPosition, -900, 0)
    nrm = _mel.create_material_expression(mat, unreal.MaterialExpressionVertexNormalWS, -900, 100)
    sources = {"P": (wp, ""), "N": (nrm, "")}
    y = 200
    for name, val in _SCALARS.items():
        p = _mel.create_material_expression(mat, unreal.MaterialExpressionScalarParameter, -900, y)
        p.set_editor_property("parameter_name", name)
        p.set_editor_property("default_value", val)
        sources[name] = (p, "")
        y += 60
    for name, (r, g, b) in _VECTORS.items():
        p = _mel.create_material_expression(mat, unreal.MaterialExpressionVectorParameter, -900, y)
        p.set_editor_property("parameter_name", name)
        p.set_editor_property("default_value", unreal.LinearColor(r, g, b, 1.0))
        sources[name] = (p, "")
        y += 120
    names = list(sources)
    f3, f1 = unreal.CustomMaterialOutputType.CMOT_FLOAT3, unreal.CustomMaterialOutputType.CMOT_FLOAT1
    outs = [(_custom(mat, f3, "return col;", names, -300, 0), unreal.MaterialProperty.MP_BASE_COLOR),
            (_custom(mat, f1, "return rough;", names, -300, 300), unreal.MaterialProperty.MP_ROUGHNESS),
            (_custom(mat, f3, "return emis;", names, -300, 600), unreal.MaterialProperty.MP_EMISSIVE_COLOR)]
    for node, prop in outs:
        for name in names:
            src, pin = sources[name]
            _mel.connect_material_expressions(src, pin, node, name)
        _mel.connect_material_property(node, "", prop)
    _mel.recompile_material(mat)
    _eal.save_asset(mat.get_path_name())
    return mat


def instance(name, pattern, col_a, col_b, col_c=(0.3, 0.3, 0.3), **scalars):
    """Create-or-update MI_<name>; parameter edits never recompile shaders."""
    master = ensure_master()
    path = f"{MAT_DIR}/MI_{name}"
    mic = _eal.load_asset(path) if _eal.does_asset_exist(path) else None
    if mic is None:
        mic = _tools().create_asset(f"MI_{name}", MAT_DIR, unreal.MaterialInstanceConstant,
                                    unreal.MaterialInstanceConstantFactoryNew())
    if mic.get_editor_property("parent") != master:
        _mel.set_material_instance_parent(mic, master)
    values = dict(scalars, Pattern=float(pattern))
    values.setdefault("WinEmissive", _SCALARS["WinEmissive"])
    for key, val in values.items():
        _mel.set_material_instance_scalar_parameter_value(mic, key, float(val))
    for key, (r, g, b) in (("ColA", col_a), ("ColB", col_b), ("ColC", col_c)):
        _mel.set_material_instance_vector_parameter_value(mic, key, unreal.LinearColor(r, g, b, 1.0))
    _mel.update_material_instance(mic)
    _eal.save_asset(mic.get_path_name(), only_if_is_dirty=True)
    return mic


def set_wet(wet: float):
    """Wet ground: darker, glossier horizontal surfaces on every instance."""
    for p in _eal.list_assets(MAT_DIR, False):
        if "/MI_" in str(p):
            mic = _eal.load_asset(str(p))
            _mel.set_material_instance_scalar_parameter_value(mic, "Wet", float(wet))
            _mel.update_material_instance(mic)


def set_night(mics, night: bool):
    for mic in mics:
        _mel.set_material_instance_scalar_parameter_value(mic, "Night", 1.0 if night else 0.0)
        _mel.update_material_instance(mic)


def library(seed: int = 0):
    """The named surface set used by the street builder (linear-space albedo)."""
    L = {}
    L["asphalt"] = instance("Asphalt", ASPHALT, (0.045, 0.045, 0.045), (0.085, 0.083, 0.08),
                            Rough=0.82, NoiseScale=0.35)
    L["slabs"] = instance("Slabs", SLABS, (0.22, 0.21, 0.19), (0.30, 0.29, 0.27),
                          (0.10, 0.10, 0.09), Tile=0.6, Rough=0.8, NoiseScale=0.25)
    L["setts"] = instance("Setts", SLABS, (0.16, 0.15, 0.14), (0.24, 0.22, 0.2),
                          (0.07, 0.07, 0.065), Tile=0.2, Rough=0.85, NoiseScale=0.4)
    L["granite"] = instance("Granite", SLABS, (0.07, 0.07, 0.075), (0.12, 0.12, 0.125),
                            (0.04, 0.04, 0.04), Tile=0.4, Rough=0.6, NoiseScale=1.5)
    L["light_slabs"] = instance("LightSlabs", SLABS, (0.42, 0.40, 0.36), (0.50, 0.48, 0.44),
                                (0.2, 0.2, 0.19), Tile=0.9, Rough=0.75, NoiseScale=0.2)
    L["kerb"] = instance("Kerb", CONCRETE, (0.30, 0.30, 0.29), (0.38, 0.38, 0.36),
                         Tile=0.9, Rough=0.85, NoiseScale=1.5)
    L["concrete"] = instance("Concrete", CONCRETE, (0.25, 0.25, 0.24), (0.36, 0.35, 0.33),
                             Tile=3.0, Rough=0.9, NoiseScale=0.6)
    L["grass"] = instance("Grass", GRASS, (0.03, 0.07, 0.02), (0.06, 0.11, 0.03),
                          Rough=0.95, NoiseScale=0.5)
    L["hedge"] = instance("Hedge", GRASS, (0.015, 0.045, 0.012), (0.04, 0.08, 0.02),
                          Rough=0.95, NoiseScale=3.0)
    L["bark"] = instance("Bark", CONCRETE, (0.05, 0.035, 0.025), (0.09, 0.07, 0.05),
                         Tile=0.3, Rough=0.95, NoiseScale=4.0)
    L["paint_white"] = instance("PaintWhite", PAINT, (0.7, 0.7, 0.68), (0.62, 0.62, 0.6),
                                Rough=0.6, NoiseScale=2.0)
    L["paint_yellow"] = instance("PaintYellow", PAINT, (0.7, 0.5, 0.05), (0.6, 0.42, 0.04),
                                 Rough=0.6, NoiseScale=2.0)
    L["metal_dark"] = instance("MetalDark", PAINT, (0.03, 0.035, 0.035), (0.05, 0.05, 0.05),
                               Rough=0.5, NoiseScale=2.0)
    L["metal_green"] = instance("MetalGreen", PAINT, (0.02, 0.06, 0.035), (0.03, 0.07, 0.04),
                                Rough=0.5, NoiseScale=2.0)
    L["wood"] = instance("Wood", CONCRETE, (0.16, 0.09, 0.045), (0.22, 0.13, 0.07),
                         Tile=0.12, Rough=0.8, NoiseScale=6.0)
    L["lamp_lens"] = instance("LampLens", PAINT, (0.9, 0.88, 0.8), (0.9, 0.88, 0.8),
                              Rough=0.3)
    L["tyre"] = instance("Tyre", PAINT, (0.02, 0.02, 0.02), (0.03, 0.03, 0.03), Rough=0.85)
    L["glass"] = instance("CarGlass", PAINT, (0.01, 0.012, 0.015), (0.02, 0.02, 0.025), Rough=0.05)
    # Facades: brick and render, each with a window grid.
    facades = [
        ("FacadeBrickRed", BRICK, (0.20, 0.07, 0.04), (0.30, 0.11, 0.06), (0.35, 0.33, 0.30)),
        ("FacadeBrickBrown", BRICK, (0.14, 0.08, 0.05), (0.22, 0.13, 0.08), (0.30, 0.29, 0.27)),
        ("FacadeBrickYellow", BRICK, (0.38, 0.30, 0.18), (0.46, 0.37, 0.23), (0.40, 0.38, 0.35)),
        ("FacadeRenderCream", CONCRETE, (0.55, 0.50, 0.40), (0.62, 0.57, 0.47), (0.4, 0.4, 0.4)),
        ("FacadeRenderGrey", CONCRETE, (0.32, 0.33, 0.34), (0.40, 0.41, 0.42), (0.4, 0.4, 0.4)),
        ("FacadeConcrete", CONCRETE, (0.22, 0.22, 0.21), (0.30, 0.30, 0.29), (0.4, 0.4, 0.4)),
    ]
    L["facades"] = []
    for i, (name, pat, a, b, c) in enumerate(facades):
        L["facades"].append(instance(
            name, pat, a, b, c, Tile=3.2 if pat == CONCRETE else 0.6, Rough=0.88,
            NoiseScale=0.5, WinSpacing=3.0 + 0.4 * (i % 3), FloorH=3.2, Seed=float(i + seed)))
    L["facade_glass"] = [instance("FacadeGlass", CONCRETE, (0.05, 0.06, 0.07), (0.07, 0.08, 0.09),
                                  (0.2, 0.2, 0.2), Tile=1.5, Rough=0.2, NoiseScale=0.5,
                                  WinSpacing=1.5, FloorH=3.6, Seed=11.0)]
    L["facade_warehouse"] = [instance("FacadeWarehouse", CONCRETE, (0.30, 0.32, 0.33),
                                      (0.36, 0.38, 0.39), Tile=1.0, Rough=0.55, NoiseScale=0.8),
                             instance("FacadeWarehouseBlue", CONCRETE, (0.06, 0.12, 0.22),
                                      (0.08, 0.15, 0.26), Tile=1.0, Rough=0.55, NoiseScale=0.8)]
    L["gravel"] = instance("Gravel", GRASS, (0.16, 0.15, 0.13), (0.24, 0.22, 0.2),
                           Rough=0.95, NoiseScale=6.0)
    L["walls"] = [instance("WallBrick", BRICK, (0.16, 0.07, 0.045), (0.26, 0.1, 0.06),
                           (0.32, 0.31, 0.29), NoiseScale=0.6),
                  L["concrete"]]
    L["car_paints"] = [instance(f"CarPaint{i}", PAINT, c, tuple(v * 0.9 for v in c),
                                Rough=0.28, NoiseScale=3.0)
                       for i, c in enumerate([(0.6, 0.6, 0.62), (0.02, 0.02, 0.025),
                                              (0.25, 0.26, 0.28), (0.35, 0.02, 0.02),
                                              (0.02, 0.06, 0.2), (0.75, 0.75, 0.73),
                                              (0.08, 0.1, 0.12)])]
    return _with_pbr(L)


def _with_pbr(L):
    """Swap in real ambientCG sets where PBR provides one (procedural otherwise)."""
    used = {}
    for role, spec in PBR.items():
        mic = pbr_instance(f"{spec['set']}_{role}", spec["maps"], spec["tile_m"],
                           tuple(spec.get("tint", (1, 1, 1))), spec.get("aspect", 1.0))
        if mic is None:
            continue
        if role == "wall_brick":
            L["walls"] = [mic] + L["walls"][1:]
        else:
            L[role] = mic
        used[role] = spec["set"]
    L["_pbr_used"] = used
    return L


# --- texture-based PBR (ambientCG) -------------------------------------------

def _no_stream(tex):
    """[X] Streaming sized these by the ground cube's 0-1 UVs over 57 m and kept
    only low mips: world-aligned sampling then showed blurred and sharp
    triangles as a checkerboard. 1K sets are small; keep every mip resident."""
    if tex is not None and not tex.get_editor_property("never_stream"):
        tex.set_editor_property("never_stream", True)
        _eal.save_asset(tex.get_path_name())


def _import_texture(src, dest, name, srgb=True, normal=False):
    path = f"{dest}/{name}"
    if _eal.does_asset_exist(path):
        tex = _eal.load_asset(path)
        if not srgb and not normal and tex.get_editor_property("compression_settings") == \
                unreal.TextureCompressionSettings.TC_MASKS:
            tex.set_editor_property("compression_settings",
                                    unreal.TextureCompressionSettings.TC_DEFAULT)
            _eal.save_asset(tex.get_path_name())
        _no_stream(tex)
        return tex
    task = unreal.AssetImportTask()
    task.filename = src
    task.destination_path = dest
    task.destination_name = name
    task.automated = True
    task.replace_existing = True
    task.save = True
    _tools().import_asset_tasks([task])
    tex = _eal.load_asset(path)
    if tex is not None:
        if normal:
            tex.set_editor_property("compression_settings",
                                    unreal.TextureCompressionSettings.TC_NORMALMAP)
            tex.set_editor_property("srgb", False)
        elif not srgb:
            # TC_DEFAULT + linear: matches the master's LINEAR_COLOR sampler.
            tex.set_editor_property("compression_settings",
                                    unreal.TextureCompressionSettings.TC_DEFAULT)
            tex.set_editor_property("srgb", False)
        _eal.save_asset(tex.get_path_name())
        _no_stream(tex)
    return tex


def ensure_pbr_master():
    """World-aligned albedo/normal/roughness sampler (tiling in metres)."""
    path = f"{MAT_DIR}/{PBR_MASTER}"
    if _eal.does_asset_exist(path):
        return _eal.load_asset(path)
    mat = _tools().create_asset(PBR_MASTER, MAT_DIR, unreal.Material, unreal.MaterialFactoryNew())
    wp = _mel.create_material_expression(mat, unreal.MaterialExpressionWorldPosition, -1200, 0)
    nrm = _mel.create_material_expression(mat, unreal.MaterialExpressionVertexNormalWS, -1200, 100)
    tiling = _mel.create_material_expression(mat, unreal.MaterialExpressionScalarParameter, -1200, 200)
    tiling.set_editor_property("parameter_name", "TileM")
    tiling.set_editor_property("default_value", 2.0)
    uvn = _mel.create_material_expression(mat, unreal.MaterialExpressionCustom, -900, 0)
    aspect = _mel.create_material_expression(mat, unreal.MaterialExpressionScalarParameter, -1200, 300)
    aspect.set_editor_property("parameter_name", "Aspect")
    aspect.set_editor_property("default_value", 1.0)
    # World-aligned UVs in metres; Aspect = texture width/height (a 2:1 brick
    # sheet covers TileM x TileM/2).
    uvn.set_editor_property("code", "float3 w = P / 100.0 / TileM; float3 a = abs(N); float2 uv;\n"
                            "if (a.z > 0.6) uv = w.xy; else if (a.x > a.y) uv = float2(w.y, -w.z);\n"
                            "else uv = float2(w.x, -w.z);\nreturn float2(uv.x, uv.y * Aspect);")
    uvn.set_editor_property("output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT2)
    cins = []
    for nm in ("P", "N", "TileM", "Aspect"):
        ci = unreal.CustomInput()
        ci.set_editor_property("input_name", nm)
        cins.append(ci)
    uvn.set_editor_property("inputs", cins)
    _mel.connect_material_expressions(wp, "", uvn, "P")
    _mel.connect_material_expressions(nrm, "", uvn, "N")
    _mel.connect_material_expressions(tiling, "", uvn, "TileM")
    _mel.connect_material_expressions(aspect, "", uvn, "Aspect")
    tint = _mel.create_material_expression(mat, unreal.MaterialExpressionVectorParameter, -600, -200)
    tint.set_editor_property("parameter_name", "Tint")
    tint.set_editor_property("default_value", unreal.LinearColor(1, 1, 1, 1))
    specs = [("Albedo", unreal.MaterialSamplerType.SAMPLERTYPE_COLOR, 0),
             ("Normal", unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL, 300),
             ("Roughness", unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR, 600)]
    samplers = {}
    for name, stype, y in specs:
        s = _mel.create_material_expression(mat, unreal.MaterialExpressionTextureSampleParameter2D, -600, y)
        s.set_editor_property("parameter_name", name)
        s.set_editor_property("sampler_type", stype)
        # [X] A TextureSampleParameter2D without a default texture fails to
        # compile; the master had 0 instructions and every "textured" surface
        # rendered in the engine's grey default grid - the "checkerboard".
        # [X] v8 root cause (measured by a per-sampler probe, 2026-09-26): the
        # Roughness sampler was MASKS with the sRGB DefaultTexture as default;
        # a sampler-type/texture mismatch compiles to 0 instructions. Albedo
        # alone: 379, normal alone: 383, masks+sRGB: 0. Default textures now
        # match each sampler type (sRGB colour / normal map / linear).
        s.set_editor_property("texture", unreal.load_asset(
            {"Normal": "/Engine/EngineMaterials/DefaultNormal",
             "Roughness": "/Engine/EngineMaterials/T_Default_Material_Grid_M"}.get(
                name, "/Engine/EngineResources/DefaultTexture")))
        _mel.connect_material_expressions(uvn, "", s, "UVs")
        samplers[name] = s
    mul = _mel.create_material_expression(mat, unreal.MaterialExpressionMultiply, -300, -100)
    _mel.connect_material_expressions(samplers["Albedo"], "RGB", mul, "A")
    _mel.connect_material_expressions(tint, "", mul, "B")
    # Macro variation (~20 m and ~7 m, world space): breaks the visible repeat
    # of a 1-2 m texture over a 40 m yard, as weathering/staining does.
    macro = _mel.create_material_expression(mat, unreal.MaterialExpressionCustom, -600, -350)
    macro.set_editor_property("code", "float2 q = P.xy / 100.0;\n"
                              "float a = sin(q.x / 20.0 * 6.283 + 1.3 * sin(q.y / 17.0 * 6.283));\n"
                              "float b = sin(q.y / 7.0 * 6.283 + 1.7 * sin(q.x / 9.0 * 6.283));\n"
                              "return lerp(0.78, 1.16, saturate(0.5 + 0.3 * a + 0.2 * b));")
    macro.set_editor_property("output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT1)
    ci = unreal.CustomInput()
    ci.set_editor_property("input_name", "P")
    macro.set_editor_property("inputs", [ci])
    _mel.connect_material_expressions(wp, "", macro, "P")
    mul2 = _mel.create_material_expression(mat, unreal.MaterialExpressionMultiply, -150, -100)
    _mel.connect_material_expressions(mul, "", mul2, "A")
    _mel.connect_material_expressions(macro, "", mul2, "B")
    _mel.connect_material_property(mul2, "", unreal.MaterialProperty.MP_BASE_COLOR)
    _mel.connect_material_property(samplers["Normal"], "RGB", unreal.MaterialProperty.MP_NORMAL)
    _mel.connect_material_property(samplers["Roughness"], "R", unreal.MaterialProperty.MP_ROUGHNESS)
    _mel.recompile_material(mat)
    _eal.save_asset(mat.get_path_name())
    if int(_mel.get_statistics(mat).get_editor_property("num_pixel_shader_instructions")) <= 0:
        _eal.delete_asset(mat.get_path_name())
        raise RuntimeError(f"{PBR_MASTER}: failed to compile (0 pixel instructions)")
    return mat


def pbr_instance(name, maps: dict, tile_m: float = 2.0, tint=(1, 1, 1), aspect: float = 1.0):
    """maps: {"albedo": png, "normal": png, "roughness": png} (local files)."""
    master = ensure_pbr_master()
    dest = f"/Game/SurvSim/Tex/{name.split('_')[0]}"    # one import per set, shared by roles
    tex = {"Albedo": _import_texture(maps["albedo"], dest, "T_albedo"),
           "Normal": _import_texture(maps["normal"], dest, "T_normal", normal=True),
           "Roughness": _import_texture(maps["roughness"], dest, "T_rough", srgb=False)}
    path = f"{MAT_DIR}/MIT_{name}"
    mic = _eal.load_asset(path) if _eal.does_asset_exist(path) else None
    if mic is None:
        mic = _tools().create_asset(f"MIT_{name}", MAT_DIR, unreal.MaterialInstanceConstant,
                                    unreal.MaterialInstanceConstantFactoryNew())
    if mic.get_editor_property("parent") != master:
        # [X] a cached instance kept its old master: a master version bump
        # (macro variation, aspect) silently never reached the render.
        _mel.set_material_instance_parent(mic, master)
    for key, t in tex.items():
        if t is not None:
            _mel.set_material_instance_texture_parameter_value(mic, key, t)
    _mel.set_material_instance_scalar_parameter_value(mic, "TileM", float(tile_m))
    _mel.set_material_instance_scalar_parameter_value(mic, "Aspect", float(aspect))
    _mel.set_material_instance_vector_parameter_value(mic, "Tint", unreal.LinearColor(*tint, 1.0))
    _mel.update_material_instance(mic)
    _eal.save_asset(mic.get_path_name(), only_if_is_dirty=True)
    return mic

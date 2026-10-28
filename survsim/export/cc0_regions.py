"""Clothing regions for the CC0 (Quaternius) characters, baked as vertex colour.

The MetaHuman painted-clothing material (scripts/ue/survsim_clothing.py) picks
regions from the pre-skinned position of an A-pose body. The rebaked Quaternius
bodies are arms-down, several are chibi-proportioned, and 8 of 22 have no UVs,
so neither a height rule nor a UV atlas can separate sleeve from torso. The
skin weights can: every vertex is labelled by its dominant joint (plus the
source material name for hair and eyes), and the label is written as COLOR_0.r
into a copy of the glb. The in-editor material reads VertexColor (pixel-shader
safe) and paints the outfit per region; region codes are stable:

    0 skin (head, neck, hands)     1 top (torso, shoulders, upper arms)
    2 forearm (long sleeve or skin) 3 thigh / seat (bottom)
    4 shin (trousers or skin)       5 shoes      6 hair      7 eyes

Pure numpy; the source glb is never modified.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .gltf_normalize import read_accessor, read_glb, world_matrices, write_glb

N_CODES = 8
SKIN, TOP, FOREARM, THIGH, SHIN, SHOES, HAIR, EYES = range(8)
NAMES = ["skin", "top", "forearm", "thigh", "shin", "shoes", "hair", "eyes"]

_J = [  # first match wins
    (re.compile(r"(?i)foot|toe"), SHOES),
    (re.compile(r"(?i)lowerleg|^(left|right)leg$|shin|calf"), SHIN),
    (re.compile(r"(?i)upperleg|upleg|thigh"), THIGH),
    (re.compile(r"(?i)wrist|palm|hand|finger|fist|thumb|index|middle|ring|pinky"), SKIN),
    (re.compile(r"(?i)lowerarm|forearm"), FOREARM),
    (re.compile(r"(?i)upperarm|^(left|right)arm$|shoulder|clavicle"), TOP),
    (re.compile(r"(?i)neck|head"), SKIN),
    (re.compile(r"(?i)chest|torso|abdomen|spine"), TOP),
]
_M_HAIR = re.compile(r"(?i)hair|moustache|beard|eyebrow")
_M_EYE = re.compile(r"(?i)^eyes?$|pupil")


def _hips_y(gltf) -> float | None:
    W = world_matrices(gltf)
    for j in gltf["skins"][0]["joints"]:
        if re.search(r"(?i)^hips$", gltf["nodes"][j].get("name", "")):
            return float(W[j][1, 3])
    return None


def region_codes(gltf, binary) -> list[np.ndarray]:
    """One int array of region codes per skinned primitive (file order)."""
    hips = _hips_y(gltf)
    mats = [m.get("name", "") for m in gltf.get("materials", [])]
    out = []
    for n in gltf["nodes"]:
        if "mesh" not in n:
            continue
        sj = gltf["skins"][n["skin"]]["joints"] if "skin" in n else []
        for p in gltf["meshes"][n["mesh"]]["primitives"]:
            P = read_accessor(gltf, binary, p["attributes"]["POSITION"])
            codes = np.full(len(P), TOP, int)
            if sj and "JOINTS_0" in p["attributes"]:
                jj = read_accessor(gltf, binary, p["attributes"]["JOINTS_0"]).astype(int)
                ww = read_accessor(gltf, binary, p["attributes"]["WEIGHTS_0"])
                best = jj[np.arange(len(jj)), ww.argmax(1)]
                names = [gltf["nodes"][sj[q]].get("name", "") for q in best]
                for i, nm in enumerate(names):
                    for rx, c in _J:
                        if rx.search(nm):
                            codes[i] = c
                            break
                    else:           # hips / root / body: seat below the waist
                        waist = (hips if hips is not None else 0.53 * P[:, 1].max()) + 0.03
                        codes[i] = THIGH if P[i, 1] < waist else TOP
            mname = mats[p["material"]] if "material" in p and p["material"] < len(mats) else ""
            if _M_HAIR.search(mname):
                codes[:] = HAIR
            elif _M_EYE.search(mname):
                codes[:] = EYES
            out.append(codes)
    return out


def paint_file(src, dst) -> dict:
    """Copy `src` to `dst` with COLOR_0 = region code (R = code / 7, normalised u8).

    Returns the fraction of vertices per region (a sanity report: a body with
    no shoes or no top is a mapping failure, see `check_regions`).
    """
    gltf, binary = read_glb(src)
    codes = region_codes(gltf, binary)
    binary = bytearray(binary)
    k = 0
    for n in gltf["nodes"]:
        if "mesh" not in n:
            continue
        for p in gltf["meshes"][n["mesh"]]["primitives"]:
            c = codes[k]
            k += 1
            rgba = np.zeros((len(c), 4), np.uint8)
            rgba[:, 0] = np.round(c * 255.0 / (N_CODES - 1)).astype(np.uint8)
            rgba[:, 3] = 255
            binary += b"\0" * (-len(binary) % 4)
            gltf["bufferViews"].append({"buffer": 0, "byteOffset": len(binary),
                                        "byteLength": rgba.nbytes, "target": 34962})
            binary += rgba.tobytes()
            gltf["accessors"].append({"bufferView": len(gltf["bufferViews"]) - 1,
                                      "componentType": 5121, "normalized": True,
                                      "count": len(c), "type": "VEC4"})
            p["attributes"]["COLOR_0"] = len(gltf["accessors"]) - 1
    gltf["buffers"][0]["byteLength"] = len(binary) + (-len(binary) % 4)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    write_glb(dst, gltf, binary)
    allc = np.concatenate(codes) if codes else np.zeros(0, int)
    return {NAMES[i]: round(float((allc == i).mean()), 4) for i in range(N_CODES)}


def check_regions(frac: dict) -> list[str]:
    """Failures for a region report: every clothing region must exist."""
    return [f"no {r} vertices" for r in ("top", "thigh", "shin", "shoes", "skin")
            if frac.get(r, 0.0) < 0.002]


def decode(r: float) -> int:
    """Region code from a vertex colour red channel in [0, 1] (as the shader does)."""
    return round(r * (N_CODES - 1))

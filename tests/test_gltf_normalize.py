"""Re-baked Quaternius characters: exact skinning, upright walk, no jumps.

Uses two originals from ~/data/assets read-only (< 1 MB each); skipped when the
asset library is absent (CI). See docs/12, "Characters" for the root cause.
"""
from pathlib import Path

import numpy as np
import pytest

from survsim.export.gltf_normalize import (
    anim_times,
    check_character,
    read_glb,
    rebake,
    sample_animation,
    skinned_vertices,
    write_glb,
)

LIB = Path.home() / "data/assets/characters/quaternius"
CHARS = ["man_HMnuH5geEG.glb", "characters_sam_UcLErL2W37.glb"]


def _src(name):
    p = LIB / name
    if not p.is_file():
        pytest.skip(f"asset library not present: {p}")
    return p


def _walk(g):
    return min((a for a in g["animations"] if "walk" in a["name"].lower()), key=lambda a: len(a["name"]))


def test_source_file_shows_the_root_cause():
    """The raw file violates the invariants the importer relies on."""
    g, b = read_glb(_src(CHARS[0]))
    mesh_node = next(n for n in g["nodes"] if "mesh" in n)
    assert mesh_node["scale"] == [100.0, 100.0, 100.0]      # ignored by spec, not by UE
    from survsim.export.gltf_normalize import _ibms, world_matrices
    W = world_matrices(g)
    sk = g["skins"][0]
    ibm = _ibms(g, b, sk)
    G = W[sk["joints"][0]] @ ibm[0]
    consistent = sum(np.allclose(W[j] @ ibm[k], G, atol=1e-2) for k, j in enumerate(sk["joints"]))
    assert consistent < len(sk["joints"]) // 2               # rest pose != bind pose


@pytest.mark.parametrize("name", CHARS)
def test_rebaked_character_passes_checks(name, tmp_path):
    g, b = read_glb(_src(name))
    out, ob, _rep = rebake(g, b)
    dst = tmp_path / name
    write_glb(dst, out, ob)
    metrics, fails = check_character(dst)
    assert fails == [], (fails, metrics)
    assert metrics["rest_err"] < 1e-3                         # (a)
    assert 1.5 < metrics["walk_extent"][1] < 1.95             # (b)
    assert metrics["max_trunk_move"] < 0.3                    # (c)
    assert metrics["max_bone_stretch"] < 0.05


def test_bind_mode_is_exact_in_every_pose():
    g, b = read_glb(_src(CHARS[0]))
    out, ob, rep = rebake(g, b, rest="bind")
    k = rep["scale_to_m"]
    wa, wo = _walk(g), _walk(out)
    t0 = anim_times(g, b, wa)[0]
    for t in (0.0, 0.37):
        src = skinned_vertices(g, b, sample_animation(g, b, wa, t0 + t), include_rigid=False) * k
        new = skinned_vertices(out, ob, sample_animation(out, ob, wo, t))
        d = new - src
        d -= d[0]                                            # ground offset only
        assert np.abs(d).max() < 1e-4


def test_source_rest_mode_close_in_walk():
    g, b = read_glb(_src(CHARS[0]))
    out, ob, rep = rebake(g, b)                              # rest="source"
    k = rep["scale_to_m"]
    wa, wo = _walk(g), _walk(out)
    t0 = anim_times(g, b, wa)[0]
    src = skinned_vertices(g, b, sample_animation(g, b, wa, t0 + 0.37), include_rigid=False) * k
    new = skinned_vertices(out, ob, sample_animation(out, ob, wo, 0.37))
    d = new - src
    d -= np.median(d, 0)
    assert np.percentile(np.linalg.norm(d, axis=1), 99) < 0.02


def test_weapons_dropped():
    g, b = read_glb(_src(CHARS[1]))
    _, _, rep = rebake(g, b)
    assert len(rep["dropped"]) == 1          # the hand-held prop (an axe on Sam)

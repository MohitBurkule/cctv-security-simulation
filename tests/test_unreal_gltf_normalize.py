"""The armature-bake keeps every joint's world transform (rest and animated)."""

import numpy as np

from survsim.export.gltf_normalize import compose, normalize, qrot


def _world(nodes, idx, parent):
    chain = []
    while idx is not None:
        chain.append(idx)
        idx = parent.get(idx)
    T = (np.zeros(3), np.array([0, 0, 0, 1.0]), np.ones(3))
    for i in reversed(chain):
        n = nodes[i]
        T = compose(T, (np.array(n.get("translation", [0, 0, 0]), float),
                        np.array(n.get("rotation", [0, 0, 0, 1]), float),
                        np.array(n.get("scale", [1, 1, 1]), float)))
    return T


def _doc():
    h = 0.5 ** 0.5
    nodes = [{"name": "Root", "children": [1]},
             {"name": "Armature", "rotation": [-h, 0, 0, h], "scale": [100, 100, 100],
              "children": [2]},
             {"name": "Hips", "translation": [0, 0.01, 0.002], "rotation": [h, 0, 0, h],
              "children": [3]},
             {"name": "Head", "translation": [0, 0.004, 0]}]
    keys = np.array([[0, 0.01, 0.0], [0.001, 0.012, 0.003]], "<f4")
    binary = bytearray(keys.tobytes())
    gltf = {"nodes": nodes, "skins": [{"joints": [2, 3]}],
            "buffers": [{"byteLength": len(binary)}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": 2, "type": "VEC3"}],
            "animations": [{"channels": [{"sampler": 0, "target": {"node": 2, "path": "translation"}}],
                            "samplers": [{"input": 0, "output": 0}]}]}
    return gltf, binary


def test_rest_pose_world_positions_and_rotations_unchanged_scale_removed():
    gltf, binary = _doc()
    parent = {c: i for i, n in enumerate(gltf["nodes"]) for c in n.get("children", [])}
    before = [_world(gltf["nodes"], j, parent) for j in (2, 3)]
    rep = normalize(gltf, binary)
    after = [_world(gltf["nodes"], j, parent) for j in (2, 3)]
    for b, a in zip(before, after):
        assert np.allclose(b[0], a[0], atol=1e-5)          # position
        assert np.allclose(np.abs(b[1]), np.abs(a[1]), atol=1e-6)
        assert np.allclose(a[2], 1.0)                      # no residual scale
    assert "rotation" not in gltf["nodes"][1] and "scale" not in gltf["nodes"][1]
    assert rep["skins"][0]["scale"] == 100.0


def test_animated_translation_is_premultiplied():
    gltf, binary = _doc()
    h = 0.5 ** 0.5
    orig = np.frombuffer(bytes(binary), "<f4").reshape(2, 3).astype(float)
    normalize(gltf, binary)
    new = np.frombuffer(bytes(binary), "<f4").reshape(2, 3).astype(float)
    expect = 100.0 * qrot(np.array([-h, 0, 0, h]), orig)
    assert np.allclose(new, expect, atol=1e-4)

"""Bake armature-node transforms into the skeleton root of a rigged glTF.

[X] Quaternius glTFs (and many Blender exports) put rotation(-90 X) and scale
100 on the ARMATURE node that parents the skin's root joint. UE 5.8 Interchange
treats that node inconsistently between the skinned mesh and the animation, so
walkers render face-down with stretched limbs (docs/12, "Characters").

This rewrites the file so every ancestor of the top joint(s) is identity, and
the top joints' rest TRS and animation channels are pre-multiplied by the
removed transform. World-space joint transforms — and therefore the inverse bind
matrices, the skinned mesh and every animated pose — are unchanged, which the
test checks numerically. Only uniform scale is supported (asserted).

Pure stdlib + numpy; works on .glb.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_COMP = {5126: ("f", 4)}
_NCOMP = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def qmul(a, b):
    """Hamilton product, glTF order (x, y, z, w). Broadcasts over rows of b."""
    ax, ay, az, aw = a
    b = np.atleast_2d(b)
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], axis=1)


def qrot(q, v):
    """Rotate row vectors v by quaternion q (x, y, z, w)."""
    v = np.atleast_2d(v)
    x, y, z, w = q
    u = np.array([x, y, z])
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _trs(node):
    if "matrix" in node:
        raise ValueError("matrix-form nodes not supported")
    return (np.array(node.get("translation", [0, 0, 0]), float),
            np.array(node.get("rotation", [0, 0, 0, 1]), float),
            np.array(node.get("scale", [1, 1, 1]), float))


def compose(parent, child):
    """(t, r, s) of parent∘child for uniform parent scale."""
    tp, rp, sp = parent
    tc, rc, sc = child
    if not np.allclose(sp, sp[0]):
        raise ValueError(f"non-uniform armature scale {sp}")
    return (tp + sp[0] * qrot(rp, tc)[0], qmul(rp, rc)[0], sp[0] * sc)


def read_glb(path):
    data = Path(path).read_bytes()
    magic, _, _ = struct.unpack("<4sII", data[:12])
    assert magic == b"glTF"
    jlen = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + jlen])
    off = 20 + jlen
    blen = struct.unpack("<I", data[off:off + 4])[0]
    binary = bytearray(data[off + 8:off + 8 + blen])
    return gltf, binary


def write_glb(path, gltf, binary):
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    binary = bytes(binary) + b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(js) + 8 + len(binary)
    out = struct.pack("<4sII", b"glTF", 2, total)
    out += struct.pack("<I4s", len(js), b"JSON") + js
    out += struct.pack("<I4s", len(binary), b"BIN\0") + binary
    Path(path).write_bytes(out)


def _accessor_view(gltf, binary, idx):
    acc = gltf["accessors"][idx]
    if acc.get("componentType") != 5126 or acc.get("sparse"):
        raise ValueError("only dense float accessors supported")
    view = gltf["bufferViews"][acc["bufferView"]]
    n = _NCOMP[acc["type"]]
    stride = view.get("byteStride", 4 * n)
    if stride != 4 * n:
        raise ValueError("interleaved accessor")
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    arr = np.frombuffer(binary, dtype="<f4", count=acc["count"] * n, offset=start)
    return arr.reshape(acc["count"], n).astype(float), start


def _write_accessor(binary, start, arr):
    binary[start:start + arr.size * 4] = arr.astype("<f4").tobytes()


def normalize(gltf, binary):
    """Mutates gltf/binary in place; returns a report dict."""
    nodes = gltf["nodes"]
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    report = {"skins": []}
    done_tops = set()
    for skin in gltf.get("skins", []):
        joints = set(skin["joints"])
        tops = [j for j in skin["joints"] if parent.get(j) not in joints]
        for top in tops:
            if top in done_tops:
                continue
            chain = []
            p = parent.get(top)
            while p is not None:
                chain.append(p)
                p = parent.get(p)
            A = (np.zeros(3), np.array([0, 0, 0, 1.0]), np.ones(3))
            for p in reversed(chain):                      # root first
                A = compose(A, _trs(nodes[p]))
            if np.allclose(A[0], 0) and np.allclose(A[1], [0, 0, 0, 1]) and np.allclose(A[2], 1):
                continue
            for p in chain:
                for k in ("translation", "rotation", "scale", "matrix"):
                    nodes[p].pop(k, None)
            t, r, s = compose(A, _trs(nodes[top]))
            nodes[top]["translation"], nodes[top]["rotation"], nodes[top]["scale"] = (
                t.tolist(), r.tolist(), s.tolist())
            touched = set()
            for anim in gltf.get("animations", []):
                for ch in anim["channels"]:
                    if ch["target"].get("node") != top:
                        continue
                    acc = anim["samplers"][ch["sampler"]]["output"]
                    if acc in touched:
                        continue
                    touched.add(acc)
                    vals, start = _accessor_view(gltf, binary, acc)
                    path = ch["target"]["path"]
                    if path == "translation":
                        vals = A[0] + A[2][0] * qrot(A[1], vals)
                    elif path == "rotation":
                        vals = qmul(A[1], vals)
                    elif path == "scale":
                        vals = vals * A[2][0]
                    _write_accessor(binary, start, vals)
            done_tops.add(top)
            sc = float(A[2][0])
            if abs(sc - 1.0) > 1e-6:
                _remove_scale(gltf, binary, skin, top, sc, touched)
            report["skins"].append({"top": nodes[top].get("name"), "chain": [
                nodes[p].get("name") for p in chain], "scale": float(A[2][0]),
                "rotation": [round(x, 4) for x in A[1].tolist()], "anim_accessors": len(touched)})
    return report


def _remove_scale(gltf, binary, skin, top, sc, done):
    """Fold a uniform scale `sc` on the top joint into translations + IBMs.

    World joint positions are unchanged: every descendant joint's local
    translation (rest and animated) grows by `sc`, the top joint's scale
    shrinks by it, and each inverse bind matrix's upper 3 rows scale by `sc`.
    Vertex data is untouched (the mesh node transform is ignored for skins).
    """
    nodes = gltf["nodes"]
    joints = set(skin["joints"])
    for j in joints:
        if j == top:
            nodes[j]["scale"] = (np.array(nodes[j].get("scale", [1, 1, 1])) / sc).tolist()
        elif "translation" in nodes[j]:
            nodes[j]["translation"] = (np.array(nodes[j]["translation"]) * sc).tolist()
    for anim in gltf.get("animations", []):
        for ch in anim["channels"]:
            node, path = ch["target"].get("node"), ch["target"]["path"]
            if node not in joints:
                continue
            acc = anim["samplers"][ch["sampler"]]["output"]
            key = ("unscale", acc)
            if key in done:
                continue
            if node == top and path == "scale":
                vals, start = _accessor_view(gltf, binary, acc)
                _write_accessor(binary, start, vals / sc)
            elif node != top and path == "translation":
                vals, start = _accessor_view(gltf, binary, acc)
                _write_accessor(binary, start, vals * sc)
            else:
                continue
            done.add(key)
    if "inverseBindMatrices" in skin:
        m, start = _accessor_view(gltf, binary, skin["inverseBindMatrices"])
        m = m.copy()
        for col in range(4):
            for row in range(3):
                m[:, col * 4 + row] *= sc
        _write_accessor(binary, start, m)


def normalize_file(src, dst):
    gltf, binary = read_glb(src)
    rep = normalize(gltf, binary)
    write_glb(dst, gltf, binary)
    return rep


# ---------------------------------------------------------------------------
# Full "apply transforms" re-bake (2026-09-26). See docs/12, "Characters".
#
# Root cause of the face-down / stretched-arm Quaternius imports, measured on
# man_HMnuH5geEG.glb:
#   1. The skinned MESH node (a sibling of the armature, not its child) carries
#      rot(-90 X) * scale 100. glTF says a skinned mesh node's transform is
#      ignored; the vertices are in a raw space ~0.05 units tall that only the
#      inverse bind matrices map to the 4.8-unit character. An importer that
#      applies the node transform to the mesh gets a face-down, 100x mesh.
#   2. The bind pose (inverse IBMs) is a wide T-pose; the node rest pose is a
#      different pose (arms down, feet moved): only 3 of 31 joints satisfy
#      world(j) * IBM_j == meshNodeWorld. Joints without animation channels
#      (Bone, Hips, Abdomen, PoleTargets) are therefore posed differently by
#      an importer that takes its reference pose from the binds vs the nodes.
#   3. Every joint has world scale 100 and local translations of ~0.005.
# The old normalize() above fixed only (3)'s top-joint part.
#
# rebake() resolves all three by re-expressing everything in one canonical
# frame: vertices in metres, Y-up, feet at y=0; every mesh node identity at the
# scene root; joints unit-scale with rest TRS == bind pose (so W_j * IBM_j == I);
# every clip resampled to full T/R/S channels for every joint, computed from the
# evaluated source node tree so each skinned pose is the source pose * k.
# ---------------------------------------------------------------------------

_DT = {5126: "<f4", 5123: "<u2", 5121: "u1", 5125: "<u4", 5122: "<i2", 5120: "i1"}
WEAPON_RE = r"(?i)gun|pistol|rifle|knife|sword|axe|weapon|shield|bow"


def read_accessor(gltf, binary, idx, raw=False):
    """Any dense accessor -> float (or raw dtype) array of shape (count, ncomp)."""
    acc = gltf["accessors"][idx]
    n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[acc["type"]]
    dt = np.dtype(_DT[acc["componentType"]])
    view = gltf["bufferViews"][acc["bufferView"]]
    stride = view.get("byteStride", dt.itemsize * n)
    off = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    buf = np.frombuffer(bytes(binary), np.uint8)
    rows = np.lib.stride_tricks.as_strided(
        buf[off:], shape=(acc["count"], dt.itemsize * n), strides=(stride, 1))
    out = np.ascontiguousarray(rows).view(dt).reshape(acc["count"], n)
    return out.copy() if raw else out.astype(float)


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(m):
    """Rotation matrix -> unit quaternion (x, y, z, w), w >= 0 not enforced."""
    t = np.trace(m)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
        q = [0.0] * 4
        q[i] = s / 4
        q[j] = (m[j, i] + m[i, j]) / s
        q[k] = (m[k, i] + m[i, k]) / s
        q[3] = (m[k, j] - m[j, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def trs_matrix(t=(0, 0, 0), r=(0, 0, 0, 1), s=(1, 1, 1)):
    m = np.eye(4)
    m[:3, :3] = quat_to_mat(r) @ np.diag(s)
    m[:3, 3] = t
    return m


def node_local(node, override=None):
    if "matrix" in node:
        return np.array(node["matrix"], float).reshape(4, 4).T
    o = override or {}
    return trs_matrix(o.get("translation", node.get("translation", (0, 0, 0))),
                      o.get("rotation", node.get("rotation", (0, 0, 0, 1))),
                      o.get("scale", node.get("scale", (1, 1, 1))))


def world_matrices(gltf, overrides=None):
    """World matrix of every node, with optional {node: {path: value}} overrides."""
    nodes = gltf["nodes"]
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    ov = overrides or {}
    W = {}

    def w(i):
        if i not in W:
            L = node_local(nodes[i], ov.get(i))
            W[i] = w(parent[i]) @ L if i in parent else L
        return W[i]
    for i in range(len(nodes)):
        w(i)
    return W


def _slerp(a, b, u):
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + u * (b - a)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - u) * th) * a + np.sin(u * th) * b) / np.sin(th)
    return q / np.linalg.norm(q)


_SAMPLER_CACHE: dict = {}


def _sampler_data(gltf, binary, s):
    key = (id(gltf), id(binary), s["input"], s["output"])
    if key not in _SAMPLER_CACHE:
        if len(_SAMPLER_CACHE) > 20000:
            _SAMPLER_CACHE.clear()
        ti = read_accessor(gltf, binary, s["input"])[:, 0]
        vo = read_accessor(gltf, binary, s["output"])
        if s.get("interpolation", "LINEAR") == "CUBICSPLINE":
            vo = vo.reshape(len(ti), 3, -1)[:, 1]
        _SAMPLER_CACHE[key] = (ti, vo)
    return _SAMPLER_CACHE[key]


def sample_animation(gltf, binary, anim, t):
    """{node: {path: value}} at time t (LINEAR/STEP; CUBICSPLINE uses the value)."""
    out = {}
    for ch in anim["channels"]:
        node = ch["target"].get("node")
        path = ch["target"]["path"]
        if node is None or path == "weights":
            continue
        s = anim["samplers"][ch["sampler"]]
        ti, vo = _sampler_data(gltf, binary, s)
        interp = s.get("interpolation", "LINEAR")
        if t <= ti[0]:
            v = vo[0]
        elif t >= ti[-1]:
            v = vo[-1]
        else:
            k = int(np.searchsorted(ti, t, side="right")) - 1
            u = (t - ti[k]) / (ti[k + 1] - ti[k])
            if interp == "STEP":
                v = vo[k]
            elif path == "rotation":
                v = _slerp(vo[k], vo[k + 1], u)
            else:
                v = vo[k] + u * (vo[k + 1] - vo[k])
        out.setdefault(node, {})[path] = np.asarray(v, float)
    return out


def anim_times(gltf, binary, anim):
    ts = set()
    for s in anim["samplers"]:
        ts.update(np.round(read_accessor(gltf, binary, s["input"])[:, 0], 6).tolist())
    return np.array(sorted(ts))


def skinned_vertices(gltf, binary, overrides=None, include_rigid=True):
    """World positions of every mesh vertex in the pose given by overrides.

    Skinned: sum_k w_k * W(joint_k) * IBM_k * v (mesh node transform ignored,
    per the glTF spec). Rigid: W(node) * v.
    """
    W = world_matrices(gltf, overrides)
    pts = []
    for ni, n in enumerate(gltf["nodes"]):
        if "mesh" not in n or ("skin" not in n and not include_rigid):
            continue
        if "skin" in n:
            sk = gltf["skins"][n["skin"]]
            ibm = _ibms(gltf, binary, sk)
            J = np.array([W[j] @ ibm[k] for k, j in enumerate(sk["joints"])])
        for p in gltf["meshes"][n["mesh"]]["primitives"]:
            P = read_accessor(gltf, binary, p["attributes"]["POSITION"])
            ph = np.c_[P, np.ones(len(P))]
            if "skin" in n:
                jj = read_accessor(gltf, binary, p["attributes"]["JOINTS_0"]).astype(int)
                ww = read_accessor(gltf, binary, p["attributes"]["WEIGHTS_0"], raw=True)
                ww = _weights_float(ww, gltf["accessors"][p["attributes"]["WEIGHTS_0"]])
                ww = ww / np.maximum(ww.sum(1, keepdims=True), 1e-12)
                M = np.einsum("vk,vkij->vij", ww, J[jj])
                pts.append(np.einsum("vij,vj->vi", M, ph)[:, :3])
            else:
                pts.append((W[ni] @ ph.T).T[:, :3])
    return np.vstack(pts)


def _weights_float(w, acc):
    ct = acc["componentType"]
    if ct == 5121:
        return w.astype(float) / 255.0
    if ct == 5123:
        return w.astype(float) / 65535.0
    return w.astype(float)


def _ibms(gltf, binary, skin):
    if "inverseBindMatrices" not in skin:
        return np.repeat(np.eye(4)[None], len(skin["joints"]), 0)
    return read_accessor(gltf, binary, skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)


class _Builder:
    """Accumulates a fresh BIN chunk + bufferViews + accessors."""

    def __init__(self):
        self.bin = bytearray()
        self.views, self.accs = [], []

    def view(self, data: bytes, target=None):
        self.bin += b"\0" * (-len(self.bin) % 4)
        v = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target:
            v["target"] = target
        self.bin += data
        self.views.append(v)
        return len(self.views) - 1

    def accessor(self, arr, ctype=5126, typ=None, target=None, minmax=False, normalized=False):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[:, None]
        n = arr.shape[1]
        typ = typ or {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4", 16: "MAT4"}[n]
        data = np.ascontiguousarray(arr.astype(_DT[ctype])).tobytes()
        a = {"bufferView": self.view(data, target), "componentType": ctype,
             "count": int(arr.shape[0]), "type": typ}
        if normalized:
            a["normalized"] = True
        if minmax:
            a["min"] = arr.min(0).astype(float).tolist()
            a["max"] = arr.max(0).astype(float).tolist()
        self.accs.append(a)
        return len(self.accs) - 1


def rebake(gltf, binary, height_m=1.75, drop_re=WEAPON_RE, rest="source"):
    """Return (new_gltf, new_binary, report): a canonical, importer-proof glb.

    rest="source": the new bind/rest pose is the file's authored node rest pose
    (arms down for Quaternius); vertices are re-skinned into it, like Blender's
    "apply pose as rest pose". Exact in the rest pose; in other poses it differs
    from the source only where a vertex blends joints whose bind->rest deltas
    differ (measured by tests). rest="bind": keep the source bind (T-pose) as
    rest; exact in every pose, but the reference pose is a T-pose.
    """
    import re as _re
    nodes = gltf["nodes"]
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    skins = gltf["skins"]
    joints = list(skins[0]["joints"])
    jset = set(joints)
    for s in skins[1:]:
        if set(s["joints"]) != jset:
            raise ValueError("skins with different joint sets are not supported")
    W0 = world_matrices(gltf)

    def joint_parent(j):
        p = parent.get(j)
        while p is not None and p not in jset:
            p = parent.get(p)
        return p

    jpar = {j: joint_parent(j) for j in joints}
    tops = [j for j in joints if jpar[j] is None]
    ibm0 = _ibms(gltf, binary, skins[0])
    # Mesh space -> world of skin 0, from a top joint (whose rest must == bind).
    G = {}
    for si, s in enumerate(skins):
        ib = _ibms(gltf, binary, s)
        k = s["joints"].index(tops[0])
        G[si] = W0[tops[0]] @ ib[k]
    # Old bind world of each joint (skin 0).
    Bw = {j: G[0] @ np.linalg.inv(ibm0[k]) for k, j in enumerate(joints)}
    sigma = {j: np.linalg.norm(Bw[j][:3, :3], axis=0) for j in joints}
    for j in joints:
        if not np.allclose(sigma[j], sigma[j][0], rtol=1e-3):
            raise ValueError(f"non-uniform joint scale {sigma[j]} on {nodes[j].get('name')}")
    D = {j: np.diag([*(1.0 / sigma[j]), 1.0]) for j in joints}

    mesh_nodes = []
    dropped = []
    for ni, n in enumerate(nodes):
        if "mesh" not in n:
            continue
        name = n.get("name", "") + " " + gltf["meshes"][n["mesh"]].get("name", "")
        if drop_re and _re.search(drop_re, name):
            dropped.append(name.strip())
            continue
        mesh_nodes.append(ni)

    # Height / ground from the bind pose (skinned meshes only, source units).
    bind_pts = []
    for ni in mesh_nodes:
        if "skin" not in nodes[ni]:
            continue
        g = G[nodes[ni]["skin"]]
        for p in gltf["meshes"][nodes[ni]["mesh"]]["primitives"]:
            P = read_accessor(gltf, binary, p["attributes"]["POSITION"])
            bind_pts.append((g @ np.c_[P, np.ones(len(P))].T).T[:, :3])
    bind_pts = np.vstack(bind_pts)
    if rest == "source":
        bind_pts = skinned_vertices(gltf, binary, include_rigid=False)
    lo, hi = bind_pts.min(0), bind_pts.max(0)
    k = height_m / (hi[1] - lo[1])
    T = np.diag([k, k, k, 1.0])
    T[1, 3] = -k * lo[1]

    def new_world(W):
        return {j: T @ W[j] @ D[j] for j in joints}

    def locals_from(Wn):
        out = {}
        for j in joints:
            p = jpar[j]
            L = np.linalg.inv(Wn[p]) @ Wn[j] if p is not None else Wn[j]
            sc = np.linalg.norm(L[:3, :3], axis=0)
            out[j] = (L[:3, 3].copy(), mat_to_quat(L[:3, :3] / sc), sc)
        return out

    if rest == "source":
        RestW = W0
    elif rest == "bind":
        RestW = Bw
    else:
        raise ValueError(rest)
    Bn = {j: T @ RestW[j] @ D[j] for j in joints}
    bind_loc = locals_from(Bn)
    jidx = {j: i for i, j in enumerate(joints)}

    b = _Builder()
    out = {"asset": {"version": "2.0", "generator": "survsim gltf_normalize.rebake"},
           "scene": 0, "nodes": [], "meshes": [], "skins": [], "animations": []}
    for key in ("materials", "textures", "samplers"):
        if key in gltf:
            out[key] = json.loads(json.dumps(gltf[key]))
    if "images" in gltf:
        out["images"] = []
        for im in gltf["images"]:
            im = dict(im)
            if "bufferView" in im:
                v = gltf["bufferViews"][im["bufferView"]]
                o = v.get("byteOffset", 0)
                im["bufferView"] = b.view(bytes(binary[o:o + v["byteLength"]]))
            out["images"].append(im)

    # Joint nodes: new indices 0..J-1, same order as the skin.
    for j in joints:
        t, r, s = bind_loc[j]
        nd = {"name": nodes[j].get("name", f"joint{jidx[j]}"), "translation": t.tolist(),
              "rotation": r.tolist()}
        if not np.allclose(s, 1, atol=1e-5):
            nd["scale"] = s.tolist()
        kids = [jidx[c] for c in joints if jpar[c] == j]
        if kids:
            nd["children"] = kids
        out["nodes"].append(nd)
    ibm_new = np.array([np.linalg.inv(Bn[j]) for j in joints])
    ibm_acc = b.accessor(ibm_new.transpose(0, 2, 1).reshape(-1, 16), typ="MAT4")
    out["skins"].append({"name": "skin", "joints": list(range(len(joints))),
                         "skeleton": jidx[tops[0]], "inverseBindMatrices": ibm_acc})

    # Meshes: one output mesh node per source mesh node, identity transform.
    for ni in mesh_nodes:
        n = nodes[ni]
        bind_to = None
        blend = None                       # per-vertex 4x4 (rest="source", skinned)
        if "skin" in n:
            M = T @ G[n["skin"]]
            if rest == "source":
                sk = skins[n["skin"]]
                ib = _ibms(gltf, binary, sk)
                Jm = np.array([T @ W0[j] @ ib[q] for q, j in enumerate(sk["joints"])])
        else:
            a = parent.get(ni)
            while a is not None and a not in jset:
                a = parent.get(a)
            if a is None:
                M = T @ W0[ni]
            else:                       # rigid attachment -> skinned to joint a
                M = T @ RestW[a] @ np.linalg.inv(W0[a]) @ W0[ni]
                bind_to = jidx[a]
        prims = []
        for p in gltf["meshes"][n["mesh"]]["primitives"]:
            at = p["attributes"]
            na = {}
            P = read_accessor(gltf, binary, at["POSITION"])
            if "skin" in n and rest == "source":
                jj = read_accessor(gltf, binary, at["JOINTS_0"]).astype(int)
                ww = _weights_float(read_accessor(gltf, binary, at["WEIGHTS_0"], raw=True),
                                    gltf["accessors"][at["WEIGHTS_0"]])
                ww = ww / np.maximum(ww.sum(1, keepdims=True), 1e-12)
                blend = np.einsum("vk,vkij->vij", ww, Jm[jj])
            else:
                blend = np.broadcast_to(M, (len(P), 4, 4))
            R = blend[:, :3, :3]
            na["POSITION"] = b.accessor(np.einsum("vij,vj->vi", blend, np.c_[P, np.ones(len(P))])[:, :3],
                                        target=34962, minmax=True)
            if "NORMAL" in at:
                Rn = np.linalg.inv(R).transpose(0, 2, 1)
                N = np.einsum("vij,vj->vi", Rn, read_accessor(gltf, binary, at["NORMAL"]))
                N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
                na["NORMAL"] = b.accessor(N, target=34962)
            if "TANGENT" in at:
                Tg = read_accessor(gltf, binary, at["TANGENT"])
                v = np.einsum("vij,vj->vi", R, Tg[:, :3])
                v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
                na["TANGENT"] = b.accessor(np.c_[v, Tg[:, 3]], target=34962)
            for key, idx in at.items():
                if key.startswith(("TEXCOORD_", "COLOR_")) or (key == "WEIGHTS_0" and "skin" in n):
                    acc = gltf["accessors"][idx]
                    raw = read_accessor(gltf, binary, idx, raw=True)
                    na[key] = b.accessor(raw, ctype=acc["componentType"], target=34962,
                                         normalized=acc.get("normalized", False))
            if "skin" in n:            # remap skin-local joint indices -> output order
                lut = np.array([jidx[j] for j in skins[n["skin"]]["joints"]])
                raw = read_accessor(gltf, binary, at["JOINTS_0"], raw=True).astype(int)
                na["JOINTS_0"] = b.accessor(lut[raw], ctype=5123, target=34962)
            elif bind_to is not None:
                nv = len(P)
                na["JOINTS_0"] = b.accessor(np.tile([bind_to, 0, 0, 0], (nv, 1)),
                                            ctype=5123, target=34962)
                na["WEIGHTS_0"] = b.accessor(np.tile([1.0, 0, 0, 0], (nv, 1)), target=34962)
            np_ = {"attributes": na}
            if "indices" in p:
                acc = gltf["accessors"][p["indices"]]
                np_["indices"] = b.accessor(read_accessor(gltf, binary, p["indices"], raw=True)[:, 0],
                                            ctype=acc["componentType"], target=34963)
            for key in ("material", "mode"):
                if key in p:
                    np_[key] = p[key]
            prims.append(np_)
        out["meshes"].append({"name": gltf["meshes"][n["mesh"]].get("name", n.get("name")),
                              "primitives": prims})
        nd = {"name": n.get("name", "mesh"), "mesh": len(out["meshes"]) - 1}
        if "skin" in n or bind_to is not None:
            nd["skin"] = 0
        out["nodes"].append(nd)

    # Merge every skinned mesh into ONE mesh node: [V] UE Interchange imports
    # each skinned mesh node as its own SkeletalMesh, so a 5-part Quaternius
    # character spawned from the first asset rendered as a floating torso.
    first = len(joints)
    if len(out["nodes"]) - first > 1:
        prims = [p for m in out["meshes"] for p in m["primitives"]]
        out["meshes"] = [{"name": "body", "primitives": prims}]
        out["nodes"] = out["nodes"][:first] + [{"name": "body", "mesh": 0, "skin": 0}]
    roots = [jidx[t] for t in tops] + list(range(len(joints), len(out["nodes"])))
    out["scenes"] = [{"name": "Scene", "nodes": roots}]

    # Animations: resample each clip on the union of its key times, all joints.
    for anim in gltf.get("animations", []):
        ts = anim_times(gltf, binary, anim)
        if len(ts) == 0:
            continue
        tr = np.zeros((len(ts), len(joints), 3))
        ro = np.zeros((len(ts), len(joints), 4))
        sc = np.zeros((len(ts), len(joints), 3))
        for i, t in enumerate(ts):
            L = locals_from(new_world(world_matrices(gltf, sample_animation(gltf, binary, anim, t))))
            for j in joints:
                tr[i, jidx[j]], ro[i, jidx[j]], sc[i, jidx[j]] = L[j]
        for ji in range(len(joints)):          # quaternion sign continuity
            for i in range(1, len(ts)):
                if np.dot(ro[i, ji], ro[i - 1, ji]) < 0:
                    ro[i, ji] *= -1
        tin = b.accessor(ts - ts[0], minmax=True)
        chans, samps = [], []
        for ji in range(len(joints)):
            for path, arr in (("translation", tr), ("rotation", ro), ("scale", sc)):
                if path == "scale" and np.allclose(arr[:, ji], 1, atol=1e-5):
                    continue
                samps.append({"input": tin, "output": b.accessor(arr[:, ji]), "interpolation": "LINEAR"})
                chans.append({"sampler": len(samps) - 1, "target": {"node": ji, "path": path}})
        out["animations"].append({"name": anim.get("name", "clip"), "channels": chans,
                                  "samplers": samps})

    out["accessors"], out["bufferViews"] = b.accs, b.views
    out["buffers"] = [{"byteLength": len(b.bin)}]
    report = {"scale_to_m": float(k), "source_height": float(hi[1] - lo[1]),
              "joints": len(joints), "meshes": len(mesh_nodes), "dropped": dropped,
              "clips": len(out["animations"]),
              "rest_eq_bind_src": int(sum(np.allclose(W0[j], Bw[j], atol=1e-4 * sigma[j][0])
                                          for j in joints))}
    return out, b.bin, report


def rebake_file(src, dst, height_m=1.75, rest="source"):
    gltf, binary = read_glb(src)
    out, binv, rep = rebake(gltf, binary, height_m, rest=rest)
    write_glb(dst, out, binv)
    return rep


import re as _re_mod

_re_head = _re_mod.compile(r"(?i)head")
_re_foot = _re_mod.compile(r"(?i)foot|toe|ankle")
_re_trunk = _re_mod.compile(r"(?i)hip|pelvis|spine|chest|torso|abdomen|neck|head")


def _dominant_joint_names(gltf, binary):
    """Name of the highest-weight joint for every vertex (skinned meshes)."""
    names = []
    for n in gltf["nodes"]:
        if "mesh" not in n:
            continue
        for p in gltf["meshes"][n["mesh"]]["primitives"]:
            cnt = gltf["accessors"][p["attributes"]["POSITION"]]["count"]
            if "skin" not in n:
                names += [""] * cnt
                continue
            sj = gltf["skins"][n["skin"]]["joints"]
            jj = read_accessor(gltf, binary, p["attributes"]["JOINTS_0"]).astype(int)
            ww = read_accessor(gltf, binary, p["attributes"]["WEIGHTS_0"])
            best = jj[np.arange(len(jj)), ww.argmax(1)]
            names += [gltf["nodes"][sj[q]].get("name", "") for q in best]
    return names


def check_character(path, height_m=1.75, walk_word="walk"):
    """Pure-Python import checks on a (re-baked) glb. Returns a dict of metrics
    and a list of failures (empty == pass). See docs/12, "Characters"."""
    gltf, binary = read_glb(path)
    fails = []
    # (a) rest pose skinned == raw mesh (i.e. rest == bind, mesh nodes identity)
    raw = []
    for n in gltf["nodes"]:
        if "mesh" in n:
            if any(k in n for k in ("translation", "rotation", "scale", "matrix")):
                fails.append(f"mesh node {n.get('name')} has a transform")
            for p in gltf["meshes"][n["mesh"]]["primitives"]:
                raw.append(read_accessor(gltf, binary, p["attributes"]["POSITION"]))
    raw = np.vstack(raw)
    rest = skinned_vertices(gltf, binary)
    rest_err = float(np.abs(rest - raw).max())
    if rest_err > 1e-3:
        fails.append(f"rest-pose skinning moves vertices by {rest_err:.4f} m")
    W = world_matrices(gltf)
    sk = gltf["skins"][0]
    ibm = _ibms(gltf, binary, sk)
    ibm_err = float(max(np.abs(W[j] @ ibm[k] - np.eye(4)).max() for k, j in enumerate(sk["joints"])))
    if ibm_err > 1e-3:
        fails.append(f"W_j * IBM_j != I (max {ibm_err:.4f})")
    # (b) walk clip first frame: upright, right height
    walks = [a for a in gltf.get("animations", []) if walk_word in a.get("name", "").lower()]
    walks.sort(key=lambda a: len(a.get("name", "")))
    metrics = {"rest_err": rest_err, "ibm_err": ibm_err}
    if not walks:
        fails.append("no walk clip")
        return metrics, fails
    walk = walks[0]
    ov = sample_animation(gltf, binary, walk, anim_times(gltf, binary, walk)[0])
    pose = skinned_vertices(gltf, binary, ov)
    ext = pose.max(0) - pose.min(0)
    metrics.update(walk=walk.get("name"), walk_extent=ext.round(3).tolist(),
                   walk_min_y=float(pose.min(0)[1]))
    if not (0.85 * height_m <= ext[1] <= 1.1 * height_m):
        fails.append(f"walk height {ext[1]:.2f} m")
    # Upright: vertices dominated by head joints sit in the top quarter and
    # those dominated by foot joints in the bottom 15%. (A bounding-box aspect
    # test was the first idea; the source walks themselves are 1.74 x 1.04 m
    # at frame 0 -- full stride -- so aspect cannot separate upright from not.)
    dom = _dominant_joint_names(gltf, binary)
    y = pose[:, 1] - pose[:, 1].min()
    head = np.array([bool(_re_head.search(nm)) for nm in dom])
    foot = np.array([bool(_re_foot.search(nm)) for nm in dom])
    metrics.update(head_y=float(y[head].mean()) if head.any() else None,
                   foot_y=float(y[foot].mean()) if foot.any() else None)
    if not head.any() or y[head].mean() < 0.75 * ext[1]:
        fails.append(f"head not at top (head_y={metrics['head_y']})")
    if not foot.any() or y[foot].mean() > 0.15 * ext[1]:
        fails.append(f"feet not at bottom (foot_y={metrics['foot_y']})")
    if abs(pose.min(0)[1]) > 0.1 * height_m:
        fails.append(f"feet at y={pose.min(0)[1]:.2f}")
    # (c) no joint jumps between rest and the first walk frame, and no bone
    # changes length (a stretched bone is the stretched-arm bug).
    Wa = world_matrices(gltf, ov)
    move = max(float(np.linalg.norm(Wa[j][:3, 3] - W[j][:3, 3])) for j in sk["joints"])
    stretch = 0.0
    top_kids = {c for j in sk["joints"] for c in gltf["nodes"][j].get("children", [])
                if not any(j in gltf["nodes"][q].get("children", []) for q in sk["joints"])}
    for j in sk["joints"]:
        if j in top_kids or not any(j in gltf["nodes"][q].get("children", []) for q in sk["joints"]):
            continue                   # root + its direct children (IK targets, pelvis bob)
        r = np.linalg.norm(gltf["nodes"][j].get("translation", [0, 0, 0]))
        a = np.linalg.norm(ov.get(j, {}).get("translation", gltf["nodes"][j].get("translation", [0, 0, 0])))
        stretch = max(stretch, abs(a - r))
    trunk = max([float(np.linalg.norm(Wa[j][:3, 3] - W[j][:3, 3])) for j in sk["joints"]
                 if _re_trunk.search(gltf["nodes"][j].get("name", ""))] or [0.0])
    metrics.update(max_joint_move=move, max_trunk_move=trunk, max_bone_stretch=stretch)
    # Arms legitimately swing 0.6 m from a T-pose rest into the walk, so the
    # 0.3 m jump limit applies to the trunk (pelvis/spine/neck/head).
    if trunk > 0.3 * height_m / 1.75:
        fails.append(f"trunk joint moves {trunk:.2f} m between rest and walk frame 0")
    if stretch > 0.05 * height_m / 1.75:
        fails.append(f"bone stretch {stretch:.2f} m between rest and walk frame 0")
    return metrics, fails


if __name__ == "__main__":
    import sys
    print(json.dumps(rebake_file(sys.argv[1], sys.argv[2]), indent=1))

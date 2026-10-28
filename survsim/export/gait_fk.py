"""Forward kinematics on a glTF walk clip: foot tracks, stride, stance pose.

Pure numpy. Used to (1) derive a clip's real ground speed (stride per cycle /
cycle length) instead of guessing it, so play rate ∝ speed is calibrated, and
(2) pick the stance time (feet closest together) used as the idle pose when a
rig has no idle clip — a frozen mid-stride walk frame reads as wrong.
"""
from __future__ import annotations

import numpy as np

from .gltf_normalize import _accessor_view, qmul, qrot, read_glb


def _sample(times, values, t, path):
    i = int(np.clip(np.searchsorted(times, t) - 1, 0, len(times) - 2))
    t0, t1 = times[i], times[i + 1]
    a = 0.0 if t1 <= t0 else float(np.clip((t - t0) / (t1 - t0), 0, 1))
    v = (1 - a) * values[i] + a * values[i + 1]
    if path == "rotation":
        v = v / np.linalg.norm(v)
    return v


def joint_tracks(path, joint_names, n=60, anim_index=0):
    """World positions (n, 3) per named joint over one cycle, glTF units, Y-up."""
    gltf, binary = read_glb(path)
    nodes = gltf["nodes"]
    parent = {c: i for i, nd in enumerate(nodes) for c in nd.get("children", [])}
    anim = gltf["animations"][anim_index]
    chans = {}
    t_max = 0.0
    for ch in anim["channels"]:
        smp = anim["samplers"][ch["sampler"]]
        times, _ = _accessor_view(gltf, binary, smp["input"])
        vals, _ = _accessor_view(gltf, binary, smp["output"])
        chans[(ch["target"]["node"], ch["target"]["path"])] = (times[:, 0], vals)
        t_max = max(t_max, float(times[-1, 0]))
    ids = {nd.get("name"): i for i, nd in enumerate(nodes)}
    ts = np.linspace(0, t_max, n, endpoint=False)
    out = {name: np.zeros((n, 3)) for name in joint_names}
    for k, t in enumerate(ts):
        cache = {}

        def world(i, t=t, cache=cache):
            if i in cache:
                return cache[i]
            nd = nodes[i]
            trs = []
            for p, default in (("translation", [0, 0, 0]), ("rotation", [0, 0, 0, 1]),
                               ("scale", [1, 1, 1])):
                if (i, p) in chans:
                    trs.append(_sample(*chans[(i, p)], t, p))
                else:
                    trs.append(np.array(nd.get(p, default), float))
            if i in parent:
                pt, pr, ps = world(parent[i], t, cache)
                res = (pt + qrot(pr, ps * trs[0])[0], qmul(pr, trs[1])[0], ps * trs[2])
            else:
                res = tuple(trs)
            cache[i] = res
            return res

        for name in joint_names:
            out[name][k] = world(ids[name])[0]
    return ts, t_max, out


def gait_stats(path, left_foot, right_foot, up_axis=1, n=80):
    """Stride per cycle, stance time and height scale from a walk clip."""
    ts, length, tr = joint_tracks(path, [left_foot, right_foot], n)
    l, r = tr[left_foot], tr[right_foot]
    horiz = [a for a in range(3) if a != up_axis]
    d = l[:, horiz] - r[:, horiz]
    # Forward axis: the horizontal axis along which the feet separate most.
    fwd = horiz[int(np.argmax(np.ptp(d, axis=0)))]
    sep = l[:, fwd] - r[:, fwd]
    step = float(np.ptp(sep)) / 2.0            # step length: feet apart at double support
    stride = 2.0 * step                        # two steps per cycle
    stance = float(ts[int(np.argmin(np.abs(sep)))])
    heights = [p[:, up_axis] for p in (l, r)]
    return {"cycle_s": float(length), "step_length": step, "stride_per_cycle": stride,
            "ground_speed": stride / float(length) if length else None,
            "foot_min_up": float(min(h.min() for h in heights)),
            "stance_t": stance, "forward_axis": "xyz"[fwd]}

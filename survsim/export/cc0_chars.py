"""The CC0 character catalogue for renders (production default, docs/research/47 option A).

Reads the in-editor verification catalogue (scripts/ue/cc0_verify_one.py ->
Saved/survsim_characters_cc0.json) and the FK foot curves it sampled from each
character's OWN walk / run clips (configs/people/cc0_foot_curves/), and returns
render catalogue entries with a measured clip ground speed and facing:

* forward axis + sign: the horizontal component-space axis along which the
  planted foot moves, opposite to its motion (a planted foot slides backward
  under an in-place walk),
* ground speed: median planted-foot speed along that axis at rate 1 (the same
  quantity foot_slide.plan_slide drives to zero), normalised to a 1.75 m body,
* yaw_offset = -atan2(forward): the actor yaw that points the clip forward
  along the path.

Bodies are excluded if the editor check failed, if they are chibi-proportioned
(head > 22 % of stature: at CCTV standoff they read as children or toys), or if
they are on the explicit list below. Pure numpy.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CATALOG = Path.home() / "data/ue-projects/SurvSim/Saved/survsim_characters_cc0.json"
CURVES = REPO / "configs" / "people" / "cc0_foot_curves"
INDEX = Path.home() / "data/assets/index.json"
CONTACT_CM = 2.0
CHIBI_HEAD_FRAC = 0.22
REF_HEIGHT_M = 1.75


def load_curves(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    out = {"len_s": float(d["len_s"]), "t": np.asarray(d["t"], float),
           "ref_height_cm": float(d.get("ref_height_cm", 175.0))}
    for k in ("foot_l", "foot_r", "head"):
        if k in d:
            out[k] = np.asarray(d[k], float)
    for k in ("foot_l", "foot_r"):
        z = out[k][:, 2]
        out[k + "_contact"] = z < z.min() + CONTACT_CM
    return out


def gait_from_curves(c: dict) -> dict:
    """Forward direction, yaw offset and planted-foot ground speed (m/s, clip scale)."""
    vs = []
    for k in ("foot_l", "foot_r"):
        v = np.gradient(c[k][:, :2], c["t"], axis=0)
        vs.append(v[c[k + "_contact"]])
    v = np.vstack(vs)
    mean = v.mean(0)
    fwd = -mean / max(np.linalg.norm(mean), 1e-9)
    ang = math.degrees(math.atan2(fwd[1], fwd[0]))
    snap = round(ang / 90.0) * 90.0
    if abs(ang - snap) <= 5.0:          # a toe-out drift is not the walking direction
        fwd = np.array([math.cos(math.radians(snap)), math.sin(math.radians(snap))])
    along = -(v @ fwd)                      # planted-foot speed, positive
    med = float(np.median(along))
    # Hand-keyed clips do not hold the planted foot at a constant speed, so the
    # rate that minimises slide is not the median: search the speed s that
    # minimises the median residual |v + s*fwd| - what plan_slide measures on a
    # straight walk (the MetaHuman loop's 1.83 m/s was confirmed the same way).
    grid = np.linspace(0.6 * med, 1.4 * med, 161)
    res = [float(np.median(np.linalg.norm(v + g * fwd, axis=1))) for g in grid]
    speed = float(grid[int(np.argmin(res))]) / 100.0
    sep = c["foot_l"][:, :2] @ fwd - c["foot_r"][:, :2] @ fwd
    stride = float(np.ptp(sep))             # two steps per cycle, one step = ptp/2
    return {"forward": [round(float(fwd[0]), 4), round(float(fwd[1]), 4)],
            "yaw_offset": round(-math.degrees(math.atan2(fwd[1], fwd[0])), 2),
            "ground_speed_mps": round(speed, 4),
            "median_planted_mps": round(med / 100.0, 4),
            "residual_floor_cms": round(min(res), 2),
            "stride_speed_mps": round(stride / 100.0 / c["len_s"], 4),
            "cycle_s": round(c["len_s"], 4),
            "contact_frac": round(float(np.mean([c["foot_l_contact"].mean(),
                                                 c["foot_r_contact"].mean()])), 3)}


def _index() -> dict:
    if not INDEX.is_file():
        return {}
    return {a["name"]: a for a in json.loads(INDEX.read_text())["assets"]}


def cc0_catalog(catalog: Path = CATALOG, curves_dir: Path = CURVES) -> dict:
    """{name: render catalogue entry} for every verified CC0 body (excluded ones flagged)."""
    if not Path(catalog).is_file():
        return {}
    idx = _index()
    out = {}
    for name, v in json.loads(Path(catalog).read_text()).items():
        e = dict(v)
        meta = idx.get(name, {})
        e["license"] = meta.get("license", "LicenseRef-Unknown")
        e["author"] = meta.get("author")
        e["source_url"] = meta.get("source_url")
        why = list(v.get("fails") or [])
        if not str(e["license"]).startswith("CC0"):
            why.append(f"licence {e['license']}")
        wp = Path(curves_dir) / f"{name}_walk.json"
        if wp.is_file():
            c = load_curves(wp)
            g = gait_from_curves(c)
            scale = REF_HEIGHT_M * 100.0 / c["ref_height_cm"]
            e["gait"] = g
            e["walk_speed_mps"] = round(g["ground_speed_mps"] * scale, 3)
            e["yaw_offset"] = g["yaw_offset"]
            e["foot_curves"] = str(wp)
            if "head" in c:
                head_frac = 1.0 - float(c["head"][:, 2].min()) / c["ref_height_cm"]
                e["head_frac"] = round(head_frac, 3)
                if head_frac > CHIBI_HEAD_FRAC:
                    why.append(f"chibi proportions (head {head_frac:.0%} of stature)")
            if not 0.6 <= e["walk_speed_mps"] <= 2.2:
                why.append(f"walk ground speed {e['walk_speed_mps']} m/s implausible")
        else:
            why.append("no walk foot curves")
        rp = Path(curves_dir) / f"{name}_run.json"
        if rp.is_file() and e.get("run"):
            g = gait_from_curves(load_curves(rp))
            e["run_speed_mps"] = round(g["ground_speed_mps"] * REF_HEIGHT_M * 100.0
                                       / e.get("height_cm", 175.0), 3)
            if not 1.8 <= e["run_speed_mps"] <= 6.0:
                e.pop("run_speed_mps")          # no plausible run loop: walk faster instead
        e["kind"] = "cc0"
        e["exclude_reasons"] = why
        e["exclude"] = bool(why)
        out[name] = e
    return out

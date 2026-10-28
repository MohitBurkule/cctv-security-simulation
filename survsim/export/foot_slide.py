"""Residual foot slide of a person plan against the real walk clip (pure Python).

The in-editor job `footcurve` (docs/12 "Foot slide") samples the MetaHuman walk
loop's foot bones by AnimPose FK (component space, clip forward = +Y, 1.6 m
reference skeleton) into configs/people/mh_walk_foot_curves.json. Here the
clip is replayed exactly as the sequencer plays it — each walk section's play
rate from its start frame, the actor transform interpolated between keys, yaw
offset and actor scale applied — and the world velocity of whichever foot is
in contact (within 2 cm of its lowest height) is measured. A perfect match of
rate to ground speed gives 0: the planted foot stays put.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

CURVES = Path(__file__).resolve().parents[2] / "configs" / "people" / "mh_walk_foot_curves.json"
CONTACT_CM = 2.0
CLIP_REF_CM = 160.0


def load_curves(path: Path = CURVES) -> dict:
    d = json.loads(Path(path).read_text())
    out = {"len_s": d["len_s"], "t": np.array(d["t"])}
    for k in ("foot_l", "foot_r"):
        a = np.array(d[k])
        out[k] = a
        out[k + "_contact"] = a[:, 2] < a[:, 2].min() + CONTACT_CM
    return out


def clip_stance_speed_mps(curves: dict) -> float:
    """Ground speed implied by the planted foot at rate 1 (1.6 m skeleton)."""
    sp = []
    for k in ("foot_l", "foot_r"):
        v = np.gradient(curves[k], curves["t"], axis=0)
        sp.append(np.abs(v[curves[k + "_contact"], 1]).mean())
    return float(np.mean(sp)) / 100.0


def _interp_keys(entry: dict, t_frames: np.ndarray):
    f = np.array([k["frame"] for k in entry["keys"]], float)
    loc = np.array([k["location"][:2] for k in entry["keys"]], float)
    yaw = np.array(entry["yaws"], float)
    x = np.interp(t_frames, f, loc[:, 0])
    y = np.interp(t_frames, f, loc[:, 1])
    w = np.interp(t_frames, f, yaw)
    return x, y, w


def plan_slide(plan: list[dict], fps: float, curves: dict | None = None,
               yaw_offset: float = -90.0, sub: int = 6,
               per_character: dict[str, tuple[dict, float, float]] | None = None) -> dict:
    """Mean / p90 planted-foot world speed (cm/s) over all walk sections.

    `per_character` maps a character name to (curves, yaw_offset, reference
    stature cm) for rigs with their own walk clip (the CC0 bodies,
    survsim/export/cc0_chars.py); people on other characters use `curves`.
    """
    default = curves or (load_curves() if not per_character else None)
    speeds = []
    for e in plan:
        if per_character and e.get("character") in per_character:
            curves, yaw_offset_e, ref_cm = per_character[e["character"]]
        elif default is not None:
            curves, yaw_offset_e, ref_cm = default, yaw_offset, CLIP_REF_CM
        else:
            continue
        L = curves["len_s"]
        scale = e["height_cm"] / ref_cm
        for seg in e["segments"]:
            if seg["clip"] != "walk" or seg["end"] - seg["start"] < 3:
                continue
            tf = np.arange(seg["start"] * sub, seg["end"] * sub) / sub     # frames
            tf = tf[(tf >= e["keys"][0]["frame"]) & (tf <= e["keys"][-1]["frame"])]
            if len(tf) < 4:
                continue
            x, y, w = _interp_keys(e, tf)
            tau = ((tf - seg["start"]) / fps * seg["rate"]) % L
            for k in ("foot_l", "foot_r"):
                fl = np.stack([np.interp(tau, curves["t"], curves[k][:, i]) for i in (0, 1)], 1)
                cont = np.interp(tau, curves["t"], curves[k + "_contact"].astype(float)) > 0.5
                a = np.radians(w + yaw_offset_e)
                wx = x + scale * (np.cos(a) * fl[:, 0] - np.sin(a) * fl[:, 1])
                wy = y + scale * (np.sin(a) * fl[:, 0] + np.cos(a) * fl[:, 1])
                dt = 1.0 / (fps * sub)
                v = np.hypot(np.diff(wx), np.diff(wy)) / dt
                c = cont[1:] & cont[:-1] & (np.diff(tau) > 0)    # skip loop wrap
                speeds.extend(v[c].tolist())
    s = np.array(speeds) if speeds else np.zeros(1)
    return {"stance_samples": len(speeds), "slide_mean_cms": round(float(s.mean()), 2),
            "slide_p90_cms": round(float(np.percentile(s, 90)), 2),
            "slide_median_cms": round(float(np.median(s)), 2)}


def limit_turn_rate(yaws: list[float], frames: list[int], fps: float,
                    max_dps: float = 240.0, big_deg: float = 45.0) -> list[float]:
    """Turn-in-place blending: a heading change > `big_deg` is spread at no more
    than `max_dps`, started early so the body is round by the time it moves off
    (backward pass) and never overshoots (forward pass)."""
    if len(yaws) < 2:
        return list(yaws)
    out = list(yaws)
    for i in range(1, len(out)):              # forward: cap speed
        step = max_dps * max(1, frames[i] - frames[i - 1]) / fps
        d = out[i] - out[i - 1]
        if abs(yaws[i] - yaws[i - 1]) > big_deg or abs(d) > step:
            out[i] = out[i - 1] + math.copysign(min(abs(d), step), d)
    # backward: start big turns early instead of lagging behind the path
    back = list(yaws)
    for i in range(len(back) - 2, -1, -1):
        step = max_dps * max(1, frames[i + 1] - frames[i]) / fps
        d = back[i] - back[i + 1]
        if abs(d) > step:
            back[i] = back[i + 1] + math.copysign(step, d)
    return [(a + b) / 2.0 for a, b in zip(out, back)]

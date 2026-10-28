"""Submit scripts/ue/mh_lineup.py: N MetaHumans in N different outfits, one frame.

    .venv/bin/python tools/mh_lineup.py --n 8 --out out/demo_humans_v2 --name mh_lineup.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def project(cam: dict, p) -> tuple[float, float]:
    """UE world point (cm) -> pixel, pinhole with horizontal fov (UE: yaw about Z, X forward, Y right)."""
    import math
    w, h = cam["res"]
    y, pi = math.radians(cam["yaw"]), math.radians(cam["pitch"])
    f = (math.cos(pi) * math.cos(y), math.cos(pi) * math.sin(y), math.sin(pi))
    r = (-math.sin(y), math.cos(y), 0.0)
    u = (-math.sin(pi) * math.cos(y), -math.sin(pi) * math.sin(y), math.cos(pi))
    d = [p[i] - cam["loc"][i] for i in range(3)]
    zf = sum(d[i] * f[i] for i in range(3))
    k = (w / 2) / math.tan(math.radians(cam["fov"]) / 2)
    return (w / 2 + k * sum(d[i] * r[i] for i in range(3)) / zf,
            h / 2 - k * sum(d[i] * u[i] for i in range(3)) / zf)


def chroma(rgb):
    s = sum(rgb) + 1e-6
    return [v / s for v in rgb]


def hue_error(expected, measured) -> float:
    """0..1: hue distance / 180 deg, or the saturation gap when either is near grey.

    Luminance-invariant, and tolerant of the tone curve desaturating colours:
    only a wrong hue (or colour where grey was asked for) scores high.
    """
    import colorsys
    he, se, _ = colorsys.rgb_to_hsv(*[max(0.0, float(v)) for v in expected])
    hm, sm, _ = colorsys.rgb_to_hsv(*[max(0.0, float(v)) for v in measured])
    if se < 0.25:                       # grey asked for: fail only on real colour
        return max(0.0, sm - se - 0.1)
    if sm < 0.15:                       # colour asked for, grey measured
        return 1.0 - sm / 0.15 * 0.5
    d = abs(he - hm) % 1.0
    return min(d, 1.0 - d) * 2.0


def region_colour_check(img, cam: dict, people: list[dict], positions: list, tol: float = 0.25):
    """Per person: colour of the torso and thigh bands vs the outfit.

    Band = rows 28-42 % (top, below the neckline) / 50-60 % (hips, no leg gap) of the projected height,
    across a body-wide window, excluding ground-like pixels; the brightest
    third (the sun-lit side) is averaged. A dark expected colour (linear
    luminance < 0.05: black, navy, charcoal...) must measure darker than the
    ground; a brighter one must match in hue (hue_error; luminance-invariant,
    after dividing out the ground's chromaticity as white balance).
    """
    import numpy as np
    rows = []
    lin = (img[..., ::-1].astype(np.float64) / 255.0) ** 2.2
    gpatch = lin[-60:, :, :].reshape(-1, 3)
    ground = gpatch.mean(0)
    wb = np.asarray(chroma(ground)) * 3.0
    g_lum = float(ground.mean())
    for e, (x, y, z, hcm) in zip(people, positions):
        u0, v_feet = project(cam, (x, y, z))
        u1, v_head = project(cam, (x, y, z + hcm))
        hp = v_feet - v_head
        half = max(2, int(0.06 * hp))  # torso/legs only: arms and hands hang at +-0.1 hp
        out = {"character": e["character"], "height_px": round(hp, 1)}
        for part, (fa, fb), key in (("top", (0.28, 0.42), "top_rgb"),
                                    ("bottom", (0.50, 0.60), "bottom_rgb")):
            v0, v1 = int(v_head + fa * hp), int(v_head + fb * hp)
            uc = int(u1 + (u0 - u1) * (fa + fb) / 2)
            px = lin[max(0, v0):max(v0 + 1, v1), max(0, uc - half):uc + half].reshape(-1, 3)
            diff = np.abs(px - ground).sum(1) > 0.06          # not ground
            px = px[diff] if diff.sum() >= 8 else px
            lum = px.mean(1)
            lit = px[lum >= np.quantile(lum, 0.67)]
            m = lit.mean(0) / wb
            exp = np.asarray(e["outfit"][key])
            if float(exp.mean()) < 0.05:
                err = float(m.mean()) / max(g_lum, 1e-6)       # dark: relative luminance
                ok = err < 0.6
            else:
                err = hue_error(exp, m)
                ok = err <= tol
            out[part] = {"expected": e["outfit"][part if part == "top" else "bottom"],
                         "err": round(err, 3), "ok": ok}
        rows.append(out)
    return rows


def main() -> int:
    from survsim.export.outfits import outfit
    from tools.render_run import metahuman_catalog
    from tools.ue_job import submit
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hivis", type=float, default=0.1)
    ap.add_argument("--distance", type=float, default=7.0)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--camera", default="cam_01")
    ap.add_argument("--out", type=Path, default=ROOT / "out/demo_humans_v2")
    ap.add_argument("--name", default="mh_lineup.png")
    ap.add_argument("--ev-offset", type=float, default=None,
                    help="exposure bias in stops; default: meter (one probe, luma -> 100)")
    a = ap.parse_args()
    cat = metahuman_catalog()
    names = sorted(cat)
    people = []
    for i in range(a.n):
        uid = f"lineup_{i}"
        people.append({"character": names[i % len(names)], "height_cm": 160 + (i * 7) % 30,
                       "outfit": outfit(uid, a.seed, a.hivis)})
    import math

    import cv2
    req = {"people": people, "catalog": cat, "camera": a.camera, "distance_m": a.distance,
           "fov": a.fov, "out": str(a.out.resolve()), "name": a.name,
           "ev_offset": a.ev_offset or 0.0}
    if a.ev_offset is None:
        Path("/tmp/survsim_mh_lineup.json").write_text(json.dumps(dict(req, name="_meter.png")))
        r = submit(ROOT / "scripts/ue/mh_lineup.py", wait=240)
        if r.get("ok"):
            lum = float(cv2.cvtColor(cv2.imread(r["out"]), cv2.COLOR_BGR2GRAY).mean())
            req["ev_offset"] = max(-3.0, min(3.0, math.log2(100.0 / max(lum, 1.0))))
            print(f"meter: luma {lum:.0f} -> ev offset {req['ev_offset']:+.2f}")
    Path("/tmp/survsim_mh_lineup.json").write_text(json.dumps(req))
    (a.out / (Path(a.name).stem + ".json")).parent.mkdir(parents=True, exist_ok=True)
    (a.out / (Path(a.name).stem + ".json")).write_text(json.dumps(people, indent=1))
    r = submit(ROOT / "scripts/ue/mh_lineup.py", wait=240)
    if not r.get("ok"):
        print(json.dumps(r, indent=1)[:2000])
        return 1
    import cv2
    img = cv2.imread(r["out"])
    rows = region_colour_check(img, r["camera"], people, r["people_cm"])
    bad = [(i, p) for i, row in enumerate(rows) for p in ("top", "bottom") if not row[p]["ok"]]
    (a.out / (Path(a.name).stem + "_check.json")).write_text(json.dumps(rows, indent=1))
    for i, row in enumerate(rows):
        print(i, row["character"], row["height_px"], row["top"], row["bottom"])
    lum = float(img.mean())
    print(f"frame mean luma {lum:.0f}; colour check {'PASS' if not bad else f'FAIL {bad}'}")
    return 0 if not bad and 20 < lum < 235 else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""One gated frame per (site type x seed) from the v2 scene generator + contact sheet.

    .venv/bin/python tools/render_scene_variety.py --seeds 1 2 3 --out out/scene_variety_v2

Per scene: survsim.generator.layout.build_layout -> a people-free "run" dir
(layout.json, unreal/manifest.json with the layout's installer cameras,
meta/scenario.resolved.json) -> tools/render_run.py --layout (build + entrance
gate + frustum gate + metering + render) -> scripts/check_render.py. Per-scene
build/round-trip seconds go to <out>/variety.json; the sheet to <out>/sheet.png.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def scene_dir(out: Path, site: str, seed: int, camera: str | None = None) -> tuple[Path, dict]:
    from survsim.export.env_plan import layout_camera_manifest
    from survsim.generator.layout import build_layout
    lay = build_layout(site, seed).to_dict()
    d = out / f"{site}_s{seed}"
    (d / "unreal").mkdir(parents=True, exist_ok=True)
    (d / "meta").mkdir(parents=True, exist_ok=True)
    (d / "layout.json").write_text(json.dumps(lay, indent=1))
    cams = layout_camera_manifest(lay)
    (d / "unreal" / "manifest.json").write_text(json.dumps(
        {"version": 1, "scenario": f"{site}_s{seed}", "seed": seed, "fps": 10.0,
         "duration_s": 1.0, "units": "cm", "cameras": cams, "people": [], "tracks": []}))
    (d / "meta" / "scenario.resolved.json").write_text(json.dumps(
        {"environment": {"id": f"{site}_s{seed}", "site_type": site}}))
    return d, lay


def label(lay: dict) -> str:
    t, w, v = lay["time"], lay["weather"], lay["variant"]
    keys = [f"{k}={v[k]}" for k in ("shape", "street", "event", "boundary", "neighbour",
                                     "rows_along", "bus_bays", "docks") if k in v][:3]
    return (f"{lay['site_type']} s{lay['seed']}  {t['city']} {t['date']} "
            f"{t['local_hour']:.1f}h sun {t['sun_elevation']:.0f}deg {w['state']}  "
            + " ".join(keys))


def sheet(rows: list[dict], path: Path, cols: int = 3) -> None:
    from PIL import Image, ImageDraw
    tw, th = 640, 360
    n = len(rows)
    img = Image.new("RGB", (tw * cols, th * ((n + cols - 1) // cols)), (0, 0, 0))
    dr = ImageDraw.Draw(img)
    for i, r in enumerate(rows):
        x, y = (i % cols) * tw, (i // cols) * th
        if r.get("frame") and Path(r["frame"]).is_file():
            img.paste(Image.open(r["frame"]).convert("RGB").resize((tw, th)), (x, y))
        else:
            dr.text((x + 10, y + 170), "FAILED", fill=(255, 0, 0))
        dr.rectangle([x, y, x + tw, y + 16], fill=(0, 0, 0))
        dr.text((x + 4, y + 2), r["label"][:110], fill=(255, 255, 0))
    img.save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--out", type=Path, default=Path("out/scene_variety_v2"))
    a = ap.parse_args()
    from survsim.generator.layout import SITE_TYPES
    a.out.mkdir(parents=True, exist_ok=True)
    vj = a.out / "variety.json"
    old = json.loads(vj.read_text()) if vj.is_file() else []   # re-runs replace their rows
    prev_fail = {(r["site"], r["seed"]): r for r in old if r["render_rc"] or r["gate_rc"]}
    rows = [r for r in old if not ((r["site"] in (a.sites or SITE_TYPES)) and r["seed"] in a.seeds)]
    for site in a.sites or SITE_TYPES:
        for seed in a.seeds:
            d, lay = scene_dir(a.out, site, seed)
            cam = lay["cameras"][0]["id"]
            t0 = time.time()
            cmd = [sys.executable, str(ROOT / "tools/render_run.py"), str(d), "--frames", "1",
                   "--cameras", cam, "--lighting", "day", "--layout", str(d / "layout.json"),
                   "--rebuild-env", "--out", str(d / "render"), "--wait", "900"]
            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
            (d / "render_run.log").write_text(p.stdout + p.stderr)
            dt = time.time() - t0
            frames = sorted((d / "render" / cam).glob("*.png"))
            gate = subprocess.run([sys.executable, str(ROOT / "scripts/check_render.py"),
                                   str(d / "render"), "--json", str(d / "check_render.json")],
                                  cwd=ROOT, capture_output=True, text=True, check=False)
            (d / "check_render.txt").write_text(gate.stdout + gate.stderr)
            rp = d / "render_result.json"
            res = json.loads(rp.read_text()) if rp.is_file() else {}
            mp = d / "meter_result.json"          # the env is built by the probe job
            if mp.is_file():
                mres = json.loads(mp.read_text())
                for k in ("env_build_s", "unbuilt", "decals", "entrance_gate"):
                    res.setdefault(k, mres.get(k))
            row = {"site": site, "seed": seed, "camera": cam, "render_rc": p.returncode,
                   "gate_rc": gate.returncode, "round_trip_s": round(dt, 1),
                   "env_build_s": res.get("env_build_s"), "vram_mb": res.get("vram_mb"),
                   "entrance_gate": res.get("entrance_gate"), "unbuilt": res.get("unbuilt"),
                   "decals": res.get("decals"),
                   "frame": str(frames[0]) if frames else None, "label": label(lay),
                   "variant": lay["variant"], "time": lay["time"], "weather": lay["weather"]}
            pf = prev_fail.get((site, seed))
            if pf:     # keep why the earlier attempt failed (the gate's own words)
                log = pf.get("failure", "")
                row["previous_failure"] = log
            if p.returncode or gate.returncode:
                err = [ln for ln in (p.stdout + p.stderr).splitlines() if "RuntimeError" in ln]
                row["failure"] = (err[-1] if err else gate.stdout[-400:])[:600]
            rows.append(row)
            print(f"{site} s{seed}: render rc={p.returncode} gate rc={gate.returncode} "
                  f"build {row['env_build_s']} s, round trip {dt:.0f} s", flush=True)
            (a.out / "variety.json").write_text(json.dumps(rows, indent=1, default=str))
    order = {s: i for i, s in enumerate(SITE_TYPES)}
    rows.sort(key=lambda r: (order[r["site"]], r["seed"]))
    (a.out / "variety.json").write_text(json.dumps(rows, indent=1, default=str))
    sheet(rows, a.out / "sheet.png")
    bad = [f"{r['site']}_s{r['seed']}" for r in rows if r["render_rc"] or r["gate_rc"]]
    print(f"{len(rows) - len(bad)}/{len(rows)} passed; failed: {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())

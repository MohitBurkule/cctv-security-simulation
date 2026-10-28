"""Render a batch: scenarios x lighting presets, all cameras, via the persistent editor.

    .venv/bin/python tools/render_batch.py out/render_demo out/runs/disguise_and_tamper \
        --lighting day night --frames 300 --out out/batch_v1

Each run dir must already hold gt/, meta/ and unreal/manifest.json (survsim run +
survsim export). Output per item: <out>/<run>_<lighting>/{gt,meta,unreal} (symlinks
to the run) + render/<cam>/NNNN.png, then scripts/check_render.py and
`survsim video`. Prints frames and wall time per item and in total.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--lighting", nargs="+", default=["day", "night"])
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("out/batch_v1"))
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--sites", nargs="*", default=None,
                    help="site type per run (env_plan.SITE_TYPES), same order as the runs")
    ap.add_argument("--env-v1", action="store_true", help="v1 environment plans")
    ap.add_argument("--entrance-gate", choices=["fail", "warn"], default="fail")
    ap.add_argument("--id-mask", action="store_true",
                    help="render the per-person ID pass; gt/ holds mask-derived boxes "
                         "(a real dir, not a symlink to the run)")
    ap.add_argument("--window", nargs=2, type=float, metavar=("T0", "T1"), default=None,
                    help="render simulation time [T0, T1) s only (tools/render_run.py --window)")
    ap.add_argument("--tightness-gate", choices=["day", "all", "off"], default="day",
                    help="with --id-mask: run survsim.checks.box_tightness (yolov8n on CPU vs "
                         "the mask boxes) and fail the set on it for these lightings. 'day': "
                         "night sets are measured and reported but not gated - the detector "
                         "is the reference and misses dim people (box_tightness docstring)")
    ap.add_argument("--suffix", default="", help="appended to each item name, e.g. _tightgt")
    a = ap.parse_args()
    t_all = time.time()
    total = 0
    rows = []
    for ri, run in enumerate(a.runs):
        first = True
        for light in a.lighting:
            item = a.out / f"{run.name}_{light}{a.suffix}"
            item.mkdir(parents=True, exist_ok=True)
            for sub in ("gt", "meta", "unreal", "calib"):
                if sub == "gt" and a.id_mask:
                    continue
                link = item / sub
                if (run / sub).exists() and not link.exists():
                    link.symlink_to((run / sub).resolve())
            t0 = time.time()
            cmd = [PY, str(ROOT / "tools/render_run.py"), str(run), "--frames", str(a.frames),
                   "--lighting", light, "--out", str(item / "render"), "--wait", "3000"]
            if first:
                cmd.append("--rebuild-env")
            if a.sites:
                cmd += ["--site", a.sites[ri]]
            if a.env_v1:
                cmd.append("--env-v1")
            cmd += ["--entrance-gate", a.entrance_gate]
            if a.id_mask:
                cmd.append("--id-mask")
            if a.window:
                cmd += ["--window", str(a.window[0]), str(a.window[1])]
            rc = subprocess.run(cmd, cwd=ROOT, check=False).returncode
            first = False
            n = len(list((item / "render").glob("*/*.png")))
            total += n
            gate = subprocess.run([PY, str(ROOT / "scripts/check_render.py"), str(item / "render"),
                                   "--gt", str(item)], cwd=ROOT, capture_output=True, text=True,
                                  check=False)
            audit = subprocess.run([str(Path(PY).parent / "survsim"), "audit-motion", str(run)],
                                   cwd=ROOT, capture_output=True, text=True, check=False)
            text = gate.stdout + gate.stderr + "\n--- survsim audit-motion ---\n" + audit.stdout
            gate_rc = gate.returncode
            if a.id_mask and a.tightness_gate != "off" and (item / "render").is_dir():
                # CPU: the render set's VRAM budget is shared with another GPU job.
                bt = subprocess.run([PY, "-m", "survsim.checks.box_tightness", str(item),
                                     "--json", str(item / "box_tightness.json")],
                                    cwd=ROOT, capture_output=True, text=True, check=False,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
                gated = a.tightness_gate == "all" or light != "night"
                text += (f"\n--- box tightness ({'gated' if gated else 'reported, not gated'}) ---\n"
                         + bt.stdout + bt.stderr[-2000:])
                if gated and bt.returncode != 0 and gate_rc == 0:
                    gate_rc = bt.returncode
            (item / "check_render.txt").write_text(text)
            if gate_rc != 0:
                print(f"{item.name}: GATE FAILED — no video encoded", flush=True)
            if not a.no_video and gate_rc == 0:
                # Encodes are slow (degrade ~2 s/frame): one background process
                # per set so the next render starts at once.
                with open(item / "video.log", "w") as log:
                    subprocess.Popen([str(Path(PY).parent / "survsim"), "video",
                                      str(item / "render"), "--out", str(item / "video"),
                                      "--no-report"], cwd=ROOT, stdout=log, stderr=log,
                                     start_new_session=True)
            dt = time.time() - t0
            rows.append((item.name, rc, gate_rc, n, dt))
            print(f"{item.name}: render rc={rc} gate rc={gate_rc} frames={n} {dt:.0f}s",
                  flush=True)
    failed = [r[0] for r in rows if r[2] != 0]
    print(f"TOTAL frames={total} wall={time.time() - t_all:.0f}s "
          f"passed={len(rows) - len(failed)}/{len(rows)} failed={failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

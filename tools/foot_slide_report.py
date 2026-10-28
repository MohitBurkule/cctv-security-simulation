"""Residual foot slide of the render plan for runs, per character stack (docs/12 "Foot slide").

    .venv/bin/python tools/foot_slide_report.py out/runs_v2/smoke_mock ... [--chars cc0|metahuman]

Builds the same person plan tools/render_run.py renders (labels from
gt/world.parquet, the stack's characters and measured clip speeds) and replays
each walk section against the clip's FK foot curves (survsim/export/foot_slide.py).
CC0 bodies use their own walk clip's curves; MetaHumans the AS_MH walk loop's.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--chars", default="cc0", choices=["cc0", "metahuman"])
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--only", default=None, help="comma list: restrict to these characters")
    a = ap.parse_args()
    from survsim.export.anim_plan import person_plan
    from survsim.export.cc0_chars import cc0_catalog, load_curves
    from survsim.export.foot_slide import plan_slide
    from tools.render_run import label_keys, metahuman_catalog
    if a.chars == "cc0":
        chars = {k: v for k, v in cc0_catalog().items() if not v.get("exclude")}
        if a.only:
            chars = {k: v for k, v in chars.items() if k in a.only.split(",")}
        per = {k: (load_curves(Path(v["foot_curves"])), v["yaw_offset"], v["height_cm"])
               for k, v in chars.items()}
    else:
        chars, per = metahuman_catalog(baked_dir=None), None
    out = {}
    for run in a.runs:
        data = json.loads((run / "unreal" / "manifest.json").read_text())
        last = max((k["frame"] for t in data.get("tracks", []) for k in t["keys"]), default=0)
        n = min(a.frames, last + 1)
        label_keys(data, run)
        for t in data.get("tracks", []):
            t.pop("character", None)
        plan = person_plan(data, n, chars)
        r = plan_slide(plan, float(data.get("fps", 10)), per_character=per)
        out[run.name] = r
        print(f"{run.name}: {a.chars} median {r['slide_median_cms']} mean {r['slide_mean_cms']} "
              f"p90 {r['slide_p90_cms']} cm/s ({r['stance_samples']} stance samples)")
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Render gate: fail loudly, by named check, on known render failure classes.

    scripts/check_render.py out/render_demo [--gt out/render_demo] [--log editor.log] [--json r.json]

Exit 0 only if no check FAILs. docs/12 "Automated render gate".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from survsim.checks.core import format_table, worst
from survsim.checks.render import run_all


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("render_dir", type=Path)
    ap.add_argument("--gt", type=Path, help="run dir (containing gt/) or the gt dir itself")
    ap.add_argument("--log", type=Path, help="Unreal editor log to scan")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--expected", type=int, help="expected frames per camera")
    ap.add_argument("--frame-offset", type=int, default=0,
                    help="GT frame number of image index 0 (default 0: sequence frame = GT frame)")
    ap.add_argument("--detector", action="store_true", help="run yolov8n vs GT (slow-ish)")
    ap.add_argument("--min-recall", type=float, default=None)
    ap.add_argument("--det-frames", type=int, default=5)
    ap.add_argument("--no-realism", action="store_true")
    a = ap.parse_args(argv)

    gt = a.gt
    if gt is not None and (gt / "gt").is_dir():
        gt = gt / "gt"
    t0 = time.perf_counter()
    checks = run_all(a.render_dir, gt, a.log, a.expected, a.frame_offset, a.detector,
                     a.min_recall, a.det_frames, realism=not a.no_realism)
    dt = time.perf_counter() - t0
    verdict = worst(checks)
    print(format_table(checks))
    failed = [c.name for c in checks if c.failed]
    print(f"\n{verdict}: {len(failed)} failed of {len(checks)} checks in {dt:.1f}s"
          + (f" -> {', '.join(failed)}" if failed else ""))
    if a.json:
        a.json.write_text(json.dumps({"verdict": verdict, "seconds": round(dt, 2),
                                      "render_dir": str(a.render_dir),
                                      "checks": [c.as_dict() for c in checks]}, indent=1, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

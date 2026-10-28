#!/usr/bin/env python3
"""Pass/fail an in-editor job runner result file, surfacing the traceback.

    scripts/check_job_result.py /tmp/survsim_jobs/<name>.result.json [...]

The editor prints "Python script executed successfully" for scripts that did
nothing; the result file is the only evidence. Exit 1 if any job failed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from survsim.checks.core import format_table
from survsim.checks.job import check_job_result, load, traceback_of


def main(argv: list[str] | None = None) -> int:
    paths = [Path(p) for p in (argv if argv is not None else sys.argv[1:])]
    if not paths:
        print(__doc__)
        return 2
    rc = 0
    for p in paths:
        data = load(p)
        checks = check_job_result(data, p.name.removesuffix(".result.json"))
        print(format_table(checks))
        if any(c.failed for c in checks):
            rc = 1
            tb = traceback_of(data)
            if tb:
                print(f"\n--- traceback from {p} ---\n{tb.rstrip()}\n")
    print("FAIL" if rc else "PASS")
    return rc


if __name__ == "__main__":
    sys.exit(main())

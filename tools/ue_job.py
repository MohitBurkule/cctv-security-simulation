"""Submit a Python job to the persistent editor (scripts/ue/job_runner.py).

    .venv/bin/python tools/ue_job.py job.py [--wait 180] [--name foo]

Waits (bounded) for <name>.result.json, and meanwhile watches the editor log
for "LogPython: Error" so a failing job is reported at once, not at timeout.
Exit 0 only if the result says ok. A job that sets RESULT["async"] (renders)
is waited on until it calls finish().
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

JOB_DIR = Path("/tmp/survsim_jobs")
LOG_DIR = Path.home() / "data/ue-projects/SurvSim/Saved/Logs"


def _log_size():
    # The persistent editor may log to SurvSim_2.log if another instance held
    # SurvSim.log at boot; watch every current log.
    return {p: p.stat().st_size for p in LOG_DIR.glob("SurvSim*.log")}


def _new_errors(sizes):
    out = []
    for p, n in sizes.items():
        if p.exists():
            with open(p, "rb") as fh:
                fh.seek(n)
                out += [l for l in fh.read().decode("utf8", "replace").splitlines()
                        if "LogPython: Error" in l]
    return out


def editor_alive() -> bool:
    """The runner writes '<time> <pid>' to heartbeat every 5 s from its tick.

    A long synchronous job stalls the tick, so staleness alone is not death;
    the pid is the authority, the heartbeat age is reported for context.
    """
    hb = JOB_DIR / "heartbeat"
    try:
        fields = hb.read_text().split()
        os.kill(int(fields[1]), 0)
        return True
    except (OSError, ValueError):
        return False


import re

ISSUE_RE = re.compile(r"Error:|Warning:.*(Insufficient|Failed to save|Failed to convert|Transient)"
                      r"|Failed to allocate Device Memory"
                      r"|Assertion failed|Fatal")


def _log_issues(sizes):
    """Every category (LogSavePackage, LogInterchange, LogMaterial, ...), not
    just LogPython: editor toasts only ever appear in these lines."""
    out = []
    for p, n in sizes.items():
        if p.exists():
            with open(p, "rb") as fh:
                fh.seek(n)
                out += [l[:300] for l in fh.read().decode("utf8", "replace").splitlines()
                        if ISSUE_RE.search(l)]
    return out[-50:]


VRAM_LIMIT_MB = 7200   # 8 GB card: above ~7 GB Vulkan allocations fail and frames go white


def vram_mb() -> int | None:
    """GPU memory in use (nvidia-smi), or None if it cannot be read."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=10, check=False).stdout
        return int(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def _attach_issues(result_path, data, sizes):
    data["log_issues"] = _log_issues(sizes)
    data["vram_mb"] = vram_mb()
    alloc = [l for l in data["log_issues"] if "Failed to allocate" in l]
    if data["vram_mb"] and data["vram_mb"] > VRAM_LIMIT_MB:
        data["ok"] = False
        data["error"] = f"VRAM {data['vram_mb']} MB > {VRAM_LIMIT_MB} MB (white-frame risk)"
    elif alloc:
        data["ok"] = False
        data["error"] = f"{len(alloc)} GPU allocation failures in the editor log"
    try:
        result_path.write_text(json.dumps(data, indent=2, default=str))
    except OSError:
        pass
    return data


def submit(script: Path, name: str | None = None, wait: float = 180.0) -> dict:
    name = name or f"{int(time.time()*1000)}_{script.stem}"
    result = JOB_DIR / f"{name}.result.json"
    result.unlink(missing_ok=True)
    if not (JOB_DIR / "runner.ready").exists():
        raise SystemExit("runner not ready: scripts/ue_mcp.sh start")
    sizes = _log_size()
    tmp = JOB_DIR / f"{name}.py.tmp"
    shutil.copy(script, tmp)
    tmp.rename(JOB_DIR / f"{name}.py")
    t0 = time.time()
    peak = [0]
    last_q = 0.0
    while time.time() - t0 < wait:
        if time.time() - last_q > 3.0:             # peak VRAM while the job runs
            v = vram_mb()
            peak[0] = max(peak[0], v or 0)
            last_q = time.time()
        if result.exists():
            data = json.loads(result.read_text())
            if not data.get("async") or "seconds" in data:
                data["vram_peak_mb"] = peak[0]
                return _attach_issues(result, data, sizes)
        if not editor_alive():
            return {"ok": False, "error": "editor process is gone (crash?): "
                    "see ~/data/ue-projects/SurvSim/Saved/Crashes", "pending": name,
                    "vram_peak_mb": peak[0], "log_issues": _log_issues(sizes)}
        errs = _new_errors(sizes)
        if errs:
            time.sleep(1.0)
            if result.exists():
                return _attach_issues(result, json.loads(result.read_text()), sizes)
            return {"ok": False, "error": "\n".join(errs[-10:]), "from": "log",
                    "log_issues": _log_issues(sizes)}
        time.sleep(1.0)
    return {"ok": False, "error": f"timeout after {wait}s", "pending": name}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("script", type=Path)
    ap.add_argument("--wait", type=float, default=180.0)
    ap.add_argument("--name")
    a = ap.parse_args()
    data = submit(a.script, a.name, a.wait)
    print(json.dumps(data, indent=2)[:6000])
    return 0 if data.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())

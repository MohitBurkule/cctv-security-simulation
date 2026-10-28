"""Submit a job to the MetaHuman sandbox editor (see tools/mh_editor.py).

    .venv/bin/python tools/mh_job.py job.py [--wait 180]

tools/ue_job.py pointed at the SurvSimMH editor's job and log dirs, plus a
wider log watch: editor toasts and message-log entries (LogSavePackage,
LogEditor, LogMetaHuman*, LogInterchange, LogSlate, ...) never pass through
LogPython, so every new line matching LOG_ALERT anywhere in the log is copied
into the result as "log_alerts". A job whose Python succeeded but whose editor
said "Insufficient writing permission" is not a success.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ue_job

LOG_ALERT = re.compile(r"Error:|Insufficient|Failed to save|Assertion failed|Fatal")
ue_job.JOB_DIR = Path(os.environ.get("SURVSIM_JOB_DIR", "/tmp/survsim_jobs_mh"))
ue_job.LOG_DIR = Path.home() / "data/ue-projects/SurvSimMH/Saved/Logs"


def _log_size():
    return {p: p.stat().st_size for p in ue_job.LOG_DIR.glob("SurvSimMH*.log")
            if "backup" not in p.name}


ue_job._log_size = _log_size


def alerts_since(sizes: dict) -> list[str]:
    out = []
    for p in ue_job.LOG_DIR.glob("SurvSimMH*.log"):
        if "backup" in p.name:
            continue
        with open(p, "rb") as fh:
            fh.seek(sizes.get(p, 0))
            out += [ln[:400] for ln in fh.read().decode("utf8", "replace").splitlines()
                    if LOG_ALERT.search(ln)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("script", type=Path)
    ap.add_argument("--wait", type=float, default=180.0)
    ap.add_argument("--name")
    a = ap.parse_args()
    sizes = _log_size()
    data = ue_job.submit(a.script, a.name, a.wait)
    data["log_alerts"] = alerts_since(sizes)[-40:]
    print(json.dumps(data, indent=2)[:12000])
    return 0 if data.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())

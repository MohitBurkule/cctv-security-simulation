"""Start/stop the headless MetaHuman sandbox editor (project SurvSimMH).

    .venv/bin/python tools/mh_editor.py start|stop|status

A second editor next to the main one (scripts/ue_mcp.sh): its own project copy
(~/data/ue-projects/SurvSimMH), its own job dir (/tmp/survsim_jobs_mh), no MCP
server. Submit jobs with tools/mh_job.py. `start` returns as soon as the runner
writes runner.ready (bounded wait); it never kills anything but its own
process group.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

UE = "/opt/unreal-engine/Engine/Binaries/Linux/UnrealEditor"
PROJECT = Path.home() / "data/ue-projects/SurvSimMH/SurvSimMH.uproject"
JOB_DIR = Path(os.environ.get("SURVSIM_JOB_DIR", "/tmp/survsim_jobs_mh"))
PIDFILE = JOB_DIR / "editor.pid"
LOG = JOB_DIR / "editor.stdout.log"
RUNNER = Path(__file__).resolve().parents[1] / "scripts/ue/mh_job_runner.py"


def running() -> int | None:
    try:
        pid = int(PIDFILE.read_text())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def start(wait: float = 600.0) -> int:
    if (pid := running()):
        print(f"already running (pid {pid})")
        return 0
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    (JOB_DIR / "runner.ready").unlink(missing_ok=True)
    env = dict(os.environ, SURVSIM_JOB_DIR=str(JOB_DIR))
    proc = subprocess.Popen(
        ["xvfb-run", "-a", "-s", "-screen 0 1280x720x24", UE, str(PROJECT),
         f"-ExecCmds=py {RUNNER}", "-unattended", "-nosplash", "-NoSound",
         # 8 GB card shared with the production editor: a MetaHuman preview +
         # street scene hit Vulkan "Failed to allocate Device Memory" at 5.5 GB
         # and starved it. Build/bake/save need no GPU, so -nullrhi by default;
         # renders go to the production runner (docs/12 "MetaHuman route").
         "-dpcvars=r.Streaming.PoolSize=600,r.Nanite.Streaming.StreamingPoolSize=128"]
        + ([] if os.environ.get("MH_GPU") == "1" else ["-nullrhi"]),
        stdout=open(LOG, "w"), stderr=subprocess.STDOUT,  # noqa: SIM115 (outlives us) stdin=subprocess.DEVNULL,
        start_new_session=True, env=env)
    PIDFILE.write_text(str(proc.pid))
    t0 = time.time()
    while time.time() - t0 < wait:
        if (JOB_DIR / "runner.ready").exists():
            print(f"ready in {time.time()-t0:.0f}s (pid {proc.pid})")
            return 0
        if proc.poll() is not None:
            print(f"editor exited rc={proc.returncode}; see {LOG}")
            return 1
        time.sleep(3)
    print("timed out waiting for runner.ready")
    return 1


def stop() -> int:
    pid = running()
    if not pid:
        print("not running")
        return 0
    pgid = os.getpgid(pid)
    os.killpg(pgid, signal.SIGTERM)
    for _ in range(20):                       # a wedged editor ignores SIGTERM
        time.sleep(1)
        try:
            os.killpg(pgid, 0)
        except OSError:
            break
    else:
        os.killpg(pgid, signal.SIGKILL)
    PIDFILE.unlink(missing_ok=True)
    print("stopped")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "start":
        sys.exit(start(float(sys.argv[2]) if len(sys.argv) > 2 else 600.0))
    if cmd == "stop":
        sys.exit(stop())
    print(f"running (pid {running()})" if running() else "not running")

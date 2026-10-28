"""Persistent in-editor job runner: the editor boots once, jobs are files.

Started by `scripts/ue_mcp.sh start` via `-ExecCmds="py <this>"`. A slate
post-tick callback polls JOB_DIR every 0.5 s; each `<name>.py` is exec()'d in
a fresh namespace (with `unreal` and `JOB_NAME` predefined), then renamed to
`<name>.py.done`. `<name>.result.json` is always written: {"ok": true, ...}
plus anything the job put in the `RESULT` dict, or the traceback on failure.

Jobs that need to wait (an MRQ render) must not block the tick; they set
`RESULT["async"] = True` and write their own result file later via the
`finish(result_dict)` callable injected into their namespace.

Submit from the host with `tools/ue_job.py <script.py> [--wait 180]`.
"""
import json
import os
import threading
import time
import traceback

import unreal

JOB_DIR = "/tmp/survsim_jobs"
os.makedirs(JOB_DIR, exist_ok=True)
_state = {"last": 0.0, "busy": False}
BUSY_TIMEOUT_S = 1800.0


def _result_path(name):
    return os.path.join(JOB_DIR, f"{name}.result.json")


def _write(name, payload):
    tmp = _result_path(name) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(tmp, _result_path(name))


def _run(path):
    name = os.path.basename(path)[:-3]
    os.replace(path, path + ".running")
    result = {}
    t0 = time.time()

    def finish(extra=None):
        payload = dict(result)
        payload.update(extra or {})
        payload.setdefault("ok", True)
        payload["seconds"] = round(time.time() - t0, 2)
        _write(name, payload)
        _state["busy"] = False

    def guard(fn):
        """Wrap an async callback: an exception finishes the job as failed
        instead of leaving the runner busy forever."""
        def wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception:                               # noqa: BLE001
                tb = traceback.format_exc()
                unreal.log_error(f"[survsim-job] {name} callback failed\n{tb}")
                for h in list(result.get("_handles", [])):
                    try:
                        unreal.unregister_slate_post_tick_callback(h)
                    except Exception:                       # noqa: BLE001, S110
                        pass
                result.pop("_handles", None)
                finish({"ok": False, "error": tb[-4000:]})
        return wrapped

    ns = {"unreal": unreal, "JOB_NAME": name, "RESULT": result,
          "finish": finish, "guard": guard, "__name__": "__survsim_job__"}
    _state["busy_since"] = time.time()
    try:
        with open(path + ".running") as fh:
            code = compile(fh.read(), path, "exec")
        _state["busy"] = True
        exec(code, ns)                                      # noqa: S102
        if not result.get("async"):
            finish()
    except Exception:                                       # noqa: BLE001
        _state["busy"] = False
        tb = traceback.format_exc()
        unreal.log_error(f"[survsim-job] {name} failed\n{tb}")
        _write(name, {"ok": False, "error": tb[-4000:], **{k: v for k, v in result.items()}})
    os.replace(path + ".running", path + ".done")


def _beat_thread():
    """Heartbeat from a thread: '<now> <pid> <last game-thread tick>'.

    A dead editor stops both; a hung game thread (long job, GPU stall) keeps
    the first field fresh while the third goes stale - the two are no longer
    indistinguishable (tools/ue_job.py reports the tick age).
    """
    while True:
        try:
            with open(os.path.join(JOB_DIR, "heartbeat"), "w") as fh:
                fh.write(f"{time.time()} {os.getpid()} {_state.get('tick', 0.0)}")
        except OSError:
            pass
        time.sleep(5.0)


def _tick(_dt):
    now = time.time()
    _state["tick"] = now
    if _state["busy"] and now - _state.get("busy_since", now) > BUSY_TIMEOUT_S:
        unreal.log_error("[survsim-job] async job exceeded timeout; runner freed")
        _state["busy"] = False
    if now - _state["last"] < 0.5 or _state["busy"]:
        return
    _state["last"] = now
    try:
        jobs = sorted(f for f in os.listdir(JOB_DIR) if f.endswith(".py"))
    except OSError:
        return
    if jobs:
        _run(os.path.join(JOB_DIR, jobs[0]))


_state["handle"] = unreal.register_slate_post_tick_callback(_tick)
threading.Thread(target=_beat_thread, name="survsim-heartbeat", daemon=True).start()
with open(os.path.join(JOB_DIR, "runner.ready"), "w") as fh:
    fh.write(str(time.time()))
unreal.log("[survsim-job] runner polling " + JOB_DIR)

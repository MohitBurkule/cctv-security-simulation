"""Turn an in-editor job runner result file into pass/fail.

scripts/ue/job_runner.py writes ``<name>.result.json``: ``{"ok": true, ...}`` or
``{"ok": false, "error": <traceback>}``. The editor itself prints "Python script
executed successfully" for scripts that did nothing, so this file is the only
evidence a job ran. Missing/unparseable/``ok`` absent are failures, not passes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .core import FAIL, PASS, WARN, Check


def check_job_result(data: dict[str, Any] | None, name: str = "job") -> list[Check]:
    if data is None:
        return [Check(f"{name}:result", FAIL, detail="result file missing or not JSON")]
    out: list[Check] = []
    okv = data.get("ok")
    if okv is True:
        out.append(Check(f"{name}:ok", PASS, True, True))
    else:
        tb = str(data.get("error") or data.get("traceback") or "")
        last = tb.strip().splitlines()[-1] if tb.strip() else "no traceback recorded"
        out.append(Check(f"{name}:ok", FAIL, okv, True, last))
    if data.get("async") is True and "seconds" not in data:
        out.append(Check(f"{name}:async", WARN, detail="async job: final result not yet written"))
    errs = {k: v for k, v in data.items() if k.startswith("err") and k != "error" and v}
    if errs:
        out.append(Check(f"{name}:soft_errors", WARN, len(errs), 0,
                         "; ".join(f"{k}={str(v)[:80]}" for k, v in list(errs.items())[:3])))
    return out


def load(path: Path) -> dict[str, Any] | None:
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def traceback_of(data: dict[str, Any] | None) -> str:
    return str((data or {}).get("error") or (data or {}).get("traceback") or "")

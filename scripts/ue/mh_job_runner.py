"""Job runner for the MetaHuman sandbox editor (project SurvSimMH).

Identical to scripts/ue/job_runner.py except the job directory, which comes
from SURVSIM_JOB_DIR (default /tmp/survsim_jobs_mh) so a second editor can run
beside the main one without the two stealing each other's jobs. Reuses the
original source rather than forking it: the JOB_DIR assignment is substituted
and the result exec()'d; the assertion fails loudly if that line ever changes.
"""
import os

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "job_runner.py")
_DIR = os.environ.get("SURVSIM_JOB_DIR", "/tmp/survsim_jobs_mh")
with open(_SRC) as _fh:
    _code = _fh.read()
_OLD = 'JOB_DIR = "/tmp/survsim_jobs"'
assert _OLD in _code, "job_runner.py JOB_DIR line changed"
exec(compile(_code.replace(_OLD, f"JOB_DIR = {_DIR!r}"), _SRC, "exec"))  # noqa: S102

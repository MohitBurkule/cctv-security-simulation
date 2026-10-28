"""Fast automated gates that catch known render / job failure classes by code.

Each check is a small pure function returning a ``Check``. ``render.run_all``
wires them to a render directory; ``scripts/check_render.py`` is the CLI.
docs/12 "Automated render gate" lists which past failure each check guards.
"""

from .core import Check, format_table, worst
from .job import check_job_result

__all__ = ["Check", "check_job_result", "format_table", "worst"]

"""Check result type and table formatting."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

PASS, FAIL, WARN, INFO, SKIP = "PASS", "FAIL", "WARN", "INFO", "SKIP"


@dataclass
class Check:
    name: str
    status: str            # PASS | FAIL | WARN | INFO | SKIP
    value: Any = None
    threshold: Any = None
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.status == FAIL

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def ok(cond: bool) -> str:
    return PASS if cond else FAIL


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3g}"
    return "" if v is None else str(v)


def format_table(checks: Iterable[Check]) -> str:
    rows = [(c.name, c.status, _fmt(c.value), _fmt(c.threshold), c.detail) for c in checks]
    head = ("check", "status", "value", "threshold", "detail")
    w = [max(len(r[i]) for r in [head, *rows]) for i in range(4)]
    line = lambda r: "  ".join(r[i].ljust(w[i]) for i in range(4)) + "  " + r[4]
    return "\n".join([line(head), "-" * (sum(w) + 16), *map(line, rows)])


def worst(checks: Iterable[Check]) -> str:
    return FAIL if any(c.failed for c in checks) else PASS

"""Build docs/research/offline_snippets.pdf -- the reference snippets to be
able to type from memory if internet is cut during the coding round
("If internet is cut").

Every snippet is extracted from docs/research/drills/drills.py by function/
class/method *name* via inspect.getsource -- never retyped -- so the PDF
cannot drift from the code tests/test_coding_drills.py exercises. The one
exception is the argparse/pytest skeleton, which isn't in drills.py and is
copied verbatim from `93` (b) point 10.

Run: python tools/offline_snippets.py
"""
from __future__ import annotations

import inspect
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
DRILLS_DIR = ROOT / "docs" / "research" / "drills"
sys.path.insert(0, str(DRILLS_DIR))
import drills

OUT = ROOT / "docs" / "research" / "offline_snippets.pdf"
MAX_PAGES = 3


def src(name: str) -> str:
    """Verbatim source of a top-level function/class in drills.py."""
    return inspect.getsource(getattr(drills, name)).rstrip()


def lines_of(*names: str, drop_docstring: bool = False) -> str:
    """Verbatim source of one or more names, stacked with one blank line
    between them. drop_docstring removes a function's own one-line
    docstring (comment-only, not logic) when a snippet must fit the page."""
    blocks = []
    for n in names:
        s = src(n)
        if drop_docstring:
            out = []
            in_doc = False
            for ln in s.splitlines():
                stripped = ln.strip()
                if in_doc:
                    if stripped.endswith('"""'):
                        in_doc = False
                    continue  # comment-only docstring line, not logic
                if stripped.startswith('"""'):
                    if stripped.endswith('"""') and len(stripped) > 3:
                        continue  # single-line docstring
                    in_doc = True
                    continue
                out.append(ln)
            s = "\n".join(out)
        blocks.append(s)
    return "\n\n".join(blocks)


# --------------------------------------------------------------- snippets
# (title, verbatim source, one-line "test to state aloud")
SNIPPETS = [
    (
        "1. IoU (+ vectorised matrix)",
        lines_of("iou", "iou_matrix"),
        "union-zero guard returns 0.0, not NaN; matrix broadcasts (N,1,2)/(1,M,2).",
    ),
    (
        "2. NMS",
        src("nms"),
        "sort by score descending; only IoU strictly over the threshold is suppressed.",
    ),
    (
        "3. Greedy IoU matching with gate",
        src("greedy_assign"),
        "sorts the whole cost matrix once; a pair above max_cost stops the loop, not just that pair.",
    ),
    (
        "4. Kalman constant-velocity predict/update (box)",
        inspect.getsource(drills.BoxKF.predict).rstrip()
        + "\n\n"
        + inspect.getsource(drills.BoxKF.update).rstrip(),
        "state is (cx, cy, w/h, h, and their 4 velocities); noise scales with box height h.",
    ),
    (
        "5. Two-stage IoU tracker (birth/death)",
        lines_of("track", drop_docstring=True),
        "birth only from confident (high-score) unmatched detections; death is age > max_age.",
    ),
    (
        "6. MOTA + IDSW, and IDF1 via Hungarian",
        lines_of("mota_idf1", drop_docstring=True),
        "MOTA = 1-(FN+FP+IDSW)/N_gt (no-GT case is -FP, not 0/0); IDF1 is one global Hungarian match on accumulated IoU>=0.5 pair counts, not a per-frame match.",
    ),
    (
        "7. Dwell rule with hysteresis",
        src("loiter_alarms"),
        "dwell keeps counting through an exit shorter than t_off; fires once per episode, re-arms only after a real exit.",
    ),
    (
        "8. Percentile bootstrap CI + exact paired sign-flip p",
        lines_of("bootstrap_ci", "signflip_p"),
        "bootstrap CI is percentile (not BCa) and too narrow at small n; sign-flip enumerates all 2^n sign patterns exactly.",
    ),
    (
        "9. DLT homography with Hartley normalisation",
        lines_of("_hartley", "dlt_homography"),
        "normalise both point sets before building the design matrix; take the last right-singular vector, then denormalise Td^-1 Hn Ts.",
    ),
]

ARGPARSE_PYTEST_SKELETON = '''import argparse
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    ...
if __name__ == "__main__":
    main()

import pytest
def test_empty_input():
    assert f([]) == []
def test_happy_path():
    assert f([...]) == [...]'''

CHECK_SH = '''#!/bin/sh
set -e
python -m pytest -q
python -m pyflakes . 2>/dev/null || python -m ruff check .'''

SNIPPETS.append((
    "10. argparse + pytest skeleton, and check.sh",
    ARGPARSE_PYTEST_SKELETON + "\n\n" + CHECK_SH,
    "activate the venv first; run pytest -q (not -n auto) in an unfamiliar sandbox.",
))

for _title, _code, _ in SNIPPETS:
    n = len(_code.splitlines())
    if n > 32:
        raise SystemExit(f"{_title}: {n} lines, too long for a pocket page")

# --------------------------------------------------------------------- draw
PAGE_W, PAGE_H = 8.5, 11.0
INK, ACC, MUTE = "#1a1a1a", "#1f5fa8", "#555555"
MONO = "DejaVu Sans Mono"


def new_page(page_no: int):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.text(0.05, 0.975, "Offline reference snippets", fontsize=15, weight="bold", color=INK)
    fig.text(0.95, 0.975, f"page {page_no} of {MAX_PAGES}", fontsize=10, color=MUTE, ha="right")
    if page_no == 1:
        fig.text(0.05, 0.955, "docs/research/drills/drills.py, verbatim -- 93 (b) \"if internet is cut\"",
                  fontsize=9.5, color=MUTE)
    fig.subplots_adjust(0, 0, 1, 1)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    return fig, ax


LINE_H = 0.0113
GAP = 0.011
Y_TOP = 0.93
Y_BOT = 0.02


class Pager:
    """Flows text line-by-line, breaking to a new page when needed -- so a
    long snippet's tail fills the next page's top instead of leaving the
    previous page's bottom empty."""

    def __init__(self):
        self.pages = []
        self.fig = self.ax = None
        self.y = -1.0
        self._new_page()

    def _new_page(self):
        fig, ax = new_page(len(self.pages) + 1)
        self.pages.append(fig)
        self.fig, self.ax = fig, ax
        self.y = Y_TOP

    def ensure(self, need: float):
        if self.y - need < Y_BOT:
            self._new_page()

    def line(self, text, **kw):
        self.ax.text(kw.pop("x", 0.055), self.y, text, transform=self.ax.transAxes, **kw)
        self.y -= LINE_H

    def snippet(self, title, code, test_line):
        code_lines = [w for ln in code.splitlines()
                      for w in (textwrap.wrap(ln, 96, replace_whitespace=False,
                                               drop_whitespace=False) or [""])]
        test_lines = textwrap.wrap("Test to state aloud: " + test_line, 112)
        self.ensure(0.017 + LINE_H * min(2, len(code_lines)))  # keep title with >=2 code lines
        self.line(title, fontsize=12, weight="bold", color=ACC, x=0.05)
        self.y -= 0.017 - LINE_H
        for w in code_lines:
            self.ensure(LINE_H)
            self.line(w, fontsize=8.4, family=MONO, color=INK)
        self.y -= 0.002
        for w in test_lines:
            self.ensure(LINE_H)
            self.line(w, fontsize=8.4, family="sans-serif", style="italic", color=MUTE)
        self.y -= GAP


pager = Pager()
for title, code, test_line in SNIPPETS:
    pager.snippet(title, code, test_line)

# fix up the page-number footer now that the final count is known
for i, fig in enumerate(pager.pages, 1):
    for t in fig.texts:
        if t.get_text().startswith("page "):
            t.set_text(f"page {i} of {len(pager.pages)}")

if len(pager.pages) > MAX_PAGES:
    raise SystemExit(f"{len(pager.pages)} pages, over the {MAX_PAGES}-page budget")

with PdfPages(OUT) as pdf:
    for fig in pager.pages:
        pdf.savefig(fig)
        plt.close(fig)

print(f"wrote {OUT} ({len(pager.pages)} pages, {len(SNIPPETS)} snippets)")

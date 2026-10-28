#!/usr/bin/env python3
"""Compute research-corpus stats used in docs (00, README_START_HERE, CLAUDE.md, 11).

Run `python tools/corpus_stats.py` after adding/removing research docs, audit
files, experiment configs/runners, ledger rows, or figures, and paste the
printed numbers into those files. See docs/research/63 for the note.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "docs" / "research"


def word_count(paths):
    total = 0
    for p in paths:
        total += len(p.read_text(errors="ignore").split())
    return total


def main() -> None:
    md_files = sorted(RESEARCH.glob("*.md"))

    numbered = [p for p in md_files if re.match(r"^\d+_", p.name)]
    report_files = [p for p in numbered if not p.name.startswith("00_")]
    dossiers = [p for p in md_files if re.match(r"^[RDH]\d+", p.name)]
    audit_files = [p for p in md_files if p.name.startswith("AUDIT")]
    total_words = word_count(md_files)
    report_nums = sorted(int(re.match(r"^(\d+)_", p.name).group(1)) for p in report_files)

    experiments_dir = ROOT / "survsim" / "experiments"
    configs = list((experiments_dir / "configs").glob("*.yaml"))

    runners_init = (experiments_dir / "__init__.py").read_text()
    m = re.search(r"RUNNERS\s*=\s*\{(.*?)\}\s*\n", runners_init, re.DOTALL)
    runners = re.findall(r'"[a-z0-9_]+"\s*:', m.group(1)) if m else []

    log32 = RESEARCH / "32_EXPERIMENTS_LOG.md"
    h_sections = re.findall(r"^##\s+H\d+[a-z]*\b", log32.read_text(), re.MULTILINE) if log32.exists() else []

    ledger = RESEARCH / "AUDIT_LEDGER.md"
    ledger_rows = 0
    if ledger.exists():
        rows = [l for l in ledger.read_text().splitlines() if l.startswith("|")]
        ledger_rows = max(0, len(rows) - 2)  # minus header + separator

    retired = RESEARCH / "RETIRED_CLAIMS.md"
    registry_rows = 0
    if retired.exists():
        rows = [l for l in retired.read_text().splitlines() if l.startswith("|")]
        registry_rows = max(0, len(rows) - 2)

    figpack = RESEARCH / "80_FIGURE_PACK.md"
    figures = re.findall(r"^##\s+\d+\s*·", figpack.read_text(), re.MULTILINE) if figpack.exists() else []

    print(f"dossiers (R/D/H):             {len(dossiers)}")
    print(f"report files ({report_nums[0]}-{report_nums[-1]}):        {len(report_files)}")
    print(f"AUDIT_*.md files:            {len(audit_files)}")
    print(f"total *.md files in corpus:  {len(md_files)}")
    print(f"total words (docs/research): {total_words:,} (~{total_words // 1000}k)")
    print(f"experiment configs:          {len(configs)}")
    print(f"registered runners:          {len(runners)}")
    print(f"H-sections in 32 (pre-reg'd): {len(h_sections)}")
    print(f"AUDIT_LEDGER.md rows:        {ledger_rows}")
    print(f"RETIRED_CLAIMS.md rows:      {registry_rows}")
    print(f"figures (80_FIGURE_PACK.md): {len(figures)}")
    print()
    print(
        f"one-line: ~{total_words // 1000}k words, {len(dossiers)} dossiers (R/D/H), "
        f"{len(report_files)} report files ({report_nums[0]}-{report_nums[-1]}), "
        f"{len(audit_files)} AUDIT files, {len(configs)} configs / {len(runners)} runners / "
        f"{len(h_sections)} H-sections, {ledger_rows} ledger rows, {registry_rows} registry rows, "
        f"{len(figures)} figures"
    )


if __name__ == "__main__":
    main()

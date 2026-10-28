#!/usr/bin/env python3
"""Fast regex secret scan of staged files (pre-commit). Exit 1 on any finding.

Usage: secret_scan.py [--allowlist FILE] [paths...]   (no paths: git staged files)
Allowlist (scripts/hooks/secret_allowlist.txt): one entry per line, either a
path glob (whole file exempt) or `path:line` (one line exempt). `#` comments.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from pathlib import Path

PATTERNS = {
    "AWS access key id": re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
    "AWS secret access key": re.compile(
        r"aws_secret_access_key\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{40}", re.IGNORECASE
    ),
    "GCP API key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "GCP service-account key": re.compile(r"\"type\"\s*:\s*\"service_account\""),
    "GitHub token": re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,})\b"),
    "private key header": re.compile(r"-----BEGIN ([A-Z]+ )?PRIVATE KEY-----"),
}
BAD_NAMES = {
    "*.salt": "per-site salt file",
    ".env": "dotenv file",
    ".env.*": "dotenv file",
    "*.pem": "PEM key file",
    "id_rsa*": "SSH key",
    "id_ed25519*": "SSH key",
}
DEFAULT_ALLOW = Path(__file__).with_name("secret_allowlist.txt")


def load_allowlist(path: Path) -> list[str]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def _allowed(path: str, lineno: int | None, allow: list[str]) -> bool:
    for a in allow:
        if ":" in a and a.rsplit(":", 1)[1].isdigit():
            p, n = a.rsplit(":", 1)
            if lineno is not None and fnmatch.fnmatch(path, p) and int(n) == lineno:
                return True
        elif fnmatch.fnmatch(path, a):
            return True
    return False


def scan(paths: list[str], allow: list[str], read=None) -> list[str]:
    findings = []
    for p in paths:
        name = Path(p).name
        for glob, why in BAD_NAMES.items():
            if (
                fnmatch.fnmatch(name, glob)
                and name != ".env.example"
                and not _allowed(p, None, allow)
            ):
                findings.append(f"{p}: {why} must not be committed")
        try:
            text = read(p) if read else Path(p).read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for label, rx in PATTERNS.items():
                if rx.search(line) and not _allowed(p, i, allow):
                    findings.append(f"{p}:{i}: possible {label}")
    return findings


def _staged() -> tuple[list[str], object]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM", "-z"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    paths = [p for p in out.split("\0") if p]

    def read(p: str) -> str:  # scan the staged blob, not the working tree
        return subprocess.run(
            ["git", "show", f":{p}"], capture_output=True, check=True
        ).stdout.decode("utf-8", errors="ignore")

    return paths, read


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOW)
    ap.add_argument("paths", nargs="*")
    a = ap.parse_args(argv)
    paths, read = (a.paths, None) if a.paths else _staged()
    findings = scan(paths, load_allowlist(a.allowlist), read)
    if findings:
        print("[pre-commit] SECRET SCAN BLOCKED THE COMMIT:", file=sys.stderr)
        for f in findings:
            print("  " + f, file=sys.stderr)
        print(
            "  If this is a false positive, add `path` or `path:line` to "
            "scripts/hooks/secret_allowlist.txt. If real: unstage it and rotate the secret.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

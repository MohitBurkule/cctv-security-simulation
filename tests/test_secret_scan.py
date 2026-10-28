"""Pre-commit secret scanner (87 gap: no secret scanner in the hook)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests/fixtures/secret_scan"
_spec = importlib.util.spec_from_file_location("secret_scan", ROOT / "scripts/hooks/secret_scan.py")
ss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ss)


def test_fixture_flags_each_kind():
    f = ss.scan([str(FIX / "leaky.txt")], [])
    text = "\n".join(f)
    for label in ("AWS access key id", "GitHub token", "private key header", "GCP API key"):
        assert label in text
    assert len(f) == 4


def test_clean_file_passes():
    assert ss.scan([str(FIX / "clean.txt")], []) == []


def test_bad_filenames_blocked(tmp_path):
    for n in ("site.salt", ".env", ".env.prod", "id_rsa"):
        (tmp_path / n).write_text("x\n")
    f = ss.scan([str(tmp_path / n) for n in ("site.salt", ".env", ".env.prod", "id_rsa")], [])
    assert len(f) == 4


def test_allowlist_file_and_line():
    p = str(FIX / "leaky.txt")
    assert ss.scan([p], [p]) == []
    assert len(ss.scan([p], [f"{p}:2"])) == 3


def test_main_exit_code_and_message(capsys, tmp_path):
    empty = tmp_path / "allow.txt"
    empty.write_text("# none\n")
    assert ss.main(["--allowlist", str(empty), str(FIX / "leaky.txt")]) == 1
    assert "SECRET SCAN BLOCKED" in capsys.readouterr().err
    assert ss.main(["--allowlist", str(empty), str(FIX / "clean.txt")]) == 0


def test_repo_allowlist_exempts_fixtures():
    allow = ss.load_allowlist(ROOT / "scripts/hooks/secret_allowlist.txt")
    assert (
        ss.scan(
            ["tests/fixtures/secret_scan/leaky.txt"], allow, read=lambda p: (ROOT / p).read_text()
        )
        == []
    )

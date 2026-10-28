"""`!include` must stay inside the config tree (87 gap: path traversal)."""

from pathlib import Path

import pytest

from survsim.core.loader import IncludeError, load_yaml


def _tree(tmp_path: Path) -> Path:
    cfg = tmp_path / "configs"
    (cfg / "scenarios").mkdir(parents=True)
    (cfg / "people").mkdir()
    (cfg / "people" / "a.yaml").write_text("name: a\n")
    (tmp_path / "secret.yaml").write_text("k: leaked\n")
    return cfg


def test_sibling_include_works(tmp_path):
    cfg = _tree(tmp_path)
    f = cfg / "scenarios" / "s.yaml"
    f.write_text("p: !include ../people/a.yaml\n")
    assert load_yaml(f) == {"p": {"name": "a"}}


def test_traversal_refused(tmp_path):
    cfg = _tree(tmp_path)
    f = cfg / "scenarios" / "s.yaml"
    f.write_text("p: !include ../../secret.yaml\n")
    with pytest.raises(IncludeError, match="outside the config root"):
        load_yaml(f)


def test_deep_traversal_refused(tmp_path):
    cfg = _tree(tmp_path)
    f = cfg / "scenarios" / "s.yaml"
    f.write_text("p: !include ../../../../../../etc/passwd\n")
    with pytest.raises(IncludeError):
        load_yaml(f)


def test_absolute_refused(tmp_path):
    cfg = _tree(tmp_path)
    f = cfg / "scenarios" / "s.yaml"
    f.write_text(f"p: !include {tmp_path / 'secret.yaml'}\n")
    with pytest.raises(IncludeError, match="absolute"):
        load_yaml(f)


def test_nested_include_cannot_escape(tmp_path):
    cfg = _tree(tmp_path)
    (cfg / "people" / "b.yaml").write_text("x: !include ../../secret.yaml\n")
    f = cfg / "scenarios" / "s.yaml"
    f.write_text("p: !include ../people/b.yaml\n")
    with pytest.raises(IncludeError):
        load_yaml(f)


def test_symlink_escape_refused(tmp_path):
    cfg = _tree(tmp_path)
    (cfg / "people" / "link.yaml").symlink_to(tmp_path / "secret.yaml")
    f = cfg / "scenarios" / "s.yaml"
    f.write_text("p: !include ../people/link.yaml\n")
    with pytest.raises(IncludeError):
        load_yaml(f)


def test_no_configs_ancestor_confines_to_file_dir(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    (d / "ok.yaml").write_text("v: 1\n")
    f = d / "s.yaml"
    f.write_text("a: !include ok.yaml\nb: !include ../secret.yaml\n")
    (tmp_path / "secret.yaml").write_text("k: 1\n")
    with pytest.raises(IncludeError):
        load_yaml(f)


def test_repo_configs_all_load():
    root = Path(__file__).resolve().parents[1] / "configs"
    for f in (root / "scenarios").glob("*.yaml"):
        load_yaml(f)

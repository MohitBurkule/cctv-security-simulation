"""Static checks on the in-editor scripts, which cannot run under pytest.

These scripts import `unreal`, so they are only executable inside a 20 GB
editor. That makes cheap static checks disproportionately valuable: they are the
only automated feedback this code gets.
"""

import ast
from pathlib import Path

import pytest

UE_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "ue"

# Calls whose positional argument order is a known trap.
#
# unreal.Rotator's positional order in Python is (roll, pitch, yaw), NOT the
# (pitch, yaw, roll) of the C++ constructor and the editor UI.
POSITIONAL_TRAPS = {"Rotator"}


def _script_paths():
    return sorted(UE_SCRIPTS.glob("*.py"))


def _positional_trap_calls(source: str):
    """Real calls only — found by parsing, not by matching text.

    A regex over lines cannot tell a call from prose, and flagged the docstring
    that *documents* this very bug. Parsing means the scripts can explain the
    trap in their own words while genuine offenders are still caught.
    """
    offenders = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in POSITIONAL_TRAPS:
            continue
        if isinstance(func.value, ast.Name) and func.value.id != "unreal":
            continue
        if node.args:                       # any positional argument at all
            offenders.append((func.attr, node.lineno))
    return offenders


@pytest.mark.parametrize("path", _script_paths(), ids=lambda p: p.name)
def test_rotators_are_built_with_keywords(path):
    """unreal.Rotator's positional order is (roll, pitch, yaw).

    Passing (pitch, yaw, roll) positionally aimed the camera at the zenith and
    put the sun below the horizon; every render came out exactly black while
    every layer reported success. A full day went into blaming the renderer.
    See docs/BUG_black_render.md.
    """
    source = path.read_text()
    # The templated renderer carries placeholders that are not valid Python
    # until they are substituted; check the substituted form.
    source = (source.replace("@MANIFEST@", "/tmp/m.json")
                    .replace("@OUTPUT@", "/tmp/out")
                    .replace("@MAXFRAMES@", "100"))

    offenders = _positional_trap_calls(source)
    assert not offenders, (
        f"{path.name} builds a rotator positionally at line(s) "
        f"{[line for _, line in offenders]} — use pitch=/yaw=/roll=")


def test_the_check_would_actually_catch_a_positional_rotator():
    """A guard that cannot fail is worse than no guard.

    The previous regex version passed happily while matching prose, so this
    pins that the detector fires on a real call and stays quiet on text that
    merely mentions one.
    """
    assert _positional_trap_calls("unreal.Rotator(-20, 90, 0)")
    assert not _positional_trap_calls("unreal.Rotator(pitch=-20, yaw=90, roll=0)")
    assert not _positional_trap_calls('"""docs mention unreal.Rotator(a, b, c) here"""')
    assert not _positional_trap_calls("# unreal.Rotator(1, 2, 3) in a comment")


def test_every_in_editor_script_parses():
    """A syntax error here is only discoverable by launching the editor."""
    for path in _script_paths():
        source = (path.read_text().replace("@MANIFEST@", "/tmp/m.json")
                  .replace("@OUTPUT@", "/tmp/out").replace("@MAXFRAMES@", "100"))
        ast.parse(source)

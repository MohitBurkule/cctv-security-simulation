"""YAML loading with `!include`, so rigs, environments and people stay separate files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import Camera, Scenario


class IncludeError(ValueError):
    """An `!include` resolved outside the allowed config tree."""


def _confine_root(path: Path) -> Path:
    """Nearest ancestor directory named `configs`, else the file's own directory."""
    path = path.resolve()
    for parent in path.parents:
        if parent.name == "configs":
            return parent
    return path.parent


class _IncludeLoader(yaml.SafeLoader):
    """SafeLoader that resolves `!include relative/path.yaml` against the file's own dir.

    Includes are confined: absolute paths are refused, and a relative path must
    resolve (symlinks followed) inside the configs root of the top-level file.
    """

    _confine: Path | None = None

    def __init__(self, stream) -> None:  # type: ignore[no-untyped-def]
        self._root = Path(getattr(stream, "name", ".")).resolve().parent
        super().__init__(stream)


def _include(loader: _IncludeLoader, node: yaml.Node) -> Any:
    raw = str(loader.construct_scalar(node))
    confine = loader._confine or _confine_root(loader._root / "_")
    if Path(raw).is_absolute():
        raise IncludeError(f"!include {raw!r}: absolute paths are not allowed")
    path = (loader._root / raw).resolve()
    if not path.is_relative_to(confine):
        raise IncludeError(
            f"!include {raw!r} resolves to {path}, outside the config root {confine}"
        )
    with path.open() as fh:
        sub = _IncludeLoader(fh)
        sub._confine = confine
        try:
            return sub.get_single_data()
        finally:
            sub.dispose()


_IncludeLoader.add_constructor("!include", _include)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as fh:
        loader = _IncludeLoader(fh)
        loader._confine = _confine_root(Path(path))
        try:
            return loader.get_single_data()
        finally:
            loader.dispose()


def load_scenario(path: str | Path, seed: int | None = None) -> Scenario:
    """Load and validate a scenario file.

    A camera rig file may be either a bare list or a mapping with a `cameras:` key
    (the latter also carries `rig_tools:`), so both shapes are accepted here rather
    than forcing every rig file into one form.
    """
    raw = load_yaml(path)

    cams = raw.get("cameras")
    if isinstance(cams, dict):
        raw["cameras"] = cams.get("cameras", [])
        raw.setdefault("_rig_tools", cams.get("rig_tools"))
    raw.pop("_rig_tools", None)

    if seed is not None:
        raw["seed"] = seed

    scenario = Scenario.model_validate(raw)
    _validate_references(scenario)
    return scenario


def _validate_references(s: Scenario) -> None:
    """Catch the config errors that would otherwise surface as a crash mid-render."""
    known = (
        {z.id for z in s.environment.zones}
        | {w.id for w in s.environment.spawn_points}
        | {w.id for w in s.environment.exits}
        | {r.id for r in s.environment.routes}
        | {p.id for p in s.environment.props}
    )
    cam_ids = {c.id for c in s.cameras}
    problems: list[str] = []

    for person in s.population.hero:
        for i, step in enumerate(person.intent):
            extra = step.model_extra or {}
            for key in ("zone", "from", "to", "in_zone", "segment", "asset", "route"):
                ref = extra.get(key)
                if isinstance(ref, str) and ref not in known and "." in ref:
                    problems.append(f"{person.id}.intent[{i}].{key} -> unknown {ref!r}")
            cam = extra.get("camera")
            if isinstance(cam, str) and cam not in cam_ids:
                problems.append(f"{person.id}.intent[{i}].camera -> unknown {cam!r}")

    if problems:
        raise ValueError("scenario references unknown objects:\n  " + "\n  ".join(problems))


def resolve_cameras(rig_path: str | Path) -> list[Camera]:
    raw = load_yaml(rig_path)
    cams = raw.get("cameras", raw) if isinstance(raw, dict) else raw
    return [Camera.model_validate(c) for c in cams]

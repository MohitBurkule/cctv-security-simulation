"""Every YAML under configs/ must parse, and every scenario must validate.

Parametrised per file so a future config break names the offending file
instead of failing one generic test.
"""

from pathlib import Path

import pytest

from survsim.core.loader import load_scenario, load_yaml

CONFIGS_ROOT = Path(__file__).resolve().parents[1] / "configs"

ALL_YAML = sorted(CONFIGS_ROOT.rglob("*.yaml"))
SCENARIOS = sorted((CONFIGS_ROOT / "scenarios").glob("*.yaml"))


@pytest.mark.parametrize("path", ALL_YAML, ids=[str(p.relative_to(CONFIGS_ROOT)) for p in ALL_YAML])
def test_yaml_parses(path):
    # load_yaml resolves `!include`, which plain yaml.safe_load cannot.
    load_yaml(path)


@pytest.mark.parametrize("path", SCENARIOS, ids=[p.stem for p in SCENARIOS])
def test_scenario_validates(path):
    load_scenario(path)

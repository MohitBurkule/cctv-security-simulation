"""Schema check for the CC0 asset index (no network, no asset files)."""
import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fetch_cc0", ROOT / "scripts/assets/fetch_cc0.py")
fetch_cc0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_cc0)
FIXTURE = json.loads((ROOT / "tests/fixtures/asset_index_small.json").read_text())


def test_fixture_is_valid():
    assert fetch_cc0.validate_index(FIXTURE) == []


def _broken(i, **kw):
    d = copy.deepcopy(FIXTURE)
    for k, v in kw.items():
        if v is None:
            d["assets"][i].pop(k)
        else:
            d["assets"][i][k] = v
    return fetch_cc0.validate_index(d)


def test_character_needs_walk_clip_and_height():
    assert _broken(0, anim_clips=["Idle"])
    assert _broken(0, height_m=None)
    assert _broken(0, height_m=175.0)  # centimetres, not metres


def test_other_kinds_need_their_size_fields():
    assert _broken(1, size_m=[1, 2])
    assert _broken(3, resolution=None)
    assert _broken(4, kind="skybox")


def test_licence_must_be_cc0():
    assert _broken(2, license="CC-BY-4.0")
    assert _broken(2, license_url=None)


def test_duplicate_names_rejected():
    d = copy.deepcopy(FIXTURE)
    d["assets"].append(copy.deepcopy(d["assets"][2]))
    assert fetch_cc0.validate_index(d)

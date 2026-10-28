"""Appearance epochs reach the render: manifest epochs, per-epoch palettes, actor split."""
import json
from pathlib import Path

import pytest

from survsim.export.anim_plan import person_plan, split_epochs
from survsim.export.outfits import epoch_outfits, outfit, srgb_hex_to_linear
from survsim.export.unreal_scene import appearance_epochs

RUN = Path(__file__).resolve().parent.parent / "out/runs_v2/disguise_and_tamper"


def _synthetic():
    ids = [{"uid": "a", "epoch": 0, "t_from": 0.0, "t_to": 11.5, "face_visible": True,
            "outfit": json.dumps({"outer": "rain_jacket", "headwear": None,
                                  "colours": {"top": "#2b2b2b", "outer": "#0b3d91"}})},
           {"uid": "a", "epoch": 1, "t_from": 11.5, "t_to": 20.0, "face_visible": False,
            "outfit": json.dumps({"outer": None, "headwear": "cap",
                                  "colours": {"top": "#d8d8d0"}})},
           {"uid": "b", "epoch": 0, "t_from": 0.0, "t_to": 20.0, "face_visible": True}]
    world = [{"frame": f, "uid": "a", "appearance_epoch": int(f >= 116)} for f in range(1, 200)]
    return ids, world


def test_epoch_frame_comes_from_world_rows():
    ids, world = _synthetic()
    eps = appearance_epochs(ids, world, 10.0)
    assert list(eps) == ["a"]                      # b has one epoch: not listed
    assert [e["frame_start"] for e in eps["a"]] == [0, 116]
    assert eps["a"][1]["face_visible"] is False and eps["a"][1]["gt_outfit"]["headwear"] == "cap"
    # no world rows: fall back to t_from
    assert appearance_epochs(ids, [], 10.0)["a"][1]["frame_source"] == "identities.t_from"


def test_epoch_palettes_change_and_cover_face():
    ids, world = _synthetic()
    eps = appearance_epochs(ids, world, 10.0)["a"]
    base = outfit("a", 0)
    pals = epoch_outfits("a", eps, base, 0)
    assert pals[0]["top_rgb"] == base["top_rgb"] and "face_cover" not in pals[0]
    assert pals[1]["top_rgb"] == srgb_hex_to_linear("#d8d8d0")
    assert pals[1]["face_cover"]["kind"] == "cap" and pals[1]["frame_start"] == 116


def test_split_epochs_swaps_actor_at_the_frame():
    ids, world = _synthetic()
    eps = appearance_epochs(ids, world, 10.0)["a"]
    man = {"fps": 10, "seed": 0, "tracks": [
        {"uid": "a", "keys": [{"frame": f, "location": [f * 10.0, 0, 0]} for f in range(1, 200)]}]}
    plan = person_plan(man, 200)
    pals = epoch_outfits("a", eps, outfit("a", 0), 0)
    out = split_epochs(plan, {"a": pals})
    assert [e["uid"] for e in out] == ["a", "a#e1"]
    assert out[0]["visible"] == [1, 115] and out[1]["visible"] == [116, 199]
    assert out[0]["segments"] == out[1]["segments"] and out[1]["person_uid"] == "a"
    # a window that ends before the change renders only epoch 0
    assert [e["uid"] for e in split_epochs(person_plan(man, 100), {"a": pals})] == ["a"]


@pytest.mark.skipif(not (RUN / "gt/identities.parquet").is_file(), reason="run not present")
def test_disguise_and_tamper_intruder_a_epoch_frame_matches_gt():
    pd = pytest.importorskip("pandas")
    ids = pd.read_parquet(RUN / "gt/identities.parquet").to_dict("records")
    world = pd.read_parquet(RUN / "gt/world.parquet",
                            columns=["frame", "uid", "appearance_epoch"]).to_dict("records")
    fps = float(json.loads((RUN / "unreal/manifest.json").read_text())["fps"])
    eps = appearance_epochs(ids, world, fps)["intruder_a"]
    assert len(eps) >= 2
    change = eps[1]
    # the first frame the world records epoch 1, and within one frame of t_from
    assert abs(change["frame_start"] / fps - change["t_from"]) <= 1.0 / fps + 1e-9
    events = [json.loads(line) for line in (RUN / "gt/events.jsonl").read_text().splitlines()
              if line.strip()]
    ev = [e for e in events if e["label"] == "suspicious.appearance.change"
          and "intruder_a" in e["actors"]]
    assert ev and any(e["t_start"] <= change["frame_start"] / fps <= e["t_end"] for e in ev)
    assert change["face_visible"] is False

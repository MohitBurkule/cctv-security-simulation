import shutil

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _fake_render(root, cams=("cam_01", "cam_02"), n=6):
    rng = np.random.default_rng(0)
    for c in cams:
        d = root / "render" / c
        d.mkdir(parents=True)
        for i in range(n):
            img = rng.integers(0, 255, (180, 320, 3), dtype=np.uint8)
            cv2.imwrite(str(d / f"{i:04d}.png"), img)


@pytest.mark.slow
def test_video_writes_one_mp4_per_camera_with_all_frames(tmp_path):
    from survsim.video import count_frames, make_videos

    _fake_render(tmp_path)
    real = tmp_path / "real"; real.mkdir()
    cv2.imwrite(str(real / "a.jpg"), np.full((200, 300, 3), 90, np.uint8))
    cv2.imwrite(str(real / "b.jpg"), np.full((200, 300, 3), 120, np.uint8))
    res = make_videos(tmp_path, tmp_path / "out", "cctv_day_720p", real_dir=real)
    assert set(res["videos"]) == {"cam_01", "cam_02"}
    for mp4 in res["videos"].values():
        assert mp4.exists() and count_frames(mp4) == 6
    assert (tmp_path / "out" / "contact_sheet.png").exists()
    assert "in band" in (tmp_path / "out" / "realism.md").read_text()


def test_cli_realism_passthrough(tmp_path):
    from typer.testing import CliRunner

    from survsim.cli import app

    _fake_render(tmp_path, cams=("cam_01",), n=3)
    r = CliRunner().invoke(app, ["realism", "compare", "--real", str(tmp_path / "render/cam_01"),
                                 "--synthetic", str(tmp_path / "render/cam_01"), "--n", "3"])
    assert r.exit_code == 0, r.output
    assert "in band" in r.output


def test_compare_keeps_same_named_sets_apart(tmp_path, capsys):
    from survsim.realism.compare import main

    _fake_render(tmp_path / "a", cams=("cam_01",), n=2)
    _fake_render(tmp_path / "b", cams=("cam_01",), n=2)
    main(["compare", "--real", str(tmp_path / "a/render/cam_01"), "--synthetic",
          str(tmp_path / "a/render/cam_01"), "--synthetic", str(tmp_path / "b/render/cam_01")])
    band = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("in band"))
    assert band.count("=") == 2

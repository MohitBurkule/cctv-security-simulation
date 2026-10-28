"""Tests for video export and VMS metadata.

The behaviours that matter here are about the *boundary* with a real system:
cameras do not share a clock, frames get dropped in transport, and metadata must
survive a VMS rescaling the stream. Each is a place where code that works in the
lab fails on site.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from survsim.core.models import Camera, Mount
from survsim.gt.video import (
    camera_clock,
    export_camera,
    onvif_metadata,
    plan_timing,
    rtsp_command,
    write_timing_sidecar,
    write_video,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _cam(cam_id="cam_a", offset_ms=0.0, drift_ppm=0.0, loss=0.0) -> Camera:
    return Camera(
        id=cam_id, mount=Mount(pos=(0.0, 0.0, 4.0), yaw=0.0, pitch=-15.0),
        capture={"res": (64, 48), "fps": 10.0},
        stream={"clock_offset_ms": offset_ms, "drift_ppm": drift_ppm,
                "packet_loss": loss, "codec": "h264"},
    )


def _frames(n=12, w=64, h=48):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def _rows(n=6):
    return [{"frame": i, "uid": f"p{i % 2}", "bbox": (8.0, 6.0, 16.0, 24.0),
             "occlusion": 0.1, "px_height": 24.0, "label": "suspicious.loiter"}
            for i in range(n)]


# ------------------------------------------------------------------ clocks


def test_clock_offset_shifts_timestamps() -> None:
    cam = _cam(offset_ms=180.0)
    assert camera_clock(cam, 10.0) == pytest.approx(10.18)


def test_drift_accumulates_over_time() -> None:
    """30 ppm is unremarkable and still enough to matter over a long recording."""
    cam = _cam(drift_ppm=30.0)
    assert camera_clock(cam, 0.0) == pytest.approx(0.0)
    assert camera_clock(cam, 3600.0) == pytest.approx(3600.0 + 0.108)


def test_two_cameras_disagree_on_when_a_frame_index_happened() -> None:
    """The assumption most multi-camera code makes without noticing.

    A person walks ~25 cm in 180 ms, which is more than enough to break naive
    cross-camera association that matches on frame index.
    """
    a = plan_timing(_cam("a", offset_ms=0.0), 10, 10.0)
    b = plan_timing(_cam("b", offset_ms=180.0), 10, 10.0)
    assert a[5].sim_t == b[5].sim_t                      # same true instant
    assert abs(a[5].pts_s - b[5].pts_s) == pytest.approx(0.18)


def test_sidecar_warns_that_frame_indices_are_not_comparable(tmp_path) -> None:
    cam = _cam(offset_ms=120.0, drift_ppm=25.0)
    path = write_timing_sidecar(plan_timing(cam, 10, 10.0), tmp_path / "t.json", cam)
    payload = json.loads(path.read_text())

    assert payload["clock_offset_ms"] == 120.0
    assert "NOT comparable across cameras" in payload["note"]
    assert len(payload["frames"]) == 10


# ------------------------------------------------------------ frame loss


def test_packet_loss_drops_frames() -> None:
    timings = plan_timing(_cam(loss=0.5), 200, 10.0, seed=1)
    dropped = sum(t.dropped for t in timings)
    assert 60 < dropped < 140          # roughly half, allowing for sampling


def test_no_loss_keeps_every_frame() -> None:
    assert not any(t.dropped for t in plan_timing(_cam(loss=0.0), 50, 10.0))


def test_dropped_frames_are_recorded_not_hidden(tmp_path) -> None:
    """A gap a consumer has to infer is a gap a consumer will get wrong."""
    cam = _cam(loss=0.3)
    timings = plan_timing(cam, 100, 10.0, seed=2)
    payload = json.loads(write_timing_sidecar(timings, tmp_path / "t.json", cam).read_text())

    assert payload["frames_dropped"] > 0
    assert payload["frames_planned"] == 100
    assert any(f["dropped"] for f in payload["frames"])


# ------------------------------------------------------------- metadata


def test_metadata_uses_normalised_coordinates() -> None:
    """A VMS rescales the stream, and pixel boxes stop meaning anything when it does."""
    meta = onvif_metadata(_rows(3), _cam())["VideoAnalytics"]
    box = meta["Frames"][0]["Objects"][0]["BoundingBox"]

    assert box["left"] == pytest.approx(8.0 / 64)
    assert box["top"] == pytest.approx(6.0 / 48)
    assert all(0.0 <= v <= 1.0 for v in box.values())


def test_metadata_carries_identity_and_difficulty() -> None:
    objs = onvif_metadata(_rows(2), _cam())["VideoAnalytics"]["Frames"][0]["Objects"]
    assert objs[0]["ObjectId"] == "p0"
    assert objs[0]["Class"] == "Human"
    assert "occlusion" in objs[0]["Attributes"]


def test_metadata_is_stamped_on_the_camera_clock() -> None:
    cam = _cam(offset_ms=200.0)
    timings = plan_timing(cam, 6, 10.0)
    meta = onvif_metadata(_rows(6), cam, timings)["VideoAnalytics"]
    first = meta["Frames"][0]
    assert first["UtcTime"] == pytest.approx(0.2)


def test_no_metadata_is_emitted_for_frames_that_were_never_sent() -> None:
    """Claiming a detection on a frame the VMS never received is a real bug."""
    cam = _cam(loss=1.0)
    timings = plan_timing(cam, 6, 10.0, seed=3)
    meta = onvif_metadata(_rows(6), cam, timings)["VideoAnalytics"]
    assert meta["Frames"] == []


# --------------------------------------------------------------- export


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_video_is_written_and_playable(tmp_path) -> None:
    path = write_video(_frames(15), tmp_path / "v.mp4", fps=10.0)
    assert path is not None and path.exists()
    assert path.stat().st_size > 0


def test_export_without_ffmpeg_degrades_rather_than_crashing(tmp_path, monkeypatch) -> None:
    """Metadata and timing are still useful when the muxer is missing."""
    monkeypatch.setattr("survsim.gt.video.write_video", lambda *a, **k: None)
    result = export_camera(tmp_path, _cam(), _frames(8), _rows(4))

    assert result["video_written"] is False
    assert json.loads((tmp_path / "metadata" / "cam_a.onvif.json").read_text())
    assert (tmp_path / "video" / "cam_a.timing.json").exists()


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_export_writes_all_three_artefacts(tmp_path) -> None:
    result = export_camera(tmp_path, _cam(loss=0.1), _frames(20), _rows(10), seed=5)
    assert result["video_written"]
    assert result["frames_written"] + result["frames_dropped"] == 20
    assert (tmp_path / "metadata" / "cam_a.onvif.json").exists()


def test_empty_frame_list_returns_nothing(tmp_path) -> None:
    assert write_video([], tmp_path / "v.mp4", fps=10.0) is None


# ----------------------------------------------------------------- rtsp


def test_rtsp_command_is_returned_not_executed(tmp_path) -> None:
    """Starting a server is a deployment decision, not a library side effect."""
    cmd = rtsp_command(tmp_path / "v.mp4", _cam(), port=8554)
    assert cmd[0] == "ffmpeg"
    assert "-re" in cmd                         # real-time pacing
    assert cmd[-1].endswith("/cam_a")
    assert "rtsp://127.0.0.1:8554" in cmd[-1]

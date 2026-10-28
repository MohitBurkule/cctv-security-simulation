"""Per-camera video export and VMS-compatible metadata.

A dataset that only our own evaluator can read proves nothing about a deployment.
Real analytics arrives as an RTSP stream into a VMS and leaves as metadata events
on a bus, and a simulator that skips that boundary never exercises the parts that
actually break: clock drift between cameras, dropped frames, timestamps that
disagree with frame indices, and metadata whose schema the VMS rejects.

So frames are muxed into per-camera video with an explicit **timing sidecar**, and
detections are emitted in an ONVIF-Profile-M-shaped JSON that a VMS adapter can
consume.

The timing model is the part worth caring about. Each camera has its own clock
offset and drift, so frame 100 on `cam_a` and frame 100 on `cam_b` are *not*
simultaneous. Any multi-camera method that assumes they are will work perfectly
here and fail on site, which is exactly the failure this export is meant to
expose rather than hide.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.models import Camera


@dataclass
class FrameTiming:
    """When a frame was actually captured, as this camera believes."""

    frame: int
    sim_t: float            # true simulation time, which no real system knows
    pts_s: float            # presentation timestamp this camera would stamp
    dropped: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"frame": self.frame, "sim_t": round(self.sim_t, 4),
                "pts_s": round(self.pts_s, 4), "dropped": self.dropped}


def camera_clock(cam: Camera, sim_t: float) -> float:
    """Convert true time into what this camera's clock reads.

    Offset plus linear drift. A camera 180 ms out with 30 ppm drift is an
    unremarkable real installation, and it is enough to break naive cross-camera
    association at walking pace — a person covers ~25 cm in 180 ms.
    """
    offset = cam.stream.clock_offset_ms / 1000.0
    drift = cam.stream.drift_ppm * 1e-6 * sim_t
    return sim_t + offset + drift


def plan_timing(cam: Camera, n_frames: int, fps: float, seed: int = 0) -> list[FrameTiming]:
    """Decide each frame's timestamp and whether it survives transport."""
    rng = np.random.default_rng(seed)
    loss = float(cam.stream.packet_loss or 0.0)
    out: list[FrameTiming] = []
    for i in range(n_frames):
        sim_t = i / max(fps, 1e-6)
        out.append(FrameTiming(
            frame=i, sim_t=sim_t, pts_s=camera_clock(cam, sim_t),
            # Loss is per-frame and independent: a dropped frame is simply absent
            # from the video, but still present in the sidecar so the gap is
            # explicit rather than something a consumer has to infer.
            dropped=bool(rng.random() < loss),
        ))
    return out


def write_video(
    frames: Sequence[np.ndarray],
    path: Path,
    fps: float,
    codec: str = "h264",
    crf: int = 23,
) -> Path | None:
    """Mux frames into a video file. Returns None if ffmpeg is unavailable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return None

    height, width = frames[0].shape[:2]
    # Even dimensions are required by yuv420p, which is what anything downstream
    # will expect to be able to decode.
    width -= width % 2
    height -= height % 2
    encoder = {"h264": "libx264", "h265": "libx265"}.get(codec, "libx264")

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-c:v", encoder, "-crf", str(crf), "-pix_fmt", "yuv420p", str(path),
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame[:height, :width, :3]).tobytes())
        proc.stdin.close()
        proc.wait(timeout=300)
    except (FileNotFoundError, BrokenPipeError, subprocess.TimeoutExpired):
        return None
    return path if path.exists() else None


def write_timing_sidecar(timings: Sequence[FrameTiming], path: Path,
                         cam: Camera) -> Path:
    """Record the timing a consumer needs to align cameras correctly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = [t for t in timings if not t.dropped]
    payload = {
        "camera_id": cam.id,
        "fps": cam.capture.fps,
        "clock_offset_ms": cam.stream.clock_offset_ms,
        "drift_ppm": cam.stream.drift_ppm,
        "frames_planned": len(timings),
        "frames_written": len(kept),
        "frames_dropped": len(timings) - len(kept),
        # Stated explicitly because it is the assumption most multi-camera code
        # makes without noticing: frame index N is not the same instant on two
        # cameras, and pts_s is the only field that can be compared across them.
        "note": ("pts_s is this camera's own clock, including offset and drift. "
                 "Frame indices are NOT comparable across cameras; align on pts_s."),
        "frames": [t.as_dict() for t in timings],
    }
    path.write_text(json.dumps(payload, indent=1))
    return path


# ------------------------------------------------------------- VMS metadata


def onvif_metadata(
    rows: Iterable[dict[str, Any]],
    cam: Camera,
    timings: Sequence[FrameTiming] | None = None,
) -> dict[str, Any]:
    """Detections in an ONVIF-Profile-M-shaped structure.

    Not a byte-exact ONVIF XML document — that belongs in an adapter — but the
    same information model: per-frame objects with a bounding box in normalised
    coordinates, a class, and a timestamp on the camera's own clock. Normalised
    coordinates matter because a VMS re-scales streams and pixel boxes stop
    meaning anything once it does.
    """
    by_pts = {t.frame: t for t in (timings or [])}
    width, height = cam.capture.res
    frames: dict[int, list[dict[str, Any]]] = {}

    for row in rows:
        frame = int(row["frame"])
        timing = by_pts.get(frame)
        if timing is not None and timing.dropped:
            continue        # no frame was transmitted, so no metadata for it
        x, y, w, h = (float(v) for v in row["bbox"])
        frames.setdefault(frame, []).append({
            "ObjectId": row.get("uid"),
            "Class": "Human",
            "BoundingBox": {
                "left": round(x / max(width, 1), 5),
                "top": round(y / max(height, 1), 5),
                "right": round((x + w) / max(width, 1), 5),
                "bottom": round((y + h) / max(height, 1), 5),
            },
            "Attributes": {
                k: row[k] for k in ("occlusion", "px_height", "label", "visible_frac")
                if k in row
            },
        })

    return {
        "VideoAnalytics": {
            "SourceId": cam.id,
            "Resolution": {"Width": width, "Height": height},
            "Frames": [
                {
                    "UtcTime": round(by_pts[f].pts_s, 4) if f in by_pts else None,
                    "FrameIndex": f,
                    "Objects": objs,
                }
                for f, objs in sorted(frames.items())
            ],
        }
    }


def export_camera(
    out_dir: Path,
    cam: Camera,
    frames: Sequence[np.ndarray],
    rows: Sequence[dict[str, Any]],
    fps: float | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Write video, timing sidecar and VMS metadata for one camera."""
    out_dir = Path(out_dir)
    fps = fps or cam.capture.fps
    timings = plan_timing(cam, len(frames), fps, seed)
    kept = [f for f, t in zip(frames, timings) if not t.dropped]

    video = write_video(kept, out_dir / "video" / f"{cam.id}.mp4", fps,
                        cam.stream.codec, max(1, min(51, cam.stream.gop // 2 or 23)))
    sidecar = write_timing_sidecar(timings, out_dir / "video" / f"{cam.id}.timing.json", cam)

    meta_path = out_dir / "metadata" / f"{cam.id}.onvif.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(onvif_metadata(rows, cam, timings), indent=1))

    return {
        "camera": cam.id,
        "video": str(video) if video else None,
        "video_written": video is not None,
        "timing": str(sidecar),
        "metadata": str(meta_path),
        "frames_written": len(kept),
        "frames_dropped": len(frames) - len(kept),
    }


def rtsp_command(path: Path, cam: Camera, port: int = 8554) -> list[str]:
    """The command that re-streams an exported file as live RTSP.

    Returned rather than run: replaying a dataset as a live stream is how you
    exercise a real VMS ingestion path, but starting a server is a deployment
    decision, not something a library should do behind the caller's back.
    """
    return [
        "ffmpeg", "-re", "-stream_loop", "-1", "-i", str(path),
        "-c", "copy", "-f", "rtsp", f"rtsp://127.0.0.1:{port}/{cam.id}",
    ]

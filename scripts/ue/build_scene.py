"""Rebuild a SurvSim run inside Unreal Editor and queue it for rendering.

This runs **inside** the editor's embedded Python, not in our venv:

    /opt/unreal-engine/Engine/Binaries/Linux/UnrealEditor <project>.uproject \
        -run=pythonscript -script="scripts/ue/build_scene.py --manifest out/run/manifest.json"

It imports `unreal`, which only exists there. Everything that can be checked
without the editor lives in `survsim/export/unreal_scene.py` and is unit-tested;
this file is deliberately thin, because code here can only be exercised by
launching a 20 GB editor.

**It does not convert units.** The manifest arrives already in Unreal's frame —
centimetres, Y negated, yaw negated. Re-applying any of that here would double
the conversion, which renders a plausible-looking mirrored scene while ground
truth continues to describe the original. All of it is done, once, in the
exporter.

Status: **[A] written against the documented API, not yet run.** Following the
project's own rule, treat every claim here as unverified until the editor has
executed it. The two most likely breakages are the MetaHuman API surface (it is
under `Experimental/` in 5.8 and moves between releases) and the exact Movie
Render Queue setting classes.
"""

from __future__ import annotations

import argparse
import json
import sys

try:
    import unreal
except ImportError:  # pragma: no cover - only importable inside the editor
    print("This script must run inside Unreal Editor (no `unreal` module).")
    raise SystemExit(2)

EXPECTED_VERSION = 1


def log(message: str) -> None:
    unreal.log(f"[survsim] {message}")


def hex_to_linear_colour(value: str) -> "unreal.LinearColor":
    """'#rrggbb' -> LinearColor, converting sRGB to linear.

    The conversion matters. Assigning an sRGB triple straight to a linear
    material parameter makes every garment render noticeably brighter and
    washed out, which would then be measured as a realism gap and "fixed" in
    the degradation chain — compensating for a colour-space bug with a lens
    model.
    """
    text = value.lstrip("#")
    if len(text) != 6:
        return unreal.LinearColor(0.25, 0.25, 0.25, 1.0)
    srgb = [int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]

    def to_linear(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (to_linear(c) for c in srgb)
    return unreal.LinearColor(r, g, b, 1.0)


def load_manifest(path: str) -> dict:
    with open(path) as handle:
        manifest = json.load(handle)
    version = manifest.get("version")
    if version != EXPECTED_VERSION:
        raise SystemExit(
            f"manifest version {version} != expected {EXPECTED_VERSION}; "
            "refusing to build a scene from a format this script may "
            "misinterpret")
    return manifest


def spawn_cameras(manifest: dict) -> dict:
    """One CineCameraActor per configured camera, at the exported pose."""
    cameras = {}
    for spec in manifest.get("cameras", []):
        location = unreal.Vector(*spec["location"])
        # Manifest order is (pitch, yaw, roll); Rotator's positional order is
        # (roll, pitch, yaw). Keywords only — see docs/BUG_black_render.md.
        pitch, yaw, roll = spec["rotation"]
        rotation = unreal.Rotator(pitch=pitch, yaw=yaw, roll=roll)
        actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
            unreal.CineCameraActor, location, rotation)
        actor.set_actor_label(f"cam_{spec['id']}")

        component = actor.get_cine_camera_component()
        if spec.get("focal_mm"):
            component.set_editor_property("current_focal_length", float(spec["focal_mm"]))
        filmback = component.get_editor_property("filmback")
        if spec.get("sensor_mm"):
            filmback.set_editor_property("sensor_width", float(spec["sensor_mm"]))
            component.set_editor_property("filmback", filmback)

        cameras[spec["id"]] = actor
        log(f"camera {spec['id']} at {spec['location']} rot {spec['rotation']}")
    return cameras


def apply_wardrobe(actor, wardrobe: dict) -> None:
    """Drive whatever appearance control this build actually exposes.

    Written defensively on purpose. Wardrobe support was twice reported as
    working because an API name existed, so nothing here assumes a call
    succeeded: each step is attempted, and what actually applied is logged so a
    run cannot silently produce a scene of identically-dressed people.
    """
    applied = []
    mesh = actor.get_editor_property("skeletal_mesh_component") \
        if hasattr(actor, "get_editor_property") else None
    if mesh is None:
        log("no skeletal mesh component; wardrobe not applied")
        return

    for slot, key in ((0, "top_colour"), (1, "bottom_colour")):
        colour = wardrobe.get(key)
        if not colour:
            continue
        try:
            material = mesh.get_material(slot)
            if material is None:
                continue
            dynamic = mesh.create_dynamic_material_instance(slot, material)
            dynamic.set_vector_parameter_value("BaseColor", hex_to_linear_colour(colour))
            applied.append(key)
        except Exception as exc:                       # noqa: BLE001
            log(f"could not set {key}: {exc}")

    # Height is a body-proportion change, not a mesh scale; scaling the actor
    # makes a tall person uniformly wide as well, which is not what a tall
    # person looks like and would teach a height estimator the wrong prior.
    height = wardrobe.get("height_m")
    if height:
        log(f"height {height} m requested — needs the MetaHuman parametric body, "
            "not actor scale")

    log(f"wardrobe applied: {applied or 'nothing'}")


def build_sequence(manifest: dict, cameras: dict):
    """Create a Level Sequence with one transform track per person."""
    fps = int(round(float(manifest.get("fps", 25.0))))
    factory = unreal.LevelSequenceFactoryNew()
    sequence = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        asset_name=f"seq_{manifest.get('scenario', 'run')}",
        package_path="/Game/SurvSim",
        asset_class=unreal.LevelSequence,
        factory=factory)

    sequence.set_display_rate(unreal.FrameRate(fps, 1))
    duration = float(manifest.get("duration_s", 0.0))
    sequence.set_playback_start(0)
    sequence.set_playback_end(int(duration * fps) if duration else 1)

    for track in manifest.get("tracks", []):
        keys = track.get("keys", [])
        if not keys:
            continue
        # A real build spawns a MetaHuman here; the placeholder keeps the
        # geometry and timing testable before the character work lands.
        first = keys[0]
        actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
            unreal.SkeletalMeshActor, unreal.Vector(*first["location"]),
            unreal.Rotator(pitch=0.0, yaw=first["yaw"], roll=0.0))
        actor.set_actor_label(track["uid"])

        binding = sequence.add_possessable(actor)
        transform_track = binding.add_track(unreal.MovieScene3DTransformTrack)
        section = transform_track.add_section()
        section.set_range(keys[0]["frame"], keys[-1]["frame"] + 1)

        channels = section.get_all_channels()
        for key in keys:
            frame = unreal.FrameNumber(int(key["frame"]))
            x, y, z = key["location"]
            # Channel order is translation XYZ, rotation XYZ, scale XYZ.
            for channel, value in zip(channels, (x, y, z, 0.0, float(key["yaw"]), 0.0)):
                channel.add_key(frame, value)

    log(f"sequence built with {len(manifest.get('tracks', []))} people")
    return sequence


def queue_render(sequence, manifest: dict, output_dir: str) -> None:
    """Add the sequence to Movie Render Queue, one job per camera."""
    subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    queue = subsystem.get_queue()

    for spec in manifest.get("cameras", []):
        job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
        job.job_name = f"{manifest.get('scenario', 'run')}_{spec['id']}"
        job.sequence = unreal.SoftObjectPath(sequence.get_path_name())

        config = job.get_configuration()
        config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
        config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)

        output = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
        output.output_directory = unreal.DirectoryPath(f"{output_dir}/{spec['id']}")
        width, height = spec.get("resolution", [1280, 720])
        output.output_resolution = unreal.IntPoint(int(width), int(height))
        # Fixed frame rate, not the editor's: a render that adapts its timing
        # would no longer line up with ground truth computed at a fixed step.
        output.use_custom_frame_rate = True
        output.output_frame_rate = unreal.FrameRate(int(round(spec.get("fps", 25))), 1)

        log(f"queued render job for {spec['id']} -> {output_dir}/{spec['id']}")


def main(argv: list) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="/tmp/survsim_render")
    parser.add_argument("--no-render", action="store_true",
                        help="build the scene but do not queue a render")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    log(f"rebuilding '{manifest.get('scenario')}' seed {manifest.get('seed')}")

    cameras = spawn_cameras(manifest)
    sequence = build_sequence(manifest, cameras)
    if not args.no_render:
        queue_render(sequence, manifest, args.out)
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

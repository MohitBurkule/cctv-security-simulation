"""Build a minimal renderable scene, to prove the offline render path works.

The export -> editor -> Movie Render Queue path was written and unit-tested but
never executed, which made "SurvSim can produce photoreal renders" the largest
unverified claim in the repo. This script builds the smallest scene that can
falsify it: ground, sun, sky, three person-sized actors, a CCTV-posed camera,
and a Level Sequence with a camera cut and a moving actor.

Deliberately uses primitives rather than MetaHumans. The question here is
whether the *render pipeline* works headless on this machine; mixing in the
MetaHuman texture blocker would confound the answer.

    UnrealEditor SurvSim.uproject -run=pythonscript \
        -script=scripts/ue/make_test_scene.py -unattended -nosplash

Two commandlet-mode traps, both silent, both cost time:

  - **`spawn_actor_from_object` does not work here.** It returns None and logs
    only "SpawnActorFromObject. No actor was spawned." Spawn from *class* and
    assign the mesh to the component afterwards. Every subsequent call on the
    None then fails with an unrelated-looking AttributeError.
  - **`new_level` returns False if the map already exists**, leaving no valid
    world, after which every spawn silently returns None. Load the level in
    that case rather than assuming creation succeeded.

A scene that renders pure black is the classic false positive, so a
DirectionalLight, SkyLight and SkyAtmosphere are all placed — a "successful"
render of an unlit scene would otherwise look like proof.
"""

from __future__ import annotations

import json
import traceback

try:
    import unreal
except ImportError:  # pragma: no cover - editor only
    raise SystemExit("must run inside Unreal Editor")

MAP = "/Game/SurvSim/TestMap"
SEQ_PACKAGE = "/Game/SurvSim"
SEQ_NAME = "TestSeq"
RESULT = "/tmp/survsim_scene_result.json"
FPS = 25
DURATION_FRAMES = 50


def spawn_mesh(actors, mesh, location, scale, label):
    """Spawn a StaticMeshActor from class, then give it a mesh.

    The two-step dance is required: spawn_actor_from_object is inert in
    commandlet mode.
    """
    actor = actors.spawn_actor_from_class(unreal.StaticMeshActor, location)
    if actor is None:
        raise RuntimeError(f"could not spawn {label}")
    actor.static_mesh_component.set_static_mesh(mesh)
    actor.set_actor_scale3d(scale)
    actor.set_actor_label(label)
    return actor


def main() -> int:
    out: dict = {"steps": []}

    def step(name, fn):
        try:
            value = fn()
            out["steps"].append({"step": name, "ok": True, "value": str(value)[:160]})
            return value
        except Exception as exc:                            # noqa: BLE001
            out["steps"].append({"step": name, "ok": False,
                                 "error": f"{type(exc).__name__}: {exc}"[:240]})
            return None

    try:
        levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
        actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        tools = unreal.AssetToolsHelpers.get_asset_tools()

        # Delete the previous map and sequence so every run starts clean.
        # Without this, `new_level` fails on the existing asset, actors
        # accumulate on each run (24 after three runs, with duplicate suns and
        # skylights), and `create_asset` silently returns None for a name that
        # already exists — leaving the sequence unbuilt while everything else
        # looks fine.
        for asset in (MAP, f"{SEQ_PACKAGE}/{SEQ_NAME}"):
            if unreal.EditorAssetLibrary.does_asset_exist(asset):
                unreal.EditorAssetLibrary.delete_asset(asset)

        created = levels.new_level(MAP)
        out["created_level"] = bool(created)
        if not created:
            out["loaded_level"] = bool(levels.load_level(MAP))

        # delete_asset silently fails on the map when it is the loaded startup
        # map, so the level survives and actors pile up (four suns after four
        # runs). Empty whatever level we ended up in.
        stale = actors.get_all_level_actors() or []
        for actor in stale:
            actors.destroy_actor(actor)
        out["stale_actors_destroyed"] = len(stale)

        cube = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Cube.Cube")
        plane = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Plane.Plane")
        if cube is None or plane is None:
            raise RuntimeError("basic shapes not loadable")

        # Ground, 40 m across.
        spawn_mesh(actors, plane, unreal.Vector(0, 0, 0),
                   unreal.Vector(40, 40, 1), "Ground")

        # Lighting. An unlit scene renders black and would pass a naive check.
        sun = actors.spawn_actor_from_class(
            unreal.DirectionalLight, unreal.Vector(0, 0, 500),
            # Rotator's POSITIONAL order in Python is (roll, pitch, yaw), not the
            # (pitch, yaw, roll) of the C++ constructor. Passed positionally this
            # sun shone upwards from below the horizon and the camera stared
            # straight up into the resulting night sky: RGB exactly 0.0, every
            # frame, with everything reporting success. Always use keywords.
            unreal.Rotator(pitch=-45, yaw=45, roll=0))
        sun.set_actor_label("Sun")
        # Intensities are set explicitly. A spawned light keeps whatever the
        # class default is, and a scene lit only by defaults rendered pure
        # black — the renderer reported success either way, so the only signal
        # was the pixels.
        step("sun_intensity", lambda: sun.light_component.set_editor_property("intensity", 12.0))
        step("sun_is_atmosphere_sun", lambda:
             sun.light_component.set_editor_property(
                 "atmosphere_sun_light", True))
        # MOVABLE, so lighting is fully dynamic. A Stationary or Static light
        # needs a baked lightmap, and without a bake the renderer stamps a
        # "Preview" watermark across every shadow — which would then be measured
        # as part of our realism statistics and degradation calibration. It also
        # matters for the simulator proper: people and doors move, and a baked
        # solution cannot light them.
        step("sun_movable", lambda: sun.light_component.set_editor_property(
            "mobility", unreal.ComponentMobility.MOVABLE))

        sky = actors.spawn_actor_from_class(
            unreal.SkyLight, unreal.Vector(0, 0, 400))
        sky.set_actor_label("SkyLight")
        step("skylight_intensity", lambda: sky.light_component.set_editor_property(
            "intensity", 3.0))
        step("skylight_realtime", lambda: sky.light_component.set_editor_property(
            "real_time_capture", True))
        step("skylight_movable", lambda: sky.light_component.set_editor_property(
            "mobility", unreal.ComponentMobility.MOVABLE))
        step("sky_atmosphere", lambda: actors.spawn_actor_from_class(
            unreal.SkyAtmosphere, unreal.Vector(0, 0, 0)))

        # Person-sized stand-ins: 0.5 x 0.5 x 1.8 m, so framing is realistic.
        people = []
        for i in range(3):
            people.append(spawn_mesh(
                actors, cube, unreal.Vector(-300 + 300 * i, 0, 90),
                unreal.Vector(0.5, 0.5, 1.8), f"person_{i}"))
        out["n_people"] = len(people)

        # A CCTV pose: 4 m up, pitched 20 degrees down, looking along +Y.
        camera = actors.spawn_actor_from_class(
            unreal.CineCameraActor, unreal.Vector(0, -800, 400),
            unreal.Rotator(pitch=-20, yaw=90, roll=0))
        camera.set_actor_label("cam_a")

        step("save_level", lambda: levels.save_current_level())

        sequence = step("create_sequence", lambda: tools.create_asset(
            asset_name=SEQ_NAME, package_path=SEQ_PACKAGE,
            asset_class=unreal.LevelSequence,
            factory=unreal.LevelSequenceFactoryNew()))

        if sequence is not None:
            sequence.set_display_rate(unreal.FrameRate(FPS, 1))
            sequence.set_playback_start(0)
            sequence.set_playback_end(DURATION_FRAMES)

            # Without a camera cut the render uses no camera and the output is
            # whatever the editor viewport happened to show.
            cam_binding = sequence.add_possessable(camera)
            cut_track = sequence.add_track(unreal.MovieSceneCameraCutTrack)
            cut_section = cut_track.add_section()
            cut_section.set_range(0, DURATION_FRAMES)
            # MovieSceneObjectBindingID(guid=...) is NOT constructible in 5.8
            # ("call() takes at most 0 arguments"). Build it empty and set the
            # property, falling back to the Sequencer helper.
            binding_id = unreal.MovieSceneObjectBindingID()
            try:
                binding_id.set_editor_property("guid", cam_binding.get_id())
            except Exception:                                # noqa: BLE001
                binding_id = unreal.MovieSceneSequenceExtensions.get_binding_id(
                    sequence, cam_binding)
            cut_section.set_camera_binding_id(binding_id)

            # Move one actor across frame, so the output proves animation
            # rather than 50 copies of one still.
            binding = sequence.add_possessable(people[0])
            track = binding.add_track(unreal.MovieScene3DTransformTrack)
            section = track.add_section()
            section.set_range(0, DURATION_FRAMES)
            channels = section.get_all_channels()
            channels[0].add_key(unreal.FrameNumber(0), -300.0)
            channels[0].add_key(unreal.FrameNumber(DURATION_FRAMES), 300.0)
            out["keyed_channels"] = len(channels)

            step("save_sequence", lambda: unreal.EditorAssetLibrary.save_asset(
                sequence.get_path_name()))
            out["sequence_path"] = sequence.get_path_name()

        out["map_path"] = MAP
        out["actor_count"] = len(actors.get_all_level_actors() or [])
    except Exception:                                        # noqa: BLE001
        out["fatal"] = traceback.format_exc()[-1500:]

    with open(RESULT, "w") as handle:
        json.dump(out, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

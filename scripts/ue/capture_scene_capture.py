"""Capture frames with SceneCapture2D — a render path independent of MRQ.

Movie Render Queue writes pure black on this machine (docs/BUG_black_render.md):
RGB exactly 0.0 in linear EXR, alpha exactly 1.0, on both GPUs, with shaders
compiled and the scene verified correct. Eleven hypotheses eliminated.

`SceneCaptureComponent2D` reaches the renderer through entirely different code:
it renders a view into a `TextureRenderTarget2D` and is exported with
`RenderingLibrary.export_render_target`. It is also what UnrealCV uses, so if it
works it is not merely a diagnostic — it is a capture path SurvSim can adopt,
and one that is arguably a better fit anyway:

  - capture is per-camera and on demand, so a rig of eight cameras is eight
    captures at one simulation instant rather than eight separate render jobs;
  - no Level Sequence is required, so the Director can drive actor transforms
    directly and capture, keeping ground truth and pixels on the same clock;
  - the render target's format and size are ours to choose.

If this also comes out black, the fault is below both paths — in the renderer or
the Vulkan offscreen surface — and that is worth knowing precisely.

    xvfb-run -a UnrealEditor SurvSim.uproject \
        -ExecCmds="py /abs/path/scripts/ue/capture_scene_capture.py" \
        -unattended -nosplash

Writes /tmp/survsim_capture/*.png and /tmp/survsim_capture_result.json.
**Check the pixels** with tools/check_render.py; every success flag in this
pipeline has been wrong so far.
"""

from __future__ import annotations

import json

import unreal

MAP = "/Game/SurvSim/TestMap"
OUTPUT_DIR = "/tmp/survsim_capture"
RESULT = "/tmp/survsim_capture_result.json"
WIDTH, HEIGHT = 1280, 720
N_FRAMES = 10

# -ExecCmds fires during editor startup, before a level can be loaded; the work
# is deferred to a tick callback. See render_in_editor.py for the full account.
SETTLE_TICKS = 120

_state: dict = {"ticks": 0, "started": False, "handle": None}


def _write(payload: dict) -> None:
    with open(RESULT, "w") as handle:
        json.dump(payload, handle, indent=2)


def _capture() -> dict:
    out: dict = {"steps": []}

    def step(name, fn):
        try:
            value = fn()
            out["steps"].append({"step": name, "ok": True, "value": str(value)[:180]})
            return value
        except Exception as exc:                               # noqa: BLE001
            out["steps"].append({"step": name, "ok": False,
                                 "error": f"{type(exc).__name__}: {exc}"[:240]})
            return None

    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    levels.load_level(MAP)
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

    # A render target to capture into. RTF_RGBA8 keeps the readback simple;
    # a float format would be needed for HDR work later.
    target = step("create_render_target", lambda: unreal.RenderingLibrary
                  .create_render_target2d(unreal.EditorLevelLibrary.get_editor_world(), WIDTH, HEIGHT,
                                          unreal.TextureRenderTargetFormat.RTF_RGBA8))
    if target is None:
        return out

    capture = step("spawn_scene_capture", lambda: actors.spawn_actor_from_class(
        unreal.SceneCapture2D, unreal.Vector(0, -800, 400),
        unreal.Rotator(pitch=-20, yaw=90, roll=0)))
    if capture is None:
        return out

    component = capture.get_component_by_class(unreal.SceneCaptureComponent2D)
    component.set_editor_property("texture_target", target)
    # Final colour through the full post-process chain, i.e. what a camera would
    # actually deliver. LDR here rather than SceneColorHDR so the comparison
    # with the MRQ PNG output is like for like.
    component.set_editor_property(
        "capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
    component.set_editor_property("capture_every_frame", False)
    component.set_editor_property("capture_on_movement", False)

    # Move one actor between captures, so a non-black result also proves the
    # capture reflects the current scene rather than one cached frame.
    movers = [a for a in (actors.get_all_level_actors() or [])
              if a.get_actor_label().startswith("person_")]
    out["n_movers"] = len(movers)

    written = []
    for i in range(N_FRAMES):
        if movers:
            mover = movers[0]
            location = mover.get_actor_location()
            mover.set_actor_location(
                unreal.Vector(-300 + i * 60, location.y, location.z),
                False, False)
        component.capture_scene()
        name = f"capture.{i:04d}"
        ok = unreal.RenderingLibrary.export_render_target(
            capture, target, OUTPUT_DIR, f"{name}.png")
        written.append({"frame": i, "exported": bool(ok)})

    out["frames"] = written
    out["output_dir"] = OUTPUT_DIR
    return out


def _tick(delta_seconds: float) -> None:
    _state["ticks"] += 1
    if _state["ticks"] < SETTLE_TICKS or _state["started"]:
        return
    _state["started"] = True
    if _state["handle"] is not None:
        unreal.unregister_slate_post_tick_callback(_state["handle"])
        _state["handle"] = None
    try:
        payload = _capture()
        payload["completed"] = True
        _write(payload)
        unreal.log("[survsim] scene capture complete")
    except Exception as exc:                                   # noqa: BLE001
        unreal.log_error(f"[survsim] capture failed: {exc}")
        _write({"completed": False, "error": str(exc)[:400]})
    unreal.SystemLibrary.quit_editor()


def main() -> None:
    unreal.log("[survsim] scene capture queued; waiting for editor to settle")
    _write({"completed": False, "note": "not finished"})
    _state["handle"] = unreal.register_slate_post_tick_callback(_tick)


main()

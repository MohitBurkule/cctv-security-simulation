"""Render a Level Sequence from inside the running editor, via MRQ's PIE executor.

**Why not `-game`.** The documented headless path
(`-game -MoviePipelineLocalExecutorClass=...`) runs the *uncooked* project with
no compiled shaders, and Movie Render Queue does not wait for the shader
compiler. Every frame comes out pure black while the pipeline reports
`success=True` — 50 correctly-sized PNGs, a resolved camera cut, exit code 0,
and RGB identically zero with alpha 255. Filling the DerivedDataCache, switching
SM5 to SM6, adding a virtual framebuffer and setting explicit light intensities
each fixed a real misconfiguration and none of them changed the pixels.

The editor is the environment MRQ was built for: it compiles shaders on demand
and it ticks, which `-run=pythonscript` does not. `MoviePipelinePIEExecutor`
runs the render inside that loop.

    xvfb-run -a UnrealEditor SurvSim.uproject \
        -ExecCmds="py /abs/path/scripts/ue/render_in_editor.py" \
        -unattended -nosplash

Writes /tmp/survsim_render_result.json. Verify the PIXELS with
`tools/check_render.py` — this pipeline's success flag has been wrong every time
so far.
"""

from __future__ import annotations

import json

import unreal

SEQUENCE = "/Game/SurvSim/TestSeq.TestSeq"
MAP = "/Game/SurvSim/TestMap"
OUTPUT_DIR = "/tmp/survsim_render_editor"
RESULT = "/tmp/survsim_render_result.json"

WIDTH, HEIGHT = 1280, 720
WARM_UP_FRAMES = 32

_state: dict = {"finished": False, "ticks": 0, "handle": None, "started": False}

# `-ExecCmds` runs during editor STARTUP, before the editor can load a level.
# Calling load_level from there blocks forever: the script logs that it started
# and nothing further happens, while the editor carries on initialising and
# reports no error at all. So the work is deferred to a tick callback and given
# a few seconds of frames for startup to settle.
SETTLE_TICKS = 120


def _write(payload: dict) -> None:
    with open(RESULT, "w") as handle:
        json.dump(payload, handle, indent=2)


def _on_finished(executor, success: bool) -> None:
    """MRQ passes (executor, success) — a one-argument callback is rejected.

    The error surfaces as "Callable has the incorrect number of arguments
    (expected 2, got 1)" *after* the render has already been set up, so the shot
    initialises and then the whole run aborts.
    """
    unreal.log(f"[survsim] render finished success={success}")
    _state["finished"] = True
    _write({"started": True, "success": bool(success), "output_dir": OUTPUT_DIR})
    # Leave the editor once the work is done, so the run terminates on its own.
    unreal.SystemLibrary.quit_editor()


def _begin() -> None:
    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    levels.load_level(MAP)

    subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    queue = subsystem.get_queue()
    # Clear anything left from a previous run, or jobs accumulate and the
    # render silently does more work than asked.
    for job in list(queue.get_jobs()):
        queue.delete_job(job)

    job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    job.job_name = "survsim"
    job.sequence = unreal.SoftObjectPath(SEQUENCE)
    job.map = unreal.SoftObjectPath(MAP)

    config = job.get_configuration()
    config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
    # EXR as well: it is linear and written BEFORE tonemapping, so comparing the
    # two separates "nothing was rendered" from "rendered, then tonemapped to
    # black by exposure". A PNG at zero with a non-zero EXR means the scene is
    # fine and the exposure is not.
    config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_EXR)

    output = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    output.output_directory = unreal.DirectoryPath(OUTPUT_DIR)
    output.output_resolution = unreal.IntPoint(WIDTH, HEIGHT)
    output.file_name_format = "{sequence_name}.{frame_number}"
    output.override_existing_output = True

    # Warm-up matters even here: the first frames are sampled before lighting
    # and streaming converge.
    aa = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    aa.spatial_sample_count = 1
    aa.temporal_sample_count = 1
    aa.engine_warm_up_count = WARM_UP_FRAMES
    aa.render_warm_up_frames = True

    config.find_or_add_setting_by_class(unreal.MoviePipelineGameOverrideSetting)
    config.initialize_transient_settings()

    executor = unreal.MoviePipelinePIEExecutor()
    executor.on_executor_finished_delegate.add_callable_unique(_on_finished)
    unreal.log(f"[survsim] rendering {SEQUENCE} -> {OUTPUT_DIR}")
    subsystem.render_queue_with_executor_instance(executor)


def _tick(delta_seconds: float) -> None:
    """Let the editor settle, then start the render exactly once."""
    _state["ticks"] += 1
    if _state["ticks"] < SETTLE_TICKS or _state["started"]:
        return
    _state["started"] = True
    if _state["handle"] is not None:
        unreal.unregister_slate_post_tick_callback(_state["handle"])
        _state["handle"] = None
    try:
        _begin()
    except Exception as exc:                                   # noqa: BLE001
        unreal.log_error(f"[survsim] render failed to start: {exc}")
        _write({"started": True, "success": False, "error": str(exc)[:400]})
        unreal.SystemLibrary.quit_editor()


def main() -> None:
    unreal.log("[survsim] queued; waiting for the editor to settle")
    _write({"started": True, "success": None, "note": "render not finished"})
    _state["handle"] = unreal.register_slate_post_tick_callback(_tick)


main()

# Unreal Engine 5.8.2 rendered pure black — RESOLVED: it was our Rotator arguments

**Status: [V] resolved 2026-09-18.** Not an engine, driver, RHI, shader or
exposure problem. The scene script aimed the camera straight up into a night sky.

## Root cause

In the Unreal **Python** API, `unreal.Rotator(a, b, c)` takes its positional
arguments as **(roll, pitch, yaw)**. The C++ constructor, the editor UI and our
export manifest all use (pitch, yaw, roll). `make_test_scene.py` wrote:

| Call | Intended | Actually got |
|---|---|---|
| sun `Rotator(-45, 45, 0)` | pitch −45 (shining down) | roll −45, **pitch +45** — light travels *upwards*: the sun is below the horizon, so SkyAtmosphere renders night |
| camera `Rotator(-20, 90, 0)` | pitch −20, yaw 90 | roll −20, **pitch +90** — staring at the zenith |

A camera looking straight up at a night sky with no geometry in view is
*exactly* 0.0 in every colour channel, with alpha 1.0. Every symptom below is
that, faithfully rendered. The renderer never malfunctioned.

## Fix

* **Always construct rotators with keywords:** `unreal.Rotator(pitch=-20, yaw=90, roll=0)`.
  Fixed in `make_test_scene.py`, `capture_scene_capture.py` and `build_scene.py`
  (which unpacked the manifest's `[pitch, yaw, roll]` positionally — every
  exported camera would have been mis-aimed the same way).
* `tests/test_ue_scripts.py` fails on any positional `Rotator(` in `scripts/ue/`.
* Verified by pixels: MRQ 50 frames mean 157.6 / std 21.9; SceneCapture2D
  10 frames mean 60.2 / std 49.7; cubes, ground and shadows visible.

## How it was actually found (the method worth keeping)

1. `SceneCapture2D` (independent of MRQ) was also black → fault is not MRQ.
2. Captured the **G-buffer**: BaseColor and Normal were 0 with alpha 0
   everywhere → *no geometry in view at all*, so lighting/exposure are moot.
3. `clear_render_target2d` and `draw_material_to_render_target` read back
   correct colours → the GPU, shaders and readback all work.
4. So the view itself was wrong → **dumped actor transforms from the saved
   map**: `cam_a` read back as pitch 90, yaw 20. Done.

Step 4 takes one minute and should have been step 1. The earlier check
"16 actors, all present, correct labels" verified names, not poses.
**When a render is empty, read the transforms back before suspecting the engine.**

## Other things learned on the way

* **Vulkan bindless is not the cause.** `BaseLinuxEngine.ini` sets
  `BindlessConfiguration=All` for VULKAN_SM6; overriding it to `Disabled`
  (a ~40 min full shader recompile) changed nothing. Override removed.
* **Actors were accumulating.** `delete_asset` silently fails on the map while
  it is the loaded startup map; the level had 32 actors (four suns). The "16
  actors" seen earlier was already two runs' worth. The script now destroys all
  existing actors before building.
* **Zombie editors.** After an MRQ PIE render, `quit_editor()` does not
  terminate the process; two editors from earlier runs were still alive holding
  VRAM. Wrap runs in `timeout` and check `pgrep -f UnrealEditor` afterwards.
  (SceneCapture runs exit cleanly.)
* **Stale logs mislead.** Concurrent editors write `SurvSim_2.log`,
  `SurvSim_3.log`…; `SurvSim.log` may belong to a zombie. Use `ls -t`.
* Python API names: `RenderingLibrary.create_render_target2d` (lower-case d,
  needs a world context, not `None`); `SceneCapture2D` has no
  `scene_capture_component2d` — use `get_component_by_class`.
  `export_render_target` returns False even when it writes the file.
* The MRQ EXR is pre-tonemap but **not** pre-exposure, so "EXR is zero" never
  did rule out exposure by itself. (Moot here; the scene radiance really was 0.)
* Remaining cosmetics, not bugs: frames are over-exposed and show the
  "Preview" shadow watermark (unbuilt static lighting — make the lights movable).

---

# Original report (kept for the record — its conclusions are superseded)


*(Status at time of writing: unresolved.)*

## One-line statement

Unreal Engine 5.8.2 on Arch Linux, driven headless, produces image sequences
whose **RGB channels are exactly 0.0 and alpha exactly 1.0**, for every frame,
in both 8-bit PNG and 32-bit linear EXR, while the engine, Movie Render Queue
and the renderer all report complete success.

## Environment

| | |
|---|---|
| Engine | Unreal Engine **5.8.2-56702186+++UE5+Release-5.8** (Epic's prebuilt Linux binary, installed from AUR `unreal-engine-bin`, at `/opt/unreal-engine`) |
| OS | Arch Linux, kernel 7.1.3-arch2-2 |
| CPU | Intel i7-10750H |
| GPU 0 | Intel UHD Graphics (CML GT2), Mesa 26.1.5, Vulkan 1.4.354 |
| GPU 1 | NVIDIA RTX 2070 Max-Q 8 GB, **driver 610.43.03 (NVIDIA open kernel module)**, Vulkan 1.4.341 |
| Display | **Headless.** No X session. Tested under both `-RenderOffscreen` and `xvfb-run` |
| RHI | Vulkan, **SM6** (`rhifeaturelevel="SM6"`, `FEATURELEVEL=4` on map load) |

The project is a **Blueprint-only, uncooked** project created from scratch:
`~/data/ue-projects/SurvSim`, plugins PythonScriptPlugin, SequencerScripting,
MovieRenderPipeline, EditorScriptingUtilities, plus the MetaHuman set.

## What is rendered

A minimal scene built programmatically (`scripts/ue/make_test_scene.py`):

* `Plane` scaled 40× as ground (40 m), `/Engine/BasicShapes/Plane`
* 3 × `Cube` scaled (0.5, 0.5, 1.8) as person stand-ins, at z = 90 cm
* `DirectionalLight`, intensity **12**, `atmosphere_sun_light = True`
* `SkyLight`, intensity **3**, `real_time_capture = True`
* `SkyAtmosphere`
* `CineCameraActor` at (0, −800, 400), rotation (pitch −20, yaw 90, roll 0)
* A `LevelSequence` with a `MovieSceneCameraCutTrack` bound to the camera and a
  transform track animating one cube

Verified after saving by reloading the map in a fresh editor: **16 actors, all
present, correct labels**, sequence has 2 bindings and the expected tracks.

## The symptom, precisely

Both outputs come from the same render, same frame:

```
PNG  (8-bit RGBA): R min=0 max=0 | G min=0 max=0 | B min=0 max=0 | A = 255
EXR  (32-bit linear, pre-tonemap):
     R min=0.000000 max=0.000000 mean=0.000000 nonzero=0     (of 921600)
     G, B identical
     A min=1.000000 max=1.000000 mean=1.000000 nonzero=921600
```

**The EXR is the key datum.** It is written *before* tonemapping and exposure,
and it is exactly zero. So this is not an exposure or tonemapper problem. Alpha
being exactly 1.0 everywhere shows the accumulator ran and wrote a full-coverage
buffer — the colour simply never arrives.

Movie Render Queue's own log reports a fully normal render:

```
Generated new ShotInfo for Inner: cam_a Outer: no shot
Initializing Camera Cut [1/1] in [no shot] cam_a.
Initialization set state to WarmingUp due to having 32 warm up frames.
Finished setting up rendering for shot. Shot has 1 Passes.
  Total resolution: (1280x720) Individual tile resolution: (1280x720). Tile count: (1x1)
Finished processing Camera Cut [1/1].
Finished rendering last shot. Moving to Finalize to finish writing items to disk.
MoviePipelineLinearExecutorBase finished 1 jobs in +00:00:04.369.
```

Exit code 0. 50 files, correctly named, correct resolution. No error or warning
anywhere in the log relating to rendering.

## What has been ruled out

Each of these was tested and **did not change the pixels**:

| Hypothesis | Test | Result |
|---|---|---|
| No display / swapchain | `xvfb-run -a -s "-screen 0 1920x1080x24"` | still black |
| Wrong feature level | Added `+TargetedRHIs=SF_VULKAN_SM6`; log confirms SM6 | still black |
| Uncompiled shaders | Ran `-run=DerivedDataCache -fill` (DDC 0 → 8.2 MB); also ran from the **editor**, which compiles on demand and was observed compiling `TBasePassPSFNoLightMapPolicy` / `TBasePassVSFNoLightMapPolicy` | still black |
| `-game` mode lacking editor services | Rendered from inside the editor with `MoviePipelinePIEExecutor` on a deferred tick | still black |
| No lighting | Explicit intensities, atmosphere sun light, real-time sky capture | still black |
| Exposure / tonemapping | 32-bit **linear EXR** output | still black (exactly 0.0) |
| Camera not bound | MRQ log resolves `cam_a` and initialises the camera cut | n/a — binding works |
| Empty or broken scene | Reloaded saved map, enumerated 16 actors with correct labels | n/a — scene is correct |
| NVIDIA Optimus / no display on the discrete GPU | Forced the **Intel** GPU via `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.json`; log confirms `DeviceName: Intel(R) UHD Graphics (CML GT2)` | **still black** |
| Stale actors / duplicate lights | Map and sequence assets deleted before each rebuild | still black |
| Insufficient warm-up | `engine_warm_up_count = 32`, `render_warm_up_frames = True` | still black |

**Both GPUs producing identical black output is the most informative
elimination** — it makes a vendor driver bug unlikely and points at something in
the engine's offscreen render path or this project's configuration.

## Possibly relevant

* A **packaged** UE 5.7 build (UnrealZoo) rendered real images on this same
  machine earlier in the day via `-RenderOffscreen`. Later the same binary began
  hanging at frame 1 and never recovered across five clean restarts. Whether
  that is related is unknown.
* CUDA compute on the RTX 2070 works fine (PyTorch 2.11 + cu128, verified with a
  GPU matmul), so the card is functional — but that is a compute path, not
  graphics.
* The cook commandlet reports `Success - 0 error(s)` while producing **zero**
  cooked assets, with or without `-map=` and `-CookAll`. That may be a separate
  problem or a clue.

## Exact reproduction

```bash
# 1. Build the scene
/opt/unreal-engine/Engine/Binaries/Linux/UnrealEditor \
    ~/data/ue-projects/SurvSim/SurvSim.uproject \
    -run=pythonscript -script=scripts/ue/make_test_scene.py -unattended -nosplash

# 2. Render from the editor
xvfb-run -a -s "-screen 0 1920x1080x24" \
  /opt/unreal-engine/Engine/Binaries/Linux/UnrealEditor \
    ~/data/ue-projects/SurvSim/SurvSim.uproject \
    -ExecCmds="py /abs/path/scripts/ue/render_in_editor.py" \
    -unattended -nosplash -nosound

# 3. Check the PIXELS, not the exit code
.venv/bin/python tools/check_render.py /tmp/survsim_render_editor
```

Relevant files: `scripts/ue/make_test_scene.py`,
`scripts/ue/render_in_editor.py`, `scripts/ue/survsim_executor.py`,
`scripts/ue/DefaultEngine.ini.template`, `tools/check_render.py`.

## The question

**Why does UE 5.8.2's deferred renderer emit zero radiance into the Movie Render
Queue accumulator, on Linux, headless, on two different GPUs, with shaders
compiled and a correctly lit and populated scene — while reporting success?**

Specific things a reader may know that we do not:

1. Does UE 5.8 offscreen rendering on Linux require something beyond
   `SF_VULKAN_SM6` — a `DefaultGraphicsRHI` entry, a `-vulkan` flag, or an
   engine ini setting we have not set?
2. Is `MoviePipelineDeferredPassBase` the wrong pass for a headless Linux run,
   and should `MoviePipelineImagePassBase` or the newer Movie Graph pipeline be
   used instead?
3. Does the accumulator require an explicit `bAccumulatorIncludesAlpha`, a
   specific `MoviePipelineConsoleVariableSetting`, or a post-process volume to
   receive colour?
4. Is a Blueprint-only project missing a renderer module that a C++ project
   would compile, so the base pass silently produces nothing?
5. Is there a known issue with the NVIDIA **open** kernel module (610.43.03) and
   UE's Vulkan offscreen path — and would that also explain the Intel result, or
   is the Intel result evidence against it?

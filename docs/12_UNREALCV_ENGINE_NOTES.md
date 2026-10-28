# 12 · UnrealZoo / UnrealCV+ engine notes

Hard-won operational knowledge about the packaged build. **Read this before
touching the engine** — most of it was discovered by hitting the failure first.

Status tags: **[V]** verified on this machine · **[A]** API exists, not yet
exercised · **[X]** tried and does not work.

---

## 1. What the build actually is

A **packaged (cooked) UE 5.7 runtime**, not the editor. 101 maps and all assets
live inside `.pak` files; `UnrealCV+` is compiled in as a TCP server.

```
~/data/unrealzoo/extracted/UnrealZoo_UE5_7_Linux_v3.1.0/Linux/
  UnrealZoo_UE5_7.sh                       <- launcher (chmod +x needed)
  UnrealZoo_UE5_7/Binaries/Linux/
      UnrealZoo_UE5_7                      <- 294 MB executable (chmod +x needed)
      unrealcv.ini                         <- create this to pin the port
  UnrealZoo_UE5_7/Content/Paks/            <- 68 GB of cooked content
  UnrealZoo_UE5_7/Plugins/unrealcv/
```

**[V]** Zip permissions are not preserved — both the `.sh` and the binary need
`chmod +x` after extraction.

**[V]** `unzip` refuses the 72 GB archive with *"invalid zip file with overlapped
components (possible zip bomb)"* and stops part-way. Use **`bsdtar -xf`**, which
reads the zip64 archive correctly.

---

## 2. Launching

```bash
cd ~/data/unrealzoo/extracted/UnrealZoo_UE5_7_Linux_v3.1.0/Linux
export UnrealEnv=$PWD
./UnrealZoo_UE5_7.sh /Game/JapanTrainStation/Maps/JapanTrainStation_Optimised \
    -RenderOffscreen -nosound -ResX=1280 -ResY=720
```

* **[V] Pass the map on the command line.** Booting straight into a level works
  and takes ~40 s.
* **[X] Do not switch levels at runtime.** `vset /action/game/level <Map>` loads
  the map but then **wedges the command loop** — every later request, even
  `vget /level/name`, hangs until the process is killed. `UnrealCv_API.set_map()`
  additionally calls `init_map()` immediately, before streaming finishes, and
  fails with `could not convert string to float: 'error'`.
* **[V]** GUI mode: drop `-RenderOffscreen` (add `-WINDOWED`).
* **[V]** Vulkan is the only system requirement — `vulkan-icd-loader` +
  `nvidia-utils`. No engine build, no editor.

Use `scripts/engine.sh {start|stop|restart|status|wait}` rather than driving it
by hand; it encodes the traps in §8 and the health distinction below.

### [X] The build now hangs at frame 1 (2026-09-18)

After running fine for ~7 h, every subsequent start of
`JapanTrainStation_Optimised` **stops advancing at frame `[ 1]`** and never
executes a command. Confirmed across five clean restarts, including one left
untouched for 40 min to rule out shader compilation. The process is alive and
busy — `R` state, 150% CPU, 4.6 GB VRAM on the 8 GB card, Vulkan correctly on
the RTX 2070 — it simply never reaches frame 2. Cause not established.

**Three different notions of "alive", and they disagree.** This cost hours:

| Check | What it proves | Fooled by |
|---|---|---|
| port is listening | a socket is bound | a hung process holds the socket open |
| handshake received | the **network thread** is alive | a game thread stuck at frame 1 |
| a command returns | the **game thread** is executing | nothing |

UnrealCV answers its handshake from a network thread while commands run on the
game thread, so a stuck build looks healthy to the first two checks. Only issue
a real command. `engine.sh status` does, and reports `STARTING` versus `WEDGED`
by whether the process is still burning CPU — "needs more time" and "needs a
restart" are different answers and guessing wrong wastes either minutes or an
hour.

Also: UnrealCV serves **one client at a time**, and a client killed with
`kill -9` leaves that slot occupied. Repeated connection attempts after a
failed probe make the situation worse, not better.

### Port pinning (important)

**[V]** UnrealCV defaults to 9000 and **auto-increments** when the port is taken.
On this machine 9000/9001 are MinIO (docker-proxy) and Chrome grabs localhost
ports, so the server landed on 9002 and Chrome then occupied its connection slot.
Pin it instead — create `UnrealZoo_UE5_7/Binaries/Linux/unrealcv.ini`:

```ini
[UnrealCV.Core]
Port=9900
Width=1280
Height=720
FOV=90.000000
EnableInput=True
EnableRightEye=False
```

**[V] One client at a time.** The listen backlog is 1. A second connection blocks
until the first disconnects, and a killed client leaves the slot occupied until
the socket times out. Symptom: `Connection timed out` / `socket is closed` while
the engine is plainly alive. Check with `sudo ss -tnp | grep 9900`.

---

## 3. Python client

```python
from unrealcv import Client                 # raw command interface
from unrealcv.api import UnrealCv_API       # convenience wrapper (81 methods)
from unrealcv.plus_api import UnrealCvPlusAPI   # UnrealCV+ (337 methods)
```

**[V]** `from unrealcv import client` (lowercase singleton) does **not** exist in
1.3.2 — use the `Client` class.

**[V] Prefer the binary-return form** for images:

```python
data = c.request("vget /camera/0/lit png", timeout=180)   # bytes
img  = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
```

The file-path form (`vget /camera/0/lit /abs/path.png`) also works but writes
**asynchronously** — the file may not exist when the call returns. (An earlier
note in this repo claiming it "writes nothing" was wrong.)

---

## 4. Capture — verified

| Command | Result |
|---|---|
| `vget /camera/0/lit png` | **[V]** 1280×720×4 uint8, photoreal |
| `vget /camera/0/object_mask png` | **[V]** instance mask, 201 distinct colours in a full scene |
| `vget /camera/0/depth npy` | **[V]** float32 HxW, metres |
| `vget /camera/0/normal png` | **[V]** surface normals |
| `vget /object/<name>/bones` | **[V]** JSON skeleton (per-bone Pitch/Yaw/Roll/X/Y/Z/Scale) |
| `vget /object/<name>/color` | **[V]** the actor's mask colour, for mask→instance lookup |

**[V] UnrealCV issue #339 does NOT reproduce here.** Skeletal meshes with
`DisableSKMAnnotation: false` render instance masks without crashing on 5.7.
The `D08` contingency (writing our own stencil pass) is therefore **not needed**.

**Encoding is the bottleneck**, not rendering: the engine log reports
`serialize_image | format=png | encode_ms=71` per frame, plus ~17 ms render.
Roughly 10 fps per modality over TCP. For bulk capture, reduce resolution or use
the shared-memory / fast-capture paths (`get_camera_*_shared`, `set_camera_use_fast_capture`).

---

## 5. Making people look and behave different — the key capability

The default `BP_Character` is **one mesh in one outfit**, so naively spawning N
actors gives N identical clones. That is useless for training a detector. The
build solves this through **UnrealCV+ MetaHuman parametric commands**:

| Capability | Command | Status |
|---|---|---|
| list available MetaHumans | `vget /metahuman/all_paths` | **[A]** |
| create a parametric human | `vset /metahuman/parametric/create [str][str][str][f][f][f][str][str]` | **[A]** |
| **body shape** (build/height) | `vset /metahuman/parametric/body [id][preset][f][f][f]` | **[A]** |
| **clothing** | `vset /metahuman/parametric/wardrobe [id][slot][item]` | **[A]** |
| finalise the character | `vset /metahuman/parametric/assemble [str][str][str]` | **[A]** |
| rig readiness | `vget /metahuman/parametric/rig_status [id]` | **[A]** |
| hair physics | `set_object_hair_gravity`, `set_object_hair_airdrag` | **[A]** |

This maps directly onto `survsim.core.appearance.AppearanceState`
(`outfit` → wardrobe, `body` → parametric body, plus appearance epochs).

### [X] Those commands do not work in v3.1.0 — and the editor is the answer

Measured, not assumed: the packaged server **rejects every one** of the wardrobe
and animation commands above. `animation/soma_glb/apply` returns
`attached=false playing=false` for all 40 candidate clip names, and the
parametric wardrobe commands are absent from the server despite the Python
*client* exposing them. The client (v1.3.2) is simply ahead of any released
binary — HuggingFace's newest build is v3.1.0, and upstream `unrealcv` releases
(v0.1.0–v1.1.2) contain no MetaHuman commands at all.

**So this table is a record of a dead end.** The route that works is the editor.

### [A] Unreal Editor 5.8 exposes the whole parametric API to Python

Verified by introspection inside the editor (2026-09-18), no UnrealCV involved:

```python
sub = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
sub.get_body_constraints(character)   # -> Array[MetaHumanCharacterBodyConstraint]
sub.set_body_constraints(character, constraints)
sub.commit_body_state(character)
sub.build_meta_human(character, params)
```

| Our field | Editor API |
|---|---|
| `body.height_m`, `build` | `get_/set_body_constraints`, `commit_body_state` |
| `outfit` (garments) | `MetaHumanWardrobeItem` + `MetaHumanCharacterPalette.try_add_item_from_wardrobe_item` |
| skin / makeup / eyes | `commit_skin_settings`, `commit_makeup_settings`, `commit_eyes_settings` |
| finalise | `build_meta_human`, `assemble_for_preview` |

`MetaHumanWardrobeItem` carries a `principal_asset` (the mesh) and a `pipeline`
that builds it. The subsystem instantiates successfully, and 74 of its methods
carry signatures.

**The plugin set is the whole trick.** With none enabled, a project exposes
**26** MetaHuman symbols and no wardrobe API; enabling `MetaHumanCharacter`,
`MetaHumanCharacterUAF`, `MetaHumanGenerator`, `MetaHumanCrowd`,
`MetaHumanRuntime`, `MetaHumanSDK` and `RigLogic` gives **272**. We had been
reading the absence of an API in a *packaged runtime* as evidence the feature
did not exist anywhere. `scripts/ue/SurvSim.uproject.template` records the set;
`scripts/ue/probe_api.py` re-runs the introspection.

### [V] The parametric body works — measured

Verified on UE 5.8.2 (2026-09-18). `Height` driven **171.75 cm → 190.0 cm**,
persisted through `commit_body_state`, read back correctly. 30 named constraints
exist on a default character:

> Height, Chest, Bust Span, Underbust, Waist, Rise, Inseam, Hip, High Hip,
> Neck Base, Neck, Thigh, Knee, Calf, Front Interscye Length, Upper Arm Length,
> Lower Arm Length, Bicep, Elbow, Wrist, Neck to Waist, Across Shoulder,
> Shoulder Height, Shoulder to Apex, Neck Length, Forearm, Hand Circumference,
> **Fat**, **Masculine/Feminine**, **Muscularity**

Measurements are centimetres; the last three are 0–1 shape parameters. Height
accepts 135–220 cm. `Masculine/Feminine` matters to us specifically: presenting
as a different gender is a documented evasion technique and defeats
appearance-based ReID far more thoroughly than a change of coat.

```python
sub = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
ch  = tools.create_asset(..., asset_class=unreal.MetaHumanCharacter,
                         factory=unreal.MetaHumanCharacterFactoryNew())
sub.try_add_object_to_edit(ch)          # MANDATORY — see below
cons = list(sub.get_body_constraints(ch))
h = next(c for c in cons if str(c.get_editor_property("name")) == "Height")
h.set_editor_property("target_measurement", 190.0)
h.set_editor_property("is_active", True)   # `is_active`, NOT `active`
sub.set_body_constraints(ch, cons)
sub.commit_body_state(ch)
```

Two traps, both silent:

* **`try_add_object_to_edit` is mandatory.** Without it `get_body_constraints`
  returns `[]` and `commit_body_state` does nothing, logging only *"not added
  for editing"*. An empty list reads exactly like "this character has no
  parameters", which is why the first attempt concluded the API was a stub.
* **A mistyped constraint name does not raise.** It leaves that measurement at
  its default while ground truth records what was asked for. Always read the
  values back rather than trusting the write.

### [X] Auto-rigging needs an Epic account — and this is a hard blocker

`can_build_meta_human` returns False with *"Character is not rigged"*. The fix is
`request_auto_rigging(character, [rig_type, report_progress, blocking])`, which
does start ("Auto-Rigging Face Mesh") — and then blocks forever on an **Epic
OAuth device-code flow**:

```
ErrorCode=[errors.com.epicgames.account.oauth.authorization_pending]
"The authorization server request is still pending as the end user has yet
 to visit and enter the verification code."
```

Auto-rigging is a **cloud service**. It polls every 10 s for a login that cannot
happen under `-unattended`, and the verification URL is redacted from the log.

**Required user action, once:** launch the editor with a GUI, sign in to Epic
(MetaHuman/Quixel), and let it store credentials.

### [V] Rigging does NOT actually need the cloud — DNA import works offline

The cloud autorig is avoidable. The engine ships **23 DNA files**, and importing
them rigs a character locally with no login:

```python
sub.import_from_face_dna(ch, f"{ARCH}/SKM_Face.dna", unreal.ImportFromDNAParams())
sub.import_body_whole_rig(ch, f"{ARCH}/SKM_Body.dna", f"{ARCH}/SKM_Face.dna")
# both return ImportErrorCode.SUCCESS
```

* `…/MetaHumanCoreTechLib/Content/ArchetypeDNA/` — `SKM_Body.dna` (4.2 MB),
  `SKM_Face.dna` (11.2 MB), `body_head_combined.dna` (6.7 MB)
* `…/MetaHumanCharacter/Content/Optional/Body/FixedCompatibility/` — **18 body
  archetypes** on a 2×3×3 grid: `{f,m}_{srt,med,tal}_{unw,nrw,ovw}`
  (sex × short/medium/tall × under/normal/overweight). This maps directly onto
  the population generator's body diversity.

**Proof it worked:** the build error changes from *"Character is not rigged"* to
*"The Character is missing textures"*. The rig gate is passed.

Skin also synthesises locally — `commit_skin_settings` succeeds using the
**941 MB** texture-synthesis model at
`…/MetaHumanCharacter/Content/Optional/TextureSynthesis/`.

### [X] The remaining gate is source textures, and it is authenticated

`request_texture_sources` logs, unambiguously:

```
Error: User not logged in, please autorig before downloading source face textures
```

`blocking=True` does not help — it is a download, not a computation. So
`build_meta_human` still cannot complete offline, one gate later than before.

**What this changes for us:** the blocker is now *cosmetic* rather than
structural. Source textures are 2K face albedo/normal/cavity maps — detail that
is irrelevant at the 20–50 m standoff a surveillance camera works at, where a
person is tens of pixels tall. The rig, the skeleton, the parametric body and
the skin synthesis — everything that governs *shape and motion* — is available
offline. A one-time GUI sign-in still unlocks the full path and remains the
simplest route.

### [V] The editor reports success for scripts that did nothing

`-run=pythonscript` prints **"Python script executed successfully"** for a script
that produced no effect and no log output at all. The first probe "succeeded"
exactly that way. In-editor scripts must **write a result file**; stdout and exit
codes are not evidence.

### Movement, gait and stationary behaviour

| Behaviour | Mechanism | Status |
|---|---|---|
| walk a real path (engine drives locomotion + foot IK) | `vset /agent/<name>/nav/goto <x> <y> <z>` | **[A]** |
| start / stop navigating | `vset /agent/<name>/nav/start <f>`, `vset /agent/<name>/nav/stop` | **[A]** |
| is it still walking? | `vget /agent/<name>/nav/status` | **[A]** |
| **arbitrary mocap animation** (gait style, idle, fidget, look around, aggression) | `vset /animation/soma_glb/apply <actor> <glb>` | **[X] present but non-functional** — see below |
| stop the clip | `vset /animation/soma_glb/stop <actor>` | **[A]** |
| built-in discrete actions | `stand, jump, crouch, open_door, enter_vehicle, pickup` | **[V]** in config |
| head turn (look around while stationary) | `head_action` in the gym config: `[[0,0,0],[0,30,0],[0,-30,0]]`, continuous ±15° | **[A]** |
| reachable navmesh points | `vget /reachablepoints <x> <y>` | **[A]** |

### Animation: the command exists but does nothing (measured)

`tools/probe_animations.py` spawned an actor and tried **40 clip names** across
every plausible convention. **All 40 were rejected**, and
`vget /animation/soma_glb/status` reported `attached=false playing=false
duration=0.000`.

The name is the explanation: `soma_glb` wants a **GLB animation asset**, not a
clip name, and this build ships none. So gait variation is *not* available out of
the box — a capability check that only tested for the command's presence would
have reported working gait while every person walked identically.

**Route forward:** supply our own GLB mocap (Mixamo exports, AMASS/SMPL
conversions) and pass a file path. Untested, and the accepted argument form still
needs establishing.

**Lesson applied in code:** `Capabilities.animation` now means *functionally
verified* — apply a clip, then confirm the status says it is playing.
`animation_command_present` records mere presence separately.

**Stationary behaviours would therefore be expressible**: loitering = `nav/stop` +
an idle/fidget GLB clip + periodic head rotation; "looking around without
walking" = head yaw steps while the body plays a standing clip. Gait variation
(limp, brisk, slow, crouch-walk) comes from applying different GLB clips rather
than from a locomotion parameter.

**Teleporting actors each tick (what `UnrealBackend.sync` currently does) is
wrong** for realism — it slides characters without leg animation. Migrate to
`nav/goto` and let the engine animate, keeping the Director as the planner.

---

## 6. Asset discovery

`vget /pak/files`, `vget /pak/assets <package_path>`, `vset /pak/scan <mount>`,
`vget /objects/scan_assets <name>` — **[A]**. This is how to enumerate the real
inventory of characters, wardrobe items and animation clips rather than guessing
from the gym's setting JSONs (which reference only **one** character asset).

Maps: 101, listed in `~/src/unrealzoo-gym/gym_unrealcv/envs/setting/Track/*.json`.
Surveillance-relevant ones: `JapanTrainStation_Optimised`, `VictorianTrainStation`,
`ParkingLot`, `UndergroundParking`, `ContainerYard_Day/Night`, `IndustrialArea`,
`Map_ChemicalPlant_1`, `Stadium`, `supermarket`, `DowntownWest`, `Tokyo`,
`Hospital`, `Museum`, `Hangar`, `Storagehouse`.

Each setting JSON carries `safe_start` (spawn points, **centimetres**),
`height` (ground Z in cm), and the agent asset paths.

---

## 7. Coordinate conventions

| | SurvSim | Unreal |
|---|---|---|
| units | metres | **centimetres** |
| handedness | right | left |
| yaw | CCW from +X | CW from +X |

Conversion lives in `survsim/backends/unreal/backend.py` (`to_ue_location`,
`to_ue_yaw`): scale ×100, **negate Y**, **negate yaw**. Engine units must never
appear above the backend boundary. Ground level is per-map (`height` in the
setting JSON, e.g. −931.85 cm for the train station).

---

## 8. Shell traps that cost real time here

* **`pgrep -f <pattern>` matches your own wrapper.** `pgrep -f UnrealZoo_UE5_7`
  matches the bash command containing that string, so a watcher built on it
  reports "running" forever, and `pkill -f` can kill your own shell. Use
  `pgrep -x UnrealZoo_UE5_7` / `pkill -x`.
* Likewise `pkill -f unrealcv` kills the Python process whose command line
  contains "unrealcv".
* Long Python runs need `python -u`; buffered stdout is lost when a timeout kills
  the process, which looks exactly like a hang with no output.
* `/tmp` is a 16 GB tmpfs — never download or extract there. Use `~/data/`.

---

## 9. Machine specifics

RTX 2070 Max-Q **8 GB VRAM**, 31 GB RAM, 12 cores, 768 GB free. Below the plan's
24 GB BOM, so: capture at 720p, round-robin cameras, ≤12 hero agents, depth at a
lower rate. Small UnrealZoo maps are fine; a full City Sample build is not.

## Unreal MCP — Epic's built-in server (2026-09-25) [V]

UE 5.8 ships an official `ModelContextProtocol` plugin (Experimental, prebuilt
for Linux) — no third-party plugin needed. Enabled in the SurvSim `.uproject`
together with `ToolsetRegistry`, `EditorToolset`, `AnimationAssistantToolset`,
`PCG` and `PCGToolset`.

* `scripts/ue_mcp.sh start|stop|status|log` runs the editor headless under
  `xvfb-run` with `-ModelContextProtocolStartServer`.
* Streamable HTTP at `http://127.0.0.1:8127/mcp`, registered in `.mcp.json`.
  **Not the default 8000** — `fc_api` holds 8000 on this machine.
* The editor's HTTP server binds to *all interfaces* by default; the project's
  `DefaultEngine.ini` pins `[HTTPServer.Listeners] DefaultBindAddress=127.0.0.1`.
  Keep that: this endpoint can run arbitrary editor operations.
* Tool-search mode: the server exposes `list_toolsets`, `describe_toolset` and
  `call_tool`; 19 toolsets load, including actor/scene/asset/material/skeletal
  mesh, PCG, logs, a sandboxed Python tool, and a **MetaHuman** toolset.
* Verified by a real `initialize` → `tools/list` → `list_toolsets` round trip,
  not by the port being open (see §2 on why an open port proves nothing).
* Config: port/path/autostart are in the project's
  `Config/DefaultEditorPerProjectUserSettings.ini`
  (`[/Script/ModelContextProtocolEngine.ModelContextProtocolSettings]`).

## Animated humans in the editor render (2026-09-25)

* **[V]** `~/data/glb/CesiumMan.glb` imports through Interchange
  (`scripts/ue/import_gltf.py`, plain `AssetImportTask`) as SkeletalMesh +
  Skeleton + a 2.0 s walk `AnimSequence` + a `MI_Default_Opaque` material
  instance. No login. Mesh is 150.6 cm tall, Z-up, feet at 0, faces +Y (render
  applies yaw offset -90 and scales to 175 cm). RiggedFigure.glb imports too
  (1.25 s clip).
* **[V]** `SkeletalMeshActor` + `MovieSceneSkeletalAnimationTrack` sections render
  a walking person in MRQ; frame 20 of cam_01 shows three walking figures.
* **[X]** `params.set_editor_property("play_rate", 1.2)` — in 5.8 PlayRate is a
  struct (FMovieSceneTimeWarpVariant). `render_run._set_play_rate` tries the
  alternatives (one was accepted in the 2026-09-25 render: no warning logged; effect not measured) and falls back to 1.0 with a log warning.
* **[X]** Importing `survsim` inside the editor: its embedded Python has no
  `yaml`. The animation plan is computed host-side (`survsim/export/anim_plan.py`)
  and passed as `/tmp/survsim_render_plan.json`.
* **[X]** `/Engine/EngineMaterials/Grid` and `DefaultDiffuse` are checkerboards;
  used as ground they render a chessboard. The ground is flat grey for now.
* An unkeyed scale channel on a transform section resets the actor to scale 1:
  the render keys scale explicitly.
* Guard: the render raises if any person lacks an animation track
  (`tests/test_unreal_anim_plan.py`).
* **[V] detector acceptance (yolov8n COCO, conf 0.25, 5 frames/camera, run
  out/render_demo, 40 frames):** cam_01 persons per frame 0,0,1,1,3 (3 walkers
  in view); cam_02 0/5 — its people are ~40 px tall at ~25 m, below what
  yolov8n finds on this stylised mesh. Cross-camera mean abs diff 63.7 (views
  differ); first/last frame diff > 0 on both cameras (people move).
  Next: larger/closer standoff or a more realistic mesh; ground texture.

## Automated render gate (2026-09-25) [V]

Nobody should have to eyeball frames to find a failure this project has already
had. After every render run:

```bash
scripts/check_render.py <render_dir> [--gt <run_dir>] [--log <editor.log>] [--json r.json] \
    [--expected N] [--frame-offset 0] [--detector --min-recall 0.5]
scripts/check_job_result.py /tmp/survsim_jobs/<name>.result.json   # in-editor job pass/fail + traceback
```

Exit is non-zero if any check FAILs; the table names it. Code: `survsim/checks/`
(each check a pure function), tests `tests/test_checks_render.py` (each failure
staged synthetically). Frames are read at 640 px wide; GT boxes are scaled.

| check | fails when | guards |
|---|---|---|
| `frame_count` | no frames, cameras disagree, `--expected` mismatch, index gaps | partial MRQ output |
| `<cam>:blank_frames` | luma mean < 8 or std < 4 | MRQ "success=True", all-black PNGs |
| `<cam>:motion` | > 50 % of consecutive pairs identical, or first == last | frozen sequence / no playback |
| `views_differ` | min pairwise mean-abs-diff of camera mean frames ≤ 5 | missing `CameraCutTrack`: every camera the same view |
| `<cam>:exposure` | median-frame luma outside 20–235, or > 25 % clipped | blown / crushed lighting |
| `<cam>:horizon` | bottom quarter > 60 % sky (blue-bright or flat-bright) | inverted pitch |
| `<cam>:gt_boxes_in_image` | any MOT box > 2 px outside the image | projection / distortion drift |
| `<cam>:gt_person_rendered` | < 60 % of boxes ≥ 20 px differ (> 6 grey levels) from the temporal-median plate | people not spawned / not visible where GT says |
| `<cam>:animation` | < 50 % of ids change their box-normalised crop (> 3) between t and t+5 | person missing an animation track (slides or freezes) |
| `<cam>:gt_count_vs_world` | more boxes in a frame than people in world.parquet | GT duplication |
| `<cam>:gt_frame_coverage` | > 5 % of image indices (+ offset) absent from world.parquet | render/GT frame misalignment |
| `log_python_error`, `log_fatal` | any `LogPython: Error` / `Fatal` | silent in-editor exceptions |
| `log_type_convert` | WARN on `Failed to convert type` | Python↔UE struct mismatches (PlayRate) |
| `realism_band` | INFO only: stats in the docs/15 real band | degrade profile is upstream |
| `detector_recall` | yolov8n recall @IoU .5 < `--min-recall` (INFO if unset) | renders a detector cannot use |

Known limits: the "plate" is a temporal median, so a person who never moves is
part of it and reads as not-rendered; `animation` on a very textured background
can pass a rigid slide. Both are conservative in the direction of flagging.

**What it caught on first run (2026-09-25):**

* `out/render_demo` (2 × 120 frames, 12.1 s): FAIL `gt_person_rendered` on both
  cameras (3 % / 0 % of GT boxes sit on anything but empty ground — median diff
  0.2) and `animation` (static ids). The PNGs are the 2026-09-18 capture with
  grey cubes and **no people**, while `gt/` is from a later run: the directory's
  frames and GT do not describe the same render. Every other check passes
  (views differ by 8.5 — low, cameras share a flat ground plane). Realism 1/13.
* `out/demo_humans` (3 sample frames, 1.1 s): FAIL `frame_count` — not a
  sequence (cam_01 has 2 frames with a 19-frame gap, cam_02 has 1). Views differ
  by 63.7, exposure and horizon pass. Realism 3/13.
* `/tmp/survsim_jobs/*probe_rate.result.json`: PASS with a WARN — 6 `err_*`
  keys (PlayRate variant probe) that `ok: true` would otherwise hide.

Pre-commit: `scripts/hooks/install.sh` installs `scripts/hooks/pre-commit`
(ruff on staged `.py`, compileall, `pytest -n auto -x -q tests/test_checks*.py`;
~3 s). Not auto-installed — the repo has ~226 pre-existing ruff findings, so the
hook lints staged files only.

## Iterating: the persistent editor + job runner (2026-09-25) [V]

**This is the way to iterate.** Cold-starting the editor per render cost ~2 min
of boot + shader warm-up; the job runner removes it.

* `scripts/ue_mcp.sh start` boots ONE headless editor (xvfb) with the MCP
  server *and* `scripts/ue/job_runner.py` (`-ExecCmds="py …"`). The runner's
  slate tick polls `/tmp/survsim_jobs/*.py` every 0.5 s, `exec()`s each job with
  `unreal`, `RESULT`, `finish`, `guard` in scope, writes
  `<job>.result.json` (traceback on failure) and renames the job `.done`.
* `tools/ue_job.py job.py --wait 180` submits and waits **bounded**, failing
  early on a new `LogPython: Error` line in any `Saved/Logs/SurvSim*.log`
  (the persistent editor logs to `SurvSim_2.log` if another instance held
  `SurvSim.log`) or on a dead editor pid (runner heartbeat file
  `/tmp/survsim_jobs/heartbeat`, `<time> <pid>`, every 5 s).
* Async jobs (MRQ renders, delayed captures) set `RESULT["async"]=True` and call
  `finish({...})` later; wrap callbacks in `guard(fn)` so an exception finishes
  the job as failed instead of leaving the runner busy (a 30 min busy timeout is
  the backstop). **[X]** an unguarded callback exception wedged the runner.
* Jobs are **incremental**: actors live in outliner folders `Env`, `Lights`,
  `People`, `Cams`, `Tmp`; `survsim_scene.clear(group)` removes one group.
  Materials/instances/imports are cached by name (`M_SurvProc_v<N>`: bump the
  version to force a shader rebuild; instance edits never recompile).
* Measured round trips: change material + 10 frames × 1 camera **7 s**; 40
  frames × 2 cameras with an env rebuild **19 s**; 300 frames × 2 cameras
  **52 s** (≈ 11 frames/s at 720p on the 2070 Max-Q). Iterate at 10 frames /
  1 camera, then render the full set.
* **[V] `get_folder_path()` returns the Name `"None"`** (truthy string) for
  unfoldered actors — compare against `"None"`, not falsiness.
* **[X] A native crash takes the batch with it**: importing 22 glTFs in one job
  segfaulted the editor (crash dirs in `Saved/Crashes`). Import one asset per
  job (`tools/import_characters.py`); the catalogue is merged after each.

### Editor lifecycle: OOM and restart (2026-09-27)

The persistent editor is **CPU-side RAM**, not VRAM: it was OOM-killed at 14:40
by 11 CPU workers + a QAT run sharing the box with it, not by anything on the
2070. Before restarting:

* `free -g` first — restart only when **≥ 8 GB free**; on a 31 GB box with no
  swap, a restart into an already-tight box just gets killed again.
* Restart via `scripts/ue_mcp.sh start` / `stop` (never raw `pkill -f`).
* The job runner writes a heartbeat file `/tmp/survsim_jobs/heartbeat`
  (`<time> <pid>`, every 5 s); `tools/ue_job.py` treats a stale heartbeat as a
  dead editor before treating a job as failed — check it before assuming a
  hang is the job's fault rather than the editor having been killed under it.

### batch_v8 recipe still current

`tools/render_run.py` matches the batch_v8 recipe above: CC0 Quaternius is the
default (`SURVSIM_CHARS` unset ⇒ `mode = "cc0"`), MetaHuman is the explicit
opt-in (`SURVSIM_CHARS=metahuman`), and `SURVSIM_OUTFIT_OVERRIDE` (env var,
checked at line ~404) still hooks in a wardrobe override path per run.

## Street scene, materials, lighting (2026-09-25)

* **[V] Procedural PBR without textures**: `scripts/ue/survsim_materials.py`
  builds one master material from a Custom HLSL node driven by *world position*
  (metres) and `VertexNormalWS` — asphalt, 0.6 m slabs with grout, 0.2 m setts,
  running-bond brick with mortar, panelled concrete, grass/hedge, paint — plus a
  facade window grid (shopfront ground floor, framed panes, glass roughness 0.06)
  whose panes emit at night (`Night`, `WinEmissive`). No UVs needed on the
  engine's basic shapes; texel density is identical on a bin and a building.
  `MaterialEditingLibrary.create_material_expression(mat, MaterialExpressionCustom)`
  + `CustomInput(input_name=…)` + `connect_material_expressions(src, "", node, name)`.
* **[A] ambientCG texture path** (`pbr_instance`, world-aligned albedo/normal/
  roughness master `M_SurvPBR_v<N>`) is written but not yet exercised; the
  textures are in `~/data/assets/textures/ambientcg`.
* **Street** (`survsim_scene.build_street`): plaza slabs sized to the track
  envelope + 3 m, setts band, two-lane road with kerbs, centre dashes, yellow
  edge lines and a zebra, parked box-cars, 20+ lamp posts, benches, bins,
  bollards, trees, wall/hedge/railings, 3 facade rows and a distant 24-building
  ring so the horizon is never empty. Deterministic from the scenario seed.
  Clutter within 8 m in front of any camera is skipped.
* **[V] Fixed exposure**: `AEM_MANUAL` with physical-camera exposure OFF and
  `auto_exposure_bias = -EV100`. **[X]** histogram with min=max EV100 is ignored
  by SceneCapture2D; manual+bias works in both MRQ and captures. Day: sun
  40 000 lux, EV100 12.5; night: 0.3 lux cool moon, 15 000 lm 3400 K lamp spots,
  EV100 1.5, lit windows.
* **[X] Sky light `real_time_capture`** leaves shadows black in editor scene
  captures; a captured sky light + `recapture_sky()` works. MRQ has GI either way.
* **[A] HDRI**: `light(hdri=path)` imports a Poly Haven .hdr as a TextureCube for
  the sky light; not yet verified in a render.

## Characters (2026-09-25)

* **[V] PlayRate**: 5.8's `FMovieSceneTimeWarpVariant` has
  `set_fixed_play_rate(r)` / `to_fixed_play_rate()`; the render reads each rate
  back and reports mismatches (0 of all sections). Rate = speed / clip ground
  speed, clip ground speed = 1.45 m per cycle / cycle length, × height/1.75.
* **[V] CesiumMan** renders upright, grounded, walking, tinted per person.
* **[X] Quaternius (all 22, raw and armature-baked)**: import works, but in the
  render the walkers lie face-down (needs roll 90) and arms stretch from the
  shoulders to far-away hands. Mixamo-style rigs from the same library
  (`animated_human_*`) are upright after baking but show the same arm stretch.
  Established: armature node carries rot(-90 X)·scale 100 (Quaternius) or
  scale 69 (Mixamo); anim and mesh share one skeleton; track names match bones
  (case-folded); mesh/actor bounds are ~100 m regardless (physics/bone bounds —
  **bounds cannot be used to auto-fit these rigs**). `survsim/export/gltf_normalize.py`
  bakes the armature rotation and folds the uniform scale into joint translations
  and inverse bind matrices (world-preserving, tested) — that fixed nothing in
  UE, so the fault is on the Interchange side. Renders therefore use an explicit
  allow-list (`SURVSIM_CHARS`, default CesiumMan). Next: FBX via a working
  Blender (the system Blender needs libavcodec.so.63; ffmpeg is at .62).

## MetaHuman route (2026-09-25)

Goal: a realistic human instead of CesiumMan/Quaternius, **without an Epic
login**. Sandbox: a copy of the project at `~/data/ue-projects/SurvSimMH`
(`SurvSimMH.uproject`, MCP autostart off), its own editor and job dir.

```bash
.venv/bin/python tools/mh_editor.py start|stop|status   # -nullrhi by default (MH_GPU=1 to render)
.venv/bin/python tools/mh_job.py scripts/ue/mh_bake.py --wait 175   # params: /tmp/survsim_jobs_mh/mh_bake.json
.venv/bin/python tools/mh_detect_compare.py --mh <render_dir> --ref out/batch_v1/render_demo_day
```

`tools/mh_job.py` also copies every new log line matching
`Error:|Insufficient|Failed to save|Assertion failed|Fatal` into the result as
`log_alerts`. Editor toasts such as "Insufficient writing permission" never
reach LogPython.

### What ships offline (in `/opt/unreal-engine/Engine/Plugins/MetaHuman`)
* **29 preset characters** (`MetaHumanCharacter/Content/Optional/Presets/{Ada…Zuri}`),
  body textures (`Optional/BodyTextures/T_Skin_V{1,2}_Body_BC` plus surface detail).
* **Walk/run/idle clips on the MetaHuman skeleton**:
  `Optional/Animation/UEFNAnimPreset/Locomotion/AS_MH_Neutral_{Walk,Run}_{Loop,Start,Stop}_*`,
  `…Stand_Idle_Loop`. They use the same `metahuman_base_skel` as the body, so
  **no IK retargeting is needed**.
* **One garment**: `Optional/Clothing/WI_DefaultGarment` (ChaosOutfitAsset
  `OA_DefaultGarment`, with 6 body-shape cloth variants). **Grooms**: about 60
  hair/beard/brow wardrobe items (`Optional/Grooms/Bindings/*/WI_*`).
  Nothing else: no jackets, bags or shoes.

### Results, tagged
* **[X] Preset + DNA rig**: `import_body_whole_rig` on a duplicated preset
  (Ada) logged "DNA vertices mismatch with the State after whole-rig import"
  and then **hung the game thread for more than 12 min** at 98 % CPU. Only a
  SIGKILL recovered it. Use a blank `MetaHumanCharacterFactoryNew` character.
* **[V] Blank character, rig offline**: face DNA takes 9 s and body whole-rig
  60 s (one-time), both returning SUCCESS. `can_build_meta_human` then fails
  with *"missing textures, use Download Texture Sources"*. This is the
  login-gated step. **[X] build_meta_human offline.**
* **[V] Body parameters**: Height, Fat, Masculine/Feminine and Muscularity
  read back exactly as set. Body-mesh bounds scale with stature: 192 cm gives
  159.9 cm and 158 cm gives 130.2 cm, both about 0.83, because the head is
  the separate face mesh.
* **[V] The preview is the offline character.** `spawn_meta_human_actor`
  gives a `MetaHumanDefaultEditorPipelineActor` with Body (342 bones), Face
  (875) and 6 groom components. **[X] animating it**: the preview pipeline
  pins the Body pose, and `play_animation`/`set_position` move no bone.
  **[X] borrowing its meshes**: they live in `/Engine/Transient` and are
  garbage-collected or replaced on re-assembly (an actor using one ended up
  with mesh None). Saving anything that references them raises "Insufficient
  writing permission to save /Engine/Transient/BodyMesh_N".
* **[V] Bake** (`scripts/ue/mh_bake.py`): this copies the preview Body/Face
  SkeletalMeshes, each transient MID (as a MaterialInstanceConstant with its
  parameters), and the outfit into `/Game/SurvSim/MetaHumans/Baked/<name>`.
  The saved packages reference only plugin content, never `/Engine/Transient`
  (checked with AssetRegistry dependencies). About 30 MB per body plus 8 MB
  per outfit. The body material is `M_GrayTexture_Body` with the offline
  `T_Skin_V2_Body_BC` albedo and underwear textures. The face uses
  grey/placeholder textures, because the synthesised face textures are
  transient and did not persist as files.
* **[V] Wardrobe (assembly)**: `get_preview_collection(ch)` →
  `try_add_item_from_wardrobe_item("Outfits", WI_DefaultGarment)` →
  `default_instance.set_single_slot_selection(slot, key)` →
  `assemble_for_preview` adds a `ChaosClothComponent`. The baked
  `ChaosOutfitAsset` attaches to a plain actor via
  `SubobjectDataSubsystem.add_new_subobject` plus `set_leader_pose_component`.
  **Not yet rendered, so the render is still [A].**
* **[V] Animation (bones, nullrhi)**: the baked body on a SkeletalMeshActor
  with `set_update_animation_in_editor(True)` + `play_animation(walk, True)`
  animates. `set_position(t)` seeks deterministically (requested 0.4 s steps
  read back exactly), and the feet alternate lifting 7 to 21 cm through the
  4 s loop. **[X] `set_play_rate(0)`**: it freezes the pose, so later seeks
  do nothing. Keep rate 1 and seek every frame.
* **[V] Variety in a loop**: after the one-time rig, each new body (body
  parameters + skin tone u/v + preview + bake + save) takes **15–17 s**,
  with 0 VRAM under `-nullrhi`.
* **[X] VRAM**: a GPU sandbox editor holding the street scene plus a
  MetaHuman preview reached **5.5 GB**, logged about 9 400 Vulkan "Failed to
  allocate Device Memory" warnings, and wrote white frames. It does not fit
  next to the production editor (about 2.2 GB) on the 8 GB card. Build under
  `-nullrhi`; render in the production editor.
* **Rendering in the scene: done 2026-09-26, see "MetaHumans in production".** `scripts/ue/mh_prod_render.py`
  captures a plate plus N frames from a level camera. It dresses the body
  with face and outfit, walks it, cleans up, and reports bone heights and the
  light list. A nullrhi dry run on RunMap is clean (1.7 s, bones move), but
  there are no pixels yet. It needs the baked set copied into the production
  project's `Content/SurvSim/MetaHumans/Baked/`.
* **Detector baseline**: yolov8n on the current stylised figures,
  batch_v1 `render_demo_day` cam_01, boxes of 98–202 px: mean person
  confidence **0.356**, recall@0.25 **0.63** (n = 87). Compare the MetaHuman
  frames against this at the same box height with `tools/mh_detect_compare.py`.

## Truth-following render details (2026-09-25)

* **[V] PTZ / tilt**: `gt/camera_status.jsonl` (per-tick yaw, pitch, hfov) is
  turned into transform + `CurrentFocalLength` keys on the camera
  (`tools/render_run.camera_keys` → `survsim_scene.key_camera`); only cameras
  whose pose changes are keyed. cam_ptz gt_person_rendered 0.096 → 0.994.
* **[V] Output resolution per camera** must equal the GT's (some rigs are
  1080p); MRQ used a global 720p and every box of those cameras was off.
* **[V] Visibility track semantics**: `MovieSceneVisibilityTrack` bound to
  `bHiddenInGame` keys a **visible** bool (true = shown). Keying "hidden=True"
  hid every person the GT had in view and showed arrivals clustered at spawn.
  People are now shown only between their first and last key.
* **[V] Idle without an idle clip**: `survsim/export/gait_fk.py` runs forward
  kinematics on the glTF walk clip; the idle section holds the double-stance
  time (feet together) via `first_loop_start_frame_offset`, and walk sections
  ease in over 2 frames. The same FK measures the clip's real ground speed
  (CesiumMan: 0.76 m/s at 1.5 m → 0.89 m/s at 1.75 m), which calibrates play
  rate ∝ speed (read back per section: 0 mismatches).
* **[A] Stride-vs-speed in pixels** is not yet measured on renders; the FK
  numbers and the per-section rate read-back are the current evidence.

## Per-person ID mask → GT boxes (2026-09-26) [V]

* **[V]** Stencil: every `PrimitiveComponent` of a person gets
  `render_custom_depth=True`, `custom_depth_stencil_value=k` (1..215) via
  `set_editor_property` (persists into the PIE copy MRQ renders), plus
  `r.CustomDepth 3`. `survsim_scene.tag_people_stencil`.
* **[V]** Material `/Game/SurvSim/M_SurvIdMask`: post-process domain,
  `BL_REPLACING_TONEMAPPER`, SceneTexture `PPI_CUSTOM_STENCIL` → Custom HLSL
  node writing k as base-6 digits in R,G,B (level/5). The PNG comes out
  **sRGB-encoded** (level 1 → 123/124, 2 → 169/170, 3 → 203/204; ±1 dither),
  so the decoder tries linear and sRGB level tables and takes the closer.
  Mask edges align pixel-exactly with the beauty frame (checked by overlay).
* **[X]** The same material as an MRQ `additional_post_process_materials` pass
  with `{render_pass}/{frame_number}` naming: ~1.6 s per job, "Not all frames
  were fully submitted by the time rendering was torn down!", **no frames at
  all, beauty included**. What works: after the beauty queue finishes, spawn an
  unbound PostProcessVolume (priority 1000) with the material as its weighted
  blendable, save, render the same sequences again to `mask/<cam>/`, destroy
  the PPV. Deterministic sequence evaluation makes the two passes identical;
  cost ~+40 % wall (smoke_mock_day 300 f × 2 cams: 590 s incl. meter).
* **[X]** Very short renders (3–4 frames, incl. the meter probe into the
  scratchpad) hit the same "Not all frames submitted" and wrote nothing; 30
  frames were fine.

## Amodal boxes: full extent behind occluders (2026-09-26) [V]

MOT20 hand labels are **amodal** (full body even when occluded); our mask box
is the visible extent. `survsim/gt/amodal.py` adds the third box type, no
re-render, from `unreal/manifest.json` + `calib/` + the mask GT +
`render_plan.json`.

Route chosen, and why the others lose:
* A second stencil pass with occluders' stencil disabled **does not work**:
  CustomDepth is depth-tested, so the occluder still wins the pixel.
* One pass per person is exact but N× render cost (18 people = 18 passes).
* **Chosen: calibrated projected body envelope.** Per frame, at the keyed actor
  position (the manifest key the transform track was built from) and the
  *rendered* stature: a thin crown cylinder + a torso ellipse (0.10 H lateral,
  0.06 H fore-aft, at the keyed yaw) + both feet and toes from the clip's FK
  curves (`configs/people/cc0_foot_curves`) at the exact clip time
  (`(f - section_start)/fps × rate`, sections restart at `start - BLEND`, the
  same as `build_sequences`). Project, take the 2-D hull, then add the median
  per-edge residual (mask − model, in units of hull height) — per person, then
  locally over clean frames within ±20 frames on that camera. On *clean*
  frames (no other person's box within 10 % of height, not at the frame edge)
  the amodal box **is** the mask box.
* The FK phase is right: width of the projected feet correlates with mask
  width at lag 0 (r up to 0.87) and falls off either side. **[X]** modelled arm
  swing (hands as 0.6 × opposite foot) *lowered* IoU (0.85 → 0.82): 4 FK bones
  carry no arm data, so `HANDS=False`.
* Skinned-vertex projection would be better but needs the engine each frame;
  the plan + FK curves do not carry arms.

Validation (holdout: each clean frame predicted without itself or any clean
frame within ±2, as if occluded):

| set | clean rows | IoU median / p10 | IoU ≥ 0.9 | occluded rows | mask ⊆ amodal median / p10 |
|---|---|---|---|---|---|
| smoke_mock_day_tightgt | 1377 | 0.921 / 0.791 | 65 % | 103 | 0.988 / 0.919 |
| street_crowd_day_tightgt | 4059 | 0.907 / 0.765 | 53 % | 1165 | 0.979 / 0.878 |

Vertical edges are within ~1 px (±2 % of height); the error is horizontal —
arm swing on ~20 px-wide boxes, where 1.4 px per side already costs 0.1 IoU.
Global per-person calibration alone gave 0.85 (cylinder-only 0.76).

Visibility label = visible mask pixels / (amodal area × local fill ratio),
clipped to 1. Street: median 0.996, 28 % < 0.9, 3 % < 0.5; its noise floor on
clean frames is p10 ≈ 0.83 (pose-dependent fill), so treat < 0.8 as occluded.

Files: `gt/amodal/mot_<cam>.txt` (MOT format, col 9 = visibility, conf 0 where
no pixel is visible), `gt/amodal/amodal_summary.json`, and
`gt/boxes_mask.parquet` columns `visible_bbox` (= `mask_bbox`), `amodal_bbox`,
`visibility`, `clean`, `amodal_calib` (mask | person[+local] | pooled).
`tools/render_run.py --id-mask` now writes `render_plan.json` into the set and
builds the amodal GT after the mask GT. For the two batch_v7 sets the plan was
regenerated with `person_plan` (deterministic; character and height match
`render_appearance.json` for all 21 people).
**For H15 (amodal vs visible → real AP):** read `gt/amodal/mot_<cam>.txt`
instead of `gt/mot_<cam>.txt`; same frame/id/format, so it is a path swap.

**Height check wired (2026-09-26).** `check_rendered_height_matches_gt` now runs
in the render gate (`run_all`, `gt_height_matches_render`) and is printed by
`render_run`. **Both tightgt sets FAIL**: they were rendered before 92e1e97, so
every person is 1.70 m while GT says 1.22–1.89 m (smoke 3/3, street 16/18 beyond
2 cm; worst passer_by_00 +47.6 cm, loiterer_200 −19.2 cm). Mask and amodal
boxes are unaffected (they measure the render, and amodal uses the rendered
height), but `identities.parquet` `height_m` does not describe these pixels:
re-render before using height as a label.

**Standalone gate fixed (2026-09-27).** `check_height_files` only looked one
level above `render_dir`, so `scripts/check_render.py <set> --gt <set>` (which
passes the set dir itself, not its `render/` subdir) SKIPped with "the
rendered-identities join is missing"; it now checks the set dir and its parent
for `render_appearance.json`, falling back to `identities_rendered.parquet`.

## batch_v8: labels true to the pixels (2026-09-27) [V]

`tools/render_batch.py out/runs_v2/{smoke_mock,disguise_and_tamper,street_crowd,street_crowd_far}
--sites office_building perimeter_site transit_station retail_street --lighting day night
--frames 300 --id-mask --out out/batch_v8`, plus `out/runs_v2/disguise_and_tamper --lighting day
--window 95 135 --id-mask --suffix _epochs`, plus street_crowd_far re-run with
`--entrance-gate warn` (the known far-rig entrance_1 gap, batch_v5 [X]; the failed
attempts are kept in `out/batch_v8/_failed_entrance_gate/`).

What changed vs v7:
* **Heights from GT.** The runs_v2 manifests predate 92e1e97 (1.70 m for all) and
  `anim_plan.person_height_m` clamps to 1.55-1.95 m, while street_crowd has four
  people at 1.22-1.50 m. `tools/render_run.py` now reads `gt/identities.parquet`
  `height_m` and uses it unclamped (`gt_heights` / `use_gt_heights`), so the
  plan's `height_cm` (the mesh scale) equals GT. Gate `gt_height_matches_render`:
  **PASS on all 9 sets, 0 people beyond 2 cm** (v7 tightgt: FAIL, up to 47.6 cm).
* **Box tightness in the gate.** `render_batch --id-mask` runs
  `survsim.checks.box_tightness` (CPU) per set, writes `box_tightness.json`, and
  fails the set on it for day lighting (`--tightness-gate day|all|off`; night is
  reported, the detector reference misses dim people).
* **Mask-box decode speed.** `survsim/gt/mask_boxes.py` decoded a 1080p frame as an
  HxWx3x6 float array and scanned the image once per person: ~17 min per camera
  under batch CPU load (disguise_day took 4 168 s). Now LUT-decode of distinct
  colours + one nonzero pass + a spawn pool per camera: **33 s for 4 cameras,
  byte-identical MOT and parquet** (1fd38df).

| set | frames | gate | height | tightness IoU / top | amodal holdout IoU / mask-in-amodal | vis < 0.8 |
|---|---|---|---|---|---|---|
| smoke_mock day | 600 | 22/22 | PASS | 0.893 / -0.3 % | 0.914 / 0.991 | 12.5 % |
| smoke_mock night | 600 | 22/22 | PASS | 0.876 / -0.5 % (not gated) | same | 12.5 % |
| disguise day | 1 200 | 40/40 | PASS | 0.908 / -0.6 % | 0.894 / 0.965 | 14.8 % |
| disguise night | 1 200 | 40/40 | PASS | 0.915 / +0.4 % (not gated) | same | 14.8 % |
| disguise day epochs 95-135 s | 1 600 | 40/40 | PASS | 0.893 / -0.2 % | 0.907 / 0.880 | 19.4 % |
| street_crowd day | 900 | 31/31 | PASS | 0.898 / -0.1 % | 0.908 / 0.981 | 13.8 % |
| street_crowd night | 900 | 31/31 | PASS | 0.906 / -0.3 % (not gated) | same | 13.8 % |
| street_crowd_far day | 900 | 31/31 | PASS | 0.932 / -0.4 % | 0.879 / 0.970 | 17.4 % |
| street_crowd_far night | 900 | 31/31 | PASS | 0.919 / -0.4 % (not gated) | same | 17.4 % |

**Gate 9/9** (v7 7/8: far night `cam_02:motion` now passes), motion audit 0 %
stuck on all, videos for all 9, every set `licences ["CC0-1.0"]`,
`commercial_ml_ok: true`. Epochs: intruder_a e0 -> e1 at frame 1152
(`render_window.json`). Day and night share mask geometry, hence identical
amodal/visibility numbers. Sheet `out/demo_humans_v2/batch_v8_sheet.png`.

Wall: 8 473 s (first 8, incl. the two 96 s / 11 s far failures and 4 168 s for
disguise day on the old decoder) + 713 s epochs + 1 478 s far re-run = 10 664 s
for 10 000 frames. With the new decoder a set is 370-900 s. VRAM: editor peak
**3.98 GB**, 0 allocation-failure lines in the editor log; the *device* total
reached 7.7 GB at 00:30 because two other processes (an experiment job ~0.5 GB
and an unattributed ~3.0 GB process) ran at the same time - not ours; our
tightness/recall passes ran on CPU.

**Recall** (yolov8n, 10 frames/camera, conf 0.25, IoU 0.5, GT >= 20 px, CPU;
`out/demo_humans_v2/recall_v8_vs_v7.json`). The recall tool scores whatever MOT
file it is given: v8 **visible** = `gt/mot_<cam>.txt` (mask boxes), v8 **amodal**
= `gt/amodal/mot_<cam>.txt`, v7 = its `gt/` (projected hull for the plain sets).

| set | v8 visible R / P | v8 amodal R / P | v7 R / P | GT v8 / v7 |
|---|---|---|---|---|
| disguise day | 0.534 / 0.809 | 0.495 / 0.761 | 0.544 / 0.691 | 103 / 103 |
| disguise day epochs | 0.776 / 0.929 | 0.685 / 0.847 | 0.745 / 0.851 | 152 / 153 |
| disguise night | 0.505 / 0.912 | 0.456 / 0.839 | 0.485 / 0.877 | 103 / 103 |
| smoke_mock day | 0.569 / 1.000 | 0.569 / 1.000 | 0.529 / 0.964 | 51 / 51 |
| smoke_mock night | 0.569 / 0.829 | 0.569 / 0.829 | 0.471 / 0.706 | 51 / 51 |
| street_crowd day | 0.900 / 0.912 | 0.900 / 0.912 | 0.853 / 0.845 | 231 / 231 |
| street_crowd night | 0.810 / 0.935 | 0.801 / 0.925 | 0.810 / 0.899 | 231 / 231 |
| street_crowd_far day | 0.569 / 0.796 | 0.583 / 0.808 | 0.520 / 0.780 | 144 / 150 |
| street_crowd_far night | 0.569 / 0.739 | 0.562 / 0.717 | 0.527 / 0.681 | 144 / 150 |
| **all** | **0.696 / 0.879** | **0.676 / 0.857** | **0.664 / 0.819** | 1 210 / 1 223 |

By height (visible / amodal / v7): 20-60 px 0.428 / 0.436 / 0.379; 60-120 px
0.920 / 0.917 / 0.872; 120+ px 0.991 / 0.854 / 0.950. The v8 gain over v7 is the
label, not the pixels (same characters): projected-hull GT was loose, so correct
detections fell under IoU 0.5. Amodal loses at 120+ px, where the near people are
partly occluded or cut by the frame edge and the detector boxes the visible part.
Noise ~±0.03 per set.

### batch_v8 file layout (for H15: amodal vs visible labels)

Per set `out/batch_v8/<scenario>_<day|night>[_epochs]/`:
* `render/<cam>/NNNN.png` beauty (NNNN = absolute GT frame; epochs set starts at 0950)
* `mask/<cam>/NNNN.png` ID pass; `render_result.json` `stencil_ids` {uid: k}
* `gt/mot_<cam>.txt` **visible** (mask) boxes, MOT: conf 0 / vis 0 = no pixels (ignore)
* `gt/amodal/mot_<cam>.txt` **amodal** boxes, same frame/id rows, col 9 = visibility
* `gt/projected/mot_<cam>.txt` the old projected-hull boxes
* `gt/boxes_mask.parquet`: `visible_bbox`, `amodal_bbox`, `visibility`, `clean`,
  `amodal_calib`, `mask_pixels`, `source`
* `gt/identities.parquet`, `identities_rendered.parquet` (render palette per epoch,
  `render_frame_start`, `asset_licences`, `commercial_ml_ok`), `render_appearance.json`
  (character, `height_cm` = GT), `render_plan.json`, `render_window.json`
* `check_render.txt` (gate + motion audit + box tightness), `box_tightness.json`,
  `licences.json`, `video/`
H15 = train once on `gt/mot_*` and once on `gt/amodal/mot_*`; everything else equal.

## Environment variety (2026-09-25)

* `survsim/export/env_plan.py` (pure, tested) maps the scenario's site type
  (explicit `environment.site_type`, else inferred from env id / zone tags) to
  a layout: office forecourt (glazed blocks, bollards, visitor bays), station
  forecourt (long low block, canopy, gate line, boards), warehouse yard (fence
  line, road, gatehouse, containers, floodlights, tree horizon), retail
  frontage (road, parked cars, dense shopfronts), car park (bays, cars, ticket
  machine, low walls), plaza (stage, kiosks, planters, tall city). Signature
  structures go on the sides the cameras face. The sun comes from the seed
  (25–60° elevation) when the scenario gives no time of day.
* Placement rules (tested): nothing solid on any walked 1 m cell; no
  non-flat geometry in the first 10 m of any camera's ±60° view.
  **[X]** a sized item passed a pre-swapped size *and* a 90° yaw — the station
  block rotated across the plaza and filled cam_03; roads ignored yaw and cut
  through the plaza. Sized items now use yaw 0 with swapped sizes; roads honour yaw.
* **[V] Metering**: a 3-frame probe render measures luma; outside 60–140 the
  EV100 moves by log2(measured/100) once (office 19 → EV 10.6, plaza 43 → 11.8).
  Written to `<out>/meter.json`.
* Round trip per environment (build + probe + 40 frames, 1 camera): 74–114 s.
* Render gate on every variety frame: 0 failures across the 6 site types
  (`out/env_variety/sheet.png`).

## Batches and detector numbers (2026-09-26)

* `tools/render_batch.py runs… --lighting day night --frames 300 --out out/batch_vN`:
  per set the render gate, `survsim audit-motion` (appended to
  `check_render.txt`), and — only if the gate passes — a background
  `survsim video` encode. **[X]** v1/v2 encoded failed sets; the gate now blocks.
* batch_v3 (same runs, frustum gate on): 6 sets, 5 400 frames, 6/6 pass, 1 870 s wall.
* batch_v2 (runs_v2 GT: staggered arrivals/departures): 6 sets, 8 100 frames,
  all 6 pass the gate after the night fix. Night: metered EV (clamped ±3 stops,
  base 1.0), urban-glow fog inscattering, 78° 25 000 lm 3400 K spots.
  **[X]** first v2 night pass was black (no lamps in the yard layout, no metering).
* **[V] In-editor frustum gate**: after the environment is built, every static
  Env actor's bounds are tested against each camera's first 10 m (±60°); any
  violator fails the job. The planner applies the same rule with intrinsic
  footprints (container, car, gatehouse … not just their centre point).
* yolov8n person recall (conf 0.25, IoU 0.5, 5 frames/camera, GT boxes ≥ 20 px),
  `tools/eval_recall.py`:

  | set | camera | recall | precision |
  |---|---|---|---|
  | smoke_mock day | cam_01 (near) | 0.857 | 0.923 |
  | smoke_mock day | cam_02 | 0.909 | 1.0 |
  | street_crowd day | cam_01 (near) | 0.900 | 0.923 |
  | street_crowd day | cam_02 / cam_03 | 0.526 / 0.432 | 0.83 / 0.80 |
  | disguise day | cam_ptz / cam_gate | 0.833 / 0.588 | 0.83 / 0.83 |
  | disguise day | cam_yard (all < 60 px) | 0.105 | 0.67 |
  | smoke_mock night | cam_01 / cam_02 | 0.714 / 0.364 | 0.91 / 1.0 |
  | street_crowd night | cam_01 / cam_02 / cam_03 | 0.775 / 0.737 / 0.486 | 0.86 / 0.90 / 0.82 |

  People ≥ 120 px are found at 0.75–1.0; the misses are 20–60 px figures,
  which on a stylised CesiumMan are below what a COCO nano model resolves.

## Characters: Quaternius root cause and the re-bake (2026-09-26)

**[V] Fixed.** Three rebaked Quaternius characters (man_HMnu, woman_casual_jpKR,
adventurer_5EGW) render upright in the production editor, with arms down,
feet on the ground, and mesh bounds of exactly **175.0 cm** (not ~100 m). The
lineup frame is `out/char_lineup_rebake/lineup.0000.png`. It is an unlit
SceneCapture in base colour, spawned 3 km from the production scene, so it looks dark.

**Root cause.** This was measured in pure Python on `man_HMnuH5geEG.glb` with `survsim/export/gltf_normalize.py`.
The file is valid glTF, but it depends on three details of the spec that Interchange handles differently:
1. **The skinned mesh node carries rot(-90 X)·scale 100.** It is a *sibling* of the
   armature. The spec says to ignore a skinned mesh node's transform. The raw vertices
   are about 0.05 units tall, and only the IBMs map them to the 4.8-unit body. The
   previous bake only touched the armature, so it left this node unchanged. That
   explains the face-down orientation and the huge bounds.
2. **Bind pose ≠ rest pose.** The inverse IBMs give a wide T-pose, and the node TRS give
   a different pose. Only 3 of 31 joints satisfy `W_j·IBM_j == meshNodeWorld`.
   Joints without channels (Bone, Hips, Abdomen, PoleTargets) take whichever
   reference pose the importer picks, and in UE that stretched the arms.
3. Every joint has world scale 100, and its local translations are about 0.005.

**Fix: `rebake()`** (a real "apply transforms"). It evaluates the source node tree and
re-expresses everything in one frame: vertices in metres, Y-up, feet at y=0, and
height 1.75 m. It merges all skinned meshes into ONE identity mesh node, because
Interchange makes one SkeletalMesh per mesh node, so a 5-part character spawned
from asset[0] was a floating torso. Joints get unit scale, rest == bind
(`W_j·IBM_j = I`), and **every** clip is resampled to full T/R/S channels for
every joint. The new rest pose is the authored node rest (arms down),
re-skinned into the mesh. That is exact in the rest pose and within 8 mm in the walk. `rest="bind"`
is exact in every pose (1e-4), but its reference pose is a T-pose. Hand-held weapons
(knife/pistol/axe) are dropped, and other rigid attachments are skinned 100% to
their parent joint.

**Checks** (`check_character`, tests/test_gltf_normalize.py). Rest-skinned ==
raw mesh; the walk at frame 0 is 0.85–1.1 × 1.75 m tall; head-dominated vertices sit in
the top quarter and foot-dominated ones in the bottom 15%; trunk joints move < 0.3 m from rest;
and no non-root bone changes length by > 5 cm. A bounding-box aspect test (≥3×) was dropped
because the *source* walks fail it: at frame 0 they are 1.74 × 1.04 m (full stride), and chibi
bodies are wide. Arms also swing 0.6 m out of a T-pose rest, so the jump limit applies
only to the trunk.

`tools/import_characters.py --rebake` writes `~/data/assets/characters_rebake/`.
The import jobs then use those copies under the suffix `_r`. The old `_m`/`_n` imports are
still the broken ones. Not yet checked: an animated sequence render (in a
static editor world `set_position` does not advance the pose, so the 3
lineup frames are identical), plus facing/yaw_offset for the new rest frame.

## MetaHumans in production (2026-09-26) [V]

Every person in `tools/render_run.py` is now a baked MetaHuman (default;
`SURVSIM_CHARS=CesiumMan` restores the old figure).

* **Catalogue** `configs/people/metahuman_bodies.json`: MHB01/02/03, stature =
  body-mesh bounds z / 0.83 (158 / 193 / 157 cm), scaled per person to the
  scenario height. Walk loop `AS_MH_Neutral_Walk_Loop_F` is in place; its
  ground speed was measured by AnimPose FK (stance foot 1.83 m/s at rate 1,
  8 steps / 4 s) and normalised to 1.75 m -> play rate proportional to speed.
  Idle `AS_MH_Neutral_Stand_Idle_Loop` when stopped, with a per-person phase.
  Rate stays 1 + seek in the capture jobs (rate 0 freezes the pose).
* **[X] "bat-wing" shoulders**: the baked bodies need their RigLogic
  correctives. `scripts/ue/mh_postprocess.py` sets
  `/MetaHumanCharacter/Body/ABP_Body_PostProcess` (and `Face/ABP_Face_PostProcess`)
  as `post_process_anim_blueprint` on the baked meshes; runs under sequencer
  single-node tracks too. Wings gone in the lineup.
* **Clothing without a login** (`scripts/ue/survsim_clothing.py`): a region
  material painted from the **pre-skinned** position (shoes / trousers-shorts-
  skirt / top + short or long sleeves / skin / hair on the face mesh), per-person
  palettes from `survsim/export/outfits.py` (street-wear weights; hi-vis 20 % on
  perimeter/car-park sites). Written to `<set>/render_appearance.json` as ReID
  truth. Only MHB02 has the one offline garment; it is off
  (`SURVSIM_MH_GARMENT=1`).
  **[X] `PreSkinnedPosition` is vertex-shader only**: wired straight into a pixel
  Custom node the material compiles to 0 instructions and bodies render in the
  grey default material. Route it through a `VertexInterpolator`. Both masters
  now refuse to save with 0 pixel instructions (`get_statistics`).
* **Lineup check** `tools/mh_lineup.py --n 8`: 8 dressed MetaHumans, metered
  exposure, and a per-person check of the torso and hip bands against the
  outfit (hue error for colours, darker-than-ground for dark colours) — fails
  the job on mismatch. `out/demo_humans_v2/mh_lineup.png`: 16/16 pass.
  **[X] A lone SceneCapture2D capture (`capture_every_frame=False`) is white in
  daylight** (no exposure history); capture every frame.
* **Behaviour -> animation** (`survsim/export/anim_plan.py`): every REGISTRY
  behaviour maps to idle / look / run (test), postures without an offline clip
  (crouch, seated, prone ...) hold the idle loop explicitly. "look" (loiter,
  recon, lost_tourist, photograph ...) turns the whole body +-35 deg over 6 s:
  the offline rig has no control rig for the neck. Run loop above 2.3 m/s.
  Keys get their world label from `gt/world.parquet`.
  `out/demo_humans_v2/loiter_demo.mp4`: disguise_and_tamper cam_gate 14-24 s,
  intruder_a (suspicious.recon) standing and scanning while others pass.
* **VRAM / speed**: 18 MetaHumans x 300 frames x 3 cameras render at the same
  speed as CesiumMan (street_crowd day 320 s). **[X]** with textures on, MRQ
  ran the 8 GB card out of memory in `FGPUSkinCache` and the editor died;
  the render job now sets `r.SkinCache.Mode 0`, `r.Streaming.PoolSize 1000`,
  `r.SkeletalMeshLODBias 1`, frees the previous set + GC before a rebuild,
  meters at 640x360. Peak during batch_v4: **3.4 GB**. No people cap needed.
  `tools/ue_job.py` records `vram_peak_mb` and fails a job above 7 200 MB or on
  any "Failed to allocate Device Memory"; the runner's heartbeat is a thread
  (`<now> <pid> <last tick>`), so a hung game thread is not a dead editor.

### Detector: MetaHuman vs CesiumMan, same GT (batch_v4 vs batch_v3)

Same runs, same cameras, same GT boxes; only the character differs.
`tools/eval_recall.py --n 10` (conf 0.25, IoU 0.5, GT >= 20 px):

| set | camera | CesiumMan (v3) recall / prec | MetaHuman (v4) recall / prec |
|---|---|---|---|
| smoke_mock day | cam_01 / cam_02 | 0.897 / 0.70, 0.909 / 1.00 | 0.966 / 0.72, 0.909 / 0.83 |
| smoke_mock night | cam_01 / cam_02 | 0.724 / 0.95, 0.364 / 0.80 | 0.793 / 0.92, 0.591 / 0.87 |
| street_crowd day | cam_01 / 02 / 03 | 0.885, 0.526, 0.533 | 0.949, 0.821, 0.840 |
| street_crowd night | cam_01 / 02 / 03 | 0.756, 0.564, 0.627 | 0.897, 0.718, 0.800 |
| disguise day | gate / ptz / sub / yard | 0.441, 0.846, 0.500, 0.158 | 0.882, 0.769, 0.833, 0.237 |
| disguise night | gate / ptz / sub / yard | 0.265, 0.385, 0.778, 0.000 | 0.706, 0.846, 0.778, 0.000 |
| **all (770 GT boxes)** | | **0.577 / 0.852** | **0.758 / 0.884** |

Mean person confidence at matched box height (street_crowd day cam_01,
`tools/mh_detect_compare.py --height`): 50 px 0.439 -> **0.634** (recall@0.25
0.72 -> 0.96); 90 px 0.632 -> 0.705; 150 px 0.675 -> 0.765.
cam_yard (all < 60 px, night 0.000 both) is the remaining weak camera.

**[X] mh_detect_compare's plate-diff box** took in the cast shadow (and merged
with other people): real 0.8-confidence hits scored as misses. Shadows are now
excluded (luma drop with unchanged chromaticity, test) and the capture job
hides every other SkeletalMeshActor.

### Real textures + HDRI sky [X] (opt-in, procedural stays default)

`survsim/export/pbr_sets.py` maps ambientCG sets to library roles (DX normal,
roughness, physical tile, aspect) and picks a seeded Poly Haven HDRI with the
sun at its brightest upper direction; `--textures`, `--hdri auto`.

* **[X] The PBR master does not compile**: `M_SurvPBR_v6/v7` report 0 pixel
  instructions, so every "textured" render showed the engine's grey default
  grid — the "checkerboard" that did not change with any tile/tint edit
  (5 renders, identical stats). A default texture on each sampler (v8) was not
  enough; v8 now refuses to build and `--textures` fails loudly. Next: compare
  node-by-node with `M_SurvProc_v5` (compiles, 780 instructions).
* HDRI sky alone (street_crowd day cam_01, 6 frames): hist_entropy 6.06 -> 6.43
  (toward the real 7.13), rms_contrast 0.298 -> 0.296, chroma 9.4 -> 6.6 (away
  from 12.2), luma 85 -> 116; with the overcast HDRI shadows flatten. Not default.
* **[X] cached material instances kept their old parent** after a master
  version bump; `pbr_instance` now re-parents. Textures are `never_stream`
  (the ground cube's 0-1 UVs over 57 m made streaming keep only low mips).

### batch_v4 (all MetaHuman)

`tools/render_batch.py out/runs_v2/{smoke_mock,street_crowd,disguise_and_tamper}
--lighting day night --frames 300 --out out/batch_v4`: 6 sets, 5 400 frames,
**6/6 gate pass**, 1 749 s wall (v3: 1 870 s), 0 GPU allocation failures, peak
3.4 GB. Sheet: `out/demo_humans_v2/batch_v4_sheet.png`.

## Scene generator v2: sites that look like different places (2026-09-26)

`survsim/generator/layout.py` (pure Python, SI, seeded) -> `env_plan.plan_from_layout`
(plan v2, Unreal cm) -> `survsim_scene.build_from_plan_v2`. v2 is now the default in
`tools/render_run.py` (`--env-v1` for the old rim-props plan, `--layout f.json` to
build a saved layout exactly, `--env-seed`). `tools/render_batch.py --sites ...`.

**What varies per seed:** footprint shape (rect/L/U/courtyard/twin), floors, a
k x 90 deg site rotation, road + kerbs + zebra/tactile, parking rows (perpendicular or
60 deg, occupancy 35-90 %), gates at the scenario's entrances (door / ticket gates /
vehicle gate + gatehouse + barrier / street end / car-park barrier / event gate),
zone furniture (bus & smoking shelters, ticket gates, loading docks + trailers, bin
store, cycle racks, skip, pallets, kiosks, food trucks, stage), vegetation density,
boundary (open / hedge / dwarf wall / palisade / mesh / wall / heras), CC0 index props
(cones, dumpster, trash bags, construction barriers, scaled to height, plain paint),
wear decals (oil/dirt/gum/crack/tyre/puddle/leaves; DecalActor, `M_SurvDecal_v2`),
dry/damp (seasonal), a real date/hour/city with the NOAA solar position, a backdrop
ring (city blocks / sheds / tree line), and an installer rig (walls <= 25 deg off the
facade normal or 5 m poles, 3-6 m, one camera per entrance first, then coverage).

**Tests (tests/test_layout.py, 11, ~20 s, no engine):** two seeds of one type share
< 50 % of placements (10 seeds: worst pair 0.39, means 0.02-0.13); leave-one-out
nearest-centroid on 51 plan features: **0 / 60 misclassified** (min margin 1.29x);
every layout gate-passes (entrances reachable by BFS, usably visible + unoccluded
from >= 1 camera, no structure in any camera's first 10 m); every camera usably
sees >= 30 % of the walk core; wall cameras outside every block; solar known values.

**Editor gate added:** `entrance_gate` - each entrance must be framed by a camera
(host `usably_visible`) AND reachable by a line trace camera -> entrance at 1.5 m
against the built Env (self-test: a trace through the largest block must hit).

**Evidence:** `out/scene_variety_v2/sheet.png` - 6 types x 3 seeds, **18/18 render
gate + frustum gate + entrance gate**, `variety.json` has per-scene env build time
(7.7-16.7 s; round trip incl. metering probe 26-40 s). People: street_crowd as a
transit station, smoke_mock as an office forecourt, 100 frames, MetaHumans, gate
pass + motion audit 0 % stuck (`out/scene_variety_v2_people/`, build 10-13 s).
`out/scene_variety_v2_r1/sheet.png` is the superseded first pass.

**[X] found on the way (each fixed, most with a test):**
* mounts used unrotated w/d: cameras ended up *inside* yaw-90 blocks (test added);
* roof overhang and tree crowns exceeded planned footprints -> editor frustum gate
  failed; planner now pads footprints (crowns by 0.6 x height);
* a barrier arm hid the entrance from a 1 m trace; trace target is chest height;
* 12 m containers on a 3 m step fused into one striped slab (the "flag");
* decals straddling solids smeared up their sides - decals only on open ground;
* mean-luma metering was fooled by wet sun glint (median now) and shade crushed
  28 % of a frame (shadow lift, capped); the short probe under-reads wet shaded
  scenes, so day metering never opens more than 2 stops; wetness capped at 0.55
  (at 0.9 the ground mirrored the sky and tripped the horizon check);
* a 100-frame window of disguise_and_tamper has nobody in cam_sub (GT: first box at
  frame 104), so the motion check fails correctly - use a window with people.

### PBR master compiles: M_SurvPBR_v9 [V] (2026-09-26)

Root cause of v6-v8's 0 pixel instructions, isolated by a one-sampler-per-material
probe: albedo alone 379 instr., normal alone 383, **roughness as SAMPLERTYPE_MASKS
with the sRGB DefaultTexture as default: 0** (LINEAR_COLOR + sRGB default: also 0).
A sampler's type must match its default texture. v9: roughness sampler LINEAR_COLOR,
default `/Engine/EngineMaterials/T_Default_Material_Grid_M` (linear), roughness
imports TC_DEFAULT linear (existing TC_MASKS imports converted). **439 instructions.**

* 1 m proof: `out/pbr_v9/checker_1m.png` - a 2x2 checker at TileM=1 on a 10 m
  plane, top-down, with 1 m ticks: one checker period per tick.
* Textures are now the default (`--no-textures` for procedural). Realism stats on
  3 variety frames (`out/pbr_v9/before_after.png`, left procedural, right PBR):
  in-band 4->5, 3->4, 5->6 of 13; chroma rises toward the real band (6-8 -> 9-9.5).
* **[X] open:** wear decals are much fainter on the PBR asphalt than on the
  procedural one; the ambientCG asphalt reads brownish.

### Foot slide: measured, then reduced (2026-09-26)

`scripts/ue/walk_foot_curves.py` (editor job) samples the MetaHuman walk loop's
foot bones by AnimPose FK (401 samples over 4 s; clip forward +Y, 1.6 m skeleton)
into `configs/people/mh_walk_foot_curves.json`. `survsim/export/foot_slide.py`
replays each walk section exactly as the sequencer plays it (section rate from its
start, transform interpolated between keys, yaw offset, actor scale) and measures
the world speed of whichever foot is in contact (within 2 cm of its lowest point).

| run (300 frames) | before: median / mean / p90 cm/s | after |
|---|---|---|
| smoke_mock | 16.8 / 26.6 / 49.7 | **8.1** / 20.5 / 57.1 |
| street_crowd | 9.3 / 17.9 / 45.1 | **5.9** / 15.7 / 46.3 |
| disguise_and_tamper | 7.9 / 16.1 / 42.8 | **6.8** / 17.5 / 51.3 |

Changes: section rate from the section's own distance/time (merged sections kept a
stale first-key rate), rate step 0.1 -> 0.02, MIN_RATE 0.4 -> 0.15 (slow drifters
were clamped into sliding), heading 3-tap smoothed while moving (avoidance jitter
swung the planted foot), turn-in-place at <= 120 deg/s for > 45 deg heading changes
while stationary, idle<->walk crossfade 0.3 s (was 2 frames).
**[X] measured and rejected:** capping the turn rate while *walking* (60-240 deg/s)
raised slide at every setting - the body stops facing its path. A "planted-foot
mean speed" estimate of the clip's ground speed (1.714 m/s) was 7 % low: a sweep
of the assumed clip speed minimises slide at 1.84, confirming the FK value 1.83.
The p90 tail (~50 cm/s) and the small mean rise on disguise are open: they sit in
starts/stops and sharp avoidance turns, where a single loop at constant rate
cannot match the root; fixing it needs start/stop clips or foot IK.
batch_v5's first three sets were rendered before this change; the rest after.

### batch_v5 (v2 sites, PBR textures, MetaHumans) (2026-09-26)

`tools/render_batch.py out/runs_v2/{smoke_mock,disguise_and_tamper,street_crowd,street_crowd_far}
--sites office_building perimeter_site transit_station retail_street --lighting day night
--frames 300 --out out/batch_v5` (street_crowd_far re-run with `--entrance-gate warn`).
8 sets, 7 200 frames; render gate **7/8** - disguise day `cam_sub:exposure` FAIL
(25.6 % crushed vs 25 % limit: one global EV, cam_sub looks into the warehouse's
shade; per-camera exposure is the fix, not done). Motion audit 0 % stuck on all.
Peak VRAM <= 3.9 GB. Sheet `out/demo_humans_v2/batch_v5_sheet.png`; videos per
passing set in `out/batch_v5/<set>/video/`.

**[X] entrance gate on street_crowd_far:** the scenario's own far rig sees spawn.east
at 13.7-19.2 px (< 20 px) - no camera usably sees that entrance. The gate is right;
the set was rendered with `--entrance-gate warn` and the failed attempt kept
(`*_entrance_gate_failed`). The run gate now checks the scenario's spawn points,
not their projection onto the layout boundary (which sat outside far rigs).

yolov8n person recall (`tools/eval_recall.py --n 10`, conf 0.25, IoU 0.5, GT >= 20 px):

| set | recall per camera | recall / precision | GT boxes |
|---|---|---|---|
| smoke_mock day (office) | cam_01 0.862, cam_02 0.364 | 0.647 / 0.917 | 51 |
| smoke_mock night | cam_01 0.724, cam_02 0.409 | 0.588 / 0.882 | 51 |
| street_crowd day (transit) | 0.936, 0.808, 0.853 | 0.866 / 0.830 | 231 |
| street_crowd night | 0.936, 0.705, 0.813 | 0.818 / 0.883 | 231 |
| disguise day (perimeter) | gate 0.706, ptz 0.846, sub 0.833, yard 0.316 | 0.602 / 0.747 | 103 |
| disguise night | gate 0.735, ptz 0.846, sub 0.833, yard 0.0 | 0.495 / 0.911 | 103 |
| street_crowd_far day (retail) | cam_01 0.949, cam_02 0.0, cam_03 0.081 | 0.527 / 0.790 | 150 |
| street_crowd_far night | cam_01 0.949, cam_02 0.0, cam_03 0.113 | 0.540 / 0.692 | 150 |
| **all** | | **0.678 / 0.823** (tallest-first matching, not re-scored) | 1 070 |

Not comparable to batch_v4's 0.758 (different sets: the far cameras add 300 GT
boxes of 14-40 px that a nano model almost never finds; cam_02/03 at 45-60 m).
Excluding street_crowd_far: 0.734. smoke_mock cam_02 fell (0.909 -> 0.364) - its
people are 20-60 px against the new office site's busier ground. Night frames read
amber and bright near lamps (metered to the night band); a colour-temperature /
lamp-falloff pass is open.

### batch_v6: the consistent set (2026-09-26)

Same 8 sets as batch_v5, all rendered with the same code: new anim plan (foot-slide
fixes), **per-camera metered EV** (the probe is metered per camera; each CineCamera
carries its own manual-exposure post-process), night v2 (4000 K, 9 000 lm, 30 m
lamps; blue-grey sky light and fog inscatter), neutral-tinted PBR asphalt, stronger
decals. `--entrance-gate warn` for all (only street_crowd_far's east entrance is
unseen, as in v5). `out/batch_v6`, 7 200 frames, 2 747 s wall, peak VRAM < 4 GB.

* Render gate **7/8**. disguise day now passes (cam_sub metered to EV 12.98 vs
  13.5 global). **[X]** street_crowd_far night `cam_02:motion` FAIL (72.6 % of steps
  "frozen"): its people are 13-20 px at 45-60 m and, with the dimmer lamps, change
  too few pixels between frames. v5's brighter amber pools passed this. Not fixed.
* Motion audit 0 % stuck on all 8. Videos for the 7 passing sets.
* Night look (smoke_mock cam_01, frame 10, v5 -> v6): cast_b -13.7 -> -0.5,
  cast_a -2.1 -> 5.7, chroma 14.5 -> 11.8, luma 95 -> 98, blockiness 1.01 -> 1.02
  (`out/batch_v6_probe/night_before_after.png`). At the time docs/15 had **no real night
  reference**, so these were compared to the day band only (since H32: LLVIP; raw renders
  3/13 vis · 2/13 thermal in band, no night profile in band — see FACTS realism row).
* Sheet `out/demo_humans_v2/batch_v6_sheet.png`.

| set | recall per camera | recall / precision | GT |
|---|---|---|---|
| smoke_mock day | cam_01 0.828, cam_02 0.182 | 0.549 / 0.966 | 51 |
| smoke_mock night | cam_01 0.690, cam_02 0.318 | 0.529 / 0.794 | 51 |
| street_crowd day | 0.936, 0.808, 0.853 | 0.866 / 0.837 | 231 |
| street_crowd night | 0.949, 0.705, 0.773 | 0.810 / 0.866 | 231 |
| disguise day | gate 0.853, ptz 0.769, sub 0.833, yard 0.289 | 0.631 / 0.765 | 103 |
| disguise night | gate 0.735, ptz 0.846, sub 0.833, yard 0.0 | 0.495 / 0.879 | 103 |
| street_crowd_far day | cam_01 0.949, cam_02 0.0, cam_03 0.065 | 0.520 / 0.772 | 150 |
| street_crowd_far night | cam_01 0.962, cam_02 0.0, cam_03 0.032 | 0.513 / 0.748 | 150 |
| **all** | | **0.666 / 0.824** (excl. far: 0.725) — tallest-first matching; re-scored with score-order matching 0.665 / 0.823 (excl. far 0.723; docs/13, and the v7 table below) | 1 070 |

vs batch_v5 0.678 / 0.823: flat within noise (10 frames per camera); smoke_mock
cam_02 (20-60 px people on the office site) fell further, 0.364 -> 0.182.

## CC0 characters are the production default (2026-09-26) [V]

Licence compliance (docs/research/47 option A): `tools/render_run.py` now renders
every person as a **rebaked Quaternius CC0 body on its own clips**. MetaHumans
are opt-in only (`SURVSIM_CHARS=metahuman`); `SURVSIM_CHARS=a,b` is an explicit
allow-list (e.g. `CesiumMan`, CC-BY - its attribution is written to the set).

* **Import + verify, one per job** (`tools/import_characters.py --cc0` ->
  `scripts/ue/cc0_verify_one.py`, assets `/Game/SurvSim/Chars/<name>_c`,
  catalogue `Saved/survsim_characters_cc0.json`). Checks by AnimPose FK on the
  walk clip, not a flag: mesh bounds 160-190 cm (all 175.0), head bone >= 0.75 H
  and lowest foot <= 0.1 H at every one of 241 samples. **16 of 22 usable**:
  man_hmnu, man_fjhy, woman_casual_jpkr, woman_in_tank_top_xqze,
  woman_in_dress_zmyp, woman_aqsd, animated_human_c3ib, animated_woman_9kf7,
  adventurer_5egw/zwf0, punk_btal, beach_character_dojk, worker_yg2b,
  casual_character_kz3d, animated_woman_niit/qj2g. The 6 rejected are the
  chibi/cube bodies (head 32-43 % of stature; the head-bone check fails too).
* **[X] VRAM:** 22 imports in one session climbed to 7.7 GB and logged 334
  "Failed to allocate Device Memory". The job now GCs at the end and the
  importer stops above 5 GB total; re-run from a clean editor peaked at 1.3 GB.
* **Clips:** each body's own walk / idle / run (by clip-name tail:
  `..._Walk`, `Walking`, `..._Idle`, `..._Run`, `Running`). FK foot curves
  (walk + run, component space) are in `configs/people/cc0_foot_curves/`.
  `survsim/export/cc0_chars.py` derives from them: forward axis (opposite the
  planted foot's motion; all +Y, snapped when within 5 deg) -> `yaw_offset` -90;
  ground speed = the speed minimising the median planted-foot residual (the
  quantity the slide metric measures; the MetaHuman 1.83 m/s was confirmed the
  same way). Walk 0.97-1.74 m/s at rate 1 for 1.75 m (the Quaternius
  "CharacterArmature" rigs walk slowly, the "HumanArmature" ones briskly), run
  2.2-5.5 m/s. The median planted speed alone was 5-7 % off the optimum.
* **Clothing** (`survsim/export/cc0_regions.py` + `M_SurvCC0Cloth_v1`): the
  MetaHuman painted material keys regions on an A-pose height/|x| rule; these
  bodies are arms-down, some chibi, 8/22 have no UVs. Each vertex is labelled
  by its **dominant skin joint** (+ source material name for hair/eyes) and the
  code is baked as COLOR_0.r into `~/data/assets/characters_cc0/*.glb`
  (source untouched); the material reads VertexColor: skin / top / forearm
  (long sleeve or skin) / thigh / shin (trousers or skin) / shoes / hair / eyes,
  palette from `survsim/export/outfits.py` (same ReID truth as before).
  `out/char_lineup_cc0/render/cam_lineup/0010.png`: 6 bodies upright, walking
  away from the camera (facing correct), tops/trousers/skin/hair distinct.
  **[A]** `SkeletalMesh.has_vertex_colors` is not readable from Python in 5.8;
  the lineup render is the evidence the colours survived Interchange.
* **Licences in GT** (`survsim/gt/licences.py`): per person
  `asset_licences` in `render_appearance.json` and
  `<set>/identities_rendered.parquet` (`"CC0-1.0|quaternius:man_hmnu"`), and
  `<set>/licences.json` with `commercial_ml_ok` (false if ANY person depends on
  a MetaHuman body or AS_MH clip, or on an unknown licence) and
  `ATTRIBUTION.txt` when CesiumMan is used. The Recorder writes
  `LicenseRef-SurvSim-Procedural` + `meta/licences.json` for mock runs.

### Foot slide with the CC0 clips

`tools/foot_slide_report.py` rebuilds the render plan and replays it against
each person's own clip curves (`plan_slide(per_character=...)`). It reproduces
the MetaHuman numbers exactly (8.08 / 5.93 / 6.82), so the comparison is like
for like. Median / mean / p90 cm/s, 300 frames:

| run | MetaHuman (docs above) | CC0 all 16 | man_hmnu only | adventurer_5egw only |
|---|---|---|---|---|
| smoke_mock | 8.1 / 20.5 / 57.1 | 8.5 / 18.6 / 41.8 | 13.3 / 21.5 / 46.2 | 11.9 / 19.8 / 39.3 |
| street_crowd | 5.9 / 15.7 / 46.3 | 8.8 / 14.1 / 29.2 | 8.4 / 14.4 / 31.8 | 9.7 / 14.3 / 28.3 |
| disguise_and_tamper | 6.8 / 17.5 / 51.3 | 9.9 / 15.8 / 35.5 | 12.1 / 17.7 / 39.8 | 12.0 / 17.2 / 33.2 |

The median is 1-3 cm/s worse, the tail better. The floor is in the clips: on a
straight walk at the optimal rate the hand-keyed Quaternius walks still slide
8.5-9.8 cm/s (the Mixamo-rigged animated_human_c3ib / animated_woman_9kf7:
3.8-4.2), because their planted foot does not move at constant speed. Not
fixable by rate; foot IK or better clips would be.

### batch_v7: the licence-compliant (CC0) set (2026-09-26)

Same 8 sets, sites, frames and flags as batch_v6; only the people differ
(16 CC0 bodies instead of 3 MetaHumans). `out/batch_v7`, 7 200 frames, 2 732 s
wall (v6 2 747 s), peak total VRAM 4.7 GB incl. the other GPU job, 0 allocation
failures. **Every set `commercial_ml_ok: true`, licences `["CC0-1.0"]`.**

* Render gate **7/8**, the same failure as v6: street_crowd_far night
  `cam_02:motion` (71.6 % frozen; 13-20 px people at 45-60 m). Motion audit
  0 % stuck on all 8. Videos for the 7 passing sets.
* Sheet `out/demo_humans_v2/batch_v7_sheet.png`.
* yolov8n recall, both batches re-scored with the current `eval_recall`
  matching (`tools/batch_compare.py`, 10 frames/camera, conf 0.25, IoU 0.5,
  GT >= 20 px; `out/demo_humans_v2/recall_v7_vs_v6.json`):

| set | v7 CC0 recall per camera | v7 R / P | v6 MetaHuman R / P | GT |
|---|---|---|---|---|
| smoke_mock day | cam_01 0.759, cam_02 0.227 | 0.529 / 0.964 | 0.549 / 0.966 | 51 |
| smoke_mock night | cam_01 0.690, cam_02 0.182 | 0.471 / 0.706 | 0.529 / 0.794 | 51 |
| street_crowd day | 0.910, 0.769, 0.880 | 0.853 / 0.845 | 0.861 / 0.833 | 231 |
| street_crowd night | 0.910, 0.667, 0.853 | 0.810 / 0.899 | 0.810 / 0.866 | 231 |
| disguise day | gate 0.618, ptz 0.923, sub 0.833, yard 0.211 | 0.544 / 0.691 | 0.631 / 0.765 | 103 |
| disguise night | gate 0.559, ptz 0.923, sub 0.833, yard 0.105 | 0.485 / 0.877 | 0.495 / 0.879 | 103 |
| street_crowd_far day | cam_01 0.936, cam_02 0.0, cam_03 0.081 | 0.520 / 0.780 | 0.520 / 0.772 | 150 |
| street_crowd_far night | cam_01 0.949, cam_02 0.0, cam_03 0.081 | 0.527 / 0.681 | 0.513 / 0.748 | 150 |
| **all** | | **0.652 / 0.814** (excl. far 0.703) | **0.665 / 0.823** (excl. far 0.723) | 1 070 |

**The realism cost of compliance, stated plainly:** -0.013 recall overall
(-0.020 excluding the far rig), -0.009 precision. By box height (all sets):
20-60 px 0.362 vs 0.403, 60-120 px 0.862 vs 0.851, 120+ px 0.966 vs 0.957 - the
cost is entirely in the small, distant people, where the low-poly silhouette
and flat vertex-colour clothing carry less than the MetaHuman's; at >= 60 px
the CC0 bodies are detected as well or better. This is ONE nano detector's
recall on 10 frames per camera (noise ~±0.03 per set); it says the CC0 people
are not a detector-visible regression for training-set purposes, not that they
look as real. Close up they are plainly stylised (lineup frame). The v4 gap
(MetaHuman 0.758 vs CesiumMan 0.577) was a different, much weaker baseline.

## Appearance epochs reach the render (2026-09-26) [V]

Before this, the renderer stored ONE outfit per person, so `outfit_change` and
disguise epochs from `gt/identities.parquet` never reached pixels (H10 blocked).

* **Export**: the manifest carries `appearance_epochs[uid]` =
  `[{epoch, frame_start, t_from, t_to, face_visible, appearance_hash, gt_outfit}]`
  for people with ≥ 2 epochs (`survsim.export.unreal_scene.appearance_epochs`).
  `frame_start` is the first `world.parquet` frame whose `appearance_epoch` is
  that epoch (truth is what happened), else `round(t_from*fps)`. Manifests
  exported earlier: `tools/render_run.py` computes the same from the run's GT.
* **Palettes**: `outfits.epoch_outfits` — epoch 0 is the old `outfit()`; later
  epochs take the GT's visible top colour (outer if worn) or a contrasting one,
  the GT bottom if it changed, and for `face_visible=false` a face-cover proxy.
* **Render**: `anim_plan.split_epochs` gives each epoch its own actor
  (`uid#e<k>`), same keys/segments, visibility-keyed to its frame range — an
  exact-to-the-frame swap without a material track. The epoch actor gets the
  person's stencil ID under `--id-mask`.
* **Face cover**: `M_SurvCC0Cloth_v2` adds `CoverOn/CoverZ/CoverCol`; hair,
  eyes and head skin above 0.84 × mesh height take the cover colour (reads as
  a hood/cap/balaclava at CCTV range). CoverOn=0 is identical to v1.
* **`--window T0 T1`** (render_run / render_batch): sequences keyed from 0,
  playback from `frame_start`; files keep absolute GT frame numbers
  (`0950.png` = GT frame 950), so check_render needs no offset.

[V] disguise_and_tamper day, 95–135 s, 4 cams, 1600 frames, gate 39/39 pass
(`out/batch_v7/disguise_and_tamper_day_epochs`); intruder_a switches at render
frame 1152 (dark top → light top + navy cap cover).
Strip: `out/demo_humans_v2/epoch_change_strip.png`.

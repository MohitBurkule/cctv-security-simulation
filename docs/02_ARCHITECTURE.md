# 02 · System Architecture

_Last verified against code: `299e649` (AUDIT_02_03)._

Project codename: **SurvSim** (multi-person-camera-real-sim).

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  USER / RESEARCHER                                                           │
│  python scripts · YAML configs · CLI (survsim run/record/eval)               │
└──────────────┬───────────────────────────────────────────────────────────────┘
               │ survsim Python SDK (pip install -e .)
┌──────────────▼───────────────────────────────────────────────────────────────┐
│  DIRECTOR (Python, engine-agnostic)                                           │
│  • ScenarioLoader  (YAML → typed pydantic models, seeded RNG)                 │
│  • WorldModel      (nav-mesh graph, semantic zones, camera graph, clock)      │
│  • PopulationGen   (identity, body, appearance, gait, role, RF/audio traits)  │
│  • BehaviourEngine (behaviour trees / state machines / scripted timelines,    │
│                     social-force + ORCA planner, anomaly injector)            │
│  • CameraDirector  (PTZ schedules, tamper events, sync jitter)                │
│  • EnvDirector     (time-of-day, weather, lighting, crowd density curves)     │
│  • Fences          (fenced zones/gates, allowed_roles, crosses_fence acts)    │
│  • AppearanceEpochs (outfit/mask/hood changes bump an epoch for ReID GT)      │
│  • GroundTruth     (collects state each tick → writers)                       │
│  • SensorExtras    (RF beacon sim, audio events, access-control logs)         │
└──────────────┬───────────────────────────────────────────────────────────────┘
               │ Backend interface (abstract): spawn/move/animate/set_material/
               │ set_camera/capture(rgb|depth|mask|normal)/query_bones/set_weather
┌──────────────▼───────────────┐   ┌──────────────────────────┐   ┌────────────┐
│ UE5 backend (PRIMARY)        │   │ Isaac Sim backend (opt.) │   │ Mock       │
│ UnrealCV+ TCP  +  SurvSim    │   │ omni.replicator.agent    │   │ backend    │
│ UE plugin (C++/BP):          │   │                          │   │ (headless  │
│  - MetaHuman spawner         │   └──────────────────────────┘   │  2D top-   │
│  - Appearance swapper        │                                  │  down, for │
│  - Motion Matching / Anim BP │                                  │  CI & fast │
│  - Mass AI crowd bridge      │                                  │  behaviour │
│  - CCTV camera actor (PTZ,   │                                  │  iteration)│
│    post-process sensor FX)   │                                  └────────────┘
│  - GT exporter (bones, bbox) │
└──────────────┬───────────────┘
               │ frames + json per tick
┌──────────────▼───────────────────────────────────────────────────────────────┐
│  POST-PROCESS & DATASET                                                       │
│  • Sensor degradation (OpenCV/torch): noise, blur, H.264 (ffmpeg), IR,        │
│    distortion, timestamp burn-in, frame drop, tamper overlays                 │
│  • Writers: MOTChallenge, COCO, MEVA-style events, 3D poses, calib, Parquet   │
│  • Video mux (ffmpeg) per camera, RTSP re-stream (mediamtx) for "live" tests  │
└──────────────┬───────────────────────────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────────────────────────┐
│  EVAL & ALGO DEV SANDBOX                                                      │
│  • Baselines: YOLO/RT-DETR det, ByteTrack/BoT-SORT, OSNet ReID, MTMC          │
│    (Synthehicle-style), behaviour classifiers (rule / GNN / video transformer)│
│  • Metrics: mAP, HOTA/IDF1/MOTA (via TrackEval bridge, `eval/trackeval_bridge`)│
│    MTMC IDF1, ReID mAP/R1, event P/R/F1, alert latency, false-alarms/hour,     │
│    operator-load index                                                        │
│  • Alarms (`eval/alarms.py`): i-LIDS timing, actor-aware matching by GT uid   │
│  • Alarm cards + evidence pack (`eval/alarm_card.py`, `eval/evidence_pack.py`) │
│    — one card per alarm with a leak check, hash-chained WORM log             │
│  • Camera-placement optimiser, NL query benchmark                             │
│  • Experiments framework (`survsim/experiments/`) — named runners (detection, │
│    tracking, reid, alarms, night, dose, autocalib, evasion, ...) over YAML    │
│    configs, each writing a `results.json` with env/git provenance             │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 1. Core concepts & data model

### 1.1 World
* `Environment`: a UE level (map) + `semantic_zones` (polygons on ground plane, tags: `public`, `restricted`, `perimeter_fence`, `gate`, `queue`, `parking`, `sensitive_asset`), `nav_graph` (exported from UE NavMesh, plus hand-placed `waypoints`), `mount_points` (camera mount candidates), `spawn_points`, `exits`, `props` (interactable: doors, bins, cabinets, fence panels, camera housings).
* `Clock`: sim time (`t`), real-world datetime for sun position, FPS, per-camera clock offset & drift.
* Coordinate convention: right-handed, **metres**, Z-up (we convert from UE's cm left-handed at the backend boundary — never leak UE units into the SDK).

### 1.2 Person (Agent)
```
Person
 ├─ identity:      uid (stable, hidden), display_name, group_id
 ├─ body:          height_m, shoulder_w, bmi_class, age_band, sex_presentation, skin_tone(MST 1-10)
 ├─ face:          metahuman_preset, beard, glasses, mask(bool), hood(bool)
 ├─ hair:          style, colour
 ├─ outfit:        layers[ top, outer, bottom, shoes, headwear ], each {mesh, material, base_colour(RGB), pattern, wetness}
 ├─ accessories:   backpack, handbag, suitcase, umbrella, phone, weapon_like_object(prop only), tool(bolt-cutter/spray-can)
 ├─ gait:          style ∈ {normal, brisk, slow, limp_L/R, shuffle, drunk, military, crouch}, speed_mps, arm_swing, stride_len, cadence, posture(upright/slouched)
 ├─ role:          {commuter, tourist, worker, guard, vendor, protester, intruder, courier, maintenance}
 ├─ intent:        schedule of BehaviourNodes (see 03_BEHAVIOUR_CATALOG.md)
 ├─ change_events: [{t, type: outfit_swap|accessory|hair|gait|mask, params}]
 ├─ rf:            emitters[{proto: wifi_probe|ble|lte, mac, tx_power, interval_s, on/off schedule}]
 ├─ audio:         voice_profile, footstep_material, shout_prob
 └─ adversarial:   camera_awareness(0-1), face_hiding(0-1), avoid_zones, timing_gaps_cameras
```

### 1.3 Camera
```
Camera
 ├─ id, mount(pose xyz+ypr, height), type ∈ {fixed_bullet, dome, ptz, fisheye, thermal, body_cam, drone}
 ├─ optics:   focal_mm or hfov_deg, sensor_size, distortion(k1..k3,p1,p2 or fisheye), rolling_shutter_ms
 ├─ capture:  resolution, fps, exposure_mode(auto/fixed), gain_db, wdr, ir_cut(auto), night_mode
 ├─ stream:   codec(h264/h265/mjpeg), bitrate_kbps, gop, packet_loss, latency_ms, clock_offset_ms, drift_ppm
 ├─ ptz:      pan_limits, tilt_limits, zoom_range, speed, patrol_schedule[], auto_track(bool)
 ├─ degrade:  lens_dirt, rain_on_lens, dead_pixels, vignetting, spider_web, compression_level
 ├─ tamper:   schedule[{t, kind: cover|spray|tilt|unplug|glare|cut}]
 └─ overlay:  timestamp(bool, format), cam_name, osd_style
```

### 1.4 Ground truth (per tick, per camera and world)
* **World**: `t`, per-person 3D position (pelvis), velocity, heading, 3D bbox, 55-joint SMPL-X-like skeleton (from MetaHuman bones), current behaviour node + intent label, zone membership, group membership, appearance snapshot hash, RF emissions this tick, audio events.
* **Per camera**: 2D bbox (visible + full/amodal), instance mask PNG, 2D keypoints (COCO-17 + face 5 + hands optional) with visibility flags, occlusion ratio, truncation, depth map (EXR 16-bit), face bbox + face-visible flag, pixel-height, motion blur magnitude.
* **Events**: `[t_start, t_end, type, actor uids, target (zone/prop/person), severity, camera_ids_observing, description]`.
* **Calibration**: K, dist, R, t, homography to ground plane, camera graph (overlap %, mean transit time), blind-spot heatmap.

All of it written as: `frames/{cam}/{000001}.jpg|png`, `gt/{cam}/mot.txt`, `gt/coco_{cam}.json`, `gt/world.parquet`, `gt/events.jsonl`, `gt/calib/{cam}.json`, `meta/scenario.resolved.yaml`, `meta/seed.txt`, `video/{cam}.mp4`.

## 2. Tick loop (Director)

```
for tick in range(N):
    t = tick / fps
    env.update(t)                    # sun, weather, lamps
    for p in people: p.behaviour.step(dt, world) → desired velocity / action
    planner.resolve(people)          # social force + ORCA → collision-free velocities
    backend.apply(people, cameras)   # move actors, set anim params, PTZ moves, appearance changes at t
    backend.tick(dt)                 # advance engine one fixed step (deterministic)
    obs = backend.capture(cameras)   # rgb/depth/mask per camera (batched)
    gt  = backend.query_gt(people, cameras)
    sensors.emit(t, people)          # RF / audio / access events
    writers.write(tick, obs, gt, events)
```
Fixed-step, seeded RNG (numpy `Generator` per subsystem, seed derived from
master seed + subsystem name), deterministic engine stepping
(`FixedFrameRate` + UnrealCV `vset /action/game/pause` + step commands) so a
scenario replays identically. Real-time mode also supported (no capture, just
watch) for debugging.

## 3. UE5 side — SurvSim plugin

Components (C++ where perf matters, Blueprint for glue):
1. **`ASurvSimPersonActor`** — MetaHuman-based character; exposes RPC-able functions: `SetOutfit(json)`, `SetColour(slot, rgb)`, `SetAccessory(name, on)`, `SetGait(style, speed)`, `PlayAction(name, blend)`, `SetLookAt(target)`, `GetBones()`. Uses Motion Matching database per gait style; Control Rig layers for posture (crouch, limp, slouch).
2. **`ASurvSimCCTVCamera`** — SceneCapture2D + PostProcess material stack (noise, vignette, distortion, IR LUT, timestamp widget). PTZ via spring arm + zoom = FOV. `Tamper(kind)` swaps overlay/moves camera. Registers with UnrealCV as camera N.
3. **`USurvSimGTSubsystem`** — per-tick projects bones to each camera, computes visible/amodal bboxes via stencil mask + depth test, occlusion ratio, writes JSON to a ring buffer read over TCP.
4. **`USurvSimWorldSubsystem`** — spawn/despawn, zone volumes, prop interactions (door open, fence cut animation, camera cover), weather parameter binding (Ultra Dynamic Sky or built-in), nav-mesh export.
5. **Mass AI bridge** — background crowd (hundreds) driven by Mass with our density curves; foreground "hero" agents (≤ 40) fully controlled by the Director with bone-level GT. Both share the same appearance system so background crowd is realistic and diverse.
6. **Headless mode** — `-RenderOffscreen -unattended -nosound`, capture via UnrealCV; Linux + Windows; Docker (NVIDIA container toolkit) for batch farms.

## 4. Backend abstraction (Python)

```python
class Backend(Protocol):
    def load_environment(self, env: Environment) -> None
    def spawn_person(self, p: PersonSpec) -> Handle
    def update_person(self, h: Handle, pose: Pose, anim: AnimState, appearance_delta: dict|None) -> None
    def set_camera(self, cam: CameraSpec) -> None
    def ptz(self, cam_id: str, pan: float, tilt: float, zoom: float) -> None
    def set_environment(self, tod: float, weather: Weather) -> None
    def step(self, dt: float) -> None
    def capture(self, cam_ids: list[str], modalities: set[str]) -> dict[str, Frame]
    def query_gt(self) -> WorldState
```
Implementations: `MockBackend` (pure-Python top-down 2D, no GPU — CI and behaviour iteration) and
`UnrealBackend`. **The `unreal` backend route is the UE 5.8 editor + a job runner, not UnrealCV+ over
TCP** (docs/12): the packaged UnrealCV+ runtime hangs at frame 1 and executes no commands as of
2026-09-18. The editor is driven headless (`-run=pythonscript` / MRQ under `xvfb-run`) via
`scripts/ue/*.py` jobs that write a result file (stdout/exit codes prove nothing — the editor
reports success for scripts that did nothing); `scripts/check_job_result.py` and
`scripts/check_render.py` gate every render by named check. `IsaacBackend` remains unbuilt/later.

## 5. Degradation pipeline (post, Python)
Ordered, each stage parameterised & seeded:
`optics(distortion, vignette, chromatic) → exposure(gain, WDR, tone) → sensor(shot noise, read noise, hot pixels, IR mode LUT) → motion(blur from per-pixel velocity buffer, rolling shutter via depth+velocity) → weather_on_lens(rain drops, fog haze, dirt) → codec(ffmpeg h264 crf/bitrate/gop, packet loss, frame drop, duplicate frames) → osd(timestamp, camera name, tamper overlays)`.
The pipeline runs on clean frames so the same scenario can be exported at many quality levels (curriculum training).

`survsim/sensors/degrade.py` ships 7 named `PROFILES`: `clean`, `cctv_day_720p`, `cctv_legacy_sd`,
`cctv_night_ir`, `cctv_night_lowlight`, `cctv_thermal`, `cctv_fog_day`. The day/legacy/thermal/fog
profiles are tuned against real MOTChallenge/UE-render image statistics; **the two night profiles
(`cctv_night_ir`, `cctv_night_lowlight`) are random-searched against a 15-frame LLVIP slice and are
flagged overfit** — LLVIP is non-commercial (measure only, never train/ship) and the real-night
penalty is otherwise unmeasured (H32/H36).

## 6. Licensing gate
MetaHuman/SMPL-X/Reallusion/Mixamo assets forbid ML train/test use (Epic MetaHuman terms; academic
SMPL-X/BEDLAM licences ban surveillance). The default asset stack is CC0 (Quaternius/Kenney/ambientCG/
Poly Haven; `SURVSIM_CHARS=metahuman` is an explicit opt-in, engineering-only). Every rendered set
writes `licences.json` (`survsim/gt/licences.py`) with a top-level `commercial_ml_ok` flag and
per-person `asset_licences`; a set is checked against it before it leaves engineering.

## 7. Non-negotiable principles
1. **Truth is what happened, not what was planned** — record actual post-avoidance state; read boxes
   from rendered masks, not projected ones.
2. **Ground truth follows the live camera** — a slewed/tampered camera's GT matches its live pose, or
   emits nothing.
3. **Hard negatives matter as much as positives** — every scenario pairs a suspicious behaviour with
   an innocent twin that looks identical to a rule.
4. **Report operational metrics** — false alarms/camera-hour against an operator budget, i-LIDS timing
   (late = false alarm), actor-aware alarm matching by GT uid.
5. **One definition of "visible"** — `geometry.usably_visible` (in frame, in front, ≥ 20 px tall) for
   both coverage analysis and "was this observed?".
6. **Determinism** — seeded per-subsystem RNG (`RngHub`); same seed ⇒ same run.
7. **SI units above the backend boundary** — metres/degrees; only the Unreal backend converts
   (×100, negate Y, negate yaw).
8. **Licensing** — MetaHuman renders may not be used to train or test ML; §6 above is the gate.

## 6. Multi-sensor extras
* **RF**: each person's emitters emit probe events; static "sniffers" at configured positions record `(t, sniffer_id, mac, rssi)` with log-distance path loss + shadowing + wall attenuation from zone geometry. MAC randomisation modelled (optional). Truth: mac↔uid mapping.
* **Audio**: event-based (footsteps, shouting, glass break, alarm, engine) with source position; microphones record `(t, mic_id, event, spl_db, doa_deg)` using inverse-square + noise. Optional actual WAV synthesis via freesound clips + pyroomacoustics.
* **Access control / PIR / door contacts / fence vibration**: zone-crossing and prop-interaction triggers produce logs with configurable false-alarm rates.

## 7. Repo layout (target)
```
survsim/                 python package
  core/  (models, rng, clock, geometry)
  director/ (scenario, population, behaviour, planner, camera_director, env_director)
  behaviours/ (library: loiter.py, tamper.py, fight.py, ...)
  backends/ (base.py, unreal/, isaac/, mock/)
  sensors/ (degrade/, rf.py, audio.py, access.py)
  gt/ (writers: mot.py, coco.py, meva.py, parquet.py, calib.py)
  eval/ (det, mot, mtmc, reid, behaviour, alerts, placement)
  cli.py
unreal/SurvSim/          UE5 project + plugin (C++/BP), content submodule via git-lfs
configs/                 environments/, cameras/, people/, behaviours/, scenarios/
docs/
tests/                   unit (mock backend), integration (needs UE), golden GT fixtures
tools/                   nav-mesh export, zone editor (web, three.js), dataset viewer
docker/                  ue-headless.Dockerfile, eval.Dockerfile
```

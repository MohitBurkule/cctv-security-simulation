# 04 · Configuration Schema & Python API

All control is via **YAML (declarative)** + **Python (imperative)**; both map to
the same pydantic models in `survsim.core.models`. Every random choice is
drawn from a seeded generator so `seed` fully determines a run.

## 1. Scenario YAML (top level)
```yaml
schema_version: 1
name: plaza_event_crowd_v1
seed: 1234
duration_s: 600
fps: 25
engine: unreal            # unreal | isaac | mock
environment: !include ../environments/city_plaza.yaml
cameras:     !include ../cameras/plaza_12cam.yaml
time_of_day: {start: "2026-06-14T21:30:00+01:00", speed: 1.0}   # speed>1 = timelapse sun
weather:
  preset: overcast_rain_light      # clear|overcast|rain_light|rain_heavy|fog|snow|dust
  fog_density: 0.05
  rain_intensity: 0.3
  wind_mps: 4
  wetness: 0.8                     # ground puddles/reflections
  schedule: [{t: 300, preset: fog, fog_density: 0.4, ramp_s: 60}]
population:
  background:                      # Mass-AI crowd, GT = boxes only
    count: 300
    density_curve: [{t: 0, mult: 0.6}, {t: 200, mult: 1.0}, {t: 500, mult: 0.3}]
    appearance_pool: !include ../people/pool_urban_summer.yaml
  hero:                            # fully controlled, full GT
    - !include ../people/intruder_a.yaml
    - id: courier_1
      role: courier
      body: {height_m: 1.72, bmi_class: normal, skin_tone: 6, sex_presentation: m, age_band: 25-40}
      outfit: {top: tshirt, outer: hi_vis_vest, bottom: jeans, shoes: trainers, headwear: cap,
               colours: {top: "#2e4a7d", outer: "#f5e600", bottom: "#1c1c2a"}}
      accessories: [backpack_large]
      gait: {style: brisk, speed_mps: 1.6}
      rf: [{proto: wifi_probe, interval_s: 30}, {proto: ble, interval_s: 1}]
      intent:
        - {type: transit, from: spawn.north, to: zone.kiosk}
        - {type: object_abandon, object: backpack_large, zone: zone.bench_row, leave_dist_m: 40}
        - {type: outfit_change, in_zone: zone.blindspot_A, delta: {outer: none, headwear: none}}
        - {type: loiter, zone: zone.gate_B, dur_s: 240, activity: phone, look_at_cameras_p: 0.3}
        - {type: transit, to: exit.south, gait: {style: normal}}
  generators:                      # procedural hero agents
    - {template: loiterer, count: 3, zones: [zone.gate_A, zone.gate_B], dur_s: [120, 400]}
    - {template: fight_pair, count: 1, t_start: [350, 420], zone: zone.center}
    - {template: commute_wait_negative, count: 8}
anomaly_injector:
  enabled: true
  rate_per_min: 0.5
  allowed: [loiter, following, concealment, camera_tamper]
  camera_tamper: {cameras: [cam_07], kinds: [spray, tilt]}
sensors_extra:
  rf_sniffers: [{id: rf_1, pos: [12.0, 4.5, 3.0]}, {id: rf_2, pos: [-30, 10, 3]}]
  microphones:  [{id: mic_1, pos: [0, 0, 4]}]
  access_control: [{id: gate_B_turnstile, zone: zone.gate_B, false_alarm_rate_per_h: 0.2}]
output:
  root: ./out/${name}/${seed}
  modalities: [rgb, depth, instance_mask, normals]
  gt: [mot, coco, keypoints2d, pose3d, world_parquet, events, calib, rf, audio, access]
  video: {codec: h264, crf: 23, gop: 50}
  degrade_profiles: [clean, cctv_720p_h264, cctv_lowlight_ir, cctv_bad_1080p_rain]
  save_clean_and_degraded: true
```

## 2. Environment YAML
```yaml
id: city_plaza
engine_map: /Game/CitySample/Maps/Small_City_LVL      # or unrealzoo scene name
units: m
origin_offset_ue: [0, 0, 0]
nav: {source: ue_navmesh, extra_waypoints: waypoints_plaza.json}
zones:
  - {id: zone.gate_A, tag: gate, polygon: [[10,2],[14,2],[14,6],[10,6]]}
  - {id: zone.restricted_stage, tag: restricted, polygon: [...], allowed_roles: [worker, guard]}
  - {id: zone.blindspot_A, tag: blindspot, polygon: [...]}     # auto-derived too
spawn_points: [{id: spawn.north, pos: [0, 80, 0]}, ...]
exits: [...]
props:
  - {id: bench_1, type: bench, pos: [...], seats: 3}
  - {id: cam_07_mount, type: camera_mount, pos: [...], reachable_height_m: 2.6}
  - {id: fence_seg_12, type: fence_panel, cuttable: true}
lighting: {street_lamps: auto, lamp_colour_k: 3000}
```

## 3. Camera rig YAML
```yaml
cameras:
  - id: cam_01
    type: fixed_bullet
    mount: {pos: [12.3, -8.0, 4.2], yaw: 135, pitch: -22, roll: 0}
    optics: {hfov_deg: 78, distortion: {model: brown, k1: -0.21, k2: 0.04, p1: 0, p2: 0}, rolling_shutter_ms: 12}
    capture: {res: [1920, 1080], fps: 25, exposure: auto, night_mode: auto_ir}
    stream:  {codec: h264, bitrate_kbps: 2500, gop: 50, clock_offset_ms: 180, drift_ppm: 30, packet_loss: 0.002}
    degrade: {lens_dirt: 0.1, vignetting: 0.2}
    overlay: {timestamp: true, format: "%Y-%m-%d %H:%M:%S", name: "CAM01 GATE A"}
  - id: cam_07
    type: ptz
    mount: {pos: [...], yaw: 0, pitch: -15}
    optics: {hfov_deg: [2.3, 60]}                       # zoom range
    ptz: {pan: [-170, 170], tilt: [-90, 10], speed_dps: 60,
          patrol: [{preset: gate_B, dwell_s: 20}, {preset: stage, dwell_s: 15}], auto_track: false}
  - id: cam_09
    type: thermal
    capture: {res: [640, 512], fps: 9, palette: white_hot}
  - id: cam_11
    type: fisheye
    optics: {model: kannala_brandt, fov_deg: 180}
rig_tools:
  auto_place: {mounts_from: props.camera_mount, objective: coverage_of_zones, weights: {restricted: 3, gate: 2}}
```

## 4. Person template YAML (`people/intruder_a.yaml`)
```yaml
id: intruder_a
role: intruder
identity: {group: cell_1}
body: {height_m: 1.81, shoulder_w: 0.47, bmi_class: athletic, skin_tone: 3, sex_presentation: m, age_band: 25-40}
face: {metahuman: MH_Preset_017, beard: stubble, glasses: false}
hair: {style: short, colour: dark_brown}
outfit: {top: hoodie, outer: rain_jacket, bottom: cargo, shoes: boots, headwear: beanie,
         colours: {top: "#3a3a3a", outer: "#0b3d91", bottom: "#4b4b3f"}, pattern: {top: plain}}
accessories: [backpack_small, phone]
gait: {style: normal, speed_mps: 1.35, arm_swing: 0.8, posture: upright}
adversarial: {camera_awareness: 0.8, face_hiding: 0.6}
change_events:
  - {trigger: {zone: zone.blindspot_A}, delta: {outer: none, headwear: none, hair: {style: bald_cap}}, gait: {style: limp_L}}
  - {trigger: {t: 420}, delta: {face: {mask: true}}}
rf: [{proto: wifi_probe, mac: random_rotating, interval_s: 20}]
```

## 5. Python SDK (imperative equivalent)
```python
import survsim as ss

sim = ss.Simulator.from_yaml("configs/scenarios/plaza_event_crowd.yaml", seed=7)

# tweak before start
sim.cameras["cam_07"].ptz.patrol = [ss.PTZPreset("gate_B", dwell_s=30)]
p = sim.people.hero("intruder_a")
p.outfit.colours.outer = "#aa1111"
p.intent.insert(2, ss.behaviours.Loiter(zone="zone.gate_A", dur_s=180, activity="scan"))

sim.env.weather.set(preset="fog", fog_density=0.35, ramp_s=30, at=200)

# run with live hooks (algorithm dev loop)
@sim.on_tick
def my_algo(tick, frames: dict[str, ss.Frame], gt: ss.WorldState):
    dets = my_detector(frames["cam_01"].rgb)
    ...

# or batch record
sim.record(out="out/plaza/7", modalities={"rgb", "depth", "instance_mask"}, degrade=["cctv_720p_h264"])

# programmatic runtime changes
sim.at(t=310).person("courier_1").change_appearance(outer=None, headwear="cap", colours={"top": "#ffffff"})
sim.at(t=330).camera("cam_03").tamper("spray")
sim.at(t=340).spawn_person(ss.PersonSpec.random(role="tourist", seed=99), at="spawn.east",
                           intent=[ss.behaviours.Transit(to="exit.west")])

# query truth
ws = sim.gt.at(t=300)
ws.person("intruder_a").pos_xyz; ws.camera_views("intruder_a")  # -> {cam_id: bbox, visible_ratio}
sim.gt.events.query("type.startswith('hostile.tamper')")

# camera-graph helpers
sim.rig.overlap_matrix(); sim.rig.blind_spots(grid_m=0.5); sim.rig.transit_time_prior("cam_01", "cam_04")
```

## 6. CLI
```
survsim validate  configs/scenarios/x.yaml
survsim run       configs/scenarios/x.yaml --seed 3 --realtime          # watch in engine
survsim record    configs/scenarios/x.yaml --seeds 0-9 --headless --gpus 0,1
survsim degrade   out/x/3 --profile cctv_lowlight_ir
survsim stream    out/x/3 --rtsp :8554                                    # replay as live cams
survsim eval      out/x/3 --task mot --pred preds/bytetrack/ --report html
survsim viewer    out/x/3                                                 # web viewer: sync'd cams + map + events
survsim place-cameras configs/environments/substation.yaml --n 6 --objective coverage
```

## 7. Validation rules (pydantic)
* Zones must be simple polygons inside nav bounds; roles referenced must exist; behaviours must be schedulable (`dur_s` within `duration_s`); camera ids unique; colour hex valid; PTZ presets resolve to yaw/pitch/zoom; sum of hero agents ≤ backend limit (UE: 40 by default, configurable).
* `survsim validate` also dry-runs the Director on the mock backend to detect unreachable targets and behaviour deadlocks (e.g., `wait_for` a person who never arrives).

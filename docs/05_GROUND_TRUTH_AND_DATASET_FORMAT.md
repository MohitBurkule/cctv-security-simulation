# 05 · Ground Truth & Dataset Format

## 1. Directory layout per run
```
out/<scenario>/<seed>/
  meta/
    scenario.resolved.yaml      # every random choice materialised
    environment.json            # zones, props, nav graph
    rig.json                    # camera graph, overlap matrix, blind-spot grid
    people.json                 # full spec per uid incl. appearance timeline
    versions.json               # survsim, UE, plugin, asset pack hashes
  calib/<cam>.json              # K(3x3), dist{model,coeffs}, R(3x3), t(3), homography_ground(3x3), res, fps, clock_offset_ms
  frames/<profile>/<cam>/<frame:06d>.jpg          # profile ∈ clean|cctv_720p_h264|...
  depth/<cam>/<frame:06d>.exr                     # metres, float16
  mask/<cam>/<frame:06d>.png                      # uint16 instance ids (uid index), 0 = bg
  normals/<cam>/<frame:06d>.png                   # optional
  velocity/<cam>/<frame:06d>.exr                  # optional (for motion-blur & optical-flow tasks)
  video/<profile>/<cam>.mp4                       # + .sei_timestamps.json (per-frame pts, drops)
  gt/
    mot/<cam>.txt                # MOTChallenge: frame,id,x,y,w,h,conf(1),class(1),vis
    mot_amodal/<cam>.txt
    coco/<cam>.json              # boxes, keypoints(17+5), segm RLE, attributes
    world.parquet                # per tick per uid: pos, vel, heading, bbox3d, joints3d(55x3), behaviour path, zone, group, appearance_hash, visible_in[cams]
    events.jsonl                 # behaviour/event segments
    identities.parquet           # uid ↔ appearance snapshots ↔ time ranges ↔ RF macs
    rf.parquet                   # t, sniffer, mac, rssi, uid(truth)
    audio.parquet                # t, mic, event, spl, doa, uid(truth)
    access.parquet               # t, sensor, event, uid(truth or null=false alarm)
    camera_status.jsonl          # ptz pose per tick, tamper state, dropped frames
  eval/                          # filled by `survsim eval`
```

## 2. `events.jsonl` record
```json
{"event_id":"e_0042","t_start":311.2,"t_end":338.7,"class":"hostile","type":"tamper","subtype":"camera.spray",
 "label_path":"plaza.intruder_a.camera_tamper.spray","actors":["intruder_a"],"target":{"kind":"camera","id":"cam_03"},
 "zone":"zone.gate_B","severity":0.9,"observed_by":{"cam_02":{"frac_visible":0.7,"mean_px_height":142},"cam_03":{"frac_visible":0.2}},
 "difficulty":{"visibility":0.55,"lighting_lux":4.2,"occlusion":0.3,"score":0.62},
 "hard_negative_for":[]}
```

## 3. Identity & appearance truth
`identities.parquet` rows: `(uid, t_from, t_to, appearance_hash, outfit_json, face_visible_default, gait_style, group_id)`. A change event creates a new row → enables evaluating ReID **within** an appearance epoch and **across** epochs separately.

## 4. Keypoints
COCO-17 + 5 face (eyes, nose, ears already in 17; add mouth corners, chin) + optional 21×2 hands; each with `v ∈ {0 not labelled, 1 occluded, 2 visible}` computed from depth test + mask. 3D joints in world metres from MetaHuman skeleton retargeted to SMPL-X 55 joint names for compatibility with BEDLAM/AMASS tooling.

## 5. Boxes
* `visible` bbox from instance mask; `amodal` from projected 3D skeleton hull; `truncation` flag; `occlusion_ratio = 1 - visible_area/amodal_area`.
* Objects too: bags, tools, drones, vehicles get ids in a separate class range (class 2..).

## 6. Time
Global sim time `t` is truth. Each camera has `pts = t + clock_offset_ms/1000 + drift_ppm*t/1e6 (+ jitter)`; frames dropped per packet-loss model are absent from video but present in `sei_timestamps.json` as `dropped: true`. GT is always indexed by **global frame index** with a per-camera mapping so cross-camera sync is a solvable sub-problem, not assumed.

## 7. Compatibility exports
`survsim export --format {motchallenge, coco, wildtrack, meva, cvat, fiftyone, kitti3d}`. WILDTRACK export writes ground-plane occupancy grids (for MVDet-style methods). FiftyOne dataset for browsing. CVAT XML for manual correction if ever needed.

## 8. Dataset card template
Each released dataset ships `DATASET_CARD.md`: scenarios, seeds, hours, #identities, #events per type, weather/tod histogram, camera types, degrade profiles, known limitations (synthetic gap), licence (assets restrict redistribution → we distribute **generation configs + seeds**, and rendered data only where asset licences allow, e.g. UnrealZoo scenes & MetaHumans under Epic's licence for UE-derived content).

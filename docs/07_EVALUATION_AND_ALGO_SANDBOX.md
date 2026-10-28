# 07 · Evaluation Harness & Algorithm Sandbox

The simulator exists so algorithms mirroring the the vendor suite can be developed and
measured. This doc fixes tasks, baselines, metrics and reports.

## 1. Tasks & metrics
| Task | Input | Truth | Metrics | Baseline to ship |
|---|---|---|---|---|
| Person detection (per camera, per profile) | frames | `coco/<cam>.json` | COCO mAP, AP_small, AR by pixel-height bucket, by lux bucket | YOLOv8/11, RT-DETR |
| Single-camera MOT | video | `mot/<cam>.txt` | HOTA, MOTA, IDF1, IDSW, frag (TrackEval) | ByteTrack, BoT-SORT, OC-SORT |
| Multi-camera tracking (MTMC) | all cams + calib | `world.parquet` | MTMC IDF1/IDP/IDR (Synthehicle/AICity code), ground-plane MODA/MODP (WILDTRACK), handover accuracy | homography-fusion + ReID association; MVDet |
| Person ReID (same-epoch / cross-appearance-epoch / long-term) | crops | `identities.parquet` | mAP, Rank-1/5, per-attribute-change breakdown (jacket removed, mask on, gait change) | OSNet, TransReID, CLIP-ReID |
| Attribute recognition | crops | people.json | per-attribute F1 (colour, headwear, bag, gender-presentation is optional & flagged) | CLIP zero-shot |
| Behaviour / intent classification (per person-segment) | tracks + poses + zones | `events.jsonl` | segment F1 @IoU 0.5, per-type P/R, confusion with hard negatives, latency-to-detect (s) | rule engine (dwell, zone, speed, contact graph); ST-GCN on poses; VideoMAE clip classifier |
| Alerting (operator-level) | all above | events | false alarms / camera-hour, missed critical events, mean time-to-alert, **operator load index** (alerts/hour vs true events) | rule engine w/ tunable thresholds |
| Camera tamper detection | frames | `camera_status.jsonl` | P/R per tamper kind, latency | frame-statistics detector |
| Cross-camera + RF fusion | tracks + rf.parquet | mac↔uid | association accuracy, coverage gain | Hungarian on time/position |
| Camera placement optimisation | env + candidate mounts | zone coverage / MTMC score | coverage %, min pixel height in restricted zones, cost | greedy + CMA-ES |
| NL retrieval | events + attributes | queries.jsonl | Recall@k for queries like "person in red jacket loitering near gate B after 22:00" | CLIP/VLM embedding index + structured filter |

## 2. Report
`survsim eval … --report html` renders one self-contained HTML: per-camera panels, difficulty-bucketed curves, event timeline with detections overlaid, confusion matrices, sim-to-real gap section when real dataset given.

## 3. Reference splits
`configs/splits/`: `train` (envs A-D, seeds 0-79), `val` (seeds 80-89), `test_indomain` (seeds 90-99), `test_env_shift` (envs E-F), `test_appearance_shift` (only cross-epoch ReID), `test_degrade` (same seeds, harsher profiles). Never leak identity presets across splits (identity preset id hashed to split).

## 4. Sandbox conveniences
* `survsim stream` re-streams any run as N RTSP feeds with realistic clock skew ⇒ test real ingestion code.
* `sim.on_tick` hook for online algorithms; `sim.replay(run_dir)` for offline.
* `survsim viewer`: browser UI (three.js map + synced videos + event lane + GT/pred toggles). Reuses FiftyOne when installed.
* Leaderboard JSON so intern results are comparable week to week.

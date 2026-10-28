
Principle: **something end-to-end every 2 weeks**. Mock backend first so the
behaviour/GT stack is testable in CI without a GPU; UE realism grows in
parallel.

| Wk | Milestone | Deliverables | Acceptance |
|---|---|---|---|
| 0 | Setup | Workstation (RTX 4080+/24 GB VRAM ideal, 64 GB RAM, 1 TB NVMe), UE 5.4/5.5 installed, UnrealZoo binaries (subset) + `unrealzoo-gym` running, City Sample opened, MetaHuman plugin, repo/CI skeleton, this plan reviewed | `python -c "import survsim"`; UnrealCV returns an RGB frame from an UnrealZoo scene |
| 1 | Core models + Mock backend | pydantic schemas (04), RNG/clock, YAML loader with `!include`, 2D top-down mock renderer, `survsim validate/run --engine mock` | 20+ unit tests green; a scenario with 10 agents transiting renders a top-down MP4 |
| 2 | Behaviour engine v1 | BT/state machine runner, social-force + ORCA planner, behaviours: transit, stand, queue, group_walk, loiter, following, object_abandon; event/label emission | Mock run emits `events.jsonl`, `world.parquet`; deadlock checker works |
| 3 | UE backend v1 | UnrealCV+ client wrapper, spawn/move existing UnrealZoo human agents, N fixed cameras from YAML, RGB+depth+mask capture, unit conversion, deterministic stepping | Same YAML runs on mock and UE; per-camera MOT GT visually aligned (overlay tool) |
| 4 | **Demo 1**: 4 cams, 10 people, loiter + object_abandon in an UnrealZoo outdoor scene | video grid + GT overlay + events | Internal demo; ByteTrack runs on it, HOTA reported |
| 5 | SurvSim UE plugin: Person actor | MetaHuman spawner, outfit slots + colour/material swap, accessories, hair; appearance timeline & identity epochs | 50 random identities rendered contact sheet; runtime jacket-off works |
| 6 | Animation & gait | Mixamo/CMU import + retarget (IK Retargeter), Motion Matching DB per gait style, Control-Rig posture layers, action anims (phone, sit, crouch-work, fight set, spray, climb, fall) | Gait styles visibly distinct; behaviour catalogue leaves mapped to anims |
| 7 | CCTV camera actor + degrade pipeline v1 | PTZ actor w/ patrol, FOV zoom, SceneCapture; post pipeline: distortion, noise, exposure, h264, OSD; profiles | Same run exported clean + 2 profiles; FID vs WILDTRACK computed |
| 8 | **Demo 2**: transit-station scenario, 8 cams incl. PTZ, hard negatives, degrade | dataset v0.1 (1 h) + eval report (det/MOT/behaviour rule baseline) | Report reviewed; false-alarm analysis on loiter vs commute_wait |
| 9 | Hostile behaviours & props | camera_tamper (all kinds, incl. real in-engine tilt/cover), fence_cut (cuttable fence panel), asset_tamper, fight (multi-agent choreography), theft_snatch, person_down; bystander reaction rules | Each behaviour has a 10-clip smoke test with GT check |
| 10 | Environments kit | border fence (Megascans/Fab), substation, lobby, car park — zones, mounts, nav export, lighting schedules; City Sample plaza integration w/ Mass crowd density control | 6 environments load from YAML; auto-place-cameras produces coverage map |
| 11 | Weather/time + IR/thermal | weather schedule, wet materials, rain-on-lens, fog; IR mode, thermal false-colour from material tags; sun from datetime | Night fog fence probe renders convincingly; thermal profile validated by eye |
| 12 | **Demo 3**: border_fence_night_probe + substation_maintenance_vs_tamper | dataset v0.2 (3 h), MTMC baseline, ReID cross-epoch eval | Metrics table in README |
| 13 | Multi-sensor extras + streaming | RF sniffers, audio events, access logs, RTSP re-stream, viewer | Fusion baseline improves MTMC IDF1 measurably |
| 14 | Scale & batch | headless Docker, multi-seed multi-GPU recording, dataset card, splits, FiftyOne export; performance tuning (≥ 8 cams × 1080p × 25 fps offline ≥ 0.5× realtime) | 10 h dataset generated overnight |
| 15 | Adversarial & NL retrieval | adversarial modifiers, camera-aware pathing, appearance drift; NL query benchmark + VLM baseline | Report: ReID drop under disguise, retrieval R@10 |
| 16 | Wrap-up | docs, tutorials (3 notebooks), final report/presentation, optional Isaac backend spike, hand-over | All configs reproducible from seed; CI green |

**12-week variant**: merge 5+6, 9+10, drop 13 extras to RF only, drop 15 NL.

## Risk register
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| UE learning curve / C++ plugin time | high | high | start with UnrealZoo agents + Blueprint; C++ only for GT exporter; pair with a UE-experienced mentor 1 h/week |
| Asset licences block dataset redistribution | med | med | distribute configs+seeds; keep an "all-redistributable" asset subset (UnrealZoo/Megascans/MetaHuman under Epic terms, Mixamo terms checked) |
| Determinism breaks (physics, Mass crowd) | med | med | fixed timestep, seeded Mass, record actual positions as truth (truth is what happened, not what was planned) |
| Performance of N SceneCaptures | med | high | round-robin capture, lower res for background cams, offline (non-realtime) recording, `r.RHI.Threading` tuning, multi-process per GPU |
| Runtime clothing swap looks fake | med | med | do swaps in blind spots / behind occluders; pre-baked outfit variants per MetaHuman; cloth sim only hero |
| Behaviour looks robotic | med | high | Motion Matching, noise on speed/heading, idle variety, gaze layer, social-force jitter; user study |
| Scope creep (multi-sensor, NL) | high | med | milestones gate features; extras after Demo 3 only |
| Disk/VRAM | med | low | EXR depth half-float, JPEG rgb, per-profile on demand |

## Hardware / software BOM
* GPU RTX 4080/4090 (24 GB) or A5000+, 64 GB RAM, 8+ cores, 2 TB NVMe (UnrealZoo 70 GB + City Sample 100 GB + outputs).
* Ubuntu 22.04 or Windows 11; UE 5.4-5.7 (match UnrealZoo v3.1 → 5.7); Python 3.10+; ffmpeg 6; CUDA 12; Docker + NVIDIA container toolkit for batch.
* Optional: Ultra Dynamic Sky (~$50), Fab clothing packs (~$20-60 each), Mixamo (free).

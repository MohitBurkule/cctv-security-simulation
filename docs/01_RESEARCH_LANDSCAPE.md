# 01 · Research Landscape — What Exists, What We Reuse, What We Build

Goal of this document: exhaustive survey of existing engines, datasets, tools and
papers that can be reused for a **photorealistic, multi-camera, multi-person,
behaviour-controllable 3D surveillance simulator**, followed by a justified
choice of the stack.

---

## 1. Requirements the stack must satisfy (derived from the vendor's product surface)

the vendor (the vendor, "vendor lab") sells an ML suite that:

| the vendor capability | What the simulator must provide so we can develop / test it |
|---|---|
| Use existing CCTV around events / buildings / borders | Fixed, PTZ, dome, bullet, thermal, low-res, compressed, badly-placed cameras. Many of them. Realistic mounts (poles, wall corners, ceilings). |
| Identify people | Ground-truth persistent identity per person, face visibility, biometrics-like stable attributes (height, gait, build), plus deliberate disguise changes. |
| Track them (same view, across views, over time, over space) | Per-frame 2D boxes / masks / keypoints in every camera, 3D world position, camera calibration (intrinsics + extrinsics), overlapping AND non-overlapping camera topologies, time-sync jitter. |
| Classify intent / behaviour: loitering, tampering, unusual posture/movement, aggression | A scriptable **behaviour engine** with a labelled catalogue of normal and anomalous behaviours, with per-frame and per-segment labels and world-space event logs. |
| People change over time / act deliberately | Runtime appearance change (jacket off, hat on, bag handed over), gait change, group split/merge, adversarial behaviour (camera-aware avoidance, hiding face). |
| Poor lighting, weather, degraded imagery | Day/night cycle, rain/fog/snow, sensor noise, motion blur, compression, IR/thermal mode, lens dirt, dead pixels, frame drops. |
| Multi-sensor (RF, audio, wired/wireless, comms) | Synthetic RF beacons (phone MAC/BLE-like emitters carried by people), directional audio events, door/turnstile/PIR sensor logs — all time-aligned with video. |
| Broadcast / media monitoring, natural-language retrieval | A queryable event & attribute index (JSON/Parquet) so NL search ("man in red jacket loitering near gate B after 22:00") can be validated against truth. |
| Multiple environments | Border fence, critical infrastructure (substation / plant), city plaza / event venue, transit station, building lobby, car park. |

Non-functional: **extremely realistic visuals** (not toy), controllable entirely
from code (Python) + config (YAML), reproducible (seeded), scalable to
headless batch generation, runs on a single workstation GPU, permissive

---

## 2. Candidate engines / platforms

### 2.1 Unreal Engine 5 + UnrealCV / UnrealZoo  ⭐ PRIMARY CHOICE

* **UnrealZoo** ([paper](https://arxiv.org/abs/2412.20977), [repo](https://github.com/UnrealZoo/unrealzoo-gym), ICCV 2025 Highlight): 100+ photorealistic UE scenes, 10+ playable agent types incl. **human characters with everyday animations**, Python API (gym-style) built on **UnrealCV+**; observations: RGB, depth, object/instance masks, panorama, LiDAR; multi-agent heterogeneous control; Apache 2.0; Windows/Linux/macOS; ~70 GB binary bundle (v3.1); UE 5.7; 50+ FPS 2K capture.
* **UnrealCV** ([site](https://unrealcv.org/)): the underlying plugin — `vget /camera/N/lit|depth|object_mask`, spawn/move actors, set materials, set camera FOV/pose, multiple cameras. This is exactly the "get a view from every camera" primitive.
* **City Sample** (Epic, free on Fab, [docs](https://dev.epicgames.com/documentation/unreal-engine/city-sample-project-unreal-engine-demonstration)): full city + **Mass AI MetaHuman crowds** (thousands of pedestrians, LOD'd), Nanite/Lumen, day/night. Perfect "event / urban plaza" environment and a ready crowd baseline.
* **MetaHuman Creator** (free with UE): photoreal humans, body types, skin tones, hair, clothing; groom; facial rig. Runtime material/mesh swaps possible → **clothing / colour change**.
* Animation sources: Mixamo (free, huge locomotion + fight + idle library), CMU MoCap (free), **BEDLAM** ([paper](https://arxiv.org/abs/2306.16940)) SMPL-X motions, UE Motion Matching (5.4+) for natural locomotion, Control Rig for procedural posture edits (limp, crouch, hands-in-pockets).
* Weather/lighting: Ultra Dynamic Sky (paid, ~$50) or free Sky Atmosphere + Volumetric Cloud + Niagara rain; Lumen GI handles night lamps/headlights.
* Mass AI / StateTree / Behaviour Trees / Smart Objects: native tooling for scalable, scriptable NPC behaviour. Zone Graph for pedestrian lanes.
* **Pros**: best realism available for free, existing Python bridge, existing scenes & humans, industry-standard, the the vendor-relevant papers (MATRIX, UAVT, SegGen, StereoGenBench) all use UE5.
* **Cons**: heavy (100+ GB disk, RTX 3070+ recommended), C++/Blueprint learning curve, UnrealZoo's human animation set is generic — we must build the behaviour catalogue ourselves.

### 2.2 NVIDIA Isaac Sim + Replicator Agent (IRA)  — SECONDARY / BENCHMARK
* [Replicator Agent docs](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/replicator_tutorials/tutorial_replicator_agent.html), [Animated People Controller](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/action_and_event_data_generation/ext_replicator-agent/ext_omni_anim_people.html).
* Purpose-built for exactly our task ("intelligent video analytics"): YAML config of environment, N cameras, characters, command file (`Tom GoTo 10 0 0`, Idle, LookAround, Sit, Queue), annotators (2D/3D bbox, seg, skeleton, depth), randomizers.
* **Cons**: omni.anim.people is being **deprecated**; only 5-6 behaviours; RTX-only; visuals decent but less than UE5 Lumen+MetaHuman; large SDK; character library small. Good as a **reference design** for our YAML/command format and as a cross-engine validation target (Phase 5 optional).

### 2.3 CARLA (UE4/UE5, MIT)
* Walker blueprints (~40), `WalkerAIController` (`go_to_location`, `set_max_speed`), bone retrieval API, weather API, RGB/depth/semantic/instance sensors, 8 towns. Recent work ([ARCANE-PedSynth](https://arxiv.org/pdf/2605.24950)) uses hybrid AI/manual pedestrian control.
* **Cons**: vehicle-centric, pedestrians low-fidelity, weak indoor/critical-infra scenes, limited appearance control. Useful only as a quick "town street" fallback.

### 2.4 Unity: PeopleSansPeople / Perception / MultiviewX / CrowdSim
* [PeopleSansPeople](https://github.com/Unity-Technologies/PeopleSansPeople) (Apache 2.0): 28 RenderPeople humans, 39 clips, Shader-Graph clothing texture randomizers, COCO output; images not video; Unity 2020.3.
* [MultiviewX](https://github.com/hou-yz/MultiviewX): Unity, replicates WILDTRACK camera rig; PersonX models.
* [GMVD](https://github.com/jeetv/GMVD): GTA V + Unity, 7 scenes, varied camera layouts.
* CrowdSim / CrowdSim2 ([paper](https://arxiv.org/pdf/2304.05090)): Unity crowd generator for detection.
* **Verdict**: good ideas to copy (randomizer design, WILDTRACK-style rig, dataset format), but Unity HDRP realism < UE5 and less free photoreal content.

### 2.5 GTA V mods: JTA / MOTSynth / MTA
* [MOTSynth](https://arxiv.org/abs/2108.09518): 768 sequences, 1.3M frames, 9.5k identities from 579 ped models with clothes/bag/mask/hair variation; boxes, masks, pose, depth. [JTA-Mods](https://github.com/fabbrimatteo/JTA-Mods) (ScenarioCreator + DatasetAnnotator via ScriptHookV), [MTA-Mod](https://github.com/koehlp/MTA-Mod) for multi-camera.
* **Verdict**: excellent proof that diversity > volume; but proprietary game, EULA risk, Windows-only, fragile mods. Use as a **dataset-format and diversity benchmark**, not as engine.

### 2.6 Blender: Infinigen / Infinigen Indoors / BlenderProc
* Infinigen Indoors ([paper](https://arxiv.org/pdf/2406.11824)): procedural photoreal interiors from Python DSL, constraint solver, 79 asset generators. BlenderProc: rendering + annotations pipeline. Humans via SMPL-X + BEDLAM assets.
* **Verdict**: offline path-tracing = minutes/frame ⇒ unusable for interactive multi-camera video at scale. Keep as **optional "hero shot" renderer** and as a source of procedural indoor layouts exported to UE (FBX/USD).

### 2.7 Behaviour / crowd simulation libraries (engine-agnostic)
* RVO2 / ORCA (Apache), Menge, PedSim (social forces), SUMO (pedestrians), MassMotion & Pathfinder (commercial; used by [SIMCD](https://www.sciencedirect.com/science/article/pii/S0957417422008065) and crowd-anomaly work) — we implement social-force + ORCA in Python for the **plan layer** and let UE Mass AI do local avoidance/animation.
* Anomaly datasets to mirror label taxonomy: UCF-Crime, ShanghaiTech, Avenue, UBnormal (synthetic, Cinema4D), MEVA (real multi-camera activity, 37 activity types), VIRAT, PETS2009, WILDTRACK, MultiviewX, DukeMTMC, MMPTRACK, SoccerNet-Tracking. **UBnormal + MEVA taxonomy** are the closest to the vendor's "loitering/tampering/aggression" list.

### 2.8 Related synthetic multi-view work (methodology to borrow)
* MATRIX ([arXiv 2511.08615](https://arxiv.org/html/2511.08615)): UE5 + AirSim, 8 moving drone cameras, multi-view pedestrian tracking.
* Synthehicle ([repo](https://github.com/fubel/synthehicle)): CARLA, multi-target multi-camera tracking dataset; MTMC evaluation code reusable.
* SegGen ([repo](https://github.com/Secure-and-Intelligent-Systems-Lab/SegGen)): UE5 PCG procedural environments + RGB/depth/seg.
* Multi-View Pedestrian Occupancy Prediction synthetic dataset ([arXiv 2412.13569](https://arxiv.org/pdf/2412.13569)).
* "Optimizing Camera Configurations for Multi-View Pedestrian Detection" ([arXiv 2312.02144](https://arxiv.org/pdf/2312.02144)) — camera-placement search, a natural downstream use of our simulator.
* Densely annotated photorealistic virtual dataset for abnormal event detection ([Springer](https://link.springer.com/chapter/10.1007/978-3-030-68799-1_1)).

---

## 3. Decision matrix

| Criterion (weight) | UE5+UnrealCV/Zoo | Isaac Sim IRA | CARLA | Unity PSP | GTA V mods | Blender |
|---|---|---|---|---|---|---|
| Photorealism (5) | 5 | 4 | 3 | 3 | 4 | 5 |
| Realtime multi-cam video (5) | 5 | 4 | 4 | 4 | 3 | 1 |
| Human appearance control (4) | 5 (MetaHuman) | 2 | 2 | 3 | 3 | 4 |
| Behaviour scripting (5) | 4 (BT/StateTree/Mass) | 2 (deprecated) | 2 | 2 | 2 | 2 |
| Ready environments (4) | 5 (100+ scenes, City Sample, Fab) | 3 | 3 | 2 | 5 | 3 |
| Python API (4) | 5 | 5 | 5 | 3 | 2 | 5 |
| Ground-truth breadth (4) | 4 | 5 | 4 | 4 | 3 | 5 |
| Licence / cost (3) | 5 (free, Apache) | 4 (free, NVIDIA EULA) | 5 (MIT) | 5 | 1 (EULA) | 5 |
| Hardware (2) | 3 | 2 (RTX only) | 3 | 4 | 3 | 3 |
| Community / longevity (3) | 5 | 4 | 4 | 3 | 2 | 4 |
| **Weighted total** | **178** | 141 | 131 | 121 | 111 | 132 |

**Decision: Unreal Engine 5.4+ as the engine; UnrealCV+ (from UnrealZoo) as the
Python bridge, extended by our own `SurvSim` UE plugin; MetaHumans + City
Sample crowd tech for people; our own Python "director" for behaviour and
scenario orchestration; Isaac Sim IRA YAML as the format inspiration and an
optional second backend behind the same Python API.**

---

## 4. Reuse inventory (concrete artefacts we pull in)

| Artefact | Source | Licence | Use |
|---|---|---|---|
| UnrealCV+ plugin & Python client | UnrealZoo | Apache 2.0 | camera capture, actor control, masks, depth |
| UnrealZoo scene binaries (subset: `SuburbNeighborhood_Day/Night`, `ModularSciFiVillage`, `Greek_Island`, `TrainStation`, `ContainerYard`, `Factory` etc.) | UnrealZoo | Apache 2.0 (scenes have per-asset Fab licences, redistribution restrictions) | ready environments for early milestones |
| City Sample project | Epic / Fab | Epic Content Licence (free, UE-only) | urban plaza / event environment + Mass crowd |
| MetaHuman Creator presets + MetaHuman Animator | Epic | Epic licence | person appearance |
| Mixamo animation library | Adobe | free w/ account | locomotion, fights, idles, gestures |
| CMU MoCap (BVH) | CMU | free | extra gait variety |
| BEDLAM motions (AMASS-derived) | MPI | research licence | gait / body-shape diversity, clothing sim refs |
| Quixel Megascans (free for UE) | Epic | Epic licence | fences, concrete, industrial props |
| Fab free assets: fences, CCTV camera meshes, turnstiles, substation props | Fab | per-asset | environment building |
| RVO2 (Python port) | Apache | local avoidance in planner |
| TrackEval, motmetrics, py-motmetrics | MIT | evaluation |
| MOTChallenge / COCO / MEVA schema | public | dataset format |
| Isaac Sim IRA config schema | NVIDIA docs | inspiration for our YAML |

---

## 5. Gaps we must build ourselves

1. **Behaviour & intent engine** — hierarchical: scenario → roles → behaviour trees → motion primitives → animation; with anomaly injection and labels. Nothing off-the-shelf covers loitering/tampering/aggression with labels.
2. **Appearance mutation at runtime** — clothing colour/material swap, garment add/remove, accessories (hat, hood, mask, bag), hairstyle; with persistent hidden identity.
3. **CCTV sensor model** — PTZ control, lens distortion, rolling shutter, H.264 artefacts, low-light noise, IR/thermal false-colour, timestamp overlay, frame drops, tamper (spray, cover, tilt).
4. **Multi-camera topology tooling** — camera placement from YAML, calibration export, overlap graph, blind-spot map, camera-to-camera transit time truth.
5. **Ground-truth writer** — synchronized MOT/COCO/keypoint/3D/identity/behaviour/event outputs + per-camera calib + world map.
6. **Multi-sensor extras** — RF emitter simulation, audio event generation, access-control sensor logs.
7. **Evaluation harness** — detection, MOT, MTMC, ReID, behaviour classification, alert precision/recall vs operator load, camera-placement optimisation.
8. **Environment kit** — border fence line, substation, station, lobby, plaza; each with camera mounts and semantic zones (restricted, entrance, queue, perimeter).

Everything else is reuse.

---

## Sources
- [UnrealZoo paper](https://arxiv.org/abs/2412.20977) · [unrealzoo-gym](https://github.com/UnrealZoo/unrealzoo-gym) · [UnrealCV](https://unrealcv.org/)
- [Isaac Sim Replicator Agent](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/replicator_tutorials/tutorial_replicator_agent.html) · [Animated People Controller](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/action_and_event_data_generation/ext_replicator-agent/ext_omni_anim_people.html) · [Omniverse Replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html)
- [City Sample](https://dev.epicgames.com/documentation/unreal-engine/city-sample-project-unreal-engine-demonstration) · [City Sample crowds tutorial](https://dev.epicgames.com/community/learning/tutorials/aw5o/unreal-engine-5-5-how-to-setup-city-sample-crowds-metahumans-massai-part-1)
- [MOTSynth](https://arxiv.org/abs/2108.09518) · [JTA-Mods](https://github.com/fabbrimatteo/JTA-Mods) · [MTA-Mod](https://github.com/koehlp/MTA-Mod)
- [PeopleSansPeople](https://github.com/Unity-Technologies/PeopleSansPeople) · [Unity Perception](https://github.com/Unity-Technologies/com.unity.perception) · [MultiviewX](https://github.com/hou-yz/MultiviewX) · [GMVD](https://github.com/jeetv/GMVD)
- [BEDLAM](https://arxiv.org/abs/2306.16940) · [Infinigen Indoors](https://arxiv.org/pdf/2406.11824)
- [Synthehicle](https://github.com/fubel/synthehicle) · [SegGen](https://github.com/Secure-and-Intelligent-Systems-Lab/SegGen) · [MATRIX](https://arxiv.org/html/2511.08615) · [Camera-config optimisation](https://arxiv.org/pdf/2312.02144)
- [SIMCD](https://www.sciencedirect.com/science/article/pii/S0957417422008065) · [CrowdSim2](https://arxiv.org/pdf/2304.05090) · [ARCANE-PedSynth (CARLA)](https://arxiv.org/pdf/2605.24950)
- [a CCTV-analytics vendor]((vendor source removed)) · [the vendor Our Work]((vendor source removed))

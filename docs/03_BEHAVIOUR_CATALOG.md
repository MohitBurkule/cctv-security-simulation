# 03 · Behaviour & Intent Catalogue

_Last verified against code: `299e649` (AUDIT_02_03)._

Every behaviour is a node usable in a person's `intent` timeline or triggered by
the anomaly injector. Each has: parameters, motion primitives it composes,
required animations, GT label(s) emitted, and the detection cue it exercises
(what an the vendor-style module would have to see). Labels follow a 3-level taxonomy:
`class` (normal / suspicious / hostile), `type`, `subtype`.

## 0. Motion primitives (leaf actions; engine-executed)
`walk_to(target, gait, speed)` · `stand(idle_style, dur)` · `turn_to(yaw)` · `look_at(target|camera, dur)` · `sit(seat)` · `crouch(dur)` · `lie_down(dur)` · `run_to(target)` · `interact(prop, anim)` · `carry(object)` · `drop(object)` · `pickup(object)` · `hand_over(object, person)` · `gesture(name)` (wave, point, phone_call, smoke, argue, shove, punch, kick, push, grab) · `climb(prop)` · `jump` · `fall` · `wait_for(person|event)` · `follow(person, dist)` · `flee(from)` · `change_appearance(delta)`.

## 1. Normal behaviours (background & hard negatives)
| Type | Params | Composition | Cue exercised |
|---|---|---|---|
| `transit` | origin, dest, gait, speed_jitter | walk_to along nav path, occasional look_around | baseline tracking, ReID |
| `commute_wait` | zone=queue/platform, dur 30-900 s, phone_use p | stand/idle variants, small shuffles, sit | **loitering false-positive** hard negative |
| `queue` | lane waypoints, service_time dist | follow leader, step forward | crowding vs anomaly |
| `group_walk` | group_id, formation(line/abreast/V), chatter | coordinated walk_to, look_at peers | group tracking, split/merge |
| `meet_and_greet` | partner, dur | walk_to, wave, stand facing, hug/handshake | close-contact vs aggression negative |
| `patrol_guard` (role guard) | zones (default: every fenced area), offset_m 1.0, check_s, rounds | walk the fence ring 1 m outside, in by the gate, check the asset, out by the gate (`normal.patrol`) | **line-cross / probe / cut / climb hard negative** (H40b): same fence path and a real crossing, by someone entitled |
| `maintenance_visit` (role worker) | zone, dur_s 60-240 | gate → inside → kneel at asset → gate (`normal.maintenance.visit`) | **zone_intrusion hard negative** (H40b): deep, stationary, authorised |
| `staff_escort` (role worker) | target, gap_m | follow a badged colleague through the fence gate to the job, stop when their work ends (`normal.maintenance.escort`) | **fence_tailgate hard negative** (H54): identical geometry and timing, only the badge differs |
| `vehicle_move` | to, speed_mps | drive a route, faster/larger than a pedestrian (`normal.vehicle.transit`) | background vehicle traffic; twin of `vehicle_ram`'s cruise phase |
| `wildlife` (object_class animal) | zone, radius_m, top_mps 2.5-5, dur_s | fast bursts, stops, direction changes, goes under fences, leaves (`distractor.wildlife`); height < 1 m; MOT class 8 (distractor, consider 0) | **i-LIDS sterile-zone distractor**: never a truth event, exercises height/age gating |
| `maintenance` | asset, tool | kneel/crouch at asset, interact(long) | **tamper false-positive** |
| `photograph` | target, dur | stand, gesture(camera), look_at | reconnaissance negative |
| `smoke_break` | zone, dur 3-8 min | stand, gesture(smoke), pace 2-4 m | loitering negative |
| `lost_tourist` | random waypoints, map look | erratic path, stop-start, look_around | unusual-movement negative |
| `child_play` | zone, energetic | run, jump, zigzag | erratic motion negative |
| `drunk_walk` | sway amp | gait=drunk, stop, lean | posture negative |
| `disabled_mobility` | wheelchair/crutch | slow, different silhouette | detector fairness |
| `weather_reaction` | rain | run for cover, umbrella open, hood up | appearance change negative |
| `sit_rest` | bench, dur | sit, phone, sleep (lie) | lying-down negative |

**Fences (H40b, `survsim/director/fences.py`).** Zones tagged `restricted` / `perimeter_fence` are fenced
(set `fenced: false` for a painted-line zone). Overlapping fenced polygons form one area with one gate prop.
An unauthorised person (role not in `allowed_roles`) is stopped at the fence on the actual position, and
`transit` routes round it. Only behaviours with `crosses_fence` get through: intrusion, cut, climb,
climb_attempt, vehicle_ram and wildlife. `perimeter_site` has a fenced compound whose outer 3 m is the
sterile strip.

## 2. Suspicious behaviours
| Type | Params | Composition | GT label | Cue |
|---|---|---|---|---|
| `loiter` | zone, dur ≥ T_loiter (e.g. 120 s), radius, activity(idle/pace/phone/scan) | stand/pace within radius, periodic look_around, look_at cameras p | `suspicious.loiter.{idle,pace,scan}` | dwell-time per zone, revisit |
| `loiter_intermittent` | leave-and-return n, gap | loiter → transit out of view → return | `suspicious.loiter.revisit` | cross-camera re-ID over time |
| `perimeter_probe` | fence segment, approach_dist | approach, walk along fence, touch, look over, retreat | `suspicious.probe.fence` | proximity to restricted line |
| `recon` | targets[] (cameras/doors/guards), dwell | walk_to vantage, look_at each target ≥ 3 s, gesture(photo/phone), note-taking | `suspicious.recon` | gaze/orientation to assets |
| `tailgating` (class `Tailgate`) | target person, door | follow(target, 1 m), enter on same badge event | `suspicious.access.tailgate` | door event ↔ people count |
| `fence_tailgate` | zone/segment (gate), wait_s | wait outside a fence gate, follow an admitted person through it within seconds, dwell at the asset, leave by the gate | `suspicious.access.tailgate` | access-control record vs person count at the gate; twin `staff_escort` (H54) |
| `following` (class `Follow`) | target, dist 5-15 m, dur | follow with speed matching, stop when target stops | `suspicious.follow` | trajectory correlation |
| `object_abandon` | bag, zone | carry → put down → walk away ≥ 20 m, no return | `suspicious.object.abandon` | static object + owner leaving |
| `object_handoff` | partner, object | meet, hand_over, split directions | `suspicious.object.handoff` | brief contact, object transfer |
| `concealment` | mask/hood/hat at t | change_appearance(face_hiding↑) when near camera | `suspicious.appearance.conceal` | ReID robustness, face-hiding |
| `outfit_change` | out-of-view zone, new outfit | walk to blind spot → swap top/outer → re-emerge | `suspicious.appearance.change` | **cross-camera ReID under clothing change** |
| `camera_avoidance` | awareness 1.0 | path planning penalises camera FOV cells, head down | `suspicious.evasion.camera` | coverage gaps analysis |
| `counter_flow` | crowd direction | walk against dominant flow | `suspicious.motion.counterflow` | flow field anomaly |
| `erratic_motion` | zigzag, sudden stops | random-walk within zone | `suspicious.motion.erratic` | trajectory entropy |
| `wrong_zone_entry` / `zone_intrusion` | restricted zone, no role=worker, dur_s, depth_m 3.5 | cross the boundary at the nearest point (not the gate), go ≥ 3.5 m in, visit the asset, pause, leave the way it came (H40b; was: stop at the edge) | `suspicious.access.intrusion` | zone rules + role/badge truth |
| `climb_attempt` | fence/wall | approach, climb (partial), drop back | `suspicious.probe.climb` | posture/vertical motion |
| `fence_climb` | segment (panel or fence point), climb_s 6-12, dwell_s | approach, climb (feet cross the line in the last third), ≥ 3.5 m in, asset, pause, leave; generator `climb_arc` (probe → climb → run), twin `patrol_guard` | `hostile.breach.climb` | posture + crossing |
| `vehicle_dropoff_loiter` | vehicle actor | exit vehicle, loiter, re-enter | `suspicious.vehicle.loiter` | vehicle+person association |
| `crowd_form` | n people converge | gather within 3 m radius, stand | `suspicious.crowd.gather` | density spike |
| `night_presence` | tod ∈ closed hours | any presence in zone | `suspicious.temporal.afterhours` | schedule context |

## 3. Hostile / tamper / aggression behaviours
| Type | Params | Composition | GT label | Cue |
|---|---|---|---|---|
| `camera_tamper` | camera_id, kind ∈ {cover, spray, tilt, cut, glare, unplug} | approach camera mount, reach/climb, interact(anim per kind) → camera degrade event | `hostile.tamper.camera.{kind}` | frame stats (dark, blur, shift), person under camera |
| `fence_cut` | segment, tool=bolt_cutter, dur 20-90 s, dwell_s | crouch at fence, interact, fence panel opens, pass through ≥ 3.5 m, asset, pause, leave (H40b: really crosses; twin `patrol_guard`) | `hostile.tamper.fence.cut` | posture + duration + zone |
| `asset_tamper` | prop (cabinet, valve, transformer), dur | open panel, kneel, tool use | `hostile.tamper.asset` | proximity to sensitive_asset + posture |
| `vandalism` | wall/surface, dur_s | spray/scratch gesture at a surface | `hostile.vandalism` | repetitive arm motion |
| `theft_snatch` | victim, object | approach fast, grab, flee (run) | `hostile.theft.snatch` | speed spike + object transfer |
| `argument` | partner(s), escalation curve | stand facing, gesture(argue), step-in, point | `hostile.aggression.verbal` | posture, proximity, gesture |
| `fight` | participants n≥2, dur, intensity | shove → punch/kick → grapple → fall; bystanders form ring / flee | `hostile.aggression.fight` | rapid limb motion, falls, crowd reaction |
| `crowd_rush` | target point | run converge on a point (deliberately weak avoidance — a real surge) | `hostile.crowd.rush` | flow anomaly |
| `person_down` | cause: fall/faint/attack | fall → lie_down, others gather or ignore | `hostile.medical.person_down` | lying posture; twin `sit_rest` (seated first, never bare prone) |
| `throw_object` | target zone | wind-up, throw, projectile | `hostile.throw` | arm motion + projectile |
| `drone_overflight` | to, speed_mps, altitude_m | flight path over the perimeter, no foot contact | `hostile.airspace.drone` | small object detection; breaks the ground-plane assumption (homography localisation) |
| `vehicle_ram` | vehicle actor, barrier | high speed approach, accelerating from a normal cruise, breaches barrier | `hostile.vehicle.ram` | speed vector; twin `vehicle_move`'s cruise |

Not yet behaviour classes (catalogue rows only, no `REGISTRY` entry): `assault_weapon_like`,
`push_over_barrier`.

## 4. Deliberate adversarial modifiers (composable with any behaviour)
* `appearance_drift(t_curve)` — gradual: jacket unzipped → removed → tied at waist; hat on/off; glasses; hair tied.
* `gait_mask` — change stride/cadence/limp on cue to defeat gait ReID.
* `camera_aware_timing` — perform key action only when PTZ points away or in camera-graph blind windows.
* `decoy_partner` — a second agent creates distraction (fall, argument) in another camera.
* `crowd_cover` — perform action only when local density > threshold.
* `role_disguise` — wear hi-vis/uniform matching `worker` role while being `intruder`.

## 5. Scenario templates (compose many agents)
1. **border_fence_night_probe**: 4 km fence segment section (≈400 m rendered), 6 fixed + 2 PTZ thermal-capable cameras, animals as distractors, 2 intruders (recon → fence_cut → cross → flee), 1 patrol guard, fog 0.3, moonlight.
2. **substation_maintenance_vs_tamper**: 5 cameras, 2 legit maintenance workers, 1 intruder in hi-vis doing asset_tamper, 1 drone_intrusion.
3. **plaza_event_crowd** (City Sample): 12 cameras, 300 background, 30 hero agents incl. object_abandon, fight, theft_snatch, outfit_change, protest gather.
4. **transit_station_rush**: 8 cameras incl. fisheye, queues, tailgating at gates, loiter vs commute_wait hard negatives, person_down.
5. **corporate_lobby_afterhours**: 4 indoor cameras, cleaner role, tailgating, camera_tamper (spray), concealment.
6. **car_park_multilevel**: low light, vehicle_dropoff_loiter, theft, camera occlusion by pillars, non-overlapping cameras → transit-time ReID.

## 6. Label emission rules
* Per tick each person has exactly one `active_behaviour` (leaf) and a stack of parents; GT stores the full path (`plaza.intruder.loiter.pace`).
* Events are emitted at behaviour start/end with `severity` (0-1) and `confidence` = 1.0 (truth) but also a **plausible-observer difficulty score** computed from visibility, pixel height, occlusion and lighting so downstream evaluations can bucket by difficulty.
* Hard negatives are tagged `hard_negative_for: [loiter, tamper]` so precision analysis is meaningful.
* Aggression intensity, loiter dwell seconds, distance-to-asset are stored as continuous attributes per tick for regression-style tasks.

## 7. Animation asset requirements (procurement list)
Locomotion sets (walk/jog/run/sprint × normal/brisk/slow/limp/drunk/crouch/military) · idles (10+) · phone use · smoking · sit/stand transitions · kneel/crouch work loops · climb fence (low/high) · fall variants · fight set (Mixamo "Fighting" pack: punches, kicks, hits, grapple) · shove/push · throw · spray paint · bolt cutter · put down/pick up bag · hand over · wave/handshake/hug · look-around head layer (Control Rig aim) · umbrella open · jacket remove (or cut to mesh swap behind occluder) · wheelchair/crutch (optional).

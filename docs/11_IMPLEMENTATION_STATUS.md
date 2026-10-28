# 11 · Implementation status — plan vs. code

**1079 tests collected (1053 selected by default; working tree incl. in-flight H59, 2026-09-27). Fast suite: 1052 passed, 1 xfailed, 0 failed (`pytest -q -n auto`, 87 s).** ~12,000 lines of package code, 32 documents. Everything
runs with no GPU and no engine except the Unreal path, which is exercised
against a fake UnrealCV client in tests, statically checked by
`tests/test_ue_scripts.py`, and verified by pixels through `tools/render_run.py`.

Tests that need downloaded data (MOT20) skip cleanly when it is absent, so CI
stays offline-capable.

---

## Capability coverage against what the company does

| the vendor capability | State | Where |
|---|---|---|
| Person detection | done — stratified AP, FPPI → alarms/camera-hour | `eval/detection.py` |
| Single-camera tracking | done — MOTA/IDF1/HOTA with DetA/AssA split; M1 parity with reference TrackEval (`eval/trackeval_bridge.py`, MOT20 HOTA 0.443, docs/13) | `eval/tracking.py` |
| Multi-camera tracking | done — global IDF1 plus `handoff_gap` | `eval/tracking.evaluate_mtmc` |
| Re-identification | done — scored within *and across* appearance change | `eval/tracking.reid_by_epoch` |
| Appearance epochs in the render | done — per-epoch palettes, actor swap at the GT frame, face-cover proxy, `--window` renders (docs/12) | `export/outfits.epoch_outfits`, `anim_plan.split_epochs` |
| Behaviour / intent | done — 50 behaviours (55 catalogue names), learned classifier, rule baseline; H40b added fence_climb, patrol_guard, maintenance_visit, wildlife and fenced restricted zones; H54 added fence_tailgate (follows a badged worker through a fence gate) and its twin staff_escort | `director/behaviours.py`, `eval/behaviour.py` |
| Aggression | done — argument, fight, theft, person-down, flee | `director/behaviours.py` |
| Camera tampering | done — 6 kinds; tilt invalidates calibration | `director/camera_director.py` |
| Degraded imagery | done — full imaging chain, 7 profiles, day profiles calibrated; `cctv_night_lowlight` / `cctv_thermal` fitted to LLVIP (H32) but **flagged overfit** (hold-out 2/13 vis, 3/13 IR; not met) | `sensors/degrade.py` |
| Experiments framework | done — `survsim experiment <name>`, pre-registered H1–H59 register (44 configs incl. in-flight h59, 29 registered runners; since H42 added evasion, ptz_slew (`tests/test_ptz_slew.py`), sensor_fusion (H53/H54/H59, `tests/test_sensor_fusion_h59.py`), dose_fixedstep `synthetic_mode` (H52); every config has a runner); results in `out/experiments/<name>/results.json`, log in `docs/research/32` | `survsim/experiments/` |
| Claim hygiene | done — `docs/research/RETIRED_CLAIMS.md` registry, enforced by `tests/test_docs_retired_claims.py`; `tests/test_docs_numbers_in_facts.py` ties decimals in 60, 65 v2, docs/13 to `FACTS_VERIFIED.md`; `AUDIT_LEDGER.md` | `docs/research/` |
| Real night reference | LLVIP subset at `~/data/real_cctv_night/{llvip_vis,llvip_ir}` — **non-commercial licence: measure only**, never train or ship. Night profiles overfit (H32); real night penalty unmeasured until H36 (download sign-off) | H32, H36 |
| Thermal / IR | done — emissivity, crossover, Johnson criteria | `sensors/thermal.py` |
| RF monitoring | done — MAC randomisation, path loss, association | `sensors/rf.py` |
| Audio | done — SPL, DOA error, named nuisance sources | `sensors/audio.py` |
| Multi-sensor fusion | done — with correlated-nuisance penalty | `eval/fusion.py` |
| NL media search | done — auditable parser, recall@k, MRR | `eval/retrieval.py` |
| Alarm architecture | done — threshold→context→persistence→corroboration→budget | `eval/pipeline.py` |
| Alarm cards + evidence log (H60) | slice 1 done — per-alarm card (86 §7), `mode=real` key whitelist + leak test (0 on smoke_mock), sha256-chained JSONL log with hashed operator id and verifier; `survsim alarm-cards`. Slice 2: per-site `--salt-file` (real mode refuses the public salt), `head.json` + fail-closed verify, RFC 8785-style canonical JSON, H22 `alarms_scored.jsonl` (score/θ/rank/verdict) carded via `--alarms`, config/model/clip hashes. Not yet: H33 per-alarm output, H22 alarms bound to a run dir with frames, signed head | `eval/alarm_card.py`, `eval/evidence_pack.py` |
| Scene generation | done — stratified, validated, repairing | `generator/` |
| Evasion / disguise | done — 52 techniques, confound audit | `director/evasion.py` |
| Crowd analytics | done — density bands, fundamental diagram, surge | `director/crowd.py` |
| Vehicles and drones | done — speed and altitude modelled separately | `director/behaviours.py` |
| VMS integration | done — video, timing sidecars, ONVIF metadata | `gt/video.py` |

## What is NOT done, and why

**Blocked on external artefacts — not on code:**

- **MetaHuman source textures.** An Epic-authenticated download. Everything that
  governs *shape and motion* works offline: the parametric body is [V] (height
  driven 171.75 → 190.0 cm across 30 named constraints) and rigging works from
  the DNA files the engine ships (`scripts/ue/rig_from_dna.py`). Textures are
  2K face albedo — cosmetic at the 20–50 m standoff where a person is tens of
  pixels tall.
- **The packaged UnrealZoo runtime is a dead end.** Its wardrobe commands are
  absent from the server and `animation/soma_glb/apply` rejects all 40 candidate
  clip names. Superseded by the editor route; not worth further effort.

**[V] Rendering works end to end (2026-09-18).** `survsim export <run>` writes an
Unreal manifest and `tools/render_run.py` rebuilds and renders it, one job per
camera, verified by pixels. The day-long "black render" was ours:
`unreal.Rotator(a, b, c)` is positionally **(roll, pitch, yaw)** in Python, so
the camera was aimed at the zenith and the sun placed below the horizon —
faithfully rendered night. See `docs/BUG_black_render.md`.

**Genuinely not built:**

- **Per-style gait parameters are argued, not measured.** Walking *speed* is
  measured from boxes (docs/14 §9), *cadence and stride* from pose (§10), and
  gait is now a six-parameter biometric that is scored for ReID and defeatable
  by a deliberate limp (§12, `survsim/gait/signature.py`). But the per-style
  multipliers in `STYLE_EFFECTS` — what a limp or a shuffle does to each
  parameter — are reasoned rather than measured, because no public dataset
  gives per-style gait parameters for surveillance footage. They are kept in
  one table so they can be disputed. Measuring them needs footage of the same
  people walking normally and then differently.
- **Editor render pipeline (2026-09-25, docs/12)** — persistent editor + job
  runner (7 s for 10 frames, ~11 frames/s at 720p); procedural PBR materials
  (asphalt, slabs, setts, brick, concrete, window-grid facades lit at night);
  six site-type environments from `survsim/export/env_plan.py`, and since
  2026-09-26 **scene generator v2** (`survsim/generator/layout.py`: per-type
  parametrised layouts, roads/kerbs/crossings/parking, gates at scenario
  entrances, boundary, decals, solar sun, installer rig, entrance gate; 18/18
  variety frames gated, docs/12 "Scene generator v2"); since 2026-09-28
  **14 site types** (8 added in `survsim/generator/layout_sites_v2.py`: loading
  dock, school gate, rail platform, hospital entrance, utility yard,
  construction site, farm yard, residential close; 3 layout variants each,
  per-type queue zone for the hard negative, `batch.SITE_PROFILES`;
  `batch.SITE_TYPES` stays the original 6 so experiment worlds are unchanged,
  `tests/test_site_types_v2.py`); day/night
  presets with fixed, metered exposure; PTZ pose and per-camera resolution
  follow the GT; people hidden outside their key range; play rate calibrated
  by FK and read back. Render gate (`scripts/check_render.py`) run on every set.
  **2026-09-26: every rendered person is a MetaHuman [V]** (baked offline
  bodies MHB01-03, painted street-wear clothing per identity, RigLogic
  post-process, walk/run/idle loops chosen from speed and the behaviour label,
  loiter/recon turn to look around). batch_v4 (3 scenarios x day/night, 5 400
  frames): gate 6/6, yolov8n recall 0.758 vs 0.577 for CesiumMan on the same
  770 GT boxes (docs/12 "MetaHumans in production").
  **2026-09-26 (later): the CC0 stack is the default [V]** - 16 verified
  rebaked Quaternius bodies on their own walk/idle/run clips (FK-measured
  ground speed and facing), clothing painted by skin-weight regions baked as
  vertex colour, `asset_licences` per person and `commercial_ml_ok` per set.
  MetaHumans are opt-in (`SURVSIM_CHARS=metahuman`). batch_v7 (same 8 sets as
  v6): gate 7/8 (same far-night failure), every set commercial_ml_ok, yolov8n
  recall 0.652 vs 0.665 for MetaHuman v6 - the cost is in 20-60 px people
  (0.362 vs 0.403); foot-slide median 8.5-9.9 vs 5.9-8.1 cm/s (clip floor
  ~9 cm/s), p90 better (docs/12 "CC0 characters are the production default").
  **Still open:** ambientCG textures / HDRI sky [X] — the texture master does
  not compile (0 instructions, guarded; opt-in `--textures`), and the HDRI sky
  moved realism stats away from the real band; procedural stays default.
  Only three baked bodies (two ~1.58 m, one ~1.93 m, scaled per person).
- **Detection/MOT/ReID baselines on our own renders** — the harness is built
  (`survsim dataset` / `survsim baseline`, docs/13_BASELINES.md: YOLO11n COCO
  mAP, numpy ByteTrack -> MOTA/IDF1/HOTA, OSNet ReID rank-1/mAP) and has a real
  MOT20 reference row. Only the *numbers on renders* wait: mock datasets are
  flagged `synthetic_topdown: true` and their numbers measure plumbing only.
- **Real night detection penalty** — unmeasured: H32's night profiles are
  overfit (hold-out 2/13 vis, 3/13 IR), so the Δ it reports is not a real penalty.
- **H36 full LLVIP held-out fit** — waits on sign-off for the ~3.5 GB download.
- **H41 keypoint auto-calibration** — needs a head/foot keypoint GT export first.
- **A live RTSP server.** `gt/video.rtsp_command` returns the command; running it
  is left to the caller because starting a server is a deployment decision.

**Approximations worth knowing about:**

- Occlusion uses a body silhouette (head ellipse over a tapered torso, ~75% of the
  bounding box) rather than a rectangle, but has no arms or pose. Pixel-accurate
  occlusion needs the Unreal backend's rendered mask.
- The mock backend has no scene lighting, so degradation profiles are applied to
  flat renders there; only the Unreal backend gives real illumination.

## Bugs found by building and measuring

Each was silent, and each would have corrupted downstream results.

1. **Inverted pitch sign** — `pitch: -20` tilted cameras *upward*; three of four
   cameras in a realistic rig saw nothing. A self-consistent principal-ray test
   could not catch it; the regression test now pins the sign to ground geometry.
2. **Behaviour labels never overrode** — `label` was a dataclass field, so every
   behaviour reported `"normal"`.
3. **Two conflicting definitions of "visible"** — a disguise staged in a blind
   spot was logged as observed. Unified on `geometry.usably_visible`.
4. **Blind spots that were not blind** — the rig analyser caught what inspection
   had missed; placement now treats them as a hard constraint.
5. **Distortion applied to pixels but not to projection** — ground truth drifted
   from people toward the frame edge. Now matches OpenCV to 1e-6.
6. **Capability inferred from a command list** — twice. Wardrobe and gait were
   both reported working because the command existed. Capability checks are now
   functional, and an unverified build is assumed broken.
7. **Realism targets that were wrong** — reasoned-out ranges pushed the
   degradation pipeline *away* from real footage (1.304 vs 1.072 for an
   undegraded render). Now calibrated from real frames.
8. **A distance metric that optimised the wrong thing** — normalising by the mean
   let near-zero statistics dominate, so the first parameter search tuned
   `clipped_fraction` and ignored sharpness. Now normalised by real spread.
9. **Degenerate evaluation reported as chance** — a test split with no positives
   returned AUC 0.5, indistinguishable from an honest result. Now flagged.
10. **NL search returning everything** — an unparsed query matched the whole
    index, reading as a confident answer to a question never understood.
11. **Thermal range without a resolution limit** — reported a cheap core
    detecting people at 800 m where they subtend under one pixel.
12. **Gait was a label, not a signal** — `Gait.style` nudged walking speed and
    nothing else, so `soft_biometrics` (the function answering "what survives a
    disguise") omitted the cue surveillance actually uses, the gait evasion
    techniques degraded nothing measurable, and two "normal" walkers were
    numerically identical — making any gait ReID score either perfect or
    chance. Now six measured parameters (docs/14 §12).
13. **Crowd density counted only visible people** — MOT20 read as 0.149/m²
    (free-flowing) while the same run measured shuffling speeds. A crowd is
    crowded *because* its members occlude each other. Caught only because two
    related quantities disagreed; a single wrong number would have survived.
14. **Camera-avoidance deadlocked** — a purely radial repulsion field cancels
    goal attraction exactly when a camera sits between the person and their
    destination, so the evader stalled in the open and never arrived.
15. **Every camera in the rig rendered the same view** — Movie Render Queue
    ignores a job's camera unless the sequence carries a
    `MovieSceneCameraCutTrack`. Near-invisible: the PNGs differed by
    anti-aliasing noise so checksums passed, while every frame's mean and
    standard deviation matched to two decimals. An eight-camera rig would have
    become one viewpoint copied eight times, silently invalidating every
    multi-camera tracking, handoff and overlap number.
16. **A regression guard that fired on prose** — the positional-`Rotator` check
    matched the docstring documenting the bug. A guard that cries wolf trains
    people to ignore it; it now walks the AST.
17. **A health check that tested the wrong thread** — UnrealCV answers its
    handshake from a network thread while commands run on the game thread, so a
    build stuck at frame 1 reported as healthy. Hours lost. docs/12 §2.
18. **`World.zone_of` resolved overlapping zones by config order, not size** —
    a worker standing inside the small `zone.substation` (nested in the large
    `zone.yard`, `allowed_roles: [worker, guard]`) was recorded as being in
    `zone.yard`, because `zone_of` returned the first polygon in declaration
    order that contained the point rather than the smallest one. Every
    consumer of `world_df["zone"]` (GT writers, `ZoneIntrusion`/
    `wrong_zone_entry`'s "have I arrived" check, `PathLoiterRule`'s role
    exemption) inherited the wrong zone. It also silently broke
    `wrong_zone_entry` targeting a nested zone: the behaviour never registered
    "inside" and so never completed. Fixed by resolving to the
    smallest-area containing zone, tie-broken on zone id for determinism
    (`survsim/director/world.py`); `PathLoiterRule` dropped its own private
    smallest-zone recomputation and now trusts `world_df["zone"]` directly.
    Regression test: `tests/test_eval.py::test_zone_of_resolves_nested_zones_to_the_smallest_one`.
19. **`loiter_arc` walked to the exact zone centroid** — `Loiter.begin`
    resolved its dwell target with `world.resolve_point`, the exact centroid,
    while every other destination in the simulator (`Transit`'s, via
    `world.sample_goal`) gets seeded jitter. `CommuteWait` and `SmokeBreak`
    both subclass `Loiter` without overriding `begin`, so the hard negative
    walked to the same exact point too — but "distance to zone centroid" was
    still a real simulation artefact docs/research/58's forbidden-feature
    list had to exclude by hand rather than by construction, and any future
    code computing that distance from the *true* centroid instead of from
    where the person actually stood would have quietly reintroduced a leak.
    Fixed: `Loiter.begin` now calls `world.sample_goal`, with a bounded retry
    (checked via `world.zone_of`) so a corner sample plus the pacing radius
    can't walk the person out of a small zone. Regression tests:
    `tests/test_eval.py::test_loiter_and_commute_wait_destination_jitter_is_indistinguishable`
    (KS test, p > 0.05 over 200 samples each) and
    `::test_loiter_destination_sampling_is_deterministic`.
20. **`alarms.score` matched actor-blind** — it only checked label + timing
    window, so two same-label events starting within `tolerance_s` of each
    other (e.g. a `disguise_swap` scan loiter and a plain loiter close
    together) competed for one match; the second person's alarm scored as a
    repeat. This capped even a label oracle's recall at 0.90 on the H58 recall
    set (docs/research/58 §4) — a scorer property, not a detector failing.
    Added `score(..., actor_aware: bool = False)`: when True and an alarm
    carries a `uid`, it only matches an event whose `actors` contains that
    `uid`; an alarm with no `uid` still falls back to window-only matching, so
    nothing regresses for rules that never emitted actor identity. Default
    stays `False` and is asserted bit-identical
    (`test_actor_aware_default_false_is_bit_identical_to_before`). Rerunning
    H12/H58 with `actor_aware=True` is pending — see docs/13 "actor_aware
    scorer option" for what would need to change and why the cached numbers
    above are not simply recomputable.
21. **`gt.coco.from_camera_rows` exported wildlife as COCO "person"** — it had
    no class filter, unlike `survsim dataset` (`IGNORE_CLASSES` in
    `survsim/datasets/build.py`); backend `observe()` rows now carry
    `object_class`, and non-person rows are skipped by default
    (`include_non_person=True` opts back in). Regression test:
    `tests/test_coco_and_detection.py::test_from_camera_rows_skips_wildlife_rows_by_default`.
22. **TrackEval parity harness crashed with `KeyError: 'deta_mean'`** — `2fdecd6` (H24)
    added bridge-only `deta_mean`/`assa_mean` to `run_trackeval`; `tools/trackeval_parity.py::score()`
    looked every COMBINED key up on `TrackingResult`. Now compares only `TrackingResult` fields
    (no metric change; parity re-run max |Δ| 9.99e-16). Regression test:
    `tests/test_trackeval_parity_keys.py`.
23. **`dose_fixedstep` silently ignored `--arm`** — it was missing from `ARM_RUNNERS`, so
    a single-arm re-run quietly ran every arm. Fixed in `385f363` (H51); H26 arms reproduced 15/15.
24. **Security hygiene (87 gaps)** — `!include` followed `../` and absolute paths anywhere on disk; now
    confined to the `configs/` root (`IncludeError`, `tests/test_loader_include.py`). Pre-commit gained a
    staged-file secret scan (`scripts/hooks/secret_scan.py`, allowlist `secret_allowlist.txt`,
    `tests/test_secret_scan.py`). Exact versions frozen in `requirements-lock.txt`. Fixed in `a633a1d`.

25. **Two config files stale since they were last edited by hand** —
    `configs/scenarios/border_fence_night_probe.yaml` referenced `spawn.east`,
    `zone.center`, `zone.gate_A`, `exit.west` that `border_fence.yaml` never
    defined, and `configs/behaviours/templates.yaml` had an invalid flow-mapping
    (`to: random_zone_tag: queue|gate`) that made it unparsable YAML. Fixed by
    adding the missing zone/spawn/exit definitions and nesting the flow-mapping
    value; `tests/test_configs_valid.py` now parametrises every `configs/*.yaml`
    parse and every scenario validate so a future break names the file.
26. **Alarm-card guard gaps (AUDIT_87 §5.1–5.2)** — `leak_count` matched GT values only
    as whole JSON strings and `gt_values` omitted `identities.parquet`, so a uid inside a
    longer string (`calib/loiterer_1_cam.json`) or a role like `intruder` passed; and
    `verify` read the log alone, so an edited `cards.jsonl` still verified. Fixed in
    e3f7e13: substring match over every string field (allowlist + min length 4), all string
    columns of `identities.parquet`, and `evidence_pack.verify_cards` (CLI `chain_ok`
    requires it), which names the offending `alarm_id`. Tests in `tests/test_alarm_card.py`.
27. **`evasion.bootstrap_delta` resampled frames, not clips** — H47's Δ-recall CI treated
    every frame of a 30 s, 10 fps clip as an independent draw, when frames within a clip are
    strongly autocorrelated; AUDIT_H47 §2 caught it by hand-recomputing a clip-cluster CI
    (crowd_cover 26n [−0.254, −0.154] vs the frame-level [−0.210, −0.192] — ~2–5× narrower
    than the true one) and AUDIT_91 flagged the glossary describing it as "clip-cluster"
    when the code was frame-level. Fixed at source: `bootstrap_delta(..., cluster=...)`
    now resamples whole (seed, camera) clips by default (`cluster="clip"`), with
    `cluster="frame"` kept only for explicit opt-in comparison; `evasion.cmd_score` writes
    the clip-cluster CI as `results.json`'s primary `delta` and the frame-level CI under a
    separate `delta_frame_level` key. Regression test:
    `tests/test_evasion_h47.py::test_bootstrap_clip_vs_frame_cluster` (frames forced
    perfectly correlated within each clip; asserts the clip-cluster CI is materially wider
    than the frame-level one). A re-score from `det_cache/` (no rendering, < 2 min) matches
    AUDIT_H47's hand computation exactly: crowd_cover 26n −0.201 [−0.254, −0.154].

**Disk incident (2026-09-27).** The root disk filled from another job's renders: H51's sweep died
with `OSError: No space left on device` (one arm JSON written empty, deleted, re-trained) and H47's
hood_up seed 3 render was truncated (245/300 frames; slow/crowd/wall seed 3 never rendered).
Check `df` before any render or training sweep; delete scored PNG sets (32 §H47, §H51).

**`/tmp` quota incident (2026-09-27).** A stale 6.8 GB batch copy left in `/tmp` (which is a
16 GB tmpfs, not disk) tripped the quota at 11:13, alongside agent transcripts/scratch that
also live there. Fix was deleting the stale copy; rule going forward is never stage render
batches under `/tmp` — see docs/10 "Operating limits learned".

**Editor OOM incident (2026-09-27).** The persistent Unreal editor was OOM-killed at 14:40 when
11 CPU workers + a QAT run + the operator-sim editor ran concurrently on the 31 GB, no-swap box —
CPU-side RAM pressure, not VRAM. Restart only via `scripts/ue_mcp.sh` once `free -g` shows
≥ 8 GB free (docs/12 "Editor lifecycle").

## Verification added

- **ID mask respects fences/walls (2026-09-28) [V].** `ID_MASK_DEPTH_TEST`
  (scripts/ue/survsim_scene.py, default on) zeroes stencil IDs where the
  person's CustomDepth lies behind scene depth. Site 09 A/B (out/verify/idmask_*):
  people behind the palisade 84→40, 62→30, 76→34, 63→23 ID pixels, ≥ 97 % of
  removed pixels on the projected pales. `build_tight_gt` now reads
  `env_plan.json` and writes `static_cover` / `occluder=static` and the
  static-corrected vis; `filter_idmap` is applied only to masks rendered without
  the depth test (it over-removes ~40 % vs native). docs/BUG_gt_visibility_fences.md;
  test `tests/test_idmask_depth_palisade.py`.
- **MOT visibility (column 9) comes from the render, not the mock (2026-09-28) [V].**
  `build_amodal` now rewrites `gt/mot_<cam>.txt` column 9 with mask pixels /
  expected pixels (`write_render_visibility`). Before, it carried the mock
  painter's `visible_frac` (fences = opaque L x L boxes): showcase_all_night and
  _dusk had 625 conf-1 rows at 0.000 with rendered pixels. Existing sets are
  not rewritten in place; v2 at out/gt_static_occ/showcase_all_night_v2.
  docs/BUG_night_visibility_zero.md; test `tests/test_night_visibility_zero.py`.
- **Static-occlusion GT follows the live PTZ pose (2026-09-28) [V].** Fence depth
  maps are built per frame from `gt/camera_status.jsonl` (live yaw/pitch/hfov),
  not the configured calib; PTZ frames with no recorded pose are flagged
  `pose_unknown`. Showcase cam_ptz: 175 / 502 rows change static_cover.
  `tests/test_gt_static_occlusion_ptz.py`; AUDIT_GT_FENCE_VISIBILITY "PTZ".
- **GT boxes on rendered sets come from the rendered pixels (2026-09-26) [V].**
  Before, a rendered set's `gt/` was a symlink to the mock run: boxes were the
  projected hull of an upright 3-D box at the Director's height (the rendered
  body is 1.70 m for everyone — `unreal_scene.wardrobe_from_appearance`
  default), so they were loose at the top (+7 % of height, docs/research/32
  §H6(c)). `render_batch --id-mask` now renders a per-person CustomDepth-stencil
  ID pass and `survsim/gt/mask_boxes.py` writes tight, occlusion-truncated
  boxes (projected file kept in `gt/projected/`, both in
  `gt/boxes_mask.parquet`; zero-pixel people become MOT ignore rows).
  Gate: `python -m survsim.checks.box_tightness <set>` (median IoU vs zero-shot
  YOLOv8n >= 0.75, |top offset| <= 3 %). smoke_mock_day 0.781 / +4.8 % → 0.884 /
  −0.6 %; street_crowd_day 0.712 (FAIL) → 0.911. Sets: `out/batch_v7/*_tightgt`.
  **Tight boxes did not raise real AP**: synthetic-only fine-tune on the two
  sets (370 imgs, 2 seeds, MOT20-02 @640) — loose 0.290 ± 0.024, tight
  0.266 ± 0.004 (`out/experiments/h1_{loose,tight}gt/`). Label fit is fixed;
  it was not the transfer bottleneck on this data (MOT20's hand boxes are
  amodal and wider: median w/h +14 % [+6, +21] vs batch_v8 mask boxes pooled
  over >= 50 px, but +22 % at 50-100 px and n.s. at 100-200 px — docs/13 "Box
  aspect"; so "tight" is not "MOT20-like").
  **Amodal boxes [V]** (`survsim/gt/amodal.py`, docs/12 "Amodal boxes"):
  third box type per person-frame (full extent behind occluders) + a
  visibility fraction, from a calibrated projected body envelope with FK feet;
  holdout IoU vs mask on clean frames 0.921 / 0.907 (smoke / street), mask ⊆
  amodal median 0.99 / 0.98. `gt/amodal/mot_<cam>.txt`; H15 is a path swap.
  Height gate wired: both tightgt sets FAIL (rendered at 1.70 m, pre-92e1e97).
  **batch_v8 [V] (2026-09-27): every label true to the pixels.** 9 sets
  (4 scenarios x day/night + disguise 95-135 s epochs), all `--id-mask`
  (visible + amodal boxes + visibility), GT heights (unclamped, 1.22-1.89 m;
  `tools/render_run.gt_heights`), CC0, commercial_ml_ok. Gate **9/9** incl.
  `gt_height_matches_render` (0 beyond 2 cm) and box tightness (IoU
  0.876-0.932, gated on day); motion audit 0 % stuck. yolov8n recall 0.696
  (visible GT) / 0.676 (amodal GT) vs v7 0.664. `out/batch_v8`; H15 layout in
  docs/12 "batch_v8". Mask-box decode 70 min -> 33 s per 4-cam set (1fd38df).

- **Tracking metrics are validated against their definitions**, not against
  themselves: dropping 10% of boxes must cost exactly 0.10 MOTA; an identity
  swap must leave MOTA > 0.99 while IDF1 falls and the damage lands in AssA not
  DetA; HOTA must lie between its components. 30 tests, 8 of them against real
  MOT20 footage — where the perfect-tracking case would expose a greedy matcher
  inventing identity switches among overlapping bodies.
- **Coding-drill reference implementations are parity-tested against survsim's
  own code on toy inputs** (`docs/research/drills/drills.py`,
  `tests/test_coding_drills.py`).
- **FID reports its noise floor.** A bare FID is uninterpretable: "38" is
  neither good nor bad until you know what two samples of the *same*
  distribution score. `fid_report` returns the floor and the ratio, and refuses
  to quietly return an under-sampled number — Inception features are 2048-d, so
  a covariance from 12 images is rank-deficient and `sqrtm` returns confident
  noise. That is what we nearly published.

## Measured results

See `docs/14_FIRST_RESULTS.md`. The headline findings:

- A single scene cannot support behaviour learning (3 suspicious identities gives
  AUC 0.46); ten scenes gives 0.72, and the 4-scene demo reaches 0.85.
- Precision stays near 0.07 regardless, because suspicious segments are 3.3% of
  the data. This is a base rate, not a model deficiency.
- The alarm architecture takes the same model from 21.67 to 0.00 false alarms per
  camera-hour, precision 0.037 → 1.000, with recall preserved in the demo run.
  **111× from architecture versus 4× from replacing a rule with a learned model**
  is the ratio that should drive where effort goes.

## Running it

```bash
survsim demo                      # whole chain end to end, reports every stage
survsim generate --count 20       # 20 diverse validated scenes
survsim rig <scenario>            # coverage, overlap, blind spots
survsim eval <run>                # alarms per camera-hour vs operator budget
survsim evasion --pairs 30        # disguise personas + confound audit
```

## Motion audit

`survsim audit-motion <run_dir>` (`survsim/eval/motion_audit.py`) reads
`gt/world.parquet`, measures per-second speed from *positions* (not recorded
velocity), and reports stuck episodes (>3 s stationary under a walking label,
or with no behaviour left), their probable cause, walking-speed distribution
(moving `normal.transit` seconds) and t=0 clustering. Writes
`<run_dir>/motion_audit.json`. Regression: `tests/test_motion_audit.py`
(street_crowd 60 s, stuck fraction < 2 %).

**[V] Root causes found (2026-09-25)** in `out/batch_v1`:

1. **Goal congestion** — every walker sent to the *same* coordinate (zone
   centroid, exact exit point) with a 0.5 m arrival radius. Reciprocal
   repulsion lets only one person ever arrive; the rest stand pinned at the
   goal for the rest of the run, still labelled `normal.transit`.
2. **Plan exhausted** — after the final transit to an exit, the agent stayed in
   the world, standing on the exit, labelled `normal.idle`.
3. **Everyone at t=0 on a spawn point** — all generated people were spawned at
   t=0 on one of two spawn coordinates.

No avoidance deadlock between walkers in open space was observed, but the
symmetric repulsion can lock a head-on pair, so it was fixed too.

Fixes: per-person destinations (`World.sample_goal`: random interior point of
a zone, 1.5 m spread at spawns/exits); transit stall detection (no progress for
1.5 s → accept arrival within 2.5 m, else re-plan to a fresh goal, max 4);
keep-right sidestep yield in `planner.steer`; agents whose plan ends at an exit
depart (`world.agents` = present now, `world.roster` = everyone); generated
people arrive at uniform times over `[-population.warmup_s (30 s), 0.75·duration)`
— per-entrance Poisson conditioned on count — and the world is simulated through
the warm-up before frame 0. Heroes stay on their scripted timeline at t=0;
generated intruders arrive in the first quarter so positives are on camera.
Warm-up events/epochs/sensor rows are dropped or clamped to t=0.

| run | stuck fraction | stationary while walking | stuck s by cause | t=0 within 1 m of another | transit speed p10 / median / p90 (m/s) |
|---|---|---|---|---|---|
| street_crowd before (batch_v1 day) | 34.6 % | 24.2 % | congestion 224, plan-exhausted 150 | 100 % (median NN 0.006 m) | 0.96 / 1.08 / 1.30 |
| street_crowd after | **0.0 %** | 0.5 % | — | 0 % (median NN 4.3 m) | 0.95 / 1.10 / 1.30 |
| disguise_and_tamper before (batch_v1 day) | 49.1 % | 35.6 % | congestion 895, plan-exhausted 400 | 100 % | — |
| disguise_and_tamper after | **0.0 %** | 0.6 % | — | 3 scripted heroes only | 0.85 / 1.19 / 1.49 |

Caveat for the render pipeline: people now enter and leave mid-run, so a
track's first/last frame is no longer 0/N. The renderer must spawn/hide actors
at those bounds (tools/ and scripts/ue/ not changed here).

## Licensing

**MetaHuman renders may not be used to train or test ML** (Epic MetaHuman/UE
terms: may not use MetaHuman characters or animation curves "to build or
enhance any database or train or test artificial intelligence, machine
learning..."). batch_v4–v6 and anything animated with `AS_MH_*` clips are
internal engineering evidence only — never a training, validation or
customer-facing benchmark set. SMPL-X/BEDLAM academic licences ban
surveillance use; Reallusion/Mixamo standard terms ban ML. **Compliant
default going forward: rebaked Quaternius CC0 characters + CC0 Kenney,
ambientCG, Poly Haven (CesiumMan is CC-BY: attribute).** New asset ⇒ record
its licence in `~/data/assets/index.json` and check its ML clause first.
**Implemented 2026-09-26:** CC0 is the render default; every set carries
`licences.json` (`commercial_ml_ok`) and per-person `asset_licences`
(`survsim/gt/licences.py`); batch_v7 is the first commercially usable set.
Not done from the docs/47 list: `--licence-restricted` gating of the `mh_*`
scripts, and eval refusing "product" numbers on a non-commercial set.
Full analysis: `docs/research/47_CHARACTER_AND_ASSET_LICENSING.md`.

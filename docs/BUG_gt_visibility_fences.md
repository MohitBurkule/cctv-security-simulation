# BUG: GT visibility ignores fences; edge slivers count as usable (2026-09-28)

Found by AUDIT_H69 (R68): sites 09 (dusk) and 17 (night) have people 21–29 px
tall standing behind palisade railings with GT visibility 0.8–1.0; site 05's
only targets are 3.5–6 px-wide slivers at the right image edge with vis > 0.

## Evidence

* **[V] These sets have no ID mask.** `out/showcase/sites/*/set/` holds
  `render/` and `render_meter/` only — no `mask/`, no `gt/boxes_mask.parquet`.
  The H69 GT is `run/gt/mot_cam_00.txt`, the Director/mock projection, whose
  column 9 is the mock painter's `visible_frac`.
* **[V] The mock painter never sees the fences.** `MockBackend._scenery_boxes`
  paints only `scenario.environment.props`. The fences are site dressing in
  `set/env_plan.json` (`fence_run`, style `palisade`: 11 pieces on site 09),
  built by `scripts/ue/survsim_scene.py` and never passed back to the Director.
* **[V] The fences are in front of the people.** Frame 20 crops (brightened
  3x, GT boxes drawn): on 09 both people are seen through 1 px pales at
  ~2.5 px pitch; on 17 all three are behind two fence layers. The projected
  fence geometry (below) overlays the rendered pales on 05, 09 and 17 with no
  visible offset.
* **[V] Numbers after projecting the fences** (per-person silhouette fraction
  behind a nearer occluder, 4x supersampled): site 09 median visibility
  0.839 -> 0.332, 67/80 rows now `occluder=static`; site 17 0.288 -> 0.167
  (45/120 static; its person 3 goes 1.0 -> ~0.53); site 05 1.0 -> 0.52, and
  9/36 rows are edge slivers.

## Root cause

1. **[V] Director-side visibility (all mask-less sets, incl. every H69 site):**
   occluders known to the render (env_plan dressing) are unknown to the GT.
   Principle 1 violated: the label describes the plan's scenery, not what was
   rendered.
2. **[V] Mask-side visibility (sets rendered with `--id-mask`):** the ID pass
   is `PPI_CUSTOM_STENCIL`. Only people set `render_custom_depth`, so the
   CustomDepth pass contains people only; a fence, wall or hedge writes no
   custom depth and cannot remove a person's stencil. Person-on-person
   occlusion is correct because both people write custom depth (that is what
   docs/12 "CustomDepth is depth-tested" observed). Not verified on a rendered
   set: no mask set with a static occluder in front of a person was inspected
   (the editor is busy; no re-render done). Verified 2026-09-28, below.
3. **[V] Slivers:** nothing flagged a box cut by the frame edge to a few pixels
   wide; `usably_visible` is 3-D and height-only.

## Fix

* **[V] `survsim/gt/static_occlusion.py`** rebuilds `fence_run` (palisade:
  7 cm pales every 20 cm + rail at 0.8 h; other styles as a panel),
  `wall_run` and `hedge` from `env_plan.json` exactly as the scene builder
  does, rasterises them through `run/calib` into a nearest-depth map
  (SS = 4), and for each GT row measures `static_cover` = fraction of the
  mock body silhouette behind an occluder nearer than the person
  (`world.parquet` position, mid-body depth). `vis = vis_old x (1 -
  static_cover)`; `occluder = static` when the static deficit is >= 0.10 and
  exceeds the person/edge deficit already in `vis_old`. CLI:
  `python -m survsim.gt.static_occlusion <site> --out <dir>` writes a corrected
  MOT (unusable rows conf 0) + `static_occlusion.parquet`; inputs untouched.
* **[V] One usability definition:** `geometry.box_usable(w, h)` =
  `h >= MIN_USABLE_PX_HEIGHT` (the same 20 px) and `w >= min(8,
  0.25 h)` on the in-frame box. Deviation from a flat 8 px width: an upright
  person exactly 20–24 px tall is ~6–7 px wide (all of site 09), and a flat
  8 px floor would have made the width rule stricter than the height rule.
* **[V] Post-hoc mask filter** `filter_idmap(idmap, zbuf, depth_by_id)` zeros a
  person's ID pixels where a static occluder is nearer (>= half of the
  subsamples covered). Run on the site 09 no-depth-test mask: 84->24, 62->14,
  76->18, 63->16 px vs the native depth test's 40/30/34/23 — it over-removes
  (projected pales cover more than the rendered ones), so it is applied only to
  masks rendered without the depth test. Wired into `build_tight_gt`
  (`depth_tested=` from the render result's `id_mask_depth_test`; results
  without the key are treated as untested).
* **[V] `build_tight_gt` static semantics:** with `set/env_plan.json` present,
  each mask row gets `static_cover`, `occluder` (same rule as
  `correct_rows`) in `boxes_mask.parquet`, and MOT column 9 = vis x (1 -
  static_cover).
* **[V] Export-side fix (2026-09-28):**
  `scripts/ue/survsim_scene.py` `ID_MASK_DEPTH_TEST = True` — the ID material
  also samples `PPI_CUSTOM_DEPTH` and `PPI_SCENE_DEPTH` and writes 0 where the
  person's custom depth lies behind the scene depth (tolerance 0.5 % + 2 cm).
  Default on.

## Verification (2026-09-28)

Site 09 (transit_station s1333, overcast/dusk, cam_00, 640x360, CC0) re-rendered
with `--id-mask`, 10 frames with the depth test (`out/verify/idmask_depth/`) and
3 frames without (`out/verify/idmask_nodepth/`, `render_run.py
--no-id-mask-depth-test`). Only two people (stencil 2, 3) are in view, both
behind the palisade (the brief said three; the camera sees two).

| frame | id | no depth test px | depth test px | removed | removed on projected pales |
|---|---|---|---|---|---|
| 1 | 2 | 84 | 40 | 44 | 44 |
| 1 | 3 | 62 | 30 | 32 | 32 |
| 2 | 2 | 76 | 34 | 42 | 42 |
| 2 | 3 | 63 | 23 | 41 | 40 |

Crops (8x, brightened): without the test the ID pass is two whole silhouettes
through the railings; with it only vertical strips between the pales remain,
aligned with the gaps in the beauty frame. Boxes barely change (height and
width set by the unoccluded strips), pixel counts halve. Render gate: 11/13
pass; `gt_frame_coverage` fails by construction (10 of 40 GT frames rendered)
and `gt_person_rendered` 0.417 < 0.6 (dim people ~7 px wide behind pales at
dusk). `render_run.py`'s amodal step crashed on this out-of-tree set (no
`unreal/manifest.json` beside it), after mask GT was written.
Regression: `tests/test_idmask_depth_palisade.py` (3.3 KB crop fixture).

## Limits

* Static cover and person occlusion are combined as independent; the mock
  silhouette has no arms/pose.
* **PTZ (fixed 2026-09-28) [V].** The depth map used to come from
  `calib/<cam>.json` (the configured pose) for every frame, so a slewed PTZ was
  scored against fences it was not looking at (principle 2). `live_static` now
  builds K/R/t per frame from `gt/camera_status.jsonl` (live yaw/pitch/hfov;
  position and roll from calib) for every camera, also used by
  `build_tight_gt`; a fixed camera with no status row falls back to calib; a PTZ
  frame with no recorded pose gets `occluder="pose_unknown"` (vis unchanged,
  `static_cover` null; `strict=True` raises `PoseUnknown`). Showcase dusk/night
  cam_ptz: static_cover changes on 175 / 502 rows (details in
  docs/research/AUDIT_GT_FENCE_VISIBILITY.md "PTZ").
  Regression: `tests/test_gt_static_occlusion_ptz.py`.
* Only fence/wall/hedge kinds are modelled; cars, shelters and blocks still
  are not.

Regression tests: `tests/test_gt_static_occlusion.py` (bar pattern -> vis < 0.5,
5 px sliver -> unusable, palisade projection, mask filter, material flag).
H69 re-score: docs/research/32 "H69 · post-fix GT" (unregistered).

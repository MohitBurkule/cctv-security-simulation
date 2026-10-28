# BUG: showcase night/dusk MOT visibility 0.000 on visible people

Reported by the demo director (DIRECTOR_QUEUE row 44, 2026-09-28 07:10):
`out/gt_static_occ/showcase_all_night` (fence_audit.py) shows `vis_old 0.00`
for people who are plainly unoccluded on cam_01.

## Reproduction [V]
- cam_01: 459 of 1,358 MOT rows have column 9 = 0.000; **456** of them have
  rendered ID-mask pixels (`boxes_mask.parquet` mask_pixels > 0, median
  render visibility 1.0). cam_02: 169 such rows. 625 conf-1 zero rows in total.
- Crops of the render (brightened x2.5), largest box per track:
  frame 97 id 3 (563 px), frame 98 id 5, frame 53 id 2, frame 96 id 13 — person
  visible and in front of the fence (id 13 partly behind another person);
  frame 43 id 1 is behind the palisade but visible between pales (render vis 0.999).

## Root cause [V]
`gt/mot_<cam>.txt` column 9 was never render-derived. `mask_boxes._camera_boxes`
replaces the box with the mask box but copies column 9 (`vis = float(r[8])`)
from the mock backend's `visible_frac`. The mock painter
(`backends/mock/backend.py::_scenery_boxes`) paints every prop as an opaque
image-space rectangle, and a fence of length L as an L x L square footprint, so
anyone near or in front of a long fence reads fully occluded. Violates
principle 1 (truth from rendered masks).

Ruled out: [V] not low light (dusk MOT is byte-identical: same 459 zeros);
[V] not a missing mask set (mask pixels present on 456/459);
[V] not PTZ live-pose keying (cam_01/cam_02 are fixed; fence_audit skips cam_ptz);
[V] not the audit script column (it reads column 9 correctly — the column is wrong).

## Fix [V]
`survsim/gt/amodal.py::write_render_visibility` (called at the end of
`build_amodal`) rewrites column 9 for conf-1 rows with the render visibility
(mask pixels / expected pixels, which already includes fence cover through the
depth-tested or filtered mask). Rows with no mask pixels stay 0.000 / conf 0.
Regression test: `tests/test_night_visibility_zero.py`.

## Re-derivation (out/gt_static_occ/showcase_all_night_v2) [V]
2,176 rows (cam_01 + cam_02): 1,568 change column 9, 1,220 by > 0.1,
625 go 0 -> positive; vis_old >= 0.5 rows 1,412 -> 2,034. After the static
fence correction, rows crossing below 0.5 go 4 -> 410 and occluder=static
10 -> 470 — the old mock value had already (wrongly, as a solid block) absorbed
the fence. [A] the night mask predates ID_MASK_DEPTH_TEST, so vis x (1 - cover)
is the right combination; for depth-tested sets it would double count.
Dusk: identical GT, same counts. Existing set directories are not rewritten.

## Numbers that move
- H32c: "362 GT (h >= 20 px, vis > 0)" becomes **449** with render visibility —
  87 visible people were dropped. Recall not re-run [A]; qualifier in 32 §H32c
  and FACTS.
- H69 did not use night-set visibility (per brief) — not checked further.

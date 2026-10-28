# CLAUDE.md — SurvSim project guide

Photorealistic multi-camera, multi-person **surveillance simulator** for
developing and evaluating CCTV analytics (detection, tracking, multi-camera
re-identification, behaviour/intent, degraded imagery, multi-sensor fusion).
Built to mirror what a production CCTV ML suite does, so algorithms can be developed
against ground truth that real CCTV can never provide.

> This repo is the simulator. Analytics, evaluation and experiments live in the companion repo `cctv-security`; research docs referenced below live there.

## Read these first
| Doc | Why |
|---|---|
| `docs/12_UNREALCV_ENGINE_NOTES.md` | **Engine operational knowledge. Read before touching Unreal — it will save hours.** |
| `docs/11_IMPLEMENTATION_STATUS.md` | What is built vs. still planned |
| `docs/10_ENVIRONMENT_SETUP.md` | This machine, install state, hardware limits |
| `docs/02_ARCHITECTURE.md`, `docs/03_BEHAVIOUR_CATALOG.md` | Design intent |

## Environment
```bash
source .venv/bin/activate          # Python 3.11 (system python is 3.14 — too new for CV wheels)
pytest -q -n auto                  # ~1050 fast tests (1053 selected / 1079 collected), ~1.5 min; `pytest -m slow` for the ~20 s-to-8 min ones
survsim validate configs/scenarios/smoke_mock.yaml
survsim run   configs/scenarios/smoke_mock.yaml --out out/run
survsim rig   configs/scenarios/disguise_and_tamper.yaml     # coverage, blind spots
survsim eval  out/run --threshold-s 25                        # alarms per camera-hour
```
Arch Linux, `pacman`/`pikaur`. Engine binaries in `~/data/unrealzoo/` (not in git).
GPU is an **8 GB** RTX 2070 Max-Q — below the 24 GB the plan assumed, so keep
capture at 720p and hero agents ≤ 12.

## Architecture in one paragraph
A Python **Director** (engine-agnostic) owns time, people, behaviours and ground
truth. It talks to a **Backend** — `mock` (pure Python, top-down, no GPU, used for
CI and behaviour work) or `unreal` (UnrealCV+ over TCP). Ground truth is written
by `survsim/gt/writers.py` in MOT / parquet / JSONL form. `survsim/sensors/degrade.py`
turns clean renders into CCTV-looking frames. `survsim/eval/` scores alarms the way
security evaluations do, not the way papers do.

```
Director (behaviours, planner, appearance epochs, camera director)
   -> Backend (mock | unreal)
   -> Recorder (MOT, world.parquet, events.jsonl, identities.parquet, calib)
   -> degrade profiles -> eval (i-LIDS timing, false alarms/camera-hour)
```

## Non-negotiable principles
1. **Truth is what happened, not what was planned.** Record actual agent state
   after avoidance, and read boxes from rendered masks (which include occlusion)
   rather than projecting an assumed box.
2. **Ground truth must follow the live camera.** A slewed PTZ head or a tampered
   camera no longer matches its configured pose; a blinded camera emits nothing.
3. **Hard negatives are as important as positives.** Every scenario pairs
   suspicious behaviour with innocent behaviour that looks identical to a rule
   (`commute_wait` vs `loiter`). Precision is meaningless without them.
4. **Report operational metrics.** False alarms per camera-hour against an
   operator budget, and i-LIDS timing (an alarm later than the tolerance is a
   *false alarm*, not a hit). Frame-level AUC flatters useless systems. An
   alarm on the wrong person is not a hit either — `alarms.score()` matches by
   actor identity (`actor_aware`) whenever alarms carry a `uid`. The
   `position_radius_m` mode (for tracker-fed uids) is a diagnostic only: its
   false-credit rate exceeds 30 % at 2 m where actor and bystander share the
   radius by construction (docs/research/32 §H62) — never a replacement.
5. **One definition of "visible"** — `geometry.usably_visible` (in frame, in
   front, and ≥ 20 px tall). Coverage analysis and "was this act observed?" must
   never disagree.
6. **Determinism.** Seeded per-subsystem RNG (`RngHub`); same seed ⇒ same run.
7. **SI units above the backend boundary.** Metres and degrees; the Unreal
   backend converts (×100, negate Y, negate yaw).

## Current state
See `docs/11_IMPLEMENTATION_STATUS.md` for the full table — keep that file and
this paragraph in step, because a stale "not yet" list causes exactly the
rediscovery these docs exist to prevent.

Working: core models, YAML loader, **behaviour engine (50 classes / 55 catalogue
names — the catalogue is complete; restricted zones are fenced, intrusions cross)**, planner, appearance epochs, population
generator, camera director (PTZ + 6 tamper kinds), rig analysis, scene
generator, mock backend, GT writers, degradation profiles, alarm evaluation,
**RF and audio sensors**, **COCO export with keypoints**, evasion taxonomy
(52 techniques), crowd model, and the Unreal backend (spawn / sync / capture /
mask-derived boxes / bones). Tracking metrics are validated against their
definitions, against real MOT20 footage, and against reference TrackEval
(`eval/trackeval_bridge.py`, M1 parity: MOT20 HOTA 0.443, docs/13). **Experiments framework**
(`survsim experiment`, H1–H59, 44 configs / 29 runners, every config has a runner, docs/research/32) and **7 degrade profiles**, including
`cctv_night_lowlight` / `cctv_thermal` fitted to LLVIP (H32) — **flagged overfit**
(hold-out 2/13 vis, 3/13 IR); raw "night" renders are not dark (luma 90). Renders use **CC0 Quaternius
bodies by default** (MetaHumans opt-in, see Licensing). Retired claims:
`docs/research/RETIRED_CLAIMS.md` + `tests/test_docs_retired_claims.py`. LLVIP night
reference in `~/data/real_cctv_night` (non-commercial: measure only, never train/ship).

Not yet, and *why*:
- **Real night penalty** — unmeasured; H32 night profiles are overfit (2/13, 3/13 hold-out).
  H36 full LLVIP fit needs sign-off for the ~3.5 GB download.
- **H41 keypoint auto-calibration** — needs head/foot keypoint GT export first.
- **Jetson latency** — no hardware; quantised-export numbers are desktop only.
- **H29 identities** — clothes-change ReID needs ~40 held-out changers (M15 renders, GPU + editor); 15 ids gave no real transfer.
- **Real gait from CCTV** — MOT20 downloaded, extraction not written.

## The engine situation (read before touching it)
The packaged runtime **hangs at frame 1** as of 2026-09-18 and executes no
commands — docs/12 §2 has the evidence and the three-way distinction between
"port open", "handshake answered" and "command executed" that makes this easy to
misdiagnose. Use `scripts/engine.sh`, never raw `pkill -f`.

We have **cooked `.pak` content only** — no `.uproject`, zero `.uasset`/`.umap`
outside the paks. Cooking is one-way, so those 101 maps **cannot** be re-opened
or re-cooked in an editor.

**[V] Unreal Editor 5.8.2 is installed** at `/opt/unreal-engine` (2026-09-18).
`scripts/verify_unreal_editor.sh` confirms the binary runs and that the
MetaHuman, MetaHumanCharacter, RigLogic and ControlRig plugins are present —
the parametric body/wardrobe stack the packaged runtime never exposed.
`Engine/Build/BatchFiles/Linux/{Build,GenerateProjectFiles}.sh` are present, so
plugins can be compiled against it.

### The wardrobe API is there — it was never an engine limitation

**[A] Verified present by introspection (2026-09-18).** A project at
`~/data/ue-projects/SurvSim` with the MetaHuman plugins enabled exposes:

* `MetaHumanWardrobeItem`, `MetaHumanWardrobeItemFactory`,
  `MetaHumanCharacterWardrobeIndividualAssets`
* `MetaHumanCharacterEditorSubsystem` — `commit_body_state`,
  `commit_face_state`, `commit_skin_settings`, `build_meta_human`
* `MetaHumanCharacterPalette.try_add_item_from_wardrobe_item`
* `ChaosOutfitAsset`, and MetaHuman**Crowd** outfit-assembly types

With **no plugins enabled the same project exposes 26 MetaHuman symbols and no
wardrobe API at all; enabling them gives 272.** The capability was never missing
from the engine — it was missing from our `.uproject`. Enable
`MetaHumanCharacter`, `MetaHumanCharacterUAF`, `MetaHumanGenerator`,
`MetaHumanCrowd`, `MetaHumanRuntime`, `MetaHumanSDK`, `RigLogic`.

**Still [A], not [V].** Symbols existing is not a dressed character. This is the
third time a capability looked available from a name; it is only verified when a
MetaHuman has been built, dressed and rendered. `scripts/ue/probe_api.py`
re-runs the introspection.

**The editor lies about success.** `-run=pythonscript` prints "Python script
executed successfully" for a script that did nothing and logged nothing. Have
in-editor scripts **write a result file**; stdout and exit codes prove nothing.

**[V] The editor renders headless (2026-09-18)** — MRQ and SceneCapture2D both
produce real frames under `xvfb-run`. The day-long "pure black render" was our
bug: **`unreal.Rotator(a, b, c)` is positionally (roll, pitch, yaw) in Python.**
Always use `pitch=`/`yaw=`/`roll=` keywords (a test enforces it), and when a
render is empty, dump actor transforms before suspecting the engine.
`docs/BUG_black_render.md` has the full account.

No UnrealCV needed for any of this.

Note the install is **manual-download only** — the AUR PKGBUILD uses a
`manual::` DLAGENT because Epic puts the Linux build behind an account login.

## Known correction — now itself corrected

An earlier version of this file claimed UnrealCV+ exposes
`metahuman/parametric/wardrobe`, `metahuman/parametric/body` and
`animation/soma_glb/apply` over the socket, and that the editor was therefore
unnecessary. **That is measured false.** The Python *client* carries those
method names; the packaged v3.1.0 *server* rejects every one of them, and
`soma_glb/apply` reports `attached=false playing=false` for all 40 candidate
clip names (docs/12 §5).

That claim was made by reading a command list, which is the same mistake the
working-style note below warns about — it is why capability is now tagged
[V]/[A]/[X] and why an unverified build is assumed broken.

**The editor route is the real one**, and the parametric body is [V] verified
there: height driven 171.75 → 190.0 cm across 30 named constraints. Rigging
also works offline from the DNA files the engine ships
(`scripts/ue/rig_from_dna.py`); only *source textures* still need an Epic
login, and those are cosmetic at surveillance standoff.

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
**[V] Now the default (2026-09-26):** `tools/render_run.py` renders 16 CC0
Quaternius bodies (`tools/import_characters.py --cc0`); `SURVSIM_CHARS=metahuman`
is the opt-in. Every set writes `licences.json` with `commercial_ml_ok` and
per-person `asset_licences` — check it before a set leaves engineering.
batch_v7 (CC0) recall 0.652 vs 0.665 for batch_v6 (MetaHuman), docs/12.
Full analysis: `docs/research/47_CHARACTER_AND_ASSET_LICENSING.md`.

## Working style
- Verify against the engine rather than assuming; tag findings **[V]** verified,
  **[A]** API exists but untried, **[X]** tried and broken.
- Before quoting a number in docs, grep `docs/research/RETIRED_CLAIMS.md`; `tests/test_docs_retired_claims.py` enforces it (add a row when an audit retires a claim); `tests/test_docs_numbers_in_facts.py` requires every decimal in 60, 65 v2 and docs/13 to be in `FACTS_VERIFIED.md` or the derived allowlist. `docs/research/AUDIT_LEDGER.md` lists every correction and what is still unverified.
- Shared docs (32, 66, FACTS): append or Edit your own section only — whole-file writes have clobbered other agents' work twice.
- Add a regression test whenever a bug is found — the inverted-pitch bug survived
  a self-consistent test and silently pointed every camera at the sky.
- **Run the render gate after every render** —
  `scripts/check_render.py <render_dir> --gt <run_dir> [--log <editor.log>]`
  fails by named check on blank frames, duplicated camera views, sky-only
  horizon, people missing where GT says, unanimated people, log errors
  (docs/12 "Automated render gate"). In-editor jobs: `scripts/check_job_result.py
  <result.json>`. Don't eyeball frames to find a failure class the gate knows.
- Tests: `pytest -n auto -q` (pytest-xdist installed). `scripts/hooks/install.sh`
  installs a <30 s pre-commit hook (ruff on staged files, compileall, check tests).
- Commit and push as work completes; a stop hook enforces a clean tree.

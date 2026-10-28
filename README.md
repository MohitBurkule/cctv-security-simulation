# cctv-security-simulation (SurvSim)

Photorealistic multi-camera, multi-person **CCTV security simulator** for developing and
evaluating video analytics (detection, tracking, multi-camera re-identification,
behaviour/intent, degraded imagery, multi-sensor fusion) against ground truth that real
CCTV can never provide.

Everything is controlled from **YAML + Python** and is **seeded / reproducible**: people
(body, clothes, gait, intent, runtime appearance changes), cameras (optics, PTZ, tamper,
degradation), environment (time, weather) and ground truth.

The analytics, evaluation harness and experiments live in the companion repo
**cctv-security**, which depends on this package.

## Samples

Ground truth drawn on rendered frames (boxes come from rendered masks, so occlusion is honest):

![GT overlay, day](docs/images/01_gt_triptych_day.png)
![GT overlay, crowd](docs/images/02_gt_triptych_crowd.png)

Same scenario, day vs night:

![Day vs night](docs/images/03_day_vs_night.png)

Camera tamper, and image-to-ground homography:


![Homography](docs/images/05_homography.png)

Procedurally generated sites (office, transit, perimeter, retail street, car park, event plaza; 3 seeds each):

![Site gallery](docs/images/site_gallery.png)
![Infinite sites](docs/images/infinite_sites_v2.png)

Behaviour, evasion and tamper catalogues:

![Behaviours](docs/images/behaviour_gallery.png)
![Evasion techniques](docs/images/evasion_gallery.png)
![Tamper kinds](docs/images/tamper_gallery.png)

## Quick start
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q -n auto
survsim validate configs/scenarios/smoke_mock.yaml
survsim run configs/scenarios/smoke_mock.yaml --out out/run
survsim rig configs/scenarios/disguise_and_tamper.yaml   # coverage, blind spots
```

## Layout
| Path | Content |
|---|---|
| `survsim/core`, `director`, `generator` | models, behaviours, planner, population, camera director, site generator |
| `survsim/backends` | `mock` (pure Python, no GPU) and `unreal` (UnrealCV+ / editor) |
| `survsim/gt`, `sensors`, `realism` | ground-truth writers, degradation/RF/audio/thermal, realism stats |
| `survsim/export`, `scripts/ue` | Unreal scene/character export and editor jobs |
| `configs/` | scenarios, cameras, environments, people |
| `docs/` | design docs 01-12 (architecture, behaviour catalogue, GT format, engine notes) |

See `CLAUDE.md` for the project principles and licensing notes (CC0 character stack by default).

## Licence
GNU Affero General Public License v3.0 or later. See [LICENSE](LICENSE).

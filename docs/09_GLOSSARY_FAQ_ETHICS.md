# 09 · Glossary, FAQ, Ethics & Licensing

## Glossary
* **Hero agent** — fully Director-controlled person with bone-level GT (≤ ~40 per scene).
* **Background agent** — Mass-AI crowd member; GT = boxes/ids only.
* **Appearance epoch** — time range where a person's appearance hash is constant.
* **Camera graph** — nodes = cameras, edges = overlap % or transit-time prior.
* **Degrade profile** — named pipeline of sensor degradations applied to clean frames.
* **Hard negative** — normal behaviour designed to resemble an anomaly.
* **MTMC** — multi-target multi-camera tracking. **ReID** — re-identification. **HOTA/IDF1** — MOT metrics. **MST** — Monk Skin Tone scale.

## FAQ
* *Why not Isaac Sim, it already has people SDG?* Its people extension is deprecated, has ~6 behaviours, weaker humans/appearance control; we keep it as an optional backend behind the same API.
* *Why not Unity?* Less free photoreal content; PeopleSansPeople is image-only, old Unity. We borrow its randomizer design.
* *Can I run without a GPU?* Yes: mock backend (top-down) runs everything except pixels — behaviours, GT, eval.
* *How do I add a behaviour?* Subclass `survsim.behaviours.Behaviour`, implement `plan(world, person) -> Iterator[Primitive]`, declare `labels`, register in the catalogue, add a smoke test and an anim mapping in `unreal/.../AnimMap.json`.
* *How do I add an environment?* Import/build UE map, drop `SurvSimZoneVolume`s and `SurvSimMount`s, run `survsim env export` → YAML.
* *Determinism?* Same seed + same versions.json ⇒ identical GT; pixels identical modulo GPU driver nondeterminism (documented).

## Ethics & responsible use
* Synthetic humans only; no real faces; MetaHuman presets are fictional. No training data scraped from real people.
* Skin-tone, age, body and mobility diversity are first-class parameters; eval reports break metrics down by these to expose bias.
* Behaviour labels describe **actions**, never demographics; "intent" is a scripted ground truth of the simulation, and docs warn against equating appearance with intent.
* Project is for developing and stress-testing detection/tracking/behaviour algorithms and camera-placement tooling in defence/civil-security contexts; no capability here is a weapon or targets real individuals.

## Licensing summary
| Component | Licence | Redistribution of derived renders |
|---|---|---|
| UnrealZoo code | Apache 2.0 | yes |
| UnrealZoo scene binaries | Apache 2.0 wrapper; assets per Fab/Marketplace licences | renders generally OK, raw assets no |
| UE5 / MetaHuman / City Sample / Megascans | Epic Content Licence (free for UE projects) | renders OK; MetaHuman use limited to UE-rendered content |
| Mixamo | Adobe terms (free, incl. commercial) | yes |
| CMU MoCap | free | yes |
| BEDLAM/AMASS | research-only | check before dataset release |
| Our code | Apache 2.0 (proposed) | — |

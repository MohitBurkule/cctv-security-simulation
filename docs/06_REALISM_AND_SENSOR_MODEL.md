# 06 · Realism, Lighting, Weather & CCTV Sensor Model

## 1. Rendering (UE5)
* Lumen GI + reflections, Nanite geometry, Virtual Shadow Maps, TSR/DLSS off for capture (deterministic), `r.ScreenPercentage 100`, motion blur **off** in-engine (we add physically-parameterised blur in post from velocity buffer so it is controllable), auto-exposure **off** (fixed EV per camera → our exposure model handles it).
* Materials: Megascans PBR; wet surfaces via weather-driven material parameter collection (roughness ↓, puddles); dirt/decal layers on cameras' housings for visual realism.
* MetaHuman LOD: hero agents LOD0-2 (strands or cards hair), background LOD3+ via Mass. Clothing: MetaHuman default + Marvelous Designer/CLO exports or Fab clothing packs; cloth sim (Chaos Cloth) only on hero agents in close-range cameras.
* Time of day: Sun Position Calculator plugin from real datetime + lat/lon of environment; street lamps auto-toggle by lux; interior lights on schedules.

## 2. Weather presets (parameter table)
| Preset | fog_density | rain | snow | wind | wetness | cloud | lux_day | notes |
|---|---|---|---|---|---|---|---|---|
| clear | 0.0 | 0 | 0 | 1 | 0 | 0.1 | 80k | hard shadows |
| overcast | 0.01 | 0 | 0 | 3 | 0.2 | 0.9 | 10k | flat lighting |
| rain_light | 0.03 | 0.3 | 0 | 4 | 0.8 | 1.0 | 5k | umbrellas p=0.4, hoods p=0.3 |
| rain_heavy | 0.08 | 1.0 | 0 | 8 | 1.0 | 1.0 | 2k | lens drops, run-for-cover |
| fog | 0.3-0.6 | 0 | 0 | 0.5 | 0.4 | 1.0 | 3k | thermal advantage |
| snow | 0.05 | 0 | 0.7 | 5 | 0.3 | 1.0 | 15k | ground albedo ↑, footprints |
| dust | 0.2 | 0 | 0 | 12 | 0 | 0.3 | 30k | border/desert |
| night_clear / night_overcast | … | | | | | | 0.1-5 | IR mode auto |

## 3. CCTV sensor model (post-process, Python, GPU-optional)
Implemented as `survsim.sensors.degrade.Pipeline([...])`, each stage a class with `__call__(frame_bundle, rng)`, where `frame_bundle` = rgb + depth + velocity + camera spec.
1. **Lens**: Brown-Conrady / Kannala-Brandt distortion (OpenCV `remap`), vignetting, chromatic aberration, softness (PSF blur radius as f(zoom)).
2. **Exposure**: EV from scene lux; auto-exposure lag (1st-order, 300-800 ms), WDR tone curve, clipping; PTZ zoom darkening.
3. **Sensor**: Poisson shot noise (photon count from EV & gain), Gaussian read noise, fixed-pattern noise, hot/dead pixels; ISP: demosaic artefacts (optional), denoise smearing (bilateral), sharpening halos.
4. **IR / thermal**: IR-cut off ⇒ desaturate + IR illuminator falloff (radial, range 30-60 m) + retro-reflective hi-vis blooming; thermal cams: emissivity-based grey mapping from material tags (skin 36 °C, clothes ambient+Δ, engines hot, glass reflective) + palettes.
5. **Motion**: per-pixel blur from velocity buffer × exposure time; rolling shutter skew from row time × velocity.
6. **Weather on lens**: raindrop refraction sprites, fog haze (depth-based transmittance), dirt/smudge alpha maps, spider webs at night (IR).
7. **Compression**: ffmpeg libx264/libx265 with per-camera CRF/bitrate/GOP/B-frames; **packet loss** ⇒ macroblock corruption / freeze frames / green blocks via decoding a damaged stream; frame drops & duplicates; variable FPS.
8. **OSD**: timestamp (font, position, drift), camera name, PTZ status, tamper text ("VIDEO LOSS").
9. **Tamper renders**: cover (dark + housing texture), spray (semi-opaque smear growing over 3 s), tilt (camera actor rotated in-engine → real), glare (torch/laser bloom), unplug (frozen last frame / blue screen / SMPTE bars), cut (no frames).

Degrade profiles in `configs/degrade/*.yaml`; the same clean run exported at
several profiles gives curriculum & robustness test sets.

## 4. Realism validation plan
* **FID/KID** between rendered frames and real CCTV sets (WILDTRACK, MEVA, PETS, DukeMTMC) per profile — target: synthetic-clean vs real gap shrinks after degrade.
* **Sim-to-real transfer**: train YOLO/ByteTrack on SurvSim only → test on WILDTRACK/MOT17; compare to MOTSynth baselines reported in the MOTSynth paper.
* **Human study** (internal, 5 people): 2AFC real-vs-synthetic thumbnails per profile; target ≥ 35 % fooled at 720p CCTV profile.
* **Physics sanity**: pedestrian speeds 0.8-1.8 m/s distribution vs literature; personal space ≥ 0.4 m; queue spacing.

## 5. Diversity budget (per 10-hour dataset)
≥ 500 unique hero identities · 10 skin tones balanced · 5 age bands · ≥ 60 garment meshes × unbounded colours/patterns · 8 gait styles · 6 environments · all weather presets · 24 h time-of-day coverage · every behaviour type ≥ 200 instances · hard-negative:positive ≥ 3:1.

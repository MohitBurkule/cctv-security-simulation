# SurvSim environment — machine-local setup notes (Arch Linux)

## This machine
| Item | Value | Note |
|---|---|---|
| OS | Arch Linux, kernel 7.1.3 | pacman + pikaur |
| CPU | 12 cores | |
| RAM | 31 GB | BOM asks 64 GB; fine for ≤8 cameras, tight for City Sample |
| GPU | RTX 2070 Max-Q, **8 GB VRAM**, driver 610.43.03 | BOM asks 24 GB. Mitigations below |
| Disk | 953 GB, 768 GB free | enough for UnrealZoo (72 GB zip + ~100 GB extracted) |
| `/tmp` | tmpfs 16 GB | **never download here** — use `~/data/` |
| System Python | 3.14.6 | too new for CV wheels → project uses a pinned venv |

### 8 GB VRAM mitigations (apply from week 3)
- Capture at 1280×720 per camera, upscale only where the scenario needs 1080p.
- Round-robin camera capture (never N SceneCaptures resident at once).
- Background crowd via Mass at low LOD; hero agents ≤ 12 rather than 40.
- Prefer UnrealZoo's smaller scenes; City Sample only in cut-down sublevels.
- Lumen on "Medium", Nanite on, virtual shadow maps at reduced resolution.
- Depth EXR at 5 Hz, not per frame (see D08: full-rate EXR ≈ 144 GB/camera-hour).

## Installed
```
pacman: uv python-pip base-devel vulkan-icd-loader vulkan-tools blender
        (ffmpeg, gcc, git, nvidia-utils already present)
```
`lib32-nvidia-utils` was skipped — multilib repo is disabled and it is not needed.

## Python environment
The system Python is 3.14, which has no wheels for most of the CV stack, so the
project pins its own interpreter:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
source .venv/bin/activate     # or use .venv/bin/python directly
```

**Reproducible install from the lock.** `requirements-lock.txt` pins every package at the
version installed in this `.venv` (generated with `uv pip freeze --python .venv/bin/python`;
local editable installs stripped). To rebuild the same environment:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
uv pip install --python .venv/bin/python --no-deps -e .
```

Regenerate it (same command, then strip `-e` lines) whenever a dependency is deliberately changed.

## Renderer tiers
1. **Mock backend** — pure Python top-down renderer. No GPU. Used for behaviour
   development, ground-truth correctness and CI. Works today.
2. **Blender** — installed via pacman. Offline path-traced "hero" frames and a
   cross-check on projection/calibration maths. Minutes per frame, not for video.
3. **UnrealZoo (UE5.7 prebuilt)** — the real-time path. **No engine build needed**:
   the release ships packaged scenes with the UnrealCV+ server inside, so Vulkan
   drivers are the only system requirement. Engine source (~150 GB, hours to
   compile) is only required later if we author custom maps.

### UnrealZoo binaries
```
source:   https://huggingface.co/datasets/UnrealZoo/UnrealZoo_UE5
file:     UnrealZoo_UE5_7_Linux_v3.1.0.zip   (72.25 GB)
target:   ~/data/unrealzoo/
extract:  ~/data/unrealzoo/UnrealEnv/
export UnrealEnv=/home/mohit/data/unrealzoo/UnrealEnv
```
Other variants in the same dataset repo: `UE5_7_Win64_v3.1.0`,
`UE5_6_Linux_v3.0.2`, `UE5_6_Win64_v3.0.1` (~73 GB each).

Client library: `git clone https://github.com/UnrealZoo/unrealzoo-gym` then
`pip install -e .` into the same venv (Python 3.9+ per upstream; 3.11 works).

## Known risk carried from research
`D08` records UnrealCV issue **#339 — "Any skeletal mesh causes a 100% crash for
segmentation"** on UE 5.3/5.6, plus several instance-mask defects (#330 gamma/8-bit
ID loss, #331 TAA/Lumen mask noise, #282 PIE-vs-Standalone divergence). Verify
segmentation on a skeletal mesh **before** building on it; the fallback is our own
stencil pass inside the SurvSim plugin, scheduled for week 3.

## Operating limits learned (2026-09-27)
Three incidents in one day (root disk full ~10:05, `/tmp` quota hit ~11:13, editor
OOM-killed ~14:40) turned these into hard rules, not guidelines.

| Resource | Limit | Rule |
|---|---|---|
| Root disk | 953 GB total | Keep `out/experiments/*` to **≤ ~20 GB per experiment**; `df` before any render or training sweep; delete scored PNG sets once a batch is audited. Batches v1–v5 (~85 GB) were deleted with the user's OK; v6–v8 are kept as the current evidence base. |
| `/tmp` | 16 GB tmpfs | **Never** copy render batches or scratch there — a stale 6.8 GB batch copy tripped the quota at 11:13. Agent transcripts and scratchpads also live in `/tmp`; treat it as small and disposable, not storage. |
| RAM | 31 GB, no swap headroom | ≤ 6 workers per agent/sweep; run **one heavy job at a time** (the 14:40 OOM was 11 CPU workers + a QAT run + the operator-sim editor together). Check `free -g` before starting another heavy job, and only restart the persistent editor when ≥ 8 GB is free. |
| GPU | 8 GB VRAM (RTX 2070 Max-Q) | Treat 7.2 GB resident as the guard rail — leave headroom for the OS compositor and any concurrent CPU-side job that also touches the device. |

Reproducible installs and secret hygiene, in force since `a633a1d`/`09a5e4f`:
```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
uv pip install --python .venv/bin/python --no-deps -e .
```
The pre-commit hook (`scripts/hooks/secret_scan.py`, allowlist
`secret_allowlist.txt`) blocks staged secrets; see `tests/test_secret_scan.py`.

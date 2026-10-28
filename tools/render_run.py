"""Render a SurvSim run through the persistent Unreal editor, from our venv.

    scripts/ue_mcp.sh start                       # once per session
    .venv/bin/python tools/render_run.py out/render_demo --frames 10 --cameras cam_01
    .venv/bin/python tools/render_run.py out/render_demo --frames 40 --lighting night

Computes the animation plan here (the editor's Python has no yaml, so survsim
cannot be imported there), writes a JSON request, submits
scripts/ue/render_run.py as a job to scripts/ue/job_runner.py, waits (bounded)
while grepping the editor log for "LogPython: Error", then verifies the PIXELS
rather than any success flag — every success flag in this pipeline has been
wrong at some point.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

JOB = ROOT / "scripts" / "ue" / "render_run.py"
REQUEST = Path("/tmp/survsim_render_request.json")
PLAN = Path("/tmp/survsim_render_plan.json")
# Stylised/broken imports kept out of renders: cube/chibi bodies, a 5 cm mesh,
# and a mis-oriented armature (docs/12, "Characters").
EXCLUDE = {"CesiumMan", "cube_woman_character_75ik", "cube_guy_character_k1ic",
           "animated_woman_9kf7", "animated_human_c3ib", "characters_matt_66kq",
           "characters_sam_ucle", "lis_gjuw", "character_animated_dgoc"}
STRIDE_CYCLE_M = 1.45
REF_HEIGHT_M = 1.75
FEET = {"CesiumMan": ("leg_joint_L_5", "leg_joint_R_5")}
# glTF metres -> UE cm (x100) times the x100 root-bone scale these rigs keep.
ROOT_SCALE_CM = 10000.0  # [V] measured: rendered size is index height_m x 1e4 cm
INDEX = Path.home() / "data/assets/index.json"
PLAN_CATALOG_KEYS = ("height_cm", "foot_curves", "walk_len_s", "run_len_s", "stance_s",
                     "yaw_offset", "z_offset_cm")
CAT_RENDER = Path("/tmp/survsim_catalog_render.json")
CATALOG = Path.home() / "data/ue-projects/SurvSim/Saved/survsim_characters.json"
MH_BODIES = ROOT / "configs" / "people" / "metahuman_bodies.json"
MH_BAKED = "/Game/SurvSim/MetaHumans/Baked"
BAKED_DIR = Path.home() / "data/ue-projects/SurvSim/Content/SurvSim/MetaHumans/Baked"


def label_keys(manifest: dict, run_dir: Path) -> int:
    """Copy each key's world behaviour label (gt/world.parquet) onto the manifest key.

    The animation plan picks idle / look-around / run from it (anim_plan.
    BEHAVIOUR_ANIM). Matched on (uid, round(t * fps)); returns keys labelled.
    """
    path = run_dir / "gt" / "world.parquet"
    if not path.is_file():
        return 0
    import pandas as pd
    fps = float(manifest.get("fps", 10))
    w = pd.read_parquet(path, columns=["t", "uid", "label"])
    lab = {(u, round(t * fps)): lb for u, t, lb in zip(w["uid"], w["t"], w["label"])}
    n = 0
    for tr in manifest.get("tracks", []):
        for k in tr["keys"]:
            lb = lab.get((tr["uid"], round(float(k.get("t", k["frame"] / fps)) * fps)))
            if lb:
                k["label"] = lb
                n += 1
    return n


def metahuman_catalog(spec_path: Path = MH_BODIES, baked_dir: Path | None = BAKED_DIR) -> dict:
    """Catalogue entries for the baked MetaHuman bodies (kind "metahuman").

    Stature from the measured body-mesh bounds (/0.83: the head is the face
    mesh). The walk clip is in place; its measured ground speed is normalised
    to a 1.75 m person so anim_plan's height scaling gives rate proportional to
    speed. A body whose baked folder is missing on disk is left out.
    """
    spec = json.loads(spec_path.read_text())
    common = spec["_common"]
    out = {}
    for name, v in spec.items():
        if name.startswith("_"):
            continue
        if baked_dir is not None and not (baked_dir / name).is_dir():
            continue
        e = {k: val for k, val in common.items() if not k.startswith("clip_")}
        base = f"{MH_BAKED}/{name}"
        e.update(mesh=f"{base}/SKM_{name}_Body", face=f"{base}/SKM_{name}_Face",
                 xmax_cm=v.get("xmax_cm", 46.0),
                 height_cm=round(v["body_bounds_z_cm"] / 0.83, 1),
                 walk_speed_mps=round(common["clip_ground_speed_mps"] * 1.75
                                      / common["clip_ref_stature_m"], 3))
        if os.environ.get("SURVSIM_MH_GARMENT") == "1":
            e["outfit"] = f"{base}/OA_{name}_0"
        out[name] = e
    return out


def run_appearance_epochs(manifest: dict, run_dir: Path) -> dict:
    """The manifest's `appearance_epochs`, else computed from the run's GT (a
    manifest exported before epochs were carried)."""
    if manifest.get("appearance_epochs") is not None:
        return manifest["appearance_epochs"]
    ident = run_dir / "gt" / "identities.parquet"
    if not ident.is_file():
        return {}
    import pandas as pd

    from survsim.export.unreal_scene import appearance_epochs
    rows = pd.read_parquet(ident).to_dict("records")
    world = run_dir / "gt" / "world.parquet"
    wr = pd.read_parquet(world, columns=["frame", "uid", "appearance_epoch"]).to_dict("records") \
        if world.is_file() else []
    return appearance_epochs(rows, wr, float(manifest.get("fps", 10)))


def gt_heights(run_dir: Path) -> dict[str, float]:
    """{uid: height_m} from the run's gt/identities.parquet (the Director's truth).

    92e1e97 fixed the export, but manifests exported before it still say 1.70 m
    for everyone, and anim_plan.person_height_m clamps to 1.55-1.95 m (street_crowd
    has four people at 1.22-1.5 m). The render must draw the GT stature, so it is
    read here and wins over both.
    """
    ident = run_dir / "gt" / "identities.parquet"
    if not ident.is_file():
        return {}
    import pandas as pd
    df = pd.read_parquet(ident, columns=["uid", "height_m"]).dropna()
    return {str(u): float(h) for u, h in zip(df["uid"], df["height_m"])}


def use_gt_heights(data: dict, heights: dict[str, float]) -> int:
    """Put GT heights into the manifest wardrobes and make anim_plan use them
    unclamped. Returns the number of people whose height changed."""
    from survsim.export import anim_plan
    n = 0
    for person in data.get("people", []):
        h = heights.get(person.get("uid"))
        if h is None:
            continue
        w = person.setdefault("wardrobe", {}) or {}
        person["wardrobe"] = w
        if w.get("height_m") is None or abs(float(w["height_m"]) - h) > 1e-6:
            n += 1
        w["height_m"] = h
    orig = getattr(anim_plan, "_person_height_m_orig", anim_plan.person_height_m)
    anim_plan._person_height_m_orig = orig

    def person_height_m(manifest, uid, seed=0):
        return heights[uid] if uid in heights else orig(manifest, uid, seed)
    anim_plan.person_height_m = person_height_m
    return n


def camera_keys(run_dir: Path, fps: float, n_frames: int) -> dict:
    """Live camera pose per frame from gt/camera_status.jsonl (PTZ slews, tilts).

    Ground truth follows the live camera (CLAUDE.md principle 2), so the render
    must too. Only cameras whose pose actually changes are returned. Unreal
    convention: yaw negated, pitch as-is (survsim.export.unreal_scene).
    """
    path = run_dir / "gt" / "camera_status.jsonl"
    if not path.is_file():
        return {}
    per: dict[str, dict[int, dict]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        f = round(float(r["t"]) * fps)
        if f >= n_frames or r.get("yaw") is None:
            continue
        per.setdefault(r["camera"], {})[f] = {
            "frame": f, "yaw": -float(r["yaw"]), "pitch": float(r["pitch"]),
            "hfov_deg": float(r.get("hfov_deg") or 0) or None}
    out = {}
    for cam, keys in per.items():
        vals = {(k["yaw"], k["pitch"], k["hfov_deg"]) for k in keys.values()}
        if len(vals) > 1:
            out[cam] = [keys[f] for f in sorted(keys)]
    return out


PRESET_EV = {"day": 13.0, "overcast": 11.5, "night": 1.0}
LUMA_BAND, LUMA_TARGET = (75.0, 125.0), 100.0   # correct toward 100 outside this
# Night CCTV is darker by design but must still show people (gate: luma >= 20).
NIGHT_BAND, NIGHT_TARGET = (40.0, 120.0), 70.0


def _ev_for(files, base: float, lighting: str) -> tuple[float, float, float]:
    """EV100 for a set of probe frames: (ev, median luma, 15th-percentile luma)."""
    import math

    import numpy as np
    from PIL import Image
    imgs = [np.asarray(Image.open(f).convert("L")) for f in files]
    if not imgs:
        return base, LUMA_TARGET, LUMA_TARGET
    # Median, not mean: a wet ground's sun glint (a small saturated patch)
    # pulled the mean into band while most of the frame was dark
    # ([X] 2026-09-26 scene_variety_v2 car_park_s1: mean 61, frame visibly black).
    measured = float(np.mean([np.median(i) for i in imgs]))
    p15 = float(np.mean([np.percentile(i, 15) for i in imgs]))
    ev = base
    band, target = (NIGHT_BAND, NIGHT_TARGET) if lighting == "night" else (LUMA_BAND, LUMA_TARGET)
    if not band[0] <= measured <= band[1]:
        ev = round(base + max(-3.0, min(3.0, math.log2(max(measured, 0.5) / target))), 2)
    if lighting != "night":
        # Shadow lift: open up until the 15th percentile would reach ~35
        # ([X] transit_station s3: 27.8 % crushed), never pushing the median
        # past 150, and never more than 2 stops (short probes under-read wet
        # shaded scenes: event_plaza s3).
        gain = 2.0 ** (base - ev)
        if p15 * gain < 25.0:
            lift = min(math.log2(35.0 / max(p15 * gain, 1.0)),
                       math.log2(150.0 / max(measured * gain, 1.0)))
            if lift > 0:
                ev = round(ev - min(lift, 1.5), 2)
        ev = max(ev, base - 2.0)
    return ev, measured, p15


def meter(request: dict, out_root: Path, wait: float) -> float:
    """Probe-render 3 frames, measure mean luma, correct EV100 once.

    A fixed EV assumes a sunlit scene; a low sun behind tall blocks leaves the
    view in shade at luma ~30. EV100 moves by log2(measured / target) (higher EV
    = darker). Both values are written to <out>/meter.json.
    """


    from tools.ue_job import submit
    base = PRESET_EV.get(request["lighting"], 13.0)
    probe_dir = out_root.parent / (out_root.name + "_meter")
    # 640x360: luma is resolution-independent and the probe must not be the
    # VRAM peak of the whole set.
    probe = dict(request, n_frames=min(request.get("frame_start", 0) + 3, request["n_frames"]), out=str(probe_dir), ev100=base,
                 probe_res=[640, 360])
    REQUEST.write_text(json.dumps(probe))
    r = submit(JOB, name=f"{int(time.time())}_meter", wait=wait)
    (out_root.parent / "meter_result.json").write_text(json.dumps(r, indent=1, default=str))
    if not r.get("ok"):
        # [X] a crashed probe used to fall through to "luma = target".
        raise SystemExit(f"meter probe failed: {str(r.get('error'))[:500]}")
    ev, measured, p15 = _ev_for(sorted(probe_dir.glob("*/*.png")), base, request["lighting"])
    # Per camera: each view has its own light (cam_sub looking into a shed's
    # shade crushed 25.6 % under one global EV, batch_v5 [X]).
    cam_ev = {}
    for cd in sorted(p for p in probe_dir.iterdir() if p.is_dir()):
        cam_ev[cd.name] = _ev_for(sorted(cd.glob("*.png")), base, request["lighting"])[0]
    request["camera_ev"] = cam_ev
    # Lights (incl. night spots) are placed by the probe job itself before it
    # renders, so the probe meters the final lighting.
    (out_root / "meter.json").write_text(json.dumps(
        {"base_ev100": base, "probe_luma": round(measured, 1), "probe_p15": round(p15, 1),
         "ev100": ev, "camera_ev": cam_ev,
         "clamp_stops": 3.0}))
    print(f"meter: luma {measured:.1f} at EV {base} -> EV {ev}")
    request["rebuild_env"] = False
    return ev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--cameras", nargs="*", default=None)
    ap.add_argument("--lighting", default="day", choices=["day", "overcast", "night"])
    ap.add_argument("--ev100", type=float, default=None)
    # [X] 2026-09-26 (docs/12 "Real textures + sky"): on the yard layouts the
    # ambientCG sets and HDRI sky moved hist_entropy / rms_contrast / chroma
    # AWAY from the real-CCTV band (flat grey concrete, overcast fill), so both
    # are opt-in and the procedural set stays the default.
    ap.add_argument("--hdri", default="none",
                    help="Poly Haven .hdr for the sky light; 'auto' = seeded pick from the "
                         "asset index for day/overcast, 'none' = procedural sky (default)")
    # 2026-09-26: M_SurvPBR_v9 compiles (439 instr.; v8's 0 was a sampler-type /
    # default-texture mismatch) and moves 3/3 test frames +1 stat into the real band.
    ap.add_argument("--textures", action=argparse.BooleanOptionalAction, default=True,
                    help="ambientCG PBR sets where the index has one (default on; "
                         "--no-textures for the procedural set)")
    ap.add_argument("--rebuild-env", action="store_true")
    ap.add_argument("--no-meter", dest="meter", action="store_false",
                    help="skip the probe render that corrects EV100 for day scenes")
    ap.add_argument("--site", default=None, help="force a site type (env_plan.SITE_TYPES)")
    ap.add_argument("--env-v1", action="store_true",
                    help="the v1 environment plan (rim props round an open square) instead of "
                         "the v2 site layout (survsim/generator/layout.py)")
    ap.add_argument("--env-seed", type=int, default=None,
                    help="layout seed (default: the run's seed)")
    ap.add_argument("--entrance-gate", choices=["fail", "warn"], default="fail",
                    help="warn: record an entrance no camera usably sees instead of failing "
                         "(a scenario whose own rig leaves an entrance uncovered)")
    ap.add_argument("--layout", type=Path, default=None,
                    help="a saved layout JSON (Layout.to_dict()) to build exactly")
    ap.add_argument("--legacy-street", action="store_true",
                    help="the fixed street instead of the scenario's environment plan")
    ap.add_argument("--no-relight", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--wait", type=float, default=900.0)
    ap.add_argument("--window", nargs=2, type=float, metavar=("T0", "T1"), default=None,
                    help="render only simulation time [T0, T1) s (e.g. 95 135 around an "
                         "appearance change); overrides --frames. Frames keep their absolute "
                         "GT numbers (render/<cam>/0950.png = GT frame 950)")
    ap.add_argument("--id-mask", action="store_true",
                    help="also render the per-person stencil ID pass and write mask-derived "
                         "GT boxes to <out>/../gt (survsim.gt.mask_boxes)")
    ap.add_argument("--id-mask-depth-test", action=argparse.BooleanOptionalAction, default=None,
                    help="override survsim_scene.ID_MASK_DEPTH_TEST for this render (A/B checks)")
    args = ap.parse_args()

    from survsim.export.anim_plan import person_plan
    from tools.ue_job import submit

    manifest = (args.run_dir / "unreal" / "manifest.json").resolve()
    if not manifest.is_file():
        print(f"no manifest; run: survsim export {args.run_dir}")
        return 1
    if not CATALOG.is_file():
        print("no character catalogue; submit scripts/ue/import_gltf.py first")
        return 1
    chars = {k: v for k, v in json.loads(CATALOG.read_text()).items() if "error" not in v}
    for name, info in chars.items():
        # Ground speed of the walk clip at rate 1.0 for a 1.75 m person: one
        # cycle is two steps, ~1.45 m for that height.
        info.setdefault("walk_speed_mps", round(STRIDE_CYCLE_M / max(0.5, info["walk_len_s"]), 3))
        feet = FEET.get(name)
        if feet and Path(info.get("source", "")).is_file():
            # [V] measured from the clip by forward kinematics, not guessed.
            from survsim.export.gait_fk import gait_stats
            g = gait_stats(info["source"], *feet)
            scale = REF_HEIGHT_M / (info["height_cm"] / 100.0)
            info["walk_speed_mps"] = round(g["ground_speed"] * scale, 3)
            info["stance_s"] = g["stance_t"]
            info["gait_fk"] = g
    # Character stack (docs/research/47, "Recommended stack"): the DEFAULT is
    # the licence-compliant CC0 set - rebaked Quaternius bodies driven by their
    # own walk/idle/run clips, region-painted clothing (survsim/export/cc0_*).
    # SURVSIM_CHARS=metahuman opts into the baked MetaHumans (Epic's terms bar
    # AI training/testing: such a set is marked commercial_ml_ok=false);
    # SURVSIM_CHARS=a,b,c is an explicit allow-list (e.g. CesiumMan, CC-BY).
    from survsim.export.cc0_chars import cc0_catalog
    mode = os.environ.get("SURVSIM_CHARS", "cc0")
    cc0 = cc0_catalog()
    chars = {k: v for k, v in chars.items() if not k.endswith(("_m", "_n"))}
    chars.update(cc0)
    mh = {}
    if mode == "metahuman":
        mh = metahuman_catalog()
        chars.update(mh)
        allow = sorted(mh)
    elif mode == "cc0":
        allow = sorted(k for k, v in cc0.items() if not v.get("exclude"))
    else:
        allow = mode.split(",")
    if not allow:
        print(f"no usable characters for SURVSIM_CHARS={mode}; run tools/import_characters.py --cc0")
        return 1
    for name, info in chars.items():
        info["exclude"] = name not in allow
    print(f"characters ({mode}): {len(allow)} usable: {', '.join(allow)}")
    idx = {re.sub(r"[^A-Za-z0-9_]", "_", a["name"]) + "_m": a
           for a in json.loads(INDEX.read_text())["assets"]} if INDEX.is_file() else {}
    for name, info in chars.items():
        if name in idx and idx[name].get("height_m"):
            # Bone sockets read nonsense on these rigs; the index measured the
            # glTF mesh bounds, which the baked copy preserves (x100 cm).
            info["height_cm"] = round(float(idx[name]["height_m"]) * ROOT_SCALE_CM, 2)
        for k in ("roll", "pitch", "yaw"):
            v = os.environ.get(f"SURVSIM_{k.upper()}_OFFSET")
            if v is not None and name != "CesiumMan" and name not in mh and name not in cc0:
                info[f"{k}_offset"] = float(v)
    CAT_RENDER.write_text(json.dumps(chars))
    data = json.loads(manifest.read_text())
    last = max((k["frame"] for t in data.get("tracks", []) for k in t["keys"]), default=0)
    n_frames = min(args.frames, last + 1)
    frame_start = 0
    if args.window:
        fps_ = float(data.get("fps", 10))
        frame_start = max(0, round(args.window[0] * fps_))
        n_frames = min(last + 1, round(args.window[1] * fps_))
        if n_frames <= frame_start:
            print(f"empty window {args.window} (run ends at frame {last})")
            return 1
        print(f"window: {args.window[0]}-{args.window[1]} s = frames {frame_start}..{n_frames - 1}")
    label_keys(data, args.run_dir)
    hts = gt_heights(args.run_dir)
    print(f"heights: {len(hts)} from GT, {use_gt_heights(data, hts)} differed from the manifest")
    people = person_plan(data, n_frames, chars)
    site = None
    scen_path = args.run_dir / "meta" / "scenario.resolved.json"
    if scen_path.is_file():
        from survsim.export.env_plan import infer_site_type
        site = args.site or infer_site_type(
            json.loads(scen_path.read_text()).get("environment") or {})
    from survsim.export.outfits import hivis_probability, outfit
    for e in people:
        if chars[e["character"]].get("kind") in ("metahuman", "cc0"):
            e["outfit"] = outfit(e["uid"], int(data.get("seed", 0)), hivis_probability(site))
    # Experiment hook (H47 evasion): palette overrides {"all"|uid: {...}} merged
    # onto the drawn outfit; recorded in render_appearance.json like any outfit.
    if ov_path := os.environ.get("SURVSIM_OUTFIT_OVERRIDE"):
        ov = json.loads(Path(ov_path).read_text())
        for e in people:
            if e.get("outfit"):
                e["outfit"] = {**e["outfit"], **ov.get("all", {}), **ov.get(e["uid"], {})}
        print(f"outfit override: {ov_path}")
    # Appearance epochs (outfit_change / disguise): one palette per epoch, the
    # person split into one actor per epoch shown from the epoch's first frame.
    epochs = run_appearance_epochs(data, args.run_dir)
    from survsim.export.anim_plan import split_epochs
    from survsim.export.outfits import epoch_outfits
    palettes = {e["uid"]: epoch_outfits(e["uid"], epochs[e["uid"]], e["outfit"],
                                        int(data.get("seed", 0)))
                for e in people if e["uid"] in epochs and e.get("outfit")}
    people = split_epochs(people, palettes)
    for uid, pals in palettes.items():
        print(f"epochs {uid}: " + "; ".join(
            f"e{p['epoch']}@frame {p['frame_start']} top={p['top']} face_cover="
            f"{(p.get('face_cover') or {}).get('kind')}" for p in pals))
    PLAN.write_text(json.dumps({"n_frames": n_frames, "people": people}))
    env_plan = None
    scen_path = args.run_dir / "meta" / "scenario.resolved.json"
    if not args.legacy_street and (scen_path.is_file() or args.site or args.layout):
        from survsim.export.env_plan import (
            entrance_camera_candidates,
            lighting_for,
            plan_for_run,
            plan_from_layout,
            plan_v2_for_run,
            walk_envelope,
            walked_cells,
        )
        scen = json.loads(scen_path.read_text()) if scen_path.is_file() else {}
        if args.layout:
            lay = json.loads(args.layout.read_text())
            env_plan = plan_from_layout(lay, data["cameras"], walked_cells(data),
                                        lighting_for(scen, args.lighting))
            env_plan["entrance_cameras"] = entrance_camera_candidates(env_plan, data["cameras"])
        elif args.env_v1:
            env_plan = plan_for_run(scen, data, walk_envelope(data), preset=args.lighting,
                                    site_type=args.site)
        else:
            env_plan = plan_v2_for_run(scen, data, walk_envelope(data), preset=args.lighting,
                                       site_type=args.site, seed=args.env_seed)
        (out_root_early := (args.out or (args.run_dir / "render")).resolve()).mkdir(
            parents=True, exist_ok=True)
        (out_root_early.parent / "env_plan.json").write_text(json.dumps(env_plan, indent=1))
        print(f"environment: {env_plan['site_type']} -> {env_plan['layout']} "
              f"({len(env_plan['items'])} items, wet={env_plan['weather']['wet']})")
    from survsim.export import pbr_sets
    pbr = pbr_sets.pbr_library() if args.textures else {}
    hdri = None if args.hdri in (None, "none") else args.hdri
    if hdri == "auto":
        pick = pbr_sets.pick_hdri(int(data.get("seed", 0))) if args.lighting != "night" else None
        hdri = pick["file"] if pick else None
    if hdri and env_plan is not None:
        sun = pbr_sets.sun_for_hdri(hdri)
        if sun["peak_ratio"] > 50:          # a visible sun: light the scene from it
            env_plan.setdefault("lighting", {})
            env_plan["lighting"]["sun_yaw"], env_plan["lighting"]["sun_pitch"] = \
                pbr_sets.ue_sun_rotation(sun)
        print(f"hdri: {Path(hdri).name} sun {sun}")
    print(f"textures: {sorted(pbr)}")
    cam_keys = camera_keys(args.run_dir, float(data.get("fps", 10)), n_frames)
    out_root = (args.out or (args.run_dir / "render")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    # Appearance truth for ReID: which body and clothes each identity wore,
    # and the licence of every asset the person depends on (docs/research/47).
    from survsim.gt.licences import dataset_licences, licence_column, person_licences
    lic = {e.get("person_uid", e["uid"]): person_licences(e["character"], chars[e["character"]])
           for e in people}
    appearance = {}
    for e in people:
        uid = e.get("person_uid", e["uid"])
        rec = appearance.setdefault(uid, {
            "character": e["character"], "height_cm": e["height_cm"],
            "outfit": e.get("outfit"), "colour": e.get("colour"),
            "asset_licences": lic[uid]})
        if uid in palettes:
            rec["outfit"] = palettes[uid][0]
            rec["epochs"] = [dict(p, rendered=any(
                x.get("person_uid") == uid and x.get("epoch") == p["epoch"] for x in people),
                actor=uid if not p["epoch"] else f"{uid}#e{p['epoch']}",
                t_from=next(q["t_from"] for q in epochs[uid] if q["epoch"] == p["epoch"]),
                appearance_hash=next(q["appearance_hash"] for q in epochs[uid]
                                     if q["epoch"] == p["epoch"]))
                for p in palettes[uid]]
    (out_root.parent / "render_appearance.json").write_text(json.dumps(appearance, indent=1))
    # Kept with the set: the amodal GT (survsim.gt.amodal) replays each person's
    # clip phase and yaw from it, so it must be the plan that was rendered.
    (out_root.parent / "render_plan.json").write_text(json.dumps(
        {"n_frames": n_frames, "people": people,
         "catalog": {c: {k: chars[c].get(k) for k in PLAN_CATALOG_KEYS}
                     for c in {e["character"] for e in people}}}))
    # Kept out of render_appearance.json, whose keys are all uids.
    (out_root.parent / "render_window.json").write_text(json.dumps(
        {"frame_start": frame_start, "frame_end": n_frames - 1, "fps": float(data.get("fps", 10)),
         "window_s": list(args.window) if args.window else None,
         "epoch_changes": {u: [(p["epoch"], p["frame_start"]) for p in ps]
                           for u, ps in palettes.items()}}, indent=1))
    summary = dict(dataset_licences(lic), characters=mode)
    (out_root.parent / "licences.json").write_text(json.dumps(summary, indent=1))
    if summary["attributions"]:
        (out_root.parent / "ATTRIBUTION.txt").write_text("\n".join(summary["attributions"]) + "\n")
    ident = args.run_dir / "gt" / "identities.parquet"
    if ident.is_file():
        import pandas as pd
        df = pd.read_parquet(ident)
        df["asset_licences"] = [licence_column(lic.get(u, [])) or "LicenseRef-Unknown|unrendered"
                                for u in df["uid"]]
        df["commercial_ml_ok"] = summary["commercial_ml_ok"]
        # Per-epoch render palette and the render frame the epoch starts at.
        pal = {(u, p["epoch"]): p for u, ps in palettes.items() for p in ps}
        base = {u: r.get("outfit") for u, r in appearance.items()}
        rows = [pal.get((u, int(ep))) or (base.get(u) if int(ep) == 0 else None)
                for u, ep in zip(df["uid"], df["epoch"])]
        df["render_outfit"] = [json.dumps(r) if r else None for r in rows]
        df["render_frame_start"] = [int(r.get("frame_start", 0)) if r else -1 for r in rows]
        df["render_face_cover"] = [((r or {}).get("face_cover") or {}).get("kind") for r in rows]
        df.to_parquet(out_root.parent / "identities_rendered.parquet", index=False)
    print(f"licences: {summary['asset_licences']} commercial_ml_ok={summary['commercial_ml_ok']}")
    request = {
        "manifest": str(manifest), "plan": str(PLAN), "catalog": str(CAT_RENDER),
        "out": str(out_root), "n_frames": n_frames, "cameras": args.cameras,
        "lighting": args.lighting, "ev100": args.ev100, "hdri": hdri, "pbr": pbr,
        "rebuild_env": args.rebuild_env, "camera_keys": cam_keys,
        "env_plan": env_plan, "relight": not args.no_relight,
        "entrance_gate": args.entrance_gate, "frame_start": frame_start}
    t0 = time.time()
    if args.meter and args.ev100 is None:
        request["ev100"] = meter(request, out_root, args.wait)
    request["id_mask"] = bool(args.id_mask)
    request["mask_out"] = str(out_root.parent / "mask")
    if args.id_mask_depth_test is not None:
        request["id_mask_depth_test"] = bool(args.id_mask_depth_test)
    REQUEST.write_text(json.dumps(request))
    result = submit(JOB, name=f"{int(t0)}_render", wait=args.wait)
    print(json.dumps(result, indent=2)[:3000])
    print(f"round trip {time.time() - t0:.1f} s")
    result["round_trip_s"] = round(time.time() - t0, 1)
    (out_root.parent / "render_result.json").write_text(json.dumps(result, indent=1, default=str))
    if not result.get("ok") or not result.get("render_success", False):
        return 1

    gt_root = args.run_dir
    if args.id_mask:
        from survsim.gt.mask_boxes import build_tight_gt
        # Results from before the flag was recorded rendered without the depth test.
        summ = build_tight_gt(out_root.parent, args.run_dir, result["stencil_ids"],
                              depth_tested=bool(result.get("id_mask_depth_test", False)))
        print(f"mask GT: {json.dumps(summ['cameras'])}")
        gt_root = out_root.parent
        from survsim.gt.amodal import build_amodal
        am = build_amodal(out_root.parent)
        print(f"amodal GT: clean IoU holdout {am.get('clean_iou_holdout')}, occluded "
              f"containment {am.get('occluded_containment')}")
    # Rendered stature vs the Director's (92e1e97: a 1.70 m fallback rendered
    # every person at one height). Also in the gate below; printed here so a
    # FAIL is visible even when the gate output is filtered.
    from survsim.checks.render import check_height_files
    hc = check_height_files(args.run_dir / "gt", out_root)
    print(f"height check: {hc.status} {hc.detail}")
    # The render gate (scripts/check_render.py): named checks against GT.
    import subprocess
    cmd = [sys.executable, str(ROOT / "scripts/check_render.py"), str(out_root)]
    if (gt_root / "gt").is_dir():
        cmd += ["--gt", str(gt_root)]
    gate = subprocess.run(cmd, capture_output=True, text=True, check=False)
    lines = gate.stdout.strip().splitlines()
    print("\n".join(l for l in lines if "FAIL" in l or "WARN" in l) or lines[-1:] and lines[-1])
    ok = gate.returncode == 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Import index.json characters into the persistent editor, ONE per job.

    .venv/bin/python tools/import_characters.py [--max 8]

A crash in Interchange on one glTF kills the editor; one-per-job means the
catalogue keeps every character imported before it, and the crashing file is
named in the output. Characters already in the catalogue are skipped.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.ue_job import submit

INDEX = Path.home() / "data/assets/index.json"
CATALOG = Path.home() / "data/ue-projects/SurvSim/Saved/survsim_characters.json"
SCRIPT = ROOT / "scripts/ue/import_gltf.py"
REBAKE_DIR = Path.home() / "data/assets/characters_rebake"
CC0_DIR = Path.home() / "data/assets/characters_cc0"
CC0_JOB = ROOT / "scripts/ue/cc0_verify_one.py"
CC0_CATALOG = Path.home() / "data/ue-projects/SurvSim/Saved/survsim_characters_cc0.json"


def cc0_all(max_n: int, wait: float, redo: bool = False) -> int:
    """Production default (docs/47 option A): region-paint each rebaked CC0 body
    (survsim/export/cc0_regions.py) and import + verify it, ONE per job."""
    from survsim.export.cc0_regions import check_regions, paint_file
    done = json.loads(CC0_CATALOG.read_text()) if CC0_CATALOG.exists() else {}
    n = 0
    for d in json.loads(INDEX.read_text())["assets"]:
        if d["kind"] not in ("character", "human", "person") or not d.get("rebaked_path"):
            continue
        if not d.get("license", "").startswith("CC0"):
            continue
        name = re.sub(r"[^A-Za-z0-9_]", "_", d["name"])
        if name in done and not redo:
            continue
        if n >= max_n:
            break
        src = INDEX.parent / d["rebaked_path"]
        dst = CC0_DIR / src.name
        frac = paint_file(src, dst)
        bad = check_regions(frac)
        job = Path("/tmp/survsim_cc0_one.py")
        job.write_text(f"ONLY = {name!r}\nSRC = {str(dst)!r}\n" + CC0_JOB.read_text())
        from tools.ue_job import vram_mb
        if (vram_mb() or 0) > 5000:
            print(f"[X] VRAM {vram_mb()} MB > 5000 before {name}: stop (restart the editor)")
            return 1
        res = submit(job, name=f"cc0_{name}", wait=wait)
        print(name, "regions", "ok" if not bad else bad,
              json.dumps({k: v for k, v in res.items() if k not in ("seconds",)})[:400], flush=True)
        n += 1
        if "editor process is gone" in str(res.get("error", "")):
            print(f"[X] {name} crashed the editor")
            return 1
    return 0


def rebake_all() -> dict:
    """Re-bake every Quaternius glb into REBAKE_DIR and run the pure-Python checks."""
    from survsim.export.gltf_normalize import check_character, rebake_file
    REBAKE_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for d in json.loads(INDEX.read_text())["assets"]:
        if d["kind"] not in ("character", "human", "person"):
            continue
        src = INDEX.parent / d["path"]
        dst = REBAKE_DIR / src.name
        rep = rebake_file(src, dst)
        metrics, fails = check_character(dst)
        out[d["name"]] = {"report": rep, "metrics": metrics, "fails": fails}
        print(d["name"], "OK" if not fails else fails, flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=100)
    ap.add_argument("--wait", type=float, default=170)
    ap.add_argument("--remeasure", action="store_true",
                    help="re-run already-imported ones (cheap: import is cached)")
    ap.add_argument("--rebake", action="store_true", help="re-bake + check only (no editor)")
    ap.add_argument("--cc0", action="store_true",
                    help="paint regions + import + verify the rebaked CC0 bodies (default stack)")
    a = ap.parse_args()
    if a.cc0:
        return cc0_all(a.max, a.wait, a.remeasure)
    if a.rebake:
        res = rebake_all()
        return 0 if any(not v["fails"] for v in res.values()) else 1
    data = json.loads(INDEX.read_text())
    # Re-baked copies (survsim/export/gltf_normalize.rebake) import upright;
    # the older armature-only bake (characters_norm, "_m") does not.
    norm = REBAKE_DIR
    names = [re.sub(r"[^A-Za-z0-9_]", "_", d["name"])
             + ("_r" if (norm / Path(d["path"]).name).is_file() else "")
             for d in data["assets"] if d["kind"] in ("character", "human", "person")]
    done = json.loads(CATALOG.read_text()) if CATALOG.exists() else {}
    job = Path("/tmp/survsim_import_one.py")
    n = 0
    for name in names:
        if name in done and "error" not in done[name] and not a.remeasure:
            continue
        if n >= a.max:
            break
        src = SCRIPT.read_text().replace("from __future__ import annotations\n", "")
        src = src.replace('"~/data/assets/characters_norm"', f'"{REBAKE_DIR}"').replace(
            '+ "_m"', '+ "_r"')
        job.write_text(f"ONLY = {name!r}\n" + src)
        res = submit(job, name=f"import_{name}", wait=a.wait)
        print(name, json.dumps({k: v for k, v in res.items() if k != "seconds"})[:300], flush=True)
        n += 1
        if "editor process is gone" in str(res.get("error", "")):
            print(f"[X] {name} crashed the editor; restart it and rerun (it will be skipped "
                  "only once it is in the catalogue — remove it from the run by hand)")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

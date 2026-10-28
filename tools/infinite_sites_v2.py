#!/usr/bin/env python3
"""Demo segment 19b "infinite sites v2": 16 seeded sites of the 8 new site types
(survsim/generator/layout_sites_v2.py), 2 per type, mixed time of day, rendered
through the persistent editor with the tools/infinite_sites.py pipeline.

    .venv/bin/python tools/infinite_sites_v2.py auto <i> [<i> ...]   # prep + render + gate
    .venv/bin/python tools/infinite_sites_v2.py reseed <i>           # re-seed once after a failed gate
    .venv/bin/python tools/infinite_sites_v2.py video                # 19b mp4 + gallery png

Sites go to out/showcase/sites_v2/<nn>_<type>_s<seed>/.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import infinite_sites as IS

from survsim.generator.batch import ALL_SITE_TYPES, SITE_TYPES_V2

OUT = ROOT / "out" / "showcase" / "sites_v2"
IS.OUT = OUT
N = 16
# (hour, render_run --lighting, label): the pair of each type gets two different times
TOD = [("10:30", "day", "day"), ("19:15", "overcast", "dusk"), ("14:00", "day", "day"),
       ("22:30", "night", "night")]
_RS = OUT / "reseed.json"


def _reseeds() -> dict[int, int]:
    return {int(k): v for k, v in json.loads(_RS.read_text()).items()} if _RS.is_file() else {}


def site(i: int) -> dict:
    fam = SITE_TYPES_V2[i // 2]
    seed = _reseeds().get(i, 2000 + 53 * i)
    hour, light, tod = TOD[(i + i // 2) % 4]
    n_people = 3 + (seed * 7) % 6          # 3..8
    return {"i": i, "type": fam, "seed": seed, "hour": hour, "lighting": light, "tod": tod,
            "people": n_people, "dir": OUT / f"{i:02d}_{fam}_s{seed}"}


IS.site = site
IS.VARIANT_KEYS = ("dock", "school", "platform", "hospital", "utility", "stage", "farm", "street")


def reseed(i: int) -> int:
    rs = _reseeds()
    rs[i] = 2000 + 53 * i + 7919
    OUT.mkdir(parents=True, exist_ok=True)
    _RS.write_text(json.dumps(rs))
    return IS.auto(i)


def video() -> int:
    from PIL import Image, ImageDraw, ImageFont
    sites = IS._ok_sites(N)
    n_pass = len(sites)
    seen: set = set()
    if len(sys.argv) > 2 and sys.argv[2] == "--one-per-type":
        sites = [s for s in sites if not (s["type"] in seen or seen.add(s["type"]))]
    n = len(sites)
    cols = 4 if n > 9 else 3                 # 4x4 for up to 16, 3x3 for the cut-down set
    tw, th = 640 // cols, 360 // cols
    tried = len([d for d in OUT.glob("[0-9][0-9]_*") if (d / "site.json").is_file()])
    F, FB = IS.FONT, IS.FONT_B
    cache: dict = {}

    def tile(k, f):
        key = (k, f % 30)
        if key not in cache:
            im = Image.open(sites[k]["frames"][f % 30]).convert("RGB").resize((tw, th), Image.LANCZOS)
            cache[key] = IS._label(im, sites[k], False)
        return cache[key]

    frames = []
    fill_per = 5
    total_fill = n * fill_per + 20
    hdr = ImageFont.truetype(FB, 13)
    for f in range(total_fill):
        can = Image.new("RGB", (640, 360), (12, 12, 14))
        shown = min(n, f // fill_per + 1)
        for k in range(shown):
            can.paste(tile(k, f), ((k % cols) * tw, (k // cols) * th))
        dr = ImageDraw.Draw(can, "RGBA")
        dr.rectangle([0, 0, 430, 18], fill=(0, 0, 0, 180))
        dr.text((5, 2), f"{shown} sites, {len({x['type'] for x in sites})} of 8 NEW types, Unreal render (SIMULATED, CC0)",
                font=hdr, fill=(255, 255, 255))
        frames.append(can)
    for k, s in enumerate(sites):                   # fly-through, 0.8 s per site
        for f in range(8):
            im = Image.open(s["frames"][(f * 3) % 30]).convert("RGB").resize((640, 360))
            frames.append(IS._label(im, dict(s, i=k, of=n), True))
    card = Image.new("RGB", (640, 360), (10, 10, 12))
    card.paste(frames[total_fill - 1].resize((320, 180)), (160, 24))
    dr = ImageDraw.Draw(card)
    dr.text((320, 222), f"{len(ALL_SITE_TYPES)} site types in the generator, 13 with a gate-passing render",
            font=ImageFont.truetype(FB, 15), fill=(255, 255, 255), anchor="mm")
    dr.text((320, 244), "× ∞ seeds (SIMULATED, CC0)", font=ImageFont.truetype(FB, 15), fill=(255, 255, 255),
            anchor="mm")
    dr.text((320, 265), f"{n_pass} of {tried} seeded v2 sites passed the render gate; {n} shown",
            font=ImageFont.truetype(F, 14), fill=(220, 220, 220), anchor="mm")
    dr.text((320, 290), "new: " + ", ".join(t.replace("_", " ") for t in SITE_TYPES_V2[:4]),
            font=ImageFont.truetype(F, 11), fill=(160, 160, 160), anchor="mm")
    dr.text((320, 306), ", ".join(t.replace("_", " ") for t in SITE_TYPES_V2[4:]),
            font=ImageFont.truetype(F, 11), fill=(160, 160, 160), anchor="mm")
    miss = [t.replace("_", " ") for t in SITE_TYPES_V2 if t not in {x["type"] for x in sites}]
    if miss:
        dr.text((320, 326), "no gate-passing render yet: " + ", ".join(miss),
                font=ImageFont.truetype(F, 11), fill=(200, 140, 90), anchor="mm")
    frames += [card] * 30
    tmp = OUT / "_frames"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    for j, im in enumerate(frames):
        im.save(tmp / f"{j:05d}.png")
    dst = ROOT / "docs/research/demo/segments/19b_infinite_sites_v2.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "10", "-i",
                    str(tmp / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "25",
                    "-preset", "slow", str(dst)], check=True)
    shutil.rmtree(tmp)
    print(f"{dst}: {len(frames) / 10:.1f} s, {dst.stat().st_size / 1e6:.2f} MB, {n} sites")
    cw, ch = 320, 180
    sheet = Image.new("RGB", (cw * 4, ch * -(-n // 4)), (12, 12, 14))
    for k, s in enumerate(sites):
        im = Image.open(s["frames"][15]).convert("RGB").resize((cw, ch), Image.LANCZOS)
        sheet.paste(IS._label(im, dict(s, i=k, of=n), True), ((k % 4) * cw, (k // 4) * ch))
    g = ROOT / "docs/research/demo/gallery/infinite_sites_v2.png"
    sheet.save(g, optimize=True)
    print(g)
    return 0


if __name__ == "__main__":
    op = sys.argv[1]
    if op == "auto":
        rc = 0
        for a in sys.argv[2:]:
            r = IS.auto(int(a))
            print("SITE", a, "rc", r, flush=True)
            rc |= r
        raise SystemExit(rc)
    if op == "reseed":
        raise SystemExit(reseed(int(sys.argv[2])))
    if op == "video":
        raise SystemExit(video())

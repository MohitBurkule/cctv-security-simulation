#!/usr/bin/env python3
"""Fetch a CC0 asset library into ~/data/assets and build index.json.

Idempotent: files already on disk are not re-downloaded; index.json is always
rebuilt from what is on disk, with every glTF re-validated.

Sources (all CC0, no login):
  * poly.pizza  - Quaternius characters/props (licence read from each model page)
  * kenney.nl   - Car Kit, City Kit Roads (License.txt inside each zip)
  * ambientCG   - PBR materials, 1K-JPG
  * Poly Haven  - HDRIs, 2k .hdr

    python scripts/assets/fetch_cc0.py [--root ~/data/assets] [--index-only]
"""
from __future__ import annotations

import argparse
import http.client
import json
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

CC0_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
KINDS = {"character", "prop", "vehicle", "texture", "hdri"}
UA = {"User-Agent": "survsim-asset-fetch/1.0"}

# poly.pizza model ids by Quaternius, chosen 2026-09-25. Only models whose page
# says CC0 and whose author is Quaternius are kept; characters additionally need
# a skin and a clip whose name contains "walk".
PP_CHARACTERS = [
    "HMnuH5geEG", "fjHyMd5Wxw", "jpKRgGDxhk", "XqzeZGB7iU", "zMyPlQXBzq",
    "AQsd9ngvKU", "c3Ibh9I3udk", "9kF7eTDbhO", "5EGWBMpuXq", "BTALZymknF",
    "DojKLcO34E", "Yg2bQZO6Hj", "ZwF0K7WBmu", "kZ3DmIoGip", "nIItLV9nxS",
    "qJ2gsTUBHL", "DgOCW9ZCRJ", "66kQ4dBBC7", "UcLErL2W37", "gjuwleUT1U",
    "75ikp7NEDx", "K1IczhnvQ5",
]
PP_PROPS = ["7uSlZo3n9Y", "53AXNH8J9N", "eitNk4I4R1", "cM3aJPU9NS", "0lxF8Dl1jU", "apWcPbwhlq",
            "Fg3Nimfcfg", "aDIrUbMbW3", "df8uCl1YpK"]

KENNEY = {
    # pack page -> [(kind, glb stem)]
    "car-kit": [("vehicle", s) for s in
                ("sedan", "suv", "taxi", "police", "hatchback-sports", "delivery", "van")],
    "city-kit-roads": [("prop", s) for s in
                       ("construction-barrier", "construction-cone", "construction-fence",
                        "dumpster", "light-square", "light-curved", "traffic-light")],
}
AMBIENTCG = ["Asphalt033", "Road007", "Concrete034", "Concrete048", "Bricks105",
             "PavingStones151", "Metal063", "Tiles141"]
POLYHAVEN = ["kloofendal_overcast", "overcast_soil_puresky", "bethnal_green_entrance",
             "abandoned_parking"]


# ----------------------------------------------------------------- validation
def validate_index(index: dict) -> list[str]:
    """Return a list of schema problems (empty == valid). No file access."""
    errs: list[str] = []
    assets = index.get("assets")
    if not isinstance(assets, list):
        return ["index.assets must be a list"]
    names = set()
    for i, a in enumerate(assets):
        p = f"assets[{i}]"
        for k in ("kind", "name", "path", "format", "license", "license_url", "source_url"):
            if not isinstance(a.get(k), str) or not a.get(k):
                errs.append(f"{p}.{k} missing/empty")
        kind = a.get("kind")
        if kind not in KINDS:
            errs.append(f"{p}.kind {kind!r} not in {sorted(KINDS)}")
        if (kind, a.get("name")) in names:
            errs.append(f"{p} duplicate name {a.get('name')!r}")
        names.add((kind, a.get("name")))
        if a.get("license") != "CC0-1.0":
            errs.append(f"{p}.license must be CC0-1.0")
        if not isinstance(a.get("anim_clips", []), list):
            errs.append(f"{p}.anim_clips must be a list")
        if kind == "character":
            h = a.get("height_m")
            if not isinstance(h, (int, float)) or not 0.1 < h < 10:
                errs.append(f"{p}.height_m invalid: {h!r}")
            if not any("walk" in c.lower() for c in a.get("anim_clips", [])):
                errs.append(f"{p} character without a walk clip")
        elif kind in ("prop", "vehicle"):
            s = a.get("size_m")
            if not (isinstance(s, list) and len(s) == 3 and all(isinstance(v, (int, float)) and v >= 0 for v in s)):
                errs.append(f"{p}.size_m must be [x,y,z] >= 0")
        elif kind in ("texture", "hdri"):
            r = a.get("resolution")
            if not (isinstance(r, list) and len(r) == 2 and all(isinstance(v, int) and v > 0 for v in r)):
                errs.append(f"{p}.resolution must be [w,h] ints")
    return errs


# ------------------------------------------------------------------- helpers
def get(url: str, tries: int = 4) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read()
        except (OSError, http.client.HTTPException) as e:  # truncated/reset transfers
            if k == tries - 1:
                raise
            print(f"  retry {k + 1} {url}: {e}")
            time.sleep(2 * (k + 1))
    raise RuntimeError(url)


def download(url: str, dest: Path, tries: int = 8) -> Path:
    """Stream to <dest>.part, resuming with HTTP Range after truncated transfers
    (ambientCG reliably cuts connections at ~4 MB)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    total = None
    for k in range(tries):
        have = tmp.stat().st_size if tmp.exists() else 0
        if total is not None and have >= total:
            break
        hdrs = dict(UA, **({"Range": f"bytes={have}-"} if have else {}))
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdrs), timeout=120) as r:
                if have and r.status != 206:  # server ignored Range: start over
                    have = 0
                    tmp.unlink()
                cl = r.headers.get("Content-Length")
                total = have + int(cl) if cl else None
                with open(tmp, "ab") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
            if total is None or tmp.stat().st_size >= total:
                break
        except (OSError, http.client.HTTPException) as e:
            print(f"  resume {k + 1} {url}: {e}")
            time.sleep(1 + k)
    else:
        raise RuntimeError(f"download failed: {url}")
    tmp.rename(dest)
    print("  got", dest)
    return dest


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_DTYPE = {5120: "i1", 5121: "u1", 5122: "i2", 5123: "u2", 5125: "u4", 5126: "f4"}


def _accessor(g, blob: bytes, idx: int):
    import numpy as np
    a = g.accessors[idx]
    bv = g.bufferViews[a.bufferView]
    n = _NCOMP[a.type]
    dt = np.dtype(_DTYPE[a.componentType])
    start = (bv.byteOffset or 0) + (a.byteOffset or 0)
    stride = bv.byteStride or n * dt.itemsize
    raw = np.frombuffer(blob, dtype=np.uint8, count=stride * (a.count - 1) + n * dt.itemsize,
                        offset=start)
    rows = np.lib.stride_tricks.as_strided(raw, (a.count, n * dt.itemsize), (stride, 1))
    out = np.ascontiguousarray(rows).view(dt).reshape(a.count, n).astype(np.float64)
    if a.normalized and dt.kind in "iu":
        out /= np.iinfo(dt).max
    return out


def _node_local(nd):
    import numpy as np
    if nd.matrix:
        return np.array(nd.matrix, dtype=float).reshape(4, 4).T
    t = np.array(nd.translation or [0, 0, 0], float)
    x, y, z, w = nd.rotation or [0, 0, 0, 1]
    sc = np.array(nd.scale or [1, 1, 1], float)
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    m = np.eye(4)
    m[:3, :3] = r * sc
    m[:3, 3] = t
    return m


def rest_pose_extents(g) -> list[float]:
    """World-space AABB extents of the default scene in its rest (node) pose.

    Skinned meshes are linear-blend skinned with the rest joint transforms, as the
    glTF spec requires (the skinned node's own transform is ignored). Naive loaders
    that skip skinning get Quaternius characters 2.5x too tall.
    """
    import numpy as np
    blob = g.binary_blob() or b""
    world = {}

    def walk(i, parent):
        world[i] = parent @ _node_local(g.nodes[i])
        for c in g.nodes[i].children or []:
            walk(c, world[i])

    for r in g.scenes[g.scene or 0].nodes:
        walk(r, np.eye(4))
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for i, nd in world.items():
        node = g.nodes[i]
        if node.mesh is None:
            continue
        skin = g.skins[node.skin] if node.skin is not None else None
        if skin is not None:
            ibm = (_accessor(g, blob, skin.inverseBindMatrices).reshape(-1, 4, 4).transpose(0, 2, 1)
                   if skin.inverseBindMatrices is not None else np.tile(np.eye(4), (len(skin.joints), 1, 1)))
            jm = np.stack([world[j] @ ibm[k] for k, j in enumerate(skin.joints)])
        for prim in g.meshes[node.mesh].primitives:
            v = _accessor(g, blob, prim.attributes.POSITION)
            vh = np.c_[v, np.ones(len(v))]
            if skin is not None and prim.attributes.JOINTS_0 is not None:
                J = _accessor(g, blob, prim.attributes.JOINTS_0).astype(int)
                W = _accessor(g, blob, prim.attributes.WEIGHTS_0)
                W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
                M = np.einsum("vk,vkij->vij", W, jm[J])
                p = np.einsum("vij,vj->vi", M, vh)[:, :3]
            else:
                p = (vh @ nd.T)[:, :3]
            lo, hi = np.minimum(lo, p.min(0)), np.maximum(hi, p.max(0))
    return [round(float(x), 4) for x in (hi - lo)]


def gltf_info(path: Path) -> dict:
    """Open with pygltflib (and trimesh as a second loader); return skins, clips and
    rest-pose extents in metres (glTF is Y-up, so extents[1] is height)."""
    import pygltflib
    import trimesh
    g = pygltflib.GLTF2().load(str(path))
    if g.buffers and g.buffers[0].uri is None and not g.binary_blob():
        raise ValueError("empty binary buffer")
    trimesh.load(str(path), force="scene")  # must also open in trimesh
    clips = [a.name or f"clip{i}" for i, a in enumerate(g.animations)]
    return {"skins": len(g.skins), "clips": clips, "extents": rest_pose_extents(g)}


# ------------------------------------------------------------------- sources
def polypizza(root: Path, ids: list[str], kind: str, meta: dict) -> None:
    for mid in ids:
        page_url = f"https://poly.pizza/m/{mid}"
        cache = meta.setdefault(mid, {})
        if not cache.get("glb"):
            html = get(page_url).decode("utf-8", "replace")
            title = re.search(r'og:title" content="([^"]*)', html)
            author = re.search(r'/u/([A-Za-z0-9_]+)', html)
            glb = re.search(r'https://static\.poly\.pizza/[^"]*\.glb', html)
            cache.update(title=title.group(1).split(" - Free Model")[0].strip() if title else mid,
                         author=author.group(1) if author else "",
                         cc0="CC0" in html, glb=glb.group(0) if glb else "")
        if not (cache["cc0"] and cache["author"] == "Quaternius" and cache["glb"]):
            print("  skip (licence/author/url)", mid, cache)
            continue
        dest = root / f"{kind}s" / "quaternius" / f"{slug(cache['title'])}_{mid}.glb"
        download(cache["glb"], dest)
        cache.update(kind=kind, path=str(dest.relative_to(root)), source_url=page_url)


def kenney(root: Path, meta: dict) -> None:
    for pack, items in KENNEY.items():
        page = f"https://kenney.nl/assets/{pack}"
        pdir = root / "kenney" / pack
        if not (pdir / "License.txt").exists():
            m = re.search(r'https://kenney\.nl/media/pages/assets/[^"]*\.zip',
                          get(page).decode("utf-8", "replace"))
            if not m:
                print("  no zip link on", page)
                continue
            z = download(m.group(0), root / "_downloads" / Path(m.group(0)).name)
            with zipfile.ZipFile(z) as zf:
                for n in zf.namelist():
                    if n == "License.txt" or n.startswith("Models/GLB format/"):
                        zf.extract(n, pdir)
        lic = (pdir / "License.txt").read_text(errors="replace")
        assert "CC0" in lic, f"{pack}: License.txt does not say CC0"
        for kind, stem in items:
            f = pdir / "Models" / "GLB format" / f"{stem}.glb"
            if f.exists():
                meta[f"kenney:{pack}:{stem}"] = {
                    "kind": kind, "title": f"kenney_{stem}", "path": str(f.relative_to(root)),
                    "source_url": page, "author": "Kenney"}


def ambientcg(root: Path, out: list) -> None:
    from PIL import Image
    for aid in AMBIENTCG:
        d = root / "textures" / "ambientcg" / aid
        if not d.exists() or not any(d.iterdir()):
            z = download(f"https://ambientcg.com/get?file={aid}_1K-JPG.zip",
                         root / "_downloads" / f"{aid}_1K-JPG.zip")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(d)
        maps = sorted(p.name for p in d.iterdir() if p.suffix.lower() in (".jpg", ".png"))
        color = next((m for m in maps if "Color" in m), maps[0])
        w, h = Image.open(d / color).size
        out.append({"kind": "texture", "name": aid, "path": str(d.relative_to(root)), "format": "jpg",
                        "license": "CC0-1.0", "license_url": "https://docs.ambientcg.com/license/",
                        "source_url": f"https://ambientcg.com/view?id={aid}", "anim_clips": [],
                        "resolution": [w, h], "maps": maps})


def polyhaven(root: Path, out: list) -> None:
    for hid in POLYHAVEN:
        dest = root / "hdri" / "polyhaven" / f"{hid}_2k.hdr"
        if not dest.exists():
            files = json.loads(get(f"https://api.polyhaven.com/files/{hid}"))
            download(files["hdri"]["2k"]["hdr"]["url"], dest)
        with open(dest, "rb") as f:  # Radiance header: "-Y h +X w"
            head = f.read(4096).decode("latin-1")
        m = re.search(r"-Y (\d+) \+X (\d+)", head)
        out.append({"kind": "hdri", "name": hid, "path": str(dest.relative_to(root)), "format": "hdr",
                        "license": "CC0-1.0", "license_url": "https://polyhaven.com/license",
                        "source_url": f"https://polyhaven.com/a/{hid}", "anim_clips": [],
                        "resolution": [int(m.group(2)), int(m.group(1))] if m else [0, 0]})


def build_models(root: Path, meta: dict, out: list) -> None:
    for key, m in meta.items():
        if "path" not in m:
            continue
        try:
            info = gltf_info(root / m["path"])
        except Exception as e:  # noqa: BLE001 - report and drop broken files
            print("  INVALID glTF", m["path"], e)
            continue
        kind = m["kind"]
        entry = {"kind": kind, "name": slug(m["title"]), "path": m["path"], "format": "glb",
                     "license": "CC0-1.0", "license_url": CC0_URL, "source_url": m["source_url"],
                     "author": m["author"], "anim_clips": info["clips"]}
        if kind == "character":
            if not info["skins"] or not any("walk" in c.lower() for c in info["clips"]):
                print("  drop character (no skin/walk)", m["path"])
                continue
            entry["name"] = f"{slug(m['title'])}_{key[:4].lower()}"
            entry["height_m"] = info["extents"][1]  # glTF is Y-up
            entry["skins"] = info["skins"]
            entry["scale_to_1_75m"] = round(1.75 / entry["height_m"], 5)
        else:
            entry["size_m"] = info["extents"]
        out.append(entry)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.home() / "data" / "assets")
    args = ap.parse_args()
    root = args.root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    meta_path = root / "_sources.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    print("poly.pizza characters"); polypizza(root, PP_CHARACTERS, "character", meta)
    print("poly.pizza props"); polypizza(root, PP_PROPS, "prop", meta)
    print("kenney"); kenney(root, meta)
    meta_path.write_text(json.dumps(meta, indent=1))

    assets: list = []
    build_models(root, meta, assets)
    print("ambientCG"); ambientcg(root, assets)
    print("polyhaven"); polyhaven(root, assets)

    index = {"version": 1, "root": str(root), "assets": assets}
    errs = validate_index(index)
    for e in errs:
        print("SCHEMA:", e)
    (root / "index.json").write_text(json.dumps(index, indent=1))
    counts: dict = {}
    for a in assets:
        counts[a["kind"]] = counts.get(a["kind"], 0) + 1
    print("counts", counts)
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())

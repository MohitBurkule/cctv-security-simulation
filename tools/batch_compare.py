"""Compare two render batches: detector recall table + a contact sheet.

    CUDA_VISIBLE_DEVICES= .venv/bin/python tools/batch_compare.py out/batch_v7 out/batch_v6 \
        --sheet out/demo_humans_v2/batch_v7_sheet.png --json out/demo_humans_v2/recall_v7_vs_v6.json

Both batches are scored with the CURRENT tools/eval_recall.py matching (same
frames per camera, conf 0.25, IoU 0.5, GT >= 20 px), so the difference is the
render, not the scorer. Sets present in both batches only. The sheet is frame
150 of every camera of every set of the first batch, labelled; each set's
licences.json verdict is printed on its row.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _sum(r: dict) -> tuple[int, int, int]:
    tp = sum(v.get("tp", 0) for v in r.values())
    gt = sum(v.get("gt", 0) for v in r.values())
    fp = sum(v.get("fp", 0) for v in r.values())
    return tp, gt, fp


def sheet(batch: Path, out: Path, frame: int = 150, tile_w: int = 320) -> None:
    from PIL import Image, ImageDraw
    sets = sorted(p for p in batch.iterdir() if (p / "render").is_dir())
    rows = []
    for s in sets:
        cams = sorted(c for c in (s / "render").iterdir() if c.is_dir())
        tiles = []
        for c in cams:
            pngs = sorted(c.glob("*.png"))
            if not pngs:
                continue
            im = Image.open(pngs[min(frame, len(pngs) - 1)]).convert("RGB")
            im = im.resize((tile_w, round(im.height * tile_w / im.width)))
            ImageDraw.Draw(im).text((4, 4), f"{s.name} {c.name}", fill=(255, 255, 0))
            tiles.append(im)
        lic = s / "licences.json"
        verdict = json.loads(lic.read_text()).get("commercial_ml_ok") if lic.is_file() else None
        rows.append((s.name, tiles, verdict))
    ncol = max(len(t) for _, t, _ in rows)
    th = max(t.height for _, ts, _ in rows for t in ts)
    W, H = ncol * tile_w, len(rows) * (th + 14)
    img = Image.new("RGB", (W, H), (20, 20, 20))
    d = ImageDraw.Draw(img)
    for r, (name, tiles, verdict) in enumerate(rows):
        y = r * (th + 14)
        d.text((4, y), f"{name}  commercial_ml_ok={verdict}", fill=(200, 200, 200))
        for c, t in enumerate(tiles):
            img.paste(t, (c * tile_w, y + 14))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("new", type=Path)
    ap.add_argument("old", type=Path)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--sheet", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    from tools.eval_recall import evaluate
    model = ("/tmp/claude-1000/-home-mohit-multi-person-camera-real-sim/"
             "5f6f75e9-f418-4115-9d7d-4e03fa563d1f/scratchpad/yolov8n.pt")
    model = model if Path(model).exists() else "yolov8n.pt"
    res = {}
    names = sorted(p.name for p in a.new.iterdir() if (p / "render").is_dir()
                   and (a.old / p.name / "render").is_dir())
    print("| set | new recall per camera | new R / P | old R / P | GT |")
    print("|---|---|---|---|---|")
    tot = {"new": [0, 0, 0], "old": [0, 0, 0]}
    for nm in names:
        r = {}
        for key, root in (("new", a.new), ("old", a.old)):
            r[key] = evaluate(root / nm / "render", root / nm / "gt", a.n, model)
            for i, v in enumerate(_sum(r[key])):
                tot[key][i] += v
        res[nm] = r
        tpn, gtn, fpn = _sum(r["new"])
        tpo, gto, fpo = _sum(r["old"])
        cams = ", ".join(f"{c} {v.get('recall', '-')}" for c, v in r["new"].items())
        print(f"| {nm} | {cams} | {tpn / max(1, gtn):.3f} / {tpn / max(1, tpn + fpn):.3f} | "
              f"{tpo / max(1, gto):.3f} / {tpo / max(1, tpo + fpo):.3f} | {gtn} |", flush=True)
    for key in ("new", "old"):
        tp, gt, fp = tot[key]
        res[f"_all_{key}"] = {"recall": round(tp / max(1, gt), 3),
                              "precision": round(tp / max(1, tp + fp), 3), "gt": gt}
    print(f"| **all** | | **{res['_all_new']['recall']} / {res['_all_new']['precision']}** | "
          f"**{res['_all_old']['recall']} / {res['_all_old']['precision']}** | {res['_all_new']['gt']} |")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(res, indent=1))
    if a.sheet:
        sheet(a.new, a.sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

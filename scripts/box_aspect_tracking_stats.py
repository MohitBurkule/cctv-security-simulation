"""Box aspect (w/h) and tracking-statistics comparison: MOT20 train GT vs batch_v8.

Resolves the docs/11 "MOT20 boxes are 14 % wider" claim (docs/13 "Box aspect")
and supplies the density/occlusion table of docs/research/73.

    python scripts/box_aspect_tracking_stats.py --out out/experiments/box_aspect
Bootstrap resamples whole tracks (sequence, id): frames of one track are not
independent. CPU only, ~1 min.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MOT20 = Path.home() / "data/mot/extracted/MOT20/train"
V8 = Path("out/batch_v8")
# unique geometry: night sets share GT with day; far_day cam_01 == street_crowd_day cam_01
V8_SEQS = [("disguise_and_tamper_day", None), ("smoke_mock_day", None),
           ("street_crowd_day", None), ("street_crowd_far_day", {"mot_cam_01.txt"})]
BINS = [(20, 50), (50, 100), (100, 200), (200, 10_000)]
FPS = {"mot20": 25.0, "v8": 10.0}


def load(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", ndmin=2)


def png_size(path: Path) -> tuple[int, int]:
    b = path.read_bytes()[16:24]
    return int.from_bytes(b[:4], "big"), int.from_bytes(b[4:], "big")


def interior(a: np.ndarray, w: int, h: int, m: float = 3.0) -> np.ndarray:
    """rows whose box does not touch the image border: MOT20 GT is clipped at
    the border, so edge boxes have a truncated, meaningless aspect"""
    return a[(a[:, 2] > m) & (a[:, 3] > m) & (a[:, 2] + a[:, 4] < w - m) & (a[:, 3] + a[:, 5] < h - m)]


def mot20_rows():
    out = []
    for seq in sorted(MOT20.iterdir()):
        a = load(seq / "gt/gt.txt")
        a = a[(a[:, 6] == 1) & (a[:, 7] == 1)]
        ini = dict(ln.split("=", 1) for ln in (seq / "seqinfo.ini").read_text().split() if "=" in ln)
        out.append((seq.name, a, int(ini["imWidth"]), int(ini["imHeight"])))
    return out


def v8_rows(kind: str):
    out = []
    for s, skip in V8_SEQS:
        d = V8 / s / "gt" / ("amodal" if kind == "amodal" else "")
        for f in sorted(d.glob("mot_cam_*.txt")):
            if skip and f.name in skip:
                continue
            a = load(f)
            cam = f.stem[4:]
            w, h = png_size(next((V8 / s / "render" / cam).glob("*.png")))
            out.append((f"{s}/{cam}", a, w, h))
    return out


def tracks(rows, min_vis, bin_):
    """per-track arrays of aspect ratios within the height bin"""
    tr = []
    for name, a, w, h in rows:
        a = interior(a, w, h)
        m = (a[:, 8] >= min_vis) & (a[:, 5] >= bin_[0]) & (a[:, 5] < bin_[1]) & (a[:, 4] > 0)
        b = a[m]
        for tid in np.unique(b[:, 1]):
            r = b[b[:, 1] == tid]
            tr.append(r[:, 4] / r[:, 5])
    return tr


def boot_ratio(ta, tb, n=2000, seed=0):
    """median aspect of A / median aspect of B, resampling tracks"""
    rng = np.random.default_rng(seed)
    pa, pb = np.concatenate(ta), np.concatenate(tb)
    est = np.median(pa) / np.median(pb)
    bs = []
    for _ in range(n):
        ia = rng.integers(0, len(ta), len(ta))
        ib = rng.integers(0, len(tb), len(tb))
        bs.append(np.median(np.concatenate([ta[i] for i in ia]))
                  / np.median(np.concatenate([tb[i] for i in ib])))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"ratio": round(float(est), 4), "ci": [round(float(lo), 4), round(float(hi), 4)],
                "med_a": round(float(np.median(pa)), 4), "med_b": round(float(np.median(pb)), 4),
                "n_tracks_a": len(ta), "n_tracks_b": len(tb), "n_boxes_a": len(pa), "n_boxes_b": len(pb)}


def tracking_stats(rows, fps, vis_rows=None):
    ppf, lens, gaps, vis, hs = [], [], [], [], []
    for name, a, _w, _h in rows:
        a = a[a[:, 8] > 0] if vis_rows is None else a
        _, c = np.unique(a[:, 0], return_counts=True)
        ppf.extend(c.tolist())
        for tid in np.unique(a[:, 1]):
            f = np.sort(a[a[:, 1] == tid, 0])
            lens.append((f[-1] - f[0] + 1) / fps)
            g = np.diff(f) - 1
            gaps.extend((g[g > 0] / fps).tolist())
        vis.extend(a[:, 8].tolist())
        hs.extend(a[:, 5].tolist())
    ppf, lens, vis, hs = map(np.asarray, (ppf, lens, vis, hs))
    return {"people_per_frame_mean": round(float(ppf.mean()), 1),
                "people_per_frame_p90": float(np.percentile(ppf, 90)),
                "vis_mean": round(float(vis.mean()), 3), "vis_lt_0_5": round(float((vis < 0.5).mean()), 3),
                "track_len_s_median": round(float(np.median(lens)), 1),
                "track_len_s_p90": round(float(np.percentile(lens, 90)), 1), "n_tracks": len(lens),
                "box_h_median_px": round(float(np.median(hs)), 1),
                "gaps_per_track": round(len(gaps) / max(len(lens), 1), 3),
                "_ppf": ppf.tolist(), "_lens": lens.tolist(), "_vis": vis.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/experiments/box_aspect")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m20 = mot20_rows()
    res = {"aspect": {}, "note": "ratio = median(w/h) MOT20 / median(w/h) batch_v8; track bootstrap"}
    for kind in ("mask", "amodal"):
        v8 = v8_rows(kind)
        for lo, hi in BINS + [(50, 10_000), (100, 10_000)]:
            key = f"{kind}_h{lo}-{hi}"
            ta = tracks(m20, 0.9, (lo, hi))
            tb = tracks(v8, 0.9, (lo, hi))
            if len(ta) < 5 or len(tb) < 5:
                res["aspect"][key] = {"skipped": f"tracks {len(ta)}/{len(tb)}"}
                continue
            res["aspect"][key] = boot_ratio(ta, tb, args.n_boot)
            print(key, res["aspect"][key])
    st = {"MOT20 train": tracking_stats(m20, FPS["mot20"]),
          "batch_v8 (amodal)": tracking_stats(v8_rows("amodal"), FPS["v8"])}
    res["tracking"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in st.items()}
    for k, v in res["tracking"].items():
        print(k, v)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(out / "dists.npz", **{f"{k.split()[0]}{kk}": np.asarray(vv)
                                              for k, v in st.items() for kk, vv in v.items()
                                              if kk.startswith("_")})
    # per-bin aspect distributions for the figure
    dd = {}
    for lbl, rows in (("MOT20", m20), ("v8mask", v8_rows("mask")), ("v8amodal", v8_rows("amodal"))):
        for lo, hi in BINS:
            t = tracks(rows, 0.9, (lo, hi))
            dd[f"{lbl}_{lo}"] = np.concatenate(t) if t else np.zeros(0)
    np.savez_compressed(out / "aspect.npz", **dd)


if __name__ == "__main__":
    main()

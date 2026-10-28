"""Per-image statistics that characterise CCTV footage.

Content-light, signal-level numbers: they say whether a frame has the *imaging
chain* of a real camera (noise, softness, compression, exposure, colour cast),
not whether its *content* is real. docs/15_REALISM_STATS.md has the measured
real distribution and what these numbers cannot see.

Resolution matters for every spatial statistic (downscaling averages noise away
and sharpens edges), so all of them are computed after resizing to a common
analysis height, ``ANALYSIS_HEIGHT``. Blockiness is the exception: the 8x8 codec
grid only exists at native resolution, so it is measured before resizing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

ANALYSIS_HEIGHT = 720
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class FrameStats:
    noise_sigma: float    # robust (MAD) sigma of the high-pass luma residual, 0-255 units
    lap_var_log10: float  # log10 Laplacian variance: sharpness proxy
    mtf_ratio: float      # spectral energy 0.25-0.5 cyc/px over 0.05-0.25 cyc/px: MTF proxy
    edge_width: float     # mean 10-90 % rise width (px) across strong edges
    luma_mean: float
    rms_contrast: float   # luma std / luma mean
    dynamic_range: float  # luma p99 - p1
    luma_skew: float      # histogram shape
    hist_entropy: float   # bits, 256-bin luma histogram
    chroma_mean: float    # mean per-pixel Lab chroma (colourfulness)
    cast_a: float         # mean Lab a* (green<0<magenta), global colour cast
    cast_b: float         # mean Lab b* (blue<0<yellow)
    blockiness: float     # 8x8 boundary gradient / interior gradient, native res

    @staticmethod
    def keys() -> list[str]:
        return list(FrameStats.__annotations__)

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _to_bgr(img: np.ndarray) -> np.ndarray:
    x = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return np.clip(x[:, :, :3], 0, 255).astype(np.uint8)


def _resize(x: np.ndarray, height: int = ANALYSIS_HEIGHT) -> np.ndarray:
    h, w = x.shape[:2]
    if h == height:
        return x
    interp = cv2.INTER_AREA if h > height else cv2.INTER_LINEAR
    return cv2.resize(x, (round(w * height / h), height), interpolation=interp)


def blockiness(grey: np.ndarray) -> float:
    """Mean |horizontal+vertical gradient| on 8-px boundaries over the rest.

    1.0 means no grid; JPEG/H.264 at CCTV bitrates gives > 1. Measured on native
    resolution luma because any resize smears the grid away.
    """
    g = grey.astype(np.float32)
    dx = np.abs(np.diff(g, axis=1))
    dy = np.abs(np.diff(g, axis=0))
    bx = (np.arange(dx.shape[1]) % 8) == 7
    by = (np.arange(dy.shape[0]) % 8) == 7
    num = dx[:, bx].mean() + dy[by, :].mean()
    den = dx[:, ~bx].mean() + dy[~by, :].mean() + 1e-6
    return float(num / den)


def noise_sigma(grey: np.ndarray) -> float:
    """Robust noise estimate: MAD of a Laplacian-like residual (Immerkaer 1996)."""
    g = grey.astype(np.float32)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = np.abs(cv2.filter2D(g, -1, k)[1:-1, 1:-1])
    # Only flat regions: edges and texture would otherwise read as noise. A
    # median on 8-bit data quantises to steps of 0.25, so use Immerkaer's mean
    # form, sigma = sqrt(pi/2) * mean|r| / 6, over the flattest 50 %.
    grad = cv2.GaussianBlur(np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1)),
                            (0, 0), 2)[1:-1, 1:-1]
    flat = grad <= np.percentile(grad, 50)
    return float(np.sqrt(np.pi / 2) * r[flat].mean() / 6.0)


def _band_ratio(grey: np.ndarray) -> float:
    g = grey.astype(np.float32)
    g = g - g.mean()
    h, w = g.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    p = np.abs(np.fft.rfft2(g * win)) ** 2
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    f = np.sqrt(fy**2 + fx**2)
    hi = p[(f >= 0.25) & (f < 0.5)].sum()
    lo = p[(f >= 0.05) & (f < 0.25)].sum() + 1e-9
    return float(hi / lo)


def edge_width(grey: np.ndarray, max_edges: int = 400) -> float:
    """Mean 10-90 % width of strong horizontal-gradient edges (px)."""
    g = cv2.GaussianBlur(grey.astype(np.float32), (0, 0), 0.5)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    mag = np.abs(gx)
    thr = np.percentile(mag, 99.5)
    ys, xs = np.nonzero((mag >= thr) & (mag == cv2.dilate(mag, np.ones((1, 7), np.uint8))))
    keep = (xs > 8) & (xs < g.shape[1] - 9)
    ys, xs = ys[keep], xs[keep]
    if len(xs) == 0:
        return float("nan")
    idx = np.linspace(0, len(xs) - 1, min(max_edges, len(xs))).astype(int)
    widths = []
    for y, x in zip(ys[idx], xs[idx]):
        prof = g[y, x - 8:x + 9]
        lo, hi = prof.min(), prof.max()
        if hi - lo < 20:
            continue
        p = (prof - lo) / (hi - lo)
        if prof[-1] < prof[0]:
            p = 1 - p
        above10 = np.argmax(p >= 0.1)
        above90 = np.argmax(p >= 0.9)
        if above90 > above10:
            widths.append(above90 - above10)
    return float(np.mean(widths)) if widths else float("nan")


def frame_stats(img: np.ndarray) -> FrameStats:
    x = _to_bgr(img)
    native_grey = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    blk = blockiness(native_grey)
    x = _resize(x)
    grey = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0
    mean = float(grey.mean())
    std = float(grey.std())
    hist = np.bincount(grey.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    hist /= hist.sum()
    nz = hist[hist > 0]
    p1, p99 = np.percentile(grey, [1, 99])
    return FrameStats(
        noise_sigma=noise_sigma(grey),
        lap_var_log10=float(np.log10(cv2.Laplacian(grey, cv2.CV_32F).var() + 1e-6)),
        mtf_ratio=_band_ratio(grey),
        edge_width=edge_width(grey),
        luma_mean=mean,
        rms_contrast=std / (mean + 1e-6),
        dynamic_range=float(p99 - p1),
        luma_skew=float(((grey - mean) ** 3).mean() / (std**3 + 1e-9)),
        hist_entropy=float(-(nz * np.log2(nz)).sum()),
        chroma_mean=float(np.sqrt(a**2 + b**2).mean()),
        cast_a=float(a.mean()),
        cast_b=float(b.mean()),
        blockiness=blk,
    )


def temporal_noise(frames: Sequence[np.ndarray]) -> float:
    """Sensor temporal noise from consecutive frames (0-255 units).

    Sigma of the frame difference, /sqrt(2) because two independent noise
    draws are subtracted. Pixels where something moved are excluded by keeping
    only the smallest 90 % of |diff| and correcting for that truncation of a
    half-normal (E|d| below its p90 = 0.6574 sigma). A median would be robust
    too but quantises to whole grey levels on 8-bit data. A clean render, or
    noise frozen by a fixed seed, reads ~0.
    """
    vals = []
    prev = None
    for f in frames:
        g = cv2.cvtColor(_resize(_to_bgr(f)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None and prev.shape == g.shape:
            d = np.abs(g - prev)[8:-8, 8:-8].ravel()
            d = d[d <= np.percentile(d, 90)]
            vals.append(float(d.mean()) / 0.6574 / np.sqrt(2))
        prev = g
    return float(np.median(vals)) if vals else float("nan")


def list_images(d: Path) -> list[Path]:
    return sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def band(values: Iterable[float]) -> dict[str, float]:
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if v.size == 0:
        return {"median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90))}

"""Measure how close a render is to real CCTV.

Without a metric, "make it more realistic" is an opinion. This module gives three
complementary numbers, because any single one is gameable:

1. **FID** — distance between Inception feature distributions. The standard
   synthetic-data metric, but it needs a reasonable sample count and is sensitive
   to content as well as style, so a train station vs a car park scores badly even
   if both look equally real.
2. **Signal statistics** — noise power, edge density, blockiness, colour spread.
   These are content-light and catch exactly what a clean render lacks: a render
   has no sensor noise and no compression blocking, and that shows up here even
   with one image.
3. **Per-statistic gap table** — so the number tells you *what* to fix rather than
   just that something is wrong.

A low FID with badly mismatched noise statistics means the look is right but the
sensor model is wrong, which is a different fix from the reverse.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ------------------------------------------------------------- signal statistics


@dataclass
class ImageStats:
    """Content-light statistics that separate a clean render from real CCTV."""

    noise_sigma: float        # high-frequency residual after denoising
    blockiness: float         # 8x8 boundary discontinuity, the H.264/JPEG signature
    edge_density: float       # fraction of pixels on an edge
    laplacian_var: float      # focus/sharpness proxy
    saturation_mean: float
    saturation_std: float
    luma_mean: float
    luma_std: float
    dark_fraction: float      # pixels below 16/255, crushed shadows
    clipped_fraction: float   # pixels above 250/255, blown highlights
    hf_energy: float          # normalised high-frequency spectral energy

    def as_vector(self) -> np.ndarray:
        return np.array(list(asdict(self).values()), dtype=float)

    @staticmethod
    def keys() -> list[str]:
        return list(ImageStats.__annotations__.keys())


def image_stats(img: np.ndarray) -> ImageStats:
    x = img[:, :, :3] if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    x = np.clip(x, 0, 255).astype(np.uint8)
    grey = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Noise: residual after a denoiser that preserves edges; a clean render ~0.
    den = cv2.bilateralFilter(grey.astype(np.uint8), 5, 30, 30).astype(np.float32)
    noise_sigma = float(np.std(grey - den))

    # Blockiness: discontinuity across 8x8 boundaries vs within blocks.
    _h, w = grey.shape
    hb = grey[:, 7:w - 1:8]
    hn = grey[:, 8:w:8]
    across = float(np.mean(np.abs(hb - hn))) if hb.shape == hn.shape and hb.size else 0.0
    within = float(np.mean(np.abs(np.diff(grey, axis=1)))) + 1e-6
    blockiness = across / within

    edges = cv2.Canny(x, 60, 160)
    edge_density = float((edges > 0).mean())
    laplacian_var = float(cv2.Laplacian(grey, cv2.CV_32F).var())

    f = np.fft.fftshift(np.abs(np.fft.fft2(grey - grey.mean())))
    cy, cx = np.array(f.shape) // 2
    yy, xx = np.mgrid[0:f.shape[0], 0:f.shape[1]]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    hf = float(f[r > min(f.shape) * 0.25].sum() / (f.sum() + 1e-9))

    # HSV saturation is (max-min)/max, which is unstable in near-black: pixels
    # like (4,4,4) vs (6,2,5) differ by two counts yet report high saturation, so
    # a correctly monochrome IR frame would look colourful. Measure only where
    # there is enough signal for the ratio to mean anything.
    lit = hsv[:, :, 2] >= 20
    sat = hsv[:, :, 1][lit] if lit.any() else np.zeros(1, dtype=np.float32)

    return ImageStats(
        noise_sigma=noise_sigma,
        blockiness=blockiness,
        edge_density=edge_density,
        laplacian_var=laplacian_var,
        saturation_mean=float(sat.mean()),
        saturation_std=float(sat.std()),
        luma_mean=float(grey.mean()),
        luma_std=float(grey.std()),
        dark_fraction=float((grey < 16).mean()),
        clipped_fraction=float((grey > 250).mean()),
        hf_energy=hf,
    )


def stats_gap(real: list[ImageStats], fake: list[ImageStats]) -> dict[str, dict[str, float]]:
    """Per-statistic comparison, normalised so the values are commensurable."""
    r = np.stack([s.as_vector() for s in real])
    f = np.stack([s.as_vector() for s in fake])
    out: dict[str, dict[str, float]] = {}
    for i, key in enumerate(ImageStats.keys()):
        rm, fm = float(r[:, i].mean()), float(f[:, i].mean())
        scale = abs(rm) + 1e-6
        out[key] = {
            "real": round(rm, 4),
            "render": round(fm, 4),
            "rel_gap": round((fm - rm) / scale, 4),
            "abs_gap": round(fm - rm, 4),
        }
    return out


def stats_distance(real: list[ImageStats], fake: list[ImageStats]) -> float:
    """Distance in units of the real set's own spread. Lower is better.

    Normalising by the *mean* — the obvious choice — makes statistics with
    near-zero means dominate: real `clipped_fraction` is about 0.017, so being
    0.017 out is a relative gap of 1.0, while being 200 out on `laplacian_var`
    (mean ~400) scores 0.5. Optimising against that chases the two smallest
    quantities and ignores sharpness and noise entirely, which is exactly what
    happened when this was first used to tune a profile.

    Normalising by the real standard deviation asks the right question instead:
    how many typical between-frame variations away from real footage is this?
    """
    r = np.stack([s.as_vector() for s in real])
    f = np.stack([s.as_vector() for s in fake])
    spread = np.maximum(r.std(axis=0), np.abs(r.mean(axis=0)) * 0.05 + 1e-6)
    return float(np.mean(np.abs(f.mean(axis=0) - r.mean(axis=0)) / spread))


# ----------------------------------------------------------------------- FID


# The Inception pool3 feature is 2048-dimensional, so a covariance estimated
# from fewer than 2048 images is rank-deficient by construction. `sqrtm` still
# returns a number for such a matrix, and that number looks exactly like a real
# FID — which is the trap: our first attempt would have reported FID from 12
# frames with a straight face.
FID_FEATURE_DIM = 2048
FID_MIN_SAMPLES = 2048


def _load_inception(device: str | None = None):
    """The FID backbone, plus the device it lives on."""
    import torch
    from torchvision.models import Inception_V3_Weights, inception_v3

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model, device


def _transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((299, 299), antialias=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def inception_features_from_images(images: Sequence[np.ndarray], batch: int = 32,
                                   device: str | None = None) -> np.ndarray:
    """Pool3 features for in-memory BGR images.

    Exists so a comparison never has to round-trip through PNG files it will
    read back immediately. Writing 7,700 1080p PNGs to measure a degradation
    sweep made the run disk-bound for hours and produced nothing that was
    wanted afterwards.
    """
    import torch

    model, device = _load_inception(device)
    tf = _transform()

    feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            chunk = [tf(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                     for im in images[i:i + batch] if im is not None]
            if chunk:
                feats.append(model(torch.stack(chunk).to(device)).detach().cpu().numpy())
    return np.concatenate(feats) if feats else np.zeros((0, FID_FEATURE_DIM))


def _inception_features(paths: list[Path], batch: int = 32,
                        device: str | None = None) -> np.ndarray:
    """Pool3 features from InceptionV3, the standard FID backbone."""
    import torch

    model, device = _load_inception(device)
    tf = _transform()

    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            imgs = []
            for p in paths[i:i + batch]:
                im = cv2.imread(str(p))
                if im is None:
                    continue
                imgs.append(tf(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
            if imgs:
                out = model(torch.stack(imgs).to(device))
                feats.append(out.detach().cpu().numpy())
    return np.concatenate(feats) if feats else np.zeros((0, FID_FEATURE_DIM))


def _trace_sqrt_product(a_centred: np.ndarray, b_centred: np.ndarray,
                        scale: float) -> float:
    """trace(sqrtm(S1 @ S2)) without ever forming a matrix square root.

    Writing S1 = A Aᵀ and S2 = B Bᵀ, the eigenvalues of S1 S2 are those of
    Aᵀ B Bᵀ A = M Mᵀ with M = Aᵀ B, which are exactly the squared singular
    values of M. So the trace of the matrix square root is the sum of M's
    singular values.

    This is an identity, not an approximation, and it is worth the four lines:
    `sqrtm` on the 2048×2048 Inception covariance is a Schur decomposition at
    O(d³), and a full FID sweep spent hours in it. M is (n × n) in the number of
    *images*, so with 1,100 images per side the cost drops by well over an order
    of magnitude. It is also better conditioned — with fewer images than feature
    dimensions the covariance is singular, which is precisely where `sqrtm` is
    least well behaved.
    """
    # A = Ac.T / sqrt(n1-1) is (d x n1) and B = Bc.T / sqrt(n2-1) is (d x n2),
    # so M = A.T @ B is (n1 x n2) — sized by the number of IMAGES, not by the
    # 2048 feature dimensions. Forming it the other way round yields a d x d
    # matrix that is both slower and simply not the quantity wanted.
    m = (a_centred @ b_centred.T) * scale
    return float(np.linalg.svd(m, compute_uv=False).sum())


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Frechet distance between two Gaussians fitted to the feature sets."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mu1, mu2 = a.mean(0), b.mean(0)
    s1 = np.cov(a, rowvar=False)
    s2 = np.cov(b, rowvar=False)

    # Same centring and normalisation np.cov uses, so the identity above
    # reproduces trace(sqrtm(s1 @ s2)) for these exact covariance estimates.
    ac = a - mu1
    bc = b - mu2
    scale = 1.0 / np.sqrt(max(len(a) - 1, 1) * max(len(b) - 1, 1))
    trace_covmean = _trace_sqrt_product(ac, bc, scale)

    return float(((mu1 - mu2) ** 2).sum()
                 + np.trace(np.atleast_2d(s1)) + np.trace(np.atleast_2d(s2))
                 - 2.0 * trace_covmean)


def fid(real_paths: list[Path], fake_paths: list[Path]) -> float | None:
    """FID between two image sets. Returns None if torch is unavailable.

    Prefer `fid_report`, which carries the sample counts and the noise floor.
    A bare FID is not interpretable on its own — see that function.
    """
    try:
        a = _inception_features(list(real_paths))
        b = _inception_features(list(fake_paths))
    except ImportError:
        return None
    if len(a) < 2 or len(b) < 2:
        return None
    return frechet_distance(a, b)


def fid_report(real_paths: Sequence[Path], fake_paths: Sequence[Path],
               seed: int = 0) -> dict[str, Any]:
    """FID plus the two things needed to interpret it: a floor and a warning.

    A bare FID number is meaningless without a reference. "FID 38" is not good
    or bad until you know what two samples of *the same* distribution score,
    because that floor is never zero at finite sample size and it moves with
    sample count, image resolution and content diversity. So this splits the
    real set in half and measures real-against-real, giving the floor for this
    dataset at this sample count. The honest quantity to report is the ratio:
    FID 38 against a floor of 35 means our images are nearly indistinguishable
    from real ones; against a floor of 2, it means they are not close.

    It also refuses to pretend about sample size. Inception features are 2048-d,
    so a covariance from fewer than 2048 images is rank-deficient and `sqrtm`
    returns a confident-looking number computed from noise. Below that threshold
    the result is flagged `reliable=False` with the reason attached rather than
    returned bare — the same discipline applied to degenerate AUC.
    """
    real_paths, fake_paths = list(real_paths), list(fake_paths)
    try:
        real = _inception_features(real_paths)
        fake = _inception_features(fake_paths)
    except ImportError:
        return {"fid": None, "reliable": False,
                "reason": "torch is not installed; FID cannot be computed"}

    if len(real) < 2 or len(fake) < 2:
        return {"fid": None, "reliable": False,
                "reason": f"too few images ({len(real)} real, {len(fake)} fake)"}

    value = frechet_distance(real, fake)

    # The floor: two halves of the real set, which differ only by sampling.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(real))
    half = len(real) // 2
    floor: float | None = None
    if half >= 2:
        floor = frechet_distance(real[order[:half]], real[order[half:2 * half]])

    n = min(len(real), len(fake))
    reliable = n >= FID_MIN_SAMPLES
    reason = None if reliable else (
        f"only {n} images per set; Inception features are {FID_FEATURE_DIM}-d, "
        f"so the covariance is rank-deficient below {FID_MIN_SAMPLES} and this "
        "number is computed largely from noise")

    return {
        "fid": round(value, 3),
        "floor": round(floor, 3) if floor is not None else None,
        # How far we are from real, measured in units of "how far real is from
        # itself". 1.0 would mean indistinguishable at this sample size.
        "ratio_to_floor": round(value / floor, 3) if floor else None,
        "n_real": len(real), "n_fake": len(fake),
        "reliable": reliable, "reason": reason,
    }


# -------------------------------------------------------------------- reporting


def compare_sets(real_dir: Path, fake_dir: Path, out_json: Path | None = None) -> dict:
    """Full comparison of two image folders."""
    exts = ("*.png", "*.jpg", "*.jpeg")
    real_paths = sorted(p for e in exts for p in Path(real_dir).glob(e))
    fake_paths = sorted(p for e in exts for p in Path(fake_dir).glob(e))
    if not real_paths or not fake_paths:
        raise FileNotFoundError(f"need images in both {real_dir} and {fake_dir}")

    real_stats = [image_stats(cv2.imread(str(p))) for p in real_paths]
    fake_stats = [image_stats(cv2.imread(str(p))) for p in fake_paths]

    report = {
        "n_real": len(real_paths),
        "n_render": len(fake_paths),
        "stats_distance": round(stats_distance(real_stats, fake_stats), 4),
        "per_statistic": stats_gap(real_stats, fake_stats),
        "fid": None,
        "fid_trustworthy": len(real_paths) >= 10 and len(fake_paths) >= 10,
    }
    value = fid(real_paths, fake_paths)
    if value is not None:
        report["fid"] = round(value, 3)

    if out_json:
        Path(out_json).write_text(json.dumps(report, indent=2))
    return report

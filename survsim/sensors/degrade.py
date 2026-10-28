"""CCTV sensor degradation: turn a clean render into something that looks like CCTV.

A game engine renders a *clean* image: no sensor noise, perfect focus, no
compression, ideal exposure. Real CCTV is none of those things, and the gap is
most of why models trained on synthetic data fail on real footage. This module
applies the imaging chain in physical order, each stage parameterised and seeded.

Order matters and follows the real optical/electronic path:

    scene radiance
      -> lens      (distortion, vignetting, chromatic aberration, defocus)
      -> exposure  (gain, low light, tone curve, clipping)
      -> sensor    (photon shot noise, read noise, fixed-pattern, hot pixels)
      -> motion    (blur from exposure time)
      -> IR/night  (desaturation + IR illuminator falloff)
      -> weather   (fog/haze by depth, rain on lens, dirt)
      -> ISP       (denoise smear, sharpening halos)
      -> codec     (H.264 blocking, banding)
      -> OSD       (timestamp, camera name burn-in)

Applying these in the wrong order gives a wrong image: compressing before adding
noise, for example, produces clean blocks instead of noisy ones, which is exactly
backwards from a real camera.
"""

from __future__ import annotations

import subprocess
import tempfile
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------- lens


def barrel_distort(img: np.ndarray, k1: float = -0.18, k2: float = 0.02) -> np.ndarray:
    """Brown-Conrady radial distortion. Real CCTV lenses are wide and bend lines."""
    h, w = img.shape[:2]
    fx = fy = max(w, h)
    k = np.array([[fx, 0, w / 2.0], [0, fy, h / 2.0], [0, 0, 1.0]])
    dist = np.array([k1, k2, 0.0, 0.0, 0.0])
    map1, map2 = cv2.initUndistortRectifyMap(k, -dist, None, k, (w, h), cv2.CV_32FC1)
    return cv2.remap(img, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


@lru_cache(maxsize=64)
def _vignette_r2(h: int, w: int) -> np.ndarray:
    """r**2 field for `vignette`, cached per resolution (identical every call)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2) / np.sqrt(2)
    return (r**2)[..., None].astype(np.float32)


def vignette(img: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """Corner falloff: cos^4 law, the standard optical model."""
    h, w = img.shape[:2]
    mask = 1.0 - strength * _vignette_r2(h, w)
    return np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.float32)


def chromatic_aberration(img: np.ndarray, px: float = 1.2) -> np.ndarray:
    """Scale the red and blue planes apart slightly — visible on cheap optics."""
    h, w = img.shape[:2]
    out = img.astype(np.float32).copy()
    for ch, scale in ((0, 1.0 + px / w), (2, 1.0 - px / w)):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
        out[:, :, ch] = cv2.warpAffine(out[:, :, ch], m, (w, h), borderMode=cv2.BORDER_REFLECT)
    return out


def defocus(img: np.ndarray, radius: float = 0.8) -> np.ndarray:
    """Slight softness: fixed-focus cameras are rarely sharp at every distance."""
    if radius <= 0:
        return img.astype(np.float32)
    k = int(max(3, round(radius * 4) | 1))
    return cv2.GaussianBlur(img.astype(np.float32), (k, k), radius)


# ----------------------------------------------------------------------- exposure


def exposure(img: np.ndarray, ev: float = 0.0, gamma: float = 1.0) -> np.ndarray:
    """Analogue gain in EV stops, then a tone curve. Clips like a real sensor."""
    out = img.astype(np.float32) * (2.0**ev)
    if gamma != 1.0:
        out = 255.0 * np.power(np.clip(out / 255.0, 0, 1), gamma)
    return np.clip(out, 0, 255)


# ------------------------------------------------------------------------- sensor


def sensor_noise(
    img: np.ndarray,
    rng: np.random.Generator,
    iso_gain: float = 4.0,
    read_noise: float = 2.2,
    fpn: float = 0.6,
    hot_pixel_rate: float = 2e-5,
    fixed_rng: np.random.Generator | None = None,
    grain_sigma: float = 0.0,
) -> np.ndarray:
    """Photon shot noise + read noise + fixed-pattern noise + hot pixels.

    Shot noise is Poisson and therefore *signal dependent* — bright areas are
    noisier in absolute terms, dark areas noisier relative to signal. Adding plain
    Gaussian noise instead is the usual shortcut and it looks wrong at night,
    which is precisely the condition surveillance models struggle with.
    """
    x = np.clip(img, 0, 255).astype(np.float32)
    electrons = np.maximum(x / max(iso_gain, 1e-6), 1e-6)
    # numpy's Poisson sampler is a per-element rejection algorithm and is by
    # far the most expensive line in the degrade chain (~0.3 s/frame at
    # 720p). Poisson(lambda) -> Normal(lambda, lambda) by the CLT is accurate
    # to a few percent once lambda is not tiny, so use the (vectorised, cheap)
    # Gaussian approximation there and reserve true Poisson sampling for the
    # genuinely low-photon-count pixels (dark scenes, night profile) where the
    # skew actually matters and looks different.
    shot = np.empty_like(electrons)
    big = electrons >= 20.0
    read_var = read_noise * read_noise
    if big.any():
        # Shot noise (Gaussian-approximated for these pixels) and read noise
        # are independent Gaussians, so their sum is one Gaussian with the
        # summed variance: one rng.normal draw scaled by the combined sigma
        # replaces what used to be two separate draws (one here, one for read
        # noise over the whole frame below).
        combined_sigma = np.sqrt(electrons[big] + read_var / max(iso_gain, 1e-6) ** 2)
        shot[big] = rng.normal(0.0, 1.0, int(big.sum())) * combined_sigma + electrons[big]
    if not big.all():
        small = ~big
        # Low-photon-count pixels keep true (skewed) Poisson shot noise, plus
        # its own read-noise draw sized to just this subset.
        shot[small] = (rng.poisson(electrons[small]).astype(np.float32)
                       + rng.normal(0.0, read_noise, int(small.sum())) / max(iso_gain, 1e-6))
    out = shot * iso_gain
    if grain_sigma > 0:
        # Demosaicing, the ISP's spatial denoise and 4:2:0 chroma make real
        # CCTV noise spatially correlated ("grain"), not white per-pixel. White
        # noise puts far too much energy above 0.25 cycles/px (docs/15).
        # Blurring the field lowers its sigma; rescale to keep the level.
        n = out - x
        nb = cv2.GaussianBlur(n.astype(np.float32), (0, 0), grain_sigma)
        out = x + nb * (n.std() / max(float(nb.std()), 1e-6))

    if fpn > 0:  # column-wise offset, a real CMOS artefact that survives denoising
        # Fixed pattern belongs to the sensor: pass ``fixed_rng`` (seeded per
        # camera) so it stays put across a sequence instead of flickering.
        cols = (fixed_rng or rng).normal(0.0, fpn, (1, x.shape[1], 1))
        out = out + cols

    if hot_pixel_rate > 0:
        n = int(x.shape[0] * x.shape[1] * hot_pixel_rate)
        if n:
            ys = rng.integers(0, x.shape[0], n)
            xs = rng.integers(0, x.shape[1], n)
            out[ys, xs] = rng.choice([0.0, 255.0], n)[:, None]
    return np.clip(out, 0, 255)


# ------------------------------------------------------------------------- motion


def motion_blur(img: np.ndarray, length: float = 0.0, angle_deg: float = 0.0) -> np.ndarray:
    """Linear blur standing in for subject motion during the exposure."""
    if length < 1:
        return img.astype(np.float32)
    k = int(max(3, round(length) | 1))
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, m, (k, k))
    s = kernel.sum()
    return cv2.filter2D(img.astype(np.float32), -1, kernel / (s if s else 1))


# ---------------------------------------------------------------------- night/IR


def ir_night(
    img: np.ndarray,
    rng: np.random.Generator,
    illuminator_range: float = 0.75,
    depth: np.ndarray | None = None,
) -> np.ndarray:
    """Monochrome IR mode with illuminator falloff.

    At night the IR-cut filter lifts and the image goes monochrome, lit by onboard
    IR LEDs. Their intensity falls off steeply with distance, so the far field goes
    black regardless of scene content — a major reason night detection collapses.
    """
    grey = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = grey.shape
    if depth is not None:
        d = cv2.resize(depth.astype(np.float32), (w, h))
        falloff = 1.0 / (1.0 + (d / max(d.mean(), 1e-6)) ** 2)
    else:
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        falloff = np.clip(1.0 - (r / max(illuminator_range, 1e-6)) ** 2, 0.05, 1.0)
    lit = grey * falloff
    return np.clip(np.repeat(lit[..., None], 3, axis=2), 0, 255)


def person_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    """Binary person mask (float32 0/1, ``shape`` = (h, w)) from a render's
    colour-coded instance mask (any non-black RGB pixel is a person)."""
    if mask is None:
        return None
    m = mask[..., :3].max(axis=2) if mask.ndim == 3 else mask
    m = (m > 0).astype(np.float32)
    if m.shape != tuple(shape):
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m


def thermal(img: np.ndarray, rng: np.random.Generator, mask: np.ndarray | None = None,
            sensor_w: int = 640, person: float = 0.9, bg_gain: float = 0.35,
            bg_offset: float = 0.2, halo_px: float = 5.0, halo: float = 0.12,
            netd: float = 0.012, agc_gamma: float = 1.6, agc_hi_pct: float = 99.7,
            fixed_rng: np.random.Generator | None = None) -> np.ndarray:
    """LWIR (8-14 um) thermal camera look, grey-scale radiometric-style.

    There is no emitted-heat channel in a visible render, so temperature is
    *modelled*: the background is a smoothed, compressed function of visible
    luma (emissivity/solar-load proxy: little contrast, no texture), people
    from the render's person mask are hot (~skin/clothing at 30 C over a 10 C
    street), with a soft halo (the lens/ISP bloom real uncooled cores show).
    The field is sampled at the microbolometer's resolution (``sensor_w`` wide,
    e.g. 640x512), NETD noise and a column pattern are added there, then the
    camera's AGC (percentile stretch + gamma) maps it to 8 bits and it is
    upscaled back to the output size - which is why thermal video is soft.
    Without a mask nobody is hot: only use it with ``mask`` for people.
    """
    x = np.clip(img[:, :, :3], 0, 255).astype(np.uint8)
    h, w = x.shape[:2]
    grey = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    temp = bg_offset + bg_gain * cv2.GaussianBlur(grey, (0, 0), 2.5)
    m = person_mask(mask, (h, w))
    if m is not None and m.any():
        body = person + 0.08 * (grey - grey.mean())       # clothing folds, faint
        temp = temp * (1 - m) + body * m
        temp = temp + halo * cv2.GaussianBlur(m, (0, 0), max(halo_px, 0.1)) * (1 - m)
    sh = max(8, round(h * sensor_w / w))
    t = cv2.resize(temp, (sensor_w, sh), interpolation=cv2.INTER_AREA)
    t = t + rng.normal(0.0, netd, t.shape).astype(np.float32)
    t = t + (fixed_rng or rng).normal(0.0, netd * 0.5, (1, sensor_w)).astype(np.float32)
    lo, hi = np.percentile(t, [0.5, agc_hi_pct])
    t = np.clip((t - lo) / max(hi - lo, 1e-6), 0, 1) ** agc_gamma * 255.0
    t = cv2.resize(t, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(np.repeat(t[..., None], 3, axis=2), 0, 255)


def desaturate(img: np.ndarray, saturation: float = 1.0) -> np.ndarray:
    """Chroma loss: blend toward luma (low light starves the colour channels)."""
    if saturation >= 1.0:
        return img
    x = img.astype(np.float32)
    g = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None]
    return np.clip(g + saturation * (x - g), 0, 255)


# ------------------------------------------------------------------------ weather


def haze(img: np.ndarray, density: float = 0.25, depth: np.ndarray | None = None) -> np.ndarray:
    """Atmospheric scattering: I = I0*t + A*(1-t), t = exp(-beta*d)."""
    x = np.clip(img, 0, 255).astype(np.float32)
    h, w = x.shape[:2]
    if depth is not None:
        d = cv2.resize(depth.astype(np.float32), (w, h))
        d = d / max(np.percentile(d, 95), 1e-6)
    else:
        d = np.tile(np.linspace(0.2, 1.0, h)[:, None], (1, w))  # far field near horizon
    t = np.exp(-density * 3.0 * d)[..., None]
    airlight = 220.0
    return np.clip(x * t + airlight * (1 - t), 0, 255)


def lens_dirt(img: np.ndarray, rng: np.random.Generator, amount: float = 0.12) -> np.ndarray:
    """Smudges and dust on the dome — almost universal on real installed cameras."""
    if amount <= 0:
        return img.astype(np.float32)
    h, w = img.shape[:2]
    blobs = np.zeros((h, w), np.float32)
    for _ in range(int(18 * amount * 10)):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        rad = int(abs(rng.normal(14, 9))) + 3
        cv2.circle(blobs, (int(cx), int(cy)), rad, float(abs(rng.normal(0.4, 0.25))), -1)
    blobs = cv2.GaussianBlur(blobs, (0, 0), 9)[..., None]
    x = img.astype(np.float32)
    return np.clip(x * (1 - 0.5 * blobs * amount * 6) + 190 * blobs * amount * 3, 0, 255)


# --------------------------------------------------------------------------- ISP


def isp(img: np.ndarray, denoise: float = 5.0, sharpen: float = 0.45) -> np.ndarray:
    """Camera ISP: aggressive denoise that smears detail, then sharpening halos.

    The smear-then-sharpen signature is highly characteristic of CCTV and is what
    makes real footage look "processed" compared with a clean render.
    """
    x = np.clip(img, 0, 255).astype(np.uint8)
    if denoise > 0:
        x = cv2.bilateralFilter(x, 5, denoise * 2, denoise * 2)
    if sharpen > 0:
        blur = cv2.GaussianBlur(x, (0, 0), 1.2)
        x = cv2.addWeighted(x, 1 + sharpen, blur, -sharpen, 0)
    return np.clip(x, 0, 255).astype(np.float32)


# ------------------------------------------------------------------------- codec


def h264(img: np.ndarray, crf: int = 32, scale: float | None = None) -> np.ndarray:
    """Real H.264 blocking by round-tripping the frame through ffmpeg.

    JPEG is the usual stand-in, but its 8x8 artefacts differ from H.264's
    deblocked macroblocks; since detectors are demonstrably sensitive to codec
    artefacts, the honest move is to use the actual codec.
    """
    x = np.clip(img, 0, 255).astype(np.uint8)
    if scale and scale != 1.0:
        h, w = x.shape[:2]
        small = cv2.resize(x, (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2))
    else:
        small = x if x.shape[0] % 2 == 0 and x.shape[1] % 2 == 0 else x[: x.shape[0] // 2 * 2, : x.shape[1] // 2 * 2]

    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "a.png", Path(td) / "c.png"
        # Compression level only affects how fast we *write* this throwaway
        # file; PNG is lossless, so the pixels ffmpeg reads back are identical
        # regardless of level, and level 1 trades disk time for CPU we're not
        # short of here.
        cv2.imwrite(str(src), small, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        try:
            # Encode and decode as one pipeline (encoder's stdout feeds the
            # decoder's stdin directly) instead of two subprocess.run calls
            # round-tripping through an intermediate .mp4 file: this is the
            # same two ffmpeg invocations bit-for-bit (verified: 0 px diff),
            # just without the extra process-wait and disk round trip.
            enc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                 "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
                 "-f", "mp4", "-movflags", "frag_keyframe+empty_moov", "pipe:1"],
                stdout=subprocess.PIPE)
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", "pipe:0", str(dst)],
                stdin=enc.stdout, check=True, capture_output=True, timeout=120)
            enc.stdout.close()
            if enc.wait(timeout=120) != 0:
                raise subprocess.CalledProcessError(enc.returncode, "ffmpeg")
            out = cv2.imread(str(dst))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            # Fall back to JPEG so the pipeline still runs without ffmpeg.
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, max(5, 100 - crf * 2)])
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else small
    if out is None:
        out = small
    if out.shape[:2] != x.shape[:2]:
        out = cv2.resize(out, (x.shape[1], x.shape[0]))
    return out.astype(np.float32)


def h264_sequence(frames: list[np.ndarray], crf: int = 32, fps: float = 25.0,
                  scale: float | None = None, gop: int | None = None,
                  bf: int = 2, preset: str = "fast") -> list[np.ndarray]:
    """Real H.264 GOP structure (I/P/B frames) for a WHOLE sequence in one
    ffmpeg encode -> decode round trip, instead of one ffmpeg pair per frame.

    Per-frame intra-only "compression" was never just slow, it was also less
    realistic: a real DVR predicts across frames with a GOP of ~25-50, and that
    inter-frame prediction is what produces the temporal blocking/smearing seen
    in real CCTV, not just per-frame blockiness. Frames are piped through as
    raw video (rawvideo bgr24 both ways) so there are no per-frame subprocess
    spawns or intermediate files.
    """
    if not frames:
        return []
    x0 = np.clip(frames[0], 0, 255).astype(np.uint8)
    h, w = x0.shape[:2]
    if scale and scale != 1.0:
        sw, sh = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    else:
        sw, sh = w // 2 * 2, h // 2 * 2
    gop = gop or int(np.clip(round(fps * 2), 25, 50))

    def _small(f: np.ndarray) -> np.ndarray:
        x = np.clip(f, 0, 255).astype(np.uint8)
        if (sw, sh) != (x.shape[1], x.shape[0]):
            x = cv2.resize(x, (sw, sh))
        return np.ascontiguousarray(x)

    try:
        enc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{sw}x{sh}", "-r", str(fps), "-i", "pipe:0",
             "-c:v", "libx264", "-crf", str(crf), "-g", str(gop), "-bf", str(bf),
             "-preset", preset, "-pix_fmt", "yuv420p",
             "-f", "mp4", "-movflags", "frag_keyframe+empty_moov", "pipe:1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        dec = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", "pipe:0",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
            stdin=enc.stdout, stdout=subprocess.PIPE)
        enc.stdout.close()  # dec holds its own dup; drop ours so EOF propagates

        write_err: list[BaseException] = []

        def _write() -> None:
            try:
                for f in frames:
                    enc.stdin.write(_small(f).tobytes())
            except BaseException as exc:  # noqa: BLE001 - surfaced to caller below
                write_err.append(exc)
            finally:
                try:
                    enc.stdin.close()
                except OSError:
                    pass

        writer = threading.Thread(target=_write)
        writer.start()
        raw = dec.stdout.read()
        writer.join(timeout=180)
        enc_rc = enc.wait(timeout=180)
        dec_rc = dec.wait(timeout=180)
        if write_err or enc_rc != 0 or dec_rc != 0:
            raise subprocess.CalledProcessError(enc_rc or dec_rc or 1, "ffmpeg")

        frame_bytes = sh * sw * 3
        n = len(raw) // frame_bytes
        out = [np.frombuffer(raw[i * frame_bytes:(i + 1) * frame_bytes], np.uint8)
               .reshape(sh, sw, 3).copy() for i in range(n)]
        # A short read (decoder drops a trailing partial GOP) pads with the
        # last good frame instead of desyncing the sequence against the OSD
        # clock, which is keyed to frame index.
        while len(out) < len(frames):
            out.append(out[-1].copy() if out else np.zeros((sh, sw, 3), np.uint8))
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        # Fall back to the old per-frame path so the pipeline still runs.
        return [h264(f, crf, scale) for f in frames]

    if (sw, sh) != (w, h):
        out = [cv2.resize(o, (w, h)) for o in out]
    return [o.astype(np.float32) for o in out]


# --------------------------------------------------------------------------- OSD


def osd(img: np.ndarray, text: str = "CAM01", timestamp: str = "2026-09-17 21:14:03") -> np.ndarray:
    """Burnt-in timestamp and camera name, as nearly every real DVR produces."""
    x = np.clip(img, 0, 255).astype(np.uint8).copy()
    h, w = x.shape[:2]
    f, sc, th = cv2.FONT_HERSHEY_SIMPLEX, max(0.4, w / 1800), 1
    for pos, s in (((10, 26), text), ((w - int(w * 0.34), h - 14), timestamp)):
        cv2.putText(x, s, pos, f, sc, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(x, s, pos, f, sc, (235, 235, 235), th, cv2.LINE_AA)
    return x.astype(np.float32)


# ----------------------------------------------------------- camera automatics
#
# Real cameras are not static transfer functions. Auto-exposure hunts, auto
# white balance wanders as the scene changes, and a CMOS rolling shutter under
# mains lighting leaves slow horizontal banding. These are the fc-synthetic
# "camera automatics" effects: small, slow, and seed-deterministic, so the same
# (sequence_seed, frame_index) always gives the same frame.


def drift(sequence_seed: int, t: float, period_s: float, channels: int = 1) -> np.ndarray:
    """Smooth zero-mean drift in [-1, 1]: three incommensurate sinusoids.

    A function of time rather than a random walk, so frame k can be rendered
    without rendering frames 0..k-1 and two workers agree on every frame.
    """
    r = np.random.default_rng([sequence_seed, 0xA11E])
    phase = r.uniform(0, 2 * np.pi, (channels, 3))
    freq = np.array([1.0, 2.7, 6.3]) / max(period_s, 1e-6)
    w = np.array([0.6, 0.3, 0.1])
    return (w * np.sin(2 * np.pi * freq * t + phase)).sum(axis=1)


def camera_automatics(img: np.ndarray, sequence_seed: int, t: float, ae_ev: float = 0.0,
                      awb: float = 0.0, period_s: float = 40.0,
                      wb_gains: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """AE (global EV) and AWB (per-channel gain) drift, plus a static WB cast (BGR)."""
    x = img.astype(np.float32)
    ev = ae_ev * drift(sequence_seed, t, period_s)[0]
    gains = np.asarray(wb_gains, np.float32) * (1.0 + awb * drift(sequence_seed + 1, t, period_s * 1.7, 3))
    gains = gains / gains[1]  # AWB normalises to green; exposure is separate
    return np.clip(x * (2.0**ev) * gains[None, None, :], 0, 255)


def auto_exposure(img: np.ndarray, target: float = 0.0, max_ev: float = 2.0,
                  levels: float = 0.0) -> np.ndarray:
    """Metered AE toward a target mean luma, then a partial auto-levels stretch.

    A render is exposed however the artist lit it; a CCTV camera meters the
    scene and pulls it toward mid-grey, and its ISP stretches a flat histogram
    (the "auto contrast"/WDR stage). ``target`` 0 disables AE; ``levels`` in
    [0, 1] blends toward a p1..p99 -> 8..247 stretch.
    """
    x = img.astype(np.float32)
    if levels > 0:
        lo, hi = np.percentile(x.mean(axis=2), [1, 99])
        if hi - lo > 4:
            st = (x - lo) * (239.0 / (hi - lo)) + 8.0
            x = np.clip(x * (1 - levels) + st * levels, 0, 255)
    if target > 0:
        # Meter on the median, as centre-weighted/average metering is robust to
        # a few specular highlights; scale in linear-ish space, soft-clip top.
        ev = np.clip(np.log2(target / max(float(np.median(x.mean(axis=2))), 1.0)), -max_ev, max_ev)
        x = x * 2.0**ev
    return np.clip(x, 0, 255)


def rolling_exposure(img: np.ndarray, sequence_seed: int, t: float, amount: float = 0.0,
                     bands: float = 1.3, hz: float = 0.35) -> np.ndarray:
    """Slow row-wise gain ripple drifting down the frame (rolling shutter vs mains)."""
    if amount <= 0:
        return img.astype(np.float32)
    h = img.shape[0]
    ph = np.random.default_rng([sequence_seed, 0x7011]).uniform(0, 2 * np.pi)
    rows = 1.0 + amount * np.sin(2 * np.pi * (bands * np.arange(h) / h + hz * t) + ph)
    return np.clip(img.astype(np.float32) * rows[:, None, None].astype(np.float32), 0, 255)


def dark_flat_quantise(img: np.ndarray, strength: float = 0.0, block: int = 8) -> np.ndarray:
    """Pull dark, flat 8x8 blocks toward their DC value.

    Encoders spend bits where the eye looks; dark and flat areas get coarse
    quantisation, so noise there collapses into visible blocks. A uniform CRF
    round trip does not reproduce that bias, so it is added explicitly.
    """
    if strength <= 0:
        return img.astype(np.float32)
    x = img.astype(np.float32)
    h, w = x.shape[:2]
    hh, ww = h // block * block, w // block * block
    core = x[:hh, :ww]
    sh, sw = hh // block, ww // block
    # Per-block mean and std, computed the fast way instead of the reshape +
    # ufunc.reduce over axes (1, 3) this replaced (the single largest cost in
    # the degrade chain after sensor noise, ~0.13 s/frame at 720p): for an
    # exact-integer downscale factor, cv2.resize(..., INTER_AREA) IS the
    # per-block mean (it is how area interpolation is defined), computed by
    # OpenCV's vectorised C++ box-sum rather than a Python-side numpy
    # reduction. Block variance follows from E[x^2] - E[x]^2 using the same
    # trick on the squared image, then both are broadcast back to per-pixel
    # with a nearest-neighbour upsample (constant within each block, matching
    # the original's block-wise assignment).
    dc_small = cv2.resize(core, (sw, sh), interpolation=cv2.INTER_AREA)
    sq_small = cv2.resize(core * core, (sw, sh), interpolation=cv2.INTER_AREA)
    var_small = np.clip(sq_small - dc_small * dc_small, 0.0, None)
    act_small = np.sqrt(var_small).mean(axis=-1, keepdims=True)
    luma_small = dc_small.mean(axis=-1, keepdims=True)
    wgt_small = (strength * np.clip(1.0 - luma_small / 90.0, 0, 1)
                 * np.clip(1.0 - act_small / 12.0, 0, 1)).astype(np.float32)
    dc = cv2.resize(dc_small, (ww, hh), interpolation=cv2.INTER_NEAREST)
    wgt = cv2.resize(wgt_small, (ww, hh), interpolation=cv2.INTER_NEAREST)
    if wgt.ndim == 2:
        wgt = wgt[..., None]
    out = x.copy()
    out[:hh, :ww] = core * (1 - wgt) + dc * wgt
    return out


# ------------------------------------------------------------------------ profile


@dataclass
class Profile:
    """A named CCTV look. Values are deliberately physical, not arbitrary sliders."""

    name: str = "cctv_day_1080p"
    resolution: tuple[int, int] | None = (1280, 720)
    k1: float = -0.16
    k2: float = 0.015
    vignette: float = 0.30
    chroma_px: float = 1.0
    defocus: float = 0.7
    ev: float = 0.0
    gamma: float = 1.0
    iso_gain: float = 3.0
    read_noise: float = 2.0
    fpn: float = 0.5
    hot_pixels: float = 1e-5
    motion_px: float = 0.0
    motion_angle: float = 0.0
    night_ir: bool = False
    haze: float = 0.0
    dirt: float = 0.10
    denoise: float = 5.0
    sharpen: float = 0.45
    crf: int = 30
    codec_scale: float | None = None
    osd: bool = True
    osd_text: str = "CAM01"
    # Camera automatics and codec bias (all off by default = old behaviour).
    ae_drift_ev: float = 0.0
    awb_drift: float = 0.0
    drift_period_s: float = 40.0
    wb_gains: tuple[float, float, float] = (1.0, 1.0, 1.0)   # BGR static cast
    rolling: float = 0.0
    dark_flatten: float = 0.0
    ae_target: float = 0.0          # metered mean luma; 0 = no AE
    auto_levels: float = 0.0
    grain_sigma: float = 0.0
    fps: float = 25.0
    # Sequence-level codec structure (only used by apply_sequence's batched
    # h264_sequence; Profile.apply's per-frame intra codec ignores these).
    gop: int | None = None
    bf: int = 2
    # Night variants (H32, docs/15): chroma loss for visible low light, and a
    # thermal (LWIR) mode that needs the person mask. ``mono`` forces a
    # grey output after the codec (a thermal camera has no chroma at all).
    saturation: float = 1.0
    thermal: bool = False
    thermal_kw: tuple = ()
    mono: bool = False

    def apply(self, img: np.ndarray, depth: np.ndarray | None = None, seed: int = 0,
              camera=None, frame_index: int | None = None,
              sequence_seed: int | None = None, mask: np.ndarray | None = None) -> np.ndarray:
        """Degrade one frame (single-frame callers): pre-codec chain, then the
        old per-frame intra-only codec round trip, then OSD.

        ``seed`` drives per-frame randomness (shot/read noise). For a video,
        pass a fixed ``sequence_seed`` and the ``frame_index``: drift, rolling
        banding, fixed-pattern noise and lens dirt then belong to the camera and
        stay coherent across frames while the sensor noise changes every frame.
        Without them the call behaves as a single still (t = 0).
        """
        x = self._pre_codec(img, depth=depth, seed=seed, camera=camera,
                            frame_index=frame_index, sequence_seed=sequence_seed, mask=mask)
        x = h264(x, self.crf, self.codec_scale)
        x = self.post_codec(x)
        if self.osd:
            x = osd(x, self.osd_text)
        return np.clip(x, 0, 255).astype(np.uint8)

    def _pre_codec(self, img: np.ndarray, depth: np.ndarray | None = None, seed: int = 0,
                   camera=None, frame_index: int | None = None,
                   sequence_seed: int | None = None, mask: np.ndarray | None = None) -> np.ndarray:
        """Every stage up to (not including) the codec and OSD.

        Split out of ``apply`` so ``apply_sequence`` can run this per-frame
        chain as before and then batch just the codec stage across the whole
        sequence in one ffmpeg round trip (see ``h264_sequence``).
        """
        rng = np.random.default_rng(seed)
        sseq = seed if sequence_seed is None else sequence_seed
        t = 0.0 if frame_index is None else frame_index / self.fps
        x = img[:, :, :3].astype(np.float32)

        # Lens first, at the camera's native resolution, through the SAME model
        # ground truth is projected with (sensors/lens.py). This stage used to
        # apply the profile's own k1/k2 with an invented focal length, which
        # moved corner pixels 26.6 px against a pinhole ground truth. The
        # profile's k1/k2 no longer distort anything: a lens belongs to the
        # camera, so put it in `optics.distortion` and ground truth follows.
        if camera is not None:
            from .lens import distort_image
            x = distort_image(x, camera)

        if self.resolution:
            x = cv2.resize(x, self.resolution, interpolation=cv2.INTER_AREA)
            if depth is not None:
                depth = cv2.resize(depth, self.resolution, interpolation=cv2.INTER_NEAREST)

        if self.thermal:
            x = thermal(x, rng, mask=mask, fixed_rng=np.random.default_rng([sseq, 0x7E]),
                        **dict(self.thermal_kw))
        x = vignette(x, self.vignette)
        x = chromatic_aberration(x, self.chroma_px)
        x = defocus(x, self.defocus)
        x = auto_exposure(x, self.ae_target, levels=self.auto_levels) \
            if (self.ae_target or self.auto_levels) else x
        x = exposure(x, self.ev, self.gamma)
        if self.ae_drift_ev or self.awb_drift or self.wb_gains != (1.0, 1.0, 1.0):
            x = camera_automatics(x, sseq, t, self.ae_drift_ev, self.awb_drift,
                                  self.drift_period_s, self.wb_gains)
        x = desaturate(x, self.saturation)
        x = rolling_exposure(x, sseq, t, self.rolling)
        if self.haze > 0:
            x = haze(x, self.haze, depth)
        if self.night_ir:
            x = ir_night(x, rng, depth=depth)
        cam_rng = None if sequence_seed is None else np.random.default_rng([sequence_seed, 0xF1])
        x = sensor_noise(x, rng, self.iso_gain, self.read_noise, self.fpn, self.hot_pixels,
                         fixed_rng=cam_rng, grain_sigma=self.grain_sigma)
        if self.motion_px > 0:
            x = motion_blur(x, self.motion_px, self.motion_angle)
        if self.dirt > 0:
            x = lens_dirt(x, cam_rng or rng, self.dirt)
        x = isp(x, self.denoise, self.sharpen)
        x = dark_flat_quantise(x, self.dark_flatten)
        return x

    def post_codec(self, x: np.ndarray) -> np.ndarray:
        if self.mono:
            g = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            x = np.repeat(g[..., None], 3, axis=2)
        return x


PROFILES: dict[str, Profile] = {
    "clean": Profile(name="clean", resolution=None, k1=0, k2=0, vignette=0, chroma_px=0,
                     defocus=0, iso_gain=1e-6, read_noise=0, fpn=0, hot_pixels=0,
                     dirt=0, denoise=0, sharpen=0, crf=1, osd=False),
    # A decent modern camera in daylight: still compressed, still processed.
    # Tuned by searching against 12 real MOTChallenge frames, not against
    # reasoned-out targets. The earlier hand-tuned version was *further* from real
    # footage than the undegraded render (1.304 vs 1.072 in units of the real
    # set's spread) because it softened heavily on the assumption that CCTV is
    # blurry. Measurement disagreed: defocus 0 wins, real frames keep far more
    # edge detail than the legacy-4CIF mental model suggests.
    # Retuned 2026-09-25 against image statistics of 80 real MOT frames (docs/15):
    # metered AE + auto-levels (renders are exposed flat/bright), gamma, lower
    # gain, lower CRF, plus AE/AWB drift and dark-flat codec bias. 12/14 stats
    # in the real p10-p90 band on the proxy set; mtf_ratio and edge_width
    # still HIGH (unfinished, see docs/15).
    # Retuned 2026-09-25 (docs/15 "Sequence codec batching, take 2") against
    # the batched-GOP sequence codec on real UE renders (80/300 frames,
    # out/batch_v1/street_crowd_day/render/cam_02): short GOP (more I-frames,
    # less inter-frame smoothing of sensor noise) plus spatially-correlated
    # grain (survives P/B prediction far better than white per-pixel noise —
    # see grain_sigma in sensor_noise). 9/14 docs-table stats in band on that
    # real render, including all 5 of the spatial imaging-chain stats pinned
    # by tests/test_realism_stats.py. temporal_noise still overshoots the
    # band (grain_sigma high enough to survive compression at all overshoots
    # once it does) — see docs/15 for the measurement and the JPEG-derived
    # real-band caveat.
    "cctv_day_720p": Profile(name="cctv_day_720p", resolution=(1280, 720), crf=29,
                             iso_gain=2.7, read_noise=2.0, defocus=0.82, gamma=1.0,
                             denoise=5.0, sharpen=0.5, dirt=0.08, hot_pixels=0,
                             ae_target=95.0, auto_levels=1.0, dark_flatten=0.3,
                             ae_drift_ev=0.15, awb_drift=0.05, rolling=0.01,
                             grain_sigma=3.0, gop=8, bf=0),
    # The far more common reality: older 4CIF-era camera, heavy compression.
    # Retuned 2026-09-25 against the batched-GOP codec (N=10 real-frame sweep
    # over gop/bf/crf/grain_sigma, docs/15): 9/14 docs-table stats in band.
    "cctv_legacy_sd": Profile(name="cctv_legacy_sd", resolution=(704, 576), k1=-0.24, k2=0.04,
                              vignette=0.42, defocus=1.4, iso_gain=6.0, read_noise=3.2,
                              fpn=1.1, dirt=0.2, denoise=8.0, sharpen=0.7, crf=34,
                              codec_scale=0.75, osd_text="CAM03 PLATFORM",
                              grain_sigma=1.5, gop=8, bf=2),
    # Night IR: monochrome, illuminator falloff, high gain, motion smear.
    # gop/bf retuned 2026-09-25 the same way (docs/15); best found 6/14
    # against the *day* real-render sweep set, since no real night/IR
    # reference frames exist yet (docs/15 still flags this as a gap) — most
    # of the misses are luma/chroma/contrast, expected because this profile
    # is deliberately monochrome and far darker than the day proxy set.
    # grain_sigma is capped at 0.3, not the sweep's nominal winner (4.5):
    # sensor_noise draws noise independently per (BGR) channel, so on a
    # monochrome-collapsed night frame a large grain_sigma reintroduces
    # visible chroma noise (test_night_is_monochrome_and_noisier_than_day /
    # test_ir_output_is_effectively_monochrome caught this at 4.5 — chroma
    # went from ~0.7 to 8.3). 0.3 is the largest value that keeps it there.
    "cctv_night_ir": Profile(name="cctv_night_ir", resolution=(1280, 720), night_ir=True,
                             ev=-0.4, iso_gain=11.0, read_noise=5.0, fpn=1.6,
                             motion_px=5.0, defocus=1.1, denoise=9.0, sharpen=0.6,
                             crf=30, dirt=0.16, osd_text="CAM07 YARD  IR",
                             grain_sigma=0.3, gop=8, bf=2),
    # H32 (docs/15, 32): visible low-light street camera (IR-cut in, sodium /
    # LED street lighting). Random-searched against the first 15 LLVIP-visible
    # frames only (measurement only: LLVIP is non-commercial); the last 10 are
    # the hold-out the pre-registration scores.
    "cctv_night_lowlight": Profile(name="cctv_night_lowlight", resolution=(1280, 720),
                                   ae_target=48.65, gamma=1.02, saturation=0.56, iso_gain=9.76,
                                   read_noise=2.56, fpn=1.0, defocus=1.52, denoise=1.56,
                                   sharpen=0.16, crf=39, wb_gains=(0.70, 1.0, 1.03),
                                   grain_sigma=1.48, gop=8, bf=2, dirt=0.1, auto_levels=0.39,
                                   osd_text="CAM07 STREET"),
    # H32e: cctv_night_lowlight with ONLY noise (s 0.7: iso_gain*0.49, read_noise/fpn*0.7) and CRF
    # fitted to LLVIP tuning-15 sigma+blockiness (in-sample). Fit kept crf 39; HN failed 2/13 on the
    # hold-out; recall 0.018 -- NOT a usable night profile (docs/research/32 §H32e).
    "cctv_night_lowlight_v2": Profile(name="cctv_night_lowlight_v2", resolution=(1280, 720),
                                      ae_target=48.65, gamma=1.02, saturation=0.56, iso_gain=9.76 * 0.49,
                                      read_noise=2.56 * 0.7, fpn=0.7, defocus=1.52, denoise=1.56,
                                      sharpen=0.16, crf=39, wb_gains=(0.70, 1.0, 1.03),
                                      grain_sigma=1.48, gop=8, bf=2, dirt=0.1, auto_levels=0.39,
                                      osd_text="CAM07 STREET"),
    # H32: uncooled LWIR thermal (640-wide microbolometer, AGC, NETD). Needs
    # the person mask (apply_sequence(masks=...)); tuned like the one above
    # against the first 15 LLVIP-thermal frames.
    "cctv_thermal": Profile(name="cctv_thermal", resolution=(1280, 720), thermal=True, mono=True,
                            thermal_kw=(("person", 0.776), ("bg_gain", 0.579), ("bg_offset", 0.2),
                                        ("halo_px", 5.05), ("halo", 0.219), ("netd", 0.0121),
                                        ("agc_gamma", 1.159), ("agc_hi_pct", 99.03)),
                            vignette=0.128, chroma_px=0, k1=0, k2=0, defocus=0.745, iso_gain=1e-6,
                            read_noise=0, fpn=0, hot_pixels=0, dirt=0, denoise=0, sharpen=0.386,
                            crf=33, gop=8, bf=2, osd_text="CAM07 THERMAL"),
    # Fog/rain daytime: low contrast, wet, hazy.
    "cctv_fog_day": Profile(name="cctv_fog_day", resolution=(1280, 720), haze=0.45,
                            iso_gain=4.0, defocus=1.0, dirt=0.25, crf=32),
    # Showcase (docs/research/95): 640x360 dusk-in-rain DVR look. There is no
    # droplet model, so "rain on lens" is approximated by heavy lens dirt +
    # haze + softness; the lamp flicker is EMULATED as fast AE drift + mains
    # rolling banding (no scripted light in the render). Label it as such.
    "cctv_showcase_rain_360p": Profile(name="cctv_showcase_rain_360p", resolution=(640, 360),
                                       haze=0.12, dirt=0.12, defocus=0.8, iso_gain=1.0,
                                       read_noise=0.6, fpn=0.2, crf=28, ae_drift_ev=0.35,
                                       drift_period_s=3.0, rolling=0.06, awb_drift=0.04,
                                       grain_sigma=0.0, gop=8, bf=0, fps=5.0,
                                       osd_text="SITE-N"),
}


def apply_profile(img: np.ndarray, profile: str | Profile, depth=None, seed: int = 0,
                  camera=None, frame_index: int | None = None,
                  sequence_seed: int | None = None, mask=None) -> np.ndarray:
    p = PROFILES[profile] if isinstance(profile, str) else profile
    return p.apply(img, depth=depth, seed=seed, camera=camera, frame_index=frame_index,
                   sequence_seed=sequence_seed, mask=mask)


def apply_sequence(frames, profile: str | Profile, sequence_seed: int = 0, depths=None,
                   camera=None, masks=None):
    """Degrade a video: camera-level effects coherent, sensor noise fresh per frame.

    The per-frame chain (lens..ISP) runs exactly as ``Profile.apply`` does, one
    frame at a time. The codec stage is different: instead of one ffmpeg
    encode/decode pair per frame (bit-exact but ~2 s/frame and, worse, not what
    a real DVR does — a single stream with GOP/P/B prediction), the whole
    sequence's pre-codec frames are batched through ONE ffmpeg round trip
    (``h264_sequence``), which is both far faster and more realistic. This
    means ``apply_sequence`` must materialise the pre-codec frames before it
    can start yielding — it is no longer frame-lazy.
    """
    p = PROFILES[profile] if isinstance(profile, str) else profile
    pre = []
    depths_list = list(depths) if depths is not None else None
    masks_list = list(masks) if masks is not None else None
    for i, f in enumerate(frames):
        d = None if depths_list is None else depths_list[i]
        m = None if masks_list is None else masks_list[i]
        x = p._pre_codec(f, depth=d, seed=hash((sequence_seed, i)) & 0x7FFFFFFF, camera=camera,
                         frame_index=i, sequence_seed=sequence_seed, mask=m)
        pre.append(np.clip(x, 0, 255).astype(np.uint8))

    decoded = h264_sequence(pre, p.crf, p.fps, p.codec_scale, gop=p.gop, bf=p.bf)

    for x in decoded:
        x = p.post_codec(x)
        if p.osd:
            x = osd(x, p.osd_text)
        yield np.clip(x, 0, 255).astype(np.uint8)

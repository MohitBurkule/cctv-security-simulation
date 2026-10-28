"""Gait as a measurable signal, not a label.

`Gait.style` has until now been a *name*. It nudged walking speed and nothing
else, which left three things quietly broken:

  - **Gait could not serve as a biometric.** `appearance.soft_biometrics`
    returns body measurements only, so the cue most famous for surviving a
    change of clothes was absent from the one place appearance-change
    evaluation looks.
  - **The gait evasion techniques degraded nothing.** `evasion.fake_limp`
    claims to degrade gait recognition by 0.85, but with no gait signal there
    was nothing to degrade — the technique was an annotation.
  - **Two people with the same style were identical.** Any gait matcher
    evaluated on that data would score either perfectly or at chance, and
    neither number would mean anything.

This module makes gait a vector of measurable quantities, with two properties
that matter more than the specific numbers:

**Individuality.** Real gait is idiosyncratic — that is precisely why gait
recognition works at all. Every person gets a stable, seeded offset on each
parameter, so two "normal" walkers are similar but not identical. Without this,
gait ReID is a meaningless test.

**Calibration.** Cadence and step length come from real CCTV rather than from
intuition: pose estimation over MOT20 gives 123 spm at a 0.39 m step in a crowd,
against ~110 spm and ~0.75 m for free walking (docs/14 §10). The crowd response
is the surprising part and is modelled explicitly — people keep their leg rhythm
and shorten their stride. They do not walk in slow motion.

Style effects are stated as *multipliers on a normal walk*, so the claim each
one makes is legible and arguable rather than buried in a constant.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from survsim.director.crowd import FREE_CADENCE_SPM, FREE_STEP_LENGTH_M

# Per-parameter effect of each gait style, as a multiplier on a normal walk
# except `asymmetry`, which is absolute because a normal walk has none.
#
# Provenance differs per row and is marked, because a reader deserves to know
# which numbers carry evidence. `cadence` and `step` for a normal walk are
# MEASURED (docs/14 §10). `carrying_bag` and `heavy_coat` are MEASURED from
# CASIA-B across 123-124 subjects, paired within subject (docs/14 §13). The rest
# are ARGUED — no public dataset gives parameters for a deliberate limp or a
# shuffle — and live here so they can be disputed rather than scattered through
# the code.
MEASURED_STYLES = frozenset({"normal", "carrying_bag", "heavy_coat"})

STYLE_EFFECTS: dict[str, dict[str, float]] = {
    #              cadence  step   asymmetry  arm_swing  sway   stance_width
    "normal":    {"cadence": 1.00, "step": 1.00, "asymmetry": 0.00,
                  "arm_swing": 1.00, "sway": 1.00, "stance_width": 1.00},
    "brisk":     {"cadence": 1.15, "step": 1.12, "asymmetry": 0.00,
                  "arm_swing": 1.25, "sway": 0.90, "stance_width": 0.95},
    "slow":      {"cadence": 0.82, "step": 0.85, "asymmetry": 0.00,
                  "arm_swing": 0.75, "sway": 1.05, "stance_width": 1.05},
    # A limp is the asymmetric case: one stance phase is shortened, so left and
    # right steps differ in length and duration. That asymmetry is the signal.
    "limp_L":    {"cadence": 0.88, "step": 0.80, "asymmetry": 0.35,
                  "arm_swing": 0.70, "sway": 1.35, "stance_width": 1.10},
    "limp_R":    {"cadence": 0.88, "step": 0.80, "asymmetry": -0.35,
                  "arm_swing": 0.70, "sway": 1.35, "stance_width": 1.10},
    "shuffle":   {"cadence": 1.05, "step": 0.55, "asymmetry": 0.05,
                  "arm_swing": 0.40, "sway": 0.80, "stance_width": 1.15},
    "drunk":     {"cadence": 0.85, "step": 0.90, "asymmetry": 0.20,
                  "arm_swing": 1.40, "sway": 3.00, "stance_width": 1.45},
    "military":  {"cadence": 1.12, "step": 1.05, "asymmetry": 0.00,
                  "arm_swing": 1.60, "sway": 0.55, "stance_width": 0.90},
    "crouch":    {"cadence": 0.95, "step": 0.65, "asymmetry": 0.05,
                  "arm_swing": 0.55, "sway": 1.20, "stance_width": 1.30},
    "run":       {"cadence": 1.55, "step": 1.70, "asymmetry": 0.00,
                  "arm_swing": 1.70, "sway": 0.85, "stance_width": 0.85},

    # --- MEASURED from CASIA-B, paired within subject (docs/14 §13) ---
    #
    # Carrying a bag: cadence x1.182 (SE 0.032, z=+5.8), leg spread x0.962, and
    # an asymmetry increase of +0.040 (SE 0.004, z=+9.9) across 123 subjects.
    # This is the empirical basis for `evasion.weighted_asymmetry`: a weight on
    # one side really does make the walk lopsided, and the effect is large
    # relative to between-person variation.
    "carrying_bag": {"cadence": 1.18, "step": 0.85, "asymmetry": 0.040,
                     "arm_swing": 0.70, "sway": 1.10, "stance_width": 0.96},

    # Wearing a coat: cadence x1.218 (z=+8.1), leg spread x0.985, and an
    # asymmetry change of -0.0001 (z=-0.1) across 124 subjects — statistically
    # indistinguishable from zero.
    #
    # That null result is the important one. It is direct evidence for the claim
    # gait ReID rests on: changing clothes alters appearance without altering
    # the *symmetry* of the walk. It also shows the two are separable, since a
    # bag changes gait while a coat does not, so "carrying something" and
    # "wearing something different" are genuinely different evasions.
    "heavy_coat":  {"cadence": 1.22, "step": 0.82, "asymmetry": 0.000,
                    "arm_swing": 0.80, "sway": 1.05, "stance_width": 0.99},
}

# How much each parameter varies between individuals walking the "same" way.
# These set the difficulty of gait ReID: too small and everyone is identical
# (matching is chance), too large and style is unrecognisable (matching is
# trivial). Loosely after the human-variation literature.
INDIVIDUAL_SPREAD = {
    "cadence_spm": 0.07,        # relative
    "step_length_m": 0.09,
    "asymmetry": 0.04,          # absolute
    "arm_swing": 0.15,
    "sway_amplitude_m": 0.20,
    "stance_width_m": 0.12,
}

# Baselines for a 1.70 m adult walking normally.
BASE_SWAY_M = 0.035
BASE_STANCE_WIDTH_M = 0.12

PARAMETERS = ("cadence_spm", "step_length_m", "asymmetry", "arm_swing",
              "sway_amplitude_m", "stance_width_m")


@dataclass(frozen=True)
class GaitSignature:
    """One person's measurable walking parameters."""

    cadence_spm: float
    step_length_m: float
    asymmetry: float            # signed; +ve favours the left leg
    arm_swing: float
    sway_amplitude_m: float
    stance_width_m: float
    style: str = "normal"

    @property
    def speed_mps(self) -> float:
        """Implied walking speed. Not stored: it is determined by the others.

        Storing it as well would let it drift out of step with cadence and
        stride, and then which of the three is right becomes unanswerable.
        """
        return self.cadence_spm / 60.0 * self.step_length_m

    def as_vector(self) -> np.ndarray:
        return np.array([getattr(self, p) for p in PARAMETERS], dtype=float)

    def as_dict(self) -> dict[str, Any]:
        return {**{p: round(float(getattr(self, p)), 4) for p in PARAMETERS},
                "style": self.style,
                "speed_mps": round(self.speed_mps, 3)}


def _person_offsets(person_id: str, spread: float = 1.0) -> dict[str, float]:
    """Stable per-person deviations, derived from the identity itself.

    Hashed rather than drawn from a shared RNG so a person's gait is the same
    whenever they appear — across cameras, across appearance epochs, and across
    runs. A gait biometric that changed between cameras would make cross-camera
    gait matching impossible for the wrong reason.
    """
    digest = hashlib.sha256(str(person_id).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return {name: float(rng.normal(0.0, sd * spread))
            for name, sd in INDIVIDUAL_SPREAD.items()}


def signature_for(person_id: str, style: str = "normal",
                  height_m: float = 1.70, density_per_m2: float = 0.0,
                  individuality: float = 1.0) -> GaitSignature:
    """Build a person's gait signature from who they are and how they walk.

    Height scales stride and sway: a taller person's legs are longer, so the
    same walk covers more ground. Density shortens the step while leaving
    cadence alone, which is what was measured on real crowded footage rather
    than what a speed-only model would predict.
    """
    effects = STYLE_EFFECTS.get(style, STYLE_EFFECTS["normal"])
    offsets = _person_offsets(person_id, individuality)
    stature_scale = height_m / 1.70

    cadence = FREE_CADENCE_SPM * effects["cadence"] * (1.0 + offsets["cadence_spm"])
    step = (FREE_STEP_LENGTH_M * effects["step"] * stature_scale
            * (1.0 + offsets["step_length_m"]))

    # Crowding: cadence holds (or rises slightly), the step absorbs the loss.
    # docs/14 section 10 — a slow crowd is not a slow-motion crowd.
    if density_per_m2 > 0.0:
        from survsim.director.crowd import gait_for_density
        crowd_cadence, crowd_step = gait_for_density(density_per_m2)
        cadence *= crowd_cadence / FREE_CADENCE_SPM
        step *= crowd_step / FREE_STEP_LENGTH_M

    return GaitSignature(
        cadence_spm=max(cadence, 1.0),
        step_length_m=max(step, 0.02),
        # Asymmetry is absolute: a normal walk has none, so a multiplicative
        # individual offset on zero would leave everyone perfectly symmetric.
        asymmetry=effects["asymmetry"] + offsets["asymmetry"],
        arm_swing=max(effects["arm_swing"] * (1.0 + offsets["arm_swing"]), 0.0),
        sway_amplitude_m=max(BASE_SWAY_M * effects["sway"] * stature_scale
                             * (1.0 + offsets["sway_amplitude_m"]), 0.0),
        stance_width_m=max(BASE_STANCE_WIDTH_M * effects["stance_width"]
                           * stature_scale * (1.0 + offsets["stance_width_m"]),
                           0.01),
        style=style,
    )


def population_spread(signatures: Sequence[GaitSignature]) -> np.ndarray:
    """Per-parameter standard deviation across a population.

    Distances must be normalised by how much a parameter varies between people,
    not by its magnitude. Cadence is ~110 and asymmetry ~0.04, so an unnormalised
    Euclidean distance is a cadence comparison with rounding noise attached.
    """
    if len(signatures) < 2:
        return np.ones(len(PARAMETERS))
    matrix = np.array([s.as_vector() for s in signatures])
    return np.maximum(matrix.std(axis=0), 1e-6)


def gait_distance(a: GaitSignature, b: GaitSignature,
                  spread: np.ndarray | None = None) -> float:
    """Distance between two gait signatures, in population standard deviations.

    The unit is deliberate: "1.0" means these two differ by as much as two
    random people typically do, which is interpretable without knowing anything
    about the parameters.
    """
    if spread is None:
        spread = np.ones(len(PARAMETERS))
    return float(np.sqrt(np.mean(((a.as_vector() - b.as_vector()) / spread) ** 2)))


def observe(signature: GaitSignature, noise: float = 0.15,
            seed: int = 0) -> GaitSignature:
    """A noisy measurement of a gait signature, as a camera would obtain it.

    Matching clean signatures against each other measures nothing: it asks
    whether two copies of the same numbers are equal. Real gait estimation from
    surveillance video is noisy, occluded and view-dependent, and a gait ReID
    score is only meaningful against an observation that carries that error.

    `noise` is the fraction of the *population* spread added per parameter, so
    0.15 means measurement error around a seventh of between-person variation.
    """
    rng = np.random.default_rng(seed)
    values = signature.as_vector()
    spread = np.array([
        FREE_CADENCE_SPM * INDIVIDUAL_SPREAD["cadence_spm"],
        FREE_STEP_LENGTH_M * INDIVIDUAL_SPREAD["step_length_m"],
        INDIVIDUAL_SPREAD["asymmetry"],
        INDIVIDUAL_SPREAD["arm_swing"],
        BASE_SWAY_M * INDIVIDUAL_SPREAD["sway_amplitude_m"],
        BASE_STANCE_WIDTH_M * INDIVIDUAL_SPREAD["stance_width_m"],
    ])
    noisy = values + rng.normal(0.0, noise * spread)
    return GaitSignature(
        cadence_spm=max(float(noisy[0]), 1.0),
        step_length_m=max(float(noisy[1]), 0.02),
        asymmetry=float(noisy[2]),
        arm_swing=max(float(noisy[3]), 0.0),
        sway_amplitude_m=max(float(noisy[4]), 0.0),
        stance_width_m=max(float(noisy[5]), 0.01),
        style=signature.style,
    )


def gait_similarity_matrix(query: Sequence[GaitSignature],
                           gallery: Sequence[GaitSignature],
                           spread: np.ndarray | None = None) -> np.ndarray:
    """Query-by-gallery similarity, ready for `eval.tracking.evaluate_reid`.

    Similarity is negated distance. Feeding this into the existing ReID scorer
    rather than writing a second one means gait is scored by exactly the same
    rank-1/rank-5/mAP definitions as appearance, so the two are comparable —
    which is the entire point of measuring gait at all.
    """
    if not len(query) or not len(gallery):
        return np.zeros((len(query), len(gallery)))
    if spread is None:
        spread = population_spread(list(query) + list(gallery))

    q = np.array([s.as_vector() for s in query]) / spread
    g = np.array([s.as_vector() for s in gallery]) / spread
    # Pairwise RMS difference, matching gait_distance's definition.
    diff = q[:, None, :] - g[None, :, :]
    return -np.sqrt(np.mean(diff ** 2, axis=-1))

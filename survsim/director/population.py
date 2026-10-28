"""Procedural people.

MOTSynth's result was that *diversity* matters more than volume for closing the
synthetic-to-real gap, so identities are sampled across skin tone, age, body and
garment colour rather than drawn from a handful of presets. Colour is sampled in
HSV — uniform RGB clusters around muddy mid-greys and produces a crowd that looks
nothing like a street.

Identities are also split deterministically by hash, so the same person preset can
never appear in both a train and a test split.
"""

from __future__ import annotations

import colorsys
import hashlib
from typing import Any

import numpy as np

from ..core.models import BehaviourSpec, Body, Gait, Outfit, Person

DEFAULT_POOL: dict[str, Any] = {
    "age_band": {"18-25": 0.25, "25-40": 0.35, "40-60": 0.25, "60+": 0.10, "child": 0.05},
    "sex_presentation": {"m": 0.48, "f": 0.48, "x": 0.04},
    "bmi_class": {"slim": 0.25, "normal": 0.45, "heavy": 0.20, "athletic": 0.10},
    "outfit": {
        "top": {"tshirt": 0.4, "shirt": 0.2, "polo": 0.1, "hoodie": 0.15, "tank": 0.15},
        "outer": {"none": 0.6, "light_jacket": 0.2, "rain_jacket": 0.1, "hi_vis_vest": 0.1},
        "bottom": {"jeans": 0.4, "shorts": 0.25, "chinos": 0.2, "skirt": 0.15},
        "shoes": {"trainers": 0.6, "boots": 0.15, "formal": 0.15, "sandals": 0.1},
        "headwear": {"none": 0.7, "cap": 0.2, "beanie": 0.05, "hood_up": 0.05},
    },
    "accessories": {
        "none": 0.35, "backpack_small": 0.2, "handbag": 0.15,
        "tote": 0.1, "suitcase": 0.05, "backpack_large": 0.05, "umbrella": 0.1,
    },
    "gait": {"normal": 0.75, "brisk": 0.12, "slow": 0.08, "limp_L": 0.025, "limp_R": 0.025},
}

# Mean stature by age band in metres, with the spread real populations show.
_HEIGHT = {
    "child": (1.30, 0.12), "18-25": (1.72, 0.09), "25-40": (1.72, 0.09),
    "40-60": (1.70, 0.09), "60+": (1.66, 0.09),
}


def _choice(rng: np.random.Generator, weights: dict[str, float]) -> str:
    keys = list(weights)
    probs = np.array([weights[k] for k in keys], dtype=float)
    return str(rng.choice(keys, p=probs / probs.sum()))


def random_colour(rng: np.random.Generator) -> str:
    """A plausible garment colour.

    Saturation and value are drawn from Beta distributions so most clothes are
    muted but a few are vivid, which is what a real crowd looks like.
    """
    h = float(rng.uniform(0, 1))
    s = float(rng.beta(2, 3))
    v = float(rng.beta(3, 2))
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def split_of(preset_id: str, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> str:
    """Assign an identity to train/val/test by hash so splits never leak."""
    digest = hashlib.blake2b(preset_id.encode(), digest_size=8).digest()
    x = int.from_bytes(digest, "big") / float(1 << 64)
    train, val, _ = ratios
    if x < train:
        return "train"
    return "val" if x < train + val else "test"


ANIMALS = {"fox": (0.35, 0.45), "dog": (0.40, 0.75), "badger": (0.25, 0.35),
           "deer_fawn": (0.60, 0.95), "cat": (0.22, 0.30)}


def random_animal(uid: str, rng: np.random.Generator) -> Person:
    """A wildlife distractor: non-person class, shoulder height < 1 m.

    `identity["class"] = "animal"` is what ground truth reads (Agent.object_class):
    MOT rows mark it consider=0/class=8 (distractor), world.parquet and
    identities.parquet carry object_class="animal".
    """
    species = str(rng.choice(sorted(ANIMALS)))
    lo, hi = ANIMALS[species]
    return Person(id=uid, role="wildlife", identity={"class": "animal", "species": species},
                  body=Body(height_m=float(rng.uniform(lo, hi)), shoulder_w=0.25),
                  gait=Gait(style="run", speed_mps=float(rng.uniform(1.5, 3.0))))


def random_person(
    uid: str,
    rng: np.random.Generator,
    pool: dict[str, Any] | None = None,
    role: str = "commuter",
) -> Person:
    pool = {**DEFAULT_POOL, **(pool or {})}
    outfit_pool = {**DEFAULT_POOL["outfit"], **pool.get("outfit", {})}

    age = _choice(rng, pool["age_band"])
    mean_h, sd_h = _HEIGHT.get(age, (1.70, 0.09))
    height = float(np.clip(rng.normal(mean_h, sd_h), 1.15, 2.05))

    def garment(slot: str) -> str | None:
        value = _choice(rng, outfit_pool[slot])
        return None if value == "none" else value

    outfit = Outfit(
        top=garment("top") or "tshirt",
        outer=garment("outer"),
        bottom=garment("bottom") or "jeans",
        shoes=garment("shoes") or "trainers",
        headwear=garment("headwear"),
        colours={
            "top": random_colour(rng),
            "outer": random_colour(rng),
            "bottom": random_colour(rng),
        },
    )

    accessory = _choice(rng, pool["accessories"])
    gait_style = _choice(rng, pool["gait"])
    # Taller people walk slightly faster; keep everyone inside the observed
    # 0.8-1.8 m/s band for ordinary walking.
    base_speed = 1.15 + 0.25 * (height - 1.55)
    speed = float(np.clip(rng.normal(base_speed, 0.12), 0.8, 1.8))
    if gait_style == "brisk":
        speed = float(min(1.8, speed * 1.25))
    elif gait_style in ("slow", "limp_L", "limp_R"):
        speed = float(max(0.6, speed * 0.75))

    return Person(
        id=uid,
        role=role,
        body=Body(
            height_m=round(height, 3),
            shoulder_w=round(float(np.clip(rng.normal(0.45, 0.035), 0.32, 0.58)), 3),
            bmi_class=_choice(rng, pool["bmi_class"]),  # type: ignore[arg-type]
            skin_tone=int(rng.integers(1, 11)),
            sex_presentation=_choice(rng, pool["sex_presentation"]),  # type: ignore[arg-type]
            age_band=age,
        ),
        outfit=outfit,
        accessories=[] if accessory == "none" else [accessory],
        gait=Gait(style=gait_style, speed_mps=round(speed, 3)),  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------- templates

TEMPLATES: dict[str, dict[str, Any]] = {
    "loiterer": {
        "role": "intruder",
        "intent": [
            {"type": "transit", "from": "$spawn", "to": "$zone"},
            {"type": "loiter", "zone": "$zone", "dur_s": "$dur_s", "activity": "pace"},
            {"type": "transit", "to": "$exit"},
        ],
    },
    # Deliberately identical in dwell time to `loiterer`. A rule that separates
    # them by duration alone is measuring nothing.
    "commute_wait_negative": {
        "role": "commuter",
        "hard_negative_for": ["loiter"],
        "intent": [
            {"type": "transit", "from": "$spawn", "to": "$zone"},
            {"type": "commute_wait", "zone": "$zone", "dur_s": "$dur_s", "activity": "idle"},
            {"type": "transit", "to": "$exit"},
        ],
    },
    "passer_by": {
        "role": "commuter",
        "intent": [
            {"type": "transit", "from": "$spawn", "to": "$zone"},
            {"type": "transit", "to": "$exit"},
        ],
    },
    "disguise_runner": {
        "role": "intruder",
        "intent": [
            {"type": "transit", "from": "$spawn", "to": "$zone"},
            {"type": "loiter", "zone": "$zone", "dur_s": "$dur_s", "activity": "scan"},
            {"type": "transit", "to": "$blindspot"},
            {"type": "outfit_change", "in_zone": "$blindspot",
             "delta": {"outer": "none", "headwear": "cap"}},
            {"type": "transit", "to": "$exit"},
        ],
    },
}


def expand_generator(
    entry: dict[str, Any],
    rng: np.random.Generator,
    env_points: dict[str, list[str]],
    index_offset: int = 0,
) -> list[Person]:
    """Turn one `population.generators` entry into concrete Person objects."""
    name = entry.get("template")
    if name not in TEMPLATES:
        raise KeyError(f"unknown template {name!r}; have {sorted(TEMPLATES)}")
    template = TEMPLATES[name]
    count = int(entry.get("count", 1))
    people: list[Person] = []

    for i in range(count):
        uid = entry.get("id_prefix", name) + f"_{index_offset + i:02d}"
        person = random_person(uid, rng, entry.get("pool"), role=template.get("role", "commuter"))
        person.hard_negative_for = list(template.get("hard_negative_for", []))

        bindings = {
            "$zone": _pick(entry.get("zones"), env_points.get("zones", []), rng),
            "$spawn": _pick(entry.get("spawns"), env_points.get("spawns", []), rng),
            "$exit": _pick(entry.get("exits"), env_points.get("exits", []), rng),
            "$blindspot": _pick(entry.get("blindspots"), env_points.get("blindspots", []), rng),
            "$dur_s": _sample_range(entry.get("dur_s", 120.0), rng),
        }
        person.intent = [
            BehaviourSpec.model_validate(
                {k: bindings.get(v, v) if isinstance(v, str) else v for k, v in step.items()}
            )
            for step in template["intent"]
        ]
        people.append(person)
    return people


def _pick(explicit: Any, fallback: list[str], rng: np.random.Generator) -> str | None:
    options = explicit if isinstance(explicit, list) and explicit else fallback
    return str(rng.choice(options)) if options else None


def _sample_range(spec: Any, rng: np.random.Generator) -> float:
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return float(rng.uniform(float(spec[0]), float(spec[1])))
    return float(spec)

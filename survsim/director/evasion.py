"""Evasion techniques, and the innocent people who look exactly like them.

Someone trying not to be recognised has a large, cheap toolkit: a covid mask, a
hood, sunglasses, a shaved head, a borrowed hi-vis vest, a deliberate limp, a
face turned away from the lens. This module enumerates that toolkit so scenarios
can recreate it, and records *which perception cue each technique actually
defeats* — a scarf destroys face recognition but leaves gait intact; a bulky coat
destroys body-shape matching but leaves the face; a wheelchair changes the
silhouette so completely that person detectors trained on upright pedestrians
degrade.

**The central design constraint: an attribute is not an intent.**

Every technique here has innocent users, usually far more of them than hostile
ones. People wear masks because they are ill, sunglasses because it is sunny,
hi-vis because it is their job, a niqab because of faith, and use a wheelchair
because they need one. A dataset where "wearing a mask" predicts "criminal"
teaches a model to be a bigot, and it will fail in deployment because the base
rate of hostile intent among mask-wearers is minuscule.

So each technique carries a `benign_rate` (how much more common the innocent use
is) and `benign_contexts` (why), and the generator below deliberately produces
matched pairs: the same appearance, opposite intent. `audit_confounds()` then
measures whether any attribute leaks the label, and a scenario that fails the
audit should be regenerated rather than shipped.

Evidence for the framing: research dossier R03 records that behavioural deception
cues perform at roughly chance (~54% across 400+ studies), and that the base rate
of genuine hostile intent makes attribute-driven suspicion overwhelmingly
false-positive. Intent must come from *behaviour over time in context*, never
from appearance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

# Perception cues an evasion technique can degrade.
CUES = ("face", "gait", "body_shape", "clothing", "silhouette", "height")


@dataclass(frozen=True)
class Technique:
    """One way to change how you are perceived."""

    id: str
    category: str
    label: str
    # How much each cue is degraded, 0 = untouched, 1 = destroyed.
    degrades: dict[str, float] = field(default_factory=dict)
    # Appearance delta applied to AppearanceState (and hence to the renderer).
    delta: dict[str, Any] = field(default_factory=dict)
    # Relative frequency of innocent vs hostile use. 20 means that for every
    # hostile user there are ~20 entirely innocent ones.
    benign_rate: float = 10.0
    benign_contexts: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    props: tuple[str, ...] = ()
    # Legally or ethically sensitive attributes must never be used as evidence.
    protected: bool = False

    def degrade(self, cue: str) -> float:
        return float(self.degrades.get(cue, 0.0))


def _t(*a, **k) -> Technique:
    return Technique(*a, **k)


# ------------------------------------------------------------ face occlusion

FACE = [
    _t("mask_surgical", "face_occlusion", "surgical/covid mask",
       {"face": 0.85}, {"face": {"mask": True}}, benign_rate=200.0,
       benign_contexts=("illness", "air quality", "post-pandemic habit", "clinical work"),
       conflicts=("scarf_face", "balaclava", "niqab", "respirator")),
    _t("respirator", "face_occlusion", "FFP3 / dust respirator",
       {"face": 0.9}, {"face": {"mask": True}}, benign_rate=60.0,
       benign_contexts=("construction", "spraying", "dust"), conflicts=("mask_surgical",)),
    _t("scarf_face", "face_occlusion", "scarf pulled over the face",
       {"face": 0.8}, {"face": {"mask": True}, "accessories": ["scarf"]}, benign_rate=40.0,
       benign_contexts=("cold weather", "wind", "fashion"), conflicts=("mask_surgical",)),
    _t("balaclava", "face_occlusion", "balaclava",
       {"face": 0.97, "clothing": 0.2}, {"headwear": "balaclava", "face": {"balaclava": True}},
       benign_rate=3.0, benign_contexts=("skiing", "motorcycling", "extreme cold"),
       conflicts=("mask_surgical", "niqab", "helmet_motorcycle")),
    _t("niqab", "face_occlusion", "niqab / face veil",
       {"face": 0.95, "body_shape": 0.6, "clothing": 0.5},
       {"headwear": "niqab", "face": {"mask": True}}, benign_rate=5000.0,
       benign_contexts=("religious observance — overwhelmingly the only reason",),
       conflicts=("mask_surgical", "balaclava"), protected=True),
    _t("burqa", "face_occlusion", "burqa",
       {"face": 0.97, "body_shape": 0.85, "gait": 0.4, "clothing": 0.7, "silhouette": 0.6},
       {"outer": "burqa", "face": {"mask": True}}, benign_rate=5000.0,
       benign_contexts=("religious observance — overwhelmingly the only reason",),
       conflicts=("mask_surgical", "balaclava", "niqab"), protected=True),
    _t("sunglasses", "face_occlusion", "sunglasses",
       {"face": 0.45}, {"face": {"glasses": "sun"}}, benign_rate=300.0,
       benign_contexts=("sunshine", "glare", "light sensitivity", "fashion")),
    _t("goggles", "face_occlusion", "goggles",
       {"face": 0.6}, {"face": {"glasses": "goggles"}}, benign_rate=50.0,
       benign_contexts=("industrial work", "swimming", "skiing", "cycling")),
    _t("clear_glasses", "face_occlusion", "prescription glasses",
       {"face": 0.15}, {"face": {"glasses": "clear"}}, benign_rate=2000.0,
       benign_contexts=("vision correction",)),
    _t("hood_up", "face_occlusion", "hood raised",
       {"face": 0.6, "silhouette": 0.2}, {"headwear": "hood_up"}, benign_rate=150.0,
       benign_contexts=("rain", "cold", "fashion"), conflicts=("helmet_motorcycle",)),
    _t("cap_low", "face_occlusion", "cap pulled low",
       {"face": 0.5}, {"headwear": "cap"}, benign_rate=250.0,
       benign_contexts=("sun", "rain", "habit"), conflicts=("hood_up", "helmet_motorcycle")),
    _t("helmet_motorcycle", "face_occlusion", "full-face motorcycle helmet",
       {"face": 0.98, "silhouette": 0.2}, {"headwear": "helmet"}, benign_rate=400.0,
       benign_contexts=("riding a motorcycle — legally required",),
       conflicts=("cap_low", "hood_up", "balaclava")),
    _t("umbrella_shield", "face_occlusion", "umbrella angled at the camera",
       {"face": 0.8, "silhouette": 0.3}, {"accessories": ["umbrella"]}, benign_rate=100.0,
       benign_contexts=("rain", "strong sun"), props=("umbrella",)),
    _t("hand_over_face", "face_occlusion", "hand raised to the face",
       {"face": 0.7}, {}, benign_rate=80.0,
       benign_contexts=("phone call", "sneeze", "yawn", "sun in eyes")),
]

# ------------------------------------------------------------ head and hair

HEAD = [
    _t("shave_head", "appearance_change", "shaved head",
       {"face": 0.2}, {"hair": {"style": "bald", "colour": "none"}}, benign_rate=500.0,
       benign_contexts=("style", "baldness", "chemotherapy", "military grooming")),
    _t("shave_beard", "appearance_change", "beard removed",
       {"face": 0.4}, {"face": {"beard": "none"}}, benign_rate=800.0,
       benign_contexts=("routine grooming",)),
    _t("grow_beard", "appearance_change", "full beard",
       {"face": 0.35}, {"face": {"beard": "full"}}, benign_rate=800.0,
       benign_contexts=("routine grooming", "religious observance"), protected=True),
    _t("wig", "appearance_change", "wig / changed hairstyle",
       {"face": 0.3}, {"hair": {"style": "wig"}}, benign_rate=30.0,
       benign_contexts=("medical hair loss", "fashion", "performance")),
    _t("hair_dye", "appearance_change", "hair colour changed",
       {"face": 0.15}, {"hair": {"colour": "dyed"}}, benign_rate=900.0,
       benign_contexts=("fashion",)),
    _t("hijab", "appearance_change", "hijab / headscarf",
       {"face": 0.2, "clothing": 0.3}, {"headwear": "hijab"}, benign_rate=5000.0,
       benign_contexts=("religious observance — overwhelmingly the only reason",),
       protected=True),
    _t("turban", "appearance_change", "turban",
       {"face": 0.15, "clothing": 0.3}, {"headwear": "turban"}, benign_rate=5000.0,
       benign_contexts=("religious observance — overwhelmingly the only reason",),
       protected=True),
    _t("presentation_change", "appearance_change", "changed gender presentation",
       {"face": 0.5, "body_shape": 0.4, "clothing": 0.8, "gait": 0.3},
       {"outfit_style": "alternate"}, benign_rate=400.0,
       benign_contexts=("gender identity and expression — a protected characteristic, "
                        "never an indicator",), protected=True),
]

# ------------------------------------------------------------------ silhouette

BODY = [
    _t("bulky_coat", "silhouette", "oversized padded coat",
       {"body_shape": 0.8, "silhouette": 0.5, "clothing": 0.4}, {"outer": "bulky_coat"},
       benign_rate=300.0, benign_contexts=("cold weather",)),
    _t("layered_removal", "appearance_change", "removes an outer layer mid-route",
       {"clothing": 0.9, "body_shape": 0.3}, {"outer": "none"}, benign_rate=60.0,
       benign_contexts=("warming up", "entering a building")),
    _t("reversible_jacket", "appearance_change", "reversible jacket turned inside out",
       {"clothing": 0.95}, {"colours": {"outer": "#2b2b2b"}}, benign_rate=2.0,
       benign_contexts=("genuinely rare innocently",)),
    _t("bag_front", "silhouette", "backpack worn on the chest",
       {"body_shape": 0.5, "silhouette": 0.4}, {"accessories": ["backpack_front"]},
       benign_rate=120.0, benign_contexts=("theft-avoidance on transit", "cycling")),
    _t("bag_swap", "appearance_change", "changes or discards a bag",
       {"silhouette": 0.4, "clothing": 0.2}, {"accessories": []}, benign_rate=50.0,
       benign_contexts=("shopping", "handing luggage to a companion")),
    _t("heeled_shoes", "silhouette", "raised footwear",
       {"height": 0.5, "gait": 0.3}, {"shoes": "heeled"}, benign_rate=600.0,
       benign_contexts=("everyday footwear",)),
]

# ----------------------------------------------------------------------- gait

GAIT = [
    _t("fake_limp", "gait", "deliberate limp",
       {"gait": 0.85}, {"gait": {"style": "limp_L"}}, benign_rate=25.0,
       benign_contexts=("genuine injury", "chronic condition")),
    _t("slow_deliberate", "gait", "unnaturally slow walk",
       {"gait": 0.5}, {"gait": {"style": "slow"}}, benign_rate=200.0,
       benign_contexts=("age", "fatigue", "browsing", "carrying something heavy")),
    _t("brisk_march", "gait", "fast purposeful stride",
       {"gait": 0.5}, {"gait": {"style": "brisk"}}, benign_rate=400.0,
       benign_contexts=("running late", "fitness")),
    _t("shuffle", "gait", "shuffling step",
       {"gait": 0.7}, {"gait": {"style": "shuffle"}}, benign_rate=150.0,
       benign_contexts=("age", "illness", "ill-fitting shoes")),
    _t("weighted_asymmetry", "gait", "heavy bag on one side alters the gait",
       {"gait": 0.6, "silhouette": 0.3}, {"accessories": ["suitcase"]}, benign_rate=500.0,
       benign_contexts=("luggage", "shopping", "tools")),
    _t("crouch_walk", "gait", "crouched movement",
       {"gait": 0.9, "height": 0.7, "silhouette": 0.6}, {"gait": {"style": "crouch"}},
       benign_rate=2.0, benign_contexts=("searching for something dropped", "gardening")),
]

# --------------------------------------------------------------- role disguise

ROLE = [
    _t("hi_vis_worker", "role_disguise", "hi-vis vest and hard hat",
       {"clothing": 0.6}, {"outer": "hi_vis_vest", "headwear": "hardhat"}, benign_rate=800.0,
       benign_contexts=("actual maintenance", "construction", "marshalling"),
       props=("clipboard",)),
    _t("plumber_kit", "role_disguise", "workwear plus tool bag",
       {"clothing": 0.5}, {"outer": "workwear", "accessories": ["tool_bag"]}, benign_rate=600.0,
       benign_contexts=("actual trades work",), props=("tool_bag",)),
    _t("carrying_ladder", "role_disguise", "carrying a ladder",
       {"silhouette": 0.6}, {"accessories": ["ladder"]}, benign_rate=400.0,
       benign_contexts=("window cleaning", "maintenance", "decorating"), props=("ladder",)),
    _t("courier", "role_disguise", "courier uniform and parcel",
       {"clothing": 0.55}, {"outer": "courier_jacket", "accessories": ["parcel"]},
       benign_rate=900.0, benign_contexts=("actual delivery",), props=("parcel",)),
    _t("cleaner", "role_disguise", "cleaning tabard and trolley",
       {"clothing": 0.5}, {"outer": "tabard", "accessories": ["cleaning_trolley"]},
       benign_rate=700.0, benign_contexts=("actual cleaning shift",)),
    _t("security_uniform", "role_disguise", "security guard uniform",
       {"clothing": 0.6}, {"outer": "security_jacket"}, benign_rate=300.0,
       benign_contexts=("actual guard",)),
    # Impersonating police is a criminal offence in itself, so the benign rate is
    # genuinely near zero. This is one of the few attributes that is evidential.
    _t("police_uniform", "role_disguise", "police uniform",
       {"clothing": 0.7}, {"outer": "police_jacket"}, benign_rate=0.02,
       benign_contexts=("a real officer",)),
    _t("paramedic", "role_disguise", "paramedic uniform",
       {"clothing": 0.6}, {"outer": "paramedic_jacket"}, benign_rate=0.2,
       benign_contexts=("a real paramedic",)),
]

# -------------------------------------------------------------- mobility aids

MOBILITY = [
    _t("wheelchair", "mobility_aid", "wheelchair user",
       {"gait": 1.0, "height": 0.9, "silhouette": 0.9, "body_shape": 0.6},
       {"mobility": "wheelchair"}, benign_rate=5000.0,
       benign_contexts=("disability — overwhelmingly the only reason",),
       conflicts=("fake_limp", "crouch_walk", "brisk_march"), protected=True),
    _t("crutches", "mobility_aid", "crutches",
       {"gait": 0.9, "silhouette": 0.4}, {"mobility": "crutches"}, benign_rate=2000.0,
       benign_contexts=("injury", "disability"), conflicts=("fake_limp",), protected=True),
    _t("walking_stick", "mobility_aid", "walking stick",
       {"gait": 0.6}, {"mobility": "stick"}, benign_rate=3000.0,
       benign_contexts=("age", "disability"), protected=True),
    _t("pushchair", "mobility_aid", "pushing a pram",
       {"gait": 0.5, "silhouette": 0.6}, {"accessories": ["pushchair"]}, benign_rate=4000.0,
       benign_contexts=("with a child",)),
]

# --------------------------------------------------- camera-aware behaviours

CAMERA = [
    _t("face_away", "camera_behaviour", "turns away when facing a camera",
       {"face": 0.9}, {}, benign_rate=20.0,
       benign_contexts=("looking at a companion", "checking a display behind them")),
    _t("head_down", "camera_behaviour", "walks with head down",
       {"face": 0.75}, {}, benign_rate=300.0,
       benign_contexts=("looking at a phone", "rain", "shyness")),
    _t("wall_hugging", "camera_behaviour", "keeps to walls and awnings",
       {"face": 0.3, "silhouette": 0.2}, {}, benign_rate=40.0,
       benign_contexts=("sheltering from rain", "avoiding crowds")),
    _t("crowd_cover", "camera_behaviour", "moves only inside dense groups",
       {"face": 0.5, "silhouette": 0.5}, {}, benign_rate=60.0,
       benign_contexts=("simply walking in a busy place",)),
    _t("ptz_timing", "camera_behaviour", "moves while the PTZ is pointed elsewhere",
       {"face": 0.6, "silhouette": 0.6}, {}, benign_rate=0.05,
       benign_contexts=("coincidence only — requires knowledge of the patrol",)),
    _t("blindspot_change", "camera_behaviour", "changes appearance in a blind spot",
       {"clothing": 0.95, "face": 0.3}, {}, benign_rate=0.1,
       benign_contexts=("coincidence only",)),
]

ALL: list[Technique] = FACE + HEAD + BODY + GAIT + ROLE + MOBILITY + CAMERA
BY_ID: dict[str, Technique] = {t.id: t for t in ALL}
BY_CATEGORY: dict[str, list[Technique]] = {}
for _t_ in ALL:
    BY_CATEGORY.setdefault(_t_.category, []).append(_t_)


# --------------------------------------------------------------- combinations


def compatible(ids: Iterable[str]) -> bool:
    """True if no chosen technique conflicts with another."""
    chosen = list(ids)
    for i in chosen:
        t = BY_ID[i]
        if any(c in chosen for c in t.conflicts):
            return False
    return True


def residual_cues(ids: Iterable[str]) -> dict[str, float]:
    """How much of each perception cue survives the chosen techniques.

    Degradations compose multiplicatively — two partial face occlusions leave less
    than either alone, but nothing ever becomes negative. The result tells you
    which identification route is still open, which is the whole point: if `face`
    is 0.05 but `gait` is 0.9, gait recognition is the answer, not a better face
    model.
    """
    out = {c: 1.0 for c in CUES}
    for i in ids:
        for cue, amount in BY_ID[i].degrades.items():
            out[cue] *= max(0.0, 1.0 - float(amount))
    return {k: round(v, 4) for k, v in out.items()}


def sample_combination(
    rng: np.random.Generator,
    n_techniques: int = 2,
    categories: Iterable[str] | None = None,
    allow_protected: bool = True,
    max_attempts: int = 200,
) -> list[str]:
    """Draw a compatible set of techniques."""
    pool = [t for t in ALL
            if (categories is None or t.category in set(categories))
            and (allow_protected or not t.protected)]
    if not pool:
        return []
    for _ in range(max_attempts):
        pick = list(rng.choice([t.id for t in pool],
                               size=min(n_techniques, len(pool)), replace=False))
        if compatible(pick):
            return pick
    return [str(rng.choice([t.id for t in pool]))]


@dataclass
class Persona:
    """A generated person: their disguise, their intent, and why they look that way."""

    uid: str
    techniques: list[str]
    intent: str                       # "benign" | "hostile"
    rationale: str                    # why an innocent person would look like this
    residual: dict[str, float]
    protected_attributes: list[str]

    def delta(self) -> dict[str, Any]:
        """Merged appearance delta for AppearanceState."""
        out: dict[str, Any] = {}
        for i in self.techniques:
            for k, v in BY_ID[i].delta.items():
                if isinstance(v, dict):
                    out[k] = {**out.get(k, {}), **v}
                elif isinstance(v, list):
                    out[k] = list(out.get(k, [])) + v
                else:
                    out[k] = v
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "techniques": self.techniques,
            "intent": self.intent,
            "rationale": self.rationale,
            "residual_cues": self.residual,
            "protected_attributes": self.protected_attributes,
            "appearance_delta": self.delta(),
        }


def generate_matched_population(
    n_pairs: int = 20,
    seed: int = 0,
    n_techniques: int = 2,
    allow_protected: bool = True,
) -> list[Persona]:
    """Generate matched pairs: identical appearance, opposite intent.

    Pairing is the mechanism that stops a model learning "mask ⇒ criminal". For
    every hostile persona there is a benign persona wearing exactly the same
    things, so appearance carries no information about the label and the model is
    forced to learn from behaviour instead.
    """
    rng = np.random.default_rng(seed)
    people: list[Persona] = []
    for i in range(n_pairs):
        ids = sample_combination(rng, n_techniques, allow_protected=allow_protected)
        residual = residual_cues(ids)
        protected = [x for x in ids if BY_ID[x].protected]
        reasons = [c for x in ids for c in BY_ID[x].benign_contexts]
        rationale = "; ".join(dict.fromkeys(reasons)) or "ordinary daily choice"
        for intent in ("benign", "hostile"):
            people.append(Persona(
                uid=f"{intent[0]}{i:03d}",
                techniques=list(ids),
                intent=intent,
                rationale=rationale,
                residual=residual,
                protected_attributes=protected,
            ))
    return people


# -------------------------------------------------------------------- audit


def audit_confounds(people: list[Persona]) -> dict[str, Any]:
    """Check that appearance does not predict intent in a generated population.

    For each technique, compare how often it appears among hostile vs benign
    personas. A balanced set gives a ratio near 1.0. Anything far from 1.0 is a
    shortcut the model will find instead of learning behaviour, and the set should
    be regenerated. Protected attributes are reported separately because any
    imbalance there is not merely a modelling bug.
    """
    hostile = [p for p in people if p.intent == "hostile"]
    benign = [p for p in people if p.intent == "benign"]
    rows: dict[str, dict[str, float]] = {}
    worst = 1.0
    for t in ALL:
        h = sum(t.id in p.techniques for p in hostile)
        b = sum(t.id in p.techniques for p in benign)
        if h == 0 and b == 0:
            continue
        ratio = (h + 0.5) / (b + 0.5)
        rows[t.id] = {"hostile": h, "benign": b, "ratio": round(ratio, 3),
                      "protected": t.protected}
        worst = max(worst, ratio, 1 / ratio)

    protected_imbalance = {
        k: v for k, v in rows.items() if v["protected"] and abs(v["ratio"] - 1.0) > 0.25
    }
    return {
        "n_hostile": len(hostile),
        "n_benign": len(benign),
        "per_technique": rows,
        "worst_imbalance": round(worst, 3),
        "balanced": worst <= 1.5,
        "protected_imbalance": protected_imbalance,
        "protected_clean": not protected_imbalance,
    }

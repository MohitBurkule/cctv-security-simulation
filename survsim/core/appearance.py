"""Appearance state and identity epochs.

A person's *identity* is permanent and hidden; their *appearance* is not. When
someone removes a jacket in a blind spot, every appearance-based re-identification
method should fail, and the ground truth has to say so — otherwise a dataset
silently scores clothes matching as identity matching.

An **appearance epoch** is a maximal time span over which a person's appearance
hash is constant. Recording epochs lets evaluation separate two very different
questions: can the system re-identify someone *within* an epoch (easy, appearance
works), and *across* one (hard, needs shape, gait or face).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .models import Outfit, Person

# Attributes that survive a change of clothes. Anything outside this set is fair
# game for a disguise and must not be relied on for long-term identity.
SOFT_BIOMETRIC_KEYS = ("height_m", "shoulder_w", "bmi_class", "skin_tone", "age_band")


@dataclass
class AppearanceState:
    """Mutable appearance of one person, plus the epoch counter."""

    outfit: Outfit
    accessories: list[str] = field(default_factory=list)
    hair: dict[str, Any] = field(default_factory=dict)
    face: dict[str, Any] = field(default_factory=dict)
    gait_style: str = "normal"
    epoch: int = 0
    epoch_started_s: float = 0.0

    @classmethod
    def from_person(cls, person: Person) -> AppearanceState:
        return cls(
            outfit=person.outfit.model_copy(deep=True),
            accessories=list(person.accessories),
            hair=dict(person.hair),
            face=dict(person.face),
            gait_style=person.gait.style,
        )

    # ------------------------------------------------------------------ hash

    def snapshot(self) -> dict[str, Any]:
        return {
            "outfit": self.outfit.model_dump(),
            "accessories": sorted(self.accessories),
            "hair": self.hair,
            "face": self.face,
            "gait_style": self.gait_style,
        }

    @property
    def hash(self) -> str:
        blob = json.dumps(self.snapshot(), sort_keys=True, default=str)
        return hashlib.blake2b(blob.encode(), digest_size=8).hexdigest()

    @property
    def face_visible(self) -> bool:
        """False when the face is deliberately covered — a mask, balaclava or hood."""
        if self.face.get("mask") or self.face.get("balaclava"):
            return False
        headwear = (self.outfit.headwear or "").lower()
        return "balaclava" not in headwear and "hood" not in headwear

    @property
    def dominant_colour(self) -> str | None:
        """Colour an operator would broadcast: outer layer if worn, else the top."""
        colours = self.outfit.colours
        if self.outfit.outer:
            return colours.get("outer") or colours.get("top")
        return colours.get("top")

    # ----------------------------------------------------------------- apply

    def apply(self, delta: dict[str, Any], t: float) -> bool:
        """Apply a change; return True if it actually altered the appearance.

        A delta value of None or the string "none" removes a garment, which is how
        "takes the jacket off" is expressed in scenario YAML.
        """
        before = self.hash

        for slot in ("top", "outer", "bottom", "shoes", "headwear"):
            if slot in delta:
                value = delta[slot]
                setattr(self.outfit, slot, None if value in (None, "none") else value)

        if "colours" in delta:
            self.outfit.colours = {**self.outfit.colours, **(delta["colours"] or {})}
        if "accessories" in delta:
            self.accessories = [a for a in (delta["accessories"] or []) if a not in (None, "none")]
        if "hair" in delta:
            self.hair = {**self.hair, **(delta["hair"] or {})}
        if "face" in delta:
            self.face = {**self.face, **(delta["face"] or {})}
        if "gait" in delta and isinstance(delta["gait"], dict):
            self.gait_style = delta["gait"].get("style", self.gait_style)

        changed = self.hash != before
        if changed:
            self.epoch += 1
            self.epoch_started_s = t
        return changed


def soft_biometrics(person: Person, include_gait: bool = True) -> dict[str, Any]:
    """The identity cues a change of clothes cannot alter.

    Gait is included, and it is the most important entry here — it is the cue
    surveillance actually uses when clothing changes, and it was missing while
    this function returned body measurements alone. Body shape survives a
    disguise but is hard to measure at range; gait survives it too and is
    measurable from the motion a camera already records.

    The crucial difference from the body entries: gait is **the one soft
    biometric a person can deliberately change**. Somebody who adopts a limp
    alters it (`director/evasion.fake_limp`), and that is a real degradation
    rather than an annotation — see `gait/signature.py`. So it is recorded
    separately rather than blended into a single score, letting evaluation ask
    how much identification depended on a cue an evader controls.
    """
    body = person.body.model_dump()
    out: dict[str, Any] = {k: body[k] for k in SOFT_BIOMETRIC_KEYS if k in body}
    if include_gait:
        from ..gait.signature import signature_for
        signature = signature_for(person.id, style=person.gait.style,
                                  height_m=person.body.height_m)
        # Flattened with a prefix rather than nested. These rows become parquet
        # columns, and a consumer building a ReID feature matrix wants columns,
        # not a struct it has to unpack — nested fields also depend on struct
        # support that not every parquet reader offers.
        out.update({f"gait_{k}": v for k, v in signature.as_dict().items()})
    return out

"""Drive MetaHuman appearance and gait over UnrealCV+.

This is what turns the Director's *specification* of a varied crowd into a varied
crowd on screen. Without it every spawned actor is the same mesh in the same
clothes, and a detector trained on that learns "person = one particular outfit".

Three capabilities, each probed at runtime because a given build may not have
them:

* `metahuman/parametric/body`     — height and build
* `metahuman/parametric/wardrobe` — clothing per slot
* `animation/soma_glb/apply`      — mocap clips, which is how gait style is set

Gait deserves a note. It is *not* a speed parameter: a limp, a shuffle and a
brisk march differ in cadence, stride asymmetry and arm swing, and those live in
the motion clip. Driving "walk faster" instead produces the same animation played
quicker, which reads as wrong to a human and, more importantly, gives a gait
recognition model nothing real to learn from.

Every call degrades gracefully: if the server does not support a command, the
capability is recorded as unavailable and the run continues with a documented
limitation rather than failing or — worse — silently pretending the crowd is
diverse when it is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...core.appearance import AppearanceState
from ...core.models import Person

# Gait style -> mocap clip. Names follow the SmartLocomotion/SOMA convention; the
# resolver falls back through the list until the build accepts one.
GAIT_CLIPS: dict[str, tuple[str, ...]] = {
    "normal": ("walk_neutral", "Walk_N", "walk"),
    "brisk": ("walk_fast", "Walk_Fast", "jog"),
    "slow": ("walk_slow", "Walk_Slow", "walk"),
    "limp_L": ("walk_limp_left", "Limp_L", "walk_injured"),
    "limp_R": ("walk_limp_right", "Limp_R", "walk_injured"),
    "shuffle": ("walk_shuffle", "Shuffle", "walk_slow"),
    "drunk": ("walk_drunk", "Stagger", "walk_shuffle"),
    "military": ("walk_march", "March", "walk_fast"),
    "crouch": ("crouch_walk", "Crouch_Walk", "crouch"),
    "run": ("run", "Run", "jog"),
}

# Stationary behaviours. A loiterer who simply stops is the single most important
# case for surveillance analytics, because motion-based detection loses them.
IDLE_CLIPS: dict[str, tuple[str, ...]] = {
    "idle": ("idle", "Idle", "stand"),
    "scan": ("idle_look_around", "LookAround", "idle"),
    "phone": ("idle_phone", "Phone", "idle"),
    "wait": ("idle_wait", "Idle_Wait", "idle"),
}

WARDROBE_SLOTS = ("top", "outer", "bottom", "shoes", "headwear")


@dataclass
class Capabilities:
    """What this particular build actually supports.

    `animation` means *functionally verified*, not merely present in the command
    list. That distinction cost real time here: `animation/soma_glb/apply` exists
    on the v3.1.0 build and accepts the syntax, but rejects every clip name
    (0 of 40 tried) and reports `attached=false playing=false duration=0.000` —
    it wants a GLB asset that the build does not ship. A presence check would have
    reported gait control as working while every person walked identically.
    """

    body: bool = False
    wardrobe: bool = False
    animation: bool = False            # verified to actually play something
    animation_command_present: bool = False
    navigation: bool = False
    probed: bool = False
    verified: bool = False             # was a functional check run at all?
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"body": self.body, "wardrobe": self.wardrobe, "animation": self.animation,
                "animation_command_present": self.animation_command_present,
                "navigation": self.navigation, "verified": self.verified,
                "notes": self.notes}

    @property
    def diverse_appearance(self) -> bool:
        return self.body or self.wardrobe


def _ok(response: Any) -> bool:
    text = str(response).strip().lower()
    return bool(text) and "error" not in text and "not supported" not in text


class AppearanceDriver:
    """Applies appearance, gait and navigation to spawned actors."""

    def __init__(self, request) -> None:
        self._req = request
        self.caps = Capabilities()
        self._clip_cache: dict[str, str | None] = {}

    # ----------------------------------------------------------------- probe

    def probe(self, sample_actor: str | None = None) -> Capabilities:
        """Ask the server which commands exist, rather than assuming."""
        listing = str(self._req("vget /unrealcv/list_cmd", timeout=45))
        self.caps.body = "/metahuman/parametric/body" in listing
        self.caps.wardrobe = "/metahuman/parametric/wardrobe" in listing
        self.caps.animation_command_present = "/animation/soma_glb/apply" in listing
        self.caps.navigation = "/nav/goto" in listing or "/agent/" in listing
        self.caps.probed = True

        # Presence is not function. Only a successful play-and-check proves the
        # build can actually animate; without an actor to test on, assume it
        # cannot rather than silently producing identical walkers.
        if sample_actor is not None and self.caps.animation_command_present:
            self.caps.animation = self.verify_animation(sample_actor)
            self.caps.verified = True
        else:
            self.caps.animation = False
            if self.caps.animation_command_present:
                self.caps.notes.append(
                    "animation command present but unverified: pass a spawned actor to "
                    "probe() to confirm it plays, or gait will be silently uniform")

        for name, present in (("body", self.caps.body), ("wardrobe", self.caps.wardrobe),
                              ("animation", self.caps.animation),
                              ("navigation", self.caps.navigation)):
            if not present:
                self.caps.notes.append(f"{name} control unavailable in this build")
        if not self.caps.diverse_appearance:
            self.caps.notes.append(
                "no parametric appearance: every actor will render identically, so this "
                "run is unsuitable for training appearance-based models"
            )
        return self.caps

    def verify_animation(self, actor: str) -> bool:
        """Apply a clip and confirm the server reports it actually playing.

        On UnrealZoo v3.1.0 this returns False: every candidate name is rejected
        because the command expects a GLB asset the build does not ship. Supplying
        our own GLB mocap files is the route to real gait variation.
        """
        for names in GAIT_CLIPS.values():
            for name in names:
                if not _ok(self._req(f"vset /animation/soma_glb/apply {actor} {name}",
                                     timeout=45)):
                    continue
                status = str(self._req(f"vget /animation/soma_glb/status {actor}", timeout=30))
                if "playing=true" in status.lower() or "attached=true" in status.lower():
                    self._clip_cache[names[0]] = name
                    return True
        self.caps.notes.append(
            "animation verified non-functional: no clip name is accepted and status "
            "reports nothing attached, so every person will walk identically")
        return False

    # ------------------------------------------------------------ appearance

    def apply_body(self, actor: str, person: Person) -> bool:
        """Height and build. Height is the one identity cue clothing cannot hide."""
        if not self.caps.body:
            return False
        body = person.body
        # Normalise stature into a -1..1 shaping parameter around a 1.70 m mean.
        height_param = max(-1.0, min(1.0, (body.height_m - 1.70) / 0.25))
        build = {"slim": -0.6, "normal": 0.0, "heavy": 0.7, "athletic": 0.35}.get(
            body.bmi_class, 0.0)
        preset = "female" if body.sex_presentation == "f" else "male"
        return _ok(self._req(
            f"vset /metahuman/parametric/body {actor} {preset} "
            f"{height_param:.3f} {build:.3f} {body.skin_tone / 10.0:.3f}", timeout=45))

    def apply_wardrobe(self, actor: str, appearance: AppearanceState) -> dict[str, bool]:
        """Clothing, slot by slot. An empty slot removes the garment."""
        if not self.caps.wardrobe:
            return {}
        results: dict[str, bool] = {}
        outfit = appearance.outfit
        for slot in WARDROBE_SLOTS:
            item = getattr(outfit, slot, None)
            value = item if item else "none"
            results[slot] = _ok(self._req(
                f"vset /metahuman/parametric/wardrobe {actor} {slot} {value}", timeout=45))
        colour = appearance.dominant_colour
        if colour:
            results["colour"] = _ok(self._req(
                f"vset /metahuman/parametric/wardrobe {actor} colour {colour.lstrip('#')}",
                timeout=45))
        return results

    def assemble(self, actor: str) -> bool:
        """Rebuild the character after body/wardrobe edits."""
        if not self.caps.diverse_appearance:
            return False
        return _ok(self._req(f"vset /metahuman/parametric/assemble {actor} rig lod0", timeout=90))

    def apply_appearance(self, actor: str, person: Person, appearance: AppearanceState) -> dict:
        """Full appearance pass for one actor."""
        out = {"body": self.apply_body(actor, person),
               "wardrobe": self.apply_wardrobe(actor, appearance)}
        out["assembled"] = self.assemble(actor)
        return out

    # ------------------------------------------------------------------ gait

    def _resolve_clip(self, actor: str, candidates: tuple[str, ...]) -> str | None:
        """Try clip names until one is accepted; remember the answer."""
        key = candidates[0]
        if key in self._clip_cache:
            return self._clip_cache[key]
        for name in candidates:
            if _ok(self._req(f"vset /animation/soma_glb/apply {actor} {name}", timeout=45)):
                self._clip_cache[key] = name
                return name
        self._clip_cache[key] = None
        return None

    def apply_gait(self, actor: str, gait_style: str) -> str | None:
        """Set the walking style by choosing a motion clip, not a speed."""
        if not self.caps.animation:
            return None
        return self._resolve_clip(actor, GAIT_CLIPS.get(gait_style, GAIT_CLIPS["normal"]))

    def apply_idle(self, actor: str, activity: str = "idle") -> str | None:
        """Stationary behaviour — loitering, scanning, waiting."""
        if not self.caps.animation:
            return None
        return self._resolve_clip(actor, IDLE_CLIPS.get(activity, IDLE_CLIPS["idle"]))

    def stop_animation(self, actor: str) -> bool:
        if not self.caps.animation:
            return False
        return _ok(self._req(f"vset /animation/soma_glb/stop {actor}", timeout=30))

    # ------------------------------------------------------------ navigation

    def walk_to(self, actor: str, x_cm: float, y_cm: float, z_cm: float) -> bool:
        """Walk the navmesh to a point, letting the engine animate the locomotion.

        Strongly preferred over setting the location every tick: teleporting slides
        the character with no leg motion, which looks wrong and destroys any
        gait signal the footage might otherwise carry.
        """
        if not self.caps.navigation:
            return False
        return _ok(self._req(
            f"vset /agent/{actor}/nav/goto {x_cm:.1f} {y_cm:.1f} {z_cm:.1f}", timeout=45))

    def stop_walking(self, actor: str) -> bool:
        if not self.caps.navigation:
            return False
        return _ok(self._req(f"vset /agent/{actor}/nav/stop", timeout=30))

    def is_walking(self, actor: str) -> bool:
        if not self.caps.navigation:
            return False
        return "moving" in str(self._req(f"vget /agent/{actor}/nav/status", timeout=30)).lower()

    def look_at_yaw(self, actor: str, yaw_deg: float) -> bool:
        """Turn the head — how 'looking around while standing still' is expressed."""
        return _ok(self._req(f"vset /object/{actor}/rotation 0 {yaw_deg:.1f} 0", timeout=30))

"""Tests for the MetaHuman appearance / gait driver.

These run against a fake transport, so they verify the command protocol and the
degradation behaviour without needing the engine. The point is that a build
lacking these features must be *reported*, never silently pretended away — a run
that renders twelve identical people is unusable for training and the caller has
to know.
"""

from __future__ import annotations

from survsim.backends.unreal.appearance import (
    GAIT_CLIPS,
    IDLE_CLIPS,
    AppearanceDriver,
)
from survsim.core.appearance import AppearanceState
from survsim.core.models import Body, Outfit, Person

FULL_LISTING = (
    "vset /metahuman/parametric/body [str] [str] [float] [float] [float]\n"
    "vset /metahuman/parametric/wardrobe [str] [str] [str]\n"
    "vset /metahuman/parametric/assemble [str] [str] [str]\n"
    "vset /animation/soma_glb/apply [str] [Anything]\n"
    "vset /animation/soma_glb/stop [str]\n"
    "vset /agent/[str]/nav/goto [float] [float] [float]\n"
    "vget /agent/[str]/nav/status\n"
)
BARE_LISTING = "vget /camera/0/lit\nvset /object/[str]/location [float] [float] [float]\n"


class FakeServer:
    """Records commands and replies; can be told which commands fail."""

    def __init__(self, listing: str = FULL_LISTING, fail: tuple[str, ...] = (),
                 animation_works: bool = True) -> None:
        self.listing = listing
        self.fail = fail
        self.animation_works = animation_works
        self.commands: list[str] = []

    def __call__(self, cmd: str, timeout: float = 60) -> str:
        self.commands.append(cmd)
        if cmd.startswith("vget /unrealcv/list_cmd"):
            return self.listing
        if any(f in cmd for f in self.fail):
            return "error: unsupported"
        if "soma_glb/status" in cmd:
            return ("attached=true playing=true duration=1.20"
                    if self.animation_works else
                    "attached=false playing=false duration=0.000")
        if "nav/status" in cmd:
            return "moving"
        return "ok"

    def sent(self, needle: str) -> list[str]:
        return [c for c in self.commands if needle in c]


def _person() -> Person:
    return Person(id="p1", body=Body(height_m=1.90, bmi_class="athletic", skin_tone=8,
                                     sex_presentation="f"),
                  outfit=Outfit(top="hoodie", outer="rain_jacket", bottom="jeans",
                                headwear="cap", colours={"outer": "#0b3d91"}))


# ------------------------------------------------------------------- probing


def test_probe_detects_full_capabilities() -> None:
    driver = AppearanceDriver(FakeServer())
    caps = driver.probe(sample_actor="a")
    assert caps.body and caps.wardrobe and caps.animation and caps.navigation
    assert caps.diverse_appearance and caps.verified


def test_animation_present_but_broken_is_not_reported_as_working() -> None:
    """The real v3.1.0 case: the command exists and every clip name is rejected."""
    driver = AppearanceDriver(FakeServer(animation_works=False))
    caps = driver.probe(sample_actor="a")
    assert caps.animation_command_present is True
    assert caps.animation is False
    assert any("walk identically" in n for n in caps.notes)


def test_unverified_animation_is_assumed_broken() -> None:
    """Without an actor to test on, assume it cannot animate rather than guess."""
    caps = AppearanceDriver(FakeServer()).probe()
    assert caps.animation is False
    assert any("unverified" in n for n in caps.notes)


def test_probe_on_a_bare_build_reports_the_limitation_loudly() -> None:
    """A build with no wardrobe renders clones; that must be surfaced, not hidden."""
    driver = AppearanceDriver(FakeServer(listing=BARE_LISTING))
    caps = driver.probe()
    assert not caps.diverse_appearance
    assert any("render identically" in n for n in caps.notes)
    assert any("unsuitable for training" in n for n in caps.notes)


def test_no_commands_are_sent_when_unsupported() -> None:
    server = FakeServer(listing=BARE_LISTING)
    driver = AppearanceDriver(server)
    driver.probe()
    assert driver.apply_body("a", _person()) is False
    assert driver.apply_wardrobe("a", AppearanceState.from_person(_person())) == {}
    assert driver.apply_gait("a", "limp_L") is None
    assert driver.walk_to("a", 1, 2, 3) is False
    assert not server.sent("parametric")
    assert not server.sent("soma_glb")


# ---------------------------------------------------------------- appearance


def test_body_encodes_height_and_build() -> None:
    server = FakeServer()
    driver = AppearanceDriver(server)
    driver.probe()
    assert driver.apply_body("a", _person())

    cmd = server.sent("parametric/body")[0].split()
    assert cmd[2] == "a" and cmd[3] == "female"
    height_param, build, tone = float(cmd[4]), float(cmd[5]), float(cmd[6])
    assert height_param > 0.5           # 1.90 m is well above the 1.70 m mean
    assert build > 0                    # athletic
    assert tone == 0.8                  # skin tone 8/10


def test_height_parameter_is_clamped() -> None:
    server = FakeServer()
    driver = AppearanceDriver(server)
    driver.probe()
    tall = _person()
    tall.body.height_m = 2.60           # absurd, must not produce a wild parameter
    driver.apply_body("a", tall)
    value = float(server.sent("parametric/body")[0].split()[4])
    assert -1.0 <= value <= 1.0


def test_wardrobe_sets_every_slot_and_clears_empty_ones() -> None:
    server = FakeServer()
    driver = AppearanceDriver(server)
    driver.probe()

    person = _person()
    person.outfit.headwear = None
    state = AppearanceState.from_person(person)
    results = driver.apply_wardrobe("a", state)

    assert set(results) >= {"top", "outer", "bottom", "shoes", "headwear"}
    sent = " ".join(server.sent("wardrobe"))
    assert "a top hoodie" in sent
    assert "a headwear none" in sent       # removal is explicit
    assert "colour 0b3d91" in sent


def test_appearance_change_is_reflected_in_the_next_wardrobe_call() -> None:
    """Taking the jacket off must actually reach the renderer."""
    server = FakeServer()
    driver = AppearanceDriver(server)
    driver.probe()

    state = AppearanceState.from_person(_person())
    driver.apply_wardrobe("a", state)
    assert "a outer rain_jacket" in " ".join(server.sent("wardrobe"))

    server.commands.clear()
    state.apply({"outer": "none"}, t=10.0)
    driver.apply_wardrobe("a", state)
    assert "a outer none" in " ".join(server.sent("wardrobe"))


# ---------------------------------------------------------------------- gait


def test_gait_selects_a_motion_clip_not_a_speed() -> None:
    server = FakeServer()
    driver = AppearanceDriver(server)
    driver.probe(sample_actor="a")
    clip = driver.apply_gait("a", "limp_L")
    assert clip == GAIT_CLIPS["limp_L"][0]
    assert server.sent("soma_glb/apply")


def test_gait_falls_back_through_clip_name_candidates() -> None:
    """Builds name clips differently; try alternatives before giving up."""
    server = FakeServer(fail=("walk_limp_left",))
    driver = AppearanceDriver(server)
    driver.probe(sample_actor="a")
    assert driver.apply_gait("a", "limp_L") == GAIT_CLIPS["limp_L"][1]


def test_unknown_gait_falls_back_to_normal() -> None:
    driver = AppearanceDriver(FakeServer())
    driver.probe(sample_actor="a")
    assert driver.apply_gait("a", "moonwalk") == GAIT_CLIPS["normal"][0]


def test_clip_resolution_is_cached() -> None:
    server = FakeServer()
    driver = AppearanceDriver(server)
    driver.probe(sample_actor="a")
    driver.apply_gait("a", "normal")
    first = len(server.sent("soma_glb/apply"))
    driver.apply_gait("a", "normal")
    assert len(server.sent("soma_glb/apply")) == first    # no re-probing


def test_stationary_activities_have_clips() -> None:
    """Loitering while standing still is the case motion detection loses."""
    driver = AppearanceDriver(FakeServer())
    driver.probe(sample_actor="a")
    assert driver.apply_idle("a", "scan") == IDLE_CLIPS["scan"][0]
    assert driver.apply_idle("a", "phone") == IDLE_CLIPS["phone"][0]


# ---------------------------------------------------------------- navigation


def test_walk_to_uses_navigation_rather_than_teleporting() -> None:
    server = FakeServer()
    driver = AppearanceDriver(server)
    driver.probe()
    assert driver.walk_to("a", 100.0, 200.0, 0.0)
    assert server.sent("nav/goto")
    assert not server.sent("/location")      # teleporting would kill the leg motion


def test_walking_status_is_readable() -> None:
    driver = AppearanceDriver(FakeServer())
    driver.probe()
    assert driver.is_walking("a") is True

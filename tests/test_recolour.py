"""Tests for image-space garment recolouring.

The properties that make it worth doing: it must actually change the colour, it
must preserve the shading that makes cloth look like cloth, it must only touch
the person it was told to, and it must report honestly how much real variety it
produced.
"""

from __future__ import annotations

import cv2
import numpy as np

from survsim.core.models import Outfit, Person
from survsim.sensors.recolour import (
    BODY_BANDS,
    RecolourSpec,
    band_mask,
    diversity_report,
    hex_to_bgr,
    person_mask,
    recolour_frame,
    recolour_person,
)

MASK_A = (10, 20, 30)
MASK_B = (200, 40, 60)


def _scene(h: int = 120, w: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """A grey person on black, with vertical shading so we can check it survives."""
    image = np.zeros((h, w, 3), np.uint8)
    mask = np.zeros((h, w, 3), np.uint8)
    shading = np.linspace(60, 200, h).astype(np.uint8)
    image[:, 20:60] = np.repeat(shading[:, None], 40, axis=1)[:, :, None]
    mask[:, 20:60] = MASK_A
    return image, mask


def _spec(**slots) -> RecolourSpec:
    return RecolourSpec(uid="p0", mask_colour=MASK_A, slots=slots or {"top": "#c81414"})


# --------------------------------------------------------------------- basics


def test_hex_converts_to_bgr() -> None:
    assert hex_to_bgr("#ff0000") == (0, 0, 255)
    assert hex_to_bgr("00ff00") == (0, 255, 0)


def test_person_mask_selects_only_that_actor() -> None:
    _, mask = _scene()
    mask[0:10, 0:10] = MASK_B
    selected = person_mask(mask, MASK_A)
    assert selected[:, 20:60].any()
    assert not selected[0:10, 0:10].any()


def test_band_mask_splits_the_body_vertically() -> None:
    _, mask = _scene()
    person = person_mask(mask, MASK_A)
    top = band_mask(person, BODY_BANDS["top"])
    bottom = band_mask(person, BODY_BANDS["bottom"])

    assert top.any() and bottom.any()
    assert not (top & bottom).any()                     # bands must not overlap
    assert np.nonzero(top)[0].mean() < np.nonzero(bottom)[0].mean()


def test_band_mask_of_empty_person_is_empty() -> None:
    assert not band_mask(np.zeros((10, 10), bool), BODY_BANDS["top"]).any()


# ------------------------------------------------------------------ recolour


def test_recolouring_changes_the_garment_colour() -> None:
    image, mask = _scene()
    out, changed = recolour_person(image, mask, _spec(top="#c81414"))

    assert changed > 0
    band = band_mask(person_mask(mask, MASK_A), BODY_BANDS["top"])
    before = image[band].astype(int)
    after = out[band].astype(int)
    # The red channel should rise well above the others.
    assert after[:, 2].mean() > before[:, 2].mean() + 25
    assert after[:, 2].mean() > after[:, 0].mean() + 25


def test_shading_is_preserved() -> None:
    """Flattening the luminance is what makes recoloured synthetic data look fake."""
    image, mask = _scene()
    out, _ = recolour_person(image, mask, _spec(top="#1414c8"))

    band = band_mask(person_mask(mask, MASK_A), BODY_BANDS["top"])
    before_l = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0][band].astype(float)
    after_l = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[:, :, 0][band].astype(float)

    assert after_l.std() > before_l.std() * 0.85        # gradient survives
    assert abs(after_l.mean() - before_l.mean()) < 12   # not globally brightened


def test_only_the_named_band_is_touched() -> None:
    image, mask = _scene()
    out, _ = recolour_person(image, mask, _spec(top="#c81414"))
    shoes = band_mask(person_mask(mask, MASK_A), BODY_BANDS["shoes"])
    assert np.array_equal(out[shoes], image[shoes])


def test_other_people_and_background_are_untouched() -> None:
    image, mask = _scene()
    image[0:15, 0:15] = 90
    mask[0:15, 0:15] = MASK_B
    out, _ = recolour_person(image, mask, _spec(top="#c81414"))

    assert np.array_equal(out[0:15, 0:15], image[0:15, 0:15])   # the other actor
    assert np.array_equal(out[:, 0:20], image[:, 0:20])         # background


def test_absent_person_is_a_no_op() -> None:
    image, mask = _scene()
    out, changed = recolour_person(image, mask, RecolourSpec("ghost", (7, 7, 7),
                                                            {"top": "#ffffff"}))
    assert changed == 0
    assert np.array_equal(out, image)


def test_strength_controls_how_far_the_colour_moves() -> None:
    image, mask = _scene()
    weak, _ = recolour_person(image, mask, RecolourSpec("p", MASK_A, {"top": "#c81414"}, 0.2))
    strong, _ = recolour_person(image, mask, RecolourSpec("p", MASK_A, {"top": "#c81414"}, 1.0))

    band = band_mask(person_mask(mask, MASK_A), BODY_BANDS["top"])
    assert strong[band][:, 2].mean() > weak[band][:, 2].mean()


def test_alpha_channel_is_preserved() -> None:
    image, mask = _scene()
    rgba = np.dstack([image, np.full(image.shape[:2], 255, np.uint8)])
    out, _ = recolour_person(rgba, mask, _spec(top="#c81414"))
    assert out.shape[2] == 4


# ----------------------------------------------------------------- integration


def test_two_people_get_different_clothes() -> None:
    """The whole point: identical actors must stop looking identical."""
    image, mask = _scene(120, 120)
    image[:, 70:110] = image[:, 20:60]
    mask[:, 70:110] = MASK_B

    out, report = recolour_frame(image, mask, [
        RecolourSpec("p0", MASK_A, {"top": "#c81414"}),
        RecolourSpec("p1", MASK_B, {"top": "#1414c8"}),
    ])
    assert report["p0"] > 0 and report["p1"] > 0

    a = band_mask(person_mask(mask, MASK_A), BODY_BANDS["top"])
    b = band_mask(person_mask(mask, MASK_B), BODY_BANDS["top"])
    assert out[a][:, 2].mean() > out[a][:, 0].mean()    # one reads red
    assert out[b][:, 0].mean() > out[b][:, 2].mean()    # the other blue


def test_spec_is_built_from_the_generated_wardrobe() -> None:
    """Colours the population generator already chose should drive the pixels."""
    person = Person(id="p9", outfit=Outfit(top="tshirt", outer="rain_jacket",
                                           colours={"top": "#123456", "outer": "#abcdef",
                                                    "bottom": "#222222"}))
    spec = RecolourSpec.from_person(person, MASK_A)
    assert spec.uid == "p9"
    assert spec.slots["top"] == "#abcdef"     # the outer layer is what is seen
    assert spec.slots["bottom"] == "#222222"


def test_diversity_report_does_not_overstate_near_identical_colours() -> None:
    near = [RecolourSpec(f"p{i}", MASK_A, {"top": c}) for i, c in
            enumerate(["#808080", "#828282", "#7e7e7e"])]
    varied = [RecolourSpec(f"p{i}", MASK_A, {"top": c}) for i, c in
              enumerate(["#c81414", "#1414c8", "#14c814"])]

    assert diversity_report(near)["distinct_at_10"] == 1
    assert diversity_report(varied)["distinct_at_10"] == 3
    assert (diversity_report(varied)["mean_pairwise_chroma_distance"]
            > diversity_report(near)["mean_pairwise_chroma_distance"])


def test_diversity_report_states_the_limitation() -> None:
    report = diversity_report([RecolourSpec("a", MASK_A, {"top": "#c81414"}),
                               RecolourSpec("b", MASK_B, {"top": "#1414c8"})])
    assert "garment shape unchanged" in report["note"]

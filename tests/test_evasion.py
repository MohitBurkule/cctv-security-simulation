"""Tests for the evasion taxonomy.

The important tests here are not about code paths, they are about the dataset
property that makes the whole thing scientifically usable: appearance must not
predict intent.
"""

from __future__ import annotations

import numpy as np
import pytest

from survsim.director.evasion import (
    ALL,
    BY_ID,
    CUES,
    audit_confounds,
    compatible,
    generate_matched_population,
    residual_cues,
    sample_combination,
)


def test_every_technique_is_well_formed() -> None:
    for t in ALL:
        assert t.id and t.category and t.label
        assert t.benign_rate > 0, f"{t.id} needs a benign rate"
        assert t.benign_contexts, f"{t.id} must document who does this innocently"
        for cue, v in t.degrades.items():
            assert cue in CUES, f"{t.id} degrades unknown cue {cue}"
            assert 0.0 <= v <= 1.0


def test_benign_contexts_are_tuples_not_bare_strings() -> None:
    """A missing trailing comma makes a tuple a string, which then iterates per
    character and renders rationales as "a; c; t; u; l". Silent and ugly."""
    for t in ALL:
        assert isinstance(t.benign_contexts, tuple), f"{t.id}: benign_contexts must be a tuple"
        for c in t.benign_contexts:
            assert len(c) > 1, f"{t.id}: {c!r} looks like a string iterated per character"


def test_conflicts_are_symmetric_enough_to_prevent_impossible_combinations() -> None:
    """You cannot wear a surgical mask and a balaclava at once."""
    assert not compatible(["mask_surgical", "balaclava"])
    assert not compatible(["wheelchair", "brisk_march"])
    assert compatible(["mask_surgical", "sunglasses"])


def test_residual_cues_compose_multiplicatively() -> None:
    face_only = residual_cues(["sunglasses"])["face"]
    both = residual_cues(["sunglasses", "hood_up"])["face"]
    assert both < face_only < 1.0
    assert both == pytest.approx((1 - 0.45) * (1 - 0.6), rel=1e-6)


def test_balaclava_destroys_face_but_leaves_gait() -> None:
    """The point of tracking residual cues: know which route is still open."""
    r = residual_cues(["balaclava"])
    assert r["face"] < 0.05
    assert r["gait"] == 1.0


def test_wheelchair_changes_silhouette_and_gait_completely() -> None:
    r = residual_cues(["wheelchair"])
    assert r["gait"] == 0.0
    assert r["silhouette"] <= 0.1
    assert r["height"] <= 0.2


def test_sampled_combinations_are_always_compatible() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        ids = sample_combination(rng, n_techniques=3)
        assert compatible(ids)


def test_protected_attributes_can_be_excluded() -> None:
    rng = np.random.default_rng(1)
    for _ in range(100):
        ids = sample_combination(rng, n_techniques=3, allow_protected=False)
        assert not any(BY_ID[i].protected for i in ids)


# ------------------------------------------------- the constraint that matters


def test_matched_population_makes_appearance_uninformative() -> None:
    """Every hostile persona has a benign twin wearing exactly the same things.

    If this fails, a classifier can reach high accuracy by reading clothing alone,
    which is both scientifically worthless and discriminatory in deployment.
    """
    people = generate_matched_population(n_pairs=40, seed=7)
    audit = audit_confounds(people)

    assert audit["n_hostile"] == audit["n_benign"] == 40
    assert audit["balanced"], f"appearance leaks intent: {audit['worst_imbalance']}"
    assert audit["worst_imbalance"] == pytest.approx(1.0, abs=1e-6)


def test_protected_attributes_are_never_imbalanced() -> None:
    people = generate_matched_population(n_pairs=60, seed=3)
    audit = audit_confounds(people)
    assert audit["protected_clean"], audit["protected_imbalance"]


def test_audit_detects_a_deliberately_biased_population() -> None:
    """The audit must actually catch bias, or it is decoration."""
    people = generate_matched_population(n_pairs=20, seed=5)
    for p in people:
        if p.intent == "hostile":
            p.techniques = ["balaclava"]
        else:
            p.techniques = ["clear_glasses"]

    audit = audit_confounds(people)
    assert not audit["balanced"]
    assert audit["per_technique"]["balaclava"]["ratio"] > 5


def test_benign_rates_reflect_reality_for_protected_attributes() -> None:
    """Religious dress and mobility aids must be modelled as overwhelmingly innocent."""
    for tid in ("niqab", "burqa", "hijab", "wheelchair", "walking_stick"):
        assert BY_ID[tid].benign_rate >= 1000, tid
        assert BY_ID[tid].protected


def test_police_impersonation_is_the_rare_evidential_case() -> None:
    """Impersonating police is itself an offence, unlike wearing a mask."""
    assert BY_ID["police_uniform"].benign_rate < 1.0
    assert BY_ID["mask_surgical"].benign_rate > 100


def test_persona_delta_merges_without_losing_slots() -> None:
    people = generate_matched_population(n_pairs=30, seed=11)
    for p in people:
        d = p.delta()
        assert isinstance(d, dict)
        for key in ("face", "hair", "gait"):
            if key in d:
                assert isinstance(d[key], dict)


def test_generation_is_deterministic() -> None:
    a = generate_matched_population(n_pairs=15, seed=42)
    b = generate_matched_population(n_pairs=15, seed=42)
    assert [p.techniques for p in a] == [p.techniques for p in b]

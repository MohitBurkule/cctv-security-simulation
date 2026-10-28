from __future__ import annotations


def test_documented_evasion_and_behaviour_counts_match_code() -> None:
    """Guard against CLAUDE.md / docs/11 repeating a stale headline count.

    This has already happened once: CLAUDE.md said "40 techniques" when the
    code had grown to 52, and docs/11 said "32 behaviours" when the registry
    had grown to 44 classes / 49 catalogue names. Both docs are free-form
    prose, so this pins the exact numbers with a regex against the same
    strings a human reads, rather than trusting the prose to stay in sync by
    hand.
    """
    import re
    from pathlib import Path

    from survsim.director.behaviours import REGISTRY
    from survsim.director.evasion import ALL as EVASION_TECHNIQUES

    repo_root = Path(__file__).resolve().parents[1]

    n_evasion = len(EVASION_TECHNIQUES)
    n_behaviour_classes = len(set(REGISTRY.values()))
    n_behaviour_names = len(REGISTRY)

    claude_md = (repo_root / "CLAUDE.md").read_text()
    match = re.search(r"evasion taxonomy\s*\n?\(\s*(\d+)\s*techniques\)", claude_md)
    assert match, "CLAUDE.md's evasion-technique count sentence was not found"
    assert int(match.group(1)) == n_evasion, (
        f"CLAUDE.md says {match.group(1)} evasion techniques; "
        f"survsim.director.evasion.ALL has {n_evasion}"
    )

    status = (repo_root / "docs" / "11_IMPLEMENTATION_STATUS.md").read_text()
    match = re.search(
        r"Behaviour / intent \| done — (\d+) behaviours \((\d+) catalogue names\)",
        status,
    )
    assert match, "docs/11's behaviour-count row was not found"
    assert int(match.group(1)) == n_behaviour_classes, (
        f"docs/11 says {match.group(1)} behaviours; "
        f"survsim.director.behaviours.REGISTRY has {n_behaviour_classes} classes"
    )
    assert int(match.group(2)) == n_behaviour_names, (
        f"docs/11 says {match.group(2)} catalogue names; "
        f"survsim.director.behaviours.REGISTRY has {n_behaviour_names} entries"
    )

"""Asset licence provenance for ground truth (docs/research/47).

Every rendered person depends on a character asset; its licence decides whether
frames containing that person may train or test a model that is sold. This
module turns the character a person was rendered with into an `asset_licences`
record, and a whole set into a dataset-level `commercial_ml_ok` flag:

* CC0-1.0 (Quaternius bodies, Kenney, ambientCG, Poly Haven) and our own
  procedural assets: permitted, no conditions.
* CC-BY-4.0 (CesiumMan, Khronos sample): permitted WITH attribution - the
  attribution string is emitted alongside the data.
* MetaHuman (Epic MetaHuman licence): the terms forbid using MetaHuman
  characters/animation to build a database or train/test AI -> not OK.
* Anything unknown is NOT OK: provenance must be proven, not assumed.

Pure Python; used by tools/render_run.py (per-set licences.json and the
identities table) and by the mock-backend Recorder.
"""
from __future__ import annotations

from typing import Any

# licence id -> (commercial ML use permitted, needs attribution)
LICENCE_TERMS: dict[str, tuple[bool, bool]] = {
    "CC0-1.0": (True, False),
    "CC-BY-4.0": (True, True),
    "LicenseRef-SurvSim-Procedural": (True, False),
    "LicenseRef-Epic-MetaHuman": (False, False),
    "LicenseRef-Unknown": (False, False),
}
CESIUMMAN_ATTRIBUTION = (
    "CesiumMan by Cesium (https://cesium.com), from the Khronos glTF-Sample-Assets "
    "repository, licensed CC-BY 4.0 (https://creativecommons.org/licenses/by/4.0/).")
PROCEDURAL = {"source": "survsim-procedural", "licence": "LicenseRef-SurvSim-Procedural"}


def normalise_licence(text: str | None) -> str:
    """Free-text licence ("CC-BY-4.0 (Khronos sample)", "CC0-1.0") -> SPDX-style id."""
    t = (text or "").strip().upper().replace(" ", "")
    if t.startswith("CC0"):
        return "CC0-1.0"
    if t.startswith(("CC-BY-4.0", "CCBY4.0", "CC-BY4")):
        return "CC-BY-4.0"
    if "METAHUMAN" in t:
        return "LicenseRef-Epic-MetaHuman"
    return "LicenseRef-Unknown"


def character_licence(name: str, entry: dict[str, Any] | None) -> dict[str, str]:
    """{source, licence} for one catalogue character (render catalogue entry)."""
    entry = entry or {}
    kind = entry.get("kind")
    if kind == "metahuman":
        return {"source": f"metahuman:{name}", "licence": "LicenseRef-Epic-MetaHuman"}
    if name == "CesiumMan":
        return {"source": "khronos:CesiumMan", "licence": "CC-BY-4.0"}
    lic = normalise_licence(entry.get("license"))
    src = entry.get("author") or ("quaternius" if kind == "cc0" else "unknown")
    return {"source": f"{str(src).lower()}:{name}", "licence": lic}


def person_licences(character: str, entry: dict[str, Any] | None,
                    extra: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    """Every asset a rendered person depends on (body + animation share the file)."""
    out = [character_licence(character, entry)]
    if (entry or {}).get("kind") == "metahuman" and "AS_MH_" in str((entry or {}).get("walk")):
        out.append({"source": "metahuman:AS_MH_locomotion", "licence": "LicenseRef-Epic-MetaHuman"})
    return out + list(extra or [])


def dataset_licences(per_person: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    """Set-level summary: the licence ids used, `commercial_ml_ok`, attributions."""
    ids = sorted({a["licence"] for recs in per_person.values() for a in recs})
    ok = bool(ids) and all(LICENCE_TERMS.get(i, (False, False))[0] for i in ids)
    attributions = []
    if any(a["licence"] == "CC-BY-4.0" and "CesiumMan" in a["source"]
           for recs in per_person.values() for a in recs):
        attributions.append(CESIUMMAN_ATTRIBUTION)
    blockers = sorted({a["source"] for recs in per_person.values() for a in recs
                       if not LICENCE_TERMS.get(a["licence"], (False, False))[0]})
    return {"asset_licences": ids, "commercial_ml_ok": ok, "attributions": attributions,
            "blocking_assets": blockers[:50]}


def licence_column(recs: list[dict[str, str]]) -> str:
    """Compact per-row form for parquet: "CC0-1.0|quaternius:man_hmnu;..."."""
    return ";".join(f"{a['licence']}|{a['source']}" for a in recs)

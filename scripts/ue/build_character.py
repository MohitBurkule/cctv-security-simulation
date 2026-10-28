"""Build a MetaHuman with a specified body, inside Unreal Editor.

    /opt/unreal-engine/Engine/Binaries/Linux/UnrealEditor \
        ~/data/ue-projects/SurvSim/SurvSim.uproject \
        -run=pythonscript -script="scripts/ue/build_character.py" \
        -unattended -nosplash -nullrhi

**[V] The parametric body works.** Verified on UE 5.8.2: a character created
from `MetaHumanCharacterFactoryNew` exposes 30 named body constraints, and
driving `Height` from 171.75 cm to 190.0 cm persists through
`commit_body_state` and reads back correctly. That closes a capability this
project recorded as blocked for most of its life — the wardrobe and body API
was never absent from the engine, only from the packaged runtime we were
driving over a socket.

Two things cost time getting here and are worth stating plainly:

  - **`try_add_object_to_edit(character)` is mandatory.** Without it,
    `get_body_constraints` returns `[]` and `commit_body_state` silently does
    nothing, logging only "not added for editing". An empty list reads exactly
    like "this character has no parameters", which is why the first attempt
    concluded the API was a stub.
  - **The property is `is_active`, not `active`.** Assigning the wrong name
    raises, which is the good case; the dangerous neighbour is a mistyped
    *constraint name*, which does not raise — it just leaves that measurement at
    its default while ground truth records the value you asked for.

Still open: `can_build_meta_human` returns False with "Character is not rigged".
A blank character has a body but no rig, so it cannot yet be assembled into a
renderable skeletal mesh. That needs rig/source content (MetaHuman assets ship
separately from the engine), and is the next thing to solve.
"""

from __future__ import annotations

import json
import traceback

try:
    import unreal
except ImportError:  # pragma: no cover - editor only
    raise SystemExit("must run inside Unreal Editor")

RESULT = "/tmp/survsim_character_result.json"

# Kept in step with survsim.export.unreal_scene.BUILD_SHAPES. Duplicated rather
# than imported because the editor's Python cannot see our venv, and a wrong
# shape is far less harmful than a fragile sys.path dance.
DEFAULT_TARGETS = {"Height": 175.0, "Fat": 0.5, "Muscularity": 0.5}


def apply_body(subsystem, character, targets: dict) -> dict:
    """Assign measurement targets by name, reporting what actually applied.

    Returns the values read back afterwards rather than the values requested.
    A constraint name the engine does not recognise is skipped silently by the
    API, so "what we asked for" and "what the character is" are different
    questions and only the second one matters.
    """
    constraints = list(subsystem.get_body_constraints(character))
    by_name = {str(c.get_editor_property("name")): c for c in constraints}

    unknown = sorted(set(targets) - set(by_name))
    for name, value in targets.items():
        c = by_name.get(name)
        if c is None:
            continue
        c.set_editor_property("target_measurement", float(value))
        c.set_editor_property("is_active", True)

    subsystem.set_body_constraints(character, constraints)
    subsystem.commit_body_state(character)

    after = {str(c.get_editor_property("name")):
             float(c.get_editor_property("target_measurement"))
             for c in subsystem.get_body_constraints(character)}
    return {
        "applied": {n: after.get(n) for n in targets if n in by_name},
        "unknown_constraints": unknown,
        "available": sorted(by_name),
    }


def main() -> int:
    out = {}
    try:
        subsystem = unreal.get_editor_subsystem(
            unreal.MetaHumanCharacterEditorSubsystem)
        tools = unreal.AssetToolsHelpers.get_asset_tools()

        character = tools.create_asset(
            asset_name="SurvSimCharacter", package_path="/Game/SurvSim",
            asset_class=unreal.MetaHumanCharacter,
            factory=unreal.MetaHumanCharacterFactoryNew())

        # Mandatory: without this every call below is a silent no-op.
        added = subsystem.try_add_object_to_edit(character)
        out["added_for_editing"] = bool(added)
        if not added:
            out["error"] = "could not add character for editing"
        else:
            out.update(apply_body(subsystem, character, DEFAULT_TARGETS))
            out["can_build"] = bool(
                subsystem.can_build_meta_human(character, True))
            out["can_build_note"] = (
                "False is expected on a blank character: it has a body but no "
                "rig, and rig/source content ships separately from the engine")
    except Exception:
        out["fatal"] = traceback.format_exc()[-1500:]

    # Written to a file because the editor reports "Python script executed
    # successfully" for scripts that did nothing at all.
    with open(RESULT, "w") as handle:
        json.dump(out, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

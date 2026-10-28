"""Report what the editor's Python API actually exposes, by introspection.

Run inside the editor, never in our venv:

    /opt/unreal-engine/Engine/Binaries/Linux/UnrealEditor \
        ~/data/ue-projects/SurvSim/SurvSim.uproject \
        -run=pythonscript -script=scripts/ue/probe_api.py -unattended -nosplash -nullrhi

It writes /tmp/ue_mh2_result.json. **Writing a file is the point.** The editor
reports "Python script executed successfully" for a script that produced no
effect whatsoever, so stdout and an exit code prove nothing — the first run of
this probe "succeeded" and logged nothing at all. A result file on disk is the
only evidence that the code really ran.

The finding that motivated it: with no plugins enabled the project exposes 26
MetaHuman symbols and no wardrobe API. Enabling MetaHumanCharacter,
MetaHumanCharacterUAF, MetaHumanGenerator, MetaHumanCrowd and MetaHumanRuntime
in the .uproject takes that to 272, including MetaHumanWardrobeItem and
MetaHumanCharacterEditorSubsystem.commit_body_state. The capability was never
missing from the engine — it was missing from our project file.

Caveat, and it matters: this verifies symbols EXIST. It does not verify that
calling them dresses a character. Twice in this project a capability was
reported working because a name was present. Treat everything here as [A] until
a character has actually been built and rendered.
"""
import json
out = {}
try:
    import unreal
    names = dir(unreal)
    out["n_symbols"] = len(names)
    out["metahuman"] = sorted(n for n in names if "MetaHuman" in n)
    for kw in ("Wardrobe", "Outfit", "Garment", "BodyPart", "Palette", "Crowd"):
        out[f"kw_{kw}"] = sorted(n for n in names if kw.lower() in n.lower())[:20]
    for cls in ("MetaHumanCharacter", "MetaHumanCharacterEditorSubsystem",
                "MetaHumanCharacterPalette", "MetaHumanCrowdGenerator"):
        obj = getattr(unreal, cls, None)
        out[cls] = ([m for m in dir(obj) if not m.startswith("_")][:45]
                    if obj is not None else None)
except Exception as exc:
    out["error"] = repr(exc)
with open("/tmp/ue_mh2_result.json", "w") as fh:
    json.dump(out, fh, indent=2)

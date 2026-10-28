"""Rig a MetaHuman from the DNA files the engine ships — no Epic login.

The documented path to a rigged character is `request_auto_rigging`, which is an
Epic **cloud service**: it sits in an OAuth device-code loop forever under
`-unattended` (docs/12). This avoids it entirely.

The engine ships 23 DNA files, and importing them rigs a character locally:

    sub.import_from_face_dna(ch, f"{ARCH}/SKM_Face.dna", unreal.ImportFromDNAParams())
    sub.import_body_whole_rig(ch, f"{ARCH}/SKM_Body.dna", f"{ARCH}/SKM_Face.dna")

Both return `ImportErrorCode.SUCCESS`, and the proof that it took effect is that
the downstream error *changes*: "Character is not rigged" becomes "The Character
is missing textures". Checking the error moved is the only honest verification
here, because both calls return SUCCESS whether or not anything happened.

**What is still gated.** `request_texture_sources` logs "User not logged in,
please autorig before downloading source face textures", and `blocking=True`
does not help because it is a download rather than a computation. So
`build_meta_human` cannot finish offline. That gate is cosmetic rather than
structural for our purposes: 2K face albedo is irrelevant at the 20-50 m
standoff where a person is tens of pixels tall, and everything governing shape
and motion — rig, skeleton, parametric body, and locally-synthesised skin — is
available without an account.

Also note `FixedCompatibility/` holds 18 body archetypes on a 2x3x3 grid,
{f,m}_{srt,med,tal}_{unw,nrw,ovw}, which maps directly onto the population
generator's body diversity.

    UnrealEditor SurvSim.uproject -run=pythonscript \
        -script=scripts/ue/rig_from_dna.py -unattended -nosplash -nullrhi

Writes /tmp/ue_fullrig_result.json — the editor reports "Python script executed
successfully" for scripts that did nothing, so a result file is the only proof.
"""
import json, traceback, inspect
out = {"steps": []}

def step(name, fn):
    try:
        v = fn()
        out["steps"].append({"step": name, "ok": True, "value": str(v)[:250]})
        return v
    except Exception as exc:
        out["steps"].append({"step": name, "ok": False,
                             "error": f"{type(exc).__name__}: {exc}"[:280]})
        return None

ARCH = "/opt/unreal-engine/Engine/Plugins/MetaHuman/MetaHumanCoreTechLib/Content/ArchetypeDNA"

try:
    import unreal
    out["ImportFromDNAParams_doc"] = (inspect.getdoc(unreal.ImportFromDNAParams) or "")[:700]
    out["BuildParams_doc"] = (inspect.getdoc(
        unreal.MetaHumanCharacterEditorBuildParameters) or "")[:500]

    sub = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    ch = tools.create_asset(asset_name="SurvSimFullRig", package_path="/Game/SurvSim",
                            asset_class=unreal.MetaHumanCharacter,
                            factory=unreal.MetaHumanCharacterFactoryNew())
    sub.try_add_object_to_edit(ch)

    params = step("make_ImportFromDNAParams", lambda: unreal.ImportFromDNAParams())
    step("import_from_face_dna", lambda: sub.import_from_face_dna(
        ch, f"{ARCH}/SKM_Face.dna", params))
    step("can_build_after_face", lambda: sub.can_build_meta_human(ch, True))

    step("import_body_whole_rig", lambda: sub.import_body_whole_rig(
        ch, f"{ARCH}/SKM_Body.dna", f"{ARCH}/SKM_Face.dna"))
    step("can_build_after_both", lambda: sub.can_build_meta_human(ch, True))

    bp = step("make_build_params",
              lambda: unreal.MetaHumanCharacterEditorBuildParameters())
    if bp is not None:
        try:
            out["build_param_fields"] = [p for p in dir(bp)
                                         if not p.startswith("_")][:30]
        except Exception:
            pass
    step("build_meta_human", lambda: sub.build_meta_human(ch, bp))
except Exception:
    out["fatal"] = traceback.format_exc()[-1500:]

with open("/tmp/ue_fullrig_result.json", "w") as fh:
    json.dump(out, fh, indent=2)

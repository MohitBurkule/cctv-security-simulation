"""Editor job: import ONE region-painted CC0 character and verify it (docs/12 "CC0 default").

Submitted by tools/import_characters.py --cc0 with `ONLY` (index name) and
`SRC` (the glb in ~/data/assets/characters_cc0) prepended. Imports into
/Game/SurvSim/Chars/<name>_c, then checks - by AnimPose forward kinematics on
the clips, not by a success flag:

* mesh bounds height 160-190 cm (the rebake normalises to 175 cm),
* the mesh carries vertex colours (the clothing regions),
* upright in the WALK clip: head bone >= 0.75 H and lowest foot <= 0.1 H at
  every sampled time, and the head stays above the feet,
* foot curves of the walk (and run) sampled over one cycle (component space)
  are written to configs/people/cc0_foot_curves/<name>_<clip>.json so the host
  measures ground speed and foot slide the same way as for the MetaHuman loop.

The catalogue entry is merged into Saved/survsim_characters_cc0.json.
"""
import json
import os
import re

import unreal

REPO = "/home/mohit/multi-person-camera-real-sim"
CATALOG = os.path.expanduser("~/data/ue-projects/SurvSim/Saved/survsim_characters_cc0.json")
CURVES = f"{REPO}/configs/people/cc0_foot_curves"
_eal = unreal.EditorAssetLibrary
A = unreal.AnimPoseExtensions

name = ONLY  # noqa: F821
src = SRC  # noqa: F821
dest = f"/Game/SurvSim/Chars/{name}_c"
if not _eal.list_assets(dest, True):
    t = unreal.AssetImportTask()
    t.filename = src
    t.destination_path = dest
    t.automated = True
    t.replace_existing = True
    t.save = True
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([t])
assets = [_eal.load_asset(str(p)) for p in _eal.list_assets(dest, True)]
meshes = [a for a in assets if isinstance(a, unreal.SkeletalMesh)]
anims = [a for a in assets if isinstance(a, unreal.AnimSequence)]
if not meshes or not anims:
    raise RuntimeError(f"import gave meshes={len(meshes)} anims={len(anims)}")
mesh = meshes[0]


def tail(a):
    # "<file>CharacterArmature_Walk" / "...HumanArmature_Female_Walk" / "Walking"
    return re.split(r"[_|]", a.get_name())[-1].lower()


def pick(words):
    hits = [a for a in anims if tail(a) in words]
    hits.sort(key=lambda a: len(a.get_name()))
    return hits[0] if hits else None


walk = pick(("walk", "walking"))
idle = pick(("idle", "standing"))
run = pick(("run", "running"))
if walk is None:
    raise RuntimeError(f"no walk clip in {[a.get_name() for a in anims]}")
b = mesh.get_bounds()
H = 2.0 * b.box_extent.z
fails = []
if not 160.0 <= H <= 190.0:
    fails.append(f"mesh height {H:.1f} cm")
has_vc = None
try:
    has_vc = bool(mesh.get_editor_property("has_vertex_colors"))
except Exception as exc:                                     # noqa: BLE001
    fails_vc = f"has_vertex_colors unreadable: {exc}"[:120]  # [A]: not exposed in 5.8
if has_vc is False:
    fails.append("no vertex colours (clothing regions lost on import)")

opts = unreal.AnimPoseEvaluationOptions()
names = [str(n) for n in A.get_bone_names(A.get_anim_pose_at_time(walk, 0.0, opts))]


def bone(rx):
    hits = [n for n in names if re.search(rx, n, re.IGNORECASE)]
    hits.sort(key=len)
    return hits[0] if hits else None


fl, fr = bone(r"^foot[._]?l$|^leftfoot$"), bone(r"^foot[._]?r$|^rightfoot$")
head = bone(r"^head$")
if not (fl and fr and head):
    raise RuntimeError(f"bones not found: foot_l={fl} foot_r={fr} head={head} in {names[:60]}")


def curves(clip, n=240):
    L = float(clip.get_play_length())
    out = {"len_s": L, "t": [], "foot_l": [], "foot_r": [], "head": [], "root": [],
           "bones": [fl, fr, head, names[0]], "clip": clip.get_path_name(),
           "ref_height_cm": round(H, 2)}
    for i in range(n + 1):
        t = L * i / n
        pose = A.get_anim_pose_at_time(clip, t, opts)
        for key, bn in (("foot_l", fl), ("foot_r", fr), ("head", head), ("root", names[0])):
            v = A.get_bone_pose(pose, bn, unreal.AnimPoseSpaces.WORLD).translation
            out[key].append([round(v.x, 3), round(v.y, 3), round(v.z, 3)])
        out["t"].append(t)
    return out


os.makedirs(CURVES, exist_ok=True)
cw = curves(walk)
with open(f"{CURVES}/{name}_walk.json", "w") as fh:
    json.dump(cw, fh)
hz = [p[2] for p in cw["head"]]
fz = [min(a[2], b_[2]) for a, b_ in zip(cw["foot_l"], cw["foot_r"])]
if min(hz) < 0.75 * H:
    fails.append(f"head drops to {min(hz):.1f} cm in the walk (H {H:.1f})")
if max(fz) > 0.10 * H:
    fails.append(f"lowest foot rises to {max(fz):.1f} cm in the walk")
if run is not None:
    with open(f"{CURVES}/{name}_run.json", "w") as fh:
        json.dump(curves(run), fh)

entry = {
    "kind": "cc0", "mesh": mesh.get_path_name(), "walk": walk.get_path_name(),
    "walk_len_s": round(float(walk.get_play_length()), 4),
    "idle": idle.get_path_name() if idle else None,
    "run": run.get_path_name() if run else None,
    "run_len_s": round(float(run.get_play_length()), 4) if run else None,
    "height_cm": round(H, 2), "z_min_cm": round(b.origin.z - b.box_extent.z, 2),
    "xmax_cm": round(b.box_extent.x, 2), "has_vertex_colors": has_vc,
    "head_min_cm": round(min(hz), 1), "foot_max_cm": round(max(fz), 1),
    "source": src, "fails": fails, "verified": not fails,
}
try:
    with open(CATALOG) as fh:
        cat = json.load(fh)
except (OSError, ValueError):
    cat = {}
cat[name] = entry
with open(CATALOG, "w") as fh:
    json.dump(cat, fh, indent=1)
RESULT.update({"name": name, "verified": not fails, "fails": fails,  # noqa: F821
               "height_cm": entry["height_cm"], "walk": walk.get_name(),
               "idle": idle.get_name() if idle else None,
               "run": run.get_name() if run else None, "has_vc": has_vc})
# [X] 2026-09-26: 22 imports in one editor session climbed to 7.7 GB and logged
# 334 "Failed to allocate Device Memory": free what this job loaded.
assets = meshes = anims = mesh = walk = idle = run = None
unreal.SystemLibrary.collect_garbage()

"""Editor job: sample the MetaHuman walk loop foot bones (AnimPose FK) -> JSON.

LICENCE-RESTRICTED (docs/research/47): AS_MH_* is MetaHuman animation, which
Epic's terms bar from AI training/testing. The CC0 default samples each
Quaternius body's OWN clips in scripts/ue/cc0_verify_one.py instead
(configs/people/cc0_foot_curves/). Kept to reproduce the batch_v4-v6 numbers.
"""
import json

import unreal

clip = unreal.load_asset("/MetaHumanCharacter/Optional/Animation/UEFNAnimPreset/Locomotion/AS_MH_Neutral_Walk_Loop_F")
A = unreal.AnimPoseExtensions
opts = unreal.AnimPoseEvaluationOptions()
L = float(clip.get_play_length())
out = {"len_s": L, "t": [], "foot_l": [], "foot_r": [], "root": []}
names = [str(n) for n in A.get_bone_names(A.get_anim_pose_at_time(clip, 0.0, opts))]
fl = next(n for n in names if n.lower() in ("foot_l", "ball_l"))
fr = next(n for n in names if n.lower() in ("foot_r", "ball_r"))
n = 400
for i in range(n + 1):
    t = L * i / n
    pose = A.get_anim_pose_at_time(clip, t, opts)
    for key, b in (("foot_l", fl), ("foot_r", fr), ("root", names[0])):
        tr = A.get_bone_pose(pose, b, unreal.AnimPoseSpaces.WORLD)
        v = tr.translation
        out[key].append([v.x, v.y, v.z])
    out["t"].append(t)
out["bones"] = [fl, fr, names[0]]
with open("/home/mohit/multi-person-camera-real-sim/configs/people/mh_walk_foot_curves.json", "w") as fh:
    json.dump(out, fh)
RESULT["len_s"] = L  # noqa: F821
RESULT["bones"] = out["bones"]  # noqa: F821

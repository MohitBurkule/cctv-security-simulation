"""Write a close-range line-up manifest: every usable character walking past one camera.

    .venv/bin/python tools/char_lineup.py out/char_lineup
    .venv/bin/python tools/render_run.py out/char_lineup --frames 20

For checking scale, orientation, animation and tint per imported character
at ~8 m, where a mistake is obvious (docs/12, "Characters").
"""
import json
import sys
from pathlib import Path

CATALOG = Path.home() / "data/ue-projects/SurvSim/Saved/survsim_characters.json"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.render_run import EXCLUDE

ONLY_N = "--raw" not in sys.argv

args = [a for a in sys.argv[1:] if not a.startswith("--")]
out = Path(args[0] if args else "out/char_lineup")
PICK = args[1].split(",") if len(args) > 1 else None
chars = [k for k, v in json.loads(CATALOG.read_text()).items()
         if "error" not in v and k.removesuffix("_m") not in EXCLUDE
         and (k.endswith("_m") or not ONLY_N)][:14]
if PICK:
    chars = PICK
fps, n = 10, 40
tracks = []
for i, name in enumerate(chars):
    y = (i - (len(chars) - 1) / 2) * 120.0
    speed = 1.0 + 0.1 * (i % 4)            # m/s, to see cadence change
    keys = [{"frame": f, "location": [200.0 + f * speed * 100 / fps - 200, y, 0.0], "yaw": 0.0}
            for f in range(n)]
    tracks.append({"uid": f"p{i}_{name}", "character": name, "keys": keys})
manifest = {"version": 1, "scenario": "char_lineup", "seed": 0, "fps": fps,
            "duration_s": n / fps, "units": "unreal_cm", "people": [],
            "tracks": tracks,
            "cameras": [{"id": "cam_lineup", "location": [-900.0, 0.0, 250.0],
                         "rotation": [-12.0, 0.0, 0.0], "resolution": [1280, 720],
                         "fps": fps, "fov_deg": 70.0}]}
(out / "unreal").mkdir(parents=True, exist_ok=True)
(out / "unreal" / "manifest.json").write_text(json.dumps(manifest))
print(out, len(tracks), "characters:", ", ".join(chars))

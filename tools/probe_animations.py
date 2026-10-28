"""Find out which animation clips a build actually accepts.

`GAIT_CLIPS` in the appearance driver is a list of *candidate* names per gait
style, tried in order. That fallback chain is a guess until it has been tested
against a real build, and a guess in the locomotion path is exactly the kind of
thing that silently produces a dataset where every person walks identically.

This spawns one throwaway actor, tries every candidate, and reports which names
the server accepts so the mapping can be pinned to reality.

    .venv/bin/python tools/probe_animations.py [port]
"""

from __future__ import annotations

import json
import sys

from unrealcv import Client

from survsim.backends.unreal.appearance import GAIT_CLIPS, IDLE_CLIPS

TEST_ACTOR = "anim_probe_actor"
CHARACTER = "/Game/SmartLocomotion/Blueprints/BP_Character.BP_Character_C"

# Names worth trying beyond our candidates, covering the common conventions.
EXTRA_GUESSES = (
    "Walk", "walk", "Run", "run", "Idle", "idle", "Jog", "jog",
    "Walk_Fwd", "MM_Walk", "Anim_Walk", "A_Walk", "Locomotion_Walk",
)


def main(port: int = 9900) -> int:
    client = Client(("127.0.0.1", port))
    client.connect(timeout=30)
    if not client.isconnected():
        print("not connected — is the packaged build running?")
        return 1

    def req(cmd: str, timeout: float = 45) -> str:
        return str(client.request(cmd, timeout=timeout))

    listing = req("vget /unrealcv/list_cmd", timeout=120)
    if "/animation/soma_glb/apply" not in listing:
        print("this build has no soma_glb animation command; nothing to probe")
        client.disconnect()
        return 2

    spawned = req(f"vset /objects/spawn_from_path {CHARACTER} {TEST_ACTOR}", timeout=90).strip()
    actor = spawned if "error" not in spawned.lower() and spawned else TEST_ACTOR
    print(f"probe actor: {actor}")

    candidates: list[str] = []
    for group in (GAIT_CLIPS, IDLE_CLIPS):
        for names in group.values():
            candidates.extend(names)
    candidates.extend(EXTRA_GUESSES)
    candidates = list(dict.fromkeys(candidates))

    accepted: list[str] = []
    rejected: list[str] = []
    for name in candidates:
        res = req(f"vset /animation/soma_glb/apply {actor} {name}", timeout=30).lower()
        (accepted if ("error" not in res and "not" not in res) else rejected).append(name)

    print(f"\naccepted {len(accepted)}/{len(candidates)}")
    for name in accepted:
        print("  +", name)
    print(f"\nrejected {len(rejected)}")
    for name in rejected[:25]:
        print("  -", name)

    status = req(f"vget /animation/soma_glb/status {actor}", timeout=30)
    print("\nstatus reply:", status[:300])

    with open("out/animation_probe.json", "w") as fh:
        json.dump({"accepted": accepted, "rejected": rejected,
                   "status_reply": status[:500]}, fh, indent=1)
    print("\nwritten out/animation_probe.json")

    req(f"vset /object/{actor}/hide", timeout=30)
    client.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 9900))

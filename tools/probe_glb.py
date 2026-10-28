"""Find out what argument form `animation/soma_glb/apply` actually wants.

`probe_animations.py` established that this build rejects every *clip name* we
could think of, reporting `attached=false playing=false`. The command exists,
so something is accepted; the open question recorded in docs/12 was whether it
wants a GLB asset on disk instead of a name.

This probe answers that empirically. It takes a real rigged, animated GLB
(Khronos ships CC-licensed ones — CesiumMan is a walking human) and tries every
plausible argument form the command might parse: bare path, quoted path,
file:// URL, JSON payload, name=value pairs. After each attempt it reads the
status back, because the command returning "ok" while leaving the actor
`attached=false` is precisely how the gait capability got misreported twice.

    .venv/bin/python tools/probe_glb.py [glb_path] [port]

Exit codes: 0 a form worked, 2 command absent, 3 nothing accepted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from unrealcv import Client

TEST_ACTOR = "glb_probe_actor"
CHARACTER = "/Game/SmartLocomotion/Blueprints/BP_Character.BP_Character_C"
DEFAULT_GLB = Path.home() / "data" / "glb" / "CesiumMan.glb"


def argument_forms(actor: str, glb: Path) -> list[tuple[str, str]]:
    """Every plausible spelling, labelled so the report says which one worked."""
    p = str(glb)
    return [
        ("bare path", f"vset /animation/soma_glb/apply {actor} {p}"),
        ("quoted path", f'vset /animation/soma_glb/apply {actor} "{p}"'),
        ("file url", f"vset /animation/soma_glb/apply {actor} file://{p}"),
        ("path then loop", f"vset /animation/soma_glb/apply {actor} {p} 1"),
        ("json payload", "vset /animation/soma_glb/apply {}".format(
            json.dumps({"actor": actor, "path": p, "loop": True}))),
        ("kv pairs", f"vset /animation/soma_glb/apply actor={actor} path={p}"),
        ("stem only", f"vset /animation/soma_glb/apply {actor} {glb.stem}"),
    ]


def main(argv: list[str]) -> int:
    glb = Path(argv[0]) if argv else DEFAULT_GLB
    port = int(argv[1]) if len(argv) > 1 else 9900

    if not glb.is_file():
        print(f"no GLB at {glb}")
        return 1

    client = Client(("127.0.0.1", port))
    client.connect(timeout=30)
    if not client.isconnected():
        print("not connected — is the packaged build running?")
        return 1

    def req(cmd: str, timeout: float = 60) -> str:
        return str(client.request(cmd, timeout=timeout))

    try:
        if "/animation/soma_glb/apply" not in req("vget /unrealcv/list_cmd", timeout=120):
            print("this build has no soma_glb command")
            return 2

        spawned = req(f"vset /objects/spawn_from_path {CHARACTER} {TEST_ACTOR}", timeout=90).strip()
        actor = spawned if spawned and "error" not in spawned.lower() else TEST_ACTOR
        print(f"probe actor: {actor}\nGLB:         {glb} ({glb.stat().st_size / 1e3:.0f} kB)\n")

        working: list[str] = []
        for label, cmd in argument_forms(actor, glb):
            reply = req(cmd).strip()
            # The reply alone is not evidence. Read the status back: a command
            # that answers "ok" and leaves the actor unattached has done nothing.
            status = req(f"vget /animation/soma_glb/status {actor}").strip()
            attached = "true" in status.lower().split("attached")[-1][:12] if "attached" in status.lower() else False
            print(f"  {label:<14} reply={reply[:60]!r}")
            print(f"  {'':<14} status={status[:90]!r}")
            if attached:
                working.append(label)

        req(f"vset /objects/delete {actor}")

        if working:
            print(f"\nACCEPTED argument form(s): {working}")
            return 0
        print("\nNo argument form attached the animation. This build does not "
              "drive gait via soma_glb; treat the capability as unavailable.")
        return 3
    finally:
        client.disconnect()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

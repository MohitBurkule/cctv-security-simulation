"""Enumerate the human assets a build can actually spawn.

With no parametric wardrobe, appearance diversity has to come from *different
character assets*. This lists what exists so the population generator can draw
from real variety instead of cloning one mesh.

    .venv/bin/python tools/list_characters.py [port]
"""

from __future__ import annotations

import json
import re
import sys

from unrealcv import Client

PERSON_HINTS = ("character", "human", "metahuman", "person", "man", "woman",
                "ped", "citizen", "civilian", "npc", "mannequin", "bp_char")


def main(port: int = 9900) -> int:
    client = Client(("127.0.0.1", port))
    client.connect(timeout=30)
    if not client.isconnected():
        print("not connected")
        return 1

    def req(cmd: str, timeout: float = 90) -> str:
        return str(client.request(cmd, timeout=timeout))

    print("=== metahuman/all_paths ===")
    mh = req("vget /metahuman/all_paths", timeout=120)
    print(mh[:1500] if mh.strip() else "(empty)")

    print("\n=== metahuman/filter_batch ===")
    print(req("vget /metahuman/filter_batch", timeout=120)[:800])

    print("\n=== pak files ===")
    paks = req("vget /pak/files", timeout=120)
    print(paks[:600])

    print("\n=== scanning pak asset paths for people ===")
    found: set[str] = set()
    for package in ("/Game", "/Game/MetaHumans", "/Game/SmartLocomotion", "/Game/Characters"):
        try:
            res = req(f"vget /pak/assets {package}", timeout=120)
        except Exception as exc:  # noqa: BLE001
            print(f"  {package}: {type(exc).__name__}")
            continue
        paths = re.findall(r"/Game[A-Za-z0-9_/.\-]*", res)
        people = [p for p in paths if any(h in p.lower() for h in PERSON_HINTS)]
        found.update(people)
        print(f"  {package}: {len(paths)} assets, {len(people)} person-like")

    print(f"\n=== {len(found)} distinct person-like assets ===")
    for path in sorted(found)[:60]:
        print(" ", path)

    if found:
        with open("out/character_assets.json", "w") as fh:
            json.dump(sorted(found), fh, indent=1)
        print("\nwritten out/character_assets.json")

    client.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 9900))

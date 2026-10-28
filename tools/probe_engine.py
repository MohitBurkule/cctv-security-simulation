"""Probe a running UnrealZoo build for the capabilities SurvSim depends on.

Run this once against any new build before trusting a dataset produced from it:
if parametric appearance is missing, every rendered person is a clone and the
footage cannot train an appearance model.

    .venv/bin/python tools/probe_engine.py [port]
"""

from __future__ import annotations

import json
import re
import sys

from unrealcv import Client

from survsim.backends.unreal.appearance import AppearanceDriver

KEYWORDS = ("metahuman", "soma", "nav", "wardrobe", "parametric", "animation")


def main(port: int = 9900) -> int:
    client = Client(("127.0.0.1", port))
    client.connect(timeout=30)
    if not client.isconnected():
        print(f"could not connect on {port}; is the packaged build running?")
        return 1

    def req(cmd: str, timeout: float = 60) -> str:
        return str(client.request(cmd, timeout=timeout))

    print("level:", req("vget /level/name", timeout=45).strip())

    listing = req("vget /unrealcv/list_cmd", timeout=120)
    print(f"command listing: {len(listing)} bytes")
    for kw in KEYWORDS:
        hits = sorted(set(re.findall(r"/[A-Za-z0-9_\[\]/]*" + kw + r"[A-Za-z0-9_\[\]/]*", listing)))
        print(f"  {kw:11s} {len(hits):3d} {hits[:5]}")

    driver = AppearanceDriver(req)
    caps = driver.probe()
    print("capabilities:", json.dumps(caps.as_dict(), indent=1))

    if caps.body or caps.wardrobe:
        print("metahuman paths:", req("vget /metahuman/all_paths", timeout=90)[:400])
    else:
        print("NOTE: no parametric appearance — spawned people will be identical.")

    client.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 9900))

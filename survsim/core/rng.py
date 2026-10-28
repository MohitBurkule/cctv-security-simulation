"""Seeded randomness.

Every subsystem draws from its own generator derived from the master seed and the
subsystem name, so adding a camera cannot change which trousers a pedestrian wears.
Determinism is a hard requirement: a scenario must replay identically from its seed.
"""

from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(master: int, subsystem: str) -> int:
    """Stable 63-bit seed for a named subsystem. Independent of import order."""
    h = hashlib.blake2b(f"{master}:{subsystem}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") >> 1


class RngHub:
    """Named generators, created on demand and cached."""

    def __init__(self, master: int) -> None:
        self.master = int(master)
        self._cache: dict[str, np.random.Generator] = {}

    def get(self, subsystem: str) -> np.random.Generator:
        if subsystem not in self._cache:
            self._cache[subsystem] = np.random.default_rng(derive_seed(self.master, subsystem))
        return self._cache[subsystem]

    def __call__(self, subsystem: str) -> np.random.Generator:
        return self.get(subsystem)

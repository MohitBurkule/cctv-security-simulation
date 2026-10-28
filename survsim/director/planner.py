"""Local navigation: steer toward a goal while keeping people out of each other.

A reciprocal-velocity model (the cheap half of ORCA) is enough at the Director
level — the engine's own avoidance handles the last few centimetres and the
animation. Speeds stay inside the 0.8-1.8 m/s band that the pedestrian
literature reports for real walkers.
"""

from __future__ import annotations

import numpy as np

PERSONAL_SPACE_M = 0.45


def steer(
    pos: np.ndarray,
    goal: np.ndarray,
    speed: float,
    others: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a velocity in m/s toward `goal`, pushed away from `others` (Nx2)."""
    to_goal = goal - pos
    dist = float(np.linalg.norm(to_goal))
    if dist < 1e-6:
        return np.zeros(2)
    desired = to_goal / dist * min(speed, dist)

    if others is not None and len(others):
        delta = pos - others
        d = np.linalg.norm(delta, axis=1)
        close = (d > 1e-6) & (d < PERSONAL_SPACE_M * 2.5)
        if close.any():
            # Repulsion falls off with distance; strongest right at personal space.
            weights = (PERSONAL_SPACE_M * 2.5 - d[close]) / (PERSONAL_SPACE_M * 2.5)
            push = (delta[close] / d[close, None] * weights[:, None]).sum(axis=0)
            desired = desired + push * speed * 0.8
            # Pure repulsion is symmetric, so two people meeting head-on (or a
            # walker behind a stationary one) cancel to zero and stand locked.
            # Keep-right yield rule: sidestep perpendicular to the heading when
            # someone is ahead, which breaks the symmetry deterministically.
            heading = to_goal / dist
            ahead = (-delta[close] @ heading) > 0.0
            if ahead.any():
                right = np.array([heading[1], -heading[0]])
                desired = desired + right * speed * 0.5 * float(weights[ahead].max())

    if rng is not None:
        desired = desired + rng.normal(0.0, 0.02, size=2)

    mag = float(np.linalg.norm(desired))
    return desired / mag * speed if mag > speed else desired


def arrived(pos: np.ndarray, goal: np.ndarray, tol: float = 0.5) -> bool:
    return bool(np.linalg.norm(goal - pos) <= tol)

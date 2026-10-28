"""Background crowd: many people, cheaply, with honest ground truth.

Hero agents get full behaviour trees, appearance epochs and per-joint truth, and
they cost enough that a few dozen is the practical limit. But a concourse at
rush hour has hundreds of people, and their presence is not decoration — it is
the thing that makes the problem hard. Crowding causes most real occlusion, it is
where trackers swap identities, and it is the reason a detector tuned on sparse
footage collapses at an event.

So background agents are simulated with a cheap flow model rather than behaviour
trees: they follow a route through the site, avoid each other locally, and leave.
They produce boxes and identities but no behaviour labels, and the ground truth
says so — a background agent is explicitly marked, so nobody can accidentally
train a behaviour classifier on people who were never given a behaviour.

Density is specified in the way the literature and the operators both use it:
**people per square metre**. 0.5/m² is a busy pavement, 2/m² is a crowded
concourse, 4/m² is the level at which crowd-safety guidance starts to worry, and
above 5/m² people cannot move independently at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

# Levels of service, roughly after the pedestrian planning literature. These are
# the numbers a venue's safety case is written against, so a crowd simulator that
# cannot express them cannot answer the question a venue actually asks.
DENSITY_BANDS = (
    ("free_flow", 0.0, 0.3),
    ("busy", 0.3, 0.8),
    ("crowded", 0.8, 2.0),
    ("dense", 2.0, 4.0),
    ("critical", 4.0, 99.0),
)


def density_band(people_per_m2: float) -> str:
    for name, low, high in DENSITY_BANDS:
        if low <= people_per_m2 < high:
            return name
    return DENSITY_BANDS[-1][0]


# Density at which movement stops entirely.
#
# [V] Checked against real footage. Fitting `v = v0(1 - d/d_jam)` to 206k speed
# samples from 1,476 tracks across the four annotated MOT20 sequences gives
# d_jam = 4.53 /m², and the *shape* of our curve matches the measured one to
# within 7% across 0.15–2.0 /m² (see `survsim/gait/extract.py` and docs/14).
# 5.4 is retained as the default because it is the value the pedestrian-planning
# literature uses and because a single crowded dataset should not overturn it,
# but it is now a parameter rather than a buried constant.
#
# The free speed could NOT be validated the same way: that measurement returned
# 0.64 m/s, which is an artefact of measuring displacement in the image plane.
# Motion along the camera axis is foreshortened, so radial walking reads slow,
# and MOT20 ships no calibration to correct it. Shape survives that bias because
# it is scale-free; absolute speed does not.
JAM_DENSITY_PER_M2 = 5.4


def speed_for_density(free_speed: float, people_per_m2: float,
                      jam_density: float = JAM_DENSITY_PER_M2) -> float:
    """Walking speed falls as density rises — the fundamental diagram.

    Roughly linear until movement becomes impossible. This is why a crowded scene
    is not simply a sparse scene with more people in it: everyone is *slower*, and
    motion-based detection degrades accordingly.
    """
    factor = max(0.05, 1.0 - people_per_m2 / max(jam_density, 1e-6))
    return float(free_speed * factor)


# Free-walking gait, from the pedestrian literature.
FREE_CADENCE_SPM = 110.0
FREE_STEP_LENGTH_M = 0.75


def gait_for_density(people_per_m2: float,
                     free_cadence_spm: float = FREE_CADENCE_SPM,
                     jam_density: float = JAM_DENSITY_PER_M2,
                     ) -> tuple[float, float]:
    """Cadence and step length at a given density. Returns (steps/min, metres).

    **A slow crowd is not a slow-motion crowd**, and this is the measurement
    that says so. Pose-estimated gait from MOT20 (`survsim/gait/cadence.py`,
    docs/14 §10) gives 123 steps/min with a 0.39 m step, against roughly
    110 spm and 0.75 m for free walking. People in a crowd keep their leg
    rhythm — if anything they raise it slightly — and take **much shorter
    steps**.

    That distinction is invisible in `speed_for_density`, which only knows that
    speed halved, and it matters in three places:

      - **Animation.** Playing a walk cycle at half rate to reach half speed
        produces a person moving in slow motion, which no pedestrian does and
        which any gait-based system would find obviously synthetic.
      - **Gait biometrics.** Cadence is roughly *preserved* under crowding
        while speed is not, making it the more stable identity cue — and the
        thing a deliberate gait change has to defeat.
      - **Motion detection.** Limb motion stays fast even when the body
        barely translates, so per-frame motion energy does not fall with speed.

    Cadence is modelled as slightly rising with density and step length as
    absorbing the rest, so that cadence x step length still reproduces
    `speed_for_density` — the three quantities must satisfy their identity.
    """
    speed = speed_for_density(free_cadence_spm / 60.0 * FREE_STEP_LENGTH_M,
                              people_per_m2, jam_density)

    # Measured: cadence at ~0.7/m^2 was ~1.12x the free-walking figure. Modelled
    # as a mild rise saturating quickly, because people cannot indefinitely
    # increase step rate as space runs out.
    crowding = min(people_per_m2 / max(jam_density, 1e-6), 1.0)
    cadence = free_cadence_spm * (1.0 + 0.15 * min(crowding * 3.0, 1.0))

    # Step length takes the remainder, so the identity holds by construction.
    step_length = speed / max(cadence / 60.0, 1e-6)
    return float(cadence), float(max(step_length, 0.02))


@dataclass
class CrowdAgent:
    """A background person: position, goal, and nothing else."""

    uid: str
    pos: np.ndarray
    goal: np.ndarray
    free_speed: float
    height_m: float
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    active: bool = True

    @property
    def pos3(self) -> np.ndarray:
        return np.array([self.pos[0], self.pos[1], 0.0])


@dataclass
class CrowdConfig:
    target_density_per_m2: float = 0.4
    area_m2: float = 1000.0
    free_speed_mean: float = 1.34       # the classic mean pedestrian speed
    free_speed_std: float = 0.22
    personal_space_m: float = 0.45
    respawn: bool = True                 # keep density steady as people leave
    seed: int = 0

    @property
    def target_count(self) -> int:
        return max(0, int(round(self.target_density_per_m2 * self.area_m2)))


class BackgroundCrowd:
    """A cheap flow-model crowd that keeps a target density."""

    def __init__(self, config: CrowdConfig, entries: Sequence[np.ndarray],
                 exits: Sequence[np.ndarray]) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.entries = [np.asarray(p, dtype=float)[:2] for p in entries] or [np.zeros(2)]
        self.exits = [np.asarray(p, dtype=float)[:2] for p in exits] or [np.zeros(2)]
        self.agents: list[CrowdAgent] = []
        self._next_id = 0
        self._spawn(self.cfg.target_count, scatter=True)

    # ------------------------------------------------------------- spawning

    def _spawn(self, n: int, scatter: bool = False) -> None:
        for _ in range(n):
            entry = self.entries[self.rng.integers(len(self.entries))]
            goal = self.exits[self.rng.integers(len(self.exits))]
            if scatter:
                # Seed the crowd spread along its route rather than piled at the
                # entrance, so the first frames are already a realistic scene.
                alpha = float(self.rng.uniform(0.0, 1.0))
                start = entry + (goal - entry) * alpha
                start = start + self.rng.normal(0.0, 2.0, size=2)
            else:
                start = entry + self.rng.normal(0.0, 1.0, size=2)

            self.agents.append(CrowdAgent(
                uid=f"bg_{self._next_id:04d}", pos=start, goal=goal,
                free_speed=float(np.clip(
                    self.rng.normal(self.cfg.free_speed_mean, self.cfg.free_speed_std),
                    0.6, 2.0)),
                height_m=float(np.clip(self.rng.normal(1.70, 0.09), 1.2, 2.05)),
            ))
            self._next_id += 1

    # ----------------------------------------------------------------- step

    def step(self, dt: float) -> None:
        """Advance the crowd one tick."""
        if not self.agents:
            return
        positions = np.array([a.pos for a in self.agents])
        density = len(self.agents) / max(self.cfg.area_m2, 1.0)

        for i, agent in enumerate(self.agents):
            if not agent.active:
                continue
            to_goal = agent.goal - agent.pos
            distance = float(np.linalg.norm(to_goal))
            if distance < 1.5:
                agent.active = False
                continue

            speed = speed_for_density(agent.free_speed, density)
            desired = to_goal / distance * speed

            # Local avoidance against near neighbours only: the cost of checking
            # everyone against everyone grows quadratically and buys nothing.
            delta = agent.pos - positions
            dist = np.linalg.norm(delta, axis=1)
            close = (dist > 1e-6) & (dist < self.cfg.personal_space_m * 3.0)
            if close.any():
                weights = ((self.cfg.personal_space_m * 3.0 - dist[close])
                           / (self.cfg.personal_space_m * 3.0))
                push = (delta[close] / dist[close, None] * weights[:, None]).sum(axis=0)
                desired = desired + push * speed * 0.9

            magnitude = float(np.linalg.norm(desired))
            if magnitude > speed:
                desired = desired / magnitude * speed
            agent.vel = desired
            agent.pos = agent.pos + desired * dt

        if self.cfg.respawn:
            departed = [a for a in self.agents if not a.active]
            self.agents = [a for a in self.agents if a.active]
            # Replace those who left, so density stays at the level asked for
            # rather than draining away over a long run.
            self._spawn(len(departed))

    # ------------------------------------------------------------ reporting

    @property
    def density(self) -> float:
        return len(self.agents) / max(self.cfg.area_m2, 1.0)

    def rows(self, tick: int, t: float) -> list[dict[str, Any]]:
        """Ground-truth rows, explicitly marked as background.

        No behaviour label is emitted. A background agent was never given a
        behaviour, and inventing "normal" for them would quietly pad any
        behaviour dataset with thousands of unlabelled negatives.
        """
        return [
            {
                "frame": tick, "t": round(t, 3), "uid": a.uid,
                "x": round(float(a.pos[0]), 4), "y": round(float(a.pos[1]), 4),
                "speed": round(float(np.linalg.norm(a.vel)), 4),
                "height_m": a.height_m,
                "background": True,
                "label": None,
            }
            for a in self.agents if a.active
        ]

    def report(self) -> dict[str, Any]:
        speeds = [float(np.linalg.norm(a.vel)) for a in self.agents if a.active]
        return {
            "n_agents": sum(a.active for a in self.agents),
            "density_per_m2": round(self.density, 3),
            "band": density_band(self.density),
            "mean_speed": round(float(np.mean(speeds)), 3) if speeds else 0.0,
            "free_speed_mean": self.cfg.free_speed_mean,
            "note": ("background agents carry no behaviour label by design; they "
                     "exist to create occlusion and density, not to be classified"),
        }

"""Fenced zones: restricted and sterile-strip polygons are physical obstacles.

H40 (docs/research/32) found innocent commuters routed straight through
restricted polygons, because nothing in the mock made a polygon anything more
than a label. A real restricted area has a fence and a gate. This module gives
the Director that fence:

* every zone tagged ``restricted`` or ``perimeter_fence`` is fenced, and
  overlapping fenced polygons merge into one *component* (a sterile strip nested
  in a compound is one fenced area, not two);
* each component has one gate, taken from a ``gate`` prop on its boundary if the
  environment has one, else the boundary point nearest the site centre;
* people whose role is not in any of the component's ``allowed_roles`` cannot
  step from outside to inside (`World.step` enforces it on the *actual*
  position, principle 1), and ordinary planners route round the fence with a
  small visibility graph instead of walking into it;
* behaviours that genuinely cross (``crosses_fence``) are exempt: an intruder
  climbs or cuts, an animal slips under.

Leaving a fenced area is never blocked — a trapped innocent is a worse artefact
than a fence with a push-bar exit.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

FENCED_TAGS = ("restricted", "perimeter_fence")
GATE_OUT_M = 1.2        # stand-off in front of the gate, outside
GATE_IN_M = 1.5         # first step inside the gate
CLEAR_M = 0.9           # detour corner clearance
HIT_M = 0.6             # a path segment closer than this to a fence is blocked


@dataclass
class Component:
    poly: Polygon
    zone_ids: list[str]
    allowed_roles: set[str]
    gate: np.ndarray                     # on the boundary
    gate_out: np.ndarray
    gate_in: np.ndarray
    bounds: tuple[float, float, float, float] = field(default=(0, 0, 0, 0))
    hit: Any = None                      # prepared buffered polygon for path tests
    corners: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    def contains(self, p: np.ndarray) -> bool:
        x, y = float(p[0]), float(p[1])
        x0, y0, x1, y1 = self.bounds
        if x < x0 or x > x1 or y < y0 or y > y1:
            return False
        return bool(shapely.contains_xy(self.poly, x, y))


def outward_normal(poly: Polygon, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(nearest boundary point, unit outward normal of the nearest edge)."""
    coords = np.asarray(poly.exterior.coords)
    best = (np.inf, None, None)
    for a, b in itertools.pairwise(coords):
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            continue
        s = float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
        q = a + s * ab
        d = float(np.linalg.norm(p - q))
        if d < best[0]:
            best = (d, q, ab)
    _, q, ab = best
    n = np.array([ab[1], -ab[0]]) / np.linalg.norm(ab)
    if poly.contains(Point(*(q + 0.05 * n))):
        n = -n
    return np.asarray(q, float), n


def deep_point(poly: Polygon, near: np.ndarray, depth_m: float) -> tuple[np.ndarray, float]:
    """The point >= depth_m inside `poly` nearest to `near` (or the deepest point)."""
    inner = poly.buffer(-depth_m)
    if inner.is_empty:
        # Smaller than twice the depth: the most interior point there is.
        lo, hi = 0.0, depth_m
        for _ in range(20):
            mid = (lo + hi) / 2
            if poly.buffer(-mid).is_empty:
                hi = mid
            else:
                lo = mid
        inner = poly.buffer(-lo)
        depth_m = lo
    from shapely.ops import nearest_points
    q = nearest_points(inner, Point(float(near[0]), float(near[1])))[0]
    c = np.array([q.x, q.y])
    # step slightly further in so float error never leaves it at depth - eps
    ctr = np.array(inner.centroid.coords[0])
    v = ctr - c
    n = float(np.linalg.norm(v))
    if n > 1e-6:
        c = c + v / n * min(0.3, n)
    return c, depth_m


class Fences:
    def __init__(self, zones: list[Any], props: list[Any]) -> None:
        polys = [(z, Polygon(z.polygon)) for z in zones
                 if z.tag in FENCED_TAGS and getattr(z, "fenced", True) is not False]
        self.components: list[Component] = []
        if not polys:
            return
        merged = unary_union([p for _, p in polys])
        geoms = list(getattr(merged, "geoms", [merged]))
        gates = [np.array(p.pos[:2], float) for p in props if p.type == "gate"]
        for g in geoms:
            if not isinstance(g, Polygon) or g.area <= 0:
                continue
            g = Polygon(g.exterior)
            members = [(z, p) for z, p in polys if g.buffer(1e-6).contains(p.representative_point())]
            roles = set().union(*[set(z.allowed_roles or []) for z, _ in members]) if members else set()
            gate = None
            for gp in gates:
                if g.exterior.distance(Point(*gp)) <= 1.0:
                    gate = gp
                    break
            if gate is None:
                gate = np.array(g.exterior.interpolate(g.exterior.project(Point(0.0, 0.0))).coords[0])
            q, n = outward_normal(g, gate)
            comp = Component(poly=g, zone_ids=[z.id for z, _ in members], allowed_roles=roles,
                             gate=q, gate_out=q + n * GATE_OUT_M, gate_in=q - n * GATE_IN_M,
                             bounds=g.bounds)
            comp.hit = g.buffer(HIT_M, join_style=2)
            shapely.prepare(comp.hit)
            comp.corners = np.asarray(g.buffer(CLEAR_M, join_style=2).exterior.coords)[:-1]
            self.components.append(comp)

    def __bool__(self) -> bool:
        return bool(self.components)

    # ------------------------------------------------------------------ queries

    def component_of_zone(self, zone_id: str) -> Component | None:
        return next((c for c in self.components if zone_id in c.zone_ids), None)

    def containing(self, p: np.ndarray) -> Component | None:
        return next((c for c in self.components if c.contains(p)), None)

    def nearest(self, p: np.ndarray) -> Component | None:
        if not self.components:
            return None
        pt = Point(float(p[0]), float(p[1]))
        return min(self.components, key=lambda c: c.poly.distance(pt))

    def inside_any(self, p: np.ndarray) -> bool:
        return any(c.contains(p) for c in self.components)

    # ------------------------------------------------------------ enforcement

    def clamp(self, prev: np.ndarray, new: np.ndarray, role: str) -> np.ndarray:
        """The position actually reached when stepping prev -> new against the fences."""
        for c in self.components:
            if role in c.allowed_roles or not c.contains(new) or c.contains(prev):
                continue
            # Slide along the fence (wall-following): keep the tangential part
            # of the step; head-on, turn right. Any behaviour steering straight
            # at a goal beyond a convex fence therefore works its way round.
            d = new - prev
            step = float(np.linalg.norm(d))
            _, n = outward_normal(c.poly, prev)
            tng = np.array([n[1], -n[0]])
            along = float(d @ tng)
            if abs(along) < 0.3 * step:
                right = np.array([d[1], -d[0]])
                along = step if float(right @ tng) >= 0.0 else -step
            cands = [prev + tng * along, np.array([new[0], prev[1]]), np.array([prev[0], new[1]])]
            for cand in cands:
                if not c.contains(cand) and not self.inside_any_blocking(cand, prev, role):
                    return cand
            return prev.copy()
        return new

    def inside_any_blocking(self, p: np.ndarray, prev: np.ndarray, role: str) -> bool:
        return any(c.contains(p) and not c.contains(prev) and role not in c.allowed_roles
                   for c in self.components)

    # ---------------------------------------------------------------- routing

    def route(self, start: np.ndarray, goal: np.ndarray) -> list[np.ndarray]:
        """Waypoints (ending at `goal`) that keep a walker outside every fence.

        Components that contain, or nearly contain, the start or the goal are
        not obstacles for this leg (the walker is inside, or is walking to a gate
        on the fence line). Empty list = go straight.
        """
        if not self.components:
            return []
        s, g = Point(float(start[0]), float(start[1])), Point(float(goal[0]), float(goal[1]))
        obs = [c for c in self.components
               if c.poly.distance(s) > CLEAR_M and c.poly.distance(g) > CLEAR_M]
        if not obs:
            return []
        seg = LineString([s, g])
        if not any(c.hit.intersects(seg) for c in obs):
            return []
        nodes = [np.asarray(start, float), np.asarray(goal, float)]
        for c in obs:
            for q in c.corners:
                if not any(o.hit.contains(Point(*q)) for o in obs):
                    nodes.append(np.asarray(q, float))
        n = len(nodes)

        def free(i: int, j: int) -> bool:
            ln = LineString([nodes[i], nodes[j]])
            return not any(c.hit.intersects(ln) for c in obs)

        dist = [np.inf] * n
        prev = [-1] * n
        dist[0] = 0.0
        pq = [(0.0, 0)]
        done = [False] * n
        while pq:
            d, i = heapq.heappop(pq)
            if done[i]:
                continue
            done[i] = True
            if i == 1:
                break
            for j in range(n):
                if done[j] or j == i:
                    continue
                w = float(np.linalg.norm(nodes[i] - nodes[j]))
                if d + w < dist[j] and free(i, j):
                    dist[j], prev[j] = d + w, i
                    heapq.heappush(pq, (d + w, j))
        if not np.isfinite(dist[1]):
            return []
        path, k = [], 1
        while k != 0:
            path.append(nodes[k].copy())
            k = prev[k]
        return path[::-1]

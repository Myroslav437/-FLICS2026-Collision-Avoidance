"""
World-level metrics.

Computed from a single W_0 without running the simulator:

  free_space_ratio     : fraction of X*Y not occupied by E.
  min_passage_width    : narrowest gap between two boundary elements
                         measured along the reference path pi.
  static_obstacle_count, dynamic_obstacle_count : counts of O_{t=0}.
  path_length          : total length of pi.

Validation checks that use these metrics:
  - free_space_ratio should decrease monotonically with d_bsp;
  - min_passage_width should cluster near w_pass - t_wall by construction;
  - min_passage_width must exceed W_agv + 2*d_clear in every retained world
    (navigability-safety check).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .models import WorldState


@dataclass
class WorldMetrics:
    free_space_ratio: float
    min_passage_width: float
    static_obstacle_count: int
    dynamic_obstacle_count: int
    path_length: float
    min_corridor_width: float = float("inf")
    prm_reference_attempts: int = -1

    def to_dict(self) -> dict:
        return {
            "free_space_ratio": float(self.free_space_ratio),
            "min_passage_width": float(self.min_passage_width),
            "static_obstacle_count": int(self.static_obstacle_count),
            "dynamic_obstacle_count": int(self.dynamic_obstacle_count),
            "path_length": float(self.path_length),
            "min_corridor_width": float(self.min_corridor_width),
            "prm_reference_attempts": int(self.prm_reference_attempts),
        }


# =============================================================================
# Geometry reconstruction from oriented chains
# =============================================================================

def _signed_area(chain: np.ndarray) -> float:
    """Shoelace signed area; positive for CCW, negative for CW."""
    x = chain[:, 0]
    y = chain[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def environment_occupied_geometry(world: WorldState) -> BaseGeometry:
    """
    Reconstruct the occupied shapely geometry from E's oriented chains.

    BSP exports chains with shapely's orientation convention (CCW = outer
    boundary of an occupied region; CW = hole inside that region). We
    group each CW hole with the smallest enclosing CCW exterior to
    produce valid polygons-with-holes, then union them.
    """
    exteriors: List[np.ndarray] = []
    holes: List[np.ndarray] = []
    for chain in world.environment.chains:
        if chain.shape[0] < 3:
            continue
        if _signed_area(chain) >= 0:
            exteriors.append(chain)
        else:
            holes.append(chain)

    if not exteriors:
        return unary_union([])

    # Build exterior polygons (no holes yet) and assign each hole to the
    # smallest exterior that contains the hole's first vertex.
    exterior_polys = [Polygon(e) for e in exteriors]
    sizes = [p.area for p in exterior_polys]
    grouped_holes: List[List[np.ndarray]] = [[] for _ in exterior_polys]
    for hole in holes:
        pt = Point(float(hole[0, 0]), float(hole[0, 1]))
        candidates = [
            (sizes[i], i)
            for i, poly in enumerate(exterior_polys)
            if poly.contains(pt)
        ]
        if not candidates:
            continue  # defensive; shouldn't happen for BSP output
        _, idx = min(candidates)
        grouped_holes[idx].append(hole)

    polys = [
        Polygon(ext, holes=grouped_holes[i])
        for i, ext in enumerate(exteriors)
    ]
    return unary_union(polys)


# =============================================================================
# Metrics
# =============================================================================

def free_space_ratio(world: WorldState) -> float:
    """1 - area(E) / (X * Y) (free-space check)."""
    X, Y = world.environment.size
    occupied = environment_occupied_geometry(world)
    return float(1.0 - occupied.area / (X * Y))


def min_passage_width(world: WorldState) -> float:
    """
    Minimum passage width measured along pi.

    At each sample point p on pi with local heading theta (the direction
    pi is travelling at p), we cast two rays perpendicular to theta
    (theta +/- pi/2) until they hit the occupied boundary. Their combined
    length is the local corridor width -- the physical gap between the
    two boundary segments flanking the AGV. We sample pi at 0.1 m
    intervals and return the minimum.

    This is the "gap between boundary segments on pi" used for the
    navigability-safety check; by construction it clusters near w_pass
    when pi crosses a doorway of width w_pass.
    """
    from shapely.geometry import LineString

    occupied = environment_occupied_geometry(world)
    if occupied.is_empty:
        return float("inf")
    boundary = occupied.boundary

    path = world.reference_path.positions
    if path.shape[0] < 2:
        return float("inf")
    X, Y = world.environment.size
    ray_length = float(np.hypot(X, Y))  # upper bound guaranteed to cross any wall

    step = 0.1
    best = float("inf")
    for i in range(path.shape[0] - 1):
        a, b = path[i], path[i + 1]
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        if seg_len <= 0:
            continue
        tangent = seg / seg_len
        normal = np.array([-tangent[1], tangent[0]])
        n_samples = max(2, int(np.ceil(seg_len / step)))
        ts = np.linspace(0.0, 1.0, n_samples)
        for t in ts:
            p = a + t * seg
            width = _corridor_width_at(p, normal, ray_length, boundary)
            if width < best:
                best = width
    return float(best)


def _corridor_width_at(
    p: np.ndarray, normal: np.ndarray, ray_length: float, boundary
) -> float:
    """
    Cast perpendicular rays from `p` along +/- `normal` and return the sum
    of the distances to the first intersection with `boundary`. A ray with
    no intersection (path outside walls; shouldn't happen on pi) falls
    back to `ray_length`.
    """
    from shapely.geometry import LineString, Point

    widths = 0.0
    for sign in (+1.0, -1.0):
        end = p + sign * normal * ray_length
        ray = LineString([tuple(p), tuple(end)])
        inter = ray.intersection(boundary)
        if inter.is_empty:
            widths += ray_length
            continue
        # Take the intersection point nearest to p.
        pt = Point(float(p[0]), float(p[1]))
        if hasattr(inter, "geoms"):
            d = min(pt.distance(g) for g in inter.geoms)
        else:
            d = pt.distance(inter)
        widths += d
    return widths


def compute_metrics(world: WorldState) -> WorldMetrics:
    # The BSP-stage corridor observation and the l_pi_min rejection-loop
    # attempt count are attached to `world` in-process by generate_world;
    # they are not serialized on WorldState. If the world was reloaded
    # from disk the attributes are absent; we report inf / -1 to signal
    # "not measured in this process".
    bsp_mc = getattr(world, "_bsp_min_corridor_width", float("inf"))
    attempts = getattr(world, "_prm_reference_attempts", -1)
    return WorldMetrics(
        free_space_ratio=free_space_ratio(world),
        min_passage_width=min_passage_width(world),
        static_obstacle_count=len(world.static_obstacles),
        dynamic_obstacle_count=len(world.dynamic_obstacles),
        path_length=world.reference_path.total_length(),
        min_corridor_width=float(bsp_mc),
        prm_reference_attempts=int(attempts),
    )

"""
BSP-based environment generation.

Implements the recursive Binary Space Partitioning procedure: boundary
elements (walls) are placed along partition edges, and rooms or passages
are carved within leaf partitions. t_wall is the thickness of boundary
elements placed along partition edges and room perimeters.

Concretely, the generator outputs walls as thin (t_wall-thick) axis-aligned
slabs:

  (a) the domain perimeter;
  (b) one slab along every internal BSP partition line, with a w_pass-wide
      doorway carved through it;
  (c) one slab around every leaf's inner room, with a w_pass-wide doorway
      carved to the surrounding corridor.

Depth semantics (low values 1-2 yield open layouts):

  d_bsp = 0 : empty bounded hall - only the outer perimeter walls, no
              internal subdivision and no carved rooms.
  d_bsp = 1 : one carved room inside the bounded hall (0 BSP splits,
              1 leaf = the whole domain, 1 inner room).
  d_bsp = k : k-1 levels of recursive subdivision, then rooms carved in
              each leaf.

Internally the recursion takes an effective max depth of (d_bsp - 1) so
that the recursion tree at d_bsp = k has at most k levels of rooms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely.geometry.base import BaseGeometry

from .config import EnvironmentParams
from .models import EnvironmentStructure


Rect = Tuple[float, float, float, float]  # (x0, y0, x1, y1)


@dataclass
class _BSPNode:
    cell: Rect
    depth: int
    left: Optional["_BSPNode"] = None
    right: Optional["_BSPNode"] = None
    split_axis: Optional[int] = None  # 0 = vertical (x-split), 1 = horizontal (y-split)
    split_pos: Optional[float] = None
    room_rect: Optional[Rect] = None  # inner-room rectangle (leaf nodes only)

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


# =============================================================================
# BSP construction
# =============================================================================

def _can_split(cell: Rect, axis: int, params: EnvironmentParams) -> bool:
    x0, y0, x1, y1 = cell
    s_min_w, s_min_h = params.min_partition
    w, h = x1 - x0, y1 - y0
    rho_min, rho_max = params.split_ratio
    if axis == 0:
        return (rho_min * w >= s_min_w) and ((1.0 - rho_max) * w >= s_min_w) \
            and h >= s_min_h
    return (rho_min * h >= s_min_h) and ((1.0 - rho_max) * h >= s_min_h) \
        and w >= s_min_w


def _build_bsp(
    cell: Rect,
    depth: int,
    max_depth: int,
    params: EnvironmentParams,
    rng: np.random.Generator,
) -> _BSPNode:
    """Recursive BSP construction with alternating axes (axis = depth % 2)."""
    node = _BSPNode(cell=cell, depth=depth)
    if depth >= max_depth:
        return node

    axis = depth % 2
    chosen_axis = None
    for a in (axis, 1 - axis):
        if _can_split(cell, a, params):
            chosen_axis = a
            break
    if chosen_axis is None:
        return node

    x0, y0, x1, y1 = cell
    rho = float(rng.uniform(*params.split_ratio))
    if chosen_axis == 0:
        split_pos = x0 + rho * (x1 - x0)
        left_cell = (x0, y0, split_pos, y1)
        right_cell = (split_pos, y0, x1, y1)
    else:
        split_pos = y0 + rho * (y1 - y0)
        left_cell = (x0, y0, x1, split_pos)
        right_cell = (x0, split_pos, x1, y1)

    node.split_axis = chosen_axis
    node.split_pos = split_pos
    node.left = _build_bsp(left_cell, depth + 1, max_depth, params, rng)
    node.right = _build_bsp(right_cell, depth + 1, max_depth, params, rng)
    return node


def _collect_leaves(node: _BSPNode) -> List[_BSPNode]:
    if node.is_leaf:
        return [node]
    return _collect_leaves(node.left) + _collect_leaves(node.right)


def _collect_internal(node: _BSPNode) -> List[_BSPNode]:
    if node.is_leaf:
        return []
    return [node] + _collect_internal(node.left) + _collect_internal(node.right)


# =============================================================================
# Wall and doorway geometry
# =============================================================================

def _perimeter_walls(X: float, Y: float, t: float) -> BaseGeometry:
    """Four t-thick slabs forming the domain perimeter."""
    slabs = [
        box(0.0, 0.0, X, t),          # bottom
        box(0.0, Y - t, X, Y),        # top
        box(0.0, 0.0, t, Y),          # left
        box(X - t, 0.0, X, Y),        # right
    ]
    return unary_union(slabs)


def _partition_wall(node: _BSPNode, t: float) -> BaseGeometry:
    """Thin wall slab along the split line of an internal node."""
    x0, y0, x1, y1 = node.cell
    if node.split_axis == 0:  # vertical split at x = split_pos
        return box(node.split_pos - t / 2.0, y0, node.split_pos + t / 2.0, y1)
    return box(x0, node.split_pos - t / 2.0, x1, node.split_pos + t / 2.0)


def _partition_doorway(
    node: _BSPNode, params: EnvironmentParams, rng: np.random.Generator,
) -> Optional[BaseGeometry]:
    """
    Doorway rectangle punched through the partition wall at `node`'s split.

    Width w_pass parallel to the wall; thickness slightly wider than the
    wall slab so the carve-out cleanly removes the wall.
    """
    x0, y0, x1, y1 = node.cell
    t = params.wall_thickness
    w = params.passage_width
    if node.split_axis == 0:
        # Wall is vertical; doorway lies along y.
        lo = y0 + t + w / 2.0
        hi = y1 - t - w / 2.0
        if hi <= lo:
            return None
        yc = float(rng.uniform(lo, hi))
        return box(node.split_pos - t, yc - w / 2.0, node.split_pos + t, yc + w / 2.0)
    # Horizontal wall; doorway lies along x.
    lo = x0 + t + w / 2.0
    hi = x1 - t - w / 2.0
    if hi <= lo:
        return None
    xc = float(rng.uniform(lo, hi))
    return box(xc - w / 2.0, node.split_pos - t, xc + w / 2.0, node.split_pos + t)


def _room_rect(
    cell: Rect, params: EnvironmentParams, rng: np.random.Generator
) -> Tuple[Optional[Rect], float]:
    """
    Inner-room rectangle inside a leaf cell.

    Corridor-width enforcement (c_min):
      nominal_corridor = p_room * partition_size   (per axis)
      actual_corridor  = max(nominal_corridor, c_min)
      room_size        = partition_size - 2*actual_corridor - 2*t_wall

    If ``actual_corridor >= partition_size / 2`` (or the derived room size
    is non-positive) no room fits - return (None, inf) and the caller must
    skip room placement in this leaf. This is rare at reasonable parameter
    values.

    Returns
    -------
    (room_rect, actual_corridor_min)
        room_rect is None if no room fits. actual_corridor_min is the
        smaller of the two axis-aligned actual corridor values (inf if
        no room).
    """
    x0, y0, x1, y1 = cell
    w, h = x1 - x0, y1 - y0
    p = float(rng.uniform(*params.room_padding))
    t = params.wall_thickness
    c_min = params.min_corridor

    corridor_x = max(p * w, c_min)
    corridor_y = max(p * h, c_min)

    if corridor_x * 2.0 >= w or corridor_y * 2.0 >= h:
        return None, float("inf")
    room_w = w - 2.0 * corridor_x - 2.0 * t
    room_h = h - 2.0 * corridor_y - 2.0 * t
    if room_w <= 0.0 or room_h <= 0.0:
        return None, float("inf")

    pad_x = corridor_x + t
    pad_y = corridor_y + t
    return (
        (x0 + pad_x, y0 + pad_y, x1 - pad_x, y1 - pad_y),
        float(min(corridor_x, corridor_y)),
    )


def _room_wall(room: Rect, t: float) -> BaseGeometry:
    """t-thick wall slab around the inner room's perimeter (ring)."""
    rx0, ry0, rx1, ry1 = room
    outer = box(rx0 - t, ry0 - t, rx1 + t, ry1 + t)
    inner = box(rx0, ry0, rx1, ry1)
    return outer.difference(inner)


def _room_doorway(
    room: Rect, params: EnvironmentParams, rng: np.random.Generator,
) -> BaseGeometry:
    """
    Single doorway of width w_pass punched through one side of the inner
    room's wall, connecting the room interior to the surrounding corridor.

    The side is chosen uniformly; the doorway position along the chosen
    side is uniform within the valid range.
    """
    rx0, ry0, rx1, ry1 = room
    t = params.wall_thickness
    w = params.passage_width
    # Side: 0=bottom, 1=top, 2=left, 3=right
    sides: List[int] = []
    if (rx1 - rx0) >= w:
        sides.extend([0, 1])
    if (ry1 - ry0) >= w:
        sides.extend([2, 3])
    if not sides:
        # Degenerate: room smaller than w_pass. Fall back to any side with a
        # minimally-sized door; the door will slightly exceed the room width
        # but still cut the wall open so the room is reachable.
        sides = [0, 1, 2, 3]
    side = int(rng.choice(sides))
    if side in (0, 1):
        lo = rx0 + 0.0
        hi = rx1 - 0.0
        xc = float(rng.uniform(min(lo, hi - w), max(lo + w, hi)) if hi - lo >= w else (lo + hi) / 2.0)
        if side == 0:
            return box(xc - w / 2.0, ry0 - t, xc + w / 2.0, ry0 + t)
        return box(xc - w / 2.0, ry1 - t, xc + w / 2.0, ry1 + t)
    lo = ry0 + 0.0
    hi = ry1 - 0.0
    yc = float(rng.uniform(min(lo, hi - w), max(lo + w, hi)) if hi - lo >= w else (lo + hi) / 2.0)
    if side == 2:
        return box(rx0 - t, yc - w / 2.0, rx0 + t, yc + w / 2.0)
    return box(rx1 - t, yc - w / 2.0, rx1 + t, yc + w / 2.0)


# =============================================================================
# Chain extraction
# =============================================================================

def _polygon_to_chain(poly: Polygon) -> List[np.ndarray]:
    """
    Return the exterior and interior rings of a Polygon as vertex arrays,
    preserving shapely's orientation (CCW exterior, CW holes). Downstream
    code reconstructs the occupied-area signed sum using this convention.
    """
    oriented = orient(poly, sign=1.0)
    chains: List[np.ndarray] = []
    ext = np.asarray(oriented.exterior.coords)
    if len(ext) > 1 and np.allclose(ext[0], ext[-1]):
        ext = ext[:-1]
    chains.append(ext)
    for interior in oriented.interiors:
        ring = np.asarray(interior.coords)
        if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        chains.append(ring)
    return chains


def _geometry_to_chains(geom: BaseGeometry) -> List[np.ndarray]:
    chains: List[np.ndarray] = []
    if geom.is_empty:
        return chains
    if isinstance(geom, Polygon):
        return _polygon_to_chain(geom)
    if hasattr(geom, "geoms"):
        for sub in geom.geoms:
            chains.extend(_geometry_to_chains(sub))
        return chains
    raise TypeError(f"Unexpected geometry type: {type(geom)}")


# =============================================================================
# Top-level entry
# =============================================================================

@dataclass
class BSPEnvironment:
    """Result of `generate_environment`: E plus the derived free-space geometry.

    ``min_corridor_width`` records the smallest actual corridor width
    applied during room placement (c_min clamp). If no rooms were carved
    (e.g. d_bsp = 0, or every leaf skipped), the value is +inf.
    """

    environment: EnvironmentStructure
    free_space: BaseGeometry
    min_corridor_width: float = float("inf")


def generate_environment(
    params: EnvironmentParams, rng: np.random.Generator
) -> BSPEnvironment:
    """
    Procedural environment generation.

    Walls = domain perimeter union partition walls (with doorways) union
    inner-room walls (with doorways). Free space = domain - walls.
    """
    X, Y, t = params.X, params.Y, params.wall_thickness

    walls: List[BaseGeometry] = [_perimeter_walls(X, Y, t)]
    doorways: List[BaseGeometry] = []
    min_corridor_observed: float = float("inf")

    # d_bsp = 0: empty bounded hall. No BSP recursion, no carved rooms; only
    # the outer perimeter walls remain. d_bsp >= 1: recurse with effective
    # max depth = d_bsp - 1 so the recursion tree has at most d_bsp levels
    # of rooms.
    if params.bsp_depth >= 1:
        root = _build_bsp(
            cell=(0.0, 0.0, X, Y),
            depth=0,
            max_depth=params.bsp_depth - 1,
            params=params,
            rng=rng,
        )

        for node in _collect_internal(root):
            walls.append(_partition_wall(node, t))
            door = _partition_doorway(node, params, rng)
            if door is not None:
                doorways.append(door)

        for leaf in _collect_leaves(root):
            room_rect, corridor = _room_rect(leaf.cell, params, rng)
            if room_rect is None:
                # c_min enforcement: partition too small for a room under
                # the required minimum corridor. Skip placement; the
                # partition remains empty free space.
                continue
            leaf.room_rect = room_rect
            if corridor < min_corridor_observed:
                min_corridor_observed = corridor
            walls.append(_room_wall(leaf.room_rect, t))
            doorways.append(_room_doorway(leaf.room_rect, params, rng))

    occupied = unary_union(walls).difference(unary_union(doorways))
    domain = box(0.0, 0.0, X, Y)
    free_space = domain.difference(occupied)

    chains = _geometry_to_chains(occupied)
    if not chains:
        raise RuntimeError(
            "BSP generation produced an empty environment E. "
            "Check that Omega_env leaves room for at least the perimeter wall."
        )
    env = EnvironmentStructure(chains=chains, size=(X, Y))
    return BSPEnvironment(
        environment=env,
        free_space=free_space,
        min_corridor_width=min_corridor_observed,
    )

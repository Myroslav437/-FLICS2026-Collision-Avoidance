"""
PRM-based path generation.

Implements the Probabilistic Roadmap procedure:
    (1) inflate the boundary elements of E by a clearance margin d_clear
    (2) sample n_samples random points in the inflated free space
    (3) connect each sample to its k_nn nearest neighbors via collision-free
        straight-line edges (edges longer than l_max are rejected)
    (4) sample start and goal positions in the free space
    (5) insert start and goal into the roadmap and find the shortest path
        via Dijkstra's algorithm
    (6) assign target speeds to each segment from D_v*

The module is parameterised so the same routine is used for the AGV
reference path pi (Omega_path) and for dynamic-obstacle trajectories
xi_i (Omega_obs PRM parameters).
"""

from __future__ import annotations

from typing import Tuple

import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

from .models import WaypointPath


class PRMGenerationError(RuntimeError):
    """Raised when PRM cannot produce a valid path (e.g., no path exists)."""


# =============================================================================
# Free-space utilities
# =============================================================================

def inflate_free_space(free_space: BaseGeometry, clearance: float) -> BaseGeometry:
    """
    Shrink the walkable region by `clearance`, which is equivalent to
    inflating all boundary elements (walls) by the same amount.
    """
    if clearance < 0:
        raise ValueError("clearance must be non-negative")
    if clearance == 0:
        return free_space
    return free_space.buffer(-clearance)


def largest_component(geom: BaseGeometry) -> BaseGeometry:
    """
    Return the largest connected component of `geom` by area.

    Inflating a free-space MultiPolygon by d_clear can disconnect regions
    where passages are only marginally wider than 2*d_clear. The AGV can
    only reach one connected component from a given start, so restricting
    PRM sampling to the largest component is the canonical choice; any
    start/goal sampled here will be connected.
    """
    if geom.is_empty:
        return geom
    if hasattr(geom, "geoms"):
        best = max(geom.geoms, key=lambda g: g.area)
        return best
    return geom


def sample_point_in(
    region: BaseGeometry,
    rng: np.random.Generator,
    max_attempts: int = 1000,
) -> np.ndarray:
    """
    Uniformly sample a point inside a shapely region using rejection sampling
    over the region's bounding box.
    """
    if region.is_empty:
        raise PRMGenerationError("Cannot sample from an empty region")
    prepared = prep(region)
    minx, miny, maxx, maxy = region.bounds
    for _ in range(max_attempts):
        x = float(rng.uniform(minx, maxx))
        y = float(rng.uniform(miny, maxy))
        if prepared.contains(Point(x, y)):
            return np.array([x, y])
    raise PRMGenerationError(
        f"Failed to sample a point inside the region after {max_attempts} attempts"
    )


# =============================================================================
# Roadmap construction + shortest-path query
# =============================================================================

def _edge_is_collision_free(
    a: np.ndarray, b: np.ndarray, prepared_region
) -> bool:
    """True iff segment [a, b] lies entirely inside the prepared region."""
    segment = LineString([tuple(a), tuple(b)])
    return prepared_region.contains(segment)


def build_roadmap(
    inflated_free: BaseGeometry,
    n_samples: int,
    k_nn: int,
    l_max: float,
    rng: np.random.Generator,
) -> Tuple[nx.Graph, np.ndarray]:
    """
    Construct the PRM roadmap inside `inflated_free`.

    Returns
    -------
    graph : networkx.Graph with node ids in [0, n_samples) and edges carrying
            the Euclidean distance as edge weight.
    points: array of shape (n_samples, 2) giving each node's coordinates.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if k_nn <= 0:
        raise ValueError("k_nn must be positive")
    if inflated_free.is_empty:
        raise PRMGenerationError(
            "Inflated free space is empty; path planning is infeasible"
        )

    prepared = prep(inflated_free)
    # Sample n_samples points inside the inflated free space.
    points = np.empty((n_samples, 2), dtype=float)
    for i in range(n_samples):
        points[i] = sample_point_in(inflated_free, rng)

    graph = nx.Graph()
    for i in range(n_samples):
        graph.add_node(i, pos=tuple(points[i]))

    # k-nearest-neighbor connections with collision- and length-filtering.
    tree = KDTree(points)
    # query k_nn + 1 to account for self-match at distance 0
    k_query = min(k_nn + 1, n_samples)
    dists, idxs = tree.query(points, k=k_query)
    if k_query == 1:
        dists = dists.reshape(-1, 1)
        idxs = idxs.reshape(-1, 1)
    for i in range(n_samples):
        for d, j in zip(dists[i], idxs[i]):
            if j == i:
                continue
            if d > l_max:
                continue
            if graph.has_edge(i, int(j)):
                continue
            if _edge_is_collision_free(points[i], points[j], prepared):
                graph.add_edge(i, int(j), weight=float(d))
    return graph, points


def insert_terminal(
    graph: nx.Graph,
    points: np.ndarray,
    node_ids: np.ndarray,
    terminal: np.ndarray,
    k_nn: int,
    l_max: float,
    prepared_region,
    node_id: int,
) -> None:
    """
    Insert start or goal into the roadmap, connecting it to its k
    nearest roadmap nodes via collision-free edges.

    `points` and `node_ids` describe the candidate roadmap nodes for
    connection (coordinate + graph node id, aligned row-wise). Callers that
    want to restrict terminals to a subset of the roadmap (e.g., the
    largest connected component) pass only those nodes here.

    Raises PRMGenerationError if the terminal cannot be connected.
    """
    graph.add_node(node_id, pos=tuple(terminal))
    if len(points) == 0:
        raise PRMGenerationError("Roadmap is empty; cannot insert terminal")
    tree = KDTree(points)
    k_query = min(k_nn, len(points))
    dists, idxs = tree.query(terminal.reshape(1, 2), k=k_query)
    dists = np.atleast_1d(dists.reshape(-1))
    idxs = np.atleast_1d(idxs.reshape(-1))
    added = 0
    for d, j in zip(dists, idxs):
        if d > l_max:
            continue
        neighbor_id = int(node_ids[int(j)])
        if _edge_is_collision_free(terminal, points[int(j)], prepared_region):
            graph.add_edge(node_id, neighbor_id, weight=float(d))
            added += 1
    if added == 0:
        raise PRMGenerationError(
            f"Terminal point {terminal.tolist()} could not be connected to the "
            f"roadmap within l_max={l_max}"
        )


def largest_connected_component(
    graph: nx.Graph, points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (component_points, component_node_ids) for the largest connected
    component of `graph`. k-NN roadmaps on a non-convex free space routinely
    split into disjoint clusters; terminals must be inserted into the same
    cluster or Dijkstra will fail with NetworkXNoPath.
    """
    if graph.number_of_nodes() == 0:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=int)
    component = max(nx.connected_components(graph), key=len)
    ids = np.fromiter(sorted(component), dtype=int)
    return points[ids], ids


def sample_terminals_on_roadmap(
    cc_points: np.ndarray,
    rng: np.random.Generator,
    min_separation: float,
    max_attempts: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pick start and goal positions from the largest-connected-component nodes
    of the roadmap.

    Using existing roadmap nodes guarantees both points are reachable from
    each other (they already belong to the same connected subgraph), which
    removes the arbitrary-rejection failure mode of sampling independent
    points and hoping they land near the roadmap.

    `min_separation` enforces a minimum Euclidean distance between start and
    goal so the planner does not produce a trivial degenerate path.
    """
    n = cc_points.shape[0]
    if n < 2:
        raise PRMGenerationError(
            "Roadmap largest component has fewer than 2 nodes; "
            "cannot sample start/goal"
        )
    for _ in range(max_attempts):
        i, j = rng.choice(n, size=2, replace=False)
        a = cc_points[int(i)]
        b = cc_points[int(j)]
        if np.linalg.norm(b - a) >= min_separation:
            return a.copy(), b.copy()
    raise PRMGenerationError(
        f"Failed to sample start/goal with min_separation={min_separation} "
        f"after {max_attempts} attempts"
    )


# =============================================================================
# Top-level: produce a WaypointPath from (start, goal) via PRM + Dijkstra
# =============================================================================

def plan_waypoint_path(
    inflated_free: BaseGeometry,
    start: np.ndarray,
    goal: np.ndarray,
    n_samples: int,
    k_nn: int,
    l_max: float,
    speed_min: float,
    speed_max: float,
    rng: np.random.Generator,
) -> WaypointPath:
    """
    Plan a waypoint path with explicit start/goal.

    Build a PRM over `inflated_free`, connect start and goal to the
    roadmap's largest connected component, find the Dijkstra shortest
    path, and assign a speed per segment drawn from
    Uniform(speed_min, speed_max).

    Raises PRMGenerationError if start or goal cannot be connected or if
    no path is found.
    """
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if start.shape != (2,) or goal.shape != (2,):
        raise ValueError("start and goal must be 2D points")

    graph, points = build_roadmap(
        inflated_free, n_samples=n_samples, k_nn=k_nn, l_max=l_max, rng=rng,
    )
    return _finish_plan(
        graph=graph,
        points=points,
        inflated_free=inflated_free,
        start=start,
        goal=goal,
        n_samples=n_samples,
        k_nn=k_nn,
        l_max=l_max,
        speed_min=speed_min,
        speed_max=speed_max,
        rng=rng,
    )


def plan_random_path(
    inflated_free: BaseGeometry,
    n_samples: int,
    k_nn: int,
    l_max: float,
    speed_min: float,
    speed_max: float,
    min_separation: float,
    rng: np.random.Generator,
) -> WaypointPath:
    """
    End-to-end PRM planning with start and goal sampled from the
    roadmap's largest connected component. Used for dynamic-obstacle
    trajectories xi_i (Omega_obs); the reference path pi (Omega_path)
    goes through ``plan_reference_path`` to enforce l_pi_min.
    """
    graph, points = build_roadmap(
        inflated_free, n_samples=n_samples, k_nn=k_nn, l_max=l_max, rng=rng,
    )
    cc_points, _ = largest_connected_component(graph, points)
    start, goal = sample_terminals_on_roadmap(cc_points, rng, min_separation)
    return _finish_plan(
        graph=graph,
        points=points,
        inflated_free=inflated_free,
        start=start,
        goal=goal,
        n_samples=n_samples,
        k_nn=k_nn,
        l_max=l_max,
        speed_min=speed_min,
        speed_max=speed_max,
        rng=rng,
    )


# Implementation-internal cap on the l_pi_min rejection loop. Bounds
# the loop so that an infeasible (world seed, Omega) combination
# surfaces as an explicit PRMGenerationError rather than an infinite
# loop. If this limit trips under default Omega the l_pi_min value
# should be revisited, not silently relaxed.
_REFERENCE_PATH_MAX_ATTEMPTS = 100


def plan_reference_path(
    inflated_free: BaseGeometry,
    n_samples: int,
    k_nn: int,
    l_max: float,
    speed_min: float,
    speed_max: float,
    min_path_length: float,
    rng: np.random.Generator,
) -> Tuple[WaypointPath, int]:
    """
    PRM planning of the AGV reference path pi with the start-goal pair
    resampled until the Dijkstra shortest path between them has total
    length >= ``min_path_length`` (l_pi_min).

    Differs from ``plan_random_path`` in two ways:
      1. Terminals are drawn as two distinct nodes of the roadmap's
         largest connected component without an up-front Euclidean
         minimum-separation check; the binding filter is the
         post-Dijkstra path length, not the start-goal straight-line
         distance.
      2. On each attempt the pair is inserted, the shortest path is
         computed, and if its total length is below ``min_path_length``
         the terminals are removed and a new pair is drawn. The cap on
         the number of attempts is implementation-internal; exhausting
         it raises PRMGenerationError rather than silently falling back
         to a shorter path.

    Returns (path, n_attempts) where ``n_attempts`` is the number of
    start-goal pairs sampled before one satisfied l_pi_min (1 on the
    first-try-success case).
    """
    if min_path_length < 0:
        raise ValueError("min_path_length must be non-negative")

    graph, points = build_roadmap(
        inflated_free, n_samples=n_samples, k_nn=k_nn, l_max=l_max, rng=rng,
    )
    cc_points, cc_ids = largest_connected_component(graph, points)
    if cc_points.shape[0] < 2:
        raise PRMGenerationError(
            "Roadmap largest component has fewer than 2 nodes; "
            "cannot sample start/goal"
        )
    prepared = prep(inflated_free)
    start_id = n_samples
    goal_id = n_samples + 1

    for attempt in range(1, _REFERENCE_PATH_MAX_ATTEMPTS + 1):
        i, j = rng.choice(cc_points.shape[0], size=2, replace=False)
        start = cc_points[int(i)].copy()
        goal = cc_points[int(j)].copy()

        insert_terminal(
            graph, cc_points, cc_ids, start, k_nn, l_max, prepared, start_id
        )
        insert_terminal(
            graph, cc_points, cc_ids, goal, k_nn, l_max, prepared, goal_id
        )

        try:
            node_path = nx.shortest_path(
                graph, source=start_id, target=goal_id, weight="weight"
            )
        except nx.NetworkXNoPath:
            graph.remove_node(start_id)
            graph.remove_node(goal_id)
            continue

        positions = np.asarray(
            [graph.nodes[nid]["pos"] for nid in node_path], dtype=float
        )
        positions = _simplify_collinear(positions)
        if positions.shape[0] < 2:
            graph.remove_node(start_id)
            graph.remove_node(goal_id)
            continue
        deltas = np.diff(positions, axis=0)
        total_length = float(np.sum(np.linalg.norm(deltas, axis=1)))

        if total_length >= min_path_length:
            n_segments = positions.shape[0] - 1
            if speed_min == speed_max:
                speeds = np.full(n_segments, float(speed_min), dtype=float)
            else:
                speeds = rng.uniform(speed_min, speed_max, size=n_segments)
            return WaypointPath(positions=positions, speeds=speeds), attempt

        graph.remove_node(start_id)
        graph.remove_node(goal_id)

    raise PRMGenerationError(
        f"Reference-path generation: {_REFERENCE_PATH_MAX_ATTEMPTS} start-goal "
        f"resampling attempts exhausted without satisfying "
        f"min_path_length={min_path_length:.3f}. The constraint may be "
        f"infeasible for this roadmap; check the world seed and Omega_path "
        f"(notably min_path_length, clearance, and domain size)."
    )


def _finish_plan(
    graph: nx.Graph,
    points: np.ndarray,
    inflated_free: BaseGeometry,
    start: np.ndarray,
    goal: np.ndarray,
    n_samples: int,
    k_nn: int,
    l_max: float,
    speed_min: float,
    speed_max: float,
    rng: np.random.Generator,
) -> WaypointPath:
    """Insert terminals, run Dijkstra, simplify, and assign segment speeds."""
    cc_points, cc_ids = largest_connected_component(graph, points)
    if cc_points.shape[0] == 0:
        raise PRMGenerationError("Roadmap has no connected nodes")

    prepared = prep(inflated_free)
    start_id = n_samples
    goal_id = n_samples + 1
    insert_terminal(graph, cc_points, cc_ids, start, k_nn, l_max, prepared, start_id)
    insert_terminal(graph, cc_points, cc_ids, goal, k_nn, l_max, prepared, goal_id)

    try:
        node_path = nx.shortest_path(
            graph, source=start_id, target=goal_id, weight="weight"
        )
    except nx.NetworkXNoPath as exc:
        raise PRMGenerationError(
            "No path exists between start and goal in the roadmap"
        ) from exc

    positions = np.asarray(
        [graph.nodes[nid]["pos"] for nid in node_path], dtype=float
    )
    positions = _simplify_collinear(positions)

    n_segments = positions.shape[0] - 1
    if n_segments <= 0:
        raise PRMGenerationError(
            "Planned path has zero segments; start and goal may coincide"
        )
    if speed_min == speed_max:
        speeds = np.full(n_segments, float(speed_min), dtype=float)
    else:
        speeds = rng.uniform(speed_min, speed_max, size=n_segments)
    return WaypointPath(positions=positions, speeds=speeds)


def _simplify_collinear(
    points: np.ndarray, tolerance: float = 1e-8
) -> np.ndarray:
    """
    Remove intermediate waypoints that are collinear with their neighbors.

    PRM paths often include multiple nodes on a straight line; collapsing
    them reduces the number of segments without changing the geometry.
    """
    if points.shape[0] <= 2:
        return points
    keep = [0]
    for i in range(1, points.shape[0] - 1):
        a, b, c = points[keep[-1]], points[i], points[i + 1]
        # Cross product of (b - a) and (c - b); near zero => collinear.
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > tolerance:
            keep.append(i)
    keep.append(points.shape[0] - 1)
    return points[keep]

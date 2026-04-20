"""
Obstacle generation.

Given the environment free space, the AGV reference path pi, and
Omega_obs, this module produces the obstacle set O_{t=0} = {o_1, ..., o_m},
where each o_i = <S_i, xi_i> consists of:

  S_i   : spatial extent (circle or square of inscribed radius r_i)
  xi_i  : trajectory -- |xi_i| = 0 for static obstacles; PRM-generated for
          dynamic obstacles

Placement respects clearance constraints:
  - each obstacle boundary is at least d_min from every environment
    boundary element and from every other obstacle boundary;
  - each obstacle's initial position is at least path_clearance from
    the AGV reference path pi.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry

from .config import ObstacleParams
from .models import Obstacle, ObstacleShape, SpatialDescriptor, WaypointPath
from .prm import (
    PRMGenerationError,
    inflate_free_space,
    largest_component,
    plan_random_path,
    sample_point_in,
)


# =============================================================================
# Shape geometry in world frame
# =============================================================================

def obstacle_footprint(
    shape: ObstacleShape, radius: float, centroid: np.ndarray
) -> BaseGeometry:
    """
    Shapely geometry for a static obstacle's world-frame boundary.

    Each shape is parameterised by the inscribed-circle radius r_i -- for
    circles r_i is the radius; for squares r_i is the half-side length.
    Squares are placed axis-aligned; rotating them would require a
    trajectory heading which static obstacles lack.
    """
    cx, cy = float(centroid[0]), float(centroid[1])
    if shape == ObstacleShape.CIRCLE:
        return Point(cx, cy).buffer(radius)
    if shape == ObstacleShape.SQUARE:
        return box(cx - radius, cy - radius, cx + radius, cy + radius)
    raise ValueError(f"Unsupported shape: {shape}")


# =============================================================================
# Shape sampling
# =============================================================================

def _sample_shape(
    params: ObstacleParams, rng: np.random.Generator
) -> ObstacleShape:
    labels = list(params.shape_weights.keys())
    weights = np.asarray([params.shape_weights[k] for k in labels], dtype=float)
    probabilities = weights / weights.sum()
    choice = rng.choice(len(labels), p=probabilities)
    return ObstacleShape(labels[int(choice)])


def _sample_radius(
    params: ObstacleParams, rng: np.random.Generator
) -> float:
    return float(rng.uniform(params.radius_min, params.radius_max))


# =============================================================================
# Placement with clearance constraints
# =============================================================================

def _try_place_centroid(
    placement_region: BaseGeometry,
    footprint_fn,
    existing: List[BaseGeometry],
    path_geom: LineString,
    d_min: float,
    path_clearance: float,
    rng: np.random.Generator,
    max_attempts: int,
) -> Tuple[np.ndarray, BaseGeometry]:
    """
    Sample a centroid and build the obstacle footprint; reject samples that
    violate clearance to other obstacles, environment boundary, or pi.

    `placement_region` is the free-space region shrunk by the obstacle's
    bounding radius so the footprint is guaranteed to fit inside E's free
    space with d_min clearance; `footprint_fn(centroid) -> BaseGeometry`
    builds the world-frame footprint at the sampled centroid.

    Returns (centroid, footprint).
    Raises PRMGenerationError if no valid placement is found.
    """
    for _ in range(max_attempts):
        centroid = sample_point_in(placement_region, rng)
        footprint = footprint_fn(centroid)
        # Inter-obstacle clearance
        if any(footprint.distance(g) < d_min for g in existing):
            continue
        # Reference-path clearance
        if path_geom.distance(footprint) < path_clearance:
            continue
        return centroid, footprint
    raise PRMGenerationError(
        f"Failed to place obstacle after {max_attempts} attempts"
    )


def _static_obstacle(
    free_space: BaseGeometry,
    existing_footprints: List[BaseGeometry],
    path_geom: LineString,
    params: ObstacleParams,
    rng: np.random.Generator,
) -> Tuple[Obstacle, BaseGeometry]:
    """
    Produce a single static obstacle (xi_i = (w_0)).
    """
    shape = _sample_shape(params, rng)
    radius = _sample_radius(params, rng)
    spatial = SpatialDescriptor(shape=shape, inscribed_radius=radius)

    # Shrink the free space by (bounding_radius + d_min) so the sampled
    # centroid guarantees the obstacle footprint clears environment geometry
    # by at least d_min.
    margin = spatial.bounding_radius() + params.min_clearance
    placement = largest_component(free_space.buffer(-margin))
    if placement.is_empty:
        raise PRMGenerationError(
            f"No placement region for static obstacle of radius {radius:.3f}"
        )

    centroid, footprint = _try_place_centroid(
        placement_region=placement,
        footprint_fn=lambda c: obstacle_footprint(shape, radius, c),
        existing=existing_footprints,
        path_geom=path_geom,
        d_min=params.min_clearance,
        path_clearance=params.path_clearance,
        rng=rng,
        max_attempts=params.max_place_attempts,
    )
    trajectory = WaypointPath.stationary(centroid)
    return Obstacle(spatial=spatial, trajectory=trajectory), footprint


def _dynamic_obstacle(
    free_space: BaseGeometry,
    existing_footprints: List[BaseGeometry],
    path_geom: LineString,
    params: ObstacleParams,
    rng: np.random.Generator,
) -> Tuple[Obstacle, BaseGeometry]:
    """
    Produce a single dynamic obstacle with a PRM-generated trajectory.
    Initial-position constraints match the static case; the trajectory
    itself crosses free space without respect to pi (it would defeat the
    purpose of dynamic obstacles to keep them away from the AGV's planned
    path).
    """
    shape = _sample_shape(params, rng)
    radius = _sample_radius(params, rng)
    spatial = SpatialDescriptor(shape=shape, inscribed_radius=radius)

    # Trajectory generation: PRM roadmap over the obstacle-inflated free
    # space using Omega_obs.
    traj_region = largest_component(
        inflate_free_space(free_space, params.trajectory_clearance + radius)
    )
    if traj_region.is_empty:
        raise PRMGenerationError(
            "Obstacle trajectory region is empty; "
            "trajectory_clearance may exceed passage widths"
        )
    trajectory = plan_random_path(
        inflated_free=traj_region,
        n_samples=params.prm_samples,
        k_nn=params.prm_neighbors,
        l_max=params.max_edge_length,
        speed_min=params.dyn_speed_min,
        speed_max=params.dyn_speed_max,
        min_separation=max(2.0 * radius, params.max_edge_length),
        rng=rng,
    )

    # Initial position footprint must respect d_min and path_clearance.
    initial_centroid = trajectory.positions[0]
    footprint = obstacle_footprint(shape, radius, initial_centroid)
    if any(footprint.distance(g) < params.min_clearance for g in existing_footprints):
        raise PRMGenerationError(
            "Dynamic obstacle initial position violates inter-obstacle clearance"
        )
    if path_geom.distance(footprint) < params.path_clearance:
        raise PRMGenerationError(
            "Dynamic obstacle initial position violates path clearance"
        )
    return Obstacle(spatial=spatial, trajectory=trajectory), footprint


# =============================================================================
# Top-level: produce the obstacle set O_{t=0}
# =============================================================================

def generate_obstacles(
    free_space: BaseGeometry,
    reference_path: WaypointPath,
    params: ObstacleParams,
    rng: np.random.Generator,
) -> Tuple[List[Obstacle], int]:
    """
    Generate the initial obstacle set O_{t=0}.

    Returns
    -------
    obstacles : list of n_static + n_dynamic obstacles
    retries   : total number of placement retries performed (diagnostic;
                a high value indicates the parameter set is nearly
                infeasible for the environment).
    """
    path_geom = LineString([tuple(p) for p in reference_path.positions])
    existing: List[BaseGeometry] = []
    obstacles: List[Obstacle] = []
    retries = 0

    # Dynamic obstacles first: their trajectories use more of the free space
    # and are more likely to fail on tight environments, so we want to fail
    # fast rather than after placing all the static ones.
    for _ in range(params.n_dynamic):
        attempt = 0
        while True:
            try:
                obs, fp = _dynamic_obstacle(
                    free_space, existing, path_geom, params, rng
                )
                break
            except PRMGenerationError:
                attempt += 1
                retries += 1
                if attempt >= params.max_place_attempts:
                    raise
        obstacles.append(obs)
        existing.append(fp)

    for _ in range(params.n_static):
        obs, fp = _static_obstacle(
            free_space, existing, path_geom, params, rng
        )
        obstacles.append(obs)
        existing.append(fp)

    return obstacles, retries

"""
LiDAR simulation: ideal scan  S  and distortion  D.

Ideal scan
----------
For each ray we perform a ray-vs-segment intersection test against every
segment of E and of each obstacle boundary (in world-frame coordinates
at time t). The first hit within [r_min, r_max] is returned; otherwise
the reading is `inf` (no return).

Distortion
----------
Under Lambda_D:
- sigma_r: additive Gaussian noise on the range, `r_hat = r + N(0, sigma^2)`;
- p_miss: each valid ray drops to `inf` with probability p_miss;
- p_ghost: each ray becomes a spurious reading with probability p_ghost
  (a random range uniform in [r_min, r_max], regardless of whether the
  clean ray had a real return).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from world_generator.models import EnvironmentStructure, Obstacle

from ..config import LiDARDistortion, LiDARGeometry
from ..types import LiDARScan


# =============================================================================
# Segment collection from E and O_t  (helpers)
# =============================================================================

def _chain_segments(chain: np.ndarray) -> np.ndarray:
    """
    Turn a closed polygonal chain (shape (k, 2)) into an array of
    segments shape (k, 2, 2); segment i connects chain[i] -> chain[(i+1)%k].
    """
    starts = chain
    ends = np.roll(chain, -1, axis=0)
    return np.stack([starts, ends], axis=1)


def _environment_segments(env: EnvironmentStructure) -> np.ndarray:
    """Concatenate all environment segments into a single (N, 2, 2) array."""
    if not env.chains:
        return np.empty((0, 2, 2), dtype=float)
    return np.concatenate(
        [_chain_segments(c) for c in env.chains], axis=0
    )


def obstacle_world_polygon(obstacle: Obstacle, progress: float) -> np.ndarray:
    """
    Compute an obstacle's world-frame boundary polygon at trajectory
    progress `progress` in [0, |xi_i|]. Returns a (k, 2) array.

    The obstacle's heading is taken as the direction of the current
    trajectory segment (xi_i is a waypoint path; heading along segment
    j is atan2 of (w_j - w_{j-1})).
    """
    local = obstacle.spatial.boundary_vertices()
    pts = obstacle.trajectory.positions
    if obstacle.trajectory.is_stationary:
        centroid = pts[0]
        heading = 0.0
    else:
        deltas = np.diff(pts, axis=0)
        seg_len = np.linalg.norm(deltas, axis=1)
        total = float(np.sum(seg_len))
        if total <= 0:
            centroid = pts[0]
            heading = 0.0
        else:
            clamped = max(0.0, min(progress, total))
            cum = np.cumsum(seg_len)
            seg_idx = int(np.searchsorted(cum, clamped, side="right"))
            seg_idx = min(seg_idx, len(seg_len) - 1)
            start = pts[seg_idx]
            end = pts[seg_idx + 1]
            seg_travel = clamped - (cum[seg_idx] - seg_len[seg_idx])
            fraction = seg_travel / seg_len[seg_idx] if seg_len[seg_idx] > 0 else 0.0
            centroid = start + fraction * (end - start)
            heading = float(np.arctan2(end[1] - start[1], end[0] - start[0]))
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + centroid


# =============================================================================
# Ray-vs-segment intersection (vectorised)
# =============================================================================

def _first_ray_hit(
    origin: np.ndarray,
    direction_unit: np.ndarray,
    segments: np.ndarray,
    r_min: float,
    r_max: float,
) -> float:
    """
    Return the smallest t in [r_min, r_max] for which
    origin + t * direction_unit lies on any given segment, else inf.

    `segments` has shape (N, 2, 2); each row is [[x1, y1], [x2, y2]].
    """
    if segments.shape[0] == 0:
        return float("inf")

    a = segments[:, 0, :]
    b = segments[:, 1, :]
    v = b - a
    w = origin - a
    d = direction_unit

    denom = d[0] * v[:, 1] - d[1] * v[:, 0]
    parallel = np.abs(denom) < 1e-12
    denom_safe = np.where(parallel, 1.0, denom)

    t = (v[:, 1] * (-w[:, 0]) - v[:, 0] * (-w[:, 1])) / denom_safe
    u = (d[0] * (-w[:, 1]) - d[1] * (-w[:, 0])) / denom_safe

    valid = (~parallel) & (t >= r_min) & (t <= r_max) & (u >= 0.0) & (u <= 1.0)
    if not np.any(valid):
        return float("inf")
    return float(np.min(t[valid]))


# =============================================================================
# Ideal scan  S
# =============================================================================

def ideal_scan(
    environment: EnvironmentStructure,
    obstacles_world_polygons: Iterable[np.ndarray],
    position: np.ndarray,
    heading: float,
    geometry: LiDARGeometry,
    sim_time: float,
) -> LiDARScan:
    """
    L_circ_t = S(E, O_t, P_t, theta_t, Lambda_S).

    Parameters
    ----------
    environment              : E
    obstacles_world_polygons : iterable of (k_i, 2) polygons, each obstacle's
                               boundary in world frame at time t (already
                               positioned and rotated via obstacle_world_polygon)
    position, heading        : (P_t, theta_t)
    geometry                 : Lambda_S
    sim_time                 : t (seconds); stored on the scan for telemetry
    """
    env_segs = _environment_segments(environment)
    poly_segs = [_chain_segments(p) for p in obstacles_world_polygons]
    if poly_segs:
        obs_segs = np.concatenate(poly_segs, axis=0)
    else:
        obs_segs = np.empty((0, 2, 2), dtype=float)
    all_segs = np.concatenate([env_segs, obs_segs], axis=0)

    n_rays = geometry.n_rays
    half = geometry.theta_fov / 2.0
    if n_rays > 1:
        angles = np.linspace(-half, +half, n_rays)
    else:
        angles = np.array([0.0])

    ranges = np.empty(n_rays, dtype=float)
    origin = np.asarray(position, dtype=float)
    for i, alpha in enumerate(angles):
        world_angle = heading + alpha
        direction = np.array([np.cos(world_angle), np.sin(world_angle)])
        ranges[i] = _first_ray_hit(
            origin, direction, all_segs, geometry.r_min, geometry.r_max
        )
    return LiDARScan(
        ranges=ranges,
        angles=angles,
        time=float(sim_time),
        pose_position=origin.copy(),
        pose_heading=float(heading),
    )


# =============================================================================
# Distortion  D
# =============================================================================

def distort_scan(
    scan: LiDARScan,
    distortion: LiDARDistortion,
    geometry: LiDARGeometry,
    rng: np.random.Generator,
) -> LiDARScan:
    """
    L_hat_t = D(L_circ_t, Lambda_D).

    Order of operations:
    1. Add Gaussian range noise (std dev sigma_r) to finite readings.
    2. Apply p_miss: each ray independently becomes inf with p = p_miss.
    3. Apply p_ghost: each ray independently becomes a uniform
       [r_min, r_max] reading with p = p_ghost (overrides its prior
       value, clean or missed).
    """
    ranges = scan.ranges.copy()
    n = ranges.shape[0]

    if distortion.sigma_r > 0:
        mask = np.isfinite(ranges)
        if np.any(mask):
            noise = rng.normal(
                loc=0.0, scale=distortion.sigma_r, size=int(mask.sum())
            )
            ranges[mask] = ranges[mask] + noise

    if distortion.p_miss > 0:
        drop = rng.random(n) < distortion.p_miss
        ranges[drop] = np.inf

    if distortion.p_ghost > 0:
        ghost = rng.random(n) < distortion.p_ghost
        if np.any(ghost):
            ghost_ranges = rng.uniform(
                low=geometry.r_min,
                high=geometry.r_max,
                size=int(ghost.sum()),
            )
            ranges[ghost] = ghost_ranges

    return LiDARScan(
        ranges=ranges,
        angles=scan.angles.copy(),
        time=scan.time,
        pose_position=scan.pose_position.copy(),
        pose_heading=scan.pose_heading,
    )

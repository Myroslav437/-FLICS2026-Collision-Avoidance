"""
Shared helper: filter a world-frame point cloud against the prior map M_0.

The detection stage receives the distorted scan L_hat_t and the prior
map M_0, filters out points matching known boundary geometry, and
groups remaining points into obstacle candidates.

The filter here implements the "filter points matching known boundary
geometry" step: any point whose distance to the nearest M_0 segment is
below a per-stage tolerance is dropped. What remains feeds the
clustering step owned by the concrete algorithm (EC or DB).
"""

from __future__ import annotations

import numpy as np

from ...types import PriorMap


def _chain_segments(chain: np.ndarray) -> np.ndarray:
    starts = chain
    ends = np.roll(chain, -1, axis=0)
    return np.stack([starts, ends], axis=1)


def _prior_map_segments(prior_map: PriorMap) -> np.ndarray:
    if not prior_map.chains:
        return np.empty((0, 2, 2), dtype=float)
    return np.concatenate(
        [_chain_segments(c) for c in prior_map.chains], axis=0
    )


def _point_to_segment_distance(
    points: np.ndarray, segments: np.ndarray
) -> np.ndarray:
    """
    For each point p in `points` (M, 2) compute the minimum distance
    to any segment in `segments` (N, 2, 2). Returns shape (M,).
    """
    if segments.shape[0] == 0:
        return np.full(points.shape[0], np.inf)
    a = segments[:, 0, :]          # (N, 2)
    b = segments[:, 1, :]          # (N, 2)
    ab = b - a                      # (N, 2)
    ab_len_sq = np.sum(ab * ab, axis=1)  # (N,)
    ab_len_sq = np.where(ab_len_sq > 1e-12, ab_len_sq, 1e-12)

    # For each point, compute projection onto every segment
    # ap = (M, N, 2)
    ap = points[:, None, :] - a[None, :, :]
    t = np.sum(ap * ab[None, :, :], axis=2) / ab_len_sq[None, :]
    t = np.clip(t, 0.0, 1.0)
    proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]  # (M, N, 2)
    diff = points[:, None, :] - proj
    dists = np.linalg.norm(diff, axis=2)  # (M, N)
    return np.min(dists, axis=1)


def filter_points_against_prior_map(
    points_world: np.ndarray,
    prior_map: PriorMap,
    tolerance: float,
) -> np.ndarray:
    """
    Return the subset of `points_world` farther than `tolerance` from
    every segment of M_0. Keeps points that do NOT match known
    boundary geometry -- i.e. the obstacle candidates.

    Parameters
    ----------
    points_world : (M, 2) array of scan returns in world frame
    prior_map    : M_0
    tolerance    : metres; points nearer than this to any M_0 segment
                   are considered "boundary points" and dropped.
    """
    if points_world.shape[0] == 0:
        return points_world
    segs = _prior_map_segments(prior_map)
    if segs.shape[0] == 0:
        return points_world
    dists = _point_to_segment_distance(points_world, segs)
    keep = dists > tolerance
    return points_world[keep]

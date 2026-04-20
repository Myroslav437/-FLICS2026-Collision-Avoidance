"""
Euclidean Clustering detection stage (Rusu, PCL).

Groups scan points into clusters using a fixed distance threshold d_ec:
two points belong to the same cluster iff their Euclidean distance is
<= d_ec. Equivalent to agglomerative single-linkage clustering with a
distance cutoff; here implemented via a KD-tree breadth-first
region-growing pass so the computational profile matches PCL's
pcl::EuclideanClusterExtraction.

Parameters
----------
- d_ec        : fixed distance threshold (m). We pick 0.25 m, consistent
                with the ~0.1-0.4 m obstacle radii (obstacle-sized
                cluster scale).
- min_cluster : minimum cluster size; smaller groups are dropped as
                noise. We use 3, consistent with the PCL defaults.
- map_tol     : prior-map filtering tolerance (m). Points within this
                distance of any M_0 segment are treated as known boundary
                and dropped before clustering. 0.2 m is generous enough
                to absorb sigma_map = 0.1 m (Degraded-2) plus
                sigma_r = 0.05 m without producing ghost obstacles.
"""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.spatial import cKDTree

from ...types import AGVState, DetectedObstacle, LiDARScan, PriorMap
from ..base import DetectionStage
from ..registry import register_detection

from ._prior_map_filter import filter_points_against_prior_map


class EuclideanClusteringDetection(DetectionStage):
    """Euclidean clustering detection (Rusu, PCL)."""

    name = "EC"

    def __init__(
        self,
        d_ec: float = 0.25,
        min_cluster: int = 3,
        map_tol: float = 0.2,
    ) -> None:
        if d_ec <= 0:
            raise ValueError(f"d_ec must be positive; got {d_ec}")
        if min_cluster < 1:
            raise ValueError(f"min_cluster must be >= 1; got {min_cluster}")
        if map_tol < 0:
            raise ValueError(f"map_tol must be >= 0; got {map_tol}")
        self.d_ec = float(d_ec)
        self.min_cluster = int(min_cluster)
        self.map_tol = float(map_tol)

    def reset(self, seed: int) -> None:
        # Stateless; no per-run state to clear.
        return None

    def step(
        self,
        scan: LiDARScan,
        prior_map: PriorMap,
        agv_state: AGVState,
    ) -> List[DetectedObstacle]:
        # Project scan to world frame and filter known boundary geometry.
        points_world = scan.cartesian_world()
        residual = filter_points_against_prior_map(
            points_world, prior_map, tolerance=self.map_tol
        )
        if residual.shape[0] == 0:
            return []

        return _cluster_euclidean(
            residual, d_ec=self.d_ec, min_cluster=self.min_cluster
        )


def _cluster_euclidean(
    points: np.ndarray, d_ec: float, min_cluster: int
) -> List[DetectedObstacle]:
    """Single-linkage clustering via KD-tree radius queries."""
    n = points.shape[0]
    tree = cKDTree(points)
    visited = np.zeros(n, dtype=bool)
    clusters: List[np.ndarray] = []

    for i in range(n):
        if visited[i]:
            continue
        # Region-growing breadth-first flood
        queue = [i]
        visited[i] = True
        members = []
        while queue:
            j = queue.pop()
            members.append(j)
            neighbours = tree.query_ball_point(points[j], r=d_ec)
            for k in neighbours:
                if not visited[k]:
                    visited[k] = True
                    queue.append(k)
        if len(members) >= min_cluster:
            clusters.append(points[members])

    detected: List[DetectedObstacle] = []
    for cluster_points in clusters:
        centroid = cluster_points.mean(axis=0)
        radius = float(np.max(np.linalg.norm(cluster_points - centroid, axis=1)))
        detected.append(
            DetectedObstacle(
                centroid=centroid,
                radius=radius,
                points=cluster_points,
            )
        )
    return detected


register_detection("EC", EuclideanClusteringDetection)

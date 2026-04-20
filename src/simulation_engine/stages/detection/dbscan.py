"""
DBSCAN detection stage (Ester, Kriegel, Sander, Xu).

Density-based clustering: a point is a "core point" if at least
`min_samples` points lie within `eps` of it; clusters are formed by
reachability through core points. Robust to noise and varying obstacle
densities relative to EC.

Implemented via scikit-learn's `sklearn.cluster.DBSCAN`.

Parameters
----------
- eps         : neighbourhood radius epsilon (m). We use 0.3 m so a
                typical obstacle (inscribed radius Uniform(0.1, 0.4) m)
                lands in one cluster.
- min_samples : minimum cluster size; we use 3.
- map_tol     : prior-map filtering tolerance, as for EC.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.cluster import DBSCAN

from ...types import AGVState, DetectedObstacle, LiDARScan, PriorMap
from ..base import DetectionStage
from ..registry import register_detection

from ._prior_map_filter import filter_points_against_prior_map


class DBSCANDetection(DetectionStage):
    """DBSCAN detection (Ester, Kriegel, Sander, Xu)."""

    name = "DB"

    def __init__(
        self,
        eps: float = 0.3,
        min_samples: int = 3,
        map_tol: float = 0.2,
    ) -> None:
        if eps <= 0:
            raise ValueError(f"eps must be positive; got {eps}")
        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1; got {min_samples}")
        if map_tol < 0:
            raise ValueError(f"map_tol must be >= 0; got {map_tol}")
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.map_tol = float(map_tol)

    def reset(self, seed: int) -> None:
        return None

    def step(
        self,
        scan: LiDARScan,
        prior_map: PriorMap,
        agv_state: AGVState,
    ) -> List[DetectedObstacle]:
        points_world = scan.cartesian_world()
        residual = filter_points_against_prior_map(
            points_world, prior_map, tolerance=self.map_tol
        )
        if residual.shape[0] == 0:
            return []

        clustering = DBSCAN(
            eps=self.eps, min_samples=self.min_samples
        ).fit(residual)
        labels = clustering.labels_

        detected: List[DetectedObstacle] = []
        for label in sorted(set(labels)):
            if label == -1:
                # Noise points ignored per the algorithm.
                continue
            mask = labels == label
            cluster_points = residual[mask]
            centroid = cluster_points.mean(axis=0)
            radius = float(
                np.max(np.linalg.norm(cluster_points - centroid, axis=1))
            )
            detected.append(
                DetectedObstacle(
                    centroid=centroid,
                    radius=radius,
                    points=cluster_points,
                )
            )
        return detected


register_detection("DB", DBSCANDetection)

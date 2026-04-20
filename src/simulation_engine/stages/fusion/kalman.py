"""
Kalman-filter fusion stage (Welch, Bishop).

Constant-velocity model per track; nearest-neighbour association between
the current detections and the predicted track positions.

State per track (4-vector): [x, y, vx, vy].
Transition F at timestep dt:
    [[1, 0, dt, 0],
     [0, 1, 0, dt],
     [0, 0, 1, 0],
     [0, 0, 0, 1]]
Process noise covariance Q_base is scaled by dt (discrete constant-
acceleration approximation).
Observation H = [[1, 0, 0, 0], [0, 1, 0, 0]] (position-only).
Observation noise R = diag(sigma_r^2, sigma_r^2); sigma_obs is a
stage-level hyperparameter (defaults to 0.05 m, order of the range
noise of Degraded-1 / -2).

The filter is built on `filterpy.kalman.KalmanFilter`; no wheel
reinvention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from ...types import DetectedObstacle, FusedObstacle
from ..base import FusionStage
from ..registry import register_fusion


@dataclass
class _Track:
    track_id: int
    kf: KalmanFilter
    radius: float
    missed: int = 0  # consecutive steps without a matched detection


class KalmanFilterFusion(FusionStage):
    """Kalman-filter fusion (Welch, Bishop)."""

    name = "KF"

    def __init__(
        self,
        sigma_process: float = 0.5,
        sigma_observation: float = 0.05,
        association_gate: float = 1.5,
        max_missed: int = 3,
    ) -> None:
        """
        Parameters
        ----------
        sigma_process : process-noise std dev on velocity (m/s per sqrt(s)).
                        Higher -> filter reacts faster to manoeuvres at the
                        cost of a noisier velocity estimate.
        sigma_observation : measurement noise std dev on position (m).
        association_gate : gating distance (m) for nearest-neighbour
                           association; detections farther than this from
                           every predicted track are treated as new tracks.
        max_missed : track is dropped after this many consecutive unmatched
                     steps.
        """
        if sigma_process <= 0:
            raise ValueError("sigma_process must be positive")
        if sigma_observation <= 0:
            raise ValueError("sigma_observation must be positive")
        if association_gate <= 0:
            raise ValueError("association_gate must be positive")
        if max_missed < 1:
            raise ValueError("max_missed must be >= 1")
        self.sigma_process = float(sigma_process)
        self.sigma_observation = float(sigma_observation)
        self.association_gate = float(association_gate)
        self.max_missed = int(max_missed)

        self._tracks: List[_Track] = []
        self._next_id: int = 0

    def reset(self, seed: int) -> None:
        self._tracks = []
        self._next_id = 0

    # -- helpers -------------------------------------------------------------

    def _make_kf(self, dt: float, x: float, y: float) -> KalmanFilter:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        kf.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        q = self.sigma_process ** 2
        # Discrete constant-acceleration Q for (x, y, vx, vy):
        kf.Q = q * np.array([
            [dt**3 / 3, 0.0, dt**2 / 2, 0.0],
            [0.0, dt**3 / 3, 0.0, dt**2 / 2],
            [dt**2 / 2, 0.0, dt, 0.0],
            [0.0, dt**2 / 2, 0.0, dt],
        ])
        r = self.sigma_observation ** 2
        kf.R = np.diag([r, r])
        kf.P *= 10.0
        kf.x = np.array([x, y, 0.0, 0.0])
        return kf

    # -- main step -----------------------------------------------------------

    def step(
        self,
        detections: List[DetectedObstacle],
        dt: float,
    ) -> List[FusedObstacle]:
        # Update each track's F for current dt and predict.
        for trk in self._tracks:
            trk.kf.F[0, 2] = dt
            trk.kf.F[1, 3] = dt
            trk.kf.predict()

        # Nearest-neighbour assignment via Hungarian on position distance.
        assigned_det: Dict[int, int] = {}
        if self._tracks and detections:
            cost = np.zeros((len(self._tracks), len(detections)), dtype=float)
            for ti, trk in enumerate(self._tracks):
                pred = trk.kf.x[:2]
                for di, det in enumerate(detections):
                    cost[ti, di] = float(np.linalg.norm(pred - det.centroid))
            # Hungarian handles the rectangular case naturally.
            row_ind, col_ind = linear_sum_assignment(cost)
            for ti, di in zip(row_ind, col_ind):
                if cost[ti, di] <= self.association_gate:
                    assigned_det[ti] = di

        matched_det_indices = set(assigned_det.values())

        # Update matched tracks, age unmatched ones.
        surviving: List[_Track] = []
        for ti, trk in enumerate(self._tracks):
            if ti in assigned_det:
                det = detections[assigned_det[ti]]
                trk.kf.update(det.centroid)
                trk.radius = 0.5 * (trk.radius + det.radius)
                trk.missed = 0
                surviving.append(trk)
            else:
                trk.missed += 1
                if trk.missed <= self.max_missed:
                    surviving.append(trk)

        # Create tracks for unmatched detections.
        for di, det in enumerate(detections):
            if di in matched_det_indices:
                continue
            kf = self._make_kf(dt=dt, x=det.centroid[0], y=det.centroid[1])
            surviving.append(
                _Track(track_id=self._next_id, kf=kf, radius=det.radius)
            )
            self._next_id += 1

        self._tracks = surviving

        # Emit fused outputs for all currently-live tracks.
        out: List[FusedObstacle] = []
        for trk in self._tracks:
            out.append(
                FusedObstacle(
                    track_id=trk.track_id,
                    position=trk.kf.x[:2].copy(),
                    velocity=trk.kf.x[2:4].copy(),
                    radius=trk.radius,
                )
            )
        return out


register_fusion("KF", KalmanFilterFusion)

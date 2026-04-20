"""
Per-run metrics aggregator  M = <mu_col, mu_dev, mu_goal, mu_vel, mu_comp, mu_lat>.

All six metrics are accumulated online from per-step samples so a single
pass over the simulation loop is sufficient. The engine drives the
aggregator; nothing outside the core references it.

Metric definitions
------------------
mu_col  : 1 iff the AGV footprint intersected E or any obstacle at any
          step, else 0.
mu_dev  : mean perpendicular distance from the AGV position to the
          nearest segment of pi, averaged over all steps.
mu_goal : 1 iff the AGV reached w_n within T_horizon, else 0.
mu_vel  : RMSE between fused and ground-truth velocities across the
          dynamic obstacles and all steps (m/s). NaN if the run has no
          dynamic obstacles. Each fused obstacle is matched to its
          nearest dynamic obstacle at each step (Euclidean).
mu_comp : mean per-step CPU wall time (s).
mu_lat  : 95th-percentile per-step CPU wall time (s).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from world_generator.models import WaypointPath

from ..types import FusedObstacle, RunMetrics


class MetricsAccumulator:
    """
    Streaming accumulator for the six per-run metrics.

    Call order
    ----------
        acc = MetricsAccumulator(reference_path, dt)
        for each simulation step:
            acc.record_deviation(agv_position)
            acc.record_velocity_samples(fused, dyn_centroids, dyn_velocities)
            acc.record_step_time(step_wall_time)
            if collided: acc.record_collision(step)
            if goal_reached: acc.record_goal(step)
        metrics = acc.finalise(steps_completed, duration)
    """

    def __init__(self, reference_path: WaypointPath) -> None:
        self._ref_segments = _segments_of(reference_path)
        self._dev_sum: float = 0.0
        self._dev_count: int = 0
        self._vel_sq_err_sum: float = 0.0
        self._vel_sample_count: int = 0
        self._step_times: List[float] = []
        self._collided: bool = False
        self._collision_step: Optional[int] = None
        self._goal_reached: bool = False
        self._goal_step: Optional[int] = None

    # ---- per-step samples ---------------------------------------------------

    def record_deviation(self, position: np.ndarray) -> None:
        if self._ref_segments.shape[0] == 0:
            return
        d = _min_distance_to_segments(position, self._ref_segments)
        self._dev_sum += float(d)
        self._dev_count += 1

    def record_velocity_samples(
        self,
        fused: List[FusedObstacle],
        dyn_centroids: np.ndarray,
        dyn_velocities: np.ndarray,
    ) -> None:
        """
        Accumulate squared velocity error across nearest-neighbour matched
        (fused, ground-truth-dynamic) pairs at the current step.

        Static obstacles are excluded by construction: the engine supplies
        only the dynamic obstacles' centroids and velocities.
        """
        if dyn_centroids.shape[0] == 0 or not fused:
            return
        for obs in fused:
            diff = dyn_centroids - obs.position
            d = np.linalg.norm(diff, axis=1)
            j = int(np.argmin(d))
            err = obs.velocity - dyn_velocities[j]
            self._vel_sq_err_sum += float(err[0] ** 2 + err[1] ** 2)
            self._vel_sample_count += 1

    def record_step_time(self, wall_time: float) -> None:
        self._step_times.append(float(wall_time))

    def record_collision(self, step: int) -> None:
        if not self._collided:
            self._collided = True
            self._collision_step = int(step)

    def record_goal(self, step: int) -> None:
        if not self._goal_reached:
            self._goal_reached = True
            self._goal_step = int(step)

    # ---- finalisation -------------------------------------------------------

    def finalise(self, steps: int, duration: float) -> RunMetrics:
        mu_dev = (self._dev_sum / self._dev_count) if self._dev_count > 0 else 0.0
        if self._vel_sample_count > 0:
            mu_vel = float(np.sqrt(self._vel_sq_err_sum / self._vel_sample_count))
        else:
            mu_vel = float("nan")
        if self._step_times:
            arr = np.asarray(self._step_times, dtype=float)
            mu_comp = float(arr.mean())
            mu_lat = float(np.percentile(arr, 95.0))
        else:
            mu_comp = 0.0
            mu_lat = 0.0
        return RunMetrics(
            mu_col=int(self._collided),
            mu_dev=float(mu_dev),
            mu_goal=int(self._goal_reached),
            mu_vel=mu_vel,
            mu_comp=mu_comp,
            mu_lat=mu_lat,
            steps=int(steps),
            duration=float(duration),
            collision_step=self._collision_step,
            goal_reach_step=self._goal_step,
        )


# =============================================================================
# Helpers
# =============================================================================

def _segments_of(path: WaypointPath) -> np.ndarray:
    """Shape-(N, 2, 2) segment array for a waypoint path; empty if stationary."""
    if path.is_stationary:
        return np.empty((0, 2, 2), dtype=float)
    pts = path.positions
    return np.stack([pts[:-1], pts[1:]], axis=1)


def _min_distance_to_segments(point: np.ndarray, segments: np.ndarray) -> float:
    """
    Minimum Euclidean distance from `point` to any segment in `segments`
    (shape (N, 2, 2)).
    """
    a = segments[:, 0, :]
    b = segments[:, 1, :]
    v = b - a
    p = point - a
    seg_len_sq = np.einsum("ij,ij->i", v, v)
    seg_len_sq = np.where(seg_len_sq < 1e-12, 1e-12, seg_len_sq)
    t = np.clip(np.einsum("ij,ij->i", p, v) / seg_len_sq, 0.0, 1.0)
    closest = a + t[:, None] * v
    d = np.linalg.norm(closest - point, axis=1)
    return float(np.min(d))


__all__ = ["MetricsAccumulator"]

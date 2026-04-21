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
          dynamic obstacles. Association is performed from each dynamic
          obstacle to its nearest fused track within a spatial gate (see
          `record_velocity_samples`).
mu_comp : mean per-step CPU wall time (s).
mu_lat  : 95th-percentile per-step pipeline decision time (s), where
          decision time is the sum of the three pipeline stage calls
          (detection + fusion + avoidance) only — excluding perception,
          dynamics, collision checks, and bookkeeping. See
          `record_step_time` below.
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
        self._decision_times: List[float] = []
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
        Accumulate squared velocity error per dynamic obstacle for the
        current step, per §IV-B mu_vel (RMSE averaged over dynamic
        obstacles and simulation steps).

        Association direction: for each dynamic obstacle, find its
        nearest fused track inside a spatial gate. Static obstacles and
        spurious fused tracks never contribute (iteration is driven by
        the dynamic-obstacle ground truth, not by the fused track list).

        Implementation-internal choices (the paper does not specify):

        * Gate size. Set to ``_GATE_FACTOR * max_fused_radius`` with a
          floor of ``_GATE_FLOOR`` metres, derived from the fused
          tracks' own radius estimate rather than introducing a new
          paper parameter. Any dynamic obstacle whose nearest fused
          track lies outside the gate is treated as "unmatched".
        * Unmatched handling. An unmatched dynamic obstacle is charged
          a squared error equal to the ground-truth velocity magnitude
          squared --- equivalent to assuming the fusion stage reported
          a zero-velocity (stationary) estimate. This makes the
          denominator exactly ``n_dynamic * n_steps`` and prevents a
          pipeline from lowering its mu_vel simply by failing to
          produce tracks.

        Both choices are flagged as "assumptions made" in the fix
        report.
        """
        if dyn_centroids.shape[0] == 0:
            return

        _GATE_FACTOR = 3.0
        _GATE_FLOOR = 1.0
        if fused:
            max_radius = max(float(f.radius) for f in fused)
            gate = max(_GATE_FACTOR * max_radius, _GATE_FLOOR)
            fused_positions = np.asarray(
                [f.position for f in fused], dtype=float
            )
            fused_velocities = np.asarray(
                [f.velocity for f in fused], dtype=float
            )
        else:
            gate = _GATE_FLOOR
            fused_positions = np.empty((0, 2), dtype=float)
            fused_velocities = np.empty((0, 2), dtype=float)

        for i in range(dyn_centroids.shape[0]):
            v_gt = dyn_velocities[i]
            if fused_positions.shape[0] == 0:
                sq_err = float(v_gt[0] ** 2 + v_gt[1] ** 2)
            else:
                diff = fused_positions - dyn_centroids[i]
                d = np.linalg.norm(diff, axis=1)
                j = int(np.argmin(d))
                if d[j] > gate:
                    sq_err = float(v_gt[0] ** 2 + v_gt[1] ** 2)
                else:
                    err = fused_velocities[j] - v_gt
                    sq_err = float(err[0] ** 2 + err[1] ** 2)
            self._vel_sq_err_sum += sq_err
            self._vel_sample_count += 1

    def record_step_time(
        self, wall_time: float, decision_time: float
    ) -> None:
        """
        Record both the full-step wall time (for mu_comp) and the
        pipeline-only decision time (for mu_lat). Per §IV-B, mu_lat is
        the 95th-percentile "decision time" — the sum of the three
        pipeline stage calls only — while mu_comp aggregates the full
        per-step CPU cost.
        """
        self._step_times.append(float(wall_time))
        self._decision_times.append(float(decision_time))

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
            mu_comp = float(np.asarray(self._step_times, dtype=float).mean())
        else:
            mu_comp = 0.0
        if self._decision_times:
            mu_lat = float(
                np.percentile(
                    np.asarray(self._decision_times, dtype=float), 95.0
                )
            )
        else:
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

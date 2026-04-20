"""
Vector Field Histogram (VFH) avoidance stage (Borenstein, Koren).

VFH constructs a polar histogram of obstacle density and selects the
best traversable direction. This module follows the original
Borenstein-Koren formulation, adapted to the fused-obstacle
representation the pipeline produces (the original VFH works on a
certainty grid; we work on FusedObstacle centroids, which is the common
practical adaptation used in ROS Nav2 and similar stacks).

Algorithm
---------
1. Build a polar histogram h[k] over N_sectors. Each obstacle
   contributes a weight c^2 * (a - b * d) to every sector its angular
   footprint covers, following the original quadratic weight
   (c = obstacle confidence == 1 here; a, b = linear parameters).
2. Binarize: sector k is "blocked" if h[k] > threshold.
3. Group consecutive unblocked sectors into candidate valleys.
4. Pick the valley whose direction is closest to the direction to the
   look-ahead waypoint.
5. Emit a control: linear velocity scaled by the valley width and the
   minimum obstacle distance in the chosen direction; angular velocity
   proportional to the heading error.

Hyperparameters
---------------
  n_sectors       : 72 (5 deg / sector)
  safety_radius   : 0.5 m (AGV radius + clearance)
  h_threshold     : 2.0 (dimensionless; above => blocked)
  a_const, b_const: a = 10, b = 1 (range of a - b*d over r_max = 10
                    stays positive)
  max_range       : sensor horizon for histogram accumulation (= 5 m)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from world_generator.models import WaypointPath

from ...config import SimulationConfig
from ...types import AGVState, ControlAction, FusedObstacle
from ..base import AvoidanceStage
from ..registry import register_avoidance

from ._goal_tracker import WaypointGoalTracker


class VFHAvoidance(AvoidanceStage):
    """VFH avoidance (Borenstein, Koren)."""

    name = "VFH"

    def __init__(
        self,
        n_sectors: int = 72,
        safety_radius: float = 0.5,
        h_threshold: float = 2.0,
        a_const: float = 10.0,
        b_const: float = 1.0,
        max_range: float = 5.0,
        goal_tolerance: float = 0.3,
    ) -> None:
        if n_sectors < 8:
            raise ValueError("n_sectors must be >= 8")
        if safety_radius <= 0:
            raise ValueError("safety_radius must be positive")
        if h_threshold <= 0:
            raise ValueError("h_threshold must be positive")
        if max_range <= 0:
            raise ValueError("max_range must be positive")
        if a_const <= 0 or b_const <= 0:
            raise ValueError("a_const and b_const must be positive")
        self.n_sectors = int(n_sectors)
        self.safety_radius = float(safety_radius)
        self.h_threshold = float(h_threshold)
        self.a_const = float(a_const)
        self.b_const = float(b_const)
        self.max_range = float(max_range)
        self.goal_tolerance = float(goal_tolerance)

        self._tracker: Optional[WaypointGoalTracker] = None
        self._sigma: Optional[SimulationConfig] = None

    def reset(self, sigma: SimulationConfig, seed: int) -> None:
        self._sigma = sigma
        self._tracker = None  # created on first step when we have the path

    def step(
        self,
        fused_obstacles: List[FusedObstacle],
        reference_path: WaypointPath,
        agv_state: AGVState,
        dt: float,
    ) -> ControlAction:
        assert self._sigma is not None, "VFHAvoidance.reset() not called"
        if self._tracker is None:
            self._tracker = WaypointGoalTracker(reference_path)

        target = self._tracker.current_target(agv_state.position)
        goal = self._tracker.goal

        # If essentially at the final goal, command a stop.
        if np.linalg.norm(agv_state.position - goal) <= self.goal_tolerance:
            return ControlAction(linear_velocity=0.0, angular_velocity=0.0)

        # Desired direction in world frame
        to_target = target - agv_state.position
        desired_angle_world = float(np.arctan2(to_target[1], to_target[0]))

        # Build polar histogram in the AGV frame
        histogram = np.zeros(self.n_sectors, dtype=float)
        sector_size = 2.0 * np.pi / self.n_sectors

        for obs in fused_obstacles:
            rel = obs.position - agv_state.position
            dist = float(np.linalg.norm(rel))
            if dist > self.max_range or dist < 1e-6:
                continue
            # Weight per original Borenstein-Koren
            weight = self.a_const - self.b_const * dist
            if weight <= 0.0:
                continue
            # Angular half-footprint gamma = arcsin((r_obs + r_safe) / d)
            total_r = obs.radius + self.safety_radius
            ratio = total_r / max(dist, total_r + 1e-6)
            if ratio >= 1.0:
                # Obstacle engulfs the AGV: block all sectors
                histogram += weight
                continue
            gamma = float(np.arcsin(ratio))
            angle_to_obs = float(np.arctan2(rel[1], rel[0]))
            rel_angle = (angle_to_obs - agv_state.heading)
            rel_angle = (rel_angle + np.pi) % (2.0 * np.pi) - np.pi

            low = rel_angle - gamma
            high = rel_angle + gamma
            # Mark sectors falling in [low, high] (mod 2*pi).
            for k in range(self.n_sectors):
                # sector k centre (relative to heading, in [-pi, pi])
                sc = -np.pi + (k + 0.5) * sector_size
                if _angle_in_range(sc, low, high):
                    histogram[k] += weight

        # Mask blocked sectors
        blocked = histogram > self.h_threshold

        # Find the candidate sector whose centre is closest to the
        # desired (world-frame) angle.
        desired_rel = (desired_angle_world - agv_state.heading)
        desired_rel = (desired_rel + np.pi) % (2.0 * np.pi) - np.pi

        best_k = -1
        best_err = float("inf")
        for k in range(self.n_sectors):
            if blocked[k]:
                continue
            sc = -np.pi + (k + 0.5) * sector_size
            err = abs(_angle_wrap(sc - desired_rel))
            if err < best_err:
                best_err = err
                best_k = k

        v_max = float(self._sigma.dynamics.v_agv_max)
        omega_max = float(self._sigma.dynamics.omega_agv_max)

        if best_k == -1:
            # All sectors blocked -- stop and turn in place toward target
            heading_err = _angle_wrap(desired_rel)
            omega = _saturate(heading_err / max(dt, 1e-3), omega_max)
            return ControlAction(linear_velocity=0.0, angular_velocity=omega)

        chosen_rel = -np.pi + (best_k + 0.5) * sector_size
        heading_err = chosen_rel  # already in [-pi, pi], AGV-frame target

        # Speed shaping: proportional to cos(heading error) and clearance
        nearest_clearance = _min_clearance_ahead(
            fused_obstacles, agv_state, chosen_rel, self.max_range
        )
        clearance_frac = min(1.0, nearest_clearance / max(self.max_range, 1e-6))
        speed = v_max * max(0.0, float(np.cos(heading_err))) * clearance_frac

        omega = _saturate(
            heading_err / max(dt, 1e-3), omega_max
        )
        return ControlAction(
            linear_velocity=float(max(0.0, speed)),
            angular_velocity=float(omega),
        )


# ----- helpers ---------------------------------------------------------------

def _angle_wrap(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _angle_in_range(a: float, low: float, high: float) -> bool:
    """Is angle a in [low, high] considered as an arc (low -> high CCW)?"""
    a = _angle_wrap(a)
    low = _angle_wrap(low)
    high = _angle_wrap(high)
    if low <= high:
        return low <= a <= high
    # Crosses +/- pi
    return a >= low or a <= high


def _saturate(x: float, bound: float) -> float:
    if x > bound:
        return bound
    if x < -bound:
        return -bound
    return x


def _min_clearance_ahead(
    fused: List[FusedObstacle],
    agv_state: AGVState,
    chosen_rel_angle: float,
    max_range: float,
) -> float:
    """Minimum (centre-to-centre) obstacle distance in the chosen direction cone."""
    if not fused:
        return max_range
    cone = np.pi / 6  # +/- 30 deg cone
    min_d = max_range
    for obs in fused:
        rel = obs.position - agv_state.position
        d = float(np.linalg.norm(rel))
        if d >= min_d:
            continue
        angle = float(np.arctan2(rel[1], rel[0])) - agv_state.heading
        if abs(_angle_wrap(angle - chosen_rel_angle)) <= cone:
            min_d = d
    return min_d


register_avoidance("VFH", VFHAvoidance)

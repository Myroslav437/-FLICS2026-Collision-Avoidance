"""
Dynamic Window Approach (DWA) avoidance stage (Fox, Burgard, Thrun).

DWA samples velocity pairs within kinematic constraints and selects the
trajectory maximising a weighted objective of heading, clearance, and
speed.

Algorithm (Fox, Burgard, Thrun 1997)
------------------------------------
Let current (v, omega) be the AGV's forward and angular velocities.
The dynamic window is the set of (v', omega') pairs reachable from
(v, omega) within one time step given the dynamics constraints
Delta = <v_agv_max, a_agv_max, omega_agv_max>:

    v'     in [v     - a_max * dt,        v + a_max * dt] n [0, v_max]
    omega' in [omega - alpha_max * dt, omega + alpha_max * dt]
           n [-omega_max, +omega_max]

where alpha_max is a per-step angular acceleration. (Delta only
specifies omega_agv_max; we approximate alpha_max as omega_agv_max / dt
to keep the window well-shaped.)

For each sampled (v', omega') we integrate a short-horizon trajectory
(circular arc for `look_ahead` seconds) and score it with:

    score = w_head * heading + w_clear * clearance + w_vel * velocity

where:
  - heading   = 1 - |angle to target at trajectory end| / pi
  - clearance = min distance to any obstacle along the arc,
                normalised by `max_range`; infeasible arcs
                (collision at any sampled point) are discarded
  - velocity  = v' / v_max
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from world_generator.models import WaypointPath

from ...config import SimulationConfig
from ...types import AGVState, ControlAction, FusedObstacle
from ..base import AvoidanceStage
from ..registry import register_avoidance

from ._goal_tracker import WaypointGoalTracker


class DWAAvoidance(AvoidanceStage):
    """DWA avoidance (Fox, Burgard, Thrun)."""

    name = "DWA"

    def __init__(
        self,
        n_v_samples: int = 7,
        n_omega_samples: int = 21,
        look_ahead: float = 1.5,
        integ_steps: int = 10,
        w_heading: float = 0.7,
        w_clearance: float = 0.2,
        w_velocity: float = 0.1,
        safety_radius: float = 0.5,
        max_range: float = 5.0,
        goal_tolerance: float = 0.3,
    ) -> None:
        if n_v_samples < 2 or n_omega_samples < 3:
            raise ValueError("n_v_samples >= 2 and n_omega_samples >= 3 required")
        if look_ahead <= 0 or integ_steps < 2:
            raise ValueError("look_ahead and integ_steps must be positive")
        if safety_radius <= 0:
            raise ValueError("safety_radius must be positive")
        if not np.isclose(w_heading + w_clearance + w_velocity, 1.0):
            raise ValueError("DWA score weights must sum to 1.0")
        self.n_v_samples = int(n_v_samples)
        self.n_omega_samples = int(n_omega_samples)
        self.look_ahead = float(look_ahead)
        self.integ_steps = int(integ_steps)
        self.w_heading = float(w_heading)
        self.w_clearance = float(w_clearance)
        self.w_velocity = float(w_velocity)
        self.safety_radius = float(safety_radius)
        self.max_range = float(max_range)
        self.goal_tolerance = float(goal_tolerance)

        self._tracker: Optional[WaypointGoalTracker] = None
        self._sigma: Optional[SimulationConfig] = None
        self._v_prev: float = 0.0
        self._omega_prev: float = 0.0

    def reset(self, sigma: SimulationConfig, seed: int) -> None:
        self._sigma = sigma
        self._tracker = None
        self._v_prev = 0.0
        self._omega_prev = 0.0

    def step(
        self,
        fused_obstacles: List[FusedObstacle],
        reference_path: WaypointPath,
        agv_state: AGVState,
        dt: float,
    ) -> ControlAction:
        assert self._sigma is not None, "DWAAvoidance.reset() not called"
        if self._tracker is None:
            self._tracker = WaypointGoalTracker(reference_path)

        target = self._tracker.current_target(agv_state.position)
        goal = self._tracker.goal

        if np.linalg.norm(agv_state.position - goal) <= self.goal_tolerance:
            return ControlAction(linear_velocity=0.0, angular_velocity=0.0)

        dyn = self._sigma.dynamics
        alpha_max = dyn.omega_agv_max / max(dt, 1e-3)

        v_lo = max(0.0, self._v_prev - dyn.a_agv_max * dt)
        v_hi = min(dyn.v_agv_max, self._v_prev + dyn.a_agv_max * dt)
        if v_hi < v_lo:
            v_hi = v_lo
        w_lo = max(-dyn.omega_agv_max, self._omega_prev - alpha_max * dt)
        w_hi = min(+dyn.omega_agv_max, self._omega_prev + alpha_max * dt)

        vs = np.linspace(v_lo, v_hi, self.n_v_samples)
        ws = np.linspace(w_lo, w_hi, self.n_omega_samples)

        best_score = -float("inf")
        best_v = 0.0
        best_w = 0.0
        found_feasible = False

        for v in vs:
            for w in ws:
                feasible, end_pos, end_heading, clearance = _simulate_arc(
                    agv_state.position,
                    agv_state.heading,
                    float(v),
                    float(w),
                    look_ahead=self.look_ahead,
                    integ_steps=self.integ_steps,
                    fused_obstacles=fused_obstacles,
                    safety_radius=self.safety_radius,
                )
                if not feasible:
                    continue

                to_target = target - end_pos
                desired_angle = float(np.arctan2(to_target[1], to_target[0]))
                heading_err = abs(_angle_wrap(desired_angle - end_heading))
                heading_score = 1.0 - heading_err / np.pi
                clearance_score = min(1.0, clearance / self.max_range)
                velocity_score = v / max(dyn.v_agv_max, 1e-6)

                score = (
                    self.w_heading * heading_score
                    + self.w_clearance * clearance_score
                    + self.w_velocity * velocity_score
                )
                if score > best_score:
                    best_score = score
                    best_v = float(v)
                    best_w = float(w)
                    found_feasible = True

        if not found_feasible:
            # Every sample collides -- emergency stop & rotate toward target
            to_target = target - agv_state.position
            desired_angle = float(np.arctan2(to_target[1], to_target[0]))
            heading_err = _angle_wrap(desired_angle - agv_state.heading)
            omega = _saturate(heading_err / max(dt, 1e-3), dyn.omega_agv_max)
            action = ControlAction(linear_velocity=0.0, angular_velocity=float(omega))
            self._v_prev = 0.0
            self._omega_prev = action.angular_velocity
            return action

        self._v_prev = best_v
        self._omega_prev = best_w
        return ControlAction(
            linear_velocity=best_v, angular_velocity=best_w
        )


def _simulate_arc(
    position: np.ndarray,
    heading: float,
    v: float,
    omega: float,
    look_ahead: float,
    integ_steps: int,
    fused_obstacles: List[FusedObstacle],
    safety_radius: float,
):
    """
    Forward-integrate a unicycle under (v, omega) and check clearance
    against predicted obstacle positions (constant-velocity extrapolation
    from fused state).

    Returns (feasible, end_position, end_heading, min_clearance).
    """
    dt = look_ahead / integ_steps
    pos = position.copy()
    th = heading
    min_clearance = float("inf")

    # Predicted obstacle centres across time (constant-velocity):
    centres_t0 = np.array(
        [o.position for o in fused_obstacles]
    ) if fused_obstacles else np.empty((0, 2), dtype=float)
    velocities = np.array(
        [o.velocity for o in fused_obstacles]
    ) if fused_obstacles else np.empty((0, 2), dtype=float)
    radii = np.array(
        [o.radius for o in fused_obstacles]
    ) if fused_obstacles else np.empty((0,), dtype=float)

    for step in range(integ_steps):
        t = dt * (step + 1)
        th = th + omega * dt
        pos = pos + v * dt * np.array([np.cos(th), np.sin(th)])

        if centres_t0.shape[0] > 0:
            predicted = centres_t0 + t * velocities
            d = np.linalg.norm(predicted - pos, axis=1) - radii - safety_radius
            nearest = float(np.min(d))
            if nearest < 0.0:
                return False, pos, th, 0.0
            if nearest < min_clearance:
                min_clearance = nearest

    if min_clearance == float("inf"):
        min_clearance = safety_radius * 10.0  # nothing in the way
    return True, pos, th, float(min_clearance)


def _angle_wrap(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _saturate(x: float, bound: float) -> float:
    if x > bound:
        return bound
    if x < -bound:
        return -bound
    return x


register_avoidance("DWA", DWAAvoidance)

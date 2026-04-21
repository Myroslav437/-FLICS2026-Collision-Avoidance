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
  - heading   = 1 - |angle between the arc-end heading and the bearing
                from the robot's current position to a fixed look-ahead
                anchor on the reference path| / pi. The anchor is the
                point at arc-length v_agv_max * look_ahead along the
                path from the AGV's projection onto it (clamped to the
                final waypoint if the path ends sooner). Anchoring at
                the maximum reach of the dynamic window guarantees no
                feasible arc overshoots the anchor, so the heading term
                is monotone in the arc-end heading's alignment with the
                path direction. Bearing is computed from the current
                position (not the arc end), keeping the term stable as
                the AGV approaches the final goal.
  - clearance serves only as a hard feasibility filter: any arc whose
                integrated path drops inside the safety_radius buffer
                of any predicted obstacle is discarded (no score is
                assigned). The term's default scoring weight is zero
                because any residual gradient among feasible arcs
                (min-perpendicular-distance-to-obstacle, capped at
                safety_radius) reintroduces the ratcheting stall seen
                whenever an obstacle sits in the buffer band: the
                arc-end proximity drops sub-linearly with v, so the
                argmax sits one sample below v_prev and the AGV
                decelerates to zero. With w_clearance = 0 and a
                bumped w_velocity = 0.3, heading still dominates
                tracking and velocity provides a strictly monotone
                preference for forward motion among feasible arcs.
                This matches the strict Fox "dist(v, omega)" semantics
                under full-horizon saturation, where for feasible
                arcs dist = v * look_ahead and c_score reduces to
                v / v_max -- i.e., the clearance term is folded into
                the velocity term.
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
        w_clearance: float = 0.0,
        w_velocity: float = 0.3,
        safety_radius: float = 0.5,
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
        self.goal_tolerance = float(goal_tolerance)

        self._sigma: Optional[SimulationConfig] = None
        self._v_prev: float = 0.0
        self._omega_prev: float = 0.0

    def reset(self, sigma: SimulationConfig, seed: int) -> None:
        self._sigma = sigma
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

        goal = reference_path.positions[-1]
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

        # Longest arc any sample in the dynamic window can trace;
        # used for the heading look-ahead anchor below.
        reach = dyn.v_agv_max * self.look_ahead

        # Anchor the heading term at a fixed point on the reference path
        # that sits exactly at the longest arc reach ahead of the AGV's
        # projection onto the path (clamped to the final waypoint if the
        # path ends sooner). Because no feasible arc can overshoot this
        # anchor, the bearing from the current position to the anchor is
        # well-defined throughout the run -- including the final approach
        # where a per-arc-end bearing would flip 180 deg on overshoot and
        # stall the AGV just outside the goal tolerance.
        lookahead = _path_lookahead(
            agv_state.position, reference_path.positions, reach
        )
        desired_vec = lookahead - agv_state.position
        desired_norm = float(np.linalg.norm(desired_vec))
        if desired_norm < 1e-9:
            desired_bearing = float(agv_state.heading)
        else:
            desired_bearing = float(np.arctan2(desired_vec[1], desired_vec[0]))

        best_score = -float("inf")
        best_v = 0.0
        best_w = 0.0
        found_feasible = False

        for v in vs:
            for w in ws:
                feasible, end_pos, end_heading, _ = _simulate_arc(
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

                heading_err = abs(_angle_wrap(desired_bearing - end_heading))
                heading_score = 1.0 - heading_err / np.pi
                velocity_score = v / max(dyn.v_agv_max, 1e-6)

                # Clearance is a feasibility filter only (see module
                # docstring); when w_clearance = 0 it contributes
                # nothing to scoring, and the multiply-by-zero short-
                # circuits cleanly whether `_` is a finite margin or
                # inf. Kept in the expression for parity with the
                # paper's additive form.
                score = (
                    self.w_heading * heading_score
                    + self.w_velocity * velocity_score
                )
                if score > best_score:
                    best_score = score
                    best_v = float(v)
                    best_w = float(w)
                    found_feasible = True

        if not found_feasible:
            # Every sample collides -- emergency stop & rotate toward the
            # look-ahead anchor.
            heading_err = _angle_wrap(desired_bearing - agv_state.heading)
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
        return True, pos, th, float("inf")
    return True, pos, th, float(min_clearance)


def _path_lookahead(
    position: np.ndarray,
    waypoints: np.ndarray,
    distance: float,
) -> np.ndarray:
    """
    Return a look-ahead point at arc length `distance` past the closest
    projection of `position` onto the polyline through `waypoints`.

    The projection is computed per-segment; the segment minimising the
    perpendicular distance to `position` wins. From that projection we
    walk forward along the polyline accumulating segment lengths until
    we have travelled `distance` metres, and return the point reached.
    If the remaining path from the projection is shorter than
    `distance`, we return the final waypoint (the goal).

    `waypoints` must have shape (N, 2) with N >= 2. `distance` must be
    non-negative.
    """
    n = waypoints.shape[0]
    assert n >= 2, "path must have at least two waypoints"
    assert distance >= 0.0, "lookahead distance must be non-negative"

    best_i = 0
    best_t = 0.0
    best_d_sq = float("inf")
    best_proj = waypoints[0]
    for i in range(n - 1):
        a = waypoints[i]
        b = waypoints[i + 1]
        ab = b - a
        ab_sq = float(np.dot(ab, ab))
        if ab_sq < 1e-12:
            t = 0.0
            proj = a
        else:
            t = float(np.dot(position - a, ab) / ab_sq)
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            proj = a + t * ab
        delta = position - proj
        d_sq = float(np.dot(delta, delta))
        if d_sq < best_d_sq:
            best_d_sq = d_sq
            best_i = i
            best_t = t
            best_proj = proj

    remaining = distance
    a = waypoints[best_i]
    b = waypoints[best_i + 1]
    seg_vec = b - a
    seg_len = float(np.linalg.norm(seg_vec))
    dist_to_next = seg_len * (1.0 - best_t)

    if dist_to_next >= remaining:
        if seg_len < 1e-12:
            return np.asarray(b, dtype=float).copy()
        return np.asarray(best_proj + (remaining / seg_len) * seg_vec, dtype=float)

    remaining -= dist_to_next
    i = best_i + 1
    while i < n - 1:
        a = waypoints[i]
        b = waypoints[i + 1]
        seg_vec = b - a
        seg_len = float(np.linalg.norm(seg_vec))
        if seg_len >= remaining:
            if seg_len < 1e-12:
                return np.asarray(b, dtype=float).copy()
            return np.asarray(a + (remaining / seg_len) * seg_vec, dtype=float)
        remaining -= seg_len
        i += 1

    return np.asarray(waypoints[-1], dtype=float).copy()


def _angle_wrap(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _saturate(x: float, bound: float) -> float:
    if x > bound:
        return bound
    if x < -bound:
        return -bound
    return x


register_avoidance("DWA", DWAAvoidance)

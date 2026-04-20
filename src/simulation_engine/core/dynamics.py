"""
World-step dynamics: advance AGV and obstacles by one dt.

AGV dynamics
------------
Unicycle with saturation:
    v_des, omega_des = control action
    v_max, a_max     : linear velocity and acceleration limits (Delta)
    omega_max        : angular velocity limit (Delta)
    dt               : integration step (1 / f_scan)

At each step:
    v_t  = clamp(v_des, 0, v_max)
    dv   = clamp(v_t - |v_{t-1}|, -a_max*dt, +a_max*dt)
    v_t  = |v_{t-1}| + dv
    w_t  = clamp(omega_des, -omega_max, +omega_max)
    theta_{t+1} = theta_t + w_t * dt
    P_{t+1}     = P_t + v_t * dt * (cos theta_{t+1}, sin theta_{t+1})
    v_vec_{t+1} = v_t * (cos theta_{t+1}, sin theta_{t+1})

The controller speaks (v, omega); the world stores velocity as a 2D
vector v_t.

Obstacle dynamics
-----------------
Each obstacle carries its own xi_i (WaypointPath). At each step we
integrate each obstacle's trajectory progress tau_i(t+dt) = tau_i(t)
+ v_j * dt, where v_j is the target speed on the current segment; we
clamp tau at |xi_i| and keep the obstacle at the last waypoint after
that. Static obstacles (|xi_i| = 0) are untouched.
"""

from __future__ import annotations

from typing import List

import numpy as np

from world_generator.models import Obstacle

from ..config import DynamicsParams
from ..types import AGVState, ControlAction


def step_agv(
    state: AGVState,
    action: ControlAction,
    dynamics: DynamicsParams,
    dt: float,
) -> AGVState:
    """
    Advance the AGV one step under the supplied control command and Delta
    bounds. See module docstring for the integration scheme.
    """
    current_speed = float(np.linalg.norm(state.velocity))
    v_target = max(0.0, min(action.linear_velocity, dynamics.v_agv_max))
    dv_max = dynamics.a_agv_max * dt
    dv = v_target - current_speed
    if dv > dv_max:
        dv = dv_max
    elif dv < -dv_max:
        dv = -dv_max
    v_new = max(0.0, current_speed + dv)

    omega = max(
        -dynamics.omega_agv_max,
        min(action.angular_velocity, dynamics.omega_agv_max),
    )

    theta_new = state.heading + omega * dt
    theta_new = float((theta_new + np.pi) % (2.0 * np.pi) - np.pi)
    direction = np.array([np.cos(theta_new), np.sin(theta_new)])
    position_new = state.position + v_new * dt * direction
    velocity_new = v_new * direction

    return AGVState(
        position=position_new,
        heading=theta_new,
        velocity=velocity_new,
    )


def step_obstacle_progresses(
    obstacles: List[Obstacle],
    progresses: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Advance each obstacle's trajectory progress by dt * v_current_segment.

    Parameters
    ----------
    obstacles  : list of Obstacle (each carries xi_i and speeds)
    progresses : (K,) array of current progress per obstacle
    dt         : time step
    """
    out = progresses.copy()
    for i, obs in enumerate(obstacles):
        if obs.trajectory.is_stationary:
            continue
        pts = obs.trajectory.positions
        deltas = np.diff(pts, axis=0)
        seg_len = np.linalg.norm(deltas, axis=1)
        total = float(np.sum(seg_len))
        if total <= 0:
            continue
        tau = float(out[i])
        cum = np.cumsum(seg_len)
        # Which segment are we currently on?
        clamped = max(0.0, min(tau, total))
        seg_idx = int(np.searchsorted(cum, clamped, side="right"))
        seg_idx = min(seg_idx, len(seg_len) - 1)
        v = float(obs.trajectory.speeds[seg_idx])
        tau_next = tau + v * dt
        if tau_next > total:
            tau_next = total
        out[i] = tau_next
    return out


def obstacle_velocity_at(
    obstacle: Obstacle,
    progress: float,
) -> np.ndarray:
    """
    Ground-truth velocity of an obstacle at the given progress along xi_i.

    For static obstacles returns the zero vector. For dynamic obstacles
    returns `v_j * (segment direction)` where j is the current segment.
    Returns zero once progress >= |xi_i| (obstacle has arrived).
    """
    if obstacle.trajectory.is_stationary:
        return np.zeros(2, dtype=float)
    pts = obstacle.trajectory.positions
    deltas = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(deltas, axis=1)
    total = float(np.sum(seg_len))
    if total <= 0 or progress >= total:
        return np.zeros(2, dtype=float)
    cum = np.cumsum(seg_len)
    seg_idx = int(np.searchsorted(cum, max(0.0, progress), side="right"))
    seg_idx = min(seg_idx, len(seg_len) - 1)
    v_scalar = float(obstacle.trajectory.speeds[seg_idx])
    if seg_len[seg_idx] < 1e-12:
        return np.zeros(2, dtype=float)
    direction = deltas[seg_idx] / seg_len[seg_idx]
    return v_scalar * direction

"""
Regression tests for DWA scoring pathologies.

Two distinct stall modes have been observed in benchmarking:

1. Clearance-term ratchet: a distant-but-visible obstacle within the
   LiDAR horizon (5 m) produced a residual clearance-term gradient that
   biased the argmax one velocity slot below v_prev each step until the
   AGV stopped. Fixed by normalising the clearance term by
   v_agv_max * look_ahead (the longest arc the dynamic window can
   trace) rather than by the sensor horizon.

2. Heading / look-ahead overshoot: the per-arc heading term took the
   bearing from the arc-end to the active carrot waypoint, so whenever
   the carrot was closer than the maximum arc reach (2.25 m) a full-
   speed arc would overshoot it and the bearing would flip ~pi. The
   heading term then preferred low-v arcs that stopped short, which
   manifested as the AGV freezing 0.3-0.4 m outside the goal tolerance
   or 1-3 m short of a waypoint downstream of an obstacle it had just
   cleared. Fixed by anchoring the heading term at a continuous
   look-ahead point on the reference path (arc length v_max * look_ahead
   ahead of the AGV's projection onto the path, clamped to the final
   waypoint near the goal) and by computing the bearing from the AGV's
   current position rather than the arc end -- the arc end can safely
   overshoot a closer goal without flipping the heading score.

Both tests replay hand-picked states that exercise each pathology and
assert DWA produces forward motion.
"""

from __future__ import annotations

import numpy as np

from src.simulation_engine.config import (
    DynamicsParams,
    LiDARDistortion,
    LiDARGeometry,
    LiDARParams,
    PerceptionParams,
    SimulationConfig,
)
from src.simulation_engine.stages.avoidance.dwa import DWAAvoidance
from src.simulation_engine.types import AGVState, FusedObstacle
from src.world_generator.models import WaypointPath


def _nominal_sigma() -> SimulationConfig:
    return SimulationConfig(
        name="nominal",
        lidar=LiDARParams(
            geometry=LiDARGeometry(
                r_min=0.05,
                r_max=10.0,
                theta_fov=4.71238898038469,
                delta_theta=0.008726646259971648,
                f_scan=15.0,
            ),
            distortion=LiDARDistortion(sigma_r=0.0, p_miss=0.0, p_ghost=0.0),
        ),
        perception=PerceptionParams(eta_cpl=1.0, sigma_map=0.0),
        dynamics=DynamicsParams(
            v_agv_max=1.5,
            a_agv_max=1.0,
            omega_agv_max=1.0,
            T_horizon=120.0,
        ),
    )


def test_dwa_does_not_stall_with_distant_obstacle() -> None:
    """
    Clear-path scenario: single static obstacle well beyond the maximum
    trajectory reach, AGV already moving, reference heading aligned with
    the next look-ahead waypoint. DWA must pick a linear velocity at
    least as high as v_prev (no ratchet-down).
    """
    sigma = _nominal_sigma()
    dt = 1.0 / sigma.lidar.geometry.f_scan

    # Reference path drawn from world_00038.json, through wp4 which sits
    # directly south of the stall position.
    positions = np.array(
        [
            [2.716, 28.501],
            [1.919, 27.353],
            [1.058, 24.950],
            [1.113, 23.561],
            [1.207, 20.331],
            [1.276, 16.729],
            [0.910, 12.729],
            [15.482, 3.748],
        ]
    )
    path = WaypointPath(
        positions=positions,
        speeds=np.ones(positions.shape[0] - 1),
    )

    dwa = DWAAvoidance()
    dwa.reset(sigma, seed=0)
    # Carry-in state observed at world_00038 step 199.
    v_prev = 0.3556
    dwa._v_prev = v_prev
    dwa._omega_prev = 0.0

    # The single static obstacle fused at step 199: 5.6 m south,
    # approximately on the path. Longest arc the window can trace is
    # v_max * look_ahead = 1.5 * 1.5 = 2.25 m, so the obstacle cannot
    # possibly intersect any feasible sample.
    obstacle = FusedObstacle(
        track_id=0,
        position=np.array([1.863, 16.030]),
        velocity=np.array([0.0, 0.0]),
        radius=0.16,
    )
    agv = AGVState(
        position=np.array([1.170, 22.131]),
        heading=-1.5104,
        velocity=np.array([-0.033, -0.59]),
    )

    action = dwa.step([obstacle], path, agv, dt=dt)

    # Strictly positive forward motion is required, and the argmax must
    # not sit below v_prev -- that was the pre-fix failure mode
    # (decrement of ~2 * a_max * dt / (n_v_samples - 1) per step).
    assert action.linear_velocity > 0.0, (
        f"DWA selected zero velocity: v={action.linear_velocity}"
    )
    assert action.linear_velocity >= v_prev - 1e-9, (
        f"DWA ratcheted velocity down: v={action.linear_velocity} "
        f"< v_prev={v_prev} (the observed pre-fix regression)"
    )


def test_dwa_does_not_stall_near_goal_overshoot() -> None:
    """
    Final-approach scenario with no obstacles: AGV is 0.5 m from the
    goal (outside the 0.3 m tolerance), heading pointed directly at the
    goal. The per-arc-end bearing formula flipped ~pi on overshoot and
    made low-v arcs win; the look-ahead-anchor formula keeps the
    bearing fixed from the current position and rewards forward motion.
    DWA must pick a strictly positive v.
    """
    sigma = _nominal_sigma()
    dt = 1.0 / sigma.lidar.geometry.f_scan

    # A short two-segment path. The AGV sits 0.5 m before the final
    # waypoint, heading straight at it. Any arc with v > 0 immediately
    # overshoots the goal (arc reach up to 2.25 m).
    positions = np.array(
        [
            [0.0, 0.0],
            [0.0, 5.0],
            [0.0, 10.0],
        ]
    )
    path = WaypointPath(
        positions=positions,
        speeds=np.ones(positions.shape[0] - 1),
    )

    dwa = DWAAvoidance()
    dwa.reset(sigma, seed=0)
    dwa._v_prev = 0.5
    dwa._omega_prev = 0.0

    agv = AGVState(
        position=np.array([0.0, 9.5]),
        heading=float(np.pi / 2.0),
        velocity=np.array([0.0, 0.5]),
    )

    action = dwa.step([], path, agv, dt=dt)

    assert action.linear_velocity > 0.0, (
        "DWA stalled outside the goal tolerance: "
        f"v={action.linear_velocity} at 0.5 m from goal with clear path"
    )

"""
Simulation Engine data types.

Dataclasses describing perception outputs, stage outputs, telemetry, and
per-run metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from world_generator.models import WaypointPath


# =============================================================================
# Perception outputs
# =============================================================================

@dataclass
class PriorMap:
    """
    M_0 = P(E, eta_cpl, sigma_map).

    A degraded copy of the environment structure E. Stored as a list of
    closed polygonal chains; each chain is a (k_j, 2) ndarray in world
    frame, identical in shape to EnvironmentStructure.chains but having
    undergone dropout (eta_cpl) and per-vertex jitter (sigma_map).
    """

    chains: List[np.ndarray]
    size: tuple  # (X, Y)


@dataclass
class LiDARScan:
    """
    L_hat_t = D(S(E, O_t, P_t, theta_t, Lambda_S), Lambda_D).

    Each scan is a collection of (range, angle) readings in the AGV's
    local frame (angle relative to heading). Rays that returned no
    reading (p_miss) are represented by `r = inf`; ghost rays are
    inserted in-place of valid readings.

    Fields
    ------
    ranges  : shape (N_rays,), radial distance per ray; inf = no return
    angles  : shape (N_rays,), angle relative to heading, in [-fov/2, +fov/2]
    time    : the simulation time of the scan (s)
    pose    : (P_t, theta_t) at which the scan was taken (world frame)
    """

    ranges: np.ndarray
    angles: np.ndarray
    time: float
    pose_position: np.ndarray
    pose_heading: float

    def cartesian_world(self) -> np.ndarray:
        """
        Project the valid returns to Cartesian world coordinates.

        Returns an (M, 2) array where M = number of finite-range rays.
        """
        mask = np.isfinite(self.ranges)
        r = self.ranges[mask]
        a = self.angles[mask] + self.pose_heading
        x = self.pose_position[0] + r * np.cos(a)
        y = self.pose_position[1] + r * np.sin(a)
        return np.column_stack([x, y])


# =============================================================================
# AGV runtime state (the time-varying fields of W_t and Psi(W_t))
# =============================================================================

@dataclass
class AGVState:
    """
    (P_t, theta_t, v_t) -- the AGV's time-varying pose and velocity.

    Corresponds to the last three fields of the world state W_t and is
    carried unchanged into the perceived world; self-pose is
    oracle-known.
    """

    position: np.ndarray   # (2,) world frame
    heading: float         # radians, in [-pi, pi]
    velocity: np.ndarray   # (2,) world frame


# =============================================================================
# Perceived world
# =============================================================================

@dataclass
class PerceivedWorld:
    """
    W_tilde_t = Psi(W_t) = <M_0, pi, S_agv, L_hat_t, P_t, theta_t, v_t>.

    This is what the pipeline sees. No direct access to E, O_t, or any
    ground-truth obstacle information.
    """

    prior_map: PriorMap
    reference_path: WaypointPath
    agv_extent: "object"  # AGVExtent (world_generator); cyclic-import avoidance
    lidar_scan: LiDARScan
    agv_state: AGVState


# =============================================================================
# Stage outputs
# =============================================================================

@dataclass
class DetectedObstacle:
    """
    O_hat_t element -- the output of the detection stage.

    A detected obstacle candidate derived from a single scan; no temporal
    continuity. Represented as the centroid + extent of a cluster.

    Fields
    ------
    centroid : (2,) world frame centre of the cluster
    radius   : effective radius (max distance from centroid to cluster point)
    points   : (K, 2) world-frame member points (for downstream use /
               telemetry); may be empty if stage does not keep them
    """

    centroid: np.ndarray
    radius: float
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))


@dataclass
class FusedObstacle:
    """
    O_bar_t element -- the output of the fusion stage.

    Carries the fusion stage's estimate of position, velocity, and radius.
    For pass-through fusion, velocity is zero.

    Fields
    ------
    track_id : integer identity, consistent across steps for KF; per-step
               anonymous for pass-through (simply index in the list)
    position : (2,) world-frame position estimate
    velocity : (2,) world-frame velocity estimate (0 for pass-through)
    radius   : effective radius (as in DetectedObstacle)
    """

    track_id: int
    position: np.ndarray
    velocity: np.ndarray
    radius: float


@dataclass
class ControlAction:
    """
    The output of the avoidance stage.

    A kinematic command expressed as a target linear velocity magnitude
    and a target angular velocity. The simulation core clamps these to
    the Delta bounds (v_max, a_max, omega_max) and integrates them.

    Fields
    ------
    linear_velocity  : m/s, target forward speed (>= 0)
    angular_velocity : rad/s, target angular velocity
    """

    linear_velocity: float
    angular_velocity: float


# =============================================================================
# Per-step measurements (telemetry + metric aggregation)
# =============================================================================

@dataclass
class StepTelemetry:
    """
    One step's internal state, for fire-and-forget emission.

    Fields cover both W_t (true) and W_tilde_t (perceived) subsets so the
    visualizer can overlay ground truth on perceived quantities. The
    simulation core does not consume this structure; it only writes.
    """

    step: int
    time: float
    agv_state: AGVState
    obstacle_centroids: np.ndarray   # (K, 2), ground truth O_t centroids
    obstacle_velocities: np.ndarray  # (K, 2), ground truth O_t velocities
    obstacle_radii: np.ndarray       # (K,), ground truth obstacle radii
    lidar_scan: LiDARScan
    detections: List[DetectedObstacle]
    fused: List[FusedObstacle]
    control: ControlAction
    collided: bool
    goal_reached: bool
    step_wall_time: float  # seconds, for mu_comp / mu_lat


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class RunMetrics:
    """
    M = <mu_col, mu_dev, mu_goal, mu_vel, mu_comp, mu_lat>.

    Per-run metrics:
      mu_col   : binary {0, 1} collision indicator for this run
                 (a single run is either a "collision run" or not)
      mu_dev   : mean perpendicular distance from the AGV trajectory to
                 the nearest segment of pi (meters)
      mu_goal  : binary {0, 1} goal-reached indicator for this run
                 (1 iff the AGV reached w_n within T_horizon)
      mu_vel   : RMSE between fusion-estimated and ground-truth velocities
                 across dynamic obstacles and steps (m/s). NaN if the run
                 has no dynamic obstacles.
      mu_comp  : mean per-step CPU time (seconds)
      mu_lat   : 95th-percentile per-step CPU time (seconds)
    """

    mu_col: int
    mu_dev: float
    mu_goal: int
    mu_vel: float
    mu_comp: float
    mu_lat: float

    # Ancillary fields for downstream analysis / audit
    steps: int
    duration: float  # simulation seconds covered
    collision_step: Optional[int] = None
    goal_reach_step: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "mu_col": int(self.mu_col),
            "mu_dev": float(self.mu_dev),
            "mu_goal": int(self.mu_goal),
            "mu_vel": (float(self.mu_vel) if np.isfinite(self.mu_vel)
                       else None),
            "mu_comp": float(self.mu_comp),
            "mu_lat": float(self.mu_lat),
            "steps": int(self.steps),
            "duration": float(self.duration),
            "collision_step": (None if self.collision_step is None
                               else int(self.collision_step)),
            "goal_reach_step": (None if self.goal_reach_step is None
                                else int(self.goal_reach_step)),
        }

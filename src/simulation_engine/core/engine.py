"""
Simulation core  SE(W_0, A, Sigma, s_se) -> M.

Implements the closed-loop dynamics

    W_t = f(W_{t-1}; A, Sigma)

with the perception function Psi folded into f.

Loop structure
--------------
Each step performs, in order:

  1. Observe: W_tilde_t = Psi(W_t)           (perception)
  2. Decide : a_t = A(W_tilde_t)             (pipeline stages)
  3. Act    : W_{t+1} = f(W_t, a_t)          (unicycle + obstacle integration)
  4. Record : collision, goal, metrics, telemetry.

Termination conditions (any one of):
  - the AGV reached the final waypoint w_n (within goal tolerance), or
  - the AGV collided with E or an obstacle, or
  - sim_time >= T_horizon.

Purity
------
The engine has no CLI, no plotting, and no file I/O beyond the supplied
`TelemetrySink` (which may be a `NullSink`). It consumes and returns
pure dataclasses; callers are responsible for persistence and
reporting.

Modularity
----------
Concrete stages are resolved by name through the `stages.registry`;
the engine never imports EC, VFH, DWA, etc. Adding a new algorithm
does not require changes in this module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from world_generator.models import WorldState

from ..config import SimulationConfig
from ..perception import apply_perception, degrade_prior_map
from ..perception.lidar import obstacle_world_polygon
from ..stages.registry import get_avoidance, get_detection, get_fusion
from ..types import (
    AGVState,
    FusedObstacle,
    RunMetrics,
    StepTelemetry,
)
from .collision import agv_collides, build_environment_occupancy
from .dynamics import obstacle_velocity_at, step_agv, step_obstacle_progresses
from .metrics import MetricsAccumulator
from .telemetry import NullSink, TelemetrySink, build_header


# =============================================================================
# Engine configuration
# =============================================================================

@dataclass
class PipelineSpec:
    """Identifies the three stage implementations to compose (by name)."""

    detection: str
    fusion: str
    avoidance: str


@dataclass
class EngineOptions:
    """
    Engine-side knobs that do not belong in Sigma or the pipeline
    (algorithm choice).
    """

    goal_tolerance: float = 0.3
    telemetry_sink: Optional[TelemetrySink] = None


# =============================================================================
# SimulationEngine
# =============================================================================

class SimulationEngine:
    """
    Closed-loop simulator.

    Typical usage:
        engine = SimulationEngine(pipeline=PipelineSpec("EC", "KF", "VFH"))
        metrics = engine.run(world, sigma, seed=42)
    """

    def __init__(
        self,
        pipeline: PipelineSpec,
        options: Optional[EngineOptions] = None,
    ) -> None:
        self._pipeline = pipeline
        self._options = options if options is not None else EngineOptions()

    # --- public API ----------------------------------------------------------

    def run(
        self,
        world: WorldState,
        sigma: SimulationConfig,
        seed: int,
    ) -> RunMetrics:
        """
        Run a full simulation and return M.

        Parameters
        ----------
        world : W_0 from the world generator.
        sigma : Sigma = <Lambda, Phi, Delta>.
        seed  : s_se; seeds all stochastic draws inside the
                perception function Psi. Stages use their own `reset(seed)`.
        """
        rng = np.random.default_rng(int(seed))

        # --- Perception: prior map M_0, computed once per run. ---
        prior_map = degrade_prior_map(
            environment=world.environment,
            eta_cpl=sigma.perception.eta_cpl,
            sigma_map=sigma.perception.sigma_map,
            rng=rng,
        )

        # --- Collision geometry (time-invariant, derived from E). ----------
        env_occupancy = build_environment_occupancy(world)

        # --- Instantiate pipeline stages via the registry. -----------------
        detector = get_detection(self._pipeline.detection)
        fuser = get_fusion(self._pipeline.fusion)
        avoider = get_avoidance(self._pipeline.avoidance)
        detector.reset(int(seed))
        fuser.reset(int(seed))
        avoider.reset(sigma, int(seed))

        # --- AGV state (W_0 -> mutable) ------------------------------------
        agv_state = AGVState(
            position=world.agv_position.copy(),
            heading=float(world.agv_heading),
            velocity=world.agv_velocity.copy(),
        )

        # --- Obstacle progress along xi_i ----------------------------------
        progresses = np.zeros(len(world.obstacles), dtype=float)
        dynamic_indices = [
            i for i, o in enumerate(world.obstacles) if o.is_dynamic
        ]

        # --- Timing & metrics ----------------------------------------------
        dt = 1.0 / sigma.lidar.geometry.f_scan
        T = sigma.dynamics.T_horizon
        max_steps = int(np.ceil(T / dt))
        goal_position = world.reference_path.positions[-1]
        goal_tol = self._options.goal_tolerance
        sink: TelemetrySink = self._options.telemetry_sink or NullSink()

        sink.write_header(
            build_header(
                world=world,
                sigma=sigma,
                detection_name=self._pipeline.detection,
                fusion_name=self._pipeline.fusion,
                avoidance_name=self._pipeline.avoidance,
                seed=int(seed),
            )
        )

        metrics = MetricsAccumulator(reference_path=world.reference_path)

        # -------------------------------------------------------------------
        # Main loop: W_0 -> W_1 -> ... -> W_T
        # -------------------------------------------------------------------
        sim_time = 0.0
        step_idx = 0
        collided = False
        goal_reached = False

        for step_idx in range(max_steps):
            t_step_start = time.perf_counter()

            # Ground-truth obstacle polygons at current progresses (for S).
            obstacle_polys = [
                obstacle_world_polygon(obs, float(progresses[i]))
                for i, obs in enumerate(world.obstacles)
            ]

            # 1. Psi(W_t) — perception.
            perceived = apply_perception(
                environment=world.environment,
                obstacle_polygons_world=obstacle_polys,
                reference_path=world.reference_path,
                agv_extent=world.agv_extent,
                agv_state=agv_state,
                prior_map=prior_map,
                lidar=sigma.lidar,
                sim_time=sim_time,
                rng=rng,
            )

            # 2. A(W_tilde_t) — pipeline. Per §IV-B, mu_lat is the 95th
            # percentile of the three-stage "decision time" — the sum of
            # the detection, fusion, and avoidance calls — and must not
            # include perception, dynamics, collision checks, or any
            # other per-step bookkeeping.
            t_det_start = time.perf_counter()
            detections = detector.step(
                scan=perceived.lidar_scan,
                prior_map=prior_map,
                agv_state=agv_state,
            )
            t_fus_start = time.perf_counter()
            fused: List[FusedObstacle] = fuser.step(detections, dt)
            t_avoid_start = time.perf_counter()
            control = avoider.step(
                fused_obstacles=fused,
                reference_path=world.reference_path,
                agv_state=agv_state,
                dt=dt,
            )
            t_avoid_end = time.perf_counter()
            decision_time = (
                (t_fus_start - t_det_start)
                + (t_avoid_start - t_fus_start)
                + (t_avoid_end - t_avoid_start)
            )

            # Record mu_vel samples before stepping the world so we score
            # the fusion estimate at the same time the obstacle truly is.
            dyn_centroids, dyn_velocities = _dynamic_truth(
                world, progresses, dynamic_indices
            )
            metrics.record_velocity_samples(fused, dyn_centroids, dyn_velocities)

            # 3. f(W_t, a_t) — integrate the world.
            agv_state = step_agv(agv_state, control, sigma.dynamics, dt)
            progresses = step_obstacle_progresses(world.obstacles, progresses, dt)
            sim_time += dt

            # 4. Collision and goal tests (evaluated on W_{t+1}).
            stepped_polys = [
                obstacle_world_polygon(obs, float(progresses[i]))
                for i, obs in enumerate(world.obstacles)
            ]
            footprint = world.agv_extent.footprint_world_frame(
                agv_state.position, agv_state.heading
            )
            if agv_collides(footprint, env_occupancy, stepped_polys):
                collided = True
                metrics.record_collision(step_idx)

            if np.linalg.norm(agv_state.position - goal_position) <= goal_tol:
                goal_reached = True
                metrics.record_goal(step_idx)

            # 5. Per-step bookkeeping.
            metrics.record_deviation(agv_state.position)
            step_wall = time.perf_counter() - t_step_start
            metrics.record_step_time(step_wall, decision_time)

            # Telemetry (post-step state — what the visualizer should draw).
            if world.obstacles:
                post_polys = [
                    obstacle_world_polygon(obs, float(progresses[i]))
                    for i, obs in enumerate(world.obstacles)
                ]
                all_centroids = np.array([p.mean(axis=0) for p in post_polys])
                all_velocities = np.array([
                    obstacle_velocity_at(o, float(progresses[i]))
                    for i, o in enumerate(world.obstacles)
                ])
                all_radii = np.array(
                    [o.spatial.bounding_radius() for o in world.obstacles]
                )
            else:
                all_centroids = np.empty((0, 2))
                all_velocities = np.empty((0, 2))
                all_radii = np.empty((0,))

            sink.write_step(StepTelemetry(
                step=int(step_idx),
                time=float(sim_time),
                agv_state=AGVState(
                    position=agv_state.position.copy(),
                    heading=agv_state.heading,
                    velocity=agv_state.velocity.copy(),
                ),
                obstacle_centroids=all_centroids,
                obstacle_velocities=all_velocities,
                obstacle_radii=all_radii,
                lidar_scan=perceived.lidar_scan,
                detections=detections,
                fused=fused,
                control=control,
                collided=collided,
                goal_reached=goal_reached,
                step_wall_time=step_wall,
            ))

            if collided or goal_reached:
                break

        sink.close()
        return metrics.finalise(steps=step_idx + 1, duration=sim_time)


# =============================================================================
# Helpers
# =============================================================================

def _dynamic_truth(
    world: WorldState,
    progresses: np.ndarray,
    dynamic_indices: List[int],
):
    """
    Stack ground-truth (centroid, velocity) for each dynamic obstacle at the
    pre-step progress; returns two empty arrays when there are no dynamic
    obstacles.
    """
    if not dynamic_indices:
        return np.empty((0, 2)), np.empty((0, 2))

    centroids = np.empty((len(dynamic_indices), 2), dtype=float)
    velocities = np.empty((len(dynamic_indices), 2), dtype=float)
    for k, i in enumerate(dynamic_indices):
        obs = world.obstacles[i]
        poly = obstacle_world_polygon(obs, float(progresses[i]))
        centroids[k] = poly.mean(axis=0)
        velocities[k] = obstacle_velocity_at(obs, float(progresses[i]))
    return centroids, velocities


# =============================================================================
# Convenience entry point
# =============================================================================

def run_simulation(
    world: WorldState,
    sigma: SimulationConfig,
    detection: str,
    fusion: str,
    avoidance: str,
    seed: int,
    telemetry_sink: Optional[TelemetrySink] = None,
    goal_tolerance: float = 0.3,
) -> RunMetrics:
    """
    Thin functional facade over SimulationEngine.run for ad-hoc callers
    (tests, CLI). Equivalent to:

        SimulationEngine(PipelineSpec(detection, fusion, avoidance),
                         EngineOptions(goal_tolerance, telemetry_sink)).run(
            world, sigma, seed)
    """
    engine = SimulationEngine(
        pipeline=PipelineSpec(
            detection=detection, fusion=fusion, avoidance=avoidance
        ),
        options=EngineOptions(
            goal_tolerance=goal_tolerance,
            telemetry_sink=telemetry_sink,
        ),
    )
    return engine.run(world, sigma, int(seed))


__all__ = [
    "EngineOptions",
    "PipelineSpec",
    "SimulationEngine",
    "run_simulation",
]

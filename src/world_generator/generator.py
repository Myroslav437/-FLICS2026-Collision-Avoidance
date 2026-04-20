"""
Top-level world generator.

Implements the function

    WG(Omega, s_wg) -> W_0

by sequencing five stages:

  1. Generate environment E via BSP
  2. Define AGV extent S_agv
  3. Plan AGV reference path pi via PRM
  4. Place static and dynamic obstacles
  5. Plan dynamic-obstacle trajectories xi_i via PRM

The output is the initial world state W_0 at t=0 together with a
snapshot of Omega for reproducibility audit.
"""

from __future__ import annotations

import numpy as np

from .bsp import generate_environment
from .config import Omega
from .models import AGVExtent, WorldState
from .obstacles import generate_obstacles
from .prm import (
    inflate_free_space,
    largest_component,
    plan_reference_path,
)


def generate_world(omega: Omega, seed: int, world_id: int = 0) -> WorldState:
    """
    Produce a single W_0 from (Omega, s_wg).

    Parameters
    ----------
    omega : Omega
        Full parameter set.
    seed : int
        s_wg, the world-generation seed.
    world_id : int
        Corpus index (used only for bookkeeping).
    """
    rng = np.random.default_rng(seed)

    # Stage 1: environment E
    bsp = generate_environment(omega.env, rng)

    # Stage 2: AGV extent
    agv_extent = AGVExtent(length=omega.agv.L_agv, width=omega.agv.W_agv)

    # Stage 3: reference path pi via PRM on d_clear-inflated free space
    path_region = largest_component(
        inflate_free_space(bsp.free_space, omega.path.clearance)
    )
    # Sampled start-goal pair is rejected when the shortest-path length on
    # the roadmap falls below l_pi_min. The rejection loop returns the
    # attempt count for diagnostic reporting.
    reference_path, reference_path_attempts = plan_reference_path(
        inflated_free=path_region,
        n_samples=omega.path.prm_samples,
        k_nn=omega.path.prm_neighbors,
        l_max=omega.path.max_edge_length,
        speed_min=omega.path.speed_min,
        speed_max=omega.path.speed_max,
        min_path_length=omega.path.min_path_length,
        rng=rng,
    )

    # Stages 4-5: static + dynamic obstacles
    obstacles, _retries = generate_obstacles(
        free_space=bsp.free_space,
        reference_path=reference_path,
        params=omega.obs,
        rng=rng,
    )

    # AGV initial pose: P_0 is the first waypoint of pi; theta_0 is the
    # heading of pi's first segment; v_0 = 0 (AGV starts at rest).
    agv_position = reference_path.positions[0].copy()
    agv_heading = reference_path.heading(0)
    agv_velocity = np.zeros(2, dtype=float)

    world = WorldState(
        environment=bsp.environment,
        obstacles=obstacles,
        reference_path=reference_path,
        agv_extent=agv_extent,
        agv_position=agv_position,
        agv_heading=agv_heading,
        agv_velocity=agv_velocity,
        seed=int(seed),
        world_id=int(world_id),
        parameters=omega.to_dict(),
    )
    # Attach the BSP-level c_min observation for downstream metrics. Not
    # serialized; validation re-reads it in-process after generation.
    world._bsp_min_corridor_width = float(bsp.min_corridor_width)
    # Attach the l_pi_min rejection-loop attempt count.
    # Not serialized; surfaced through compute_metrics for the manifest.
    world._prm_reference_attempts = int(reference_path_attempts)
    return world

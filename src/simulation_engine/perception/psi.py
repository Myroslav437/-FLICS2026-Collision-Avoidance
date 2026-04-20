"""
Perception function  Psi.

    Psi(W_t) = <M_0, pi, S_agv, L_hat_t, P_t, theta_t, v_t>

Composes the prior-map degradation, the ideal LiDAR scan, and the
LiDAR distortion. The AGV self-pose (P_t, theta_t, v_t) is carried
through unchanged; self-pose is oracle-known.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from world_generator.models import (
    AGVExtent,
    EnvironmentStructure,
    WaypointPath,
)

from ..config import LiDARParams, PerceptionParams
from ..types import AGVState, LiDARScan, PerceivedWorld, PriorMap

from .lidar import distort_scan, ideal_scan
from .prior_map import degrade_prior_map


def apply_perception(
    environment: EnvironmentStructure,
    obstacle_polygons_world: Iterable[np.ndarray],
    reference_path: WaypointPath,
    agv_extent: AGVExtent,
    agv_state: AGVState,
    prior_map: PriorMap,
    lidar: LiDARParams,
    sim_time: float,
    rng: np.random.Generator,
) -> PerceivedWorld:
    """
    Compute W_tilde_t = Psi(W_t) given the objective pieces of W_t.

    The prior map M_0 is computed once per run (outside this function)
    and passed in; it does not change over time.

    Parameters
    ----------
    environment            : E (for ray-vs-segment intersections)
    obstacle_polygons_world: (k_i, 2) polygons for each obstacle at time t
    reference_path         : pi
    agv_extent             : S_agv
    agv_state              : (P_t, theta_t, v_t)
    prior_map              : M_0 (pre-computed via degrade_prior_map)
    lidar                  : Lambda = <Lambda_S, Lambda_D>
    sim_time               : t (stored on the scan)
    rng                    : RNG for the distortion D
    """
    ideal = ideal_scan(
        environment=environment,
        obstacles_world_polygons=obstacle_polygons_world,
        position=agv_state.position,
        heading=agv_state.heading,
        geometry=lidar.geometry,
        sim_time=sim_time,
    )
    scan = distort_scan(
        scan=ideal,
        distortion=lidar.distortion,
        geometry=lidar.geometry,
        rng=rng,
    )
    return PerceivedWorld(
        prior_map=prior_map,
        reference_path=reference_path,
        agv_extent=agv_extent,
        lidar_scan=scan,
        agv_state=agv_state,
    )


__all__ = ["apply_perception"]

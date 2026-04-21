"""
Structured per-run report emission.

A run report is a single JSON file with a fixed schema containing every
piece of metadata and metrics required for downstream analysis of a
simulation run.

The schema is defined in ``build_report`` and is deliberately rigid:
every field must always be present, missing values are explicit ``null``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from typing import Optional

from world_generator.models import WorldState

from ..config import SimulationConfig
from ..types import RunMetrics


def compute_run_id(
    world_id: int,
    detection: str,
    fusion: str,
    avoidance: str,
    sigma_profile: str,
    seed: int,
) -> str:
    """
    Deterministic 16-hex run identifier derived from the input tuple.

    Same inputs always yield the same id, regardless of timing or host.
    """
    key = (
        f"{int(world_id)}|{detection}|{fusion}|{avoidance}|"
        f"{sigma_profile}|{int(seed)}"
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _extract_stratum(world: WorldState) -> dict:
    """
    Pull (d_bsp, n_static, n_dynamic) from the world's parameters snapshot.

    `WorldState.parameters` is the snapshot of Omega recorded at generation
    time; `bsp_depth`, `n_static`, `n_dynamic` are always present on worlds
    produced by `world_generator.generator`.
    """
    params = world.parameters or {}
    env = params.get("environment", {})
    obs = params.get("obstacles", {})
    return {
        "d_bsp": int(env["bsp_depth"]),
        "n_static": int(obs["n_static"]),
        "n_dynamic": int(obs["n_dynamic"]),
    }


def build_report(
    world: WorldState,
    world_path: str,
    sigma: SimulationConfig,
    sigma_config_path: str,
    detection: str,
    fusion: str,
    avoidance: str,
    seed: int,
    metrics: RunMetrics,
    wall_time_s: float,
    telemetry_path: Optional[str],
) -> dict:
    """
    Assemble the fixed-schema run-report dict.
    """
    run_id = compute_run_id(
        world_id=world.world_id,
        detection=detection,
        fusion=fusion,
        avoidance=avoidance,
        sigma_profile=sigma.name,
        seed=seed,
    )
    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    mu_vel = (
        float(metrics.mu_vel)
        if metrics.mu_vel is not None and _is_finite(metrics.mu_vel)
        else None
    )

    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "world": {
            "path": str(world_path),
            "id": int(world.world_id),
            "seed": int(world.seed),
            "stratum": _extract_stratum(world),
        },
        "pipeline": {
            "detection": detection,
            "fusion": fusion,
            "avoidance": avoidance,
        },
        "sigma": {
            "profile": sigma.name,
            "config_path": str(sigma_config_path),
            "parameters": sigma.to_dict(),
        },
        "seed": int(seed),
        "metrics": {
            "mu_col": int(metrics.mu_col),
            "mu_dev": float(metrics.mu_dev),
            "mu_goal": int(metrics.mu_goal),
            "mu_vel": mu_vel,
            "mu_comp": float(metrics.mu_comp),
            "mu_lat": float(metrics.mu_lat),
        },
        "execution": {
            "steps": int(metrics.steps),
            "duration": float(metrics.duration),
            "collision_step": (
                None if metrics.collision_step is None
                else int(metrics.collision_step)
            ),
            "goal_reach_step": (
                None if metrics.goal_reach_step is None
                else int(metrics.goal_reach_step)
            ),
            "wall_time_s": float(wall_time_s),
        },
        "telemetry_path": (None if telemetry_path is None else str(telemetry_path)),
    }


def write_report(report: dict, path: str) -> None:
    """Write the report to ``path``, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)


def _is_finite(x: float) -> bool:
    try:
        import math
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


__all__ = ["build_report", "compute_run_id", "write_report"]

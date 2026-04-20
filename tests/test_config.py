"""Tests for Omega parameter loading and validation."""

from __future__ import annotations

import pytest

from src.world_generator import (
    AGVParams,
    EnvironmentParams,
    ObstacleParams,
    Omega,
    PathParams,
)


def test_default_config_matches_table_i(default_omega: Omega) -> None:
    """config/default_world.yaml should reflect the base values."""
    env = default_omega.env
    assert env.X == 30.0 and env.Y == 30.0
    # d_bsp semantics: d_bsp = 0 is the empty-hall base case and
    # d_bsp in {1, 2, 3} is the stratified corpus; the default YAML
    # picks one of the corpus levels.
    assert env.bsp_depth in (1, 2, 3)
    assert env.min_partition == (6.0, 6.0)
    assert env.split_ratio == (0.3, 0.7)
    assert env.room_padding == (0.15, 0.35)
    assert env.min_corridor == 2.0
    assert env.passage_width == 4.0
    assert env.wall_thickness == 0.3

    agv = default_omega.agv
    assert agv.L_agv == 0.6 and agv.W_agv == 0.4

    path = default_omega.path
    assert path.prm_samples == 500
    assert path.prm_neighbors == 10
    assert path.max_edge_length == 5.0
    assert path.clearance == 0.5
    assert path.min_path_length == 15.0
    # v* = 1 m/s (constant).
    assert path.speed_min == path.speed_max == 1.0

    obs = default_omega.obs
    assert obs.radius_min == 0.1 and obs.radius_max == 0.4
    assert obs.min_clearance == 0.3
    assert obs.dyn_speed_min == 0.2 and obs.dyn_speed_max == 0.8


def test_env_params_rejects_passage_not_exceeding_wall() -> None:
    with pytest.raises(ValueError):
        EnvironmentParams(
            X=10, Y=10, bsp_depth=2,
            min_partition=(2.0, 2.0), split_ratio=(0.4, 0.6),
            room_padding=(0.1, 0.3), min_corridor=1.0,
            passage_width=0.05, wall_thickness=0.1,
        )


def test_env_params_rejects_bad_split_ratio() -> None:
    with pytest.raises(ValueError):
        EnvironmentParams(
            X=10, Y=10, bsp_depth=2, min_partition=(2.0, 2.0),
            split_ratio=(0.7, 0.3),  # rho_min > rho_max
            room_padding=(0.1, 0.3), min_corridor=1.0,
            passage_width=2.0, wall_thickness=0.1,
        )


def test_agv_params_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        AGVParams(L_agv=0.0, W_agv=0.4)


def test_path_params_rejects_inverted_speeds() -> None:
    with pytest.raises(ValueError):
        PathParams(
            prm_samples=500, prm_neighbors=10, max_edge_length=5.0,
            clearance=0.5, min_path_length=15.0,
            speed_min=2.0, speed_max=1.0,
        )


def test_path_params_rejects_negative_min_path_length() -> None:
    with pytest.raises(ValueError):
        PathParams(
            prm_samples=500, prm_neighbors=10, max_edge_length=5.0,
            clearance=0.5, min_path_length=-1.0,
            speed_min=1.0, speed_max=1.0,
        )


def test_obstacle_params_rejects_bad_shape_weights() -> None:
    with pytest.raises(ValueError):
        ObstacleParams(
            n_static=4, n_dynamic=2, radius_min=0.1, radius_max=0.4,
            shape_weights={"triangle": 1.0},  # not in {circle, square}
            min_clearance=0.3, path_clearance=0.3,
            prm_samples=300, prm_neighbors=10, max_edge_length=5.0,
            trajectory_clearance=0.5,
            dyn_speed_min=0.2, dyn_speed_max=0.8,
        )

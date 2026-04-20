"""
Tests for obstacle generation.

Run against a simple 20x20 m empty region so all clearance checks reflect
the placement logic in src.world_generator.obstacles, not BSP geometry.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, box

from src.world_generator import ObstacleShape, WaypointPath
from src.world_generator.config import ObstacleParams
from src.world_generator.obstacles import generate_obstacles, obstacle_footprint


def _simple_params(n_static: int = 4, n_dynamic: int = 2) -> ObstacleParams:
    return ObstacleParams(
        n_static=n_static, n_dynamic=n_dynamic,
        radius_min=0.1, radius_max=0.4,
        shape_weights={"circle": 1.0, "square": 1.0},
        min_clearance=0.3, path_clearance=0.3,
        prm_samples=150, prm_neighbors=8, max_edge_length=5.0,
        trajectory_clearance=0.3,
        dyn_speed_min=0.2, dyn_speed_max=0.8,
        max_place_attempts=200,
    )


@pytest.fixture
def open_region():
    return box(0.0, 0.0, 20.0, 20.0)


@pytest.fixture
def simple_path():
    return WaypointPath(
        positions=np.array([[2.0, 10.0], [18.0, 10.0]], dtype=float),
        speeds=np.array([1.0]),
    )


def test_obstacle_footprint_circle_radius() -> None:
    fp = obstacle_footprint(ObstacleShape.CIRCLE, 0.25, np.array([5.0, 5.0]))
    # Shapely approximates circles with polygons; tolerate the discretisation.
    assert fp.area == pytest.approx(np.pi * 0.25**2, rel=5e-3)


def test_obstacle_footprint_square_side_is_twice_radius() -> None:
    fp = obstacle_footprint(ObstacleShape.SQUARE, 0.3, np.array([5.0, 5.0]))
    assert fp.area == pytest.approx((0.6) ** 2)


def test_generate_obstacles_produces_correct_counts(open_region, simple_path) -> None:
    rng = np.random.default_rng(0)
    params = _simple_params(n_static=5, n_dynamic=3)
    obstacles, _retries = generate_obstacles(open_region, simple_path, params, rng)
    assert len(obstacles) == 8
    assert sum(1 for o in obstacles if o.is_dynamic) == 3
    assert sum(1 for o in obstacles if not o.is_dynamic) == 5


def test_generate_obstacles_respects_min_clearance(open_region, simple_path) -> None:
    rng = np.random.default_rng(1)
    params = _simple_params(n_static=6, n_dynamic=0)
    obstacles, _ = generate_obstacles(open_region, simple_path, params, rng)
    footprints = [
        obstacle_footprint(o.spatial.shape, o.spatial.inscribed_radius,
                           o.initial_centroid())
        for o in obstacles
    ]
    for i in range(len(footprints)):
        for j in range(i + 1, len(footprints)):
            assert footprints[i].distance(footprints[j]) >= params.min_clearance - 1e-9


def test_generate_obstacles_respects_path_clearance(open_region, simple_path) -> None:
    rng = np.random.default_rng(2)
    params = _simple_params(n_static=4, n_dynamic=2)
    obstacles, _ = generate_obstacles(open_region, simple_path, params, rng)
    path_geom = LineString([tuple(p) for p in simple_path.positions])
    for o in obstacles:
        fp = obstacle_footprint(
            o.spatial.shape, o.spatial.inscribed_radius, o.initial_centroid()
        )
        assert path_geom.distance(fp) >= params.path_clearance - 1e-9


def test_generate_obstacles_zero_counts(open_region, simple_path) -> None:
    rng = np.random.default_rng(3)
    params = _simple_params(n_static=0, n_dynamic=0)
    obstacles, retries = generate_obstacles(open_region, simple_path, params, rng)
    assert obstacles == []
    assert retries == 0


def test_generate_obstacles_reproducible(open_region, simple_path) -> None:
    params = _simple_params(n_static=3, n_dynamic=2)
    obs_a, _ = generate_obstacles(
        open_region, simple_path, params, np.random.default_rng(99),
    )
    obs_b, _ = generate_obstacles(
        open_region, simple_path, params, np.random.default_rng(99),
    )
    assert len(obs_a) == len(obs_b)
    for a, b in zip(obs_a, obs_b):
        np.testing.assert_allclose(a.initial_centroid(), b.initial_centroid())
        assert a.spatial.shape == b.spatial.shape

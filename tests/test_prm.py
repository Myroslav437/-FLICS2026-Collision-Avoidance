"""
Tests for PRM planning.

These tests operate on a simple rectangular free-space region (no walls)
so roadmap connectivity is guaranteed and failures point to the planner
itself rather than to the BSP environment.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box
from shapely.prepared import prep

from src.world_generator.prm import (
    PRMGenerationError,
    build_roadmap,
    inflate_free_space,
    plan_random_path,
    plan_reference_path,
    plan_waypoint_path,
)


@pytest.fixture
def open_region():
    # 20 x 20 m empty box -- plenty of room for a k-NN roadmap.
    return box(0.0, 0.0, 20.0, 20.0)


def test_inflate_free_space_shrinks_region(open_region) -> None:
    shrunk = inflate_free_space(open_region, clearance=1.0)
    assert shrunk.area < open_region.area


def test_inflate_free_space_rejects_negative_clearance(open_region) -> None:
    with pytest.raises(ValueError):
        inflate_free_space(open_region, clearance=-0.1)


def test_build_roadmap_edges_are_collision_free(open_region) -> None:
    rng = np.random.default_rng(0)
    graph, points = build_roadmap(
        open_region, n_samples=50, k_nn=5, l_max=5.0, rng=rng,
    )
    prepared = prep(open_region)
    from shapely.geometry import LineString
    for u, v in graph.edges():
        seg = LineString([tuple(points[u]), tuple(points[v])])
        assert prepared.contains(seg)


def test_plan_waypoint_path_connects_start_and_goal(open_region) -> None:
    rng = np.random.default_rng(0)
    start = np.array([1.0, 1.0])
    goal = np.array([18.0, 18.0])
    path = plan_waypoint_path(
        open_region, start, goal,
        n_samples=200, k_nn=10, l_max=5.0,
        speed_min=1.0, speed_max=1.0, rng=rng,
    )
    assert path.num_segments >= 1
    np.testing.assert_allclose(path.positions[0], start, atol=1e-9)
    np.testing.assert_allclose(path.positions[-1], goal, atol=1e-9)
    # Constant speed -> every segment gets 1.0.
    np.testing.assert_allclose(path.speeds, 1.0)


def test_plan_random_path_respects_min_separation(open_region) -> None:
    rng = np.random.default_rng(7)
    path = plan_random_path(
        open_region, n_samples=200, k_nn=10, l_max=5.0,
        speed_min=1.0, speed_max=1.0, min_separation=3.0, rng=rng,
    )
    assert path.num_segments >= 1
    d = np.linalg.norm(path.positions[-1] - path.positions[0])
    assert d >= 3.0


def test_plan_random_path_is_reproducible(open_region) -> None:
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    pa = plan_random_path(
        open_region, n_samples=150, k_nn=8, l_max=5.0,
        speed_min=1.0, speed_max=1.0, min_separation=3.0, rng=rng_a,
    )
    pb = plan_random_path(
        open_region, n_samples=150, k_nn=8, l_max=5.0,
        speed_min=1.0, speed_max=1.0, min_separation=3.0, rng=rng_b,
    )
    np.testing.assert_allclose(pa.positions, pb.positions)
    np.testing.assert_allclose(pa.speeds, pb.speeds)


def test_plan_reference_path_enforces_min_path_length(open_region) -> None:
    """
    l_pi_min rejection loop: every accepted path must have total length
    >= l_pi_min within floating-point tolerance.

    On an open 20x20 m box the roadmap connects ~all sampled pairs; a
    high l_pi_min forces the rejection loop to discard the many short
    straight-line pairs and only accept start-goal combinations whose
    shortest path clears the threshold.
    """
    min_path_length = 18.0
    tol = 1e-9
    for seed in range(5):
        rng = np.random.default_rng(seed)
        path, attempts = plan_reference_path(
            open_region,
            n_samples=200, k_nn=10, l_max=5.0,
            speed_min=1.0, speed_max=1.0,
            min_path_length=min_path_length, rng=rng,
        )
        length = float(np.sum(
            np.linalg.norm(np.diff(path.positions, axis=0), axis=1)
        ))
        assert length + tol >= min_path_length, (
            f"seed={seed}: path length {length:.4f} < l_pi_min={min_path_length}"
        )
        assert attempts >= 1


def test_plan_reference_path_returns_attempt_count(open_region) -> None:
    """min_path_length=0 should accept the first sample (attempts == 1)."""
    rng = np.random.default_rng(0)
    _, attempts = plan_reference_path(
        open_region, n_samples=200, k_nn=10, l_max=5.0,
        speed_min=1.0, speed_max=1.0, min_path_length=0.0, rng=rng,
    )
    assert attempts == 1


def test_plan_reference_path_infeasible_raises() -> None:
    """
    If l_pi_min is larger than the domain can support, the rejection
    loop must raise PRMGenerationError rather than silently falling
    back to a shorter path.
    """
    small = box(0.0, 0.0, 5.0, 5.0)
    rng = np.random.default_rng(0)
    with pytest.raises(PRMGenerationError):
        plan_reference_path(
            small, n_samples=100, k_nn=8, l_max=5.0,
            speed_min=1.0, speed_max=1.0,
            min_path_length=1000.0, rng=rng,
        )


def test_plan_waypoint_path_errors_on_empty_region() -> None:
    rng = np.random.default_rng(0)
    # Inflating a small box by a large clearance empties it.
    tiny = box(0.0, 0.0, 0.5, 0.5)
    empty = inflate_free_space(tiny, clearance=1.0)
    with pytest.raises(PRMGenerationError):
        plan_waypoint_path(
            empty, np.array([0.1, 0.1]), np.array([0.4, 0.4]),
            n_samples=50, k_nn=5, l_max=5.0,
            speed_min=1.0, speed_max=1.0, rng=rng,
        )

"""
End-to-end tests for WG(Omega, s_wg) -> W_0 and the navigability-safety
metrics.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace

import numpy as np

from src.world_generator import Omega, WorldState, generate_world
from src.world_generator.metrics import compute_metrics


def test_generate_world_reproducible(default_omega: Omega) -> None:
    """Same seed + same Omega -> identical W_0."""
    w1 = generate_world(default_omega, seed=7, world_id=0)
    w2 = generate_world(default_omega, seed=7, world_id=0)
    np.testing.assert_allclose(w1.agv_position, w2.agv_position)
    assert w1.agv_heading == w2.agv_heading
    np.testing.assert_allclose(w1.reference_path.positions,
                               w2.reference_path.positions)
    assert len(w1.obstacles) == len(w2.obstacles)
    for o1, o2 in zip(w1.obstacles, w2.obstacles):
        np.testing.assert_allclose(o1.initial_centroid(), o2.initial_centroid())


def test_generate_world_produces_expected_counts(default_omega: Omega) -> None:
    world = generate_world(default_omega, seed=0, world_id=0)
    assert len(world.static_obstacles) == default_omega.obs.n_static
    assert len(world.dynamic_obstacles) == default_omega.obs.n_dynamic


def test_agv_initial_pose_matches_reference_path(default_omega: Omega) -> None:
    """P_0 = first waypoint of pi, theta_0 = heading of pi's first segment."""
    world = generate_world(default_omega, seed=1, world_id=0)
    np.testing.assert_allclose(world.agv_position, world.reference_path.positions[0])
    assert world.agv_heading == world.reference_path.heading(0)
    np.testing.assert_allclose(world.agv_velocity, np.zeros(2))


def test_world_roundtrip_json(default_omega: Omega) -> None:
    world = generate_world(default_omega, seed=2, world_id=0)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "w.json")
        world.save(path)
        loaded = WorldState.load(path)
    np.testing.assert_allclose(loaded.agv_position, world.agv_position)
    np.testing.assert_allclose(
        loaded.reference_path.positions, world.reference_path.positions
    )
    assert len(loaded.obstacles) == len(world.obstacles)


def test_metrics_navigability_safety(default_omega: Omega) -> None:
    """
    Navigability-safety condition:
      min_passage_width(pi) > W_agv + 2 * d_clear.

    Uses the default Omega: w_pass=4.0 m and t_wall=0.3 m give a target
    passage width of ~3.7 m, comfortably above the navigability-safety
    threshold W_agv + 2*d_clear = 1.4 m.
    """
    world = generate_world(default_omega, seed=3, world_id=0)
    metrics = compute_metrics(world)
    threshold = default_omega.agv.W_agv + 2.0 * default_omega.path.clearance
    assert metrics.min_passage_width > threshold, (
        f"min_passage_width={metrics.min_passage_width:.3f} "
        f"violates navigability-safety threshold {threshold:.3f}"
    )


def test_metrics_fields_populated(default_omega: Omega) -> None:
    world = generate_world(default_omega, seed=4, world_id=0)
    m = compute_metrics(world)
    assert 0.0 < m.free_space_ratio < 1.0
    assert m.min_passage_width > 0.0
    assert m.static_obstacle_count == default_omega.obs.n_static
    assert m.dynamic_obstacle_count == default_omega.obs.n_dynamic
    assert m.path_length > 0.0


def test_different_seeds_yield_different_worlds(default_omega: Omega) -> None:
    w1 = generate_world(default_omega, seed=10, world_id=0)
    w2 = generate_world(default_omega, seed=11, world_id=1)
    # With high probability, distinct seeds produce distinct first waypoints.
    d = np.linalg.norm(w1.agv_position - w2.agv_position)
    assert d > 1e-6


def test_reference_path_respects_l_pi_min_across_seeds(default_omega: Omega) -> None:
    """
    l_pi_min enforcement: under default Omega, every generated reference
    path must have total length >= l_pi_min within floating-point
    tolerance. Uses d_bsp=1 (the stratum where naive sampling is most
    prone to trivial intra-room hops) and iterates over many seeds to
    give the rejection loop a chance to exercise.
    """
    env = replace(default_omega.env, bsp_depth=1)
    omega = replace(default_omega, env=env)
    l_pi_min = omega.path.min_path_length
    tol = 1e-9
    for seed in range(20):
        world = generate_world(omega, seed=seed, world_id=seed)
        length = world.reference_path.total_length()
        assert length + tol >= l_pi_min, (
            f"seed={seed}: reference-path length {length:.4f} "
            f"violates l_pi_min={l_pi_min:.4f}"
        )

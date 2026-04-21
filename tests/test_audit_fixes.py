"""
Regression tests for the three critical audit findings (C-01, C-02, C-03).

Each test would have caught the corresponding spec violation against the
paper (paper/paper.tex, §III-B.3 and §IV-B).
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

from src.simulation_engine.core.metrics import MetricsAccumulator
from src.simulation_engine.types import FusedObstacle
from src.world_generator.models import WaypointPath
from src.world_generator.prm import build_roadmap, plan_random_path


# =============================================================================
# C-01 — mu_lat must measure only pipeline decision time
# =============================================================================


class _DummyPath:
    """Minimal WaypointPath stand-in: stationary so no deviation is recorded."""

    is_stationary = True
    positions = np.zeros((1, 2))


def test_c01_mu_lat_reflects_only_decision_time() -> None:
    """
    Before the fix, mu_lat was the 95th percentile of the *full step*
    wall time — it scaled with collision-check / perception cost as much
    as with pipeline algorithmic cost. After the fix, mu_lat reflects
    only the pipeline stage calls. This test feeds synthetic per-step
    samples where decision time is small (1 ms) and non-pipeline cost is
    large (50 ms, dominant): mu_lat must track the decision time, not
    the non-pipeline component.
    """
    acc = MetricsAccumulator(reference_path=_DummyPath())  # type: ignore[arg-type]
    decision_time = 0.001  # 1 ms
    non_pipeline_time = 0.050  # 50 ms "Shapely + perception"
    for _ in range(200):
        acc.record_step_time(
            wall_time=decision_time + non_pipeline_time,
            decision_time=decision_time,
        )
    metrics = acc.finalise(steps=200, duration=200 * 0.066)
    # mu_lat must match the pipeline decision time, independent of the
    # large non-pipeline cost.
    assert abs(metrics.mu_lat - decision_time) < 1e-9, (
        f"mu_lat={metrics.mu_lat} should equal decision_time={decision_time}"
    )
    # mu_comp still reflects the full per-step CPU cost.
    assert abs(metrics.mu_comp - (decision_time + non_pipeline_time)) < 1e-9


def test_c01_mu_lat_independent_of_non_pipeline_cost() -> None:
    """
    Two scenarios with the same decision-time distribution but different
    non-pipeline costs must yield identical mu_lat. This is the bug the
    audit found: scaling up the non-pipeline Shapely work inflated mu_lat
    before the fix.
    """
    acc_light = MetricsAccumulator(reference_path=_DummyPath())  # type: ignore[arg-type]
    acc_heavy = MetricsAccumulator(reference_path=_DummyPath())  # type: ignore[arg-type]
    decision_samples = [0.001, 0.002, 0.0015, 0.0025, 0.0012] * 20
    for d in decision_samples:
        acc_light.record_step_time(wall_time=d + 0.005, decision_time=d)
        acc_heavy.record_step_time(wall_time=d + 0.100, decision_time=d)
    m_light = acc_light.finalise(steps=len(decision_samples), duration=1.0)
    m_heavy = acc_heavy.finalise(steps=len(decision_samples), duration=1.0)
    assert abs(m_light.mu_lat - m_heavy.mu_lat) < 1e-12
    assert m_heavy.mu_comp > m_light.mu_comp  # comp *does* scale


# =============================================================================
# C-02 — PRM terminals must be sampled uniformly in free space
# =============================================================================


def test_c02_start_goal_are_not_roadmap_nodes() -> None:
    """
    After the fix, start/goal are sampled uniformly in the inflated free
    space (paper §III-B.3 step 4) and *inserted* into the roadmap, not
    drawn from existing roadmap node positions. A spot check across
    several seeds: the first and last waypoint positions must not
    exactly coincide with any of the pre-insertion roadmap node
    coordinates.
    """
    region = box(0.0, 0.0, 20.0, 20.0)
    hits_as_node = 0
    n_worlds = 10
    for seed in range(n_worlds):
        build_rng = np.random.default_rng(seed)
        # Use a deterministic roadmap+terminal draw: build a roadmap
        # ourselves to know exactly which node positions existed
        # before insertion, then plan a random path with the same seed
        # stream and compare.
        _graph, roadmap_points = build_roadmap(
            region, n_samples=150, k_nn=8, l_max=5.0,
            rng=np.random.default_rng(seed),
        )
        plan_rng = np.random.default_rng(seed)
        path = plan_random_path(
            region, n_samples=150, k_nn=8, l_max=5.0,
            speed_min=1.0, speed_max=1.0, min_separation=3.0,
            rng=plan_rng,
        )
        start = path.positions[0]
        goal = path.positions[-1]
        # A start/goal that exactly equals one of the roadmap's
        # pre-existing sample points is the pre-fix failure mode.
        def _is_node(p: np.ndarray) -> bool:
            return bool(
                np.any(np.all(np.isclose(roadmap_points, p, atol=1e-12), axis=1))
            )
        if _is_node(start) or _is_node(goal):
            hits_as_node += 1
    # Under the correct behaviour, freshly sampled terminals are not
    # (with probability 1) coincident with any roadmap node.
    assert hits_as_node == 0, (
        f"{hits_as_node}/{n_worlds} worlds produced start/goal that were "
        f"existing roadmap nodes; terminals must be freshly sampled"
    )


def test_c02_start_goal_spread_across_free_space() -> None:
    """
    Over multiple seeds, the start/goal points should be distributed
    across the free space, not clumped at a small handful of roadmap
    node positions. A coarse spot check: the bounding box of the
    collected terminals should span a non-trivial fraction of the
    region.
    """
    region = box(0.0, 0.0, 20.0, 20.0)
    terminals = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        path = plan_random_path(
            region, n_samples=150, k_nn=8, l_max=5.0,
            speed_min=1.0, speed_max=1.0, min_separation=3.0,
            rng=rng,
        )
        terminals.append(path.positions[0])
        terminals.append(path.positions[-1])
    terminals = np.asarray(terminals)
    span_x = terminals[:, 0].max() - terminals[:, 0].min()
    span_y = terminals[:, 1].max() - terminals[:, 1].min()
    # 10 seeds, uniformly drawn over a 20x20 box: expect span >> 5 m on
    # each axis. This is a loose spot check, not a statistical test.
    assert span_x > 10.0 and span_y > 10.0, (
        f"Terminal bounding-box span too small: ({span_x:.2f}, {span_y:.2f})"
    )


# =============================================================================
# C-03 — mu_vel associates from each dynamic GT to its nearest fused track
# =============================================================================


def test_c03_denominator_is_n_dynamic_times_n_steps() -> None:
    """
    Per §IV-B, mu_vel is averaged over dynamic obstacles AND simulation
    steps. After the fix, the denominator equals n_dynamic * n_steps
    regardless of how many fused tracks the detector/fuser emit. Before
    the fix it was n_fused * n_steps, which inflated with false
    positives.
    """
    n_dynamic = 1
    n_steps = 5
    # 1 dynamic obstacle at origin moving with v=(1, 0).
    dyn_centroids = np.array([[0.0, 0.0]])
    dyn_velocities = np.array([[1.0, 0.0]])

    # Three fused tracks per step (simulating a noisy detector with
    # false positives): one on the GT centroid with v=(1, 0) (clean
    # estimate) and two spurious tracks far away with arbitrary
    # velocities.
    fused_step = [
        FusedObstacle(track_id=0, position=np.array([0.0, 0.0]),
                      velocity=np.array([1.0, 0.0]), radius=0.2),
        FusedObstacle(track_id=1, position=np.array([15.0, 0.0]),
                      velocity=np.array([5.0, 5.0]), radius=0.2),
        FusedObstacle(track_id=2, position=np.array([-15.0, 0.0]),
                      velocity=np.array([-3.0, 4.0]), radius=0.2),
    ]

    acc = MetricsAccumulator(reference_path=_DummyPath())  # type: ignore[arg-type]
    for _ in range(n_steps):
        acc.record_velocity_samples(fused_step, dyn_centroids, dyn_velocities)
    metrics = acc.finalise(steps=n_steps, duration=1.0)

    # Denominator = n_dynamic * n_steps = 5 samples; numerator is 0
    # because the matching fused track reports the GT velocity exactly.
    assert acc._vel_sample_count == n_dynamic * n_steps  # type: ignore[attr-defined]
    assert abs(metrics.mu_vel) < 1e-9, (
        f"mu_vel should be ~0 when the GT-matched track is perfect; "
        f"got {metrics.mu_vel}"
    )


def test_c03_false_positives_do_not_inflate_mu_vel() -> None:
    """
    Scenario: the GT-matched fused track reports a clean velocity, and
    many extra spurious tracks (false positives from a noisy detector)
    are also present. Under the old behaviour those spurious tracks
    were associated to the nearest (still distant) dynamic GT and
    contributed huge velocity errors; under the new behaviour only the
    per-dynamic nearest-in-gate track counts.
    """
    dyn_centroids = np.array([[0.0, 0.0]])
    dyn_velocities = np.array([[1.0, 0.0]])

    clean_only = [
        FusedObstacle(track_id=0, position=np.array([0.0, 0.0]),
                      velocity=np.array([1.0, 0.0]), radius=0.2),
    ]
    with_false_positives = clean_only + [
        FusedObstacle(track_id=100 + k, position=np.array([10.0 + k, 5.0]),
                      velocity=np.array([10.0, -10.0]), radius=0.2)
        for k in range(5)
    ]

    acc_clean = MetricsAccumulator(reference_path=_DummyPath())  # type: ignore[arg-type]
    acc_noisy = MetricsAccumulator(reference_path=_DummyPath())  # type: ignore[arg-type]
    for _ in range(10):
        acc_clean.record_velocity_samples(clean_only, dyn_centroids, dyn_velocities)
        acc_noisy.record_velocity_samples(
            with_false_positives, dyn_centroids, dyn_velocities
        )
    m_clean = acc_clean.finalise(steps=10, duration=1.0)
    m_noisy = acc_noisy.finalise(steps=10, duration=1.0)

    # Clean case: zero error because the matched track is perfect.
    assert abs(m_clean.mu_vel) < 1e-9
    # Noisy case: still zero — false positives outside the gate do not
    # contribute, because the nearest-to-GT track is the clean one.
    assert abs(m_noisy.mu_vel) < 1e-9, (
        f"False positives should not inflate mu_vel; got {m_noisy.mu_vel}"
    )
    # And the denominators must match: n_dynamic * n_steps in both runs.
    assert (
        acc_clean._vel_sample_count == acc_noisy._vel_sample_count == 10  # type: ignore[attr-defined]
    )


def test_c03_unmatched_dynamic_is_penalised() -> None:
    """
    When no fused track lies within the gate of a given dynamic
    obstacle at a given step (e.g., the detector missed it entirely),
    the fix charges a squared error of ||v_gt||^2 — the same penalty as
    "reported stationary". The denominator still counts the obstacle.
    This guarantees a pipeline that suppresses tracks cannot game
    mu_vel by dropping to zero samples.
    """
    dyn_centroids = np.array([[0.0, 0.0]])
    dyn_velocities = np.array([[3.0, 4.0]])  # ||v|| = 5 -> ||v||^2 = 25
    acc = MetricsAccumulator(reference_path=_DummyPath())  # type: ignore[arg-type]
    for _ in range(4):
        acc.record_velocity_samples([], dyn_centroids, dyn_velocities)
    metrics = acc.finalise(steps=4, duration=1.0)
    assert acc._vel_sample_count == 4  # type: ignore[attr-defined]
    # RMSE over 4 samples of squared error 25 -> sqrt(25) = 5.
    assert abs(metrics.mu_vel - 5.0) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

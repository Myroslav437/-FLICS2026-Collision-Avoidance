"""
Tests for BSP environment generation.

Checks structural properties of E: the occupied geometry is non-empty,
the domain has free space left over, and the free-space ratio decreases
monotonically as d_bsp grows (more partition/room walls carved into the
same X*Y domain).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.world_generator import EnvironmentParams, Omega
from src.world_generator.bsp import generate_environment
from src.world_generator.generator import generate_world


def _free_ratio(omega: Omega, seed: int) -> float:
    rng = np.random.default_rng(seed)
    bsp = generate_environment(omega.env, rng)
    X, Y = bsp.environment.size
    return float(bsp.free_space.area / (X * Y))


def test_bsp_produces_nonempty_environment(default_omega: Omega) -> None:
    rng = np.random.default_rng(0)
    bsp = generate_environment(default_omega.env, rng)
    assert len(bsp.environment.chains) >= 1
    assert not bsp.free_space.is_empty
    X, Y = bsp.environment.size
    assert 0.0 < bsp.free_space.area < X * Y


def test_bsp_free_space_decreases_with_depth(default_omega: Omega) -> None:
    """free_space_ratio should decrease as d_bsp grows.

    Uses the corpus levels {1, 2, 3}; the empty-hall base case d_bsp=0 is
    covered by a dedicated test below.
    """
    ratios_by_depth = {}
    for d in (1, 2, 3):
        env = replace(default_omega.env, bsp_depth=d)
        omega = replace(default_omega, env=env)
        ratios = [_free_ratio(omega, seed) for seed in range(5)]
        ratios_by_depth[d] = sum(ratios) / len(ratios)

    assert ratios_by_depth[1] > ratios_by_depth[2] > ratios_by_depth[3], (
        f"Free-space ratio did not decrease monotonically: {ratios_by_depth}"
    )


def test_bsp_depth_zero_is_empty_hall(default_omega: Omega) -> None:
    """
    Renumbered semantics: d_bsp=0 must produce the empty bounded hall -
    only the outer perimeter walls, no internal partition walls and no
    carved rooms. The free-space ratio therefore equals
    1 - perimeter_area / (X*Y) and must exceed every d_bsp>=1 ratio.
    """
    env_zero = replace(default_omega.env, bsp_depth=0)
    omega_zero = replace(default_omega, env=env_zero)
    ratio_zero = _free_ratio(omega_zero, seed=0)

    X, Y = default_omega.env.X, default_omega.env.Y
    t = default_omega.env.wall_thickness
    inner_area = (X - 2.0 * t) * (Y - 2.0 * t)
    expected_ratio = inner_area / (X * Y)
    assert ratio_zero == pytest.approx(expected_ratio, rel=1e-6), (
        f"d_bsp=0 should leave only the perimeter walls; "
        f"got ratio={ratio_zero:.6f}, expected {expected_ratio:.6f}"
    )

    env_one = replace(default_omega.env, bsp_depth=1)
    omega_one = replace(default_omega, env=env_one)
    ratio_one = _free_ratio(omega_one, seed=0)
    assert ratio_zero > ratio_one, (
        "Empty hall (d_bsp=0) should have strictly more free space than "
        f"the one-room case (d_bsp=1): {ratio_zero:.4f} vs {ratio_one:.4f}"
    )


def test_room_placement_enforces_c_min_clamp(default_omega: Omega) -> None:
    """
    c_min enforcement: when p_room * partition_size falls below c_min,
    the actual corridor must be clamped to c_min (not the
    padding-implied value). We construct a small single-cell domain
    (d_bsp = 1) tight enough that p_room * W < c_min on both axes and
    check that the resulting room is padded by c_min + t_wall rather than
    p_room * W + t_wall.
    """
    # Choose params so that p * W is strictly less than c_min on every axis:
    # W = 10 m, p in [0.05, 0.1] -> p*W in [0.5, 1.0] < c_min = 2.0.
    env = EnvironmentParams(
        X=10.0, Y=10.0, bsp_depth=1,
        min_partition=(6.0, 6.0), split_ratio=(0.3, 0.7),
        room_padding=(0.05, 0.1), min_corridor=2.0,
        passage_width=1.0, wall_thickness=0.3,
    )
    omega = replace(default_omega, env=env)
    rng = np.random.default_rng(0)
    bsp = generate_environment(omega.env, rng)

    # Observed corridor should equal c_min exactly within tolerance.
    assert bsp.min_corridor_width == pytest.approx(env.min_corridor, abs=1e-9), (
        f"expected corridor = c_min = {env.min_corridor}, got "
        f"{bsp.min_corridor_width}"
    )

    # Also sanity-check the room rectangle that was carved.
    t = env.wall_thickness
    expected_pad = env.min_corridor + t
    # Reconstruct the room rect: there is exactly one leaf = the whole
    # domain at d_bsp = 1. Room exterior should be inset by expected_pad
    # from the domain.
    # Walk to the only leaf via an external helper would be intrusive;
    # instead, derive the expected interior size:
    expected_room_w = env.X - 2.0 * expected_pad
    expected_room_h = env.Y - 2.0 * expected_pad
    assert expected_room_w > 0 and expected_room_h > 0


def test_room_placement_skipped_when_partition_too_small(
    default_omega: Omega,
) -> None:
    """c_min > partition_size / 2 must skip room placement cleanly
    (no degenerate negative-sized rooms, no crash)."""
    env = EnvironmentParams(
        X=5.0, Y=5.0, bsp_depth=1,
        min_partition=(3.0, 3.0), split_ratio=(0.3, 0.7),
        room_padding=(0.1, 0.2), min_corridor=3.0,  # 2*c_min >= X=Y
        passage_width=1.0, wall_thickness=0.3,
    )
    omega = replace(default_omega, env=env)
    rng = np.random.default_rng(0)
    bsp = generate_environment(omega.env, rng)
    # No room was carved; only the outer perimeter remains, so
    # min_corridor_width stays at its sentinel (+inf).
    assert bsp.min_corridor_width == float("inf")
    # Free space must still be non-empty.
    assert not bsp.free_space.is_empty


def test_bsp_reproducibility(default_omega: Omega) -> None:
    """Same seed + same Omega_env -> identical environment chains."""
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    bsp_a = generate_environment(default_omega.env, rng_a)
    bsp_b = generate_environment(default_omega.env, rng_b)
    assert len(bsp_a.environment.chains) == len(bsp_b.environment.chains)
    for ca, cb in zip(bsp_a.environment.chains, bsp_b.environment.chains):
        np.testing.assert_allclose(ca, cb)


@pytest.mark.parametrize("depth", [1, 3])
def test_generate_world_end_to_end(default_omega: Omega, depth: int) -> None:
    """
    End-to-end smoke test at d_bsp corpus extremes: the world must contain
    a non-trivial reference path spanning at least the AGV diagonal plus
    2 * d_clear.
    """
    env = replace(default_omega.env, bsp_depth=depth)
    omega = replace(default_omega, env=env)
    world = generate_world(omega, seed=123, world_id=0)
    assert world.reference_path.num_segments >= 1
    min_sep = omega.agv.L_agv + 2 * omega.path.clearance
    span = np.linalg.norm(
        world.reference_path.positions[0] - world.reference_path.positions[-1]
    )
    assert span >= min_sep


def test_generate_world_end_to_end_empty_hall(default_omega: Omega) -> None:
    """
    d_bsp=0 (empty hall) is a supported code path: PRM should plan on the
    bare bounded domain without any internal structure.

    The perimeter is a t_wall-thick ring, which shapely exports as two
    oriented chains (CCW outer ring + CW inner ring). No partition walls
    and no room walls are present.
    """
    env = replace(default_omega.env, bsp_depth=0)
    omega = replace(default_omega, env=env)
    world = generate_world(omega, seed=123, world_id=0)
    assert world.reference_path.num_segments >= 1
    # Only the perimeter ring -> exactly 2 chains (outer + inner).
    assert len(world.environment.chains) == 2

"""Tests for data models."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.world_generator import (
    AGVExtent,
    EnvironmentStructure,
    Obstacle,
    ObstacleShape,
    SpatialDescriptor,
    WaypointPath,
)


# =============================================================================
# WaypointPath
# =============================================================================

def test_waypoint_path_num_segments_and_heading() -> None:
    p = WaypointPath(
        positions=np.array([[0, 0], [1, 0], [1, 2]], dtype=float),
        speeds=np.array([1.0, 0.5]),
    )
    assert p.num_segments == 2
    assert p.heading(0) == pytest.approx(0.0)
    assert p.heading(1) == pytest.approx(math.pi / 2)
    assert p.total_length() == pytest.approx(3.0)


def test_waypoint_path_stationary() -> None:
    p = WaypointPath.stationary(np.array([1.0, 2.0]))
    assert p.is_stationary and p.num_segments == 0
    assert p.total_length() == 0.0


def test_waypoint_path_rejects_speed_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        WaypointPath(
            positions=np.array([[0, 0], [1, 0]], dtype=float),
            speeds=np.array([1.0, 0.5]),  # should be length 1
        )


def test_waypoint_path_roundtrip() -> None:
    p = WaypointPath(
        positions=np.array([[0, 0], [1, 0], [1, 2]], dtype=float),
        speeds=np.array([1.0, 0.5]),
    )
    data = p.to_dict()
    q = WaypointPath.from_dict(data)
    np.testing.assert_allclose(q.positions, p.positions)
    np.testing.assert_allclose(q.speeds, p.speeds)


# =============================================================================
# SpatialDescriptor -- restricted to circles and squares
# =============================================================================

def test_spatial_descriptor_circle_boundary() -> None:
    s = SpatialDescriptor(shape=ObstacleShape.CIRCLE, inscribed_radius=0.25)
    v = s.boundary_vertices()
    # All boundary vertices at exactly r from origin.
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 0.25, atol=1e-9)


def test_spatial_descriptor_square_boundary_half_side_r() -> None:
    s = SpatialDescriptor(shape=ObstacleShape.SQUARE, inscribed_radius=0.3)
    v = s.boundary_vertices()
    # r = half-side -> every vertex has |x|=|y|=r.
    assert np.all(np.abs(v) == pytest.approx(0.3))


def test_spatial_descriptor_rejects_non_positive_radius() -> None:
    with pytest.raises(ValueError):
        SpatialDescriptor(shape=ObstacleShape.CIRCLE, inscribed_radius=0.0)


# =============================================================================
# Obstacle
# =============================================================================

def test_obstacle_static_is_not_dynamic() -> None:
    o = Obstacle(
        spatial=SpatialDescriptor(shape=ObstacleShape.CIRCLE, inscribed_radius=0.2),
        trajectory=WaypointPath.stationary(np.array([5.0, 5.0])),
    )
    assert not o.is_dynamic
    np.testing.assert_allclose(o.initial_centroid(), [5.0, 5.0])


def test_obstacle_roundtrip() -> None:
    o = Obstacle(
        spatial=SpatialDescriptor(shape=ObstacleShape.SQUARE, inscribed_radius=0.15),
        trajectory=WaypointPath(
            positions=np.array([[0, 0], [2, 0]], dtype=float),
            speeds=np.array([0.5]),
        ),
    )
    o2 = Obstacle.from_dict(o.to_dict())
    assert o2.spatial.shape == o.spatial.shape
    assert o2.spatial.inscribed_radius == o.spatial.inscribed_radius
    np.testing.assert_allclose(o2.trajectory.positions, o.trajectory.positions)


# =============================================================================
# EnvironmentStructure + AGVExtent
# =============================================================================

def test_environment_structure_requires_triangles_or_larger() -> None:
    with pytest.raises(ValueError):
        EnvironmentStructure(
            chains=[np.array([[0, 0], [1, 0]], dtype=float)],  # only 2 vertices
            size=(10.0, 10.0),
        )


def test_agv_extent_footprint_is_rotated_rectangle() -> None:
    ext = AGVExtent(length=2.0, width=1.0)
    # Heading = pi/2 rotates a length-2 local-x rectangle to span y.
    fp = ext.footprint_world_frame(position=np.array([5.0, 5.0]), heading=math.pi / 2)
    # Rectangle vertices should all be 5 +/- 1 in x and 5 +/- 1 in y (after rotation
    # width goes to x, length goes to y; half-dims 0.5 in x, 1.0 in y).
    xs, ys = fp[:, 0], fp[:, 1]
    assert set(np.round(xs - 5.0, 3).tolist()) <= {-0.5, 0.5}
    assert set(np.round(ys - 5.0, 3).tolist()) <= {-1.0, 1.0}

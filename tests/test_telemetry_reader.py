"""
Unit tests for the telemetry reader.

Tests parsing of synthetic JSONL fixtures, error handling for
malformed input, and null-range → np.inf conversion.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List

import numpy as np
import pytest

from src.simulation_visualizer.reader import (  # noqa: E402
    TelemetryError,
    TelemetryRun,
    read_telemetry,
)


# =========================================================================
# Fixtures
# =========================================================================

def _make_header() -> Dict[str, Any]:
    """Build a minimal valid header record."""
    return {
        "type": "header",
        "world_id": 42,
        "world_seed": 100,
        "simulation_seed": 200,
        "sigma": {
            "name": "nominal",
            "lidar": {
                "geometry": {
                    "r_min": 0.05,
                    "r_max": 10.0,
                    "theta_fov": 4.712,
                    "delta_theta": 0.00873,
                    "f_scan": 15.0,
                },
                "distortion": {
                    "sigma_r": 0.0,
                    "p_miss": 0.0,
                    "p_ghost": 0.0,
                },
            },
            "perception": {
                "eta_cpl": 1.0,
                "sigma_map": 0.0,
            },
            "dynamics": {
                "v_agv_max": 1.5,
                "a_agv_max": 1.0,
                "omega_agv_max": 1.0,
                "T_horizon": 120.0,
            },
        },
        "pipeline": {
            "detection": "EC",
            "fusion": "KF",
            "avoidance": "VFH",
        },
        "environment": {
            "size": [30.0, 30.0],
            "chains": [
                [[0, 0], [30, 0], [30, 30], [0, 30]],
            ],
        },
        "reference_path": {
            "positions": [[1, 1], [10, 10], [20, 20]],
            "speeds": [1.0, 1.0],
        },
        "agv_extent": {"length": 0.6, "width": 0.4},
        "agv_initial": {"position": [1, 1], "heading": 0.785},
        "obstacles_initial": [
            {
                "spatial": {
                    "shape": "circle",
                    "inscribed_radius": 0.3,
                    "circle_vertex_count": 32,
                },
                "trajectory": {
                    "positions": [[5, 5]],
                    "speeds": [],
                },
            }
        ],
    }


def _make_step(step: int = 0, time: float = 0.0667) -> Dict[str, Any]:
    """Build a minimal valid step record."""
    return {
        "type": "step",
        "step": step,
        "time": time,
        "agv": {
            "position": [2.0, 2.0],
            "heading": 0.785,
            "velocity": [0.5, 0.5],
        },
        "obstacle_centroids": [[5.0, 5.0]],
        "obstacle_velocities": [[0.0, 0.0]],
        "obstacle_radii": [0.3],
        "lidar": {
            "time": time,
            "pose_position": [2.0, 2.0],
            "pose_heading": 0.785,
            "angles": [-1.0, 0.0, 1.0],
            "ranges": [3.5, None, 8.0],
        },
        "detections": [
            {
                "centroid": [5.0, 5.0],
                "radius": 0.3,
                "points": [[4.8, 5.0], [5.2, 5.0]],
            }
        ],
        "fused": [
            {
                "track_id": 0,
                "position": [5.0, 5.0],
                "velocity": [0.0, 0.0],
                "radius": 0.3,
            }
        ],
        "control": {
            "linear_velocity": 1.0,
            "angular_velocity": 0.1,
        },
        "collided": False,
        "goal_reached": False,
        "step_wall_time": 0.001,
    }


def _write_jsonl(records: List[dict], path: str) -> None:
    """Write a list of dicts as JSON Lines to *path*."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# =========================================================================
# Tests
# =========================================================================

class TestReadTelemetry:
    """Tests for ``read_telemetry``."""

    def test_valid_parse(self, tmp_path):
        """Parsing a valid JSONL file returns a well-formed TelemetryRun."""
        path = str(tmp_path / "run.jsonl")
        _write_jsonl([_make_header(), _make_step(0), _make_step(1, 0.133)], path)

        run = read_telemetry(path)
        assert isinstance(run, TelemetryRun)
        assert run.header.world_id == 42
        assert run.header.sigma_name == "nominal"
        assert len(run.steps) == 2
        assert run.steps[0].step == 0
        assert run.steps[1].step == 1

    def test_header_fields(self, tmp_path):
        """Header fields are correctly parsed."""
        path = str(tmp_path / "run.jsonl")
        _write_jsonl([_make_header(), _make_step()], path)

        hdr = read_telemetry(path).header
        assert hdr.world_seed == 100
        assert hdr.simulation_seed == 200
        assert hdr.pipeline_detection == "EC"
        assert hdr.pipeline_fusion == "KF"
        assert hdr.pipeline_avoidance == "VFH"
        assert hdr.environment_size == (30.0, 30.0)
        assert len(hdr.environment_chains) == 1
        assert hdr.agv_length == 0.6
        assert hdr.agv_width == 0.4
        assert hdr.eta_cpl == 1.0
        assert hdr.sigma_map == 0.0
        assert len(hdr.obstacles) == 1
        assert hdr.obstacles[0].shape == "circle"
        assert not hdr.obstacles[0].is_dynamic

    def test_null_ranges_become_inf(self, tmp_path):
        """LiDAR ranges of null in JSON become np.inf."""
        path = str(tmp_path / "run.jsonl")
        _write_jsonl([_make_header(), _make_step()], path)

        step = read_telemetry(path).steps[0]
        assert np.isfinite(step.lidar.ranges[0])
        assert np.isinf(step.lidar.ranges[1])
        assert np.isfinite(step.lidar.ranges[2])

    def test_missing_header(self, tmp_path):
        """A file with no header raises TelemetryError."""
        path = str(tmp_path / "run.jsonl")
        _write_jsonl([_make_step()], path)

        with pytest.raises(TelemetryError, match="step record before header"):
            read_telemetry(path)

    def test_empty_file(self, tmp_path):
        """An empty file raises TelemetryError."""
        path = str(tmp_path / "run.jsonl")
        _write_jsonl([], path)

        with pytest.raises(TelemetryError, match="no header"):
            read_telemetry(path)

    def test_no_steps(self, tmp_path):
        """A file with header but no steps raises TelemetryError."""
        path = str(tmp_path / "run.jsonl")
        _write_jsonl([_make_header()], path)

        with pytest.raises(TelemetryError, match="no step"):
            read_telemetry(path)

    def test_malformed_step_missing_field(self, tmp_path):
        """A step missing a required field raises TelemetryError."""
        path = str(tmp_path / "run.jsonl")
        bad_step = _make_step()
        del bad_step["agv"]
        _write_jsonl([_make_header(), bad_step], path)

        with pytest.raises(TelemetryError, match="agv"):
            read_telemetry(path)

    def test_duplicate_header(self, tmp_path):
        """A file with two headers raises TelemetryError."""
        path = str(tmp_path / "run.jsonl")
        _write_jsonl([_make_header(), _make_header(), _make_step()], path)

        with pytest.raises(TelemetryError, match="duplicate header"):
            read_telemetry(path)

    def test_step_lidar_cartesian(self, tmp_path):
        """LiDAR cartesian_world projects finite rays correctly."""
        path = str(tmp_path / "run.jsonl")
        _write_jsonl([_make_header(), _make_step()], path)

        step = read_telemetry(path).steps[0]
        pts = step.lidar.cartesian_world()
        # Two finite rays → 2 points
        assert pts.shape == (2, 2)

    def test_empty_obstacles(self, tmp_path):
        """Handles runs with zero obstacles gracefully."""
        hdr = _make_header()
        hdr["obstacles_initial"] = []
        step = _make_step()
        step["obstacle_centroids"] = []
        step["obstacle_velocities"] = []
        step["obstacle_radii"] = []
        path = str(tmp_path / "run.jsonl")
        _write_jsonl([hdr, step], path)

        run = read_telemetry(path)
        assert len(run.header.obstacles) == 0
        assert run.steps[0].obstacle_centroids.shape == (0, 2)

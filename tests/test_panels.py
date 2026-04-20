"""
Unit tests for the four visualization panels.

Each panel is tested against canned input to verify that setup()
and update() complete without error and create the expected
matplotlib artists.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.simulation_visualizer.reader import read_telemetry  # noqa: E402
from src.simulation_visualizer.panels import (  # noqa: E402
    MetricsPanel,
    ObjectiveWorldPanel,
    PerceptionPanel,
    PipelineOutputPanel,
)


# =========================================================================
# Shared fixture helpers
# =========================================================================

def _make_header():
    return {
        "type": "header",
        "world_id": 1,
        "world_seed": 10,
        "simulation_seed": 20,
        "sigma": {
            "name": "nominal",
            "lidar": {
                "geometry": {
                    "r_min": 0.05, "r_max": 10.0,
                    "theta_fov": 4.712, "delta_theta": 0.00873,
                    "f_scan": 15.0,
                },
                "distortion": {
                    "sigma_r": 0.0, "p_miss": 0.0, "p_ghost": 0.0,
                },
            },
            "perception": {"eta_cpl": 1.0, "sigma_map": 0.0},
            "dynamics": {
                "v_agv_max": 1.5, "a_agv_max": 1.0,
                "omega_agv_max": 1.0, "T_horizon": 120.0,
            },
        },
        "pipeline": {"detection": "EC", "fusion": "PT", "avoidance": "DWA"},
        "environment": {
            "size": [20.0, 20.0],
            "chains": [
                [[0, 0], [20, 0], [20, 20], [0, 20]],
            ],
        },
        "reference_path": {
            "positions": [[1, 1], [10, 10], [19, 19]],
            "speeds": [1.0, 1.0],
        },
        "agv_extent": {"length": 0.6, "width": 0.4},
        "agv_initial": {"position": [1, 1], "heading": 0.785},
        "obstacles_initial": [
            {
                "spatial": {
                    "shape": "circle", "inscribed_radius": 0.25,
                    "circle_vertex_count": 32,
                },
                "trajectory": {
                    "positions": [[5, 5], [10, 10]],
                    "speeds": [0.5],
                },
            },
            {
                "spatial": {
                    "shape": "square", "inscribed_radius": 0.2,
                    "circle_vertex_count": 32,
                },
                "trajectory": {
                    "positions": [[15, 15]],
                    "speeds": [],
                },
            },
        ],
    }


def _make_step(step=0, time=0.0667):
    return {
        "type": "step",
        "step": step,
        "time": time,
        "agv": {
            "position": [2.0 + step * 0.5, 2.0 + step * 0.5],
            "heading": 0.785,
            "velocity": [0.5, 0.5],
        },
        "obstacle_centroids": [[6.0, 6.0], [15.0, 15.0]],
        "obstacle_velocities": [[0.3, 0.3], [0.0, 0.0]],
        "obstacle_radii": [0.25, 0.2],
        "lidar": {
            "time": time,
            "pose_position": [2.0 + step * 0.5, 2.0 + step * 0.5],
            "pose_heading": 0.785,
            "angles": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "ranges": [3.5, None, 8.0, 5.0, None],
        },
        "detections": [
            {"centroid": [6.0, 6.0], "radius": 0.3, "points": [[5.8, 6.0]]},
        ],
        "fused": [
            {
                "track_id": 0,
                "position": [6.0, 6.0],
                "velocity": [0.3, 0.3],
                "radius": 0.3,
            },
        ],
        "control": {"linear_velocity": 1.0, "angular_velocity": 0.05},
        "collided": False,
        "goal_reached": False,
        "step_wall_time": 0.002,
    }


@pytest.fixture
def telemetry_run(tmp_path):
    """Create a synthetic telemetry file and return the parsed run."""
    path = str(tmp_path / "test_run.jsonl")
    records = [_make_header()]
    for i in range(5):
        records.append(_make_step(step=i, time=0.0667 * (i + 1)))
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return read_telemetry(path)


# =========================================================================
# Panel tests
# =========================================================================

class TestObjectiveWorldPanel:
    """Tests for the ObjectiveWorldPanel."""

    def test_setup_and_update(self, telemetry_run):
        """Panel sets up and updates without error."""
        fig, ax = plt.subplots()
        panel = ObjectiveWorldPanel()
        panel.setup(ax, telemetry_run.header)
        for i, step in enumerate(telemetry_run.steps):
            panel.update(step, i, len(telemetry_run.steps))
        plt.close(fig)

    def test_creates_axes_limits(self, telemetry_run):
        """Panel sets the correct axis limits from environment size."""
        fig, ax = plt.subplots()
        panel = ObjectiveWorldPanel()
        panel.setup(ax, telemetry_run.header)
        assert ax.get_xlim() == (0.0, 20.0)
        assert ax.get_ylim() == (0.0, 20.0)
        plt.close(fig)


class TestPerceptionPanel:
    """Tests for the PerceptionPanel."""

    def test_setup_and_update(self, telemetry_run):
        """Panel sets up and updates without error."""
        fig, ax = plt.subplots()
        panel = PerceptionPanel()
        panel.setup(ax, telemetry_run.header)
        for i, step in enumerate(telemetry_run.steps):
            panel.update(step, i, len(telemetry_run.steps))
        plt.close(fig)


class TestPipelineOutputPanel:
    """Tests for the PipelineOutputPanel."""

    def test_setup_and_update(self, telemetry_run):
        """Panel sets up and updates without error."""
        fig, ax = plt.subplots()
        panel = PipelineOutputPanel()
        panel.setup(ax, telemetry_run.header)
        for i, step in enumerate(telemetry_run.steps):
            panel.update(step, i, len(telemetry_run.steps))
        plt.close(fig)


class TestMetricsPanel:
    """Tests for the MetricsPanel."""

    def test_setup_and_update(self, telemetry_run):
        """Panel sets up and updates without error."""
        fig, ax = plt.subplots()
        panel = MetricsPanel()
        panel.setup(ax, telemetry_run.header)
        for i, step in enumerate(telemetry_run.steps):
            panel.update(step, i, len(telemetry_run.steps))
        plt.close(fig)

    def test_collision_flag_tracked(self, telemetry_run):
        """Metrics panel tracks collision properly."""
        fig, ax = plt.subplots()
        panel = MetricsPanel()
        panel.setup(ax, telemetry_run.header)

        # First step: no collision
        panel.update(telemetry_run.steps[0], 0, len(telemetry_run.steps))
        assert not panel._collided

        plt.close(fig)

    def test_deviation_accumulation(self, telemetry_run):
        """Metrics panel correctly accumulates deviation samples."""
        fig, ax = plt.subplots()
        panel = MetricsPanel()
        panel.setup(ax, telemetry_run.header)

        for i, step in enumerate(telemetry_run.steps):
            panel.update(step, i, len(telemetry_run.steps))

        assert panel._dev_count == len(telemetry_run.steps)
        assert panel._dev_sum >= 0.0
        plt.close(fig)

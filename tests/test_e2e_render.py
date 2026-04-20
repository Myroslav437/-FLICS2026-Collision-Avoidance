"""
End-to-end test: run a short simulation, capture telemetry, and render.

Verifies that:
  1. A simulation produces a valid telemetry file
  2. The visualizer can read it back
  3. Rendering to a PNG directory produces non-empty files
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")

import pytest


@pytest.fixture
def telemetry_file(tmp_path):
    """
    Run a short simulation and produce a telemetry JSONL file.

    Uses a small world from the validation set to keep the test fast.
    """
    from src.world_generator import Omega, generate_world
    from src.simulation_engine import SimulationConfig, run_simulation
    from src.simulation_engine.core.telemetry import JsonlSink

    # Generate a simple world
    omega = Omega.load(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "default_world.yaml",
        )
    )
    world = generate_world(omega, world_id=0, seed=0)

    # Load Sigma YAML
    sigma_nominal = SimulationConfig.load(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "simulation_nominal.yaml",
        )
    )

    # Run simulation with telemetry
    telemetry_path = str(tmp_path / "test_run.jsonl")
    sink = JsonlSink(path=telemetry_path)

    run_simulation(
        world, sigma=sigma_nominal,
        detection="EC", fusion="PT", avoidance="VFH",
        seed=42,
        telemetry_sink=sink,
    )

    return telemetry_path


class TestE2ERender:
    """End-to-end rendering tests."""

    def test_telemetry_produced(self, telemetry_file):
        """The simulation produces a non-empty telemetry file."""
        assert os.path.isfile(telemetry_file)
        assert os.path.getsize(telemetry_file) > 0

    def test_reader_parses(self, telemetry_file):
        """The visualizer reader can parse the telemetry file."""
        from src.simulation_visualizer.reader import read_telemetry

        run = read_telemetry(telemetry_file)
        assert run.header.world_id == 0
        assert len(run.steps) > 0

    def test_render_to_png_directory(self, telemetry_file, tmp_path):
        """Rendering to a PNG directory produces non-empty frames."""
        from src.simulation_visualizer.reader import read_telemetry
        from src.simulation_visualizer.playback import animate

        run = read_telemetry(telemetry_file)

        # Only render a small number of frames for test speed
        # Truncate the steps to max 5 for faster testing
        if len(run.steps) > 5:
            run.steps = run.steps[:5]

        output_dir = str(tmp_path / "frames")
        animate(run, speed=1.0, render_to=output_dir, dpi=72)

        # Verify frames exist
        assert os.path.isdir(output_dir)
        pngs = [f for f in os.listdir(output_dir) if f.endswith(".png")]
        assert len(pngs) == len(run.steps)

        # Verify files are non-empty
        for png in pngs:
            fpath = os.path.join(output_dir, png)
            assert os.path.getsize(fpath) > 0

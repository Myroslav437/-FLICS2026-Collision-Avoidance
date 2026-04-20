"""
Simulation Visualizer.

Post-hoc playback tool that reads the Simulation Engine's JSONL
telemetry and renders four-panel animated visualizations:

  1. Objective world (W_t)        — §III-A.6
  2. Perception (W_tilde_t)       — §III-A.7
  3. Pipeline output              — A = <A_det, A_fus, A_avoid>
  4. Metrics                      — §IV-B

Usage::

    python -m simulation_visualizer play \\
        --telemetry path/to/run.jsonl \\
        --speed 1.0 \\
        --render-to output.mp4
"""

from .reader import TelemetryHeader, TelemetryStep, TelemetryRun, read_telemetry

__all__ = [
    "TelemetryHeader",
    "TelemetryStep",
    "TelemetryRun",
    "read_telemetry",
]

"""
Per-step telemetry emitter (fire-and-forget JSON Lines).

Design
------
The simulation core does not know how, whether, or where telemetry is
consumed. A `TelemetrySink` is a narrow interface with exactly one
method, `write(step: StepTelemetry)`. The core calls it once per step;
what happens after is the sink's responsibility.

The separation lets the visualizer run entirely out-of-process: it
reads the JSONL file the sink produced and needs no engine imports.

Two concrete sinks ship in-tree:
  - `NullSink`     : no-op; the default when telemetry is disabled.
  - `JsonlSink`    : one JSON object per line, compact encoding, no
                     pretty-printing. Numpy arrays are converted to
                     nested Python lists so the file is interpretable
                     without numpy.

Header
------
`JsonlSink` writes a single *header* JSON object on the first line
(before any step). The header carries the simulation configuration,
pipeline identifiers, seed, and world id. The visualizer uses the
header to render world-frame context (walls, path).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import numpy as np

from world_generator.models import WorldState

from ..config import SimulationConfig
from ..types import (
    ControlAction,
    DetectedObstacle,
    FusedObstacle,
    LiDARScan,
    StepTelemetry,
)


# =============================================================================
# Sink interface
# =============================================================================

class TelemetrySink(Protocol):
    """Narrow sink interface; no read-back."""

    def write_header(self, header: dict) -> None: ...

    def write_step(self, step: StepTelemetry) -> None: ...

    def close(self) -> None: ...


class NullSink:
    """No-op sink; used when telemetry is disabled."""

    def write_header(self, header: dict) -> None:
        return

    def write_step(self, step: StepTelemetry) -> None:
        return

    def close(self) -> None:
        return


# =============================================================================
# JSONL sink
# =============================================================================

@dataclass
class JsonlSink:
    """
    Writes header + per-step records as JSON Lines to `path`.

    File format:
        line 0     : {"type": "header", ...}
        line i > 0 : {"type": "step", ...}

    Always compact (no indent); arrays are Python lists of floats/ints.
    """

    path: str
    _fh: Optional[Any] = None  # file handle

    def __post_init__(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")

    def write_header(self, header: dict) -> None:
        assert self._fh is not None, "JsonlSink is closed"
        payload = {"type": "header", **header}
        self._fh.write(json.dumps(payload, default=_json_default) + "\n")
        self._fh.flush()

    def write_step(self, step: StepTelemetry) -> None:
        assert self._fh is not None, "JsonlSink is closed"
        payload = {"type": "step", **_step_to_dict(step)}
        self._fh.write(json.dumps(payload, default=_json_default) + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# =============================================================================
# Serialisation helpers
# =============================================================================

def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"Cannot serialise object of type {type(obj).__name__}")


def _scan_to_dict(scan: LiDARScan) -> dict:
    ranges = scan.ranges
    finite = np.isfinite(ranges)
    return {
        "time": float(scan.time),
        "pose_position": scan.pose_position.tolist(),
        "pose_heading": float(scan.pose_heading),
        "angles": scan.angles.tolist(),
        # Encode non-finite values as None so JSON stays valid.
        "ranges": [
            (float(r) if f else None) for r, f in zip(ranges.tolist(), finite.tolist())
        ],
    }


def _detection_to_dict(d: DetectedObstacle) -> dict:
    return {
        "centroid": d.centroid.tolist(),
        "radius": float(d.radius),
        "points": d.points.tolist() if d.points.size else [],
    }


def _fused_to_dict(f: FusedObstacle) -> dict:
    return {
        "track_id": int(f.track_id),
        "position": f.position.tolist(),
        "velocity": f.velocity.tolist(),
        "radius": float(f.radius),
    }


def _control_to_dict(c: ControlAction) -> dict:
    return {
        "linear_velocity": float(c.linear_velocity),
        "angular_velocity": float(c.angular_velocity),
    }


def _step_to_dict(step: StepTelemetry) -> dict:
    return {
        "step": int(step.step),
        "time": float(step.time),
        "agv": {
            "position": step.agv_state.position.tolist(),
            "heading": float(step.agv_state.heading),
            "velocity": step.agv_state.velocity.tolist(),
        },
        "obstacle_centroids": step.obstacle_centroids.tolist(),
        "obstacle_velocities": step.obstacle_velocities.tolist(),
        "obstacle_radii": step.obstacle_radii.tolist(),
        "lidar": _scan_to_dict(step.lidar_scan),
        "detections": [_detection_to_dict(d) for d in step.detections],
        "fused": [_fused_to_dict(f) for f in step.fused],
        "control": _control_to_dict(step.control),
        "collided": bool(step.collided),
        "goal_reached": bool(step.goal_reached),
        "step_wall_time": float(step.step_wall_time),
    }


# =============================================================================
# Header construction
# =============================================================================

def build_header(
    world: WorldState,
    sigma: SimulationConfig,
    detection_name: str,
    fusion_name: str,
    avoidance_name: str,
    seed: int,
) -> dict:
    """
    Assemble the run-level header record written as the first JSONL line.
    """
    return {
        "world_id": int(world.world_id),
        "world_seed": int(world.seed),
        "simulation_seed": int(seed),
        "sigma": sigma.to_dict(),
        "pipeline": {
            "detection": str(detection_name),
            "fusion": str(fusion_name),
            "avoidance": str(avoidance_name),
        },
        "environment": world.environment.to_dict(),
        "reference_path": world.reference_path.to_dict(),
        "agv_extent": world.agv_extent.to_dict(),
        "agv_initial": {
            "position": world.agv_position.tolist(),
            "heading": float(world.agv_heading),
        },
        "obstacles_initial": [o.to_dict() for o in world.obstacles],
    }


__all__ = [
    "JsonlSink",
    "NullSink",
    "TelemetrySink",
    "build_header",
]

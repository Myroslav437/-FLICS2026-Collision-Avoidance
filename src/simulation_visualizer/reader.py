"""
Telemetry reader — parses Simulation Engine JSONL telemetry files.

The telemetry file format is defined by ``core.telemetry.JsonlSink``:
  - Line 0: ``{"type": "header", ...}`` — run-level metadata
  - Lines 1..N: ``{"type": "step", ...}`` — per-step telemetry

This module converts the JSON records into typed dataclasses that the
panel modules consume. It deliberately does *not* import from the
simulation engine or world generator packages: the JSONL file is the
sole interface.

Raises ``TelemetryError`` with a human-readable message naming the
missing or malformed field whenever the input is not well-formed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# =========================================================================
# Exceptions
# =========================================================================

class TelemetryError(Exception):
    """Raised when telemetry is missing, malformed, or inconsistent."""


# =========================================================================
# Data classes — header
# =========================================================================

@dataclass
class TelemetryObstacleSpec:
    """
    Initial obstacle specification from the header.

    Mirrors ``Obstacle.to_dict()`` from the world generator.
    """

    shape: str              # "circle" or "square"
    inscribed_radius: float
    trajectory_positions: np.ndarray  # (n+1, 2)
    trajectory_speeds: np.ndarray     # (n,)

    @property
    def is_dynamic(self) -> bool:
        """True iff the obstacle has at least one trajectory segment."""
        return self.trajectory_speeds.shape[0] > 0


@dataclass
class TelemetryHeader:
    """
    Run-level metadata from the JSONL header record.

    Contains all time-invariant information needed to set up the
    visualization panels: world geometry, reference path, AGV extent,
    initial obstacle specs, and the simulation configuration.
    """

    world_id: int
    world_seed: int
    simulation_seed: int

    # Sigma
    sigma_name: str
    sigma: Dict[str, Any]  # full nested dict

    # Pipeline
    pipeline_detection: str
    pipeline_fusion: str
    pipeline_avoidance: str

    # Environment E
    environment_size: tuple  # (X, Y)
    environment_chains: List[np.ndarray]  # list of (k, 2) arrays

    # Reference path π
    reference_path_positions: np.ndarray  # (n+1, 2)
    reference_path_speeds: np.ndarray     # (n,)

    # AGV extent S_agv
    agv_length: float
    agv_width: float

    # AGV initial pose
    agv_initial_position: np.ndarray  # (2,)
    agv_initial_heading: float

    # Obstacles at t=0
    obstacles: List[TelemetryObstacleSpec]

    # Perception parameters (for M_0 reconstruction)
    eta_cpl: float
    sigma_map: float

    # LiDAR parameters
    lidar_r_min: float
    lidar_r_max: float


# =========================================================================
# Data classes — step
# =========================================================================

@dataclass
class StepLiDAR:
    """Per-step LiDAR scan data from §III-A.5–6."""

    time: float
    pose_position: np.ndarray  # (2,)
    pose_heading: float
    angles: np.ndarray         # (N,)
    ranges: np.ndarray         # (N,); inf for misses

    def cartesian_world(self) -> np.ndarray:
        """
        Project valid (finite-range) returns to world-frame (x, y).

        Returns an (M, 2) array where M is the number of finite rays.
        """
        mask = np.isfinite(self.ranges)
        r = self.ranges[mask]
        a = self.angles[mask] + self.pose_heading
        x = self.pose_position[0] + r * np.cos(a)
        y = self.pose_position[1] + r * np.sin(a)
        return np.column_stack([x, y])


@dataclass
class StepDetection:
    """A single obstacle detection (Ô_t element)."""

    centroid: np.ndarray  # (2,)
    radius: float
    points: np.ndarray    # (K, 2) or empty (0, 2)


@dataclass
class StepFused:
    """A single fused obstacle (Ō_t element)."""

    track_id: int
    position: np.ndarray  # (2,)
    velocity: np.ndarray  # (2,)
    radius: float


@dataclass
class StepControl:
    """Control action from the avoidance stage."""

    linear_velocity: float
    angular_velocity: float


@dataclass
class TelemetryStep:
    """
    One step's telemetry record.

    Fields cover both W_t (true) and pipeline outputs so the
    visualizer can overlay ground truth on perceived quantities.
    """

    step: int
    time: float

    # AGV state
    agv_position: np.ndarray   # (2,)
    agv_heading: float
    agv_velocity: np.ndarray   # (2,)

    # Ground-truth obstacles
    obstacle_centroids: np.ndarray   # (K, 2)
    obstacle_velocities: np.ndarray  # (K, 2)
    obstacle_radii: np.ndarray       # (K,)

    # LiDAR
    lidar: StepLiDAR

    # Pipeline outputs
    detections: List[StepDetection]
    fused: List[StepFused]
    control: StepControl

    # Termination flags
    collided: bool
    goal_reached: bool

    # Timing
    step_wall_time: float


# =========================================================================
# Data class — full run
# =========================================================================

@dataclass
class TelemetryRun:
    """A complete telemetry run: header + ordered step list."""

    header: TelemetryHeader
    steps: List[TelemetryStep]


# =========================================================================
# Parsing helpers
# =========================================================================

def _require(record: dict, key: str, context: str = "") -> Any:
    """Extract *key* from *record*, raising TelemetryError if absent."""
    if key not in record:
        label = f" in {context}" if context else ""
        raise TelemetryError(f"missing required field '{key}'{label}")
    return record[key]


def _parse_header(record: dict) -> TelemetryHeader:
    """Convert a raw JSON header dict to a ``TelemetryHeader``."""
    ctx = "header"

    sigma_dict = _require(record, "sigma", ctx)
    pipeline = _require(record, "pipeline", ctx)
    env = _require(record, "environment", ctx)
    ref = _require(record, "reference_path", ctx)
    agv_ext = _require(record, "agv_extent", ctx)
    agv_init = _require(record, "agv_initial", ctx)
    obs_list = _require(record, "obstacles_initial", ctx)

    # Parse environment chains
    chains_raw = _require(env, "chains", "header.environment")
    chains = [np.asarray(c, dtype=float) for c in chains_raw]
    env_size = tuple(_require(env, "size", "header.environment"))

    # Parse reference path
    ref_pos = np.asarray(
        _require(ref, "positions", "header.reference_path"), dtype=float
    )
    ref_spd = np.asarray(
        _require(ref, "speeds", "header.reference_path"), dtype=float
    )

    # Parse obstacles
    obstacles: List[TelemetryObstacleSpec] = []
    for i, obs_dict in enumerate(obs_list):
        spatial = _require(obs_dict, "spatial", f"header.obstacles_initial[{i}]")
        traj = _require(obs_dict, "trajectory", f"header.obstacles_initial[{i}]")
        obstacles.append(TelemetryObstacleSpec(
            shape=_require(spatial, "shape", f"obstacle[{i}].spatial"),
            inscribed_radius=float(
                _require(spatial, "inscribed_radius", f"obstacle[{i}].spatial")
            ),
            trajectory_positions=np.asarray(
                _require(traj, "positions", f"obstacle[{i}].trajectory"), dtype=float
            ),
            trajectory_speeds=np.asarray(
                _require(traj, "speeds", f"obstacle[{i}].trajectory"), dtype=float
            ),
        ))

    # Perception params
    perception = _require(sigma_dict, "perception", "header.sigma")
    eta_cpl = float(_require(perception, "eta_cpl", "header.sigma.perception"))
    sigma_map_val = float(
        _require(perception, "sigma_map", "header.sigma.perception")
    )

    # LiDAR params
    lidar_dict = _require(sigma_dict, "lidar", "header.sigma")
    geom = _require(lidar_dict, "geometry", "header.sigma.lidar")
    lidar_r_min = float(_require(geom, "r_min", "header.sigma.lidar.geometry"))
    lidar_r_max = float(_require(geom, "r_max", "header.sigma.lidar.geometry"))

    return TelemetryHeader(
        world_id=int(_require(record, "world_id", ctx)),
        world_seed=int(_require(record, "world_seed", ctx)),
        simulation_seed=int(_require(record, "simulation_seed", ctx)),
        sigma_name=str(_require(sigma_dict, "name", "header.sigma")),
        sigma=sigma_dict,
        pipeline_detection=str(_require(pipeline, "detection", "header.pipeline")),
        pipeline_fusion=str(_require(pipeline, "fusion", "header.pipeline")),
        pipeline_avoidance=str(_require(pipeline, "avoidance", "header.pipeline")),
        environment_size=env_size,
        environment_chains=chains,
        reference_path_positions=ref_pos,
        reference_path_speeds=ref_spd,
        agv_length=float(_require(agv_ext, "length", "header.agv_extent")),
        agv_width=float(_require(agv_ext, "width", "header.agv_extent")),
        agv_initial_position=np.asarray(
            _require(agv_init, "position", "header.agv_initial"), dtype=float
        ),
        agv_initial_heading=float(
            _require(agv_init, "heading", "header.agv_initial")
        ),
        obstacles=obstacles,
        eta_cpl=eta_cpl,
        sigma_map=sigma_map_val,
        lidar_r_min=lidar_r_min,
        lidar_r_max=lidar_r_max,
    )


def _parse_ranges(raw: list) -> np.ndarray:
    """Convert a JSON ranges list (with None for misses) to a float array."""
    return np.array(
        [float(r) if r is not None else np.inf for r in raw],
        dtype=float,
    )


def _parse_step(record: dict) -> TelemetryStep:
    """Convert a raw JSON step dict to a ``TelemetryStep``."""
    ctx = "step"

    agv = _require(record, "agv", ctx)
    lidar_raw = _require(record, "lidar", ctx)
    det_list = _require(record, "detections", ctx)
    fused_list = _require(record, "fused", ctx)
    ctrl_raw = _require(record, "control", ctx)

    # LiDAR
    lidar = StepLiDAR(
        time=float(_require(lidar_raw, "time", "step.lidar")),
        pose_position=np.asarray(
            _require(lidar_raw, "pose_position", "step.lidar"), dtype=float
        ),
        pose_heading=float(
            _require(lidar_raw, "pose_heading", "step.lidar")
        ),
        angles=np.asarray(
            _require(lidar_raw, "angles", "step.lidar"), dtype=float
        ),
        ranges=_parse_ranges(
            _require(lidar_raw, "ranges", "step.lidar")
        ),
    )

    # Detections
    detections: List[StepDetection] = []
    for i, d in enumerate(det_list):
        pts_raw = d.get("points", [])
        pts = np.asarray(pts_raw, dtype=float) if pts_raw else np.empty((0, 2))
        if pts.ndim == 1 and pts.size == 0:
            pts = np.empty((0, 2))
        detections.append(StepDetection(
            centroid=np.asarray(
                _require(d, "centroid", f"step.detections[{i}]"), dtype=float
            ),
            radius=float(_require(d, "radius", f"step.detections[{i}]")),
            points=pts,
        ))

    # Fused
    fused: List[StepFused] = []
    for i, f in enumerate(fused_list):
        fused.append(StepFused(
            track_id=int(_require(f, "track_id", f"step.fused[{i}]")),
            position=np.asarray(
                _require(f, "position", f"step.fused[{i}]"), dtype=float
            ),
            velocity=np.asarray(
                _require(f, "velocity", f"step.fused[{i}]"), dtype=float
            ),
            radius=float(_require(f, "radius", f"step.fused[{i}]")),
        ))

    # Control
    control = StepControl(
        linear_velocity=float(
            _require(ctrl_raw, "linear_velocity", "step.control")
        ),
        angular_velocity=float(
            _require(ctrl_raw, "angular_velocity", "step.control")
        ),
    )

    obs_centroids = np.asarray(
        _require(record, "obstacle_centroids", ctx), dtype=float
    )
    obs_velocities = np.asarray(
        _require(record, "obstacle_velocities", ctx), dtype=float
    )
    obs_radii = np.asarray(
        _require(record, "obstacle_radii", ctx), dtype=float
    )

    # Ensure 2D shape for empty arrays
    if obs_centroids.ndim == 1 and obs_centroids.size == 0:
        obs_centroids = np.empty((0, 2))
    if obs_velocities.ndim == 1 and obs_velocities.size == 0:
        obs_velocities = np.empty((0, 2))

    return TelemetryStep(
        step=int(_require(record, "step", ctx)),
        time=float(_require(record, "time", ctx)),
        agv_position=np.asarray(
            _require(agv, "position", "step.agv"), dtype=float
        ),
        agv_heading=float(_require(agv, "heading", "step.agv")),
        agv_velocity=np.asarray(
            _require(agv, "velocity", "step.agv"), dtype=float
        ),
        obstacle_centroids=obs_centroids,
        obstacle_velocities=obs_velocities,
        obstacle_radii=obs_radii,
        lidar=lidar,
        detections=detections,
        fused=fused,
        control=control,
        collided=bool(_require(record, "collided", ctx)),
        goal_reached=bool(_require(record, "goal_reached", ctx)),
        step_wall_time=float(_require(record, "step_wall_time", ctx)),
    )


# =========================================================================
# Public API
# =========================================================================

def read_telemetry(path: str) -> TelemetryRun:
    """
    Read a complete telemetry JSONL file produced by the Simulation Engine.

    Parameters
    ----------
    path : str
        Filesystem path to the ``.jsonl`` telemetry file.

    Returns
    -------
    TelemetryRun
        Header and ordered list of step records.

    Raises
    ------
    TelemetryError
        If the file is empty, lacks a header, or contains malformed records.
    """
    steps: List[TelemetryStep] = []
    header: Optional[TelemetryHeader] = None

    with open(path, "r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise TelemetryError(
                    f"line {line_no}: invalid JSON — {exc}"
                ) from exc

            rec_type = record.get("type")
            if rec_type == "header":
                if header is not None:
                    raise TelemetryError(
                        f"line {line_no}: duplicate header record"
                    )
                header = _parse_header(record)
            elif rec_type == "step":
                if header is None:
                    raise TelemetryError(
                        f"line {line_no}: step record before header"
                    )
                steps.append(_parse_step(record))
            else:
                raise TelemetryError(
                    f"line {line_no}: unknown record type '{rec_type}'"
                )

    if header is None:
        raise TelemetryError("telemetry file contains no header record")
    if not steps:
        raise TelemetryError("telemetry file contains no step records")

    return TelemetryRun(header=header, steps=steps)

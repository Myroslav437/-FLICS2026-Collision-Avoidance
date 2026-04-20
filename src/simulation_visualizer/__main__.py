"""
CLI driver for the Simulation Visualizer.

Invoked as::

    python -m simulation_visualizer play \\
        --telemetry path/to/run.jsonl \\
        --speed 1.0 \\
        --render-to output.mp4 \\
        --dpi 150

Reads a JSONL telemetry file produced by the Simulation Engine and
renders a four-panel animated visualization (objective world,
perception, pipeline output, metrics).
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure the project root and src/ are on sys.path
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from simulation_visualizer.playback import animate  # noqa: E402
from simulation_visualizer.reader import TelemetryError, read_telemetry  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="simulation_visualizer",
        description=(
            "Post-hoc playback of Simulation Engine telemetry. "
            "Renders four panels: objective world W_t, perception "
            "W̃_t, pipeline output, and per-run metrics (§IV-B)."
        ),
    )

    sub = parser.add_subparsers(dest="command")
    play = sub.add_parser(
        "play",
        help="Play back or render a telemetry file.",
    )
    play.add_argument(
        "--telemetry", required=True,
        help="Path to the JSONL telemetry file.",
    )
    play.add_argument(
        "--speed", type=float, default=1.0,
        help=(
            "Playback speed multiplier. 1.0 = wall-clock rate; "
            "0 = step-through on keypress; >1 = faster."
        ),
    )
    play.add_argument(
        "--render-to", default=None,
        help=(
            "Render to file instead of live window. "
            "Supports .mp4 (requires ffmpeg), .gif (requires Pillow), "
            "or a directory path for numbered PNGs."
        ),
    )
    play.add_argument(
        "--dpi", type=int, default=150,
        help="Render DPI (default: 150).",
    )
    return parser


def main() -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command != "play":
        parser.print_help()
        return 1

    telemetry_path = args.telemetry
    if not os.path.isfile(telemetry_path):
        print(f"Error: telemetry file not found: {telemetry_path}", file=sys.stderr)
        return 1

    try:
        run = read_telemetry(telemetry_path)
    except TelemetryError as exc:
        print(f"Error reading telemetry: {exc}", file=sys.stderr)
        return 1

    print(
        f"Loaded run: world {run.header.world_id}, "
        f"sigma={run.header.sigma_name}, "
        f"pipeline={run.header.pipeline_detection}+"
        f"{run.header.pipeline_fusion}+"
        f"{run.header.pipeline_avoidance}, "
        f"{len(run.steps)} steps"
    )

    animate(
        run,
        speed=args.speed,
        render_to=args.render_to,
        dpi=args.dpi,
    )

    if args.render_to:
        print(f"Rendered to: {args.render_to}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

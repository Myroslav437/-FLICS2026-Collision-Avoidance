#!/usr/bin/env python3
"""
Render a saved WorldState to a PNG.

Takes either a single `--input world.json` or a directory, and writes one
PNG per world using `src.visualization.save_world_figure`.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

# Add project root to sys.path so the src package is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.world_generator import WorldState  # noqa: E402
from src.visualization import save_world_figure  # noqa: E402


def _out_path(world_path: str, output: str) -> str:
    """Decide where to write the PNG given `world_path` and `output` spec."""
    if output.lower().endswith(".png"):
        return output
    os.makedirs(output, exist_ok=True)
    base = os.path.splitext(os.path.basename(world_path))[0]
    return os.path.join(output, f"{base}.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="world_*.json file OR directory containing them")
    parser.add_argument("--output", required=True,
                        help="PNG file (single input) or directory (batch)")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no-trajectories", action="store_true",
                        help="Hide dynamic-obstacle trajectories xi_i")
    parser.add_argument("--no-path", action="store_true",
                        help="Hide the reference path pi")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        world_paths = sorted(glob.glob(os.path.join(args.input, "world_*.json")))
        if not world_paths:
            print(f"No world_*.json files under {args.input}", file=sys.stderr)
            return 1
    elif os.path.isfile(args.input):
        world_paths = [args.input]
    else:
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    for wp in world_paths:
        world = WorldState.load(wp)
        out = _out_path(wp, args.output)
        save_world_figure(
            world, out, dpi=args.dpi,
            show_path=not args.no_path,
            show_trajectories=not args.no_trajectories,
        )
        print(f"{wp} -> {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

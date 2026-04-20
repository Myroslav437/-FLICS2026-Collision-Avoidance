#!/usr/bin/env python3
"""
CLI entry point to run a single simulation iteration.

It can either load a pre-calculated WorldState (e.g., from the validation dataset)
or generate a new world on-the-fly using a provided YAML configuration. Then, it
configures and runs the simulation engine with the requested pipeline stages.
"""

import argparse
import os
import sys

# Ensure the project root is in the path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from world_generator.models import WorldState
from simulation_engine import SimulationConfig, run_simulation
from simulation_engine.core.telemetry import JsonlSink, NullSink

def main():
    parser = argparse.ArgumentParser(description="Run the FLICS Simulation Engine.")
    
    parser.add_argument(
        "--world", type=str, required=True,
        help="Path to a pre-generated world JSON file (e.g. data/validation_v5/world_00000.json)."
    )
    
    # Simulation configuration
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to Simulation configuration file (e.g. config/simulation_nominal.yaml)."
    )
    parser.add_argument(
        "--detection", type=str, required=True, choices=["EC", "DB"],
        help="Detection pipeline stage."
    )
    parser.add_argument(
        "--fusion", type=str, required=True, choices=["PT", "KF"],
        help="Fusion tracking pipeline stage."
    )
    parser.add_argument(
        "--avoidance", type=str, required=True, choices=["VFH", "DWA"],
        help="Collision Avoidance pipeline stage."
    )
    parser.add_argument("--seed", type=int, default=42, help="Simulation random seed.")
    
    # Outputs
    parser.add_argument(
        "--telemetry", type=str,
        help="Path to write the playback telemetry JSONL file."
    )

    args = parser.parse_args()

    print(f"Loading pre-generated world from {args.world}...")
    world = WorldState.load(args.world)

    # 2. Extract configuration logic
    sigma_profile = SimulationConfig.load(args.config)
    sink = JsonlSink(args.telemetry) if args.telemetry else NullSink()

    print(f"Starting simulation run...")
    print(f" - Pipeline: [{args.detection}] -> [{args.fusion}] -> [{args.avoidance}]")
    print(f" - Sigma Profile: {sigma_profile.name.upper()} (from {args.config})")
    
    # 3. Simulate!
    metrics = run_simulation(
        world=world,
        sigma=sigma_profile,
        detection=args.detection,
        fusion=args.fusion,
        avoidance=args.avoidance,
        seed=args.seed,
        telemetry_sink=sink
    )

    if args.telemetry:
        sink.close()
        print(f"Telemetry saved to {args.telemetry}.")

    print("\nSimulation complete. Final Metrics:")
    for metric, value in metrics.to_dict().items():
        print(f"  {metric}: {value}")


if __name__ == "__main__":
    main()

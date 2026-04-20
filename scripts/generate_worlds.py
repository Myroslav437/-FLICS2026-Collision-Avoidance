#!/usr/bin/env python3
"""
Generate a corpus of initial worlds W_0 from Omega.

The corpus can be either

  (a) a flat set of N worlds generated from a single Omega (no stratification,
      for quick smoke testing), or

  (b) the stratified corpus: BSP depth d_bsp in {1, 2, 3}, static-obstacle
      count n_static in {0, 4, 8, 16}, dynamic-obstacle count n_dynamic
      in {0, 2, 4, 8}, with `--per-stratum` worlds per stratum (48
      strata total).

Each world is saved as JSON alongside its computed metrics; a manifest
CSV summarises the corpus.
"""

import argparse
import csv
import itertools
import json
import os
import sys
import time
import traceback
import yaml
from dataclasses import replace

# Add project root to sys.path so the src package is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.world_generator import Omega, generate_world  # noqa: E402
from src.world_generator.metrics import compute_metrics  # noqa: E402
from src.world_generator.prm import PRMGenerationError  # noqa: E402


def _build_strata(base: Omega, strata_dict: dict):
    """Yield (stratum_id, omega, combination_dict) tuples for the corpus."""
    if not strata_dict:
        yield 0, base, {}
        return

    keys = list(strata_dict.keys())
    values = [strata_dict[k] for k in keys]

    stratum_id = 0
    for combination in itertools.product(*values):
        combination_dict = {k: v for k, v in zip(keys, combination)}

        # Group updates by category
        updates = {"environment": {}, "agv": {}, "path": {}, "obstacles": {}}
        for k, v in combination_dict.items():
            category, attr = k.split('.', 1)
            updates[category][attr] = v

        new_env = replace(base.env, **updates["environment"]) if updates["environment"] else base.env
        new_agv = replace(base.agv, **updates["agv"]) if updates["agv"] else base.agv
        new_path = replace(base.path, **updates["path"]) if updates["path"] else base.path
        new_obs = replace(base.obs, **updates["obstacles"]) if updates["obstacles"] else base.obs

        omega = Omega(env=new_env, agv=new_agv, path=new_path, obs=new_obs)
        yield stratum_id, omega, combination_dict
        stratum_id += 1


def _generate_one(omega: Omega, seed: int, world_id: int):
    t0 = time.perf_counter()
    world = generate_world(omega, seed=seed, world_id=world_id)
    metrics = compute_metrics(world)
    return world, metrics, time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default_world.yaml",
                        help="Omega YAML (base parameter set)")
    parser.add_argument("--strata-config", default="config/default_strata.yaml",
                        help="YAML file defining the parameter arrays to stratify over")
    parser.add_argument("--output", required=True,
                        help="Output directory")
    parser.add_argument("--mode", choices=("single", "flat", "stratified"), default="single",
                        help="Generation mode (single = flat with count 1)")
    parser.add_argument("--count", type=int, default=10,
                        help="Worlds to generate in flat mode")
    parser.add_argument("--per-stratum", type=int, default=2,
                        help="Worlds per stratum in stratified mode")
    parser.add_argument("--seed-base", type=int, default=1000,
                        help="Seed offset (s_wg = seed_base + i)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Retry budget per world on PRM/placement failure")
    args = parser.parse_args()

    base = Omega.load(args.config)
    os.makedirs(args.output, exist_ok=True)

    manifest_path = os.path.join(args.output, "manifest.csv")
    jobs = []
    
    if args.mode == "single":
        args.mode = "flat"
        args.count = 1

    strata_dict = {}
    if args.mode == "stratified":
        with open(args.strata_config, "r", encoding="utf-8") as f:
            strata_dict = yaml.safe_load(f)

    if args.mode == "flat":
        for i in range(args.count):
            jobs.append({
                "stratum": 0,
                "omega": base,
                "stratum_params": {},
                "seed": args.seed_base + i,
                "world_id": i,
            })
    else:
        world_id = 0
        for stratum_id, omega, combination_dict in _build_strata(base, strata_dict):
            for k in range(args.per_stratum):
                jobs.append({
                    "stratum": stratum_id,
                    "omega": omega,
                    "stratum_params": combination_dict,
                    "seed": args.seed_base + world_id,
                    "world_id": world_id,
                })
                world_id += 1

    # Base fieldnames
    fieldnames = [
        "world_id", "stratum", "seed",
        "d_bsp", "n_static", "n_dynamic",
        "stratum_config",
        "free_space_ratio", "min_passage_width",
        "min_corridor_width",
        "static_obstacle_count", "dynamic_obstacle_count",
        "path_length", "prm_reference_attempts",
        "gen_time_sec", "status",
    ]
    
    n_ok = n_fail = 0
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for job in jobs:
            row = {
                "world_id": job["world_id"],
                "stratum": job["stratum"],
                "seed": job["seed"],
                "d_bsp": job["omega"].env.bsp_depth,
                "n_static": job["omega"].obs.n_static,
                "n_dynamic": job["omega"].obs.n_dynamic,
                "stratum_config": json.dumps(job["stratum_params"]),
            }
            attempt = 0
            while True:
                try:
                    world, metrics, dt = _generate_one(
                        job["omega"], job["seed"] + 10007 * attempt, job["world_id"]
                    )
                    world_path = os.path.join(
                        args.output, f"world_{job['world_id']:05d}.json"
                    )
                    world.save(world_path)
                    row.update(metrics.to_dict())
                    row["gen_time_sec"] = f"{dt:.3f}"
                    row["status"] = "ok"
                    n_ok += 1
                    break
                except (PRMGenerationError, RuntimeError) as exc:
                    attempt += 1
                    if attempt > args.max_retries:
                        row["status"] = f"fail:{type(exc).__name__}"
                        n_fail += 1
                        break
            writer.writerow(row)
            fh.flush()

    print(f"Done: {n_ok} ok, {n_fail} failed. Manifest: {manifest_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)

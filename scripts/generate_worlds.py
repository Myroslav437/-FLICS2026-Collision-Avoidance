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

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from dataclasses import replace

# Add project root to sys.path so the src package is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.world_generator import Omega, generate_world  # noqa: E402
from src.world_generator.metrics import compute_metrics  # noqa: E402
from src.world_generator.prm import PRMGenerationError  # noqa: E402


# Stratification axes (d_bsp >= 1 means at least one carved room; see
# src/world_generator/bsp.py).
STRATUM_D_BSP = (1, 2, 3)
STRATUM_N_STATIC = (0, 4, 8, 16)
STRATUM_N_DYNAMIC = (0, 2, 4, 8)


def _build_strata(base: Omega):
    """Yield (stratum_id, omega) tuples for the 48-stratum corpus."""
    stratum_id = 0
    for d in STRATUM_D_BSP:
        for ns in STRATUM_N_STATIC:
            for nd in STRATUM_N_DYNAMIC:
                env = replace(base.env, bsp_depth=d)
                obs = replace(base.obs, n_static=ns, n_dynamic=nd)
                omega = Omega(env=env, agv=base.agv, path=base.path, obs=obs)
                yield stratum_id, omega, d, ns, nd
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
    parser.add_argument("--output", required=True,
                        help="Output directory")
    parser.add_argument("--mode", choices=("flat", "stratified"), default="flat")
    parser.add_argument("--count", type=int, default=10,
                        help="Worlds to generate in flat mode")
    parser.add_argument("--per-stratum", type=int, default=2,
                        help="Worlds per stratum in stratified mode")
    parser.add_argument("--seed-base", type=int, default=0,
                        help="Seed offset (s_wg = seed_base + i)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Retry budget per world on PRM/placement failure")
    args = parser.parse_args()

    base = Omega.load(args.config)
    os.makedirs(args.output, exist_ok=True)

    manifest_path = os.path.join(args.output, "manifest.csv")
    jobs = []
    if args.mode == "flat":
        for i in range(args.count):
            jobs.append({
                "stratum": 0,
                "d_bsp": base.env.bsp_depth,
                "n_static": base.obs.n_static,
                "n_dynamic": base.obs.n_dynamic,
                "omega": base,
                "seed": args.seed_base + i,
                "world_id": i,
            })
    else:
        world_id = 0
        for stratum_id, omega, d, ns, nd in _build_strata(base):
            for k in range(args.per_stratum):
                jobs.append({
                    "stratum": stratum_id,
                    "d_bsp": d,
                    "n_static": ns,
                    "n_dynamic": nd,
                    "omega": omega,
                    "seed": args.seed_base + world_id,
                    "world_id": world_id,
                })
                world_id += 1

    fieldnames = [
        "world_id", "stratum", "d_bsp", "n_static", "n_dynamic",
        "seed", "free_space_ratio", "min_passage_width",
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
                "d_bsp": job["d_bsp"],
                "n_static": job["n_static"],
                "n_dynamic": job["n_dynamic"],
                "seed": job["seed"],
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

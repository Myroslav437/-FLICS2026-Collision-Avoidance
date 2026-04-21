#!/usr/bin/env python3
"""
Parallel benchmarking runner.

Enumerates the full factorial of (world) x (8 pipelines) x (3 sigma
profiles) for a corpus of generated worlds, runs each configuration in
parallel, and writes one fixed-schema JSON report (item 2 format) per
run to the output directory.

The run seed s_se is derived deterministically from the tuple
(world_id, detection, fusion, avoidance, sigma_profile, master_seed)
using a SHA-256 digest, so results are reproducible regardless of the
scheduler order. Partial re-runs skip any run whose report already
exists unless `--force` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))


DETECTIONS = ("EC", "DB")
FUSIONS = ("PT", "KF")
AVOIDANCES = ("VFH", "DWA")
SIGMA_PROFILES = ("nominal", "degraded-1", "degraded-2")

SIGMA_CONFIG_FILES = {
    "nominal":    "config/simulation_nominal.yaml",
    "degraded-1": "config/simulation_degraded1.yaml",
    "degraded-2": "config/simulation_degraded2.yaml",
}


# =============================================================================
# Job enumeration
# =============================================================================

@dataclass(frozen=True)
class Job:
    world_path: str
    world_id: int
    detection: str
    fusion: str
    avoidance: str
    sigma_profile: str
    sigma_config_path: str
    master_seed: int
    output_dir: str
    emit_telemetry: bool

    def derived_seed(self) -> int:
        key = (
            f"{self.world_id}|{self.detection}|{self.fusion}|{self.avoidance}|"
            f"{self.sigma_profile}|{self.master_seed}"
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return int(digest[:15], 16)  # 60-bit non-negative int

    def report_basename(self) -> str:
        return (
            f"world_{self.world_id:05d}."
            f"{self.detection}.{self.fusion}.{self.avoidance}."
            f"{self.sigma_profile}.json"
        )

    def report_path(self) -> str:
        return os.path.join(self.output_dir, self.report_basename())

    def telemetry_path(self) -> Optional[str]:
        if not self.emit_telemetry:
            return None
        base = os.path.splitext(self.report_basename())[0]
        return os.path.join(self.output_dir, "telemetry", base + ".jsonl")


_WORLD_FILE_RE = re.compile(r"^world_\d+\.json$")


def _iter_world_files(corpus_dir: str) -> List[Tuple[str, int]]:
    """
    Return a sorted list of (world_path, world_id) from a corpus directory.

    Only files matching ``world_NNNNN.json`` are returned; auxiliary
    artefacts such as ``world_00038_hist.jsonl`` or
    ``world_00038_hist.report.json`` that a prior telemetry/replay pass
    may have dropped into the corpus directory are ignored.

    `world_id` is read from the world JSON itself so it matches whatever
    the generator assigned, regardless of filename.
    """
    world_paths = []
    for name in sorted(os.listdir(corpus_dir)):
        if not _WORLD_FILE_RE.match(name):
            continue
        full = os.path.join(corpus_dir, name)
        with open(full, "r", encoding="utf-8") as f:
            data = json.load(f)
        world_paths.append((full, int(data["world_id"])))
    if not world_paths:
        raise FileNotFoundError(
            f"No world_NNNNN.json files found in {corpus_dir}"
        )
    return world_paths


def enumerate_jobs(
    corpus_dir: str,
    output_dir: str,
    master_seed: int,
    emit_telemetry: bool,
) -> List[Job]:
    """Enumerate the full factorial over the corpus and sigma/pipeline sets."""
    worlds = _iter_world_files(corpus_dir)
    jobs: List[Job] = []
    for (world_path, world_id), det, fus, avo, profile in itertools.product(
        worlds, DETECTIONS, FUSIONS, AVOIDANCES, SIGMA_PROFILES
    ):
        cfg_path = os.path.join(_PROJECT_ROOT, SIGMA_CONFIG_FILES[profile])
        jobs.append(Job(
            world_path=world_path,
            world_id=world_id,
            detection=det,
            fusion=fus,
            avoidance=avo,
            sigma_profile=profile,
            sigma_config_path=cfg_path,
            master_seed=master_seed,
            output_dir=output_dir,
            emit_telemetry=emit_telemetry,
        ))
    return jobs


# =============================================================================
# Worker
# =============================================================================

def _run_one(job: Job) -> dict:
    """
    Execute a single run in the worker process and write its report.

    Must be importable at module scope for ProcessPoolExecutor to pickle it.
    Returns a small status dict used by the parent for progress logging;
    the full run details are persisted in the report file.
    """
    # Lazy imports so child workers pay import cost once, not the parent.
    from world_generator.models import WorldState
    from simulation_engine import SimulationConfig, run_simulation
    from simulation_engine.core.report import build_report, write_report
    from simulation_engine.core.telemetry import JsonlSink, NullSink

    try:
        world = WorldState.load(job.world_path)
        sigma = SimulationConfig.load(job.sigma_config_path)
        seed = job.derived_seed()

        telemetry_path = job.telemetry_path()
        if telemetry_path is not None:
            os.makedirs(os.path.dirname(telemetry_path) or ".", exist_ok=True)
            sink = JsonlSink(telemetry_path)
        else:
            sink = NullSink()

        t0 = time.perf_counter()
        metrics = run_simulation(
            world=world,
            sigma=sigma,
            detection=job.detection,
            fusion=job.fusion,
            avoidance=job.avoidance,
            seed=seed,
            telemetry_sink=sink,
        )
        wall_time_s = time.perf_counter() - t0

        if telemetry_path is not None:
            sink.close()

        report = build_report(
            world=world,
            world_path=job.world_path,
            sigma=sigma,
            sigma_config_path=job.sigma_config_path,
            detection=job.detection,
            fusion=job.fusion,
            avoidance=job.avoidance,
            seed=seed,
            metrics=metrics,
            wall_time_s=wall_time_s,
            telemetry_path=telemetry_path,
        )
        write_report(report, job.report_path())
        return {"status": "ok", "report": job.report_path()}
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "fail",
            "report": job.report_path(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", required=True,
        help="Directory containing world_*.json files.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Directory to write per-run reports into.",
    )
    parser.add_argument(
        "--master-seed", type=int, default=42,
        help="Master seed; per-run s_se derived from this and the run key.",
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--max-runs", type=int, default=None,
        help="Cap the number of runs (after skip filter) for quick validation.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if a report file already exists.",
    )
    parser.add_argument(
        "--emit-telemetry", action="store_true",
        help="Emit per-step telemetry JSONL alongside each report.",
    )
    parser.add_argument(
        "--progress-every", type=int, default=25,
        help="Print a progress line every N completed runs.",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    all_jobs = enumerate_jobs(
        corpus_dir=args.corpus,
        output_dir=args.output,
        master_seed=args.master_seed,
        emit_telemetry=args.emit_telemetry,
    )

    if args.force:
        unfiltered = all_jobs
        n_skipped_existing = 0
    else:
        unfiltered = [j for j in all_jobs if not os.path.exists(j.report_path())]
        n_skipped_existing = len(all_jobs) - len(unfiltered)

    if args.max_runs is not None:
        pending = unfiltered[: args.max_runs]
        n_capped = len(unfiltered) - len(pending)
    else:
        pending = unfiltered
        n_capped = 0

    total = len(all_jobs)
    print(
        f"Benchmark: {total} total runs, {len(pending)} pending, "
        f"{n_skipped_existing} skipped (existing reports), "
        f"{n_capped} deferred (--max-runs), workers={args.workers}"
    )
    if not pending:
        print("Nothing to do.")
        return 0

    t0 = time.perf_counter()
    n_ok = 0
    n_fail = 0
    failures: List[dict] = []

    if args.workers == 1:
        # Serial fallback, useful for debugging.
        for i, job in enumerate(pending, start=1):
            result = _run_one(job)
            if result["status"] == "ok":
                n_ok += 1
            else:
                n_fail += 1
                failures.append(result)
            if i % args.progress_every == 0 or i == len(pending):
                elapsed = time.perf_counter() - t0
                rate = i / elapsed if elapsed > 0 else 0.0
                print(
                    f"  [{i}/{len(pending)}] ok={n_ok} fail={n_fail} "
                    f"elapsed={elapsed:.1f}s rate={rate:.2f}/s"
                )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_one, job): job for job in pending}
            done = 0
            for fut in as_completed(futures):
                done += 1
                result = fut.result()
                if result["status"] == "ok":
                    n_ok += 1
                else:
                    n_fail += 1
                    failures.append(result)
                if done % args.progress_every == 0 or done == len(pending):
                    elapsed = time.perf_counter() - t0
                    rate = done / elapsed if elapsed > 0 else 0.0
                    print(
                        f"  [{done}/{len(pending)}] ok={n_ok} fail={n_fail} "
                        f"elapsed={elapsed:.1f}s rate={rate:.2f}/s"
                    )

    elapsed = time.perf_counter() - t0
    print(
        f"Done in {elapsed:.1f}s: ok={n_ok} fail={n_fail} "
        f"(reports in {args.output})"
    )
    if failures:
        print("\nFailures:")
        for f in failures[:10]:
            print(f"  {f['report']}: {f['error']}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

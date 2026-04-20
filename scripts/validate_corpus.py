#!/usr/bin/env python3
"""
Validate a generated corpus against navigability-safety conditions.

Reads `manifest.csv` and reports:
  (1) navigability-safety: min_passage_width > W_agv + 2 * d_clear
      for every retained world;
  (2) free_space_ratio trend across d_bsp (mean should decrease);
  (3) obstacle-count sanity against Omega_obs;
  (4) min_passage_width distribution vs w_pass - t_wall target.

All thresholds are read from the canonical Omega YAML (single source of
truth) -- no numerical defaults are duplicated here.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

# Add project root to sys.path so the src package is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.world_generator import Omega  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                        help="Path to manifest.csv from generate_worlds.py")
    parser.add_argument("--config", default="config/default_world.yaml",
                        help="Omega YAML (reads W_agv, d_clear, w_pass, t_wall)")
    args = parser.parse_args()

    omega = Omega.load(args.config)
    w_agv = omega.agv.W_agv
    d_clear = omega.path.clearance
    w_pass = omega.env.passage_width
    t_wall = omega.env.wall_thickness
    c_min = omega.env.min_corridor
    l_pi_min = omega.path.min_path_length

    with open(args.manifest, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("status") == "ok"]
    if not rows:
        print("manifest contains no successfully generated worlds", file=sys.stderr)
        return 1

    threshold = w_agv + 2.0 * d_clear
    violators = []
    free_by_d = {}
    mpw_all = []
    mcw_all = []
    plen_all = []
    attempts_all = []
    for r in rows:
        d = int(r["d_bsp"])
        fsr = float(r["free_space_ratio"])
        mpw = float(r["min_passage_width"])
        free_by_d.setdefault(d, []).append(fsr)
        mpw_all.append(mpw)
        mcw_raw = r.get("min_corridor_width")
        if mcw_raw not in (None, "", "inf"):
            mcw_all.append(float(mcw_raw))
        plen_raw = r.get("path_length")
        if plen_raw not in (None, ""):
            plen_all.append(float(plen_raw))
        att_raw = r.get("prm_reference_attempts")
        if att_raw not in (None, "", "-1"):
            attempts_all.append(int(att_raw))
        if mpw < threshold:
            violators.append((r["world_id"], d, mpw))

    print("=" * 64)
    print(f"Validation report for {args.manifest}")
    print(f"Omega source: {args.config}")
    print(f"Worlds: {len(rows)}")
    print("=" * 64)

    print("\n[1] Navigability-safety "
          f"(min_passage_width > W_agv + 2*d_clear = {threshold:.3f})")
    if violators:
        print(f"  FAIL: {len(violators)} / {len(rows)} worlds violate threshold")
        # Per-stratum breakdown
        by_stratum = {}
        for wid, d, mpw in violators:
            by_stratum.setdefault(d, 0)
            by_stratum[d] += 1
        for d in sorted(by_stratum):
            print(f"    d_bsp={d}: {by_stratum[d]} failures")
        for wid, d, mpw in violators[:10]:
            print(f"    world_id={wid}  d_bsp={d}  min_passage_width={mpw:.3f}")
    else:
        print(f"  PASS: all {len(rows)} worlds satisfy navigability-safety")

    print("\n[2] Free-space ratio by d_bsp (mean, should decrease with d_bsp)")
    for d in sorted(free_by_d):
        v = free_by_d[d]
        print(f"  d_bsp={d}  n={len(v):3d}  mean={statistics.mean(v):.4f}"
              f"  stdev={statistics.stdev(v) if len(v) > 1 else 0.0:.4f}")
    means = [statistics.mean(free_by_d[d]) for d in sorted(free_by_d)]
    monotone = all(a > b for a, b in zip(means, means[1:]))
    print(f"  Monotone decrease: {'PASS' if monotone else 'FAIL'}")

    print("\n[3] Obstacle counts vs Omega_obs")
    mismatches = [
        r for r in rows
        if int(r["static_obstacle_count"]) != int(r["n_static"])
        or int(r["dynamic_obstacle_count"]) != int(r["n_dynamic"])
    ]
    if mismatches:
        print(f"  FAIL: {len(mismatches)} worlds have mismatched obstacle counts")
    else:
        print("  PASS: all worlds carry exactly (n_static, n_dynamic) obstacles")

    print(f"\n[4] min_passage_width distribution (target near "
          f"w_pass - t_wall = {w_pass - t_wall:.3f})")
    print(f"  min={min(mpw_all):.3f}  median={statistics.median(mpw_all):.3f}"
          f"  max={max(mpw_all):.3f}")

    print(f"\n[5] min_corridor_width (target >= c_min = {c_min:.3f})")
    mcw_violation = False
    if not mcw_all:
        print("  (no finite corridor measurements; only d_bsp=0 worlds?)")
    else:
        tol = 1e-6
        corridor_min = min(mcw_all)
        below = [v for v in mcw_all if v < c_min - tol]
        print(f"  n={len(mcw_all)}  min={corridor_min:.4f}"
              f"  median={statistics.median(mcw_all):.4f}"
              f"  max={max(mcw_all):.4f}")
        if below:
            mcw_violation = True
            print(f"  FAIL: {len(below)} worlds below c_min "
                  f"(smallest = {min(below):.4f})")
        else:
            print(f"  PASS: all corridors >= c_min within tol")

    print(f"\n[6] Reference-path length (target >= l_pi_min = {l_pi_min:.3f})")
    plen_violation = False
    if not plen_all:
        print("  (no path_length entries in manifest)")
    else:
        tol = 1e-6
        plen_min = min(plen_all)
        below = [v for v in plen_all if v < l_pi_min - tol]
        print(f"  n={len(plen_all)}  min={plen_min:.4f}"
              f"  median={statistics.median(plen_all):.4f}"
              f"  max={max(plen_all):.4f}")
        if below:
            plen_violation = True
            print(f"  FAIL: {len(below)} worlds below l_pi_min "
                  f"(smallest = {min(below):.4f}). Enforcement is buggy.")
        else:
            print(f"  PASS: all paths >= l_pi_min within tol")

    print("\n[7] l_pi_min rejection-loop attempts per world")
    if not attempts_all:
        print("  (no prm_reference_attempts entries in manifest)")
    else:
        a_max = max(attempts_all)
        a_med = statistics.median(attempts_all)
        print(f"  n={len(attempts_all)}  median={a_med:.1f}  max={a_max}")
        # Distribution: how many worlds took > 1 attempt (diagnostic).
        multi = sum(1 for a in attempts_all if a > 1)
        print(f"  worlds needing resample (>1 attempt): {multi}/{len(attempts_all)}")
        # The cap is implementation-internal; if a run ever reaches it
        # without raising, that's a manifest anomaly.
        if a_max >= 100:
            print(f"  FLAG: max attempts ({a_max}) equals the internal cap; "
                  f"l_pi_min may be too tight for some strata.")
        elif a_max > 20:
            print(f"  FLAG: max attempts ({a_max}) exceeds typical range "
                  f"(~20); consider revisiting l_pi_min vs. domain size.")

    return 0 if (not violators and monotone and not mismatches
                 and not mcw_violation and not plen_violation) else 1


if __name__ == "__main__":
    sys.exit(main())

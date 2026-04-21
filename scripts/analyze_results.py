#!/usr/bin/env python3
"""
Analysis pipeline over a directory of simulation run reports.

Produces the artifacts required by Section V of the paper:

  RQ1 (Environment Realism):
    - Figure 1: free-space-ratio and min-passage-width distributions,
      stratified by d_bsp (world-level metrics read from world JSONs).
    - Figure 2: travel metrics mu_col, mu_dev, mu_goal as functions of
      n_static and n_dynamic under Nominal.
    - Parameter-consistency: mean free-space ratio per d_bsp, min passage
      width summary.
    - Navigability-safety: fraction of worlds satisfying
      min_passage_width > W_agv + 2*d_clear.
    - Difficulty-spread: Spearman's rho between obstacle counts and each
      travel metric.

  RQ2 (Decomposition Validity):
    - Table IV: per-stage paired differences from switching each
      algorithm with the other two stages held fixed, under Nominal.
      Median paired diff for mu_col / mu_vel / mu_dev / mu_goal plus the
      paired Wilcoxon signed-rank p-value per cell.

  RQ3 (Result Stability):
    - Table V: pairwise Kendall's tau between the 8-pipeline rankings
      under Nominal / Degraded-1 / Degraded-2, per metric.
    - Seed-stability: partition Nominal runs into 5 disjoint shards,
      rank the 8 pipelines per shard, report mean pairwise Kendall's tau
      across shards.

  Aggregate (Table III):
    - Means of each metric per (pipeline, sigma profile), 8 x 6 x 3 cells.

Outputs:
  - artifacts/analysis_summary.md
  - artifacts/analysis_data.json
  - artifacts/figures/rq1_distributions.png
  - artifacts/figures/rq1_difficulty.png
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


PIPELINES = [
    (d, f, a)
    for d in ("EC", "DB")
    for f in ("PT", "KF")
    for a in ("VFH", "DWA")
]
SIGMA_PROFILES = ("nominal", "degraded-1", "degraded-2")
METRICS = ("mu_col", "mu_dev", "mu_goal", "mu_vel", "mu_comp", "mu_lat")
TRAVEL_METRICS = ("mu_col", "mu_dev", "mu_goal")

# Navigability-safety bound from world default config:
#   W_agv = 0.4, d_clear = 0.5 -> W_agv + 2*d_clear = 1.4 m
W_AGV_PLUS_CLEARANCE_M = 1.4
W_PASS_MINUS_T_WALL_M = 3.7  # 4.0 - 0.3 from default_world.yaml


# =============================================================================
# Loading
# =============================================================================

def load_reports(reports_dir: str) -> pd.DataFrame:
    """Load every *.json run report in `reports_dir` into a long DataFrame."""
    rows: List[dict] = []
    for name in sorted(os.listdir(reports_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(reports_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            rep = json.load(f)
        row = {
            "run_id":    rep["run_id"],
            "world_id":  int(rep["world"]["id"]),
            "world_path": rep["world"]["path"],
            "d_bsp":     int(rep["world"]["stratum"]["d_bsp"]),
            "n_static":  int(rep["world"]["stratum"]["n_static"]),
            "n_dynamic": int(rep["world"]["stratum"]["n_dynamic"]),
            "detection": rep["pipeline"]["detection"],
            "fusion":    rep["pipeline"]["fusion"],
            "avoidance": rep["pipeline"]["avoidance"],
            "sigma":     rep["sigma"]["profile"],
            "seed":      int(rep["seed"]),
        }
        for m in METRICS:
            row[m] = rep["metrics"][m]
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No reports found in {reports_dir}")
    df = pd.DataFrame(rows)
    df["pipeline"] = df["detection"] + "+" + df["fusion"] + "+" + df["avoidance"]
    return df


def load_world_metrics(corpus_dir: str) -> pd.DataFrame:
    """Compute per-world structural metrics from the corpus directory."""
    from world_generator.models import WorldState
    from world_generator.metrics import compute_metrics

    rows: List[dict] = []
    for name in sorted(os.listdir(corpus_dir)):
        if not (name.endswith(".json") and name.startswith("world_")):
            continue
        world = WorldState.load(os.path.join(corpus_dir, name))
        wm = compute_metrics(world)
        params = world.parameters or {}
        env = params.get("environment", {})
        obs = params.get("obstacles", {})
        rows.append({
            "world_id":          int(world.world_id),
            "d_bsp":             int(env["bsp_depth"]),
            "n_static":          int(obs["n_static"]),
            "n_dynamic":         int(obs["n_dynamic"]),
            "free_space_ratio":  float(wm.free_space_ratio),
            "min_passage_width": float(wm.min_passage_width),
            "path_length":       float(wm.path_length),
        })
    if not rows:
        raise FileNotFoundError(f"No world_*.json in {corpus_dir}")
    return pd.DataFrame(rows)


# =============================================================================
# Aggregates (Table III)
# =============================================================================

def aggregate_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of each metric per (pipeline, sigma profile)."""
    g = df.groupby(["pipeline", "sigma"], as_index=False)[list(METRICS)].mean()
    return g


# =============================================================================
# RQ1
# =============================================================================

def rq1_world_stats(world_df: pd.DataFrame) -> dict:
    """Parameter-consistency, navigability-safety, per-d_bsp summary."""
    by_bsp = (
        world_df.groupby("d_bsp")["free_space_ratio"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    nav_safe = float(
        (world_df["min_passage_width"] > W_AGV_PLUS_CLEARANCE_M).mean()
    )
    return {
        "free_space_by_d_bsp": by_bsp.to_dict(orient="records"),
        "min_passage_width_mean": float(world_df["min_passage_width"].mean()),
        "min_passage_width_median": float(
            world_df["min_passage_width"].median()
        ),
        "min_passage_width_theoretical": W_PASS_MINUS_T_WALL_M,
        "nav_safety_threshold_m": W_AGV_PLUS_CLEARANCE_M,
        "nav_safety_fraction": nav_safe,
        "n_worlds": int(world_df.shape[0]),
    }


def rq1_difficulty_spread(df_nominal: pd.DataFrame) -> dict:
    """Spearman rho between obstacle counts and each travel metric."""
    out: Dict[str, Dict[str, dict]] = {}
    for axis in ("n_static", "n_dynamic"):
        out[axis] = {}
        for metric in TRAVEL_METRICS:
            rho, p = stats.spearmanr(df_nominal[axis], df_nominal[metric])
            out[axis][metric] = {
                "rho": (None if np.isnan(rho) else float(rho)),
                "p_value": (None if np.isnan(p) else float(p)),
            }
    return out


def rq1_figure_distributions(world_df: pd.DataFrame, path: str) -> None:
    """Figure 1: free-space ratio & min-passage-width histograms per d_bsp."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    colors = {1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}
    for dbsp, sub in world_df.groupby("d_bsp"):
        c = colors.get(int(dbsp), "gray")
        axes[0].hist(
            sub["free_space_ratio"], bins=20, alpha=0.55,
            label=f"d_bsp={int(dbsp)}", color=c,
        )
        axes[1].hist(
            sub["min_passage_width"], bins=20, alpha=0.55,
            label=f"d_bsp={int(dbsp)}", color=c,
        )
    axes[0].set_xlabel("free-space ratio")
    axes[0].set_ylabel("count")
    axes[0].set_title("Free-space ratio")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("min passage width (m)")
    axes[1].set_title("Minimum passage width")
    axes[1].axvline(
        W_PASS_MINUS_T_WALL_M, linestyle="--", color="k",
        linewidth=0.8, alpha=0.6, label=f"w_pass - t_wall = {W_PASS_MINUS_T_WALL_M}",
    )
    axes[1].axvline(
        W_AGV_PLUS_CLEARANCE_M, linestyle=":", color="red",
        linewidth=0.8, alpha=0.8, label=f"W_agv + 2 d_clear = {W_AGV_PLUS_CLEARANCE_M}",
    )
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def rq1_figure_difficulty(df_nominal: pd.DataFrame, path: str) -> None:
    """Figure 2: travel metrics vs n_static (top) and n_dynamic (bottom)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharey="col")
    for j, metric in enumerate(TRAVEL_METRICS):
        for i, axis in enumerate(("n_static", "n_dynamic")):
            ax = axes[i, j]
            g = df_nominal.groupby(axis)[metric].mean().reset_index()
            ax.plot(
                g[axis], g[metric], marker="o", color="#1f77b4", linewidth=1.5,
            )
            ax.set_xlabel(axis)
            if j == 0:
                ax.set_ylabel(f"mean {metric}")
            ax.set_title(f"{metric} vs {axis}", fontsize=9)
            ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# =============================================================================
# RQ2
# =============================================================================

def rq2_table(df_nominal: pd.DataFrame) -> pd.DataFrame:
    """
    Per-stage paired differences under Nominal.

    For each stage s with two algorithm choices (a0, a1), pair every
    (world, other-two-fixed) combination where both algorithms produced
    a run and compute delta = metric[a1] - metric[a0]. Report the median
    delta and the paired Wilcoxon signed-rank test p-value per metric.
    """
    stages = [
        ("Detection (EC->DB)", "detection", "EC",  "DB", ("fusion", "avoidance")),
        ("Fusion (PT->KF)",    "fusion",    "PT",  "KF", ("detection", "avoidance")),
        ("Avoidance (VFH->DWA)", "avoidance", "VFH", "DWA", ("detection", "fusion")),
    ]
    metrics = ["mu_col", "mu_vel", "mu_dev", "mu_goal"]
    rows: List[dict] = []
    for label, stage_col, a0, a1, other_cols in stages:
        piv = df_nominal.pivot_table(
            index=["world_id", *other_cols],
            columns=stage_col,
            values=metrics,
            aggfunc="mean",
        )
        row = {"stage": label}
        for m in metrics:
            if (m, a0) not in piv.columns or (m, a1) not in piv.columns:
                row[f"median_delta_{m}"] = None
                row[f"wilcoxon_p_{m}"] = None
                row[f"n_pairs_{m}"] = 0
                continue
            paired = piv[m].dropna(subset=[a0, a1])
            delta = (paired[a1] - paired[a0]).to_numpy()
            row[f"median_delta_{m}"] = float(np.median(delta)) if delta.size else None
            row[f"n_pairs_{m}"] = int(delta.size)
            # Wilcoxon requires at least one nonzero difference
            if delta.size == 0 or np.all(delta == 0):
                row[f"wilcoxon_p_{m}"] = None
            else:
                try:
                    w = stats.wilcoxon(delta, zero_method="wilcox")
                    row[f"wilcoxon_p_{m}"] = float(w.pvalue)
                except ValueError:
                    row[f"wilcoxon_p_{m}"] = None
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# RQ3
# =============================================================================

def _rank_pipelines(df: pd.DataFrame, metric: str) -> pd.Series:
    """
    Return a Series indexed by pipeline with the pipeline's rank under
    the given metric (mean over the rows of `df`). Lower mean = rank 1.

    For mu_goal and mu_vel the semantic direction differs, but we use
    *numeric* ranks across pipelines consistently: the absolute rank
    choice does not affect pairwise Kendall's tau (tau is invariant to
    joint monotonic transforms of the rankings), so we rank on the
    native metric value for all metrics.
    """
    means = df.groupby("pipeline")[metric].mean()
    return means.rank(method="average")


def rq3_tau_across_sigma(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Kendall's tau of 8-pipeline rankings across sigma profiles, per metric."""
    rows: List[dict] = []
    for metric in METRICS:
        ranks = {}
        for sig in SIGMA_PROFILES:
            sub = df[df["sigma"] == sig]
            if sub.empty:
                continue
            ranks[sig] = _rank_pipelines(sub, metric)
        pairs = [
            ("nominal", "degraded-1"),
            ("nominal", "degraded-2"),
            ("degraded-1", "degraded-2"),
        ]
        row = {"metric": metric}
        for a, b in pairs:
            key = f"tau_{a}_vs_{b}"
            if a not in ranks or b not in ranks:
                row[key] = None
                continue
            common = ranks[a].index.intersection(ranks[b].index)
            if len(common) < 2:
                row[key] = None
                continue
            tau, _ = stats.kendalltau(ranks[a].loc[common], ranks[b].loc[common])
            row[key] = (None if np.isnan(tau) else float(tau))
        rows.append(row)
    return pd.DataFrame(rows)


def rq3_seed_stability(df_nominal: pd.DataFrame, n_shards: int = 5) -> dict:
    """
    Partition Nominal runs into `n_shards` disjoint shards (by world_id),
    rank the 8 pipelines per shard, report per-metric mean pairwise
    Kendall's tau across shards.
    """
    if df_nominal.empty:
        return {"n_shards": n_shards, "per_metric": {}, "overall_mean_tau": None}
    world_ids = sorted(df_nominal["world_id"].unique())
    # Round-robin shard assignment for balanced shards.
    shard_of = {wid: i % n_shards for i, wid in enumerate(world_ids)}
    df = df_nominal.copy()
    df["shard"] = df["world_id"].map(shard_of)

    per_metric: Dict[str, dict] = {}
    overall_taus: List[float] = []
    for metric in METRICS:
        shard_ranks = {}
        for s in range(n_shards):
            sub = df[df["shard"] == s]
            if sub.empty:
                continue
            shard_ranks[s] = _rank_pipelines(sub, metric)
        taus: List[float] = []
        for a, b in itertools.combinations(sorted(shard_ranks), 2):
            common = shard_ranks[a].index.intersection(shard_ranks[b].index)
            if len(common) < 2:
                continue
            tau, _ = stats.kendalltau(
                shard_ranks[a].loc[common], shard_ranks[b].loc[common]
            )
            if not np.isnan(tau):
                taus.append(float(tau))
        per_metric[metric] = {
            "mean_tau": (float(np.mean(taus)) if taus else None),
            "n_pairs": len(taus),
        }
        overall_taus.extend(taus)
    return {
        "n_shards": n_shards,
        "per_metric": per_metric,
        "overall_mean_tau": (
            float(np.mean(overall_taus)) if overall_taus else None
        ),
    }


# =============================================================================
# Markdown emission
# =============================================================================

def _fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "--"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _md_table(df: pd.DataFrame) -> str:
    """Render a pandas DataFrame as a GitHub-flavoured markdown table."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def render_summary(
    agg: pd.DataFrame,
    rq1_stats: dict,
    rq1_spread: dict,
    rq2_tab: pd.DataFrame,
    rq3_tau: pd.DataFrame,
    rq3_seed: dict,
    fig1_rel: str,
    fig2_rel: str,
    n_runs: int,
    n_worlds: int,
) -> str:
    parts: List[str] = []
    parts.append(f"# Analysis summary\n")
    parts.append(
        f"Source: {n_runs} run reports over {n_worlds} worlds "
        f"({len(PIPELINES)} pipelines x {len(SIGMA_PROFILES)} sigma profiles).\n"
    )

    parts.append("\n## Table III -- Aggregate metrics (mean per pipeline x sigma)\n")
    # Pivot: pipelines as rows, (metric, sigma) columns grouped.
    agg_wide = agg.pivot(index="pipeline", columns="sigma", values=list(METRICS))
    # Flatten MultiIndex columns
    agg_wide.columns = [f"{m} [{s}]" for m, s in agg_wide.columns]
    agg_wide = agg_wide.reset_index()
    parts.append(_md_table(agg_wide))

    parts.append("\n\n## RQ1 -- Environment Realism\n")
    parts.append(f"Worlds: {rq1_stats['n_worlds']}\n")
    parts.append(f"\n![Figure 1 -- distributions]({fig1_rel})\n")
    parts.append(f"\n![Figure 2 -- difficulty spread]({fig2_rel})\n")

    parts.append("\n**Parameter consistency (mean free-space ratio by d_bsp):**\n")
    parts.append(_md_table(pd.DataFrame(rq1_stats["free_space_by_d_bsp"])))

    parts.append(
        f"\n\nMin passage width: mean = {_fmt(rq1_stats['min_passage_width_mean'])} m, "
        f"median = {_fmt(rq1_stats['min_passage_width_median'])} m, "
        f"theoretical w_pass - t_wall = {rq1_stats['min_passage_width_theoretical']} m.\n"
    )
    parts.append(
        f"\nNavigability-safety (min passage width > "
        f"{rq1_stats['nav_safety_threshold_m']} m): "
        f"{rq1_stats['nav_safety_fraction']*100:.1f}% of worlds.\n"
    )

    parts.append("\n**Difficulty-spread (Spearman rho, Nominal runs):**\n\n")
    spread_rows: List[dict] = []
    for axis, per_metric in rq1_spread.items():
        for metric, vals in per_metric.items():
            spread_rows.append({
                "axis": axis, "metric": metric,
                "spearman_rho": vals["rho"], "p_value": vals["p_value"],
            })
    parts.append(_md_table(pd.DataFrame(spread_rows)))

    parts.append("\n\n## Table IV -- RQ2 Decomposition Validity (Nominal, per-stage)\n")
    parts.append(_md_table(rq2_tab))

    parts.append("\n\n## Table V -- RQ3 Pairwise Kendall's tau across sigma profiles\n")
    parts.append(_md_table(rq3_tau))

    parts.append("\n\n**Seed-stability (5 disjoint shards of Nominal):**\n\n")
    seed_rows = [
        {"metric": m,
         "mean_tau": rq3_seed["per_metric"].get(m, {}).get("mean_tau"),
         "n_pairs": rq3_seed["per_metric"].get(m, {}).get("n_pairs", 0)}
        for m in METRICS
    ]
    parts.append(_md_table(pd.DataFrame(seed_rows)))
    parts.append(
        f"\n\nOverall mean pairwise Kendall's tau (across metrics and "
        f"shard pairs): {_fmt(rq3_seed['overall_mean_tau'])}\n"
    )
    return "".join(parts)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports", required=True,
        help="Directory of run reports (one JSON per run).",
    )
    parser.add_argument(
        "--corpus", required=True,
        help=(
            "Directory of world_*.json files used to produce the reports "
            "(world-level metrics are computed from these)."
        ),
    )
    parser.add_argument(
        "--output", default="artifacts",
        help="Output directory for analysis artifacts.",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "figures"), exist_ok=True)

    print(f"Loading reports from {args.reports} ...")
    df = load_reports(args.reports)
    print(f"  loaded {len(df)} reports")

    print(f"Computing world-level metrics from {args.corpus} ...")
    world_df = load_world_metrics(args.corpus)
    print(f"  loaded {len(world_df)} worlds")

    df_nominal = df[df["sigma"] == "nominal"].copy()
    # Join world metrics so difficulty-spread can use them if needed.
    if not df_nominal.empty:
        df_nominal = df_nominal.merge(
            world_df[["world_id", "free_space_ratio", "min_passage_width"]],
            on="world_id", how="left",
        )

    # Aggregate.
    agg = aggregate_table(df)

    # RQ1.
    rq1_stats = rq1_world_stats(world_df)
    rq1_spread = (
        rq1_difficulty_spread(df_nominal)
        if not df_nominal.empty else
        {a: {m: {"rho": None, "p_value": None} for m in TRAVEL_METRICS}
         for a in ("n_static", "n_dynamic")}
    )
    fig1_path = os.path.join(args.output, "figures", "rq1_distributions.png")
    rq1_figure_distributions(world_df, fig1_path)
    fig2_path = os.path.join(args.output, "figures", "rq1_difficulty.png")
    if not df_nominal.empty:
        rq1_figure_difficulty(df_nominal, fig2_path)
    else:
        # Produce an empty-plot placeholder when no Nominal runs exist.
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No Nominal runs available",
                ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(fig2_path, dpi=140)
        plt.close(fig)

    # RQ2.
    rq2_tab = (
        rq2_table(df_nominal)
        if not df_nominal.empty else
        pd.DataFrame(columns=["stage"])
    )

    # RQ3.
    rq3_tau = rq3_tau_across_sigma(df)
    rq3_seed = (
        rq3_seed_stability(df_nominal)
        if not df_nominal.empty else
        {"n_shards": 5, "per_metric": {}, "overall_mean_tau": None}
    )

    # Emit artifacts.
    summary = render_summary(
        agg=agg,
        rq1_stats=rq1_stats,
        rq1_spread=rq1_spread,
        rq2_tab=rq2_tab,
        rq3_tau=rq3_tau,
        rq3_seed=rq3_seed,
        fig1_rel="figures/rq1_distributions.png",
        fig2_rel="figures/rq1_difficulty.png",
        n_runs=len(df),
        n_worlds=len(world_df),
    )
    md_path = os.path.join(args.output, "analysis_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Wrote {md_path}")

    data = {
        "n_runs": len(df),
        "n_worlds": len(world_df),
        "aggregate": agg.to_dict(orient="records"),
        "rq1_world_stats": rq1_stats,
        "rq1_difficulty_spread": rq1_spread,
        "rq2_table": rq2_tab.to_dict(orient="records"),
        "rq3_tau_across_sigma": rq3_tau.to_dict(orient="records"),
        "rq3_seed_stability": rq3_seed,
    }
    data_path = os.path.join(args.output, "analysis_data.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)
    print(f"Wrote {data_path}")
    print(f"Figures in {os.path.join(args.output, 'figures')}")
    return 0


def _json_default(o):
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if np.isnan(v) else v
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Unserialisable: {type(o).__name__}")


if __name__ == "__main__":
    sys.exit(main())

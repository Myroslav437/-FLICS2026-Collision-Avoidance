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
        execu = rep.get("execution", {}) or {}
        row["collision_step"] = execu.get("collision_step")
        row["goal_reach_step"] = execu.get("goal_reach_step")
        # Timeout: neither collided nor reached goal within T_horizon.
        row["timeout"] = int(
            row["collision_step"] is None and row["goal_reach_step"] is None
        )
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

def _paired_delta(
    df: pd.DataFrame,
    stage_col: str,
    a0: str,
    a1: str,
    other_cols: Tuple[str, ...],
    metrics: Tuple[str, ...],
    filter_kwargs: Optional[dict] = None,
) -> dict:
    """
    Compute paired metric deltas when switching `stage_col` from `a0` to `a1`,
    holding `other_cols` fixed. Returns a dict with median delta, Wilcoxon
    p-value, and pair count per metric.
    """
    sub = df
    if filter_kwargs:
        for k, v in filter_kwargs.items():
            sub = sub[sub[k] == v]
    piv = sub.pivot_table(
        index=["world_id", *other_cols],
        columns=stage_col,
        values=list(metrics),
        aggfunc="mean",
    )
    row: Dict[str, Optional[float]] = {}
    for m in metrics:
        if (m, a0) not in piv.columns or (m, a1) not in piv.columns:
            row[f"median_delta_{m}"] = None
            row[f"wilcoxon_p_{m}"] = None
            row[f"n_pairs_{m}"] = 0
            continue
        paired = piv[m].dropna(subset=[a0, a1])
        delta = (paired[a1] - paired[a0]).to_numpy()
        # Mean paired difference is more informative than median for binary
        # metrics (mu_col, mu_goal) where the median is usually zero.
        row[f"median_delta_{m}"] = float(np.mean(delta)) if delta.size else None
        row[f"n_pairs_{m}"] = int(delta.size)
        if delta.size == 0 or np.all(delta == 0):
            row[f"wilcoxon_p_{m}"] = None
        else:
            try:
                w = stats.wilcoxon(delta, zero_method="wilcox")
                row[f"wilcoxon_p_{m}"] = float(w.pvalue)
            except ValueError:
                row[f"wilcoxon_p_{m}"] = None
    return row


def rq2_table(df_nominal: pd.DataFrame) -> pd.DataFrame:
    """
    Per-stage paired differences under Nominal, with Fusion sliced by
    avoidance to test property (ii).
    """
    metrics = ("mu_col", "mu_vel", "mu_dev", "mu_goal")
    specs: List[Tuple[str, str, str, str, Tuple[str, ...], Optional[dict]]] = [
        ("Detection (EC->DB)", "detection", "EC",  "DB",
         ("fusion", "avoidance"), None),
        ("Fusion (PT->KF)", "fusion", "PT", "KF",
         ("detection", "avoidance"), None),
        ("Fusion (PT->KF) | Avoidance=VFH", "fusion", "PT", "KF",
         ("detection",), {"avoidance": "VFH"}),
        ("Fusion (PT->KF) | Avoidance=DWA", "fusion", "PT", "KF",
         ("detection",), {"avoidance": "DWA"}),
        ("Avoidance (VFH->DWA)", "avoidance", "VFH", "DWA",
         ("detection", "fusion"), None),
    ]
    rows: List[dict] = []
    for label, stage_col, a0, a1, other_cols, filt in specs:
        row = {"stage": label}
        row.update(
            _paired_delta(df_nominal, stage_col, a0, a1, other_cols, metrics, filt)
        )
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


def rq3_seed_stability(df_nominal: pd.DataFrame, n_shards: int = 3) -> dict:
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

    parts.append(
        f"\n\n**Seed-stability ({rq3_seed.get('n_shards', '?')} "
        f"disjoint shards of Nominal):**\n\n"
    )
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
# Paper-value extraction (inline numbers used in paper.tex §V prose)
# =============================================================================

def compute_paper_values(
    df: pd.DataFrame,
    df_nominal: pd.DataFrame,
    agg: pd.DataFrame,
    rq1_stats: dict,
    rq2_tab: pd.DataFrame,
    rq3_tau: pd.DataFrame,
    rq3_seed: dict,
) -> dict:
    """Produce every inline number needed to replace paper.tex placeholders."""
    out: Dict[str, object] = {}

    # ---- RQ1 in-line values ----
    fsr_by_bsp = {
        int(r["d_bsp"]): float(r["mean"])
        for r in rq1_stats["free_space_by_d_bsp"]
    }
    out["rq1_fsr_d_bsp_1"] = fsr_by_bsp.get(1)
    out["rq1_fsr_d_bsp_2"] = fsr_by_bsp.get(2)
    out["rq1_fsr_d_bsp_3"] = fsr_by_bsp.get(3)
    means_in_order = [fsr_by_bsp[k] for k in sorted(fsr_by_bsp)]
    out["rq1_fsr_monotonic"] = all(
        a > b for a, b in zip(means_in_order, means_in_order[1:])
    )
    out["rq1_min_passage_mean"] = rq1_stats["min_passage_width_mean"]
    out["rq1_nav_safety_pct"] = rq1_stats["nav_safety_fraction"] * 100.0
    out["rq1_nav_safety_gt_98"] = rq1_stats["nav_safety_fraction"] > 0.98

    # Obstacle-count range for mu_col/mu_dev/mu_goal under Nominal.
    if not df_nominal.empty:
        # Combine by the maximum of obstacle counts on each axis for a
        # simple headline range: compute per-(n_static, n_dynamic) means and
        # take the min (lowest-difficulty) and max (highest-difficulty) cell.
        g = df_nominal.groupby(["n_static", "n_dynamic"])[list(TRAVEL_METRICS)].mean()
        tot_obs = g.index.get_level_values("n_static") + g.index.get_level_values("n_dynamic")
        low_mask = tot_obs == tot_obs.min()  # 0 obstacles
        high_mask = tot_obs == tot_obs.max()  # 16 + 8 = 24 obstacles
        low_row = g[low_mask].mean()
        high_row = g[high_mask].mean()
        out["rq1_mu_col_low"] = float(low_row["mu_col"])
        out["rq1_mu_col_high"] = float(high_row["mu_col"])
        out["rq1_mu_dev_low"] = float(low_row["mu_dev"])
        out["rq1_mu_dev_high"] = float(high_row["mu_dev"])
        out["rq1_mu_goal_low"] = float(low_row["mu_goal"])
        out["rq1_mu_goal_high"] = float(high_row["mu_goal"])
    else:
        for k in ("mu_col_low", "mu_col_high", "mu_dev_low", "mu_dev_high",
                  "mu_goal_low", "mu_goal_high"):
            out[f"rq1_{k}"] = None

    # Spearman-sign check.
    # Expected: mu_col, mu_dev rise with obstacle count (positive rho);
    #           mu_goal falls (negative rho).
    # ---- RQ2 in-line values ----
    # Look up deltas from rq2_tab.
    row_by_stage = {r["stage"]: r for r in rq2_tab.to_dict(orient="records")}

    def _abs(stage: str, metric: str) -> Optional[float]:
        v = row_by_stage.get(stage, {}).get(f"median_delta_{metric}")
        return None if v is None else abs(float(v))

    det = row_by_stage.get("Detection (EC->DB)", {})
    fus = row_by_stage.get("Fusion (PT->KF)", {})
    avd = row_by_stage.get("Avoidance (VFH->DWA)", {})

    out["rq2_det_delta_col"] = det.get("median_delta_mu_col")
    out["rq2_det_delta_vel"] = det.get("median_delta_mu_vel")
    out["rq2_det_delta_dev"] = det.get("median_delta_mu_dev")
    out["rq2_det_delta_goal"] = det.get("median_delta_mu_goal")
    out["rq2_fus_delta_col"] = fus.get("median_delta_mu_col")
    out["rq2_fus_delta_vel"] = fus.get("median_delta_mu_vel")
    out["rq2_fus_delta_dev"] = fus.get("median_delta_mu_dev")
    out["rq2_fus_delta_goal"] = fus.get("median_delta_mu_goal")
    out["rq2_avd_delta_col"] = avd.get("median_delta_mu_col")
    out["rq2_avd_delta_vel"] = avd.get("median_delta_mu_vel")
    out["rq2_avd_delta_dev"] = avd.get("median_delta_mu_dev")
    out["rq2_avd_delta_goal"] = avd.get("median_delta_mu_goal")

    fus_vfh = row_by_stage.get("Fusion (PT->KF) | Avoidance=VFH", {})
    fus_dwa = row_by_stage.get("Fusion (PT->KF) | Avoidance=DWA", {})
    out["rq2_fus_vfh_delta_col"] = fus_vfh.get("median_delta_mu_col")
    out["rq2_fus_vfh_delta_vel"] = fus_vfh.get("median_delta_mu_vel")
    out["rq2_fus_vfh_delta_dev"] = fus_vfh.get("median_delta_mu_dev")
    out["rq2_fus_vfh_delta_goal"] = fus_vfh.get("median_delta_mu_goal")
    out["rq2_fus_dwa_delta_col"] = fus_dwa.get("median_delta_mu_col")
    out["rq2_fus_dwa_delta_vel"] = fus_dwa.get("median_delta_mu_vel")
    out["rq2_fus_dwa_delta_dev"] = fus_dwa.get("median_delta_mu_dev")
    out["rq2_fus_dwa_delta_goal"] = fus_dwa.get("median_delta_mu_goal")

    # Property (i): direct stage sensitivity.
    #   - Detection: |delta mu_col| > |delta mu_dev| AND > |delta mu_goal|
    #   - Fusion:    |delta mu_vel| > each of col/dev/goal
    #   - Avoidance: |delta mu_dev| > |delta mu_col| and |delta mu_goal|
    #                percent-effect > |delta mu_col|
    def _cmp_gt(a, b):
        if a is None or b is None:
            return None
        return float(a) > float(b)

    p1_det = _cmp_gt(_abs("Detection (EC->DB)", "mu_col"),
                     _abs("Detection (EC->DB)", "mu_dev"))
    p1_det_goal = _cmp_gt(_abs("Detection (EC->DB)", "mu_col"),
                          _abs("Detection (EC->DB)", "mu_goal"))
    p1_fus_col = _cmp_gt(_abs("Fusion (PT->KF)", "mu_vel"),
                         _abs("Fusion (PT->KF)", "mu_col"))
    p1_fus_dev = _cmp_gt(_abs("Fusion (PT->KF)", "mu_vel"),
                         _abs("Fusion (PT->KF)", "mu_dev"))
    p1_avd = _cmp_gt(_abs("Avoidance (VFH->DWA)", "mu_dev"),
                     _abs("Avoidance (VFH->DWA)", "mu_col"))
    p1_avd_goal = _cmp_gt(abs(float(avd.get("median_delta_mu_goal") or 0.0)),
                          _abs("Avoidance (VFH->DWA)", "mu_col") or 0.0)

    checks_p1 = [p1_det, p1_det_goal, p1_fus_col, p1_fus_dev, p1_avd, p1_avd_goal]
    n_pass = sum(1 for c in checks_p1 if c is True)
    if n_pass == len(checks_p1):
        out["rq2_property_i_verdict"] = "holds"
    elif n_pass >= 3:
        out["rq2_property_i_verdict"] = "partially holds"
    else:
        out["rq2_property_i_verdict"] = "does not hold"
    out["rq2_property_i_checks"] = {
        "det_col_dominates_dev": p1_det,
        "det_col_dominates_goal": p1_det_goal,
        "fus_vel_dominates_col": p1_fus_col,
        "fus_vel_dominates_dev": p1_fus_dev,
        "avd_dev_dominates_col": p1_avd,
        "avd_goal_dominates_col": p1_avd_goal,
    }

    # Property (ii): DWA's fusion-switch effect on mu_goal exceeds VFH's.
    vfh_goal = fus_vfh.get("median_delta_mu_goal")
    dwa_goal = fus_dwa.get("median_delta_mu_goal")
    if vfh_goal is None or dwa_goal is None:
        p2_holds = None
    else:
        p2_holds = abs(float(dwa_goal)) > abs(float(vfh_goal))
    out["rq2_property_ii_verdict"] = (
        "holds" if p2_holds is True else
        "does not hold" if p2_holds is False else
        "inconclusive"
    )

    # ---- RQ3 values ----
    out["rq3_seed_overall_mean_tau"] = rq3_seed.get("overall_mean_tau")
    tau_per = rq3_seed.get("per_metric") or {}
    # Focus on the four metrics the paper references:
    per_travel = [
        tau_per.get(m, {}).get("mean_tau")
        for m in ("mu_col", "mu_dev", "mu_goal", "mu_vel")
    ]
    per_travel_valid = [v for v in per_travel if v is not None]
    min_tau_travel = min(per_travel_valid) if per_travel_valid else None
    out["rq3_seed_min_tau_travel"] = min_tau_travel

    # Classification of seed robustness.
    overall = rq3_seed.get("overall_mean_tau")
    if overall is None:
        out["rq3_seed_robustness"] = "inconclusive"
    elif overall >= 0.7:
        out["rq3_seed_robustness"] = "robust"
    elif overall >= 0.4:
        out["rq3_seed_robustness"] = "moderately robust"
    else:
        out["rq3_seed_robustness"] = "fragile"

    # Cross-sigma stability: summarize tau across three pair columns for the
    # four travel metrics (as in Table V).
    tau_rows = rq3_tau.to_dict(orient="records")
    tau_by_metric = {r["metric"]: r for r in tau_rows}
    pair_cols = ("tau_nominal_vs_degraded-1",
                 "tau_nominal_vs_degraded-2",
                 "tau_degraded-1_vs_degraded-2")
    all_taus: List[float] = []
    per_metric_min: Dict[str, float] = {}
    for m in ("mu_col", "mu_dev", "mu_goal", "mu_vel"):
        vals = [tau_by_metric.get(m, {}).get(c) for c in pair_cols]
        vals = [v for v in vals if v is not None]
        if vals:
            per_metric_min[m] = float(min(vals))
            all_taus.extend(vals)
    out["rq3_xsigma_min_tau"] = (
        float(min(per_metric_min.values())) if per_metric_min else None
    )
    out["rq3_xsigma_mean_tau"] = (
        float(np.mean(all_taus)) if all_taus else None
    )
    # Ranking stability classification: uniform if every tau in Table V >= 0.7.
    out["rq3_xsigma_uniform"] = (
        all(v >= 0.7 for v in all_taus) if all_taus else False
    )

    # ---- DWA timeout fractions per sigma ----
    timeouts: Dict[str, object] = {}
    for sig in SIGMA_PROFILES:
        sub = df[(df["sigma"] == sig) & (df["avoidance"] == "DWA")]
        if sub.empty:
            timeouts[sig] = None
        else:
            timeouts[sig] = float(sub["timeout"].mean())
    out["dwa_timeout_fraction"] = timeouts
    # Also overall timeout fraction per avoidance.
    overall_to: Dict[str, object] = {}
    for avo in ("VFH", "DWA"):
        sub = df[df["avoidance"] == avo]
        overall_to[avo] = (
            None if sub.empty else float(sub["timeout"].mean())
        )
    out["avoidance_timeout_fraction"] = overall_to

    # ---- Nominal->Degraded-2 aggregate-worsening summary ----
    agg_wide = agg.pivot(index="pipeline", columns="sigma", values=list(METRICS))
    worsen: Dict[str, Dict[str, bool]] = {}
    for pipe in agg_wide.index:
        per_metric: Dict[str, bool] = {}
        for metric in METRICS:
            try:
                n = float(agg_wide.loc[pipe, (metric, "nominal")])
                d2 = float(agg_wide.loc[pipe, (metric, "degraded-2")])
            except (KeyError, ValueError):
                continue
            if metric == "mu_goal":  # higher is better
                per_metric[metric] = d2 < n
            else:  # lower is better
                per_metric[metric] = d2 > n
        worsen[str(pipe)] = per_metric
    out["degraded2_vs_nominal_worse"] = worsen

    return out


def _tex_num(x: Optional[float], nd: int = 3, pct: bool = False) -> str:
    """Format a numeric value for inline substitution into paper.tex."""
    if x is None:
        return "--"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if np.isnan(v):
        return "--"
    if pct:
        return f"{v*100:.1f}"
    return f"{v:.{nd}f}"


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
        {"n_shards": 3, "per_metric": {}, "overall_mean_tau": None}
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

    paper_values = (
        compute_paper_values(
            df=df,
            df_nominal=df_nominal,
            agg=agg,
            rq1_stats=rq1_stats,
            rq2_tab=rq2_tab,
            rq3_tau=rq3_tau,
            rq3_seed=rq3_seed,
        )
        if not df_nominal.empty
        else {}
    )
    data = {
        "n_runs": len(df),
        "n_worlds": len(world_df),
        "aggregate": agg.to_dict(orient="records"),
        "rq1_world_stats": rq1_stats,
        "rq1_difficulty_spread": rq1_spread,
        "rq2_table": rq2_tab.to_dict(orient="records"),
        "rq3_tau_across_sigma": rq3_tau.to_dict(orient="records"),
        "rq3_seed_stability": rq3_seed,
        "paper_values": paper_values,
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

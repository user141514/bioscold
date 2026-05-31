"""
D4A3: Compare Methods - Per-method A4C metrics and bootstrap significance
Parts E-F: Method comparison summary, bootstrap
"""
import json, os, sys, warnings, random, collections
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

OUTDIR = Path(r"E:\zuhui\bioisosteric_diffusion\plan_results\routeA_chembl37k_d4a3_geometry_a4c_evaluation")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("=" * 60)
print("D4A3: Compare Methods (Parts E-F)")
print("=" * 60)

# ============================================================
# Load data
# ============================================================
print("\n=== Loading data ===")

a4c_df = pd.read_csv(OUTDIR / "d4a3_a4c_review_results.csv")
print(f"A4C review results: {len(a4c_df)} rows")

# Get eval query set with stratum info
eval_df = pd.read_csv(OUTDIR / "d4a3_eval_query_set.csv")
print(f"Eval query set: {len(eval_df)} rows")

# Merge stratum into A4C data
stratum_map = dict(zip(eval_df["query_id"], eval_df["stratum"]))
a4c_df["stratum"] = a4c_df["query_id"].map(stratum_map)

# Define buckets
REVIEW_READY_BUCKETS = ["REVIEW_READY", "REVIEW_READY_WITH_WARNING"]
HARD_REJECT_BUCKETS = ["HARD_GEOMETRY_REJECT", "HARD_CHEMISTRY_ALERT", "PROPERTY_SHIFT_WARNING"]

a4c_df["is_review_ready"] = a4c_df["a4c_bucket"].isin(REVIEW_READY_BUCKETS).astype(int)
a4c_df["is_hard_reject"] = a4c_df["a4c_bucket"].isin(HARD_REJECT_BUCKETS).astype(int)
a4c_df["is_geometry_ok"] = (a4c_df["geometry_success"] == 1).astype(int)
a4c_df["is_positive"] = a4c_df["is_exact_positive"].astype(int)

# ============================================================
# PART E: Per-Method, Per-Query Metrics
# ============================================================
print("\n=== Part E: Method Comparison ===")

methods = sorted(a4c_df["method"].unique())
print(f"Methods: {methods}")

# Per-query aggregation: for each method+query, compute metrics
per_query_metrics = []
for (qid, method), group in a4c_df.groupby(["query_id", "method"]):
    # Only consider top-10 proposals per query per method
    top10 = group[group["rank"] <= 10]

    n_top10 = len(top10)
    n_review_ready = top10["is_review_ready"].sum()
    n_hard_reject = top10["is_hard_reject"].sum()
    n_geometry_ok = top10["is_geometry_ok"].sum()
    n_positive = top10["is_positive"].sum()

    # Get stratum for this query
    qstratum = stratum_map.get(qid, "unknown")

    per_query_metrics.append({
        "query_id": qid,
        "method": method,
        "stratum": qstratum,
        "n_proposals": n_top10,
        "n_review_ready": n_review_ready,
        "n_hard_reject": n_hard_reject,
        "n_geometry_ok": n_geometry_ok,
        "n_positive_in_top10": n_positive,
        "at_least_one_review_ready": int(n_review_ready > 0),
        "at_least_one_positive": int(n_positive > 0),
        "review_ready_rate": n_review_ready / max(n_top10, 1),
        "hard_reject_rate": n_hard_reject / max(n_top10, 1),
        "geometry_ok_rate": n_geometry_ok / max(n_top10, 1),
        "positive_capture_rate": n_positive / max(n_top10, 1),
    })

per_query_df = pd.DataFrame(per_query_metrics)

# Aggregate by method
print("\nPer-method summary:")
method_agg_rows = []
for method in methods:
    mdf = per_query_df[per_query_df["method"] == method]
    n_queries = len(mdf)
    method_agg_rows.append({
        "method": method,
        "n_queries": n_queries,
        "mean_at_least_one_review_ready": mdf["at_least_one_review_ready"].mean(),
        "mean_review_ready_rate": mdf["review_ready_rate"].mean(),
        "mean_hard_reject_rate": mdf["hard_reject_rate"].mean(),
        "mean_top10_positive_rate": mdf["positive_capture_rate"].mean(),
        "mean_geometry_ok_rate": mdf["geometry_ok_rate"].mean(),
        "std_review_ready_rate": mdf["review_ready_rate"].std(),
        "std_hard_reject_rate": mdf["hard_reject_rate"].std(),
        "std_positive_capture_rate": mdf["positive_capture_rate"].std(),
    })
    print(f"  {method:30s}: n_queries={n_queries:5d}, review_ready={mdf['at_least_one_review_ready'].mean():.3f}, "
          f"review_rate={mdf['review_ready_rate'].mean():.3f}, "
          f"hard_reject={mdf['hard_reject_rate'].mean():.3f}, "
          f"positive_rate={mdf['positive_capture_rate'].mean():.3f}")

method_agg_df = pd.DataFrame(method_agg_rows)
method_agg_df.to_csv(OUTDIR / "d4a3_method_comparison_summary.csv", index=False)
print(f"\nSaved d4a3_method_comparison_summary.csv")

# By stratum
print("\nPer-method per-stratum summary:")
stratum_method_rows = []
for method in methods:
    mdf = per_query_df[per_query_df["method"] == method]
    for stratum in sorted(mdf["stratum"].unique()):
        sdf = mdf[mdf["stratum"] == stratum]
        if len(sdf) == 0:
            continue
        stratum_method_rows.append({
            "method": method,
            "stratum": stratum,
            "n_queries": len(sdf),
            "mean_at_least_one_review_ready": sdf["at_least_one_review_ready"].mean(),
            "mean_review_ready_rate": sdf["review_ready_rate"].mean(),
            "mean_hard_reject_rate": sdf["hard_reject_rate"].mean(),
            "mean_top10_positive_rate": sdf["positive_capture_rate"].mean(),
            "mean_geometry_ok_rate": sdf["geometry_ok_rate"].mean(),
        })
        print(f"  {method:30s} str={stratum:20s}: n={len(sdf):4d}, review_ready={sdf['at_least_one_review_ready'].mean():.3f}")

stratum_method_df = pd.DataFrame(stratum_method_rows)
stratum_method_df.to_csv(OUTDIR / "d4a3_method_comparison_by_stratum.csv", index=False)
print(f"Saved d4a3_method_comparison_by_stratum.csv")

# ============================================================
# PART F: Bootstrap
# ============================================================
print("\n=== Part F: Bootstrap Significance ===")

N_BOOTSTRAP = 1000
CI_ALPHA = 0.05

# Compare HGB vs attachment_frequency
hgb_metrics = per_query_df[per_query_df["method"] == "M1_canonical_HGB"]
attach_metrics = per_query_df[per_query_df["method"] == "M0_attachment_frequency"]

# Align queries
hgb_queries = set(hgb_metrics["query_id"])
attach_queries = set(attach_metrics["query_id"])
common_queries = sorted(hgb_queries & attach_queries)
print(f"Common queries for bootstrap: {len(common_queries)}")

if len(common_queries) < 100:
    print("WARNING: Too few common queries for bootstrap. Results may be unreliable.")

hgb_idx = {row["query_id"]: i for i, row in hgb_metrics.iterrows()}
attach_idx = {row["query_id"]: i for i, row in attach_metrics.iterrows()}

# Observed differences
key_metrics = ["at_least_one_review_ready", "hard_reject_rate", "review_ready_rate",
               "positive_capture_rate", "geometry_ok_rate"]
bootstrap_results = []

for metric in key_metrics:
    hgb_vals = hgb_metrics.loc[[hgb_idx[q] for q in common_queries], metric].values
    attach_vals = attach_metrics.loc[[attach_idx[q] for q in common_queries], metric].values

    if metric in ["at_least_one_review_ready"]:
        # Already a per-query binary metric
        obs_diff = np.mean(hgb_vals) - np.mean(attach_vals)
        # Bootstrap
        n = len(common_queries)
        boot_diffs = []
        for b in range(N_BOOTSTRAP):
            indices = np.random.randint(0, n, size=n)
            diff = np.mean(hgb_vals[indices]) - np.mean(attach_vals[indices])
            boot_diffs.append(diff)
    else:
        obs_diff = np.mean(hgb_vals) - np.mean(attach_vals)
        n = len(common_queries)
        boot_diffs = []
        for b in range(N_BOOTSTRAP):
            indices = np.random.randint(0, n, size=n)
            diff = np.mean(hgb_vals[indices]) - np.mean(attach_vals[indices])
            boot_diffs.append(diff)

    boot_diffs = np.array(boot_diffs)
    ci_lo = np.percentile(boot_diffs, 100 * CI_ALPHA / 2)
    ci_hi = np.percentile(boot_diffs, 100 * (1 - CI_ALPHA / 2))
    if obs_diff > 0:
        p_value = np.mean(boot_diffs <= 0)
    elif obs_diff < 0:
        p_value = np.mean(boot_diffs >= 0)
    else:
        p_value = 1.0  # No difference
    p_value = 2 * min(p_value, 1 - p_value)  # Two-sided
    p_value = min(p_value, 1.0)  # Clip to 1.0
    significant = ci_lo > 0 or ci_hi < 0

    bootstrap_results.append({
        "metric": metric,
        "hgb_mean": float(np.mean(hgb_vals)),
        "attach_mean": float(np.mean(attach_vals)),
        "observed_difference": float(obs_diff),
        "bootstrap_samples": N_BOOTSTRAP,
        "ci_95_lo": float(ci_lo),
        "ci_95_hi": float(ci_hi),
        "p_value": float(p_value),
        "significant_at_0.05": int(significant),
    })

    print(f"  {metric:40s}: HGB={np.mean(hgb_vals):.4f} vs Attach={np.mean(attach_vals):.4f}, "
          f"diff={obs_diff:.4f}, CI=[{ci_lo:.4f}, {ci_hi:.4f}], "
          f"p={p_value:.4f}, sig={significant}")

boot_df = pd.DataFrame(bootstrap_results)
boot_df.to_csv(OUTDIR / "d4a3_bootstrap_method_comparison.csv", index=False)
print(f"\nSaved d4a3_bootstrap_method_comparison.csv")

# Also compare M2 (best D4A2 ranker) vs attach freq
m2_metrics = per_query_df[per_query_df["method"] == "M2_best_D4A2_ranker"]
m2_queries = set(m2_metrics["query_id"])
m2_common = sorted(attach_queries & m2_queries)
print(f"\nM2 vs Attach common queries: {len(m2_common)}")

m2_idx = {row["query_id"]: i for i, row in m2_metrics.iterrows()}
bootstrap_results_m2 = []

for metric in key_metrics:
    m2_vals = m2_metrics.loc[[m2_idx[q] for q in m2_common], metric].values
    attach_vals = attach_metrics.loc[[attach_idx[q] for q in m2_common], metric].values

    obs_diff = np.mean(m2_vals) - np.mean(attach_vals)
    n = len(m2_common)
    boot_diffs = []
    for b in range(N_BOOTSTRAP):
        indices = np.random.randint(0, n, size=n)
        diff = np.mean(m2_vals[indices]) - np.mean(attach_vals[indices])
        boot_diffs.append(diff)

    boot_diffs = np.array(boot_diffs)
    ci_lo = np.percentile(boot_diffs, 100 * CI_ALPHA / 2)
    ci_hi = np.percentile(boot_diffs, 100 * (1 - CI_ALPHA / 2))
    if obs_diff > 0:
        p_value = np.mean(boot_diffs <= 0)
    elif obs_diff < 0:
        p_value = np.mean(boot_diffs >= 0)
    else:
        p_value = 1.0
    p_value = 2 * min(p_value, 1 - p_value)
    p_value = min(p_value, 1.0)
    significant = ci_lo > 0 or ci_hi < 0

    bootstrap_results_m2.append({
        "comparison": "M2_best_D4A2_ranker_vs_attach",
        "metric": metric,
        "method1_mean": float(np.mean(m2_vals)),
        "method2_mean": float(np.mean(attach_vals)),
        "observed_difference": float(obs_diff),
        "bootstrap_samples": N_BOOTSTRAP,
        "ci_95_lo": float(ci_lo),
        "ci_95_hi": float(ci_hi),
        "p_value": float(p_value),
        "significant_at_0.05": int(significant),
    })

    print(f"  M2 vs Attach - {metric:30s}: M2={np.mean(m2_vals):.4f} vs Attach={np.mean(attach_vals):.4f}, "
          f"diff={obs_diff:.4f}, CI=[{ci_lo:.4f}, {ci_hi:.4f}], "
          f"p={p_value:.4f}, sig={significant}")

# Write combined bootstrap results
all_bootstrap_rows = []
for r in bootstrap_results:
    all_bootstrap_rows.append({
        "comparison": "M1_HGB_vs_M0_attach_freq",
        **r
    })
for r in bootstrap_results_m2:
    all_bootstrap_rows.append(r)

boot_all_df = pd.DataFrame(all_bootstrap_rows)
boot_all_df.to_csv(OUTDIR / "d4a3_bootstrap_method_comparison.csv", index=False)
print(f"Saved d4a3_bootstrap_method_comparison.csv ({len(boot_all_df)} rows)")

print(f"\n=== Script 2 Complete ===")
print(f"Output files:")
print(f"  {OUTDIR / 'd4a3_method_comparison_summary.csv'}")
print(f"  {OUTDIR / 'd4a3_method_comparison_by_stratum.csv'}")
print(f"  {OUTDIR / 'd4a3_bootstrap_method_comparison.csv'}")

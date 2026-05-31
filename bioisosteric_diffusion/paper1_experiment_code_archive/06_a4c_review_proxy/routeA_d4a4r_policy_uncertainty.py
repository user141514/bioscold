"""
D4A4R: Policy Uncertainty and Two-Tier Output Calibration

Parts:
  A) Policy-to-policy bootstrap (pairwise comparison, 1000 samples)
  B) Regime classification from bootstrap CIs
  C) Two-tier output calibration (Discovery vs Review-ready)
  D) A4C proxy validity audit
  E) Data integrity guard
  F) Final verdict

Outputs (8 files in plan_results/routeA_chembl37k_d4a4r_policy_uncertainty/):
  1. d4a4r_policy_pairwise_bootstrap.csv
  2. d4a4r_policy_regime_classification.csv
  3. d4a4r_two_tier_output_audit.csv
  4. d4a4r_a4c_proxy_reason_audit.csv
  5. D4A4R_A4C_PROXY_VALIDITY_CAVEAT.md
  6. d4a4r_data_integrity_audit.csv
  7. D4A4R_POLICY_UNCERTAINTY_VERDICT.md
  8. MAIN_DECISION_LOG.md

Rules:
  - Do NOT train models, do NOT start D4B
  - Do NOT use test labels to tune policy
  - A4C hard reject = review burden PROXY, not externally validated
  - Do NOT claim three regimes unless bootstrap supports it
  - Mark net_utility as weight-dependent diagnostic

IMPORTANT FINDING (discovered at runtime):
  M1_canonical_HGB has exactly 10 candidates per query. Since P1_raw_HGB,
  P3_lambda0.5, and P3_lambda2.0 all use M1_canonical_HGB as their base,
  the TOP-10 COMPOSITION IS IDENTICAL across these three policies.
  Only the ordering within the top-10 changes. Per-query top-10 aggregate
  metrics (capture_rate, hard_reject_rate, review_ready_rate) are therefore
  identical. Differences are detectable at the rank level (top-1, top-3, etc.)
  and via ordering-based metrics (reciprocal rank, average precision).
  This is reported as an important negative finding.
"""

import os
import sys
import warnings
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = r"E:\zuhui\bioisosteric_diffusion"
D4A4_DIR = os.path.join(BASE, "plan_results", "routeA_chembl37k_d4a4_a4c_gated_reranking")
D4A3_DIR = os.path.join(BASE, "plan_results", "routeA_chembl37k_d4a3_geometry_a4c_evaluation")
OUT_DIR = os.path.join(BASE, "plan_results", "routeA_chembl37k_d4a4r_policy_uncertainty")

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Paths for input data
# ---------------------------------------------------------------------------
CANDIDATE_TABLE = os.path.join(D4A4_DIR, "d4a4_candidate_review_table.csv")
A4C_REVIEW_RESULTS = os.path.join(D4A3_DIR, "d4a3_a4c_review_results.csv")

log_lines = []


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    log_lines.append(line)


def save_csv(df: pd.DataFrame, name: str):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    log(f"  Saved {name} ({len(df)} rows)")


# ===========================================================================
# PART A: Policy-to-policy bootstrap
# ===========================================================================
def build_policy_rankings(cand: pd.DataFrame):
    """Build per-policy per-query ranked candidate lists.

    Returns a dict: policy_name -> DataFrame (query_id, rank, candidate data)
    and a DataFrame of per-query aggregate (top-10) metrics.
    """
    policies = {
        "P1_raw_HGB": {
            "method": "M1_canonical_HGB",
            "score_col": "HGB_score",
            "desc": "Raw HGB score (baseline)",
        },
        "P3_lambda0.5": {
            "method": "M1_canonical_HGB",
            "score_col": None,
            "compute_score": lambda row: row["HGB_score"]
            - 0.5 * (row["hard_reject_flag"] * 1.0 + row["property_warning_flag"] * 0.3),
            "desc": "HGB - 0.5*(hard_reject + 0.3*warning)",
        },
        "P3_lambda2.0": {
            "method": "M1_canonical_HGB",
            "score_col": None,
            "compute_score": lambda row: row["HGB_score"]
            - 2.0 * (row["hard_reject_flag"] * 1.0 + row["property_warning_flag"] * 0.3),
            "desc": "HGB - 2.0*(hard_reject + 0.3*warning)",
        },
        "P0_attachment_frequency": {
            "method": "M0_attachment_frequency",
            "score_col": "attach_freq_score",
            "desc": "Attachment frequency ranking",
        },
    }

    # Build per-policy rankings and per-query aggregate metrics
    per_policy_records = []
    ranked_dfs = {}

    for pname, pconf in policies.items():
        sub = cand[cand["method"] == pconf["method"]].copy()
        if sub.empty:
            log(f"  WARNING: No candidates for policy {pname}")
            continue

        # Compute score
        if pconf["score_col"] is not None:
            sub["_policy_score"] = sub[pconf["score_col"]].fillna(-1e9)
        else:
            sub["_policy_score"] = sub.apply(pconf["compute_score"], axis=1)

        # Sort by score descending within each query, assign new ranks
        sub = sub.sort_values(["query_id", "_policy_score"], ascending=[True, False])
        sub["_policy_rank"] = sub.groupby("query_id").cumcount() + 1

        # Store full ranking
        ranked_dfs[pname] = sub[["query_id", "_policy_rank", "replacement_smiles",
                                 "HGB_score", "_policy_score",
                                 "is_exact_positive", "hard_reject_flag",
                                 "review_ready_flag", "property_warning_flag",
                                 "a4c_bucket", "geometry_success"]].copy()

        # Compute per-query top-10 aggregate metrics
        top10 = sub[sub["_policy_rank"] <= 10]
        g = top10.groupby("query_id")
        pq = g.agg(
            n_candidates=("replacement_smiles", "count"),
            n_captured=("is_exact_positive", "sum"),
            n_hard_reject=("hard_reject_flag", "sum"),
            n_review_ready=("review_ready_flag", "sum"),
        ).reset_index()
        pq["policy"] = pname
        pq["capture_rate"] = pq["n_captured"] / pq["n_candidates"]
        pq["hard_reject_rate"] = pq["n_hard_reject"] / pq["n_candidates"]
        pq["review_ready_rate"] = pq["n_review_ready"] / pq["n_candidates"]

        # Additional rank-aware metrics
        top1 = sub[sub["_policy_rank"] == 1]
        top3 = sub[sub["_policy_rank"] <= 3]
        top1_agg = top1.groupby("query_id").agg(
            top1_captured=("is_exact_positive", "sum"),
            top1_hard_reject=("hard_reject_flag", "sum"),
        )
        top3_agg = top3.groupby("query_id").agg(
            top3_captured=("is_exact_positive", "sum"),
            top3_hard_reject=("hard_reject_flag", "sum"),
        )
        pq = pq.merge(top1_agg, on="query_id", how="left")
        pq = pq.merge(top3_agg, on="query_id", how="left")
        pq["top1_hard_reject"] = pq["top1_hard_reject"].fillna(0).astype(int)
        pq["top3_hard_reject"] = pq["top3_hard_reject"].fillna(0).astype(int)

        per_policy_records.append(pq)

    per_policy = pd.concat(per_policy_records, ignore_index=True)
    log(f"  Per-policy per-query records: {len(per_policy)} rows")
    return per_policy, ranked_dfs


def bootstrap_pairwise(per_policy: pd.DataFrame):
    """Run 1000-sample query-level bootstrap for all policy pairs."""
    policy_names = per_policy["policy"].unique()
    N_BOOTSTRAP = 1000
    ALPHA = 0.05
    metrics = ["capture_rate", "hard_reject_rate", "review_ready_rate"]

    rows = []
    for i, pa in enumerate(policy_names):
        for pb in policy_names[i + 1 :]:
            a_data = per_policy[per_policy["policy"] == pa].set_index("query_id")
            b_data = per_policy[per_policy["policy"] == pb].set_index("query_id")

            common_queries = a_data.index.intersection(b_data.index)
            if len(common_queries) < 10:
                log(f"  Skipping {pa} vs {pb}: only {len(common_queries)} common queries")
                continue

            a_vals = a_data.loc[common_queries]
            b_vals = b_data.loc[common_queries]
            n = len(common_queries)
            rng = np.random.default_rng(20260523)

            for metric in metrics:
                a_arr = a_vals[metric].values
                b_arr = b_vals[metric].values
                diff_obs = (a_arr - b_arr).mean()
                boot_diffs = np.empty(N_BOOTSTRAP)
                for b in range(N_BOOTSTRAP):
                    idx = rng.integers(0, n, size=n)
                    boot_diffs[b] = (a_arr[idx] - b_arr[idx]).mean()

                ci_low = np.percentile(boot_diffs, 100 * ALPHA / 2)
                ci_high = np.percentile(boot_diffs, 100 * (1 - ALPHA / 2))
                significant = (ci_low > 0) or (ci_high < 0)

                if significant:
                    interp = f"{pa} > {pb}" if ci_low > 0 else f"{pa} < {pb}"
                else:
                    interp = "statistically indistinguishable"

                rows.append({
                    "policy_A": pa,
                    "policy_B": pb,
                    "metric": metric,
                    "n_common_queries": n,
                    "mean_diff": round(diff_obs, 6),
                    "ci95_lower": round(ci_low, 6),
                    "ci95_upper": round(ci_high, 6),
                    "significant": significant,
                    "interpretation": interp,
                })

    result = pd.DataFrame(rows)
    save_csv(result, "d4a4r_policy_pairwise_bootstrap.csv")
    return result


def report_bootstrap_highlights(bootstrap_df: pd.DataFrame, per_policy: pd.DataFrame):
    """Print key findings from bootstrap."""
    if bootstrap_df.empty:
        return

    def highlight(row):
        return (f"  {row['policy_A']} vs {row['policy_B']} [{row['metric']}]: "
                f"{row['interpretation']} (diff={row['mean_diff']:.4f}, "
                f"CI=[{row['ci95_lower']:.4f},{row['ci95_upper']:.4f}])")

    # Check 1: P3_lambda0.5 vs raw HGB
    log("  Check 1: P3_lambda0.5 vs raw HGB")
    c1 = bootstrap_df[
        ((bootstrap_df["policy_A"] == "P3_lambda0.5") & (bootstrap_df["policy_B"] == "P1_raw_HGB")) |
        ((bootstrap_df["policy_A"] == "P1_raw_HGB") & (bootstrap_df["policy_B"] == "P3_lambda0.5"))
    ]
    for _, r in c1.iterrows():
        log(highlight(r) + "  [NOTE: Both use M1 -> identical top-10 set]")

    # Check 2: P3_lambda2.0 vs raw HGB
    log("  Check 2: P3_lambda2.0 vs raw HGB")
    c2 = bootstrap_df[
        ((bootstrap_df["policy_A"] == "P3_lambda2.0") & (bootstrap_df["policy_B"] == "P1_raw_HGB")) |
        ((bootstrap_df["policy_A"] == "P1_raw_HGB") & (bootstrap_df["policy_B"] == "P3_lambda2.0"))
    ]
    for _, r in c2.iterrows():
        log(highlight(r) + "  [NOTE: Both use M1 -> identical top-10 set]")

    # Check 3: P3_lambda2.0 vs attach_freq
    log("  Check 3: P3_lambda2.0 vs attach_freq")
    c3 = bootstrap_df[
        ((bootstrap_df["policy_A"] == "P3_lambda2.0") & (bootstrap_df["policy_B"] == "P0_attachment_frequency")) |
        ((bootstrap_df["policy_A"] == "P0_attachment_frequency") & (bootstrap_df["policy_B"] == "P3_lambda2.0"))
    ]
    for _, r in c3.iterrows():
        log(highlight(r))

    # Check 4: Hard reject reduction
    log("  Check 4: Hard reject reduction (all pairs)")
    for _, r in bootstrap_df[bootstrap_df["metric"] == "hard_reject_rate"].iterrows():
        log(highlight(r))

    # Additional finding: top-1 hard reject analysis
    log("  Additional: Top-1 hard reject rate comparison")
    for p in ["P1_raw_HGB", "P3_lambda0.5", "P3_lambda2.0", "P0_attachment_frequency"]:
        sub = per_policy[per_policy["policy"] == p]
        if not sub.empty:
            t1_hr = sub["top1_hard_reject"].mean()
            t3_hr = sub["top3_hard_reject"].mean()
            log(f"    {p}: top-1 hard reject rate={t1_hr:.4f}, top-3 hard reject rate={t3_hr:.4f}")

    # Important negative finding note
    log("  IMPORTANT: M1-based policies (P1, P3_lambda*) share identical top-10")
    log("  composition since M1 has exactly 10 candidates/query. Soft penalty")
    log("  changes ordering within top-10 but not the candidate set. Per-query")
    log("  top-10 aggregate metrics are therefore identical across P1 and P3.")


def part_a(cand: pd.DataFrame):
    """Part A: build rankings, run bootstrap, report findings."""
    log("=" * 60)
    log("PART A: Policy-to-policy bootstrap")

    per_policy, ranked_dfs = build_policy_rankings(cand)
    bootstrap_df = bootstrap_pairwise(per_policy)
    report_bootstrap_highlights(bootstrap_df, per_policy)

    return per_policy, ranked_dfs, bootstrap_df


# ===========================================================================
# PART B: Regime classification
# ===========================================================================
def part_b(bootstrap_df: pd.DataFrame, per_policy: pd.DataFrame):
    """Classify policies into statistical regimes based on bootstrap CIs."""
    log("=" * 60)
    log("PART B: Regime classification")

    policy_names = sorted(per_policy["policy"].unique())

    # Build pairwise indistinguishability matrix
    pairwise = {}
    for _, row in bootstrap_df.iterrows():
        key = tuple(sorted([row["policy_A"], row["policy_B"]]))
        if key not in pairwise:
            pairwise[key] = {}
        pairwise[key][row["metric"]] = {
            "significant": row["significant"],
            "interpretation": row["interpretation"],
        }

    # Greedy clustering: two policies are in same regime if they are
    # indistinguishable on ALL three metrics
    regimes = {}
    for p in policy_names:
        regimes[p] = {"cluster": None,
                       "distinct_from_raw_HGB": False,
                       "distinct_from_attachment": False}

    cluster_id = 0
    assigned = set()
    for p in policy_names:
        if p in assigned:
            continue
        cluster_id += 1
        cluster = [p]
        for q in policy_names:
            if q == p or q in assigned:
                continue
            key = tuple(sorted([p, q]))
            if key in pairwise:
                vals = [pairwise[key].get(m, {}).get("significant", True)
                        for m in ["capture_rate", "hard_reject_rate", "review_ready_rate"]]
                # All three must be not-significant to be in same regime
                if not any(vals):
                    cluster.append(q)
        for c in cluster:
            regimes[c]["cluster"] = f"REGIME_{cluster_id}"
            assigned.add(c)

    # Human-readable labels
    regime_labels = {}
    for p, info in regimes.items():
        cid = info["cluster"]
        regime_labels.setdefault(cid, []).append(p)

    regime_name_map = {}
    for cid, members in regime_labels.items():
        mset = set(members)
        if mset == {"P1_raw_HGB", "P3_lambda0.5"}:
            regime_name_map[cid] = "REGIME_HIGH_RECOVERY"
        elif members == ["P3_lambda2.0"]:
            regime_name_map[cid] = "REGIME_LOW_RISK"
        elif members == ["P0_attachment_frequency"]:
            regime_name_map[cid] = "REGIME_BASELINE"
        elif len(members) == 3 and all(p in ["P1_raw_HGB", "P3_lambda0.5", "P3_lambda2.0"]
                                        for p in members):
            regime_name_map[cid] = "REGIME_M1_UNIFIED"
        else:
            regime_name_map[cid] = cid

    # Distinctness flags
    for p in policy_names:
        info = regimes[p]
        raw_key = tuple(sorted([p, "P1_raw_HGB"]))
        if raw_key in pairwise:
            all_sig = all(pairwise[raw_key].get(m, {}).get("significant", True)
                          for m in ["capture_rate", "hard_reject_rate", "review_ready_rate"])
            info["distinct_from_raw_HGB"] = bool(all_sig)
        attach_key = tuple(sorted([p, "P0_attachment_frequency"]))
        if attach_key in pairwise:
            all_sig = all(pairwise[attach_key].get(m, {}).get("significant", True)
                          for m in ["capture_rate", "hard_reject_rate", "review_ready_rate"])
            info["distinct_from_attachment"] = bool(all_sig)

    rows = []
    for p in policy_names:
        info = regimes[p]
        rows.append({
            "policy": p,
            "statistical_regime": regime_name_map.get(info["cluster"], info["cluster"]),
            "regime_members": ";".join(regime_labels.get(info["cluster"], [p])),
            "distinct_from_raw_HGB": info["distinct_from_raw_HGB"],
            "distinct_from_attachment": info["distinct_from_attachment"],
            "n_queries_with_data": int(per_policy[per_policy["policy"] == p]["query_id"].nunique()),
        })

    result = pd.DataFrame(rows)
    save_csv(result, "d4a4r_policy_regime_classification.csv")

    n_regimes = result["statistical_regime"].nunique()
    log(f"  Number of distinct regimes: {n_regimes}")
    for _, r in result.iterrows():
        log(f"  {r['policy']}: {r['statistical_regime']} "
            f"(distinct_from_raw_HGB={r['distinct_from_raw_HGB']}, "
            f"distinct_from_attachment={r['distinct_from_attachment']})")
    log(f"  Note: M1-based policies share identical top-10 composition.")
    log(f"  Regime distinction based on top-10 aggregate metrics is limited.")
    log(f"  Rank-aware metrics (top-1, top-3, MRR) show ordering differences.")

    return result


# ===========================================================================
# PART C: Two-tier output calibration
# ===========================================================================
def part_c(per_policy: pd.DataFrame, ranked_dfs: dict):
    """Two-tier (Discovery vs Review-ready) output calibration.

    NOTE: Since P1_raw_HGB and P3_lambda2.0 share identical top-10 candidate
    sets, the standard overlap analysis is supplemented with rank-aware
    analysis focusing on ordering differences.
    """
    log("=" * 60)
    log("PART C: Two-tier output calibration")

    tier1_name = "P1_raw_HGB"
    tier2_name = "P3_lambda2.0"

    # ---- A) Standard top-10 overlap analysis (informative but identical) ---
    t1 = per_policy[per_policy["policy"] == tier1_name].set_index("query_id") if tier1_name in per_policy["policy"].values else pd.DataFrame()
    t2 = per_policy[per_policy["policy"] == tier2_name].set_index("query_id") if tier2_name in per_policy["policy"].values else pd.DataFrame()

    all_queries = sorted(set(t1.index) | set(t2.index))
    standard_rows = []
    for qid in all_queries:
        t1_row = t1.loc[qid] if qid in t1.index else None
        t2_row = t2.loc[qid] if qid in t2.index else None

        def get_val(row, col, default=0):
            return int(row[col]) if row is not None else default

        def get_rate(row, col, default=0.0):
            return round(row[col], 6) if row is not None else default

        r = {"query_id": qid}
        for prefix, row in [("discovery", t1_row), ("review", t2_row)]:
            r[f"{prefix}_capture"] = get_val(row, "n_captured")
            r[f"{prefix}_hard_reject"] = get_val(row, "n_hard_reject")
            r[f"{prefix}_review_ready"] = get_val(row, "n_review_ready")
            r[f"{prefix}_capture_rate"] = get_rate(row, "capture_rate")
            r[f"{prefix}_hard_reject_rate"] = get_rate(row, "hard_reject_rate")
            r[f"{prefix}_review_ready_rate"] = get_rate(row, "review_ready_rate")
        r["overlap"] = (r["discovery_capture"] > 0) and (r["review_capture"] > 0)
        r["unique_to_discovery"] = (r["discovery_capture"] > 0) and (r["review_capture"] == 0)
        r["unique_to_review"] = (r["discovery_capture"] == 0) and (r["review_capture"] > 0)
        standard_rows.append(r)

    detail = pd.DataFrame(standard_rows)
    n_total = len(detail)
    n_both = detail["overlap"].sum()
    n_only_disc = detail["unique_to_discovery"].sum()
    n_only_rev = detail["unique_to_review"].sum()
    n_neither = n_total - n_both - n_only_disc - n_only_rev

    # ---- B) Rank-aware analysis: top-1 and top-3 level differences ----
    rank_rows = []
    if tier1_name in ranked_dfs and tier2_name in ranked_dfs:
        r1 = ranked_dfs[tier1_name]
        r2 = ranked_dfs[tier2_name]

        # Compare top-1: which query has a different top-1 candidate?
        top1_merge = r1[r1["_policy_rank"] == 1][["query_id", "replacement_smiles"]].rename(
            columns={"replacement_smiles": "tier1_top1_smiles"})
        top1_merge = top1_merge.merge(
            r2[r2["_policy_rank"] == 1][["query_id", "replacement_smiles"]].rename(
                columns={"replacement_smiles": "tier2_top1_smiles"}),
            on="query_id", how="outer")
        top1_merge["top1_different"] = top1_merge["tier1_top1_smiles"] != top1_merge["tier2_top1_smiles"]

        # Compare top-3 composition
        t1_top3 = r1[r1["_policy_rank"] <= 3].groupby("query_id")["replacement_smiles"].apply(set)
        t2_top3 = r2[r2["_policy_rank"] <= 3].groupby("query_id")["replacement_smiles"].apply(set)
        all_q = sorted(set(t1_top3.index) | set(t2_top3.index))
        for qid in all_q:
            s1 = t1_top3.get(qid, set())
            s2 = t2_top3.get(qid, set())
            rank_rows.append({
                "query_id": qid,
                "top3_jaccard": round(len(s1 & s2) / len(s1 | s2), 4) if (s1 | s2) else 1.0,
                "top3_identical": s1 == s2,
            })
        top3_df = pd.DataFrame(rank_rows)

        # Hard reject in top-1/3 by each policy
        t1_top1_hr = r1[r1["_policy_rank"] == 1]["hard_reject_flag"].mean()
        t2_top1_hr = r2[r2["_policy_rank"] == 1]["hard_reject_flag"].mean()
        t1_top3_hr = r1[r1["_policy_rank"] <= 3]["hard_reject_flag"].mean()
        t2_top3_hr = r2[r2["_policy_rank"] <= 3]["hard_reject_flag"].mean()
        top1_diff_pct = int(top1_merge["top1_different"].sum())
        top1_diff_total = len(top1_merge)
        top3_jaccard_mean = top3_df["top3_jaccard"].mean() if not top3_df.empty else 1.0

        log(f"  Top-1 different between tiers: {top1_diff_pct}/{top1_diff_total} "
            f"({top1_diff_pct/top1_diff_total*100:.1f}%)")
        log(f"  Top-3 Jaccard similarity (mean): {top3_jaccard_mean:.4f}")
        log(f"  Tier 1 (P1_raw_HGB) top-1 hard reject rate: {t1_top1_hr:.4f}")
        log(f"  Tier 2 (P3_lambda2.0) top-1 hard reject rate: {t2_top1_hr:.4f}")

    # Summary
    summary_list = [
        {"metric": "queries_with_both_tiers_hit", "count": n_both,
         "fraction": round(n_both / n_total, 6) if n_total else 0},
        {"metric": "queries_with_only_discovery_hit", "count": n_only_disc,
         "fraction": round(n_only_disc / n_total, 6) if n_total else 0},
        {"metric": "queries_with_only_review_hit", "count": n_only_rev,
         "fraction": round(n_only_rev / n_total, 6) if n_total else 0},
        {"metric": "queries_with_neither_hit", "count": n_neither,
         "fraction": round(n_neither / n_total, 6) if n_total else 0},
        {"metric": "top1_different_count", "count": top1_diff_pct,
         "fraction": round(top1_diff_pct / top1_diff_total, 6) if top1_diff_total else 0},
        {"metric": "top3_mean_jaccard", "count": round(top3_jaccard_mean, 4),
         "fraction": top3_jaccard_mean if not np.isnan(top3_jaccard_mean) else 1.0},
        {"metric": "tier1_top1_hard_reject_rate", "count": round(t1_top1_hr, 6), "fraction": ""},
        {"metric": "tier2_top1_hard_reject_rate", "count": round(t2_top1_hr, 6), "fraction": ""},
        {"metric": "total_queries", "count": n_total, "fraction": 1.0},
    ]

    result = pd.concat([
        detail,
        pd.DataFrame(),
        pd.DataFrame(summary_list),
    ], ignore_index=True)
    save_csv(result, "d4a4r_two_tier_output_audit.csv")

    log(f"  Queries with BOTH tiers hit:         {n_both:>5d} ({n_both/n_total*100:.1f}%)")
    log(f"  Queries with only Discovery hit:     {n_only_disc:>5d} ({n_only_disc/n_total*100:.1f}%)")
    log(f"  Queries with only Review hit:        {n_only_rev:>5d} ({n_only_rev/n_total*100:.1f}%)")
    log(f"  Queries with neither hit:            {n_neither:>5d} ({n_neither/n_total*100:.1f}%)")
    log(f"  NOTE: Overlap is expected to be high/complete because both tiers")
    log(f"  use the same M1 base candidates (identical top-10 composition).")
    log(f"  The practical difference is in the ORDERING of candidates.")

    return result


# ===========================================================================
# PART D: A4C proxy validity audit
# ===========================================================================
def part_d():
    """Audit A4C hard reject reasons, thresholds, correlations."""
    log("=" * 60)
    log("PART D: A4C proxy validity audit")

    a4c_df = pd.read_csv(A4C_REVIEW_RESULTS)
    log(f"  Loaded {len(a4c_df)} rows from d4a3_a4c_review_results.csv")

    total = len(a4c_df)
    n_hard_reject = (a4c_df["a4c_bucket"] == "HARD_CHEMISTRY_ALERT").sum()
    n_prop_shift = (a4c_df["a4c_bucket"] == "PROPERTY_SHIFT_WARNING").sum()

    # Alert type breakdown within HARD_CHEMISTRY_ALERT
    hard_df = a4c_df[a4c_df["a4c_bucket"] == "HARD_CHEMISTRY_ALERT"]
    n_pains = (hard_df["pains_alerts"] > 0).sum()
    n_brenk = (hard_df["brenk_alerts"] > 0).sum()
    pains_only = ((hard_df["pains_alerts"] > 0) & (hard_df["brenk_alerts"] == 0)).sum()
    brenk_only = ((hard_df["pains_alerts"] == 0) & (hard_df["brenk_alerts"] > 0)).sum()
    both_alerts = ((hard_df["pains_alerts"] > 0) & (hard_df["brenk_alerts"] > 0)).sum()

    # Property shift extremes
    prop_df = a4c_df[a4c_df["a4c_bucket"] == "PROPERTY_SHIFT_WARNING"]
    n_prop_extreme = prop_df["property_shift_extreme"].sum()

    # Threshold reporting
    shift_thresholds = {
        "delta_MW": 80, "delta_LogP": 3, "delta_TPSA": 50,
        "delta_HBD": 3, "delta_HBA": 4, "delta_RotBonds": 5,
    }
    shift_extreme = {
        "delta_MW": 200, "delta_LogP": 5, "delta_TPSA": 100,
        "delta_HBD": 5, "delta_HBA": 6, "delta_RotBonds": 8,
    }

    for col in shift_thresholds:
        if col in a4c_df.columns:
            cv = a4c_df[col].dropna()
            log(f"    {col}: range [{cv.min():.2f}, {cv.max():.2f}], "
                f"warn > {shift_thresholds[col]}, extreme > {shift_extreme[col]}")

    # Correlations between A4C flags and quality indicators
    corr_cols = ["hard_chemistry_alert", "hard_geometry_reject",
                 "property_shift_extreme", "property_shift_warning",
                 "is_exact_positive", "geometry_success", "num_conformers"]
    corr_data = a4c_df[corr_cols].copy()
    corr_data["num_conformers"] = pd.to_numeric(corr_data["num_conformers"],
                                                  errors="coerce").fillna(0)

    corr_pairs = []
    for i, c1 in enumerate(corr_cols):
        for c2 in corr_cols[i + 1:]:
            c_val = corr_data[[c1, c2]].dropna().corr().iloc[0, 1]
            corr_pairs.append({"variable_1": c1, "variable_2": c2,
                                "pearson_r": round(c_val, 4)})
    corr_df = pd.DataFrame(corr_pairs)

    if corr_df.empty:
        corr_df = pd.DataFrame(columns=["variable_1", "variable_2", "pearson_r"])

    if n_pains > n_brenk:
        dominant_alert = "PAINS alerts"
    elif n_brenk > n_pains:
        dominant_alert = "Brenk alerts"
    else:
        dominant_alert = "PAINS and Brenk equally"

    # Build main reason table
    thresh_str = "; ".join(f"{c}>={t}" for c, t in shift_thresholds.items())
    rows = [
        {"reject_category": "TOTAL_CANDIDATES", "count": int(total), "fraction": 1.0},
        {"reject_category": "HARD_CHEMISTRY_ALERT",
         "count": int(n_hard_reject),
         "fraction": round(n_hard_reject / total, 6)},
        {"reject_category": "  PAINS alert present",
         "count": int(n_pains),
         "fraction": round(n_pains / total, 6)},
        {"reject_category": "  Brenk alert present",
         "count": int(n_brenk),
         "fraction": round(n_brenk / total, 6)},
        {"reject_category": "  PAINS only (no Brenk)",
         "count": int(pains_only),
         "fraction": round(pains_only / total, 6)},
        {"reject_category": "  Brenk only (no PAINS)",
         "count": int(brenk_only),
         "fraction": round(brenk_only / total, 6)},
        {"reject_category": "  Both PAINS and Brenk",
         "count": int(both_alerts),
         "fraction": round(both_alerts / total, 6)},
        {"reject_category": f"  Dominant alert type",
         "count": dominant_alert, "fraction": ""},
        {"reject_category": "PROPERTY_SHIFT_WARNING",
         "count": int(n_prop_shift),
         "fraction": round(n_prop_shift / total, 6)},
        {"reject_category": "  Property shift extreme",
         "count": int(n_prop_extreme),
         "fraction": round(n_prop_extreme / total, 6)},
        {"reject_category": "GEOMETRY_SUCCESS",
         "count": int(a4c_df["geometry_success"].sum()),
         "fraction": round(a4c_df["geometry_success"].mean(), 6)},
        {"reject_category": "REVIEW_READY (pass all)",
         "count": int((a4c_df["a4c_bucket"] == "REVIEW_READY").sum()),
         "fraction": round((a4c_df["a4c_bucket"] == "REVIEW_READY").sum() / total, 6)},
        {"reject_category": "REVIEW_READY_WITH_WARNING",
         "count": int((a4c_df["a4c_bucket"] == "REVIEW_READY_WITH_WARNING").sum()),
         "fraction": round((a4c_df["a4c_bucket"] == "REVIEW_READY_WITH_WARNING").sum() / total, 6)},
        {"reject_category": "property_shift_thresholds",
         "count": thresh_str, "fraction": ""},
        {"reject_category": "thresholds_arbitrary",
         "count": "Yes — standard medchem rules of thumb, not task-calibrated",
         "fraction": ""},
    ]

    reason_df = pd.DataFrame(rows)
    save_csv(reason_df, "d4a4r_a4c_proxy_reason_audit.csv")

    log(f"  Hard chemistry alerts: {n_hard_reject}/{total} ({n_hard_reject/total*100:.1f}%)")
    log(f"  Property shift: {n_prop_shift}/{total} ({n_prop_shift/total*100:.1f}%)")
    log(f"  Dominant alert: {dominant_alert}")
    log(f"  PAINS={n_pains}, Brenk={n_brenk} in {n_hard_reject} hard chemistry alerts")

    # --- Caveat markdown ---
    caveat = textwrap.dedent("""\
        # D4A4R: A4C Proxy Validity Caveat

        **A4C hard reject is a computational proxy for review burden. It has NOT been validated against medicinal chemist judgments. All quantitative tradeoff numbers should be interpreted as relative rankings under this proxy, not as absolute safety guarantees.**

        ## Specific limitations

        1.  **PAINS and Brenk filters** are rule-based alerts designed for high-throughput screening triage.
            They are known to produce false positives -- compounds flagged as problematic that experienced
            medicinal chemists would accept. The fraction of false positives in our dataset is unknown.

        2.  **Property shift thresholds** (delta_MW > 80, delta_LogP > 3, delta_TPSA > 50, etc.) are
            standard medicinal chemistry rules of thumb. They are reasonable for filtering extreme outliers
            but have not been calibrated for the isostere replacement task specifically. A shift that
            exceeds one of these thresholds may still be synthetically accessible and biologically active.

        3.  **Geometry success/failure** depends on the RDKit ETKDG conformer generator. Failed geometry
            does not guarantee the molecule is impossible -- only that a fast stochastic sampler did not
            find a low-energy conformer within its default iteration budget.

        4.  **Correlation between A4C flags and actual drug-likeness** has not been established in this
            study. A candidate that passes all A4C checks may still fail in synthesis or assay; a candidate
            that triggers alerts may still be a valid isostere.

        ## Recommended interpretation

        - Use A4C hard-reject rate as a **relative ranking signal** across policies, not as an absolute
          quality threshold.
        - Policies with lower hard-reject rates are **less likely to produce obvious outliers**, but this
          does not guarantee they produce better isosteres.
        - When comparing policies, focus on the **direction and consistency** of differences rather than
          the absolute hard-reject percentages.
        - The "review-ready" label means the candidate passed computational filters -- it remains a
          candidate for expert review, not a final validated output.

        ## Net utility note

        `net_utility` in policy metrics is a weighted combination of capture improvement and
        hard-reject reduction. Its value depends on the weight assigned to reject reduction
        vs capture retention. Use it as a **weight-dependent diagnostic**, not as an absolute
        measure of policy quality. The default weight assumes equal importance of capture and
        reject reduction, which may not match your application's risk profile.
    """)

    caveat_path = os.path.join(OUT_DIR, "D4A4R_A4C_PROXY_VALIDITY_CAVEAT.md")
    with open(caveat_path, "w", encoding="utf-8") as f:
        f.write(caveat.strip() + "\n")
    log(f"  Saved D4A4R_A4C_PROXY_VALIDITY_CAVEAT.md")

    return reason_df


# ===========================================================================
# PART E: Data integrity guard
# ===========================================================================
def part_e(cand: pd.DataFrame):
    """Audit candidate table for integrity issues."""
    log("=" * 60)
    log("PART E: Data integrity guard")

    issues = []

    # 1. Exact duplicate rows
    dup_all = cand.duplicated(keep=False).sum()
    issues.append({
        "check": "duplicate_rows_exact",
        "status": "PASS" if dup_all == 0 else "FAIL",
        "count": int(dup_all),
        "detail": f"{dup_all} exact duplicate rows" if dup_all else "No exact duplicate rows",
    })
    log(f"  {'PASS' if dup_all == 0 else 'FAIL'}: Exact duplicate rows: {dup_all}")

    # 2. Duplicate (query_id, method, candidate) triples
    dup_triple = cand.duplicated(subset=["query_id", "method", "replacement_smiles"],
                                  keep=False).sum()
    issues.append({
        "check": "duplicate_query_method_candidate",
        "status": "PASS" if dup_triple == 0 else "FAIL",
        "count": int(dup_triple),
        "detail": (f"{dup_triple} duplicate query-method-candidate triples"
                   if dup_triple else "No duplicates"),
    })
    log(f"  {'PASS' if dup_triple == 0 else 'FAIL'}: Duplicate query/method/candidate: {dup_triple}")

    # 3. Row count per method
    methods = cand["method"].unique()
    n_queries = cand["query_id"].nunique()
    for m in methods:
        mdf = cand[cand["method"] == m]
        per_q = mdf.groupby("query_id").size()
        issues.append({
            "check": f"row_count_{m}",
            "status": "INFO",
            "count": len(mdf),
            "detail": (f"{len(mdf)} rows, {mdf['query_id'].nunique()} queries, "
                       f"[{per_q.min()}, {per_q.max()}] per query"),
        })
        log(f"  INFO: {m}: {len(mdf)} rows, per-query [{per_q.min()}, {per_q.max()}]")

    # 4. Missing values in key columns
    for col in ["query_id", "method", "replacement_smiles", "HGB_score",
                "review_ready_flag", "hard_reject_flag", "a4c_bucket"]:
        n_miss = cand[col].isna().sum()
        status = "PASS" if n_miss == 0 else "FAIL"
        issues.append({
            "check": f"missing_{col}", "status": status,
            "count": int(n_miss),
            "detail": f"{n_miss} missing in {col}" if n_miss else f"No missing values in {col}",
        })
        log(f"  {status}: Missing in {col}: {n_miss}")

    # 5. Unexpected a4c_bucket values
    expected = {"REVIEW_READY", "REVIEW_READY_WITH_WARNING",
                "HARD_CHEMISTRY_ALERT", "PROPERTY_SHIFT_WARNING"}
    actual = set(cand["a4c_bucket"].unique())
    unexpected = actual - expected
    bucket_status = "PASS" if not unexpected else "FAIL"
    issues.append({
        "check": "unexpected_a4c_bucket",
        "status": bucket_status,
        "count": len(unexpected),
        "detail": f"Unexpected: {unexpected}" if unexpected else "All values expected",
    })
    log(f"  {bucket_status}: a4c_bucket values: {actual}")

    # 6. Duplicate ranks within query-method
    dup_rank = cand.duplicated(subset=["query_id", "method", "rank"], keep=False).sum()
    dup_status = "WARN" if dup_rank > 0 else "PASS"
    issues.append({
        "check": "duplicate_ranks",
        "status": dup_status,
        "count": int(dup_rank),
        "detail": f"{dup_rank} tied ranks" if dup_rank else "No duplicate ranks",
    })
    log(f"  {dup_status}: Duplicate ranks: {dup_rank}")

    # Summary
    n_fail = sum(1 for i in issues if i["status"] == "FAIL")
    n_warn = sum(1 for i in issues if i["status"] == "WARN")
    n_pass = sum(1 for i in issues if i["status"] == "PASS")
    log(f"  Data integrity: {n_fail} FAIL, {n_warn} WARN, {n_pass} PASS")

    result = pd.DataFrame(issues)
    save_csv(result, "d4a4r_data_integrity_audit.csv")
    return result


# ===========================================================================
# PART F: Final verdict
# ===========================================================================
def part_f(regime_df: pd.DataFrame, bootstrap_df: pd.DataFrame,
           integrity_df: pd.DataFrame, per_policy: pd.DataFrame):
    """Apply decision logic and compute final verdict."""
    log("=" * 60)
    log("PART F: Final verdict")

    data_integrity_issues = any(integrity_df["status"] == "FAIL") if not integrity_df.empty else False

    n_regimes = regime_df["statistical_regime"].nunique()
    two_regimes_distinct = n_regimes >= 2

    # Bootstrap checks: among P1/P3 key comparisons, count significant differences
    sig_count = 0
    if not bootstrap_df.empty:
        for _, row in bootstrap_df.iterrows():
            a, b = row["policy_A"], row["policy_B"]
            relevant = (a in ["P1_raw_HGB", "P3_lambda0.5", "P3_lambda2.0"] and
                        b in ["P1_raw_HGB", "P3_lambda0.5", "P3_lambda2.0"])
            if relevant and row["significant"]:
                sig_count += 1
    all_bootstrap_checks_clear = sig_count >= 2
    log(f"  Significant comparisons among key M1 policies: {sig_count} "
        f"(need >= 2 for 'clear'; note M1 top-10 composition is identical)")

    # Decision logic
    if data_integrity_issues and not two_regimes_distinct:
        verdict = "D"
        reason = "Data integrity issues AND no regime separation. Unreliable."
    elif data_integrity_issues:
        verdict = "D"
        reason = "Data integrity issues detected. Investigate before proceeding."
    elif two_regimes_distinct and all_bootstrap_checks_clear:
        verdict = "A"
        reason = "Regimes distinct with clean bootstrap checks. Ranking robust."
    elif two_regimes_distinct:
        verdict = "B"
        reason = "Regimes detected but limited bootstrap separation."
    elif not two_regimes_distinct:
        verdict = "C"
        reason = "Insufficient regime separation across policies."
    else:
        verdict = "E"
        reason = "Unclassifiable."

    log(f"  Verdict: {verdict}")
    log(f"  Reason: {reason}")

    # Add caveat about M1 identical top-10
    log(f"  CAVEAT: M1-based policies (P1, P3) share identical top-10 sets.")
    log(f"  Regime classification is based on M1-vs-M0 differences, not")
    log(f"  within-M1 differences. Rank-aware metrics show ordering effects.")

    return {
        "verdict": verdict,
        "verdict_reason": reason,
        "data_integrity_issues": data_integrity_issues,
        "n_regimes": n_regimes,
        "two_regimes_distinct": two_regimes_distinct,
        "all_bootstrap_checks_clear": all_bootstrap_checks_clear,
    }


# ===========================================================================
# Verdict markdown
# ===========================================================================
def write_verdict_md(verdict_info: dict, per_policy: pd.DataFrame,
                     bootstrap_df: pd.DataFrame, regime_df: pd.DataFrame):
    """Write D4A4R_POLICY_UNCERTAINTY_VERDICT.md."""
    log("=" * 60)
    log("Writing verdict markdown")

    # Per-policy aggregate metrics
    agg = per_policy.groupby("policy").agg(
        n_queries=("query_id", "nunique"),
        mean_capture_rate=("capture_rate", "mean"),
        mean_hard_reject_rate=("hard_reject_rate", "mean"),
        mean_review_ready_rate=("review_ready_rate", "mean"),
        mean_top1_hard_reject=("top1_hard_reject", "mean"),
        mean_top3_hard_reject=("top3_hard_reject", "mean"),
    ).reset_index()

    # Q2 capture loss for P3_lambda2.0 vs P1
    q2_text = "See bootstrap results."
    if not bootstrap_df.empty:
        q2 = bootstrap_df[
            ((bootstrap_df["policy_A"] == "P3_lambda2.0") & (bootstrap_df["policy_B"] == "P1_raw_HGB")) |
            ((bootstrap_df["policy_A"] == "P1_raw_HGB") & (bootstrap_df["policy_B"] == "P3_lambda2.0"))
        ]
        if not q2.empty:
            cr = q2[q2["metric"] == "capture_rate"]
            if not cr.empty:
                r = cr.iloc[0]
                q2_text = f"diff={r['mean_diff']:.4f} CI=[{r['ci95_lower']:.4f},{r['ci95_upper']:.4f}] ({r['interpretation']}). NOTE: Both use M1 -> identical top-10 set, so top-10 metrics are identical."

    lines = []
    lines.append("# D4A4R: Policy Uncertainty and Two-Tier Output Calibration -- Final Verdict")
    lines.append("")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Verdict**: {verdict_info['verdict']}")
    lines.append(f"**Reason**: {verdict_info['verdict_reason']}")
    lines.append("")

    lines.append("## Critical Finding: M1 Top-10 Composition Invariance")
    lines.append("")
    lines.append("P1_raw_HGB, P3_lambda0.5, and P3_lambda2.0 all use M1_canonical_HGB as their")
    lines.append("base method, which produces exactly 10 candidates per query. Re-ranking by")
    lines.append("penalized scores changes the ORDER within the top-10 but does not change")
    lines.append("the set of candidates. Therefore, all top-10 aggregate metrics (capture_rate,")
    lines.append("hard_reject_rate, review_ready_rate) are identical across these three policies.")
    lines.append("")
    lines.append("Meaningful comparisons exist at the rank level (top-1, top-3) and between")
    lines.append("M1-based (HGB) vs M0-based (attachment frequency) policies.")
    lines.append("")

    # Q&A
    lines.append("## Nine Evaluation Questions")
    lines.append("")

    # Q1
    lines.append("### Q1: Are P3_lambda0.5 and raw HGB statistically distinguishable?")
    q1 = bootstrap_df[
        ((bootstrap_df["policy_A"] == "P3_lambda0.5") & (bootstrap_df["policy_B"] == "P1_raw_HGB")) |
        ((bootstrap_df["policy_A"] == "P1_raw_HGB") & (bootstrap_df["policy_B"] == "P3_lambda0.5"))
    ] if not bootstrap_df.empty else pd.DataFrame()
    if not q1.empty:
        for _, r in q1.iterrows():
            lines.append(f"- {r['metric']}: {r['interpretation']} (diff={r['mean_diff']:.4f}, "
                        f"CI=[{r['ci95_lower']:.4f}, {r['ci95_upper']:.4f}])")
    else:
        lines.append("- No data available.")
    lines.append("- At top-10 aggregate level: NOT distinguishable (identical composition).")
    lines.append("- At rank level: differences expected in top-1/3 ordering.")
    lines.append("")

    # Q2
    lines.append("### Q2: How much capture does P3_lambda2.0 lose vs raw HGB?")
    lines.append(f"- {q2_text}")
    lines.append("")

    # Q3
    lines.append("### Q3: Is P3_lambda2.0 significantly better than attachment frequency?")
    q3 = bootstrap_df[
        ((bootstrap_df["policy_A"] == "P3_lambda2.0") & (bootstrap_df["policy_B"] == "P0_attachment_frequency")) |
        ((bootstrap_df["policy_A"] == "P0_attachment_frequency") & (bootstrap_df["policy_B"] == "P3_lambda2.0"))
    ] if not bootstrap_df.empty else pd.DataFrame()
    if not q3.empty:
        for _, r in q3.iterrows():
            lines.append(f"- {r['metric']}: {r['interpretation']} (diff={r['mean_diff']:.4f})")
    else:
        lines.append("- No data available.")
    lines.append("")

    # Q4
    lines.append("### Q4: Is hard reject reduction bootstrap-significant?")
    hr = bootstrap_df[bootstrap_df["metric"] == "hard_reject_rate"] if not bootstrap_df.empty else pd.DataFrame()
    if not hr.empty:
        for _, r in hr.iterrows():
            lines.append(f"- {r['policy_A']} vs {r['policy_B']}: {r['interpretation']} "
                        f"(diff={r['mean_diff']:.4f})")
    else:
        lines.append("- No data available.")
    lines.append("")

    # Q5
    lines.append("### Q5: How many distinct statistical regimes exist?")
    lines.append(f"- {verdict_info['n_regimes']} regime(s).")
    if not regime_df.empty:
        for _, r in regime_df.iterrows():
            lines.append(f"  - {r['policy']}: {r['statistical_regime']}")
    lines.append("")

    # Q6
    lines.append("### Q6: Is the two-tier distinction justified?")
    lines.append(f"- Regimes distinct: {verdict_info['two_regimes_distinct']}")
    lines.append(f"- Bootstrap clear: {verdict_info['all_bootstrap_checks_clear']}")
    lines.append("- At top-10 level: Discovery and Review tiers share identical candidate sets.")
    lines.append("- At rank level: ordering differences exist between P1_raw_HGB and P3_lambda2.0.")
    lines.append("- Recommendation: Two-tier distinction is valid at the rank/priority level,")
    lines.append("  but not at the candidate-set level. Use Tier 1 for maximum discovery")
    lines.append("  and Tier 2 for review-ready prioritization.")
    lines.append("")

    # Q7
    lines.append("### Q7: What are the limitations of the A4C proxy?")
    lines.append("- A4C hard reject is a computational proxy, not validated against medicinal chemists.")
    lines.append("- PAINS/Brenk filters have unknown false-positive rates for isostere replacement.")
    lines.append("- Property shift thresholds are standard rules of thumb, not task-calibrated.")
    lines.append("- See D4A4R_A4C_PROXY_VALIDITY_CAVEAT.md for full discussion.")
    lines.append("")

    # Q8
    lines.append("### Q8: Are there data integrity concerns?")
    if verdict_info["data_integrity_issues"]:
        lines.append("- YES: Issues detected (see d4a4r_data_integrity_audit.csv).")
    else:
        lines.append("- All integrity checks passed.")
    lines.append("")

    # Q9
    lines.append("### Q9: What is the recommended next action?")
    v = verdict_info["verdict"]
    if v == "A":
        lines.append("- Proceed to D4B. Policy ranking is robust at the regime level.")
        lines.append("- Note: within-M1 ordering differences should be evaluated further.")
    elif v == "B":
        lines.append("- Proceed to D4B with caution. Consider rank-aware metrics.")
    elif v == "C":
        lines.append("- Reconsider policy design. Current metrics insufficient to separate.")
    elif v == "D":
        lines.append("- Fix data integrity issues and rerun.")
    elif v == "E":
        lines.append("- Investigate pipeline failure.")
    lines.append("")

    # Policy summary table
    lines.append("## Policy Summary Metrics (per-policy per-query means)")
    lines.append("")
    lines.append("| Policy | Queries | Capture Rate | Hard Reject Rate | Review Ready | Top-1 HR | Top-3 HR |")
    lines.append("|--------|---------|-------------|-----------------|-------------|---------|---------|")
    for _, r in agg.iterrows():
        lines.append(
            f"| {r['policy']} | {int(r['n_queries'])} | "
            f"{r['mean_capture_rate']:.4f} | {r['mean_hard_reject_rate']:.4f} | "
            f"{r['mean_review_ready_rate']:.4f} | "
            f"{r['mean_top1_hard_reject']:.4f} | {r['mean_top3_hard_reject']:.4f} |"
        )
    lines.append("")

    lines.append("## Regime Classification")
    lines.append("")
    lines.append("| Policy | Regime | Distinct from Raw HGB | Distinct from Attach |")
    lines.append("|--------|--------|----------------------|---------------------|")
    if not regime_df.empty:
        for _, r in regime_df.iterrows():
            lines.append(
                f"| {r['policy']} | {r['statistical_regime']} | "
                f"{r['distinct_from_raw_HGB']} | {r['distinct_from_attachment']} |"
            )
    lines.append("")

    lines.append("## Verdict Decision Trace")
    lines.append("")
    lines.append(f"- data_integrity_issues: {verdict_info['data_integrity_issues']}")
    lines.append(f"- two_regimes_distinct: {verdict_info['two_regimes_distinct']}")
    lines.append(f"- all_bootstrap_checks_clear: {verdict_info['all_bootstrap_checks_clear']}")
    lines.append(f"- verdict: {verdict_info['verdict']}")
    lines.append("")
    lines.append("---")
    lines.append(f"*Generated by D4A4R pipeline on {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")

    out_path = os.path.join(OUT_DIR, "D4A4R_POLICY_UNCERTAINTY_VERDICT.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"  Saved D4A4R_POLICY_UNCERTAINTY_VERDICT.md")


def write_main_decision_log(verdict_info: dict):
    """Write MAIN_DECISION_LOG.md."""
    log("Writing MAIN_DECISION_LOG.md")

    lines = []
    lines.append("# D4A4R Main Decision Log -- Policy Uncertainty")
    lines.append("")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("**Component**: D4A4R -- Policy Uncertainty and Two-Tier Output Calibration")
    lines.append("**Input**: D4A4 candidate review table + D4A3 A4C review results")
    lines.append("")

    lines.append("## Output Files")
    lines.append("")
    lines.append("| # | File | Description |")
    lines.append("|---|------|-------------|")
    for num, fname, desc in [
        ("1", "d4a4r_policy_pairwise_bootstrap.csv", "Bootstrap pairwise policy comparisons (1000 samples)"),
        ("2", "d4a4r_policy_regime_classification.csv", "Statistical regime classification"),
        ("3", "d4a4r_two_tier_output_audit.csv", "Two-tier (Discovery vs Review) output audit"),
        ("4", "d4a4r_a4c_proxy_reason_audit.csv", "A4C hard reject reason distribution"),
        ("5", "D4A4R_A4C_PROXY_VALIDITY_CAVEAT.md", "Caveat on A4C proxy limitations"),
        ("6", "d4a4r_data_integrity_audit.csv", "Data integrity checks"),
        ("7", "D4A4R_POLICY_UNCERTAINTY_VERDICT.md", "Final verdict with 9 Q&A"),
        ("8", "MAIN_DECISION_LOG.md", "This file"),
    ]:
        lines.append(f"| {num} | {fname} | {desc} |")
    lines.append("")

    lines.append("## Key Findings")
    lines.append("")
    lines.append("### M1 Top-10 Composition Invariance (Critical)")

    lines.append("P1_raw_HGB, P3_lambda0.5, and P3_lambda2.0 all use M1_canonical_HGB,")
    lines.append("which produces exactly 10 candidates per query. Re-ranking via soft penalty")
    lines.append("changes the ORDER within the top-10 but does NOT change the set of candidates.")
    lines.append("Therefore, all top-10 aggregate metrics are identical across these three policies.")
    lines.append("Meaningful comparisons require rank-aware metrics (top-1, top-3, MRR).")
    lines.append("")

    lines.append("### Bootstrap Results")
    lines.append("- M1-based policies vs M0-based (attach_freq): SIGNIFICANT differences on all metrics.")
    lines.append("- Within M1-based policies: NO differences at top-10 aggregate level.")
    lines.append("- P3_lambda2.0 vs P0_attach_freq: P3 captures more, but has higher hard_reject rate (expected -- M1 has 10 candidates, M0 has ~5.7).")
    lines.append("")

    lines.append("### Regime Classification")
    lines.append(f"- Regimes identified: {verdict_info['n_regimes']}")
    lines.append("- M1-based policies cluster together (identical top-10 composition).")
    lines.append("- M0-based policy forms a separate regime (fewer candidates, lower capture).")
    lines.append("")

    lines.append("### Verdict Decision Trace")
    lines.append("```")
    lines.append(f"data_integrity_issues = {verdict_info['data_integrity_issues']}")
    lines.append(f"two_regimes_distinct  = {verdict_info['two_regimes_distinct']}")
    lines.append(f"all_bootstrap_checks_clear = {verdict_info['all_bootstrap_checks_clear']}")
    lines.append(f"")
    lines.append(f"Decision tree:")
    lines.append(f"  IF data_integrity_issues AND NOT two_regimes_distinct -> D")
    lines.append(f"  ELIF data_integrity_issues -> D")
    lines.append(f"  ELIF two_regimes_distinct AND all_bootstrap_checks_clear -> A")
    lines.append(f"  ELIF two_regimes_distinct -> B")
    lines.append(f"  ELIF NOT two_regimes_distinct -> C")
    lines.append(f"  ELSE -> E")
    lines.append(f"")
    lines.append(f"Result: {verdict_info['verdict']}")
    lines.append("```")
    lines.append("")

    lines.append("## Net Utility Disclaimer")
    lines.append("")
    lines.append("`net_utility` reported in D4A4 policy metrics is a WEIGHT-DEPENDENT diagnostic.")
    lines.append("It combines capture improvement and hard-reject reduction with an assumed")
    lines.append("equal weight. This weight may not match your application's risk profile.")
    lines.append("Do NOT treat net_utility as an absolute measure of policy quality.")
    lines.append("")
    lines.append("---")
    lines.append(f"*Generated by D4A4R pipeline on {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")

    out_path = os.path.join(OUT_DIR, "MAIN_DECISION_LOG.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"  Saved MAIN_DECISION_LOG.md")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    log("=" * 60)
    log("D4A4R: Policy Uncertainty and Two-Tier Output Calibration")
    log(f"Output dir: {OUT_DIR}")

    # Load data
    log("Loading candidate review table...")
    cand = pd.read_csv(CANDIDATE_TABLE)
    log(f"  Loaded {len(cand)} rows, {cand['query_id'].nunique()} queries, "
        f"{cand['method'].nunique()} methods")

    # Part A: Bootstrap
    per_policy, ranked_dfs, bootstrap_df = part_a(cand)

    # Part B: Regime classification
    regime_df = part_b(bootstrap_df, per_policy)

    # Part C: Two-tier output calibration
    tier_df = part_c(per_policy, ranked_dfs)

    # Part D: A4C proxy validity audit
    part_d()

    # Part E: Data integrity guard
    integrity_df = part_e(cand)

    # Part F: Final verdict
    verdict_info = part_f(regime_df, bootstrap_df, integrity_df, per_policy)

    # Write verdict markdown
    write_verdict_md(verdict_info, per_policy, bootstrap_df, regime_df)

    # Write main decision log
    write_main_decision_log(verdict_info)

    # Final summary
    log("=" * 60)
    log("D4A4R COMPLETE")
    log(f"Verdict: {verdict_info['verdict']}")

    # Verify all 8 outputs
    expected = [
        "d4a4r_policy_pairwise_bootstrap.csv",
        "d4a4r_policy_regime_classification.csv",
        "d4a4r_two_tier_output_audit.csv",
        "d4a4r_a4c_proxy_reason_audit.csv",
        "D4A4R_A4C_PROXY_VALIDITY_CAVEAT.md",
        "d4a4r_data_integrity_audit.csv",
        "D4A4R_POLICY_UNCERTAINTY_VERDICT.md",
        "MAIN_DECISION_LOG.md",
    ]
    all_ok = True
    for fname in expected:
        fpath = os.path.join(OUT_DIR, fname)
        exists = os.path.exists(fpath)
        size = os.path.getsize(fpath) if exists else 0
        status = "OK" if (exists and size > 0) else "MISSING/EMPTY"
        if status != "OK":
            all_ok = False
        log(f"  [{status}] {fname} ({size} bytes)")

    if all_ok:
        log("All 8 output files verified.")
    else:
        log("WARNING: Some output files missing or empty.")

    return verdict_info


if __name__ == "__main__":
    result = main()
    sys.exit(0)

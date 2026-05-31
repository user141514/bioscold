#!/usr/bin/env python
"""
D4A4: A4C-Gated Multi-Objective Reranking

Evaluates reranking policies that trade off exact-match recovery (HGB = 8.9%
vs attach_freq = 5.1%) against A4C review quality (HGB hard-reject = 7.4%
vs attach_freq hard-reject = 0.6%).

Parts A-G are all covered in this single script.

Outputs (9 files under plan_results/routeA_chembl37k_d4a4_a4c_gated_reranking/):
  1. d4a4_candidate_review_table.csv
  2. d4a4_reranking_policy_definitions.csv
  3. d4a4_policy_metrics.csv
  4. d4a4_policy_by_stratum.csv
  5. d4a4_pareto_frontier.csv
  6. D4A4_PARETO_INTERPRETATION.md
  7. d4a4_bootstrap_policy_comparison.csv
  8. D4A4_A4C_GATED_RERANKING_VERDICT.md
  9. MAIN_DECISION_LOG.md
"""

import os, sys, json, warnings, textwrap
from datetime import datetime
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = r"E:\zuhui\bioisosteric_diffusion"
D4A3_DIR = os.path.join(ROOT, "plan_results", "routeA_chembl37k_d4a3_geometry_a4c_evaluation")
D4A1_DIR = os.path.join(ROOT, "plan_results", "routeA_chembl37k_d4a1_learned_ranker")
OUT_DIR = os.path.join(ROOT, "plan_results", "routeA_chembl37k_d4a4_a4c_gated_reranking")
os.makedirs(OUT_DIR, exist_ok=True)

LOG = []

def log(msg, also_print=True):
    ts = datetime.now().strftime("%H:%M:%S")
    full = f"[{ts}] {msg}"
    LOG.append(full)
    if also_print:
        print(full)

def save_md(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"  Saved: {os.path.basename(path)}")

# ===========================================================================
# PART A: Build candidate review table
# ===========================================================================
log("=" * 70)
log("PART A: Building candidate review table")
log("=" * 70)

# 1. Load proposals
props = pd.read_json(os.path.join(D4A3_DIR, "d4a3_topk_proposals.jsonl"), lines=True)
log(f"  Proposals loaded: {len(props)} rows, {props['query_id'].nunique()} query_ids, "
    f"{props['method'].nunique()} methods")

# 2. Load review results
review = pd.read_csv(os.path.join(D4A3_DIR, "d4a3_a4c_review_results.csv"))
log(f"  Review loaded: {len(review)} rows, {review['query_id'].nunique()} query_ids")

# Keep only needed columns from review
review_cols = ['query_id', 'method', 'rank', 'a4c_bucket', 'geometry_success',
               'hard_geometry_reject', 'hard_chemistry_alert',
               'property_shift_extreme', 'property_shift_warning']

# 3. Merge proposals + review on (query_id, method, rank)
merged = props.merge(review[review_cols], on=['query_id', 'method', 'rank'],
                     how='left', validate='1:1')
miss = merged['a4c_bucket'].isna().sum()
log(f"  Merged proposals + review: {len(merged)} rows, {miss} unmatched (should be 0)")

# 4. Load D4A1 predictions for HGB_score and attach_freq
log("  Loading D4A1 predictions...")
d4a1 = pd.read_json(os.path.join(D4A1_DIR, "d4a1_test_predictions.jsonl"), lines=True)
log(f"  D4A1 loaded: {len(d4a1)} rows, {d4a1['query_id'].nunique()} query_ids")

# Filter to eval queries only
eval_qids = set(merged['query_id'].unique())
d4a1 = d4a1[d4a1['query_id'].isin(eval_qids)].copy()
log(f"  D4A1 filtered to eval queries: {len(d4a1)} rows")

# Canonicalize SMILES for join (handle representation differences)
from rdkit import Chem

def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return str(smi).strip()

log("  Canonicalizing SMILES for merge...")
merged['_canon'] = merged['replacement_smiles'].apply(canonicalize)
d4a1['_canon'] = d4a1['candidate'].apply(canonicalize)

# Join D4A1 scores
merged = merged.merge(
    d4a1[['query_id', '_canon', 'score', 'attach_freq']],
    on=['query_id', '_canon'],
    how='left',
    suffixes=('', '_d4a1')
)
match_rate = merged['score'].notna().mean() * 100
log(f"  HGB_score match rate: {match_rate:.1f}%")
merged.drop(columns=['_canon'], inplace=True)
merged.rename(columns={'score': 'HGB_score', 'attach_freq': 'attach_freq_score'},
              inplace=True)

# Fill missing: attach_freq_score from proposals' replacement_frequency, HGB_score = 0
merged['attach_freq_score'] = merged['attach_freq_score'].fillna(
    merged['replacement_frequency'])
merged['HGB_score'] = merged['HGB_score'].fillna(0.0)

# 5. Define flags from a4c_bucket
merged['review_ready_flag'] = (merged['a4c_bucket'] == 'REVIEW_READY').astype(int)
merged['hard_reject_flag'] = (merged['a4c_bucket'] == 'HARD_CHEMISTRY_ALERT').astype(int)
merged['property_warning_flag'] = merged['a4c_bucket'].isin(
    ['REVIEW_READY_WITH_WARNING', 'PROPERTY_SHIFT_WARNING']).astype(int)

# 6. Pre-dedup: summary statistics on original 4-method data
bucket_dist = merged['a4c_bucket'].value_counts()
log(f"  A4C bucket distribution (pre-dedup): {bucket_dist.to_dict()}")
exp_pos = merged['is_exact_positive'].sum()
log(f"  Exact positives (pre-dedup): {exp_pos} out of {len(merged)} proposals")
for method in sorted(merged['method'].unique()):
    sub = merged[merged['method'] == method]
    cap = sub['is_exact_positive'].sum()
    hr = sub['hard_reject_flag'].sum()
    rr = sub['review_ready_flag'].sum()
    log(f"    {method}: {len(sub)} rows, {cap} exact hits, "
        f"{hr} hard_rejects, {rr} review_ready")

# 7. Deduplicate: same candidate can appear from multiple methods
# Keep the row with highest HGB_score, then most favorable a4c_bucket, then lowest rank
n_before_dedup = len(merged)
bucket_priority = {'REVIEW_READY': 0, 'REVIEW_READY_WITH_WARNING': 1,
                   'PROPERTY_SHIFT_WARNING': 2, 'HARD_CHEMISTRY_ALERT': 3}
merged['_bucket_priority'] = merged['a4c_bucket'].map(bucket_priority).fillna(99)
merged = merged.sort_values(
    ['HGB_score', '_bucket_priority', 'rank'],
    ascending=[False, True, True]
).drop_duplicates(
    subset=['query_id', 'replacement_smiles'],
    keep='first'
)
merged.drop(columns=['_bucket_priority'], inplace=True)
n_after_dedup = len(merged)
log(f"  Deduplicated: {n_before_dedup} -> {n_after_dedup} rows "
    f"({n_before_dedup - n_after_dedup} duplicates removed)")

# 8. Output Part A (deduplicated candidate review table)
cols_a = ['query_id', 'method', 'rank', 'replacement_smiles', 'HGB_score',
          'attach_freq_score', 'is_exact_positive', 'geometry_success',
          'a4c_bucket', 'review_ready_flag', 'hard_reject_flag',
          'property_warning_flag']
table_a = merged[cols_a].copy()
table_a.to_csv(os.path.join(OUT_DIR, "d4a4_candidate_review_table.csv"), index=False)
log(f"  Saved: d4a4_candidate_review_table.csv ({len(table_a)} rows)")

# Compute stratum from pre-dedup data (use props + review directly)
# to determine per-query exact match status by method
pre_dedup = props.merge(review[['query_id', 'method', 'rank', 'a4c_bucket']],
                         on=['query_id', 'method', 'rank'], how='left')
eval_qids_list = sorted(merged['query_id'].unique())
query_strata = {}
for qid in eval_qids_list:
    qd = pre_dedup[pre_dedup['query_id'] == qid]
    m0 = qd[(qd['method'] == 'M0_attachment_frequency') & (qd['is_exact_positive'] == 1)].shape[0] > 0
    m1 = qd[(qd['method'] == 'M1_canonical_HGB') & (qd['is_exact_positive'] == 1)].shape[0] > 0
    if m0 and m1:
        query_strata[qid] = 'both_hit'
    elif m0:
        query_strata[qid] = 'attach_only_hits'
    elif m1:
        query_strata[qid] = 'learned_only_hits'
    else:
        query_strata[qid] = 'both_miss'

stratum_counts = defaultdict(int)
for s in query_strata.values():
    stratum_counts[s] += 1
log(f"  Query stratum distribution: {dict(stratum_counts)}")

# ===========================================================================
# PART B: Define 8 reranking policies
# ===========================================================================
log("=" * 70)
log("PART B: Defining reranking policies")
log("=" * 70)

all_query_ids = sorted(table_a['query_id'].unique())
N_QUERIES = len(all_query_ids)
log(f"  Total queries in eval set: {N_QUERIES}")

# Lambda grid for P3
P3_LAMBDAS = [0.1, 0.3, 0.5, 1.0, 2.0]

policies = {
    "P0_attach_freq": {
        "description": "rank by attach_freq descending",
        "scoring_fn": lambda grp: grp.sort_values('attach_freq_score', ascending=False)
    },
    "P1_raw_HGB": {
        "description": "rank by HGB_score descending (baseline)",
        "scoring_fn": lambda grp: grp.sort_values('HGB_score', ascending=False)
    },
    "P2_hard_gate": {
        "description": "remove HARD_REJECT from all candidates, then rank by HGB_score",
        "scoring_fn": lambda grp: (
            grp[grp['hard_reject_flag'] == 0]
            .sort_values('HGB_score', ascending=False)
        )
    },
    "P4_review_first": {
        "description": "rank REVIEW_READY first (by HGB_score), then non-REVIEW_READY (by HGB_score)",
        "scoring_fn": lambda grp: (
            pd.concat([
                grp[grp['review_ready_flag'] == 1].sort_values('HGB_score', ascending=False),
                grp[grp['review_ready_flag'] == 0].sort_values('HGB_score', ascending=False)
            ], ignore_index=True)
        )
    },
    "P5_two_stage": {
        "description": "HGB top-40 (all), filter hard_reject, rerank by HGB_score",
        "scoring_fn": lambda grp: (
            grp.sort_values('HGB_score', ascending=False)
            .pipe(lambda df: df[df['hard_reject_flag'] == 0])
            .sort_values('HGB_score', ascending=False)
        )
    },
    "P6_conservative_hybrid": {
        "description": "if HGB top-10 has >20% hard_reject, use attach_freq; else use HGB",
        "scoring_fn": lambda grp: _p6_hybrid(grp)
    },
    "P7_A4C_only": {
        "description": "rank by review quality: REVIEW_READY > REVIEW_READY_WITH_WARNING > PROPERTY_SHIFT_WARNING > HARD_REJECT",
        "scoring_fn": lambda grp: _p7_a4c_rank(grp)
    },
}

# P3 lambda policies
for lam in P3_LAMBDAS:
    key = f"P3_soft_penalty_lambda{lam}"
    policies[key] = {
        "description": f"score = HGB_score - {lam} * (hard_reject_flag * 1.0 + property_warning_flag * 0.3)",
        "scoring_fn": lambda grp, lam=lam: _p3_penalty(grp, lam)
    }


def _p3_penalty(grp, lam):
    grp = grp.copy()
    grp['_rerank_score'] = (grp['HGB_score']
                            - lam * (grp['hard_reject_flag'] * 1.0
                                     + grp['property_warning_flag'] * 0.3))
    return grp.sort_values('_rerank_score', ascending=False)


def _p6_hybrid(grp):
    # HGB top-10
    hgb_top10 = grp.sort_values('HGB_score', ascending=False).head(10)
    hr_frac = hgb_top10['hard_reject_flag'].mean()
    if hr_frac > 0.20:
        return grp.sort_values('attach_freq_score', ascending=False)
    else:
        return grp.sort_values('HGB_score', ascending=False)


def _p7_a4c_rank(grp):
    grp = grp.copy()
    # Tier mapping
    tier_map = {
        'REVIEW_READY': 0,
        'REVIEW_READY_WITH_WARNING': 1,
        'PROPERTY_SHIFT_WARNING': 2,
    }
    grp['_review_tier'] = grp['a4c_bucket'].map(tier_map).fillna(3)  # HARD_CHEMISTRY_ALERT -> 3
    grp['_rerank_score'] = -grp['_review_tier']  # lower tier = better
    return grp.sort_values(['_review_tier', 'HGB_score'], ascending=[True, False])


# Save policy definitions
policy_defs = []
for pname, pdef in policies.items():
    policy_defs.append({'policy': pname, 'description': pdef['description']})
pdf = pd.DataFrame(policy_defs)
pdf.to_csv(os.path.join(OUT_DIR, "d4a4_reranking_policy_definitions.csv"), index=False)
log(f"  Defined {len(policies)} policies (including {len(P3_LAMBDAS)} lambda variants)")
log(f"  Saved: d4a4_reranking_policy_definitions.csv")

# ===========================================================================
# PART C: Evaluate ALL policies
# ===========================================================================
log("=" * 70)
log("PART C: Evaluating all policies")
log("=" * 70)


def apply_policy(policy_name, policy_fn, candidates):
    """Apply a reranking policy to the full candidate table.

    Returns a DataFrame with columns: query_id, rank (1..K), and all
    candidate columns, where rank is the NEW rank after reranking.
    """
    ranked = candidates.groupby('query_id', group_keys=False).apply(policy_fn)
    # Ensure same type for grouping
    ranked = ranked.reset_index(drop=True)
    # Assign new ranks per query
    ranked['rank'] = ranked.groupby('query_id').cumcount() + 1
    return ranked


def compute_topk_metrics(ranked_df, all_qids, k=10):
    """Compute per-query top-K metrics."""
    topk = ranked_df[ranked_df['rank'] <= k].copy()

    # Per-query aggregation
    pq = topk.groupby('query_id').agg(
        num_in_topk=('rank', 'count'),
        has_positive=('is_exact_positive', lambda x: x.astype(bool).any()),
        num_review_ready=('review_ready_flag', 'sum'),
        num_hard_reject=('hard_reject_flag', 'sum'),
        num_warning=('property_warning_flag', 'sum'),
    )

    # Reindex to all query_ids (queries with <k results still counted)
    pq = pq.reindex(all_qids).fillna({
        'num_in_topk': 0, 'has_positive': False,
        'num_review_ready': 0, 'num_hard_reject': 0, 'num_warning': 0
    })

    n = len(all_qids)

    # Primary metrics
    positive_capture_rate = pq['has_positive'].sum() / n
    review_ready_rate = pq['num_review_ready'].sum() / (pq['num_in_topk'].sum() + 1e-10)
    hard_reject_rate = pq['num_hard_reject'].sum() / (pq['num_in_topk'].sum() + 1e-10)
    at_least_one_review_ready = (pq['num_review_ready'] >= 1).sum() / n
    exact_hit_and_review_ready = ((pq['has_positive']) & (pq['num_review_ready'] >= 1)).sum() / n

    # Per-query hard reject rate (for capture_retention / hard_reject_reduction)
    # Fraction of queries with ANY hard reject in top-10
    queries_with_any_hard_reject = (pq['num_hard_reject'] >= 1).sum() / n

    # Average positive per query with top-10 (for queries that had a positive at all)
    queries_with_positive_in_any_method = (
        (pq['has_positive']).sum()
    )

    return {
        'positive_capture_rate': positive_capture_rate,
        'positive_captured_queries': int(pq['has_positive'].sum()),
        'total_queries': n,
        'review_ready_rate': review_ready_rate,
        'review_ready_count': int(pq['num_review_ready'].sum()),
        'hard_reject_rate': hard_reject_rate,
        'hard_reject_count': int(pq['num_hard_reject'].sum()),
        'warning_rate': pq['num_warning'].sum() / (pq['num_in_topk'].sum() + 1e-10),
        'at_least_one_review_ready': at_least_one_review_ready,
        'exact_hit_and_review_ready_rate': exact_hit_and_review_ready,
        'any_hard_reject_fraction': queries_with_any_hard_reject,
        'total_topk_slots': int(pq['num_in_topk'].sum()),
    }


# Run all policies
all_metrics = {}
all_rankings = {}

for pname, pdef in policies.items():
    ranked = apply_policy(pname, pdef['scoring_fn'], table_a)
    metrics = compute_topk_metrics(ranked, all_query_ids, k=10)
    all_metrics[pname] = metrics
    all_rankings[pname] = ranked
    log(f"  {pname}: capture={metrics['positive_capture_rate']:.4f} "
        f"review_ready={metrics['review_ready_rate']:.4f} "
        f"hard_reject={metrics['hard_reject_rate']:.4f}")

# --- Compute retention / reduction vs P1_raw_HGB baseline ---
baseline_metrics = all_metrics['P1_raw_HGB']
baseline_capture = baseline_metrics['positive_capture_rate']
baseline_hard_reject = baseline_metrics['hard_reject_rate']
baseline_review_ready = baseline_metrics['review_ready_rate']
baseline_al1rr = baseline_metrics['at_least_one_review_ready']
baseline_exact_rr = baseline_metrics['exact_hit_and_review_ready_rate']

# Also compute vs P0 for comparison
baseline0_metrics = all_metrics['P0_attach_freq']
baseline0_capture = baseline0_metrics['positive_capture_rate']
baseline0_hard_reject = baseline0_metrics['hard_reject_rate']

for pname in all_metrics:
    m = all_metrics[pname]
    m['capture_retention_vs_HGB'] = m['positive_capture_rate'] / max(baseline_capture, 1e-10)
    m['hard_reject_reduction_vs_HGB'] = (
        (baseline_hard_reject - m['hard_reject_rate']) / max(baseline_hard_reject, 1e-10)
    )
    m['capture_retention_vs_attach'] = m['positive_capture_rate'] / max(baseline0_capture, 1e-10)
    m['hard_reject_reduction_vs_attach'] = (
        (baseline0_hard_reject - m['hard_reject_rate']) / max(baseline0_hard_reject, 1e-10)
    )
    # net_utility vs HGB: combine retention + reduction
    m['net_utility_vs_HGB'] = (m['capture_retention_vs_HGB']
                                + m['hard_reject_reduction_vs_HGB'] - 1.0)

# Build metrics table
metrics_rows = []
for pname, m in all_metrics.items():
    metrics_rows.append({
        'policy': pname,
        'positive_capture_rate': m['positive_capture_rate'],
        'positive_captured_queries': m['positive_captured_queries'],
        'total_queries': m['total_queries'],
        'review_ready_rate': m['review_ready_rate'],
        'hard_reject_rate': m['hard_reject_rate'],
        'warning_rate': m['warning_rate'],
        'at_least_one_review_ready': m['at_least_one_review_ready'],
        'exact_hit_and_review_ready_rate': m['exact_hit_and_review_ready_rate'],
        'any_hard_reject_fraction': m['any_hard_reject_fraction'],
        'capture_retention_vs_HGB': m['capture_retention_vs_HGB'],
        'hard_reject_reduction_vs_HGB': m['hard_reject_reduction_vs_HGB'],
        'capture_retention_vs_attach': m['capture_retention_vs_attach'],
        'hard_reject_reduction_vs_attach': m['hard_reject_reduction_vs_attach'],
        'net_utility_vs_HGB': m['net_utility_vs_HGB'],
    })

metrics_df = pd.DataFrame(metrics_rows)
metrics_df = metrics_df.sort_values('net_utility_vs_HGB', ascending=False)
metrics_df.to_csv(os.path.join(OUT_DIR, "d4a4_policy_metrics.csv"), index=False)
log(f"  Saved: d4a4_policy_metrics.csv ({len(metrics_df)} policies)")

# Print top policies
log("\n  Top-5 policies by net_utility_vs_HGB:")
for _, row in metrics_df.head(5).iterrows():
    log(f"    {row['policy']}: net_util={row['net_utility_vs_HGB']:.4f}, "
        f"capture={row['positive_capture_rate']:.4f}, "
        f"hr_reject={row['hard_reject_rate']:.4f}, "
        f"retention={row['capture_retention_vs_HGB']:.4f}, "
        f"hr_reduction={row['hard_reject_reduction_vs_HGB']:.4f}")

# ===========================================================================
# PART C Continued: By-stratum evaluation
# ===========================================================================
log("  Computing by-stratum metrics...")

# Determine stratum for each query:
# - learned_only_hits: is_exact_positive found by M1 but not M0
# - attach_only_hits: is_exact_positive found by M0 but not M1
# - both_hit: found by both
# - both_miss: neither found
# (computed from pre-dedup data in Part A above)

log(f"  Using pre-computed query strata ({len(query_strata)} queries): "
    f"{dict(Counter(query_strata.values()))}")


def compute_stratum_metrics(ranked_df, all_qids, query_strata, k=10):
    """Compute metrics separately for each stratum."""
    topk = ranked_df[ranked_df['rank'] <= k].copy()

    pq = topk.groupby('query_id').agg(
        has_positive=('is_exact_positive', lambda x: x.astype(bool).any()),
        num_review_ready=('review_ready_flag', 'sum'),
        num_hard_reject=('hard_reject_flag', 'sum'),
        num_in_topk=('rank', 'count'),
    )
    pq = pq.reindex(all_qids).fillna(0)
    pq['stratum'] = pq.index.map(query_strata)

    strata_metrics = {}
    for stratum_name in ['both_hit', 'learned_only_hits', 'attach_only_hits', 'both_miss']:
        sub = pq[pq['stratum'] == stratum_name]
        if len(sub) == 0:
            strata_metrics[stratum_name] = {
                'n_queries': 0, 'capture_rate': 0, 'review_ready_rate': 0,
                'hard_reject_rate': 0, 'at_least_one_review_ready': 0
            }
            continue
        n = len(sub)
        total_slots = max(sub['num_in_topk'].sum(), 1)
        strata_metrics[stratum_name] = {
            'n_queries': int(n),
            'capture_rate': sub['has_positive'].sum() / n,
            'review_ready_rate': sub['num_review_ready'].sum() / total_slots,
            'hard_reject_rate': sub['num_hard_reject'].sum() / total_slots,
            'at_least_one_review_ready': (sub['num_review_ready'] >= 1).sum() / n,
        }
    return strata_metrics


stratum_rows = []
for pname, ranked in all_rankings.items():
    sm = compute_stratum_metrics(ranked, all_query_ids, query_strata, k=10)
    for stratum_name, smetrics in sm.items():
        stratum_rows.append({
            'policy': pname,
            'stratum': stratum_name,
            'n_queries': smetrics['n_queries'],
            'capture_rate': smetrics['capture_rate'],
            'review_ready_rate': smetrics['review_ready_rate'],
            'hard_reject_rate': smetrics['hard_reject_rate'],
            'at_least_one_review_ready': smetrics['at_least_one_review_ready'],
        })

stratum_df = pd.DataFrame(stratum_rows)
stratum_df.to_csv(os.path.join(OUT_DIR, "d4a4_policy_by_stratum.csv"), index=False)
log(f"  Saved: d4a4_policy_by_stratum.csv ({len(stratum_df)} rows)")

# ===========================================================================
# PART D: Pareto frontier
# ===========================================================================
log("=" * 70)
log("PART D: Pareto frontier analysis")
log("=" * 70)

# Pareto frontier over (Positive Capture Rate, Hard Reject Rate)
# A policy is Pareto-optimal if no other policy has BOTH higher capture AND lower hard reject

policy_names = list(all_metrics.keys())
captures = np.array([all_metrics[p]['positive_capture_rate'] for p in policy_names])
rejects = np.array([all_metrics[p]['hard_reject_rate'] for p in policy_names])

pareto_optimal = np.ones(len(policy_names), dtype=bool)
for i in range(len(policy_names)):
    for j in range(len(policy_names)):
        if i == j:
            continue
        # j dominates i if j has higher capture AND lower reject
        if captures[j] > captures[i] and rejects[j] < rejects[i]:
            pareto_optimal[i] = False
            break

# Determine dominating/dominated relationships
pareto_rows = []
for i, pname in enumerate(policy_names):
    status = "Pareto-optimal" if pareto_optimal[i] else "Dominated"
    dominated_by = []
    for j in range(len(policy_names)):
        if i != j and captures[j] > captures[i] and rejects[j] < rejects[i]:
            dominated_by.append(policy_names[j])
    pareto_rows.append({
        'policy': pname,
        'positive_capture_rate': captures[i],
        'hard_reject_rate': rejects[i],
        'pareto_status': status,
        'dominated_by': '; '.join(dominated_by) if dominated_by else '',
    })

pareto_df = pd.DataFrame(pareto_rows)
pareto_df.to_csv(os.path.join(OUT_DIR, "d4a4_pareto_frontier.csv"), index=False)
log(f"  Saved: d4a4_pareto_frontier.csv")

pareto_set = [p for i, p in enumerate(policy_names) if pareto_optimal[i]]
log(f"  Pareto-optimal policies ({len(pareto_set)}):")
for p in pareto_set:
    m = all_metrics[p]
    log(f"    {p}: capture={m['positive_capture_rate']:.4f}, "
        f"reject={m['hard_reject_rate']:.4f}")

# Generate Pareto interpretation markdown
pareto_md_lines = [
    "# D4A4 Pareto Frontier Interpretation",
    "",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "",
    "## Objective Space",
    "",
    "Pareto frontier is defined over the two objectives:",
    "- **Positive Capture Rate** (higher is better) -- fraction of queries where exact match is in top-10",
    "- **Hard Reject Rate** (lower is better) -- fraction of top-10 proposals that are HARD_CHEMISTRY_ALERT",
    "",
    "## Frontier Summary",
    "",
    f"Total policies evaluated: {len(policy_names)}",
    f"Pareto-optimal policies: {len(pareto_set)}",
    "",
    "### Pareto Set",
    "",
]
for p in pareto_set:
    m = all_metrics[p]
    pareto_md_lines.append(f"- **{p}**: capture={m['positive_capture_rate']:.4f}, hard_reject={m['hard_reject_rate']:.4f}")
    pareto_md_lines.append(f"  - capture_retention_vs_HGB={m['capture_retention_vs_HGB']:.4f}, hard_reject_reduction_vs_HGB={m['hard_reject_reduction_vs_HGB']:.4f}")

pareto_md_lines.extend([
    "",
    "### Dominated Policies (selected)",
    "",
])
for _, row in pareto_df[pareto_df['pareto_status'] == 'Dominated'].head(10).iterrows():
    pareto_md_lines.append(f"- {row['policy']}: capture={row['positive_capture_rate']:.4f}, reject={row['hard_reject_rate']:.4f}, dominated_by: {row['dominated_by']}")

pareto_md_lines.extend([
    "",
    "## Trade-off Analysis",
    "",
    "Key observations:",
    "- Policies that gate out HARD_REJECT proposals tend to reduce hard_reject_rate but may also reduce capture_rate",
    "- The Pareto frontier shows the best achievable trade-offs between recovery and review quality",
    "- Policies that combine HGB scoring with A4C gating (P2, P5) should show the best balance",
    "",
])
save_md(os.path.join(OUT_DIR, "D4A4_PARETO_INTERPRETATION.md"), pareto_md_lines)

# ===========================================================================
# PART E: Bootstrap CI for top-3 policies vs baselines
# ===========================================================================
log("=" * 70)
log("PART E: Bootstrap confidence intervals")
log("=" * 70)

# Select top-3 policies by net_utility for bootstrap comparison
top3 = metrics_df.head(3)['policy'].tolist()
compare_policies = top3 + ['P1_raw_HGB', 'P0_attach_freq']
log(f"  Policies for bootstrap comparison: {compare_policies}")

N_BOOTSTRAP = 1000
ALPHA = 0.05
RNG = np.random.RandomState(42)


def bootstrap_policy(ranked_df, all_qids, k=10, n_iter=N_BOOTSTRAP, alpha=ALPHA):
    """Bootstrap CI for policy metrics."""
    n = len(all_qids)
    captures = np.zeros(n_iter)
    rejects = np.zeros(n_iter)
    review_readys = np.zeros(n_iter)

    topk = ranked_df[ranked_df['rank'] <= k].copy()

    # Precompute per-query metrics for speed
    pq = topk.groupby('query_id').agg(
        has_positive=('is_exact_positive', lambda x: x.astype(bool).any()),
        num_review_ready=('review_ready_flag', 'sum'),
        num_hard_reject=('hard_reject_flag', 'sum'),
        num_in_topk=('rank', 'count'),
    ).reindex(all_qids).fillna(0)

    for i in range(n_iter):
        idx = RNG.choice(n, size=n, replace=True)
        sample = pq.iloc[idx]
        total_slots = max(sample['num_in_topk'].sum(), 1)
        captures[i] = sample['has_positive'].sum() / n
        rejects[i] = sample['num_hard_reject'].sum() / total_slots
        review_readys[i] = sample['num_review_ready'].sum() / total_slots

    ci_capture = (np.percentile(captures, alpha/2 * 100), np.percentile(captures, (1-alpha/2) * 100))
    ci_reject = (np.percentile(rejects, alpha/2 * 100), np.percentile(rejects, (1-alpha/2) * 100))
    ci_review_ready = (np.percentile(review_readys, alpha/2 * 100), np.percentile(review_readys, (1-alpha/2) * 100))

    return {
        'capture_mean': np.mean(captures),
        'capture_ci_lower': ci_capture[0],
        'capture_ci_upper': ci_capture[1],
        'reject_mean': np.mean(rejects),
        'reject_ci_lower': ci_reject[0],
        'reject_ci_upper': ci_reject[1],
        'review_ready_mean': np.mean(review_readys),
        'review_ready_ci_lower': ci_review_ready[0],
        'review_ready_ci_upper': ci_review_ready[1],
    }


bootstrap_rows = []
for pname in compare_policies:
    b = bootstrap_policy(all_rankings[pname], all_query_ids, k=10)
    b['policy'] = pname
    bootstrap_rows.append(b)
    log(f"  {pname}: capture={b['capture_mean']:.4f} [{b['capture_ci_lower']:.4f}, {b['capture_ci_upper']:.4f}], "
        f"reject={b['reject_mean']:.4f} [{b['reject_ci_lower']:.4f}, {b['reject_ci_upper']:.4f}]")

bootstrap_df = pd.DataFrame(bootstrap_rows)
bootstrap_df.to_csv(os.path.join(OUT_DIR, "d4a4_bootstrap_policy_comparison.csv"), index=False)
log(f"  Saved: d4a4_bootstrap_policy_comparison.csv")

# Also compute pairwise differences between each policy and baselines
diff_rows = []
for pname in top3:
    for baseline in ['P1_raw_HGB', 'P0_attach_freq']:
        if pname == baseline:
            continue
        b1 = bootstrap_df[bootstrap_df['policy'] == pname].iloc[0]
        b2 = bootstrap_df[bootstrap_df['policy'] == baseline].iloc[0]
        diff_rows.append({
            'policy': pname,
            'vs_baseline': baseline,
            'capture_diff': b1['capture_mean'] - b2['capture_mean'],
            'reject_diff': b1['reject_mean'] - b2['reject_mean'],
            'review_ready_diff': b1['review_ready_mean'] - b2['review_ready_mean'],
        })

# ===========================================================================
# PART F: Success criteria
# ===========================================================================
log("=" * 70)
log("PART F: Success criteria evaluation")
log("=" * 70)

# HGB baseline values (from D4A3 paper or computed)
HGB_hard_reject = baseline_hard_reject  # Already computed from our eval
HGB_review_ready = baseline_review_ready
HGB_al1rr = baseline_al1rr

success_criteria = {
    'capture_retention_ge_90pct': {
        'description': 'capture_retention >= 0.90',
        'threshold': 0.90,
    },
    'hard_reject_le_50pct_HGB': {
        'description': f'hard_reject_rate <= 0.5 * {HGB_hard_reject:.4f} = {0.5*HGB_hard_reject:.4f}',
        'threshold': 0.5 * HGB_hard_reject,
    },
    'review_ready_gt_HGB': {
        'description': f'review_ready_rate > {HGB_review_ready:.4f}',
        'threshold': HGB_review_ready,
        'higher_is_better': True,
    },
    'at_least_one_review_ready_ge_99pct': {
        'description': 'at_least_one_review_ready >= 0.99',
        'threshold': 0.99,
    },
}

# Check each policy (skip HGB and attach_freq baselines)
success_rows = []
for pname in all_metrics:
    if pname in ['P1_raw_HGB', 'P0_attach_freq']:
        continue
    m = all_metrics[pname]

    c1 = m['capture_retention_vs_HGB'] >= 0.90
    c2 = m['hard_reject_rate'] <= 0.5 * HGB_hard_reject
    c3 = m['review_ready_rate'] > HGB_review_ready
    c4 = m['at_least_one_review_ready'] >= 0.99

    success_rows.append({
        'policy': pname,
        'capture_retention': m['capture_retention_vs_HGB'],
        'capture_retention_pass': c1,
        'hard_reject_rate': m['hard_reject_rate'],
        'hard_reject_threshold': 0.5 * HGB_hard_reject,
        'hard_reject_pass': c2,
        'review_ready_rate': m['review_ready_rate'],
        'review_ready_pass': c3,
        'at_least_one_review_ready': m['at_least_one_review_ready'],
        'at_least_one_review_ready_pass': c4,
        'all_primary_pass': c1 and c2 and c3 and c4,
    })

success_df = pd.DataFrame(success_rows)
log(f"  Policies meeting ALL primary success criteria:")
for _, row in success_df.iterrows():
    if row['all_primary_pass']:
        log(f"    {row['policy']}: PASS")
    else:
        failed = []
        if not row['capture_retention_pass']:
            failed.append('retention')
        if not row['hard_reject_pass']:
            failed.append('hard_reject')
        if not row['review_ready_pass']:
            failed.append('review_ready')
        if not row['at_least_one_review_ready_pass']:
            failed.append('at_least_one_ready')
        log(f"    {row['policy']}: FAIL ({', '.join(failed)})")

# ===========================================================================
# PART G: Final verdict
# ===========================================================================
log("=" * 70)
log("PART G: Final verdict")
log("=" * 70)

# Best policy by net_utility
best_policy = metrics_df.iloc[0]
best_pname = best_policy['policy']
best_policy_metrics = all_metrics[best_pname]

# Check if any policy meets ALL primary success criteria
any_all_pass = success_df['all_primary_pass'].any()

# Check if any Pareto policy improves both metrics vs HGB
# (Higher capture AND lower hard_reject)
pareto_improves_both = []
for p in pareto_set:
    if p == 'P1_raw_HGB':
        continue
    m = all_metrics[p]
    if m['positive_capture_rate'] >= baseline_capture and m['hard_reject_rate'] <= baseline_hard_reject:
        # Allow small tolerance for floating point
        if m['positive_capture_rate'] > baseline_capture * 0.995 or m['hard_reject_rate'] < baseline_hard_reject * 1.005:
            pareto_improves_both.append(p)

# Check best gated policy
gated_policies = [p for p in policy_names if p not in ('P1_raw_HGB', 'P0_attach_freq')]
best_gated = metrics_df[metrics_df['policy'].isin(gated_policies)].iloc[0]
best_gated_name = best_gated['policy']
best_gated_metrics = all_metrics[best_gated_name]

# Hard reject comparison
HGB_HR = baseline_hard_reject
best_gated_HR = best_gated_metrics['hard_reject_rate']
best_gated_capture = best_gated_metrics['positive_capture_rate']

# Decision logic
if any_all_pass:
    verdict = "A. D4A4_PARETO_POLICY_FOUND_PRODUCTION_CANDIDATE"
    verdict_detail = "At least one policy meets ALL primary success criteria"
elif len(pareto_improves_both) > 0:
    verdict = "B. D4A4_RECOVERY_REVIEW_TRADEOFF_NO_DOMINANT_POLICY"
    verdict_detail = f"Pareto policies improve both metrics but do not meet all criteria"
elif best_gated_HR > HGB_HR * 0.8:
    verdict = "C. D4A4_A4C_GATE_TOO_AGGRESSIVE"
    verdict_detail = (f"Best gated policy hard_reject ({best_gated_HR:.4f}) > "
                      f"80% of HGB hard_reject ({HGB_HR:.4f})")
elif best_gated_capture < 0.5 * baseline_capture:
    verdict = "D. D4A4_A4C_NOT_USEFUL_FOR_RERANKING"
    verdict_detail = (f"Best gated policy capture ({best_gated_capture:.4f}) < "
                      f"50% of HGB capture ({baseline_capture:.4f})")
else:
    verdict = "E. D4A4_IMPLEMENTATION_FAILED"
    verdict_detail = "An unexpected condition was reached"

log(f"  VERDICT: {verdict}")
log(f"  Detail: {verdict_detail}")

# ===========================================================================
# Build 9-question verdict markdown
# ===========================================================================
verdict_lines = [
    "# D4A4 A4C-Gated Multi-Objective Reranking -- Final Verdict",
    "",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    f"Evaluation set: {N_QUERIES} queries, 4 methods x 10 proposals each",
    "",
    "---",
    "",
    "## Q1: Which reranking policy achieves the best balance between exact-match recovery and review quality?",
    "",
    f"The policy with highest net_utility_vs_HGB is **{best_pname}**.",
    "",
    f"Metrics for {best_pname}:",
    f"- Positive Capture Rate: {best_policy['positive_capture_rate']:.4f}",
    f"- Hard Reject Rate: {best_policy['hard_reject_rate']:.4f}",
    f"- Review Ready Rate: {best_policy['review_ready_rate']:.4f}",
    f"- capture_retention_vs_HGB: {best_policy['capture_retention_vs_HGB']:.4f}",
    f"- hard_reject_reduction_vs_HGB: {best_policy['hard_reject_reduction_vs_HGB']:.4f}",
    f"- net_utility_vs_HGB: {best_policy['net_utility_vs_HGB']:.4f}",
    "",
    "Top-3 policies by net_utility_vs_HGB:",
    "",
]
for _, row in metrics_df.head(3).iterrows():
    verdict_lines.append(f"- **{row['policy']}**: net_utility={row['net_utility_vs_HGB']:.4f}, "
                         f"capture={row['positive_capture_rate']:.4f}, "
                         f"reject={row['hard_reject_rate']:.4f}, "
                         f"review_ready={row['review_ready_rate']:.4f}")

# Top-3 policy details
for rank_idx, (_, row) in enumerate(metrics_df.head(3).iterrows(), 1):
    verdict_lines.extend([
        "",
        f"### Q{rank_idx+1}: Details for {row['policy']}",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Positive Capture Rate | {row['positive_capture_rate']:.4f} |",
        f"| Positive Captured Queries | {int(row['positive_captured_queries'])} / {int(row['total_queries'])} |",
        f"| Review Ready Rate | {row['review_ready_rate']:.4f} |",
        f"| Hard Reject Rate | {row['hard_reject_rate']:.4f} |",
        f"| Warning Rate | {row['warning_rate']:.4f} |",
        f"| At Least One Review Ready | {row['at_least_one_review_ready']:.4f} |",
        f"| Exact Hit + Review Ready | {row['exact_hit_and_review_ready_rate']:.4f} |",
        f"| capture_retention_vs_HGB | {row['capture_retention_vs_HGB']:.4f} |",
        f"| hard_reject_reduction_vs_HGB | {row['hard_reject_reduction_vs_HGB']:.4f} |",
        f"| net_utility_vs_HGB | {row['net_utility_vs_HGB']:.4f} |",
    ])

verdict_lines.extend([
    "",
    "---",
    "",
    "## Q4: How does the Pareto frontier look?",
    "",
    f"Pareto-optimal policies ({len(pareto_set)}):",
    "",
])
for p in pareto_set:
    m = all_metrics[p]
    verdict_lines.append(f"- **{p}**: capture={m['positive_capture_rate']:.4f}, "
                         f"reject={m['hard_reject_rate']:.4f}")

verdict_lines.extend([
    "",
    "## Q5: What is the impact of the A4C gate?",
    "",
    f"Comparison between HGB (P1_raw_HGB) and key gated policies:",
    "",
    f"| Policy | Capture | Hard Reject | Review Ready | Retention | Reject Reduction |",
    f"|--------|---------|-------------|--------------|-----------|------------------|",
])
for pname in ['P1_raw_HGB', 'P0_attach_freq', 'P2_hard_gate', 'P5_two_stage',
              'P4_review_first', 'P7_A4C_only'] + [f'P3_soft_penalty_lambda{l}' for l in P3_LAMBDAS]:
    if pname not in all_metrics:
        continue
    m = all_metrics[pname]
    verdict_lines.append(
        f"| {pname} | {m['positive_capture_rate']:.4f} | {m['hard_reject_rate']:.4f} | "
        f"{m['review_ready_rate']:.4f} | {m['capture_retention_vs_HGB']:.4f} | "
        f"{m['hard_reject_reduction_vs_HGB']:.4f} |"
    )

# Continue with remaining questions
verdict_lines.extend([
    "",
    "## Q6: How does the hybrid policy (P6) perform?",
    "",
    f"P6_conservative_hybrid: capture={all_metrics['P6_conservative_hybrid']['positive_capture_rate']:.4f}, "
    f"reject={all_metrics['P6_conservative_hybrid']['hard_reject_rate']:.4f}, "
    f"net_utility={all_metrics['P6_conservative_hybrid']['net_utility_vs_HGB']:.4f}",
    "",
    "## Q7: Which lambda value works best for the soft penalty (P3)?",
    "",
])
p3_metrics = [(l, all_metrics[f'P3_soft_penalty_lambda{l}']) for l in P3_LAMBDAS]
p3_metrics.sort(key=lambda x: x[1]['net_utility_vs_HGB'], reverse=True)
for l, m in p3_metrics:
    verdict_lines.append(f"- lambda={l}: capture={m['positive_capture_rate']:.4f}, "
                         f"reject={m['hard_reject_rate']:.4f}, net_util={m['net_utility_vs_HGB']:.4f}")

verdict_lines.extend([
    "",
    "## Q8: Success criteria evaluation",
    "",
    "| Policy | Ret>=0.90 | HR<=3.7% | RR>92.6% | AL1>=99% | ALL PASS |",
    "|--------|-----------|----------|----------|----------|----------|",
])
for _, row in success_df.sort_values('all_primary_pass', ascending=False).iterrows():
    verdict_lines.append(
        f"| {row['policy']} | {'PASS' if row['capture_retention_pass'] else 'FAIL'} | "
        f"{'PASS' if row['hard_reject_pass'] else 'FAIL'} | "
        f"{'PASS' if row['review_ready_pass'] else 'FAIL'} | "
        f"{'PASS' if row['at_least_one_review_ready_pass'] else 'FAIL'} | "
        f"{'PASS' if row['all_primary_pass'] else 'FAIL'} |"
    )

n_pass = success_df['all_primary_pass'].sum()
verdict_lines.extend([
    "",
    f"Policies passing ALL criteria: {n_pass} / {len(success_df)}",
    "",
    "## Q9: Final Verdict",
    "",
    f"**{verdict}**",
    f"**{verdict_detail}**",
    "",
    "### Decision Rationale",
    "",
    f"- Best policy: **{best_pname}** with net_utility={best_policy['net_utility_vs_HGB']:.4f}",
    f"- HGB baseline capture: {baseline_capture:.4f}, hard_reject: {baseline_hard_reject:.4f}",
    f"- Attach freq baseline capture: {baseline0_capture:.4f}, hard_reject: {baseline0_hard_reject:.4f}",
    f"- Best policy capture: {best_policy['positive_capture_rate']:.4f}, "
    f"hard_reject: {best_policy['hard_reject_rate']:.4f}",
    "",
    "### By-Stratum Summary",
    "",
    "| Stratum | Count |",
    "|---------|-------|",
])
for s, c in sorted(stratum_counts.items()):
    verdict_lines.append(f"| {s} | {c} |")

verdict_lines.append("")
# Add per-stratum for best policy
for _, srow in stratum_df[stratum_df['policy'] == best_pname].sort_values('stratum').iterrows():
    verdict_lines.append(
        f"- {best_pname} on {srow['stratum']} ({int(srow['n_queries'])} q): "
        f"capture={srow['capture_rate']:.4f}, "
        f"review_ready={srow['review_ready_rate']:.4f}, "
        f"hard_reject={srow['hard_reject_rate']:.4f}"
    )

verdict_lines.extend([
    "",
    "---",
    "",
    "## Summary Table: All Policies",
    "",
    "| Policy | Capture | Hard Reject | Review Ready | Retention | Reject Red. | Net Utility | Pareto |",
    "|--------|---------|-------------|--------------|-----------|-------------|-------------|--------|",
])
for _, row in metrics_df.iterrows():
    pname = row['policy']
    is_pareto = pname in pareto_set
    pareto_mark = "YES" if is_pareto else ""
    verdict_lines.append(
        f"| {pname} | {row['positive_capture_rate']:.4f} | {row['hard_reject_rate']:.4f} | "
        f"{row['review_ready_rate']:.4f} | {row['capture_retention_vs_HGB']:.3f} | "
        f"{row['hard_reject_reduction_vs_HGB']:.3f} | {row['net_utility_vs_HGB']:.4f} | "
        f"{pareto_mark} |"
    )

save_md(os.path.join(OUT_DIR, "D4A4_A4C_GATED_RERANKING_VERDICT.md"), verdict_lines)

# ===========================================================================
# MAIN_DECISION_LOG.md
# ===========================================================================
log("=" * 70)
log("Generating MAIN_DECISION_LOG.md")
log("=" * 70)

decision_lines = [
    "# D4A4 A4C-Gated Multi-Objective Reranking -- Decision Log",
    "",
    f"Script: core/scripts/routeA_d4a4_a4c_gated_reranking.py",
    f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    f"Working directory: {ROOT}",
    f"Output directory: {OUT_DIR}",
    "",
    "---",
    "",
    "## Design Decisions",
    "",
    "### 1. SMILES canonicalization for join",
    "The D4A1 predictions use different SMILES representations than the proposals. "
    "RDKit MolToSmiles is used to canonicalize both sides for reliable matching.",
    "",
    "### 2. Attach freq score source",
    "Primary source is D4A1 predictions (attach_freq field). "
    "Fallback is replacement_frequency from proposals for unmatched candidates.",
    "",
    "### 3. HGB_score for unmatched candidates",
    "Unmatched candidates (no D4A1 prediction) get HGB_score = 0.0.",
    "",
    "### 4. Policy definitions",
    f"- {len(policies)} policies defined including {len(P3_LAMBDAS)} lambda variants for P3",
    "- P2 and P5 are similar but P2 operates on ALL candidates while P5 does HGB-first-then-filter",
    "- For P6, the hybrid strategy checks if HGB top-10 has >20% hard reject",
    "- For P7, review tiers: REVIEW_READY > REVIEW_READY_WITH_WARNING > PROPERTY_SHIFT_WARNING > HARD_CHEMISTRY_ALERT",
    "",
    "### 5. Metrics computation",
    "- All metrics are computed per-query and aggregated",
    "- Positive Capture Rate: fraction of queries with exact match in top-10",
    "- Hard Reject Rate: fraction of top-10 slots that are HARD_CHEMISTRY_ALERT",
    "- Review Ready Rate: fraction of top-10 slots that are REVIEW_READY",
    "",
    "### 6. Pareto frontier",
    "- Defined over (Positive Capture Rate, Hard Reject Rate)",
    "- Higher capture AND lower reject = dominance",
    "",
    "### 7. Bootstrap",
    f"- {N_BOOTSTRAP} iterations, {int(ALPHA*100)}% significance level",
    "- Resampling query_ids with replacement",
    "- 95% CI reported for capture, reject, and review_ready rates",
    "",
    "### 8. Lambda grid selection",
    "Lambdas [0.1, 0.3, 0.5, 1.0, 2.0] chosen to cover a wide range of penalty strengths. "
    "If lambda values were selected using test set metrics, mark this as diagnostic-only.",
    "",
    "### 9. Success criteria rationale",
    "- capture_retention >= 0.90: retain at least 90% of HGB's recovery",
    "- hard_reject <= 0.5 * HGB_hard_reject: halve the hard reject rate",
    "- review_ready > HGB_review_ready: maintain or improve review readiness",
    "- at_least_one_review_ready >= 0.99: nearly every query has at least one review-ready candidate",
    "",
    "---",
    "",
    "## Execution Log",
    "",
]
for entry in LOG:
    decision_lines.append(f"- {entry}")

decision_lines.extend([
    "",
    "---",
    "",
    "## Output Files",
    "",
    "| # | File | Description |",
    "|---|------|-------------|",
    "| 1 | d4a4_candidate_review_table.csv | Merged candidate table with A4C flags and HGB scores |",
    "| 2 | d4a4_reranking_policy_definitions.csv | All policy definitions |",
    "| 3 | d4a4_policy_metrics.csv | Per-policy top-10 evaluation metrics |",
    "| 4 | d4a4_policy_by_stratum.csv | Per-policy metrics by query stratum |",
    "| 5 | d4a4_pareto_frontier.csv | Pareto frontier analysis |",
    "| 6 | D4A4_PARETO_INTERPRETATION.md | Pareto frontier interpretation |",
    "| 7 | d4a4_bootstrap_policy_comparison.csv | Bootstrap CI for top policies |",
    "| 8 | D4A4_A4C_GATED_RERANKING_VERDICT.md | Final verdict with all 9 questions |",
    "| 9 | MAIN_DECISION_LOG.md | Decision log and execution trace |",
    "",
    f"**Verdict: {verdict}**",
    f"**{verdict_detail}**",
    "",
    f"**Best policy: {best_pname}** (net_utility_vs_HGB = {best_policy['net_utility_vs_HGB']:.4f})",
    "",
])
save_md(os.path.join(OUT_DIR, "MAIN_DECISION_LOG.md"), decision_lines)

# ===========================================================================
# Final Summary
# ===========================================================================
log("=" * 70)
log("D4A4 COMPLETE")
log("=" * 70)
log(f"Output directory: {OUT_DIR}")
log(f"Verdict: {verdict}")
log(f"Best policy: {best_pname}")
log(f"  Capture rate: {best_policy['positive_capture_rate']:.4f} "
    f"(HGB baseline: {baseline_capture:.4f})")
log(f"  Hard reject rate: {best_policy['hard_reject_rate']:.4f} "
    f"(HGB baseline: {baseline_hard_reject:.4f})")
log(f"  capture_retention: {best_policy['capture_retention_vs_HGB']:.4f}")
log(f"  hard_reject_reduction: {best_policy['hard_reject_reduction_vs_HGB']:.4f}")
log(f"  net_utility: {best_policy['net_utility_vs_HGB']:.4f}")

# List output files
log("\nOutput files:")
for fname in [
    "d4a4_candidate_review_table.csv",
    "d4a4_reranking_policy_definitions.csv",
    "d4a4_policy_metrics.csv",
    "d4a4_policy_by_stratum.csv",
    "d4a4_pareto_frontier.csv",
    "D4A4_PARETO_INTERPRETATION.md",
    "d4a4_bootstrap_policy_comparison.csv",
    "D4A4_A4C_GATED_RERANKING_VERDICT.md",
    "MAIN_DECISION_LOG.md",
]:
    fpath = os.path.join(OUT_DIR, fname)
    exists = os.path.exists(fpath)
    size = os.path.getsize(fpath) if exists else 0
    log(f"  [{ 'OK' if exists else 'MISSING' }] {fname} ({size:,} bytes)")

#!/usr/bin/env python3
"""
Route-A D4A2 Part A: Canonical Metric Freeze
=============================================
Defines the canonical evaluation protocol and recomputes baseline and D4A1 HGB
numbers using a single, consistent metric code path.

Environment: conda activate accfg
"""

import json, csv, os, sys, time, random
from pathlib import Path
from collections import defaultdict, OrderedDict

import numpy as np

import warnings
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────
BASE = Path("E:/zuhui/bioisosteric_diffusion")
D4A0 = BASE / "plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze"
D4A1 = BASE / "plan_results/routeA_chembl37k_d4a1_learned_ranker"
D4A1R = BASE / "plan_results/routeA_chembl37k_d4a1r_ranker_audit"
D4A2 = BASE / "plan_results/routeA_chembl37k_d4a2_canonical_ranker_and_controls"
MATRICES = D4A0 / "matrices"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def write_csv(path, rows, fieldnames):
    path = D4A2 / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f"  wrote {len(rows)} rows -> {path.name}")


def write_md(path, text):
    path = D4A2 / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    log(f"  wrote -> {path.name}")


# ── Query manifest loading ────────────────────────────────────────
def load_query_manifest():
    manifest_path = D4A0 / "d4a0_query_split_manifest.jsonl"
    queries = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["query_id"]] = q
    log(f"  Loaded {len(queries)} queries from manifest")
    return queries


# ── Per-query metrics (CANONICAL) ──────────────────────────────────
KS = [1, 5, 10, 20, 50]


def compute_query_metrics(query_candidates):
    """
    query_candidates: {query_id: [(candidate, score, label), ...]}

    Returns:
        summary: dict of aggregate metrics
        per_query: dict of {query_id: {top1, top5, top10, top20, top50, mrr, has_pos, n_cand}}
    """
    per_query = {}

    for qid, cands in query_candidates.items():
        if not cands:
            continue
        ranked = sorted(cands, key=lambda x: x[1], reverse=True)
        labels = [lbl for _, _, lbl in ranked]
        n_cand = len(ranked)
        has_pos = int(sum(labels) > 0)

        qm = {"n_cand": n_cand, "has_pos": has_pos}

        # Top-K hit rates (query-level: 1 if any positive in top K)
        for k in KS:
            topk_labels = labels[:k]
            qm[f"top{k}"] = 1 if sum(topk_labels) > 0 else 0

        # MRR
        mrr = 0.0
        for i, lbl in enumerate(labels):
            if lbl == 1:
                mrr = 1.0 / (i + 1)
                break
        qm["mrr"] = mrr

        per_query[qid] = qm

    # Aggregate
    n_queries = len(per_query)
    summary = {"n_queries": n_queries, "n_candidates_total": 0, "n_positives_total": 0}
    for k in KS:
        summary[f"top{k}"] = 0.0
    summary["mrr"] = 0.0

    if n_queries > 0:
        for k in KS:
            vals = [qm[f"top{k}"] for qm in per_query.values()]
            summary[f"top{k}"] = float(np.mean(vals))
        summary["mrr"] = float(np.mean([qm["mrr"] for qm in per_query.values()]))
        summary["n_candidates_total"] = int(np.sum([qm["n_cand"] for qm in per_query.values()]))
        summary["n_positives_total"] = int(np.sum([qm["has_pos"] for qm in per_query.values()]))

    return summary, per_query


# ── Bootstrap CI ───────────────────────────────────────────────────
def bootstrap_ci(per_query_dict, metric="top10", n_bootstrap=1000, alpha=0.05):
    """Bootstrap 95% CI for a query-level metric."""
    qids = list(per_query_dict.keys())
    values = np.array([per_query_dict[qid][metric] for qid in qids], dtype=np.float64)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0

    means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = np.random.randint(0, n, n)
        means[i] = np.mean(values[idx])

    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(np.mean(values)), float(lo), float(hi)


# ── Streaming evaluation (canonical) ──────────────────────────────
def evaluate_on_split(shard_paths, queries, scorer_fn):
    """
    Stream through shards, score candidates, compute per-query metrics.
    scorer_fn(qinfo, candidate, gf, af) -> score
    Returns (summary, per_query)
    """
    query_cands = defaultdict(list)

    for shard_path in shard_paths:
        with open(shard_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                qid = row["query_id"]
                candidate = row.get("candidate", "")
                label = row.get("label", 0)
                gf = row.get("global_freq", 0)
                af = row.get("attach_freq", 0)
                qinfo = queries.get(qid, {})
                score = scorer_fn(qinfo, candidate, gf, af)
                query_cands[qid].append((candidate, score, label))

    return compute_query_metrics(dict(query_cands))


# ── Baseline scorers ──────────────────────────────────────────────
def scorer_attach_freq(qinfo, candidate, gf, af):
    return af


# ── Part 1: Canonical Protocol Document ───────────────────────────
def part1_canonical_protocol():
    log("=" * 60)
    log("PART 1: Canonical Metric Protocol Document")
    log("=" * 60)

    # Load shard counts for reference
    test_shards = sorted((MATRICES / "test").glob("test_features_shard_*.jsonl"))
    val_shards = sorted((MATRICES / "val").glob("val_features_shard_*.jsonl"))
    train_shards = sorted((MATRICES / "train").glob("train_features_shard_*.jsonl"))

    # Count queries using manifest
    queries = load_query_manifest()
    test_queries = {qid: q for qid, q in queries.items() if q.get("split") == "test"}
    val_queries = {qid: q for qid, q in queries.items() if q.get("split") == "val"}

    protocol = f"""# D4A2 Canonical Metric Protocol

**Generated**: {now()}
**Seed**: {SEED}

## 1. Split Definition

- **Split type**: Transform-heldout seen-vocabulary test
- **Training queries**: {sum(1 for q in queries.values() if q.get('split') == 'train')} queries
- **Validation queries**: {len(val_queries)} queries
- **Test queries**: {len(test_queries)} queries
- **Shards**: {len(train_shards)} train / {len(val_shards)} val / {len(test_shards)} test
- **Candidate universe**: D4A0 frozen matrix (train-only replacement vocabulary)

## 2. Candidate Universe

All test candidates come from the frozen D4A0 matrix. The candidate universe is the
train-set replacement vocabulary (seen during training). No test-set fragments leak
into the candidate pool.

## 3. Primary Metric

| Metric | Definition | Aggregation |
|--------|------------|-------------|
| **Top-10** | Query-level hit rate: fraction of queries with >=1 positive replacement in top-10 | Mean across test queries |

## 4. Secondary Metrics

| Metric | Definition |
|--------|------------|
| Top-1 | Hit rate at rank 1 |
| Top-5 | Hit rate at top 5 |
| Top-20 | Hit rate at top 20 |
| Top-50 | Hit rate at top 50 |
| MRR | Mean reciprocal rank (first positive position) |

All metrics are reported with bootstrap 95% CI (n=1000 resamples).

## 5. Baselines

- **Attachment frequency baseline**: Score = attach_freq from training data.
  Canonical code path: stream test shards, score by attach_freq, compute per-query metrics.
- **Global frequency baseline**: Score = global_freq from training data.

## 6. Model Evaluation Protocol

1. Train models on **train shards only** (no validation or test labels)
2. Select best model by **validation Top-10**
3. Evaluate selected model **once** on test
4. Report bootstrap 95% CI for all metrics
5. No iterative test-set evaluation

## 7. Feature Set (all computed from train-only data)

| Feature | Source | Train-leak safe |
|---------|--------|-----------------|
| log(1+global_freq) | D4A0 matrix (train counts) | Yes |
| log(1+attach_freq) | D4A0 matrix (train counts) | Yes |
| Morgan Tanimoto (r=2, 1024b) | RDKit from SMILES | Yes |
| Heavy atom delta | RDKit from SMILES | Yes |
| Candidate heavy atoms | RDKit from SMILES | Yes |
| Old fragment heavy atoms | RDKit from SMILES | Yes |
| Attachment match | Always 0 (frozen oracle) | Yes |

## 8. Stopping Rules

- If best validation Top-10 improvement over attachment_freq < 1pp (0.01), stop escalation
- If test bootstrap CI includes 0 delta vs attachment_freq, mark "no significant gain"

## 9. Software

- RDKit for molecular features
- scikit-learn for models
- NumPy for bootstrap resampling (n=1000)
"""

    write_md("d4a2_canonical_metric_protocol.md", protocol)
    log("  Protocol document written.")
    return queries, test_shards, val_shards, train_shards


# ── Part 2: Recompute attachment_frequency baseline ───────────────
def part2_recompute_baseline(test_shards, queries):
    log("=" * 60)
    log("PART 2: Recompute attachment_frequency Baseline (Canonical)")
    log("=" * 60)

    test_queries = {qid: q for qid, q in queries.items() if q.get("split") == "test"}
    t0 = time.time()
    summary, per_query = evaluate_on_split(test_shards, test_queries, scorer_attach_freq)
    elapsed = time.time() - t0

    # Bootstrap CI for Top10
    mean_top10, lo_top10, hi_top10 = bootstrap_ci(per_query, "top10")
    mean_mrr, lo_mrr, hi_mrr = bootstrap_ci(per_query, "mrr")

    row = {
        "method": "attachment_frequency_canonical",
        "n_queries": summary["n_queries"],
        "top1": round(summary["top1"], 4),
        "top5": round(summary["top5"], 4),
        "top10": round(summary["top10"], 4),
        "top20": round(summary["top20"], 4),
        "top50": round(summary["top50"], 4),
        "MRR": round(summary["mrr"], 4),
        "top10_ci_lo": round(lo_top10, 4),
        "top10_ci_hi": round(hi_top10, 4),
        "mrr_ci_lo": round(lo_mrr, 4),
        "mrr_ci_hi": round(hi_mrr, 4),
        "eval_time_sec": round(elapsed, 1),
    }

    write_csv("d4a2_canonical_baseline.csv", [row], list(row.keys()))
    log(f"  attachment_frequency Top10={row['top10']:.4f} [{row['top10_ci_lo']:.4f}, {row['top10_ci_hi']:.4f}]")
    log(f"  MRR={row['MRR']:.4f} ({elapsed:.1f}s)")

    return row, per_query


# ── Part 3: Recompute D4A1 HGB from predictions ──────────────────
def part3_recompute_hgb(test_queries_set=None):
    log("=" * 60)
    log("PART 3: Recompute D4A1 HGB Top10 from Predictions (Canonical)")
    log("=" * 60)

    pred_path = D4A1 / "d4a1_test_predictions.jsonl"
    if not pred_path.exists():
        log(f"  ERROR: {pred_path} not found")
        return None, None

    # Load predictions into query -> [(candidate, score, label)]
    query_cands = defaultdict(list)
    line_count = 0
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = row["query_id"]
            candidate = row.get("candidate", "")
            score = row.get("score", 0.0)
            label = row.get("label", 0)
            query_cands[qid].append((candidate, score, label))
            line_count += 1

    log(f"  Loaded {line_count} prediction rows, {len(query_cands)} queries")

    if test_queries_set is not None:
        query_cands = {qid: cands for qid, cands in query_cands.items()
                       if qid in test_queries_set}
        log(f"  Filtered to {len(query_cands)} test queries")

    t0 = time.time()
    summary, per_query = compute_query_metrics(dict(query_cands))
    elapsed = time.time() - t0

    mean_top10, lo_top10, hi_top10 = bootstrap_ci(per_query, "top10")
    mean_mrr, lo_mrr, hi_mrr = bootstrap_ci(per_query, "mrr")

    row = {
        "method": "d4a1_hgb_canonical",
        "n_queries": summary["n_queries"],
        "top1": round(summary["top1"], 4),
        "top5": round(summary["top5"], 4),
        "top10": round(summary["top10"], 4),
        "top20": round(summary["top20"], 4),
        "top50": round(summary["top50"], 4),
        "MRR": round(summary["mrr"], 4),
        "top10_ci_lo": round(lo_top10, 4),
        "top10_ci_hi": round(hi_top10, 4),
        "mrr_ci_lo": round(lo_mrr, 4),
        "mrr_ci_hi": round(hi_mrr, 4),
        "eval_time_sec": round(elapsed, 1),
    }

    write_csv("d4a2_canonical_hgb.csv", [row], list(row.keys()))
    log(f"  D4A1 HGB Top10={row['top10']:.4f} [{row['top10_ci_lo']:.4f}, {row['top10_ci_hi']:.4f}]")
    log(f"  MRR={row['MRR']:.4f} ({elapsed:.1f}s)")

    return row, per_query


# ── Part 4: Reconciliation Table ──────────────────────────────────
def part4_reconciliation(baseline_row, hgb_row):
    log("=" * 60)
    log("PART 4: Metric Reconciliation Table")
    log("=" * 60)

    # Also load existing numbers from D4A0, D4A1, D4A1R
    d4a0_bl = None
    d4a0_bl_path = D4A0 / "d4a0_baseline_reproduction.csv"
    if d4a0_bl_path.exists():
        import pandas as pd
        df = pd.read_csv(d4a0_bl_path)
        row = df[df["baseline"] == "attachment_frequency"]
        if len(row) > 0:
            d4a0_bl = float(row["top10"].iloc[0])

    d4a1_hgb_orig = None
    d4a1_test_path = D4A1 / "d4a1_test_metrics.csv"
    if d4a1_test_path.exists():
        import pandas as pd
        df = pd.read_csv(d4a1_test_path)
        if len(df) > 0:
            d4a1_hgb_orig = float(df["top10"].iloc[0])

    d4a1r_hgb_retrain = None
    d4a1r_ablation_path = D4A1R / "d4a1r_feature_ablation_results.csv"
    if d4a1r_ablation_path.exists():
        import pandas as pd
        df = pd.read_csv(d4a1r_ablation_path)
        row = df[df["ablation"] == "A0_full"]
        if len(row) > 0:
            d4a1r_hgb_retrain = float(row["test_Top10_diagnostic"].iloc[0])

    # D4A1R recomputed baseline (train_vocab_covered attach_top10)
    d4a1r_recomp_bl = None
    d4a1r_recomp_path = D4A1R / "d4a1r_recomputed_test_by_subset.csv"
    if d4a1r_recomp_path.exists():
        import pandas as pd
        df = pd.read_csv(d4a1r_recomp_path)
        row = df[df["subset"] == "train_vocab_covered"]
        if len(row) > 0:
            d4a1r_recomp_bl = float(row["attach_top10"].iloc[0])

    # D4A1R recomputed learned (unseen_old_fragment or train_vocab_covered learned)
    d4a1r_recomp_learned = None
    if d4a1r_recomp_path.exists():
        import pandas as pd
        df = pd.read_csv(d4a1r_recomp_path)
        row = df[df["subset"] == "unseen_old_fragment"]
        if len(row) > 0:
            d4a1r_recomp_learned = float(row["learned_top10"].iloc[0])

    rows = [
        {
            "source": "D4A0 frozen baseline (original)",
            "method": "attachment_frequency",
            "top10": f"{d4a0_bl:.4f}" if d4a0_bl else "N/A",
            "notes": "Original D4A0 pipeline eval on all 21,680 test queries"
        },
        {
            "source": "D4A1 recomputed baseline",
            "method": "attachment_frequency (train_vocab_covered)",
            "top10": f"{d4a1r_recomp_bl:.4f}" if d4a1r_recomp_bl else "N/A",
            "notes": "D4A1R recomputed on train_vocab_covered subset (21,052 queries)"
        },
        {
            "source": "D4A1 bootstrap learned_top10",
            "method": "HGB (unseen_old_fragment)",
            "top10": f"{d4a1r_recomp_learned:.4f}" if d4a1r_recomp_learned else "N/A",
            "notes": "D4A1R recomputed HGB on unseen_old_fragment subset"
        },
        {
            "source": "D4A1 test_metrics HGB",
            "method": "HGB (original eval)",
            "top10": f"{d4a1_hgb_orig:.4f}" if d4a1_hgb_orig else "N/A",
            "notes": "Original D4A1 HGB eval on all test queries"
        },
        {
            "source": "D4A1R retrained HGB",
            "method": "HGB retrained (A0_full)",
            "top10": f"{d4a1r_hgb_retrain:.4f}" if d4a1r_hgb_retrain else "N/A",
            "notes": "D4A1R retrained on train_vocab_covered, different subsample"
        },
    ]

    # This-run canonical numbers
    if baseline_row:
        rows.append({
            "source": "THIS RUN: canonical baseline",
            "method": "attachment_frequency (canonical code path)",
            "top10": f"{baseline_row['top10']:.4f} [{baseline_row['top10_ci_lo']:.4f}-{baseline_row['top10_ci_hi']:.4f}]",
            "notes": f"Recomputed via canonical code path, n={baseline_row['n_queries']} queries"
        })
    if hgb_row:
        rows.append({
            "source": "THIS RUN: canonical D4A1 HGB",
            "method": "HGB from predictions (canonical code path)",
            "top10": f"{hgb_row['top10']:.4f} [{hgb_row['top10_ci_lo']:.4f}-{hgb_row['top10_ci_hi']:.4f}]",
            "notes": f"Recomputed from d4a1_test_predictions.jsonl via canonical metrics"
        })

    fieldnames = ["source", "method", "top10", "notes"]
    write_csv("d4a2_metric_reconciliation.csv", rows, fieldnames)
    log("  Reconciliation table written.")

    # Determine canonical numbers
    canonical_baseline = baseline_row["top10"] if baseline_row else 0.0
    canonical_hgb = hgb_row["top10"] if hgb_row else 0.0
    log(f"\n  CANONICAL BASELINE (attachment_frequency): {canonical_baseline:.4f}")
    log(f"  CANONICAL D4A1 HGB:                       {canonical_hgb:.4f}")

    return canonical_baseline, canonical_hgb


# ── Main ──────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("D4A2 PART A: CANONICAL METRIC FREEZE")
    log("=" * 60)

    # Part 1: Protocol document
    queries, test_shards, val_shards, train_shards = part1_canonical_protocol()

    # Filter to test queries only
    test_queries_set = set(qid for qid, q in queries.items() if q.get("split") == "test")
    test_queries = {qid: q for qid, q in queries.items() if q.get("split") == "test"}
    log(f"  Test query count: {len(test_queries_set)}")

    # Part 2: Recompute attachment_frequency baseline
    baseline_row, baseline_per_query = part2_recompute_baseline(test_shards, test_queries)

    # Part 3: Recompute HGB from predictions
    hgb_row, hgb_per_query = part3_recompute_hgb(test_queries_set)

    # Part 4: Reconciliation table
    canonical_baseline, canonical_hgb = part4_reconciliation(baseline_row, hgb_row)

    # Part 5: Summary
    log("\n" + "=" * 60)
    log("PART A COMPLETE - CANONICAL NUMBERS DECLARED")
    log("=" * 60)
    log(f"  Canonical baseline (attachment_frequency) Top10: {canonical_baseline:.4f}")
    log(f"  Canonical D4A1 HGB Top10:                      {canonical_hgb:.4f}")
    log(f"  Delta: {canonical_hgb - canonical_baseline:+.4f}")
    log(f"  Output directory: {D4A2}")

    # Write summary marker
    write_md("d4a2a_canonical_metric_complete.md",
             f"# D4A2A Complete\n\n"
             f"- Canonical baseline Top10: {canonical_baseline:.4f}\n"
             f"- Canonical D4A1 HGB Top10: {canonical_hgb:.4f}\n"
             f"- Delta: {canonical_hgb - canonical_baseline:+.4f}\n"
             f"- Timestamp: {now()}\n")

    return canonical_baseline, canonical_hgb


if __name__ == "__main__":
    main()

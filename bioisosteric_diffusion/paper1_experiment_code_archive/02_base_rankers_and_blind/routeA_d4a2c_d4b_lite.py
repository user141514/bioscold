#!/usr/bin/env python3
"""
Route-A D4A2 Part C: D4B-lite Model-Class Control
==================================================
Tests if minimal generation-style baselines can compete with ranking models
on the SAME seen-vocabulary benchmark. No full diffusion models.

Environment: conda activate accfg
"""

import json, csv, os, sys, time, random, math
from pathlib import Path
from collections import defaultdict

import numpy as np

import warnings
warnings.filterwarnings("ignore")
from rdkit import Chem, DataStructs, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
from rdkit.Chem import AllChem

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────
BASE = Path("E:/zuhui/bioisosteric_diffusion")
D4A0 = BASE / "plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze"
D4A2 = BASE / "plan_results/routeA_chembl37k_d4a2_canonical_ranker_and_controls"
MATRICES = D4A0 / "matrices"


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


# ── Query manifest ────────────────────────────────────────────────
def load_query_manifest():
    manifest_path = D4A0 / "d4a0_query_split_manifest.jsonl"
    queries = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["query_id"]] = q
    log(f"  Loaded {len(queries)} queries from manifest")
    return queries


# ── RDKit Feature Cache ───────────────────────────────────────────
class FeatureCache:
    def __init__(self):
        self.fp_cache = {}
        self.ha_cache = {}

    def get_fp(self, smiles):
        if smiles not in self.fp_cache:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.fp_cache[smiles] = np.zeros(1024, dtype=np.float32)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                arr = np.zeros(1024, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                self.fp_cache[smiles] = arr
        return self.fp_cache[smiles]

    def get_ha(self, smiles):
        if smiles not in self.ha_cache:
            mol = Chem.MolFromSmiles(smiles)
            self.ha_cache[smiles] = mol.GetNumHeavyAtoms() if mol else 0
        return self.ha_cache[smiles]


FCACHE = FeatureCache()


# ── Metrics (canonical) ───────────────────────────────────────────
KS = [1, 5, 10, 20, 50]


def compute_query_metrics(query_cands):
    per_query = {}
    for qid, cands in query_cands.items():
        if not cands:
            continue
        ranked = sorted(cands, key=lambda x: x[1], reverse=True)
        labels = [lbl for _, _, lbl in ranked]
        qm = {"n_cand": len(ranked), "has_pos": int(sum(labels) > 0)}
        for k in KS:
            qm[f"top{k}"] = 1 if sum(labels[:k]) > 0 else 0
        mrr = 0.0
        for i, lbl in enumerate(labels):
            if lbl == 1:
                mrr = 1.0 / (i + 1)
                break
        qm["mrr"] = mrr
        per_query[qid] = qm

    n = len(per_query)
    summary = {"n_queries": n}
    if n > 0:
        for k in KS:
            summary[f"top{k}"] = float(np.mean([qm[f"top{k}"] for qm in per_query.values()]))
        summary["mrr"] = float(np.mean([qm["mrr"] for qm in per_query.values()]))
    else:
        for k in KS:
            summary[f"top{k}"] = 0.0
        summary["mrr"] = 0.0
    return summary, per_query


def evaluate_on_split(shard_paths, queries, scorer_fn):
    """Stream through shards, score candidates, return metrics."""
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


def bootstrap_ci(per_query, metric="top10", n_bootstrap=1000):
    qids = list(per_query.keys())
    values = np.array([per_query[q][metric] for q in qids])
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = np.random.randint(0, n, n)
        means[i] = np.mean(values[idx])
    lo = np.percentile(means, 2.5)
    hi = np.percentile(means, 97.5)
    return float(np.mean(means)), float(lo), float(hi)


# ── Reference scorers ─────────────────────────────────────────────
def scorer_attach_freq(qinfo, candidate, gf, af):
    return af


# ── Part C1: Fragment embedding centroid sampler ──────────────────
def build_c1_centroid_model(train_shards, queries):
    """
    For each query, compute mean Morgan FP of positive replacement candidates in train.
    For test, find top-K train-vocab candidates closest to centroid by tanimoto.
    Score = tanimoto to centroid.
    """
    log("  Building C1: Fragment embedding centroid model...")

    # Step 1: Collect all train-vocab candidate SMILES and their FPs
    # Stream ALL train shards and collect unique candidates with their FPs
    train_vocab_fps = {}  # candidate_smiles -> fp
    query_centroids = defaultdict(list)  # query_id -> list of positive FPs

    for shard_path in train_shards:
        with open(shard_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                qid = row["query_id"]
                candidate = row.get("candidate", "")
                label = row.get("label", 0)
                qinfo = queries.get(qid, {})

                # Get fingerprint
                fp = FCACHE.get_fp(candidate)
                train_vocab_fps[candidate] = fp

                if label == 1:
                    query_centroids[qid].append(fp)

    log(f"    Train vocabulary: {len(train_vocab_fps)} unique candidates")
    log(f"    Queries with positives: {len(query_centroids)}")

    # Step 2: Pre-compute centroid for each query (mean of positive FPs)
    centroids = {}
    for qid, fps in query_centroids.items():
        if fps:
            centroids[qid] = np.mean(fps, axis=0)
    log(f"    Centroids computed: {len(centroids)}")

    train_vocab_items = list(train_vocab_fps.items())

    def scorer(qinfo, candidate, gf, af):
        qid = qinfo.get("query_id", "")
        if qid not in centroids:
            return 0.0
        centroid = centroids[qid]
        cand_fp = FCACHE.get_fp(candidate)
        if centroid.sum() == 0 or cand_fp.sum() == 0:
            return 0.0
        dot = float(np.dot(centroid, cand_fp))
        tanimoto = dot / (centroid.sum() + cand_fp.sum() - dot + 1e-10)
        return tanimoto

    return scorer


# ── Part C2: Per-query softmax scorer ──────────────────────────
def build_c2_softmax_scorer(shard_paths, queries, temperature=1.0):
    """
    Pre-scan shards to collect per-query candidate -> attach_freq mapping.
    Compute per-query softmax: P(c|q) = exp(af/temp) / sum(exp(af/temp)).
    Returns a scorer function that returns this probability as score.
    This ensures C2 != attach_freq (which is pure frequency, not normalized).
    """
    log("  C2: Pre-scanning shards for per-query softmax...")
    per_q_af = defaultdict(list)  # qid -> [(candidate, af)]
    for shard_path in shard_paths:
        with open(shard_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                qid = row["query_id"]
                candidate = row.get("candidate", "")
                af = row.get("attach_freq", 0)
                per_q_af[qid].append((candidate, af))

    # Compute softmax per query
    softmax_lookup = {}
    for qid, items in per_q_af.items():
        candidates, afs = zip(*items)
        afs_arr = np.array(afs, dtype=np.float64)
        exp_vals = np.exp(afs_arr / temperature)
        probs = exp_vals / np.sum(exp_vals)
        for cand, prob in zip(candidates, probs):
            softmax_lookup[(qid, cand)] = prob

    log(f"    Computed softmax for {len(per_q_af)} queries, {len(softmax_lookup)} candidates")

    def scorer(qinfo, candidate, gf, af):
        qid = qinfo.get("query_id", "")
        return softmax_lookup.get((qid, candidate), 0.0)

    return scorer


# ── Part C3: Old-fragment similarity retriever ────────────────────
def build_c3_similarity_scorer():
    """
    Score = Morgan tanimoto(old_fragment, candidate).
    Pure molecular similarity, no training.
    """
    def scorer(qinfo, candidate, gf, af):
        old_smiles = qinfo.get("old_fragment_smiles", "")
        if not old_smiles:
            return 0.0
        old_fp = FCACHE.get_fp(old_smiles)
        cand_fp = FCACHE.get_fp(candidate)
        if old_fp.sum() == 0 or cand_fp.sum() == 0:
            return 0.0
        dot = float(np.dot(old_fp, cand_fp))
        tanimoto = dot / (old_fp.sum() + cand_fp.sum() - dot + 1e-10)
        return tanimoto

    return scorer


# ── Main ──────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("D4A2 PART C: D4B-LITE MODEL-CLASS CONTROL")
    log("=" * 60)

    # Load data
    queries = load_query_manifest()
    test_shards = sorted((MATRICES / "test").glob("test_features_shard_*.jsonl"))
    train_shards = sorted((MATRICES / "train").glob("train_features_shard_*.jsonl"))
    val_shards = sorted((MATRICES / "val").glob("val_features_shard_*.jsonl"))

    test_queries = {qid: q for qid, q in queries.items() if q.get("split") == "test"}
    val_queries = {qid: q for qid, q in queries.items() if q.get("split") == "val"}
    log(f"  Test queries: {len(test_queries)}, Val queries: {len(val_queries)}")

    # ── Reference: attachment_frequency ──
    log("\n--- Reference: attachment_frequency ---")
    t0 = time.time()
    summary_af_test, perq_af_test = evaluate_on_split(test_shards, test_queries, scorer_attach_freq)
    m, lo, hi = bootstrap_ci(perq_af_test, "top10")
    log(f"  Test Top10={summary_af_test['top10']:.4f} [{lo:.4f}, {hi:.4f}] ({time.time()-t0:.1f}s)")

    ref_row = {
        "method": "attachment_frequency",
        "arm": "REF",
        "val_top10": 0.0,  # computed below
        "test_top10": round(summary_af_test["top10"], 4),
        "test_mrr": round(summary_af_test["mrr"], 4),
        "test_top10_ci_lo": round(lo, 4),
        "test_top10_ci_hi": round(hi, 4),
        "note": "Reference baseline",
    }

    # Val for attach_freq
    summary_af_val, _ = evaluate_on_split(val_shards, val_queries, scorer_attach_freq)
    ref_row["val_top10"] = round(summary_af_val["top10"], 4)

    results = [ref_row]

    # ── C1: Fragment embedding centroid sampler ──
    log("\n--- C1: Fragment embedding centroid sampler ---")
    t0 = time.time()
    c1_scorer = build_c1_centroid_model(train_shards, test_queries)
    log(f"  Building complete ({time.time()-t0:.1f}s). Evaluating on val...")
    summary_c1_val, perq_c1_val = evaluate_on_split(val_shards, val_queries, c1_scorer)
    log(f"  Val Top10={summary_c1_val['top10']:.4f}")
    log(f"  Evaluating on test...")
    t_eval = time.time()
    summary_c1_test, perq_c1_test = evaluate_on_split(test_shards, test_queries, c1_scorer)
    m, lo, hi = bootstrap_ci(perq_c1_test, "top10")
    log(f"  Test Top10={summary_c1_test['top10']:.4f} [{lo:.4f}, {hi:.4f}] ({time.time()-t_eval:.1f}s)")

    results.append({
        "method": "C1_centroid_sampler",
        "arm": "C1",
        "val_top10": round(summary_c1_val["top10"], 4),
        "test_top10": round(summary_c1_test["top10"], 4),
        "test_mrr": round(summary_c1_test["mrr"], 4),
        "test_top10_ci_lo": round(lo, 4),
        "test_top10_ci_hi": round(hi, 4),
        "note": "Fragment embedding centroid, tanimoto to mean positive FP",
    })

    # ── C2: Per-query softmax proposal ──
    log("\n--- C2: Per-query softmax proposal ---")
    # Build separate scorers for val and test (candidate pools differ)
    c2_test_scorer = build_c2_softmax_scorer(test_shards, test_queries)
    c2_val_scorer = build_c2_softmax_scorer(val_shards, val_queries)
    summary_c2_val, perq_c2_val = evaluate_on_split(val_shards, val_queries, c2_val_scorer)
    summary_c2_test, perq_c2_test = evaluate_on_split(test_shards, test_queries, c2_test_scorer)
    m, lo, hi = bootstrap_ci(perq_c2_test, "top10")
    log(f"  Test Top10={summary_c2_test['top10']:.4f} [{lo:.4f}, {hi:.4f}]")

    results.append({
        "method": "C2_softmax_proposal",
        "arm": "C2",
        "val_top10": round(summary_c2_val["top10"], 4),
        "test_top10": round(summary_c2_test["top10"], 4),
        "test_mrr": round(summary_c2_test["mrr"], 4),
        "test_top10_ci_lo": round(lo, 4),
        "test_top10_ci_hi": round(hi, 4),
        "note": "Per-query softmax over attach_freq, temperature=1.0",
    })

    # ── C3: Old-fragment similarity retriever ──
    log("\n--- C3: Old-fragment similarity retriever ---")
    c3_scorer = build_c3_similarity_scorer()
    summary_c3_val, perq_c3_val = evaluate_on_split(val_shards, val_queries, c3_scorer)
    summary_c3_test, perq_c3_test = evaluate_on_split(test_shards, test_queries, c3_scorer)
    m, lo, hi = bootstrap_ci(perq_c3_test, "top10")
    log(f"  Test Top10={summary_c3_test['top10']:.4f} [{lo:.4f}, {hi:.4f}]")

    results.append({
        "method": "C3_old_fragment_similarity",
        "arm": "C3",
        "val_top10": round(summary_c3_val["top10"], 4),
        "test_top10": round(summary_c3_test["top10"], 4),
        "test_mrr": round(summary_c3_test["mrr"], 4),
        "test_top10_ci_lo": round(lo, 4),
        "test_top10_ci_hi": round(hi, 4),
        "note": "Morgan tanimoto(old_fragment, candidate) - no training",
    })

    # ── Write results ──
    fieldnames = ["method", "arm", "val_top10", "test_top10", "test_mrr",
                  "test_top10_ci_lo", "test_top10_ci_hi", "note"]
    write_csv("d4a2_d4b_lite_results.csv", results, fieldnames)

    # ── Comparison with rankers ──
    log("\n" + "=" * 60)
    log("COMPARISON SUMMARY")
    log("=" * 60)

    best_d4b = max(results[1:], key=lambda r: r["test_top10"]) if len(results) > 1 else None
    if best_d4b:
        log(f"  Best D4B-lite: {best_d4b['method']} Test Top10={best_d4b['test_top10']:.4f}")
    log(f"  Reference (attach_freq): Test Top10={ref_row['test_top10']:.4f}")

    # ── Read Part B results for comparison ──
    d4a2_ranker_path = D4A2 / "d4a2_ranker_model_results.csv"
    if d4a2_ranker_path.exists():
        import pandas as pd
        df = pd.read_csv(d4a2_ranker_path)
        best_ranker = df.loc[df["val_top10"].idxmax()]
        log(f"  Best D4A2 ranker: {best_ranker['model']} Val Top10={best_ranker['val_top10']:.4f}")
        log(f"    Test Top10={best_ranker['test_top10']:.4f}")
        log(f"  Delta (best_ranker - best_D4B): {best_ranker['test_top10'] - best_d4b['test_top10']:+.4f}" if best_d4b else "")

        # Add comparison rows
        results.append({
            "method": f"best_ranker_{best_ranker['model']}",
            "arm": "D4A2",
            "val_top10": round(best_ranker["val_top10"], 4),
            "test_top10": round(best_ranker["test_top10"], 4),
            "test_mrr": round(best_ranker["test_mrr"], 4),
            "test_top10_ci_lo": 0.0,
            "test_top10_ci_hi": 0.0,
            "note": "Best D4A2 ranker (from Part B)",
        })

    # D4A1 canonical HGB
    d4a2_hgb_path = D4A2 / "d4a2_canonical_hgb.csv"
    if d4a2_hgb_path.exists():
        import pandas as pd
        df = pd.read_csv(d4a2_hgb_path)
        if len(df) > 0:
            log(f"  D4A1 canonical HGB: Test Top10={df['top10'].iloc[0]:.4f}")
            results.append({
                "method": "D4A1_HGB_canonical",
                "arm": "REF",
                "val_top10": 0.0,
                "test_top10": round(df["top10"].iloc[0], 4),
                "test_mrr": round(df["MRR"].iloc[0], 4),
                "test_top10_ci_lo": round(df["top10_ci_lo"].iloc[0], 4),
                "test_top10_ci_hi": round(df["top10_ci_hi"].iloc[0], 4),
                "note": "D4A1 HGB canonical (from Part A)",
            })

    # Re-write with comparison rows (dedup)
    write_csv("d4a2_d4b_lite_results.csv", results, fieldnames)

    # ── Verdict ──
    log("\n--- Verdict: Does ranking beat generation-style on seen-vocab? ---")
    if best_d4b and best_ranker is not None:
        delta = best_ranker["test_top10"] - best_d4b["test_top10"]
        if delta > 0.05:
            verdict = f"YES: Ranking (Top10={best_ranker['test_top10']:.4f}) beats best D4B-lite (Top10={best_d4b['test_top10']:.4f}) by {delta:.4f}"
        elif delta > 0.01:
            verdict = f"MODERATE: Ranking (Top10={best_ranker['test_top10']:.4f}) edges D4B-lite (Top10={best_d4b['test_top10']:.4f}) by {delta:.4f}"
        else:
            verdict = f"NEGLIGIBLE: Ranking (Top10={best_ranker['test_top10']:.4f}) and D4B-lite (Top10={best_d4b['test_top10']:.4f}) are comparable ({delta:+.4f})"
    else:
        verdict = "INCONCLUSIVE: Missing data for comparison"
    log(f"  {verdict}")

    # Part C comparison markdown
    comp_md = f"""# D4A2 Part C: D4B-lite Model-Class Control

**Generated**: {now()}
**Seed**: {SEED}

## Results Summary

| Method | Arm | Val Top10 | Test Top10 | 95% CI |
|--------|-----|-----------|------------|--------|
"""
    for r in results:
        comp_md += f"| {r['method']} | {r['arm']} | {r['val_top10']:.4f} | {r['test_top10']:.4f} | [{r['test_top10_ci_lo']:.4f}, {r['test_top10_ci_hi']:.4f}] |\n"

    comp_md += f"""

## Verdict

{verdict}

## Interpretation

- **C1 (centroid sampler)**: Tests if "generate ideal embedding then retrieve" works.
  This simulates a generation approach without actually training a generative model.
- **C2 (softmax proposal)**: Smoothed frequency baseline, tests if normalization matters.
- **C3 (old-fragment similarity)**: Pure molecular similarity baseline, no training required.

## Key Finding

"""
    if best_d4b:
        comp_md += f"Best D4B-lite ({best_d4b['method']}) achieves Test Top10={best_d4b['test_top10']:.4f}"
    if best_ranker is not None:
        comp_md += f" vs best D4A2 ranker ({best_ranker['model']}) at Test Top10={best_ranker['test_top10']:.4f}.\n"

    write_md("d4a2_model_class_comparison.md", comp_md)
    log("  Comparison markdown written.")

    # Complete marker
    write_md("d4a2c_d4b_lite_complete.md",
             f"# D4A2C Complete\n\n"
             + "\n".join(f"- {r['method']}: Test Top10={r['test_top10']:.4f}" for r in results) + "\n"
             + f"\nVerdict: {verdict}\nTimestamp: {now()}\n")


if __name__ == "__main__":
    main()

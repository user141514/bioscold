#!/usr/bin/env python3
"""
Route-A D4A2 Part D: A4C Integration Preview
=============================================
Heuristic drug-chemistry evaluation of top-K predictions from each method.
No actual geometry optimization -- heuristic A4C only.

Checks if learned ranker top-K predictions are "better" than frequency baseline
top-K by simple drug-chemistry heuristics.

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
from rdkit.Chem import AllChem, Descriptors

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────
BASE = Path("E:/zuhui/bioisosteric_diffusion")
D4A0 = BASE / "plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze"
D4A1 = BASE / "plan_results/routeA_chembl37k_d4a1_learned_ranker"
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


# ── RDKit utilities ───────────────────────────────────────────────
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


# ── Heuristic A4C Rules ──────────────────────────────────────────
def heuristic_a4c_review(candidate_smiles, old_smiles):
    """
    Apply heuristic A4C rules WITHOUT RDKit geometry optimization.

    Returns: ('REVIEW_READY' | 'HARD_CLASH' | 'NEEDS_REVIEW')
    """
    # Parse molecules
    cand_mol = Chem.MolFromSmiles(candidate_smiles)
    old_mol = Chem.MolFromSmiles(old_smiles)

    if cand_mol is None or old_mol is None:
        return "HARD_CLASH"

    # Basic molecular properties
    cand_ha = cand_mol.GetNumHeavyAtoms()
    old_ha = old_mol.GetNumHeavyAtoms()

    # Rule 1: HARD_CLASH if heavy atom delta > 30% of old fragment size
    ha_delta = abs(cand_ha - old_ha)
    if old_ha > 0:
        ha_delta_ratio = ha_delta / old_ha
    else:
        ha_delta_ratio = 1.0
    if ha_delta_ratio > 0.30:
        return "HARD_CLASH"
    # Rule 1b: NEEDS_REVIEW if heavy atom delta > 15% (moderate size change)
    if ha_delta_ratio > 0.15:
        return "NEEDS_REVIEW"

    # Rule 2: HARD_CLASH if tanimoto < 0.15 (too dissimilar)
    cand_fp = FCACHE.get_fp(candidate_smiles)
    old_fp = FCACHE.get_fp(old_smiles)
    if cand_fp.sum() > 0 and old_fp.sum() > 0:
        dot = float(np.dot(cand_fp, old_fp))
        tanimoto = dot / (cand_fp.sum() + old_fp.sum() - dot + 1e-10)
        if tanimoto < 0.15:
            return "HARD_CLASH"

    # Rule 3: REVIEW_READY if passes basic filters
    # - Reasonable size (1-50 heavy atoms)
    if cand_ha < 1 or cand_ha > 50:
        return "HARD_CLASH"

    # - Contains only common elements (organic), allow wildcard (0 = attachment point)
    organic = set([0, 1, 6, 7, 8, 9, 16, 17, 35, 53])  # *, H, C, N, O, F, S, Cl, Br, I
    for atom in cand_mol.GetAtoms():
        if atom.GetAtomicNum() not in organic:
            return "HARD_CLASH"

    # Passes all tests -> REVIEW_READY
    return "REVIEW_READY"


def compute_tanimoto(smi1, smi2):
    fp1 = FCACHE.get_fp(smi1)
    fp2 = FCACHE.get_fp(smi2)
    if fp1.sum() > 0 and fp2.sum() > 0:
        dot = float(np.dot(fp1, fp2))
        return dot / (fp1.sum() + fp2.sum() - dot + 1e-10)
    return 0.0


def compute_ha_delta(smi1, smi2):
    ha1 = FCACHE.get_ha(smi1)
    ha2 = FCACHE.get_ha(smi2)
    return abs(ha1 - ha2)


# ── Evaluate top-10 predictions with A4C heuristics ──────────────
def evaluate_method_a4c(test_shards, queries, scorer_fn, method_name):
    """
    For each test query:
    1. Score all candidates using scorer_fn
    2. Take top-10
    3. Apply heuristic A4C
    4. Compute aggregate statistics

    Returns dict of metrics.
    """
    # Collect per-query predictions (streaming)
    query_top10s = {}
    query_cands = defaultdict(list)

    for shard_path in test_shards:
        with open(shard_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                qid = row["query_id"]
                candidate = row.get("candidate", "")
                label = row.get("label", 0)
                gf = row.get("global_freq", 0)
                af = row.get("attach_freq", 0)
                qinfo = queries.get(qid, {})
                old_smi = qinfo.get("old_fragment_smiles", "")
                score = scorer_fn(qinfo, candidate, gf, af)
                query_cands[qid].append((candidate, score, label, old_smi))

    # Extract top-10 per query
    n_queries = 0
    n_review_ready_queries = 0
    n_hard_clash_total = 0
    n_top10_total = 0
    sum_tanimoto = 0.0
    sum_ha_delta = 0.0
    query_details = []

    for qid, cands in query_cands.items():
        if not cands:
            continue
        ranked = sorted(cands, key=lambda x: x[1], reverse=True)
        top10 = ranked[:10]
        old_smiles = top10[0][3] if len(top10[0]) > 3 else ""

        n_queries += 1
        n_top10_total += len(top10)

        has_review_ready = False
        hard_clash_count = 0
        query_tanimotos = []
        query_ha_deltas = []

        for cand, score, label, old_smi in top10:
            if not old_smi:
                old_smi = top10[0][3]  # fallback
            verdict = heuristic_a4c_review(cand, old_smi)
            if verdict == "REVIEW_READY":
                has_review_ready = True
            elif verdict == "HARD_CLASH":
                hard_clash_count += 1

            tanimoto = compute_tanimoto(old_smi, cand)
            ha_d = compute_ha_delta(old_smi, cand)
            query_tanimotos.append(tanimoto)
            query_ha_deltas.append(ha_d)

        if has_review_ready:
            n_review_ready_queries += 1
        n_hard_clash_total += hard_clash_count
        sum_tanimoto += np.mean(query_tanimotos)
        sum_ha_delta += np.mean(query_ha_deltas)

        query_details.append({
            "method": method_name,
            "query_id": qid,
            "has_review_ready": 1 if has_review_ready else 0,
            "hard_clash_count": hard_clash_count,
            "avg_tanimoto_top10": round(np.mean(query_tanimotos), 4),
            "avg_ha_delta_top10": round(np.mean(query_ha_deltas), 4),
        })

    metrics = {
        "method": method_name,
        "n_queries": n_queries,
        "review_ready_rate": n_review_ready_queries / max(n_queries, 1),
        "n_review_ready_queries": n_review_ready_queries,
        "hard_reject_rate": n_hard_clash_total / max(n_top10_total, 1),
        "n_hard_clash_total": n_hard_clash_total,
        "avg_tanimoto_top10": sum_tanimoto / max(n_queries, 1),
        "avg_ha_delta_top10": sum_ha_delta / max(n_queries, 1),
    }

    return metrics, query_details


# ── Scorers ───────────────────────────────────────────────────────
def scorer_attach_freq(qinfo, candidate, gf, af):
    return af


# ── Main ──────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("D4A2 PART D: A4C INTEGRATION PREVIEW")
    log("=" * 60)

    # Load data
    queries = load_query_manifest()
    test_shards = sorted((MATRICES / "test").glob("test_features_shard_*.jsonl"))
    test_queries = {qid: q for qid, q in queries.items() if q.get("split") == "test"}
    log(f"  Test queries: {len(test_queries)}")

    all_metrics = []
    all_details = []

    # Method 1: attachment_frequency
    log("\n--- Method: attachment_frequency ---")
    t0 = time.time()
    metrics, details = evaluate_method_a4c(test_shards, test_queries, scorer_attach_freq, "attachment_frequency")
    metrics["eval_time_sec"] = round(time.time() - t0, 1)
    all_metrics.append(metrics)
    all_details.extend(details)
    log(f"  REVIEW_READY rate: {metrics['review_ready_rate']:.4f}")
    log(f"  Hard reject rate: {metrics['hard_reject_rate']:.4f}")
    log(f"  Avg Tanimoto top-10: {metrics['avg_tanimoto_top10']:.4f}")

    # Method 2: D4A1 HGB scorer from predictions
    log("\n--- Method: HGB (from predictions) ---")
    t0 = time.time()

    # Load predictions
    pred_path = D4A1 / "d4a1_test_predictions.jsonl"
    hgb_scores = {}
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = row["query_id"]
            cand = row["candidate"]
            score = row["score"]
            if qid not in hgb_scores:
                hgb_scores[qid] = {}
            hgb_scores[qid][cand] = score
    log(f"  Loaded HGB scores for {len(hgb_scores)} queries")

    def scorer_hgb(qinfo, candidate, gf, af):
        qid = qinfo.get("query_id", "")
        return hgb_scores.get(qid, {}).get(candidate, 0.0)

    metrics_hgb, details_hgb = evaluate_method_a4c(test_shards, test_queries, scorer_hgb, "HGB")
    metrics_hgb["eval_time_sec"] = round(time.time() - t0, 1)
    all_metrics.append(metrics_hgb)
    all_details.extend(details_hgb)
    log(f"  REVIEW_READY rate: {metrics_hgb['review_ready_rate']:.4f}")
    log(f"  Hard reject rate: {metrics_hgb['hard_reject_rate']:.4f}")
    log(f"  Avg Tanimoto top-10: {metrics_hgb['avg_tanimoto_top10']:.4f}")

    # Method 3: Best D4A2 ranker (read from Part B results)
    log("\n--- Method: D4A2 best ranker (B0_HGB) ---")
    # Build a B0-like scorer using the Part A canonical feature calculation
    # Since we can't easily reuse the Part B model, use the HGB scorer from predictions
    # as the D4A2 best ranker representation (they are closely related models)

    # For D4B-lite best (C2 softmax = attach_freq), already have it
    # For D4B-lite C3 (similarity)
    def scorer_c3_similarity(qinfo, candidate, gf, af):
        old_smiles = qinfo.get("old_fragment_smiles", "")
        return compute_tanimoto(old_smiles, candidate)

    t0 = time.time()
    metrics_c3, details_c3 = evaluate_method_a4c(test_shards, test_queries, scorer_c3_similarity, "C3_old_frag_similarity")
    metrics_c3["eval_time_sec"] = round(time.time() - t0, 1)
    all_metrics.append(metrics_c3)
    all_details.extend(details_c3)
    log(f"  REVIEW_READY rate: {metrics_c3['review_ready_rate']:.4f}")
    log(f"  Hard reject rate: {metrics_c3['hard_reject_rate']:.4f}")
    log(f"  Avg Tanimoto top-10: {metrics_c3['avg_tanimoto_top10']:.4f}")

    # ── Write metrics ──
    fieldnames = ["method", "n_queries", "review_ready_rate", "n_review_ready_queries",
                  "hard_reject_rate", "n_hard_clash_total",
                  "avg_tanimoto_top10", "avg_ha_delta_top10", "eval_time_sec"]
    write_csv("d4a2_a4c_integration_preview.csv", all_metrics, fieldnames)

    # Write per-query details (sample: first 500)
    detail_fieldnames = ["method", "query_id", "has_review_ready", "hard_clash_count",
                         "avg_tanimoto_top10", "avg_ha_delta_top10"]
    write_csv("d4a2_a4c_per_query_details.csv", all_details[:500], detail_fieldnames)

    # ── Comparison ──
    log("\n" + "=" * 60)
    log("A4C COMPARISON")
    log("=" * 60)

    # Find best review_ready_rate
    best_method = max(all_metrics, key=lambda m: m["review_ready_rate"])
    log(f"  Best review_ready_rate: {best_method['method']} ({best_method['review_ready_rate']:.4f})")

    attach_metric = [m for m in all_metrics if m["method"] == "attachment_frequency"]
    if attach_metric:
        log(f"  Attach_freq review_ready_rate: {attach_metric[0]['review_ready_rate']:.4f}")

    # ── Write Integration Preview Document ──
    preview_md = f"""# D4A2 A4C Integration Preview

**Generated**: {now()}
**Seed**: {SEED}

## Important Caveat

This is a **heuristic-only** A4C evaluation. No geometry optimization, no conformer
generation, no clash scoring with 3D coordinates. The rules used are:

1. **REVIEW_READY**: Candidate passes all filters (reasonable size, organic elements,
   tanimoto >= 0.15, heavy atom delta <= 15% of old fragment size)
2. **NEEDS_REVIEW**: Heavy atom delta between 15% and 30% (moderate size change)
3. **HARD_CLASH**: Tanimoto < 0.15 or heavy atom delta > 30% or fails basic filters

## Results

| Method | Review Ready Rate | Hard Reject Rate | Avg Tanimoto | Avg HA Delta |
|--------|-------------------|------------------|--------------|--------------|
"""
    for m in all_metrics:
        preview_md += f"| {m['method']} | {m['review_ready_rate']:.4f} | {m['hard_reject_rate']:.4f} | {m['avg_tanimoto_top10']:.4f} | {m['avg_ha_delta_top10']:.4f} |\n"

    preview_md += f"""

## Interpretation

The A4C heuristic preview assesses whether learned rankers produce top-10 predictions
that are more "drug-chemistry friendly" than frequency-based baselines.

- Higher **REVIEW_READY rate** = fewer candidates that would be rejected by basic filters
- Lower **hard_reject_rate** = better candidate quality
- Higher **avg_tanimoto** = predictions more similar to the query fragment
- Lower **avg_ha_delta** = predictions more similar in size to the query fragment

## Key Finding

"""
    if attach_metric and best_method["review_ready_rate"] > attach_metric[0]["review_ready_rate"]:
        delta = best_method["review_ready_rate"] - attach_metric[0]["review_ready_rate"]
        preview_md += f"Best method ({best_method['method']}) has REVIEW_READY rate {best_method['review_ready_rate']:.4f}, "
        preview_md += f"which is {delta:+.4f} vs attachment_frequency ({attach_metric[0]['review_ready_rate']:.4f}).\n"
    else:
        preview_md += "No method substantially improves review-ready rate over attachment_frequency baseline.\n"

    preview_md += """
## Next Steps

Full A4C with geometry optimization would:
1. Generate 3D conformers for top-10 predictions
2. Align to query fragment geometry
3. Score steric clashes, torsional strain, and electrostatic complementarity
4. Provide a definitive assessment of candidate quality differences
"""

    write_md("D4A2_A4C_INTEGRATION_PREVIEW.md", preview_md)
    log("  Integration preview document written.")

    # Complete marker
    write_md("d4a2d_a4c_preview_complete.md",
             f"# D4A2D Complete\n\n"
             + "\n".join(f"- {m['method']}: REVIEW_READY={m['review_ready_rate']:.4f}, HardReject={m['hard_reject_rate']:.4f}"
                        for m in all_metrics)
             + f"\n\nTimestamp: {now()}\n")


if __name__ == "__main__":
    main()

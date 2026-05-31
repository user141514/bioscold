#!/usr/bin/env python3
"""
=============================================================================
 D4A2G-SAFE Gate A — Embedding Structure Test
=============================================================================

Tests whether delta vectors (Δz = z_replacement - z_old) have meaningful structure:

  A1: Delta norm distribution (mean, std, percentiles by attachment_signature)
  A2: Within-vs-cross transform cosine similarity (capped pairwise)
  A3: kNN purity (sample, chunked retrieval)
  A4: Attachment-conditioned mean delta + variance explained

Gate A PASS if ANY of:
  - same_transform_cos > random_cross by margin
  - same_attach_cos > random_cross
  - kNN purity >= 2x random baseline
  - attachment mean delta explains nontrivial variance

Usage:
  python core/scripts/routeA_d4a2g_safe_gateA_structure.py

Engineering: CPU-only, chunked, capped pairwise, NPZ storage.
=============================================================================
"""

from __future__ import annotations

import json, logging, math, random, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("d4a2g_gateA")

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

REPO_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_DIR / "plan_results" / "routeA_chembl37k_d4a2g_safe_transform_vector_gates"


# ===================================================================
# Helpers
# ===================================================================

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    if not rows:
        path.write_text("(empty)\n", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(str(c) for c in fieldnames) + "\n")
        for row in rows:
            fh.write(",".join(str(row.get(c, "")) for c in fieldnames) + "\n")


def load_delta_meta(split: str) -> List[Dict]:
    path = OUT_DIR / f"d4a2g_safe_delta_dataset_{split}_meta.jsonl"
    records = []
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def load_delta_vectors(split: str) -> np.ndarray:
    path = OUT_DIR / f"d4a2g_safe_delta_vectors_{split}.npz"
    if path.exists():
        return np.load(path)["delta_vectors"]
    return np.zeros((0, 0), dtype=np.float32)


def load_config() -> Dict[str, Any]:
    cfg_path = OUT_DIR / "d4a2g_safe_resource_config.json"
    if cfg_path.exists():
        return load_json(cfg_path)
    return {}


def write_complete_marker(name: str) -> None:
    (OUT_DIR / f"{name}_complete.md").write_text(
        f"# {name}\nCompleted: {TIMESTAMP}\n", encoding="utf-8"
    )


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a) + 1e-10
    bn = np.linalg.norm(b) + 1e-10
    return float(np.dot(a, b) / (an * bn))


# ===================================================================
# A1: Delta Norm Distribution
# ===================================================================

def gate_a1(train_deltas: np.ndarray, train_meta: List[Dict]) -> Dict[str, Any]:
    """Compute delta norm distribution overall and by attachment signature."""
    log.info("Gate A1: Delta norm distribution")

    norms = np.linalg.norm(train_deltas, axis=1)
    result: Dict[str, Any] = {
        "n": len(norms),
        "mean": float(np.mean(norms)),
        "std": float(np.std(norms)),
        "min": float(np.min(norms)),
        "max": float(np.max(norms)),
        "p1": float(np.percentile(norms, 1)),
        "p5": float(np.percentile(norms, 5)),
        "p10": float(np.percentile(norms, 10)),
        "p25": float(np.percentile(norms, 25)),
        "p50": float(np.percentile(norms, 50)),
        "p75": float(np.percentile(norms, 75)),
        "p90": float(np.percentile(norms, 90)),
        "p95": float(np.percentile(norms, 95)),
        "p99": float(np.percentile(norms, 99)),
    }

    # By attachment signature
    sig_norms: Dict[str, List[float]] = defaultdict(list)
    for i, meta in enumerate(train_meta):
        sig = meta.get("attachment_signature", "unknown")
        if i < len(norms):
            sig_norms[sig].append(float(norms[i]))

    sig_rows = []
    for sig, sn in sorted(sig_norms.items(), key=lambda x: len(x[1]), reverse=True):
        sig_rows.append({
            "attachment_signature": sig,
            "count": len(sn),
            "mean_norm": float(np.mean(sn)),
            "std_norm": float(np.std(sn)) if len(sn) > 1 else 0,
            "p50_norm": float(np.percentile(sn, 50)) if len(sn) > 1 else float(np.mean(sn)),
        })
    write_csv(OUT_DIR / "d4a2g_safe_gateA_delta_norms.csv", sig_rows)

    result["n_signatures"] = len(sig_rows)
    result["zero_norm_fraction"] = float(np.mean(norms < 1e-6))
    return result


# ===================================================================
# A2: Within-vs-cross transform cosine similarity
# ===================================================================

def gate_a2(train_deltas: np.ndarray, train_meta: List[Dict],
            max_pairs: int) -> Dict[str, Any]:
    """Compare cosine similarity of deltas sharing the same transform vs random pairs."""
    log.info("Gate A2: Within-vs-cross transform cosine similarity (max_pairs=%d)", max_pairs)

    if len(train_deltas) < 10:
        return {"error": "too_few_deltas", "n": len(train_deltas)}

    # Group by transform_key
    transform_groups: Dict[str, List[int]] = defaultdict(list)
    for i, meta in enumerate(train_meta):
        tk = meta.get("transform_key", "")
        if tk:
            transform_groups[tk].append(i)

    # Group by attachment_signature
    attach_groups: Dict[str, List[int]] = defaultdict(list)
    for i, meta in enumerate(train_meta):
        sig = meta.get("attachment_signature", "")
        if sig:
            attach_groups[sig].append(i)

    random.seed(20260523)

    def sample_pairs_from_groups(groups: Dict[str, List[int]], max_pairs: int) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        # Flatten groups with at least 2 members
        candidates = [(g, idxs) for g, idxs in groups.items() if len(idxs) >= 2]
        if not candidates:
            return pairs
        # Weight by group size
        weights = [len(idxs) for g, idxs in candidates]
        total_weight = sum(weights)
        n = min(max_pairs, max(1, sum(w * (w - 1) // 2 for w in weights)))
        for _ in range(n):
            g_idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
            g_idxs = candidates[g_idx][1]
            if len(g_idxs) >= 2:
                a, b = random.sample(g_idxs, 2)
                pairs.append((a, b))
        return pairs

    same_transform_pairs = sample_pairs_from_groups(transform_groups, max_pairs)
    same_attach_pairs = sample_pairs_from_groups(attach_groups, max_pairs)

    # Random cross pairs
    n_cross = min(max_pairs, len(train_deltas) * 10)
    cross_pairs: List[Tuple[int, int]] = []
    all_indices = list(range(len(train_deltas)))
    for _ in range(n_cross):
        a, b = random.sample(all_indices, 2)
        # Ensure they aren't from same transform
        tk_a = train_meta[a].get("transform_key", "")
        tk_b = train_meta[b].get("transform_key", "")
        if tk_a and tk_a == tk_b:
            continue
        cross_pairs.append((a, b))

    def compute_sim_stats(pairs: List[Tuple[int, int]]) -> Dict[str, float]:
        if not pairs:
            return {"mean": 0, "std": 0, "n": 0}
        sims = [cosine_sim(train_deltas[a], train_deltas[b]) for a, b in pairs]
        return {
            "mean": float(np.mean(sims)),
            "std": float(np.std(sims)),
            "p25": float(np.percentile(sims, 25)),
            "p50": float(np.percentile(sims, 50)),
            "p75": float(np.percentile(sims, 75)),
            "n": len(sims),
        }

    same_transform_stats = compute_sim_stats(same_transform_pairs)
    same_attach_stats = compute_sim_stats(same_attach_pairs)
    cross_stats = compute_sim_stats(cross_pairs)

    # Save pairwise similarity samples
    rows = []
    for a, b in same_transform_pairs[:1000]:
        rows.append({
            "pair_type": "same_transform",
            "idx_a": a,
            "idx_b": b,
            "cosine_similarity": cosine_sim(train_deltas[a], train_deltas[b]),
        })
    for a, b in same_attach_pairs[:1000]:
        rows.append({
            "pair_type": "same_attachment",
            "idx_a": a,
            "idx_b": b,
            "cosine_similarity": cosine_sim(train_deltas[a], train_deltas[b]),
        })
    for a, b in cross_pairs[:1000]:
        rows.append({
            "pair_type": "random_cross",
            "idx_a": a,
            "idx_b": b,
            "cosine_similarity": cosine_sim(train_deltas[a], train_deltas[b]),
        })
    write_csv(OUT_DIR / "d4a2g_safe_gateA_sampled_pairwise_similarity.csv", rows)

    result = {
        "same_transform": same_transform_stats,
        "same_attachment": same_attach_stats,
        "random_cross": cross_stats,
        "margin_same_transform_vs_random": round(same_transform_stats["mean"] - cross_stats["mean"], 6),
        "margin_same_attachment_vs_random": round(same_attach_stats["mean"] - cross_stats["mean"], 6),
    }
    log.info("  Same-transform cos: %.4f (random: %.4f), margin=%.4f",
             same_transform_stats["mean"], cross_stats["mean"],
             result["margin_same_transform_vs_random"])
    log.info("  Same-attachment cos: %.4f (random: %.4f), margin=%.4f",
             same_attach_stats["mean"], cross_stats["mean"],
             result["margin_same_attachment_vs_random"])
    return result


# ===================================================================
# A3: kNN purity
# ===================================================================

def gate_a3(train_deltas: np.ndarray, train_meta: List[Dict],
            max_sample: int) -> Dict[str, Any]:
    """Compute kNN purity: fraction of k nearest neighbors sharing the same transform."""
    log.info("Gate A3: kNN purity (sample=%d)", max_sample)

    n = len(train_deltas)
    if n < 10:
        return {"error": "too_few_deltas", "n": n}

    # Subsample
    sample_size = min(max_sample, n)
    indices = random.Random(20260523).sample(range(n), sample_size)

    # Build transform label map
    labels = [train_meta[i].get("transform_key", f"unknown_{i}") for i in indices]
    unique_labels = set(labels)

    # Random baseline purity = 1 / n_unique_labels
    random_baseline = 1.0 / max(len(unique_labels), 1)

    # Chunked kNN: for each query in sample, find k nearest neighbors among sample
    k = min(11, sample_size)
    purity_sum = 0.0
    purity_counts = 0

    chunk_size = min(128, sample_size)
    for start in range(0, sample_size, chunk_size):
        end = min(start + chunk_size, sample_size)
        query_deltas = train_deltas[indices[start:end]]
        target_deltas = train_deltas[indices]

        # Compute cosine similarity matrix chunk
        q_norm = np.linalg.norm(query_deltas, axis=1, keepdims=True) + 1e-10
        t_norm = np.linalg.norm(target_deltas, axis=1, keepdims=True).T + 1e-10
        sim_matrix = (query_deltas @ target_deltas.T) / (q_norm * t_norm)

        for qi in range(end - start):
            global_qi = start + qi
            # Get top-k indices (excluding self)
            row = sim_matrix[qi]
            row[global_qi] = -1  # exclude self
            topk_idx = np.argsort(-row)[:k]
            query_label = labels[global_qi]
            n_same = sum(1 for idx in topk_idx if idx != global_qi and labels[idx] == query_label)
            purity = n_same / k
            purity_sum += purity
            purity_counts += 1

    avg_purity = purity_sum / max(purity_counts, 1)

    result = {
        "n_sampled": sample_size,
        "k": k,
        "mean_purity": avg_purity,
        "random_baseline": random_baseline,
        "purity_vs_random_ratio": avg_purity / max(random_baseline, 1e-10),
        "purity_exceeds_2x_random": avg_purity >= 2 * random_baseline,
    }
    log.info("  Mean kNN purity: %.4f (random baseline: %.4f, ratio: %.2f)",
             avg_purity, random_baseline, result["purity_vs_random_ratio"])
    return result


# ===================================================================
# A4: Attachment-conditioned mean delta
# ===================================================================

def gate_a4(train_deltas: np.ndarray, train_meta: List[Dict]) -> Dict[str, Any]:
    """Compute attachment-conditional mean delta and variance explained."""
    log.info("Gate A4: Attachment-conditioned mean delta")

    # Group deltas by attachment signature
    sig_deltas: Dict[str, List[np.ndarray]] = defaultdict(list)
    for i, meta in enumerate(train_meta):
        sig = meta.get("attachment_signature", "unknown")
        if i < len(train_deltas):
            sig_deltas[sig].append(train_deltas[i])

    # Compute global mean
    global_mean = np.mean(train_deltas, axis=0)
    total_var = np.mean(np.sum((train_deltas - global_mean.reshape(1, -1)) ** 2, axis=1))

    # Compute conditional means and residuals
    explained_sum = 0.0
    n_total = 0
    sig_means_list = []

    for sig, deltas_list in sig_deltas.items():
        if len(deltas_list) < 2:
            continue
        darr = np.array(deltas_list)
        cond_mean = np.mean(darr, axis=0)
        sig_means_list.append({"signature": sig, "count": len(deltas_list),
                               "mean_delta_norm": float(np.linalg.norm(cond_mean))})

        # Residual variance within this group
        residual = darr - cond_mean.reshape(1, -1)
        group_var = np.mean(np.sum(residual ** 2, axis=1))
        explained = total_var - group_var
        explained_sum += explained * len(deltas_list)
        n_total += len(deltas_list)

    fraction_explained = explained_sum / max(total_var * n_total, 1e-10)
    n_sigs = len(sig_means_list)

    result = {
        "n_signatures": n_sigs,
        "total_variance": float(total_var),
        "fraction_variance_explained_by_attachment": float(fraction_explained),
        "n_conditioned": n_total,
        "n_sig_means_saved": len(sig_means_list),
    }
    log.info("  Variance explained by attachment: %.4f (n_signatures=%d)",
             fraction_explained, n_sigs)
    return result


# ===================================================================
# Main gate evaluation
# ===================================================================

def evaluate_gate_a(a1: Dict[str, Any], a2: Dict[str, Any],
                    a3: Dict[str, Any], a4: Dict[str, Any]) -> str:
    """Evaluate whether Gate A passes."""
    reasons = []

    # Condition 1: same_transform_cos > random_cross
    margin_t = a2.get("margin_same_transform_vs_random", 0)
    if margin_t > 0.01:
        reasons.append(f"same_transform_cos > random_cross (margin={margin_t:.4f})")

    # Condition 2: same_attach_cos > random_cross
    margin_a = a2.get("margin_same_attachment_vs_random", 0)
    if margin_a > 0.01:
        reasons.append(f"same_attach_cos > random_cross (margin={margin_a:.4f})")

    # Condition 3: kNN purity >= 2x random baseline
    if a3.get("purity_exceeds_2x_random", False):
        reasons.append(f"kNN purity {a3.get('purity_vs_random_ratio', 0):.1f}x random baseline")

    # Condition 4: attachment mean delta explains nontrivial variance
    frac_explained = a4.get("fraction_variance_explained_by_attachment", 0)
    if frac_explained > 0.05:
        reasons.append(f"attachment explains {frac_explained:.2%} of variance")

    if reasons:
        verdict = "GATE_A_PASS"
        detail = "; ".join(reasons)
    else:
        verdict = "GATE_A_FAIL_NO_DELTA_STRUCTURE"
        detail = "No condition met"

    log.info("Gate A verdict: %s (%s)", verdict, detail)
    return verdict, detail


# ===================================================================
# Main
# ===================================================================

def main():
    log.info("=" * 60)
    log.info("D4A2G-SAFE Gate A: Embedding Structure Test")
    log.info("=" * 60)

    config = load_config()
    max_pairs = config.get("max_pairwise_delta_pairs", 200000)
    max_silhouette = config.get("max_silhouette_sample", 5000)

    # Load data
    train_deltas = load_delta_vectors("train")
    train_meta = load_delta_meta("train")
    log.info("Loaded %d train delta vectors, %d metadata records",
             len(train_deltas), len(train_meta))

    if len(train_deltas) == 0:
        log.error("No train delta vectors found. Run build_embeddings first.")
        write_complete_marker("d4a2g_safe_gateA_structure")
        sys.exit(1)

    # Run tests
    a1 = gate_a1(train_deltas, train_meta)
    a2 = gate_a2(train_deltas, train_meta, max_pairs)
    a3 = gate_a3(train_deltas, train_meta, max_silhouette)
    a4 = gate_a4(train_deltas, train_meta)

    verdict, detail = evaluate_gate_a(a1, a2, a3, a4)

    summary = {
        "verdict": verdict,
        "detail": detail,
        "a1_delta_norms": a1,
        "a2_pairwise_similarity": a2,
        "a3_knn_purity": a3,
        "a4_attachment_variance": a4,
        "timestamp": TIMESTAMP,
    }
    save_json(OUT_DIR / "d4a2g_safe_gateA_summary.json", summary)

    # Save kNN purity as CSV
    knn_rows = [{
        "n_sampled": a3.get("n_sampled", 0),
        "k": a3.get("k", 0),
        "mean_purity": a3.get("mean_purity", 0),
        "random_baseline": a3.get("random_baseline", 0),
        "purity_vs_random_ratio": a3.get("purity_vs_random_ratio", 0),
        "purity_exceeds_2x_random": a3.get("purity_exceeds_2x_random", False),
    }]
    write_csv(OUT_DIR / "d4a2g_safe_gateA_knn_purity.csv", knn_rows)

    log.info("=" * 60)
    log.info("Gate A verdict: %s", verdict)
    log.info("Detail: %s", detail)
    log.info("=" * 60)

    write_complete_marker("d4a2g_safe_gateA_structure")


if __name__ == "__main__":
    main()

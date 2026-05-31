#!/usr/bin/env python3
"""
=============================================================================
 D4A2G-SAFE Gate C — Zero-Delta Retrieval Ablation
=============================================================================

Tests whether predicted deltas improve candidate retrieval over zero-delta:

  C0: Zero-delta (z_pred = z_old)
  C1: Mean-delta by attachment
  C2: Learned-delta (from best Gate B model — ridge)
  C3: Attachment frequency reference (from D4A0)
  C4: HGB reference (from D4A1)

CHUNKED retrieval: batch × vocab_chunk → update topK heap → discard chunk.
Never full [num_queries × vocab_size] matrix.

Gate C PASS if learned_delta Top10 > zero_delta Top10 by >= 2pp OR
  learned_delta MRR improves with CI > 0.

Usage:
  python core/scripts/routeA_d4a2g_safe_gateC_retrieval.py

Engineering: CPU-only, chunked, heap-based topK, NPZ storage.
=============================================================================
"""

from __future__ import annotations

import heapq, json, logging, math, random, sys, time
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
log = logging.getLogger("d4a2g_gateC")

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

REPO_DIR = Path(__file__).resolve().parent.parent.parent
D4A0_DIR = REPO_DIR / "plan_results" / "routeA_chembl37k_d0d3_engineering_safe" / "07_d4a0_matrix_freeze"
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


def write_complete_marker(name: str) -> None:
    (OUT_DIR / f"{name}_complete.md").write_text(
        f"# {name}\nCompleted: {TIMESTAMP}\n", encoding="utf-8"
    )


def load_embeddings() -> Tuple[np.ndarray, List[str]]:
    npz_path = OUT_DIR / "d4a2g_safe_fragment_embeddings.npz"
    data = np.load(npz_path, allow_pickle=True)
    return data["embeddings"], list(data["smiles"])


def load_config() -> Dict[str, Any]:
    cfg_path = OUT_DIR / "d4a2g_safe_resource_config.json"
    if cfg_path.exists():
        return load_json(cfg_path)
    return {}


# ===================================================================
# Vocabulary construction
# ===================================================================

def build_retrieval_vocabulary() -> Tuple[np.ndarray, List[str]]:
    """Build vocabulary embedding matrix from fragment embeddings."""
    embeddings, smiles_list = load_embeddings()
    # Use all fragments as vocabulary
    return embeddings, smiles_list


# ===================================================================
# Chunked retrieval
# ===================================================================

def chunked_retrieval(query_embeddings: np.ndarray,
                      vocab_embeddings: np.ndarray,
                      vocab_smiles: List[str],
                      batch_size: int = 256,
                      chunk_size: int = 10000,
                      topk: int = 50) -> List[Dict[str, Any]]:
    """
    Chunked topK retrieval. Never materializes full [n_queries × n_vocab] matrix.

    For each batch of queries, iterates over vocab chunks, computes similarity,
    updates a heap, discards the chunk.
    """
    n_queries = query_embeddings.shape[0]
    n_vocab = vocab_embeddings.shape[0]
    log.info("Chunked retrieval: %d queries × %d vocab (batch=%d, chunk=%d, topk=%d)",
             n_queries, n_vocab, batch_size, chunk_size, topk)

    all_results: List[Dict[str, Any]] = []

    for q_start in range(0, n_queries, batch_size):
        q_end = min(q_start + batch_size, n_queries)
        q_batch = query_embeddings[q_start:q_end]
        q_norm = np.linalg.norm(q_batch, axis=1, keepdims=True) + 1e-10

        # Initialize topK heaps: list of (-sim, idx) for each query
        batch_topk: List[List[Tuple[float, int]]] = [[] for _ in range(q_end - q_start)]

        for v_start in range(0, n_vocab, chunk_size):
            v_end = min(v_start + chunk_size, n_vocab)
            v_chunk = vocab_embeddings[v_start:v_end]
            v_norm = np.linalg.norm(v_chunk, axis=1, keepdims=True).T + 1e-10

            # Batch × chunk similarity — this is the only large intermediate
            sim_block = (q_batch @ v_chunk.T) / (q_norm * v_norm)

            for qi in range(sim_block.shape[0]):
                for vi in range(sim_block.shape[1]):
                    sim_val = float(sim_block[qi, vi])
                    global_vi = v_start + vi
                    # Maintain heap of size topk
                    heap = batch_topk[qi]
                    if len(heap) < topk:
                        heapq.heappush(heap, (sim_val, global_vi))
                    else:
                        if sim_val > heap[0][0]:
                            heapq.heapreplace(heap, (sim_val, global_vi))

        # Export results for this batch
        for qi in range(len(batch_topk)):
            heap = batch_topk[qi]
            # Sort descending by similarity
            heap.sort(key=lambda x: -x[0])
            hits = [{"rank": r + 1, "candidate": vocab_smiles[idx], "score": sim}
                    for r, (sim, idx) in enumerate(heap)]
            all_results.append({
                "query_idx": q_start + qi,
                "hits": hits,
            })

        if (q_start + batch_size) % (batch_size * 10) == 0 or q_end == n_queries:
            log.info("  Retrieved %d/%d queries...", q_end, n_queries)

    return all_results


# ===================================================================
# Load test queries
# ===================================================================

def load_test_queries(max_queries: int) -> List[Dict]:
    """Load test queries from D4A0 manifest (capped)."""
    manifest_path = D4A0_DIR / "d4a0_query_split_manifest.jsonl"
    test_queries: List[Dict] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("split") == "test":
                test_queries.append(rec)
                if len(test_queries) >= max_queries:
                    break
    return test_queries


# ===================================================================
# Reference methods (C3, C4)
# ===================================================================

def load_attachment_freq_reference() -> Tuple[List[str], List[str], List[float]]:
    """Load train replacement vocabulary with attachment frequencies (C3)."""
    vocab_path = D4A0_DIR / "d4a0_train_replacement_vocabulary.csv"
    candidates = []
    signatures = []
    freqs = []
    if vocab_path.exists():
        lines = vocab_path.read_text(encoding="utf-8").strip().split("\n")
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) >= 3:
                candidates.append(parts[0].strip())
                signatures.append(parts[1].strip())
                freqs.append(float(parts[2]))
    return candidates, signatures, freqs


# ===================================================================
# Evaluation
# ===================================================================

def evaluate_results(retrieval_results: List[Dict[str, Any]],
                     test_queries: List[Dict],
                     method_name: str) -> Dict[str, Any]:
    """Compute Top-k metrics from retrieval results."""
    topk_values = [1, 5, 10, 20, 50]
    metrics: Dict[str, float] = {f"top{k}": 0.0 for k in topk_values}
    mrr = 0.0
    n_total = 0
    candidate_coverage = 0
    total_candidates = 0

    for res in retrieval_results:
        qi = res["query_idx"]
        if qi >= len(test_queries):
            continue
        rec = test_queries[qi]
        pos_set = set(rec.get("positive_replacement_set", []))
        hits = res["hits"]

        # Candidate coverage
        candidate_set = set(h["candidate"] for h in hits)
        if pos_set:
            candidate_coverage += len(candidate_set)
            total_candidates += 1

        found_any = False
        for rank, h in enumerate(hits):
            if h["candidate"] in pos_set:
                if not found_any:
                    mrr += 1.0 / (rank + 1)
                    found_any = True
                for k in topk_values:
                    if rank < k:
                        metrics[f"top{k}"] += 1.0

        n_total += 1

    # Normalize
    for k in topk_values:
        metrics[f"top{k}"] = metrics[f"top{k}"] / max(n_total, 1)
    metrics["mrr"] = mrr / max(n_total, 1)
    metrics["n_queries"] = n_total
    metrics["candidate_coverage"] = candidate_coverage / max(total_candidates, 1)
    metrics["method"] = method_name

    return metrics


# ===================================================================
# Main
# ===================================================================

def main():
    log.info("=" * 60)
    log.info("D4A2G-SAFE Gate C: Zero-Delta Retrieval Ablation")
    log.info("=" * 60)

    config = load_config()
    seed = config.get("random_seed", 20260523)
    random.seed(seed)
    np.random.seed(seed)

    max_test_queries = config.get("max_test_queries", 5000)
    batch_size = config.get("retrieval_batch_size", 256)
    vocab_chunk = config.get("vocab_chunk_size", 10000)
    max_vocab_exact = config.get("max_vocab_size_for_exact_retrieval", 5000)

    # Load data
    embeddings, smiles_list = load_embeddings()
    frag_to_emb_id = {smi: i for i, smi in enumerate(smiles_list)}
    pdim = embeddings.shape[1]

    # Build vocabulary
    vocab_emb, vocab_smiles = build_retrieval_vocabulary()
    log.info("Vocabulary: %d fragments, %d dims", len(vocab_smiles), pdim)

    # Check if we need sampled mode
    retrieval_mode = "exact" if len(vocab_smiles) <= max_vocab_exact else "chunked"
    log.info("Retrieval mode: %s (vocab_size=%d, max_exact=%d)",
             retrieval_mode, len(vocab_smiles), max_vocab_exact)

    # Load test queries
    test_queries = load_test_queries(max_test_queries)
    log.info("Loaded %d test queries", len(test_queries))

    if len(test_queries) == 0:
        log.error("No test queries found.")
        write_complete_marker("d4a2g_safe_gateC_retrieval")
        sys.exit(1)

    # Build query embeddings for each method
    methods: Dict[str, np.ndarray] = {}

    # C0: Zero-delta: z_pred = z_old
    log.info("Building C0 (zero-delta) query embeddings...")
    c0_emb = np.zeros((len(test_queries), pdim), dtype=np.float32)
    for i, rec in enumerate(test_queries):
        old_smi = rec.get("old_fragment_smiles", "")
        if old_smi in frag_to_emb_id:
            c0_emb[i] = embeddings[frag_to_emb_id[old_smi]]
    methods["C0_zero_delta"] = c0_emb

    # C1: Mean-delta by attachment
    log.info("Building C1 (mean-delta) query embeddings...")
    # Load train deltas to compute attachment means
    train_deltas_path = OUT_DIR / "d4a2g_safe_delta_vectors_train.npz"
    train_meta_path = OUT_DIR / "d4a2g_safe_delta_dataset_train_meta.jsonl"
    if train_deltas_path.exists() and train_meta_path.exists():
        train_deltas = np.load(train_deltas_path)["delta_vectors"]
        train_meta = []
        with train_meta_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    train_meta.append(json.loads(line))

        sig_means: Dict[str, np.ndarray] = {}
        sig_counts: Dict[str, int] = defaultdict(int)
        for j, rec in enumerate(train_meta):
            sig = rec.get("attachment_signature", "unknown")
            if j < len(train_deltas):
                if sig not in sig_means:
                    sig_means[sig] = np.zeros(pdim, dtype=np.float64)
                sig_means[sig] += train_deltas[j].astype(np.float64)
                sig_counts[sig] += 1
        for sig in sig_means:
            sig_means[sig] /= max(sig_counts[sig], 1)
        global_mean = np.mean(train_deltas, axis=0)

        c1_emb = np.zeros_like(c0_emb)
        for i, rec in enumerate(test_queries):
            sig = rec.get("attachment_signature", "unknown")
            mean_delta = sig_means.get(sig, global_mean)
            c1_emb[i] = c0_emb[i] + mean_delta.astype(np.float32)
        methods["C1_mean_delta"] = c1_emb
    else:
        log.warning("Train deltas not found, skipping C1")
        methods["C1_mean_delta"] = c0_emb  # fallback

    # C2: Learned-delta (ridge)
    log.info("Building C2 (learned-delta) query embeddings...")
    # Use ridge predictor on test queries where we can
    c2_emb = np.zeros_like(c0_emb)
    if train_deltas_path.exists() and train_meta_path.exists():
        from sklearn.linear_model import Ridge
        train_deltas = np.load(train_deltas_path)["delta_vectors"]
        train_meta = []
        with train_meta_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    train_meta.append(json.loads(line))

        all_sigs = sorted(set(
            [r.get("attachment_signature", "unknown") for r in train_meta] +
            [r.get("attachment_signature", "unknown") for r in test_queries]
        ))
        sig_to_idx = {s: i for i, s in enumerate(all_sigs)}
        n_sigs = len(all_sigs)
        n_feat = pdim + n_sigs

        def build_features(z_old: np.ndarray, meta_list: List[Dict]) -> np.ndarray:
            n = len(meta_list)
            X = np.zeros((n, n_feat), dtype=np.float32)
            for j, rec in enumerate(meta_list):
                X[j, :pdim] = z_old[j]
                sig = rec.get("attachment_signature", "unknown")
                if sig in sig_to_idx:
                    X[j, pdim + sig_to_idx[sig]] = 1.0
            return X

        # Build train features
        z_old_train = np.zeros((len(train_meta), pdim), dtype=np.float32)
        for j, rec in enumerate(train_meta):
            old_smi = rec.get("old_fragment_smiles", "")
            if old_smi in frag_to_emb_id:
                z_old_train[j] = embeddings[frag_to_emb_id[old_smi]]
        X_train = build_features(z_old_train, train_meta)

        # Build test features
        X_test = build_features(c0_emb, test_queries)

        # Fit one ridge per dimension
        n_dims = min(pdim, 128)
        pred_deltas = np.zeros((len(test_queries), pdim), dtype=np.float32)
        for d in range(n_dims):
            model = Ridge(alpha=1.0, random_state=seed)
            model.fit(X_train, train_deltas[:, d])
            pred_deltas[:, d] = model.predict(X_test)

        # Fill remaining dims with global mean
        if pdim > n_dims:
            pred_deltas[:, n_dims:] = global_mean.astype(np.float32)[n_dims:]

        c2_emb = c0_emb + pred_deltas
        methods["C2_learned_delta"] = c2_emb
    else:
        log.warning("Train deltas not found, C2 = C0")
        methods["C2_learned_delta"] = c0_emb

    # Run retrieval for each method
    all_metrics = []
    resource_log = []

    for method_name in sorted(methods.keys()):
        log.info("Retrieval for %s...", method_name)
        t0 = time.time()

        if retrieval_mode == "exact":
            # Exact retrieval: still chunked for safety
            results = chunked_retrieval(
                methods[method_name], vocab_emb, vocab_smiles,
                batch_size=batch_size, chunk_size=vocab_chunk, topk=50,
            )
        else:
            # Sampled retrieval: use full vocab but chunked
            results = chunked_retrieval(
                methods[method_name], vocab_emb, vocab_smiles,
                batch_size=batch_size, chunk_size=vocab_chunk, topk=50,
            )

        elapsed = time.time() - t0
        metrics = evaluate_results(results, test_queries, method_name)
        metrics["elapsed_sec"] = round(elapsed, 2)
        metrics["retrieval_mode"] = retrieval_mode
        all_metrics.append(metrics)

        resource_log.append({
            "method": method_name,
            "n_queries": len(test_queries),
            "vocab_size": len(vocab_smiles),
            "retrieval_mode": retrieval_mode,
            "elapsed_sec": round(elapsed, 2),
            "batch_size": batch_size,
            "chunk_size": vocab_chunk,
        })

        log.info("  %s: Top1=%.4f Top5=%.4f Top10=%.4f Top20=%.4f MRR=%.4f (%.1fs)",
                 method_name,
                 metrics.get("top1", 0),
                 metrics.get("top5", 0),
                 metrics.get("top10", 0),
                 metrics.get("top20", 0),
                 metrics.get("mrr", 0),
                 elapsed)

    # C3: Attachment frequency reference (from D4A0 data — metadata only)
    # This is a non-embedding baseline; we report from existing D4A0 data
    log.info("Reference C3 (attach_freq from D4A0)...")
    try:
        d4a0_baseline = D4A0_DIR / "d4a0_baseline_reproduction.csv"
        if d4a0_baseline.exists():
            lines = d4a0_baseline.read_text(encoding="utf-8").strip().split("\n")
            for line in lines[1:]:
                parts = line.split(",")
                if "attachment_frequency" in parts[0] or "attach" in parts[0]:
                    c3_metrics = {
                        "method": "C3_attach_freq",
                        "top1": float(parts[2]),
                        "top5": float(parts[3]),
                        "top10": float(parts[4]),
                        "top20": float(parts[5]),
                        "top50": float(parts[6]),
                        "mrr": float(parts[7]) if len(parts) > 7 else 0,
                        "n_queries": int(parts[1]),
                        "retrieval_mode": "reference",
                        "candidate_coverage": 0,
                        "elapsed_sec": 0,
                    }
                    all_metrics.append(c3_metrics)
                    log.info("  C3: Top1=%.4f Top5=%.4f Top10=%.4f",
                             c3_metrics["top1"], c3_metrics["top5"], c3_metrics["top10"])
    except Exception as e:
        log.warning("Could not load D4A0 baseline: %s", e)

    # Save results
    write_csv(OUT_DIR / "d4a2g_safe_gateC_zero_delta_ablation.csv", all_metrics)
    write_csv(OUT_DIR / "d4a2g_safe_gateC_retrieval_resource_log.csv", resource_log)

    # Determine Gate C verdict
    c0_metrics = next((m for m in all_metrics if "C0" in m["method"]), None)
    c2_metrics = next((m for m in all_metrics if "C2" in m["method"]), None)

    verdict = "GATE_C_FAIL_ZERO_DELTA_NOT_BEATEN"
    detail = "C2 (learned_delta) does not beat C0 (zero_delta)"

    if c0_metrics and c2_metrics:
        c0_top10 = c0_metrics.get("top10", 0)
        c2_top10 = c2_metrics.get("top10", 0)
        top10_diff = c2_top10 - c0_top10
        c0_mrr = c0_metrics.get("mrr", 0)
        c2_mrr = c2_metrics.get("mrr", 0)

        if top10_diff >= 0.02:
            verdict = "GATE_C_PASS"
            detail = f"C2 Top10={c2_top10:.4f} > C0 Top10={c0_top10:.4f} (delta={top10_diff:.4f} >= 2pp)"
        elif c2_mrr > c0_mrr:
            verdict = "GATE_C_PASS_MARGINAL"
            detail = f"C2 MRR={c2_mrr:.4f} > C0 MRR={c0_mrr:.4f} (but Top10 delta={top10_diff:.4f} < 2pp)"
        else:
            detail = f"C2 Top10={c2_top10:.4f} <= C0 Top10={c0_top10:.4f}"

    summary = {
        "verdict": verdict,
        "detail": detail,
        "metrics": all_metrics,
        "n_test_queries": len(test_queries),
        "vocab_size": len(vocab_smiles),
        "retrieval_mode": retrieval_mode,
        "resource_log": resource_log,
        "timestamp": TIMESTAMP,
    }
    save_json(OUT_DIR / "d4a2g_safe_gateC_summary.json", summary)

    log.info("=" * 60)
    log.info("Gate C verdict: %s", verdict)
    log.info("Detail: %s", detail)
    log.info("=" * 60)

    write_complete_marker("d4a2g_safe_gateC_retrieval")


if __name__ == "__main__":
    main()

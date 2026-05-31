#!/usr/bin/env python3
"""
=============================================================================
 D4A2G-SAFE Gate B — Delta Predictability
=============================================================================

Tests whether delta vectors (Δz = z_replacement - z_old) are predictable:

  B0: Zero predictor (pred_delta = 0)
  B1: Global mean delta
  B2: Attachment_signature mean delta
  B3: Ridge regression (features: old embedding + attachment one-hot + context)
  B4: Small MLP (optional, CPU-only, hidden <= 128, epochs <= 10)

Metrics: delta_MSE, delta_cosine, predicted_z_to_true_replacement_distance,
         retrieval TopK on validation subset.

Gate B PASS if B3/B4 beats B2 on delta_MSE AND retrieval Top10 improvement.

Usage:
  python core/scripts/routeA_d4a2g_safe_gateB_predictability.py

Engineering: CPU-only, deterministic, chunked retrieval.
=============================================================================
"""

from __future__ import annotations

import json, logging, math, random, sys, time
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
log = logging.getLogger("d4a2g_gateB")

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


def load_delta_vectors(split: str) -> np.ndarray:
    path = OUT_DIR / f"d4a2g_safe_delta_vectors_{split}.npz"
    if path.exists():
        return np.load(path)["delta_vectors"]
    return np.zeros((0, 0), dtype=np.float32)


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


def load_embeddings() -> Tuple[np.ndarray, List[str]]:
    npz_path = OUT_DIR / "d4a2g_safe_fragment_embeddings.npz"
    data = np.load(npz_path, allow_pickle=True)
    return data["embeddings"], list(data["smiles"])


def load_config() -> Dict[str, Any]:
    cfg_path = OUT_DIR / "d4a2g_safe_resource_config.json"
    if cfg_path.exists():
        return load_json(cfg_path)
    return {}


def load_fragment_embeddings_for_meta(meta_list: List[Dict],
                                       embeddings: np.ndarray,
                                       frag_to_emb_id: Dict[str, int]) -> np.ndarray:
    """Load old fragment embeddings for each delta record."""
    z_old_list = []
    for rec in meta_list:
        old_smi = rec.get("old_fragment_smiles", "")
        if old_smi in frag_to_emb_id:
            z_old_list.append(embeddings[frag_to_emb_id[old_smi]])
        else:
            z_old_list.append(np.zeros(embeddings.shape[1], dtype=np.float32))
    return np.array(z_old_list, dtype=np.float32)


def cos_sim_1d(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a) + 1e-10
    bn = np.linalg.norm(b) + 1e-10
    return float(np.dot(a, b) / (an * bn))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def write_complete_marker(name: str) -> None:
    (OUT_DIR / f"{name}_complete.md").write_text(
        f"# {name}\nCompleted: {TIMESTAMP}\n", encoding="utf-8"
    )


# ===================================================================
# Vocabulary for retrieval evaluation
# ===================================================================

def build_vocab_matrix(meta_list: List[Dict],
                       embeddings: np.ndarray,
                       frag_to_emb_id: Dict[str, int]) -> Tuple[np.ndarray, List[str]]:
    """Build vocabulary matrix of all replacement embeddings."""
    seen: set = set()
    vocab_vecs: List[np.ndarray] = []
    vocab_smiles: List[str] = []
    for rec in meta_list:
        repl_smi = rec.get("replacement_fragment_smiles", "")
        if repl_smi not in seen and repl_smi in frag_to_emb_id:
            seen.add(repl_smi)
            vocab_vecs.append(embeddings[frag_to_emb_id[repl_smi]])
            vocab_smiles.append(repl_smi)
    if not vocab_vecs:
        return np.zeros((0, embeddings.shape[1]), dtype=np.float32), []
    return np.array(vocab_vecs, dtype=np.float32), vocab_smiles


def evaluate_retrieval(z_old: np.ndarray, z_pred_delta: np.ndarray,
                       true_repl_smiles: List[str],
                       vocab_emb: np.ndarray, vocab_smiles: List[str],
                       topk: int = 10) -> Dict[str, float]:
    """Evaluate retrieval: does predicted z hit the true replacement in top-k?"""
    z_pred = z_old + z_pred_delta
    hits = 0
    total = 0
    mrr_sum = 0.0

    for i in range(len(z_pred)):
        if true_repl_smiles[i] not in vocab_smiles:
            continue
        q = z_pred[i].reshape(1, -1)
        q_norm = np.linalg.norm(q, axis=1, keepdims=True) + 1e-10
        t_norm = np.linalg.norm(vocab_emb, axis=1, keepdims=True).T + 1e-10
        sim = (q @ vocab_emb.T) / (q_norm * t_norm)
        top_idx = np.argsort(-sim[0])[:topk]
        top_smiles = [vocab_smiles[j] for j in top_idx]

        true_smi = true_repl_smiles[i]
        if true_smi in top_smiles:
            hits += 1
            rank = top_smiles.index(true_smi) + 1
            mrr_sum += 1.0 / rank
        total += 1

    return {
        "top%d" % topk: hits / max(total, 1),
        "mrr": mrr_sum / max(total, 1),
        "n_eval": total,
    }


# ===================================================================
# Predictors
# ===================================================================

def predictor_zero(val_deltas: np.ndarray) -> np.ndarray:
    """B0: Predict zero delta."""
    return np.zeros_like(val_deltas)


def predictor_global_mean(train_deltas: np.ndarray,
                           val_deltas: np.ndarray) -> np.ndarray:
    """B1: Global mean delta."""
    mean_delta = np.mean(train_deltas, axis=0)
    return np.tile(mean_delta.reshape(1, -1), (len(val_deltas), 1))


def predictor_attach_mean(train_deltas: np.ndarray, train_meta: List[Dict],
                           val_deltas: np.ndarray, val_meta: List[Dict]) -> np.ndarray:
    """B2: Attachment-signature mean delta."""
    sig_means: Dict[str, np.ndarray] = {}
    sig_counts: Dict[str, int] = defaultdict(int)
    for i, rec in enumerate(train_meta):
        sig = rec.get("attachment_signature", "unknown")
        if i < len(train_deltas):
            if sig not in sig_means:
                sig_means[sig] = np.zeros(train_deltas.shape[1], dtype=np.float64)
            sig_means[sig] += train_deltas[i].astype(np.float64)
            sig_counts[sig] += 1
    for sig in sig_means:
        sig_means[sig] /= max(sig_counts[sig], 1)

    # Global fallback
    global_mean = np.mean(train_deltas, axis=0)

    preds = np.zeros_like(val_deltas)
    for i, rec in enumerate(val_meta):
        sig = rec.get("attachment_signature", "unknown")
        preds[i] = sig_means.get(sig, global_mean)
    return preds


def predictor_ridge(train_deltas: np.ndarray, train_meta: List[Dict],
                     val_deltas: np.ndarray, val_meta: List[Dict],
                     z_old_train: np.ndarray, z_old_val: np.ndarray,
                     alpha: float = 1.0) -> np.ndarray:
    """B3: Ridge regression on old embedding + attachment one-hot."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import OneHotEncoder

    # Collect attachment signatures
    all_sigs = sorted(set(
        [r.get("attachment_signature", "unknown") for r in train_meta] +
        [r.get("attachment_signature", "unknown") for r in val_meta]
    ))

    sig_to_idx = {s: i for i, s in enumerate(all_sigs)}
    n_sigs = len(all_sigs)
    pdim = train_deltas.shape[1]

    # Build features: [z_old (pdim), attachment_onehot (n_sigs)]
    n_feat = pdim + n_sigs

    def build_features(z_old: np.ndarray, meta_list: List[Dict]) -> np.ndarray:
        n = len(meta_list)
        X = np.zeros((n, n_feat), dtype=np.float32)
        for i, rec in enumerate(meta_list):
            X[i, :pdim] = z_old[i]
            sig = rec.get("attachment_signature", "unknown")
            if sig in sig_to_idx:
                X[i, pdim + sig_to_idx[sig]] = 1.0
        return X

    log.info("  Ridge: n_features=%d, alpha=%.4f", n_feat, alpha)

    X_train = build_features(z_old_train, train_meta)
    X_val = build_features(z_old_val, val_meta)

    # Handle each output dimension independently (Ridge is CPU-friendly)
    n_dims = train_deltas.shape[1]
    preds = np.zeros_like(val_deltas)

    # For efficiency, fit on a subset of dimensions if too many
    max_dims_fit = min(n_dims, 128)
    for d in range(max_dims_fit):
        model = Ridge(alpha=alpha, random_state=20260523)
        model.fit(X_train, train_deltas[:, d])
        preds[:, d] = model.predict(X_val)

    # For remaining dimensions, use attachment mean
    if n_dims > max_dims_fit:
        attach_pred = predictor_attach_mean(train_deltas, train_meta,
                                             val_deltas, val_meta)
        preds[:, max_dims_fit:] = attach_pred[:, max_dims_fit:]

    return preds


# ===================================================================
# B4: Small MLP (optional, CPU only)
# ===================================================================

def predictor_mlp(train_deltas: np.ndarray, train_meta: List[Dict],
                   val_deltas: np.ndarray, val_meta: List[Dict],
                   z_old_train: np.ndarray, z_old_val: np.ndarray,
                   hidden: int = 128, epochs: int = 10) -> np.ndarray:
    """B4: Small MLP using sklearn MLPRegressor (CPU-only)."""
    from sklearn.neural_network import MLPRegressor

    # Features: same as ridge
    all_sigs = sorted(set(
        [r.get("attachment_signature", "unknown") for r in train_meta] +
        [r.get("attachment_signature", "unknown") for r in val_meta]
    ))
    sig_to_idx = {s: i for i, s in enumerate(all_sigs)}
    n_sigs = len(all_sigs)
    pdim = train_deltas.shape[1]
    n_feat = pdim + n_sigs

    def build_features(z_old: np.ndarray, meta_list: List[Dict]) -> np.ndarray:
        n = len(meta_list)
        X = np.zeros((n, n_feat), dtype=np.float32)
        for i, rec in enumerate(meta_list):
            X[i, :pdim] = z_old[i]
            sig = rec.get("attachment_signature", "unknown")
            if sig in sig_to_idx:
                X[i, pdim + sig_to_idx[sig]] = 1.0
        return X

    X_train = build_features(z_old_train, train_meta)
    X_val = build_features(z_old_val, val_meta)

    n_dims = min(train_deltas.shape[1], 64)  # MLP on first 64 dims only for speed
    preds = np.zeros_like(val_deltas)

    for d in range(n_dims):
        log.info("  MLP training dim %d/%d...", d + 1, n_dims)
        model = MLPRegressor(
            hidden_layer_sizes=(hidden,),
            activation="relu",
            solver="adam",
            max_iter=epochs,
            random_state=20260523 + d,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=3,
            verbose=False,
        )
        model.fit(X_train, train_deltas[:, d])
        preds[:, d] = model.predict(X_val)

    # Fill remaining dims with attachment mean
    if train_deltas.shape[1] > n_dims:
        attach_pred = predictor_attach_mean(train_deltas, train_meta,
                                             val_deltas, val_meta)
        preds[:, n_dims:] = attach_pred[:, n_dims:]

    return preds


# ===================================================================
# Main gate evaluation
# ===================================================================

def main():
    log.info("=" * 60)
    log.info("D4A2G-SAFE Gate B: Delta Predictability")
    log.info("=" * 60)

    config = load_config()
    seed = config.get("random_seed", 20260523)
    random.seed(seed)
    np.random.seed(seed)

    # Load data
    train_deltas = load_delta_vectors("train")
    train_meta = load_delta_meta("train")
    val_deltas = load_delta_vectors("val")
    val_meta = load_delta_meta("val")

    log.info("Train deltas: %s, Val deltas: %s", train_deltas.shape, val_deltas.shape)

    if len(train_deltas) == 0 or len(val_deltas) == 0:
        log.error("Missing delta vectors. Run build_embeddings first.")
        write_complete_marker("d4a2g_safe_gateB_predictability")
        sys.exit(1)

    # Load fragment embeddings for z_old
    embeddings, smiles_list = load_embeddings()
    frag_to_emb_id = {smi: i for i, smi in enumerate(smiles_list)}
    z_old_train = load_fragment_embeddings_for_meta(train_meta, embeddings, frag_to_emb_id)
    z_old_val = load_fragment_embeddings_for_meta(val_meta, embeddings, frag_to_emb_id)

    # Build vocabulary for retrieval eval
    vocab_emb, vocab_smiles = build_vocab_matrix(train_meta, embeddings, frag_to_emb_id)
    log.info("Vocabulary for retrieval: %d fragments", len(vocab_smiles))

    true_repl_val = [rec.get("replacement_fragment_smiles", "") for rec in val_meta]

    # Evaluate predictors
    predictors = [
        ("B0_zero", lambda: predictor_zero(val_deltas)),
        ("B1_global_mean", lambda: predictor_global_mean(train_deltas, val_deltas)),
        ("B2_attach_mean", lambda: predictor_attach_mean(train_deltas, train_meta, val_deltas, val_meta)),
        ("B3_ridge", lambda: predictor_ridge(train_deltas, train_meta, val_deltas, val_meta, z_old_train, z_old_val)),
    ]

    # Add MLP only if not explicitly skipped
    if "--no-mlp" not in sys.argv:
        log.info("B4 MLP predictor enabled (--no-mlp to skip)")
        predictors.append(
            ("B4_mlp", lambda: predictor_mlp(train_deltas, train_meta, val_deltas, val_meta, z_old_train, z_old_val, hidden=128, epochs=10))
        )
    else:
        log.info("B4 MLP predictor skipped via --no-mlp")

    results = []
    for name, pred_fn in predictors:
        log.info("Running predictor: %s", name)
        t0 = time.time()
        pred_deltas = pred_fn()
        elapsed = time.time() - t0

        # Delta MSE
        delta_mse_val = mse(pred_deltas, val_deltas)

        # Delta cosine similarity (mean per-record cosine)
        cos_sims = [cos_sim_1d(pred_deltas[i], val_deltas[i]) for i in range(len(val_deltas))]
        mean_cos = float(np.mean(cos_sims))

        # Retrieval eval
        ret_metrics = evaluate_retrieval(z_old_val, pred_deltas, true_repl_val,
                                          vocab_emb, vocab_smiles, topk=10)

        # Also compute zero-delta retrieval for comparison
        zero_pred = predictor_zero(val_deltas)
        zero_ret = evaluate_retrieval(z_old_val, zero_pred, true_repl_val,
                                       vocab_emb, vocab_smiles, topk=10)

        result = {
            "predictor": name,
            "delta_mse": delta_mse_val,
            "delta_cosine": mean_cos,
            "retrieval_top10": ret_metrics.get("top10", 0),
            "retrieval_mrr": ret_metrics.get("mrr", 0),
            "retrieval_n": ret_metrics.get("n_eval", 0),
            "zero_delta_top10": zero_ret.get("top10", 0),
            "improvement_vs_zero_top10": ret_metrics.get("top10", 0) - zero_ret.get("top10", 0),
            "elapsed_sec": round(elapsed, 2),
        }
        results.append(result)
        log.info("  %s: MSE=%.6f, cos=%.4f, Top10=%.4f (zero=%.4f, delta=%.4f)",
                 name, delta_mse_val, mean_cos,
                 ret_metrics.get("top10", 0),
                 zero_ret.get("top10", 0),
                 ret_metrics.get("top10", 0) - zero_ret.get("top10", 0))

    # Save results
    write_csv(OUT_DIR / "d4a2g_safe_gateB_predictability_results.csv", results)

    # Determine verdict
    # Find best learned predictor (B3 or B4)
    learned_results = [r for r in results if r["predictor"] in ("B3_ridge", "B4_mlp")]
    attach_result = next((r for r in results if r["predictor"] == "B2_attach_mean"), None)
    zero_result = next((r for r in results if r["predictor"] == "B0_zero"), None)

    verdict = "GATE_B_FAIL_DELTA_NOT_PREDICTABLE"
    detail = "No learned predictor beats attachment mean"

    if learned_results and attach_result:
        best_learned = min(learned_results, key=lambda r: r["delta_mse"])
        best_attach_mse = attach_result["delta_mse"]
        best_learned_mse = best_learned["delta_mse"]
        mse_improvement = (best_attach_mse - best_learned_mse) / max(best_attach_mse, 1e-10) * 100

        # Condition: B3/B4 beats B2 on delta_MSE
        if best_learned_mse < best_attach_mse:
            # Condition: retrieval Top10 improvement
            learned_top10 = best_learned["retrieval_top10"]
            attach_top10 = attach_result["retrieval_top10"]
            top10_improvement = learned_top10 - attach_top10

            if top10_improvement >= 0.02:  # >= 2 pp
                verdict = "GATE_B_PASS"
                detail = f"{best_learned['predictor']} beats attach_mean (MSE: {best_learned_mse:.6f} vs {best_attach_mse:.6f}, Top10: {learned_top10:.4f} vs {attach_top10:.4f})"
            elif top10_improvement > 0:
                verdict = "GATE_B_PASS_MARGINAL"
                detail = f"{best_learned['predictor']} beats attach_mean marginally (Top10: {learned_top10:.4f} vs {attach_top10:.4f})"
            else:
                verdict = "GATE_B_FAIL_DELTA_NOT_PREDICTABLE"
                detail = f"{best_learned['predictor']} lower MSE but retrieval no better than attach_mean"
        else:
            detail = f"No learned predictor improves MSE over attach_mean (best learned MSE={best_learned_mse:.6f}, attach_mean MSE={best_attach_mse:.6f})"

    summary = {
        "verdict": verdict,
        "detail": detail,
        "predictors": results,
        "timestamp": TIMESTAMP,
    }
    save_json(OUT_DIR / "d4a2g_safe_gateB_summary.json", summary)

    log.info("=" * 60)
    log.info("Gate B verdict: %s", verdict)
    log.info("Detail: %s", detail)
    log.info("=" * 60)

    write_complete_marker("d4a2g_safe_gateB_predictability")


if __name__ == "__main__":
    main()

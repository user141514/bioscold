#!/usr/bin/env python3
"""
=============================================================================
 D4A2G-SAFE Transform Vector Gate — Smoke Test + Delta Dataset
=============================================================================

Combines:
  - Part 3: Smoke test (tiny caps) on Gate A/B/C
  - Part 4: Delta dataset construction (old → replacement vector differences)

Usage:
  python core/scripts/routeA_d4a2g_safe_build_embeddings.py
  python core/scripts/routeA_d4a2g_safe_build_embeddings.py --smoke_only

Engineering rules: CPU-only, chunked, NPZ storage, deterministic seed.
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
log = logging.getLogger("d4a2g_build")

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

# Paths
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


def load_jsonl_stream(path: Path) -> List[Dict]:
    """Load JSONL into memory (used for small files / smoke test)."""
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_complete_marker(name: str) -> None:
    (OUT_DIR / f"{name}_complete.md").write_text(f"# {name}\nCompleted: {TIMESTAMP}\n", encoding="utf-8")


def load_config() -> Dict[str, Any]:
    cfg_path = OUT_DIR / "d4a2g_safe_resource_config.json"
    if cfg_path.exists():
        return load_json(cfg_path)
    return {}


def load_embeddings() -> Tuple[np.ndarray, List[str]]:
    """Load NPZ embeddings and return (embeddings_array, smiles_list)."""
    npz_path = OUT_DIR / "d4a2g_safe_fragment_embeddings.npz"
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    smiles = list(data["smiles"])
    log.info("Loaded embeddings: shape=%s, n_fragments=%d", embeddings.shape, len(smiles))
    return embeddings, smiles


def build_fragment_to_embedding_id(smiles_list: List[str], embeddings: np.ndarray) -> Dict[str, int]:
    """Build SMILES → index mapping."""
    mapping = {}
    for i, smi in enumerate(smiles_list):
        mapping[smi] = i
    return mapping


# ===================================================================
# Part 3 — Smoke Test
# ===================================================================

def run_smoke_test(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run mini Gate A/B/C on a tiny sample to verify pipeline works."""
    log.info("=" * 50)
    log.info("PART 3: Smoke Test (tiny caps)")
    log.info("=" * 50)

    smoke_train = 2000
    smoke_val = 500
    smoke_test_queries = 200
    seed = config.get("random_seed", 20260523)
    random.seed(seed)
    np.random.seed(seed)

    # Load embeddings
    embeddings, smiles_list = load_embeddings()
    frag_to_emb_id = build_fragment_to_embedding_id(smiles_list, embeddings)
    pdim = embeddings.shape[1]

    # Load manifest for query samples
    manifest_path = D4A0_DIR / "d4a0_query_split_manifest.jsonl"
    all_train: List[Dict] = []
    all_val: List[Dict] = []
    all_test: List[Dict] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            s = rec.get("split", "")
            if s == "train":
                all_train.append(rec)
            elif s == "val":
                all_val.append(rec)
            elif s == "test":
                all_test.append(rec)

    # Sample tiny subsets
    random.shuffle(all_train)
    random.shuffle(all_val)
    random.shuffle(all_test)
    train_sample = all_train[:smoke_train]
    val_sample = all_val[:smoke_val]
    test_sample = all_test[:smoke_test_queries]
    log.info("Smoke sample: train=%d, val=%d, test=%d", len(train_sample), len(val_sample), len(test_sample))

    # Build smoke delta dataset
    train_deltas, train_meta = build_delta_records(train_sample, frag_to_emb_id, embeddings, pdim, "train")
    val_deltas, val_meta = build_delta_records(val_sample, frag_to_emb_id, embeddings, pdim, "val")

    if len(train_deltas) == 0 or len(val_deltas) == 0:
        log.error("SMOKE FAIL: zero delta records constructed.")
        return {"status": "SMOKE_FAIL", "reason": "zero_delta_records"}

    # Mini Gate A: delta norm distribution
    norms = np.linalg.norm(train_deltas, axis=1)
    gate_a = {
        "delta_norm_mean": float(np.mean(norms)),
        "delta_norm_std": float(np.std(norms)),
        "delta_norm_p50": float(np.percentile(norms, 50)),
        "delta_norm_p10": float(np.percentile(norms, 10)),
        "delta_norm_p90": float(np.percentile(norms, 90)),
        "n_deltas": len(norms),
    }
    log.info("Smoke Gate A: mean_delta_norm=%.4f", gate_a["delta_norm_mean"])

    # Mini Gate B: zero predictor vs mean delta
    val_norms = np.linalg.norm(val_deltas, axis=1)
    zero_mse = float(np.mean(val_norms ** 2))
    mean_delta = np.mean(train_deltas, axis=0)
    mean_pred = np.tile(mean_delta.reshape(1, -1), (len(val_deltas), 1))
    mean_mse = float(np.mean(np.sum((val_deltas - mean_pred) ** 2, axis=1)))
    gate_b = {
        "zero_predictor_mse": zero_mse,
        "mean_delta_mse": mean_mse,
        "mean_delta_norm": float(np.linalg.norm(mean_delta)),
        "improvement_vs_zero": float((zero_mse - mean_mse) / zero_mse * 100) if zero_mse > 0 else 0,
    }
    log.info("Smoke Gate B: zero_mse=%.6f mean_mse=%.6f", zero_mse, mean_mse)

    # Mini Gate C: zero-delta retrieval (tiny)
    vocab = build_vocab_set(manifest_path)
    vocab_smiles = list(vocab)
    vocab_emb = np.array([
        embeddings[frag_to_emb_id[smi]] for smi in vocab_smiles
        if smi in frag_to_emb_id
    ])
    vocab_lookup = [smi for smi in vocab_smiles if smi in frag_to_emb_id]

    if len(vocab_emb) > 0 and len(test_sample) > 0:
        zero_hits = 0
        mean_hits = 0
        total = 0
        topk = 5
        for rec in test_sample[:50]:
            old_smi = rec.get("old_fragment_smiles", "")
            if old_smi not in frag_to_emb_id:
                continue
            z_old = embeddings[frag_to_emb_id[old_smi]].reshape(1, -1)
            z_pred_mean = z_old + mean_delta.reshape(1, -1)

            pos_replacements = rec.get("positive_replacement_set", [])
            if not pos_replacements:
                continue

            sim_zero = cosine_similarity_chunked(z_old, vocab_emb)
            sim_mean = cosine_similarity_chunked(z_pred_mean, vocab_emb)

            zero_top = set(vocab_lookup[j] for j in np.argsort(-sim_zero[0])[:topk])
            mean_top = set(vocab_lookup[j] for j in np.argsort(-sim_mean[0])[:topk])

            if any(p in zero_top for p in pos_replacements):
                zero_hits += 1
            if any(p in mean_top for p in pos_replacements):
                mean_hits += 1
            total += 1

        gate_c = {
            "zero_delta_top%d" % topk: zero_hits / total if total > 0 else 0,
            "mean_delta_top%d" % topk: mean_hits / total if total > 0 else 0,
            "n_queries": total,
        }
    else:
        gate_c = {"zero_delta_top5": 0, "mean_delta_top5": 0, "n_queries": 0}

    log.info("Smoke Gate C: zero_top%d=%.4f mean_top%d=%.4f (n=%d)",
             topk, gate_c.get("zero_delta_top5", 0),
             topk, gate_c.get("mean_delta_top5", 0),
             gate_c.get("n_queries", 0))

    smoke_result = {
        "status": "SMOKE_PASS",
        "gate_a": gate_a,
        "gate_b": gate_b,
        "gate_c": gate_c,
        "smoke_train_used": len(train_deltas),
        "smoke_val_used": len(val_deltas),
        "smoke_test_queries_used": min(50, len(test_sample)),
    }

    # Determine smoke pass/fail
    if gate_a["n_deltas"] < 10:
        smoke_result["status"] = "SMOKE_FAIL"
        smoke_result["reason"] = "too_few_deltas"
    if math.isnan(gate_b["zero_predictor_mse"]) or math.isnan(gate_b["mean_delta_mse"]):
        smoke_result["status"] = "SMOKE_FAIL"
        smoke_result["reason"] = "nan_metrics"

    log.info("Smoke test status: %s", smoke_result["status"])
    return smoke_result


def cosine_similarity_chunked(query: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query (1×d) and targets (n×d)."""
    q_norm = np.linalg.norm(query, axis=1, keepdims=True) + 1e-10
    t_norm = np.linalg.norm(targets, axis=1, keepdims=True).T + 1e-10
    return (query @ targets.T) / (q_norm * t_norm)


# ===================================================================
# Part 4 — Delta Dataset
# ===================================================================

def build_vocab_set(manifest_path: Path) -> set:
    """Build set of all replacement SMILES from train vocabulary."""
    vocab_path = D4A0_DIR / "d4a0_train_replacement_vocabulary.csv"
    vocab: set = set()
    if vocab_path.exists():
        lines = vocab_path.read_text(encoding="utf-8").strip().split("\n")
        for line in lines[1:]:
            parts = line.split(",")
            if parts:
                vocab.add(parts[0].strip())
    return vocab


def compute_transform_key(old_smi: str, new_smi: str) -> str:
    """Create a deterministic transform key from old→new fragment."""
    return f"{old_smi}→{new_smi}"


def build_delta_records(queries: List[Dict], frag_to_emb_id: Dict[str, int],
                        embeddings: np.ndarray, pdim: int,
                        split: str) -> Tuple[np.ndarray, List[Dict]]:
    """Build delta vectors and metadata from query records."""
    deltas: List[np.ndarray] = []
    meta: List[Dict] = []
    pair_id = 0

    for rec in queries:
        old_smi = rec.get("old_fragment_smiles", "")
        attach_sig = rec.get("attachment_signature", "")
        query_id = rec.get("query_id", "")

        if old_smi not in frag_to_emb_id:
            continue
        old_emb_id = frag_to_emb_id[old_smi]
        z_old = embeddings[old_emb_id]

        pos_set = rec.get("positive_replacement_set", [])
        for repl_smi in pos_set:
            if repl_smi not in frag_to_emb_id:
                continue
            repl_emb_id = frag_to_emb_id[repl_smi]
            z_repl = embeddings[repl_emb_id]
            delta = z_repl - z_old
            delta_norm = float(np.linalg.norm(delta))
            transform_key = compute_transform_key(old_smi, repl_smi)

            deltas.append(delta)
            meta.append({
                "pair_id": f"{split}_{pair_id:08d}",
                "query_id": query_id,
                "split": split,
                "old_fragment_smiles": old_smi,
                "replacement_fragment_smiles": repl_smi,
                "attachment_signature": attach_sig,
                "transform_key": transform_key,
                "old_embedding_id": old_emb_id,
                "replacement_embedding_id": repl_emb_id,
                "delta_embedding_id": len(deltas) - 1,
                "delta_norm": delta_norm,
                "sampling_weight": 1.0,
            })
            pair_id += 1

    if len(deltas) == 0:
        return np.zeros((0, pdim), dtype=np.float32), []

    return np.array(deltas, dtype=np.float32), meta


def build_delta_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build full delta dataset for train and val splits."""
    log.info("=" * 50)
    log.info("PART 4: Delta Dataset Construction")
    log.info("=" * 50)

    max_train = config.get("max_train_delta_records", 50000)
    max_val = config.get("max_val_delta_records", 10000)
    seed = config.get("random_seed", 20260523)
    random.seed(seed)
    np.random.seed(seed)

    # Load embeddings
    embeddings, smiles_list = load_embeddings()
    frag_to_emb_id = build_fragment_to_embedding_id(smiles_list, embeddings)
    pdim = embeddings.shape[1]
    log.info("Embedding dim: %d, fragment mapping size: %d", pdim, len(frag_to_emb_id))

    # Load manifest
    manifest_path = D4A0_DIR / "d4a0_query_split_manifest.jsonl"
    all_train: List[Dict] = []
    all_val: List[Dict] = []
    log.info("Streaming manifest to collect train/val queries...")
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            s = rec.get("split", "")
            if s == "train":
                all_train.append(rec)
            elif s == "val":
                all_val.append(rec)

    log.info("Total train queries: %d, val queries: %d", len(all_train), len(all_val))

    # Stratify by attachment_signature
    def stratify_sample(records: List[Dict], max_records: int) -> List[Dict]:
        if len(records) <= max_records:
            return records
        # Group by attachment signature
        by_sig: Dict[str, List[Dict]] = defaultdict(list)
        for rec in records:
            sig = rec.get("attachment_signature", "unknown")
            by_sig[sig].append(rec)

        # Sample proportionally
        sampled = []
        remaining = max_records
        sigs = sorted(by_sig.keys(), key=lambda s: len(by_sig[s]), reverse=True)
        for sig in sigs:
            group = by_sig[sig]
            if remaining <= 0:
                break
            n_from_group = max(1, int(len(group) / len(records) * max_records))
            n_from_group = min(n_from_group, len(group), remaining)
            random.shuffle(group)
            sampled.extend(group[:n_from_group])
            remaining -= n_from_group

        # If we still have room, add more random
        if remaining > 0:
            already_ids = {id(r) for r in sampled}
            extra = [r for r in records if id(r) not in already_ids]
            random.shuffle(extra)
            sampled.extend(extra[:remaining])

        random.shuffle(sampled)
        log.info("Stratified sample: %d from %d records (target=%d)", len(sampled), len(records), max_records)
        return sampled

    train_sample = stratify_sample(all_train, max_train)
    val_sample = stratify_sample(all_val, max_val)

    # Build delta vectors
    train_deltas, train_meta = build_delta_records(train_sample, frag_to_emb_id, embeddings, pdim, "train")
    val_deltas, val_meta = build_delta_records(val_sample, frag_to_emb_id, embeddings, pdim, "val")

    log.info("Train delta records: %d, Val delta records: %d", len(train_deltas), len(val_deltas))

    # Save NPZ
    npz_train_path = OUT_DIR / "d4a2g_safe_delta_vectors_train.npz"
    npz_val_path = OUT_DIR / "d4a2g_safe_delta_vectors_val.npz"
    np.savez_compressed(npz_train_path, delta_vectors=train_deltas)
    np.savez_compressed(npz_val_path, delta_vectors=val_deltas)
    log.info("Saved train delta vectors: %s", npz_train_path)
    log.info("Saved val delta vectors: %s", npz_val_path)

    # Save metadata as JSONL (no embeddings inside)
    def save_meta_jsonl(meta_list: List[Dict], path: Path) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for rec in meta_list:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    save_meta_jsonl(train_meta, OUT_DIR / "d4a2g_safe_delta_dataset_train_meta.jsonl")
    save_meta_jsonl(val_meta, OUT_DIR / "d4a2g_safe_delta_dataset_val_meta.jsonl")

    # Summary
    summary = {
        "n_train_delta_records": len(train_deltas),
        "n_val_delta_records": len(val_deltas),
        "embedding_dim": pdim,
        "n_unique_old_in_train": len(set(m["old_fragment_smiles"] for m in train_meta)),
        "n_unique_repl_in_train": len(set(m["replacement_fragment_smiles"] for m in train_meta)),
        "n_attachment_signatures_train": len(set(m["attachment_signature"] for m in train_meta)),
        "mean_delta_norm_train": float(np.mean(np.linalg.norm(train_deltas, axis=1))) if len(train_deltas) > 0 else 0,
        "mean_delta_norm_val": float(np.mean(np.linalg.norm(val_deltas, axis=1))) if len(val_deltas) > 0 else 0,
        "train_npz_path": str(npz_train_path),
        "val_npz_path": str(npz_val_path),
        "timestamp": TIMESTAMP,
    }
    save_json(OUT_DIR / "d4a2g_safe_delta_dataset_summary.json", summary)
    log.info("Delta dataset summary: %s", json.dumps(summary, default=str))

    return summary


# ===================================================================
# Main
# ===================================================================

def main():
    log.info("=" * 60)
    log.info("D4A2G-SAFE Smoke Test + Delta Dataset Builder")
    log.info("=" * 60)

    smoke_only = "--smoke_only" in sys.argv

    config = load_config()
    if not config:
        log.warning("No resource config found, using defaults.")
        config = {
            "max_train_delta_records": 50000,
            "max_val_delta_records": 10000,
            "random_seed": 20260523,
            "projection_dim": 128,
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seed = config.get("random_seed", 20260523)
    random.seed(seed)
    np.random.seed(seed)

    # Part 3: Smoke test
    smoke_result = run_smoke_test(config)

    # Save smoke report
    smoke_metrics = {
        "status": smoke_result["status"],
        "gate_a": smoke_result.get("gate_a", {}),
        "gate_b": smoke_result.get("gate_b", {}),
        "gate_c": smoke_result.get("gate_c", {}),
        "timestamp": TIMESTAMP,
    }
    save_json(OUT_DIR / "d4a2g_safe_smoke_metrics.json", smoke_metrics)

    # Smoke markdown
    md_lines = [
        f"# D4A2G-SAFE Smoke Test Report",
        f"Date: {TIMESTAMP}",
        f"Status: **{smoke_result['status']}**",
        "",
        "## Gate A (Delta Norm Distribution)",
    ]
    ga = smoke_result.get("gate_a", {})
    for k, v in ga.items():
        md_lines.append(f"- {k}: {v}")
    md_lines.extend([
        "",
        "## Gate B (Zero vs Mean Predictor)",
    ])
    gb = smoke_result.get("gate_b", {})
    for k, v in gb.items():
        md_lines.append(f"- {k}: {v}")
    md_lines.extend([
        "",
        "## Gate C (Zero-Delta Retrieval)",
    ])
    gc = smoke_result.get("gate_c", {})
    for k, v in gc.items():
        md_lines.append(f"- {k}: {v}")
    md_lines.append("")
    (OUT_DIR / "d4a2g_safe_smoke_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    if smoke_result["status"] == "SMOKE_FAIL":
        log.error("SMOKE_FAIL: Smoke test did not pass.")
        write_complete_marker("d4a2g_safe_build_embeddings")
        sys.exit(1)

    if smoke_only:
        log.info("Smoke-only mode. Skipping full delta dataset construction.")
        write_complete_marker("d4a2g_safe_build_embeddings")
        return

    # Part 4: Full delta dataset
    delta_summary = build_delta_dataset(config)

    log.info("=" * 60)
    log.info("Build complete: %s", json.dumps(delta_summary, default=str))
    log.info("=" * 60)

    write_complete_marker("d4a2g_safe_build_embeddings")


if __name__ == "__main__":
    main()

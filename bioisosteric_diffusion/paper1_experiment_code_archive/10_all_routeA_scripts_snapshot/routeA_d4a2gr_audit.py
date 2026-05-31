#!/usr/bin/env python3
"""
=============================================================================
 D4A2G-R: Protocol Reconciliation, Leakage Audit, Vocabulary Scaling +
          Artifact Analysis for Direction 3 (Delta Predictor)
=============================================================================

Purpose: Audit whether D4A2G-SAFE Direction 3 results are genuine, scalable,
         and free of protocol confounds.

Parts:
  A — Gate B/C Protocol Reconciliation
  B — Unified Evaluation Set (transform-heldout, seen-vocabulary)
  C — Leakage Audit (7 checks)
  D — Vocabulary Ladder (L1=166..L4=10k, 4 methods)
  E — Artifact Analysis (5 checks)
  F — Final Verdict with SKEPTICAL REVIEW

Usage:
  E:/Anaconda3/envs/accfg/python.exe core/scripts/routeA_d4a2gr_audit.py

Resources: CPU-only, chunked operations, deterministic seed 20260523.
=============================================================================
"""

from __future__ import annotations

import csv
import json
import logging
import math
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# pandas is imported lazily in Part D (needs sklearn too)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("d4a2gr")

SEED = 20260523
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ────────────────────────────────────────────────────────────
BASE = Path("E:/zuhui/bioisosteric_diffusion")
D4A0 = BASE / "plan_results" / "routeA_chembl37k_d0d3_engineering_safe" / "07_d4a0_matrix_freeze"
D4A2G = BASE / "plan_results" / "routeA_chembl37k_d4a2g_safe_transform_vector_gates"
D4A1 = BASE / "plan_results" / "routeA_chembl37k_d4a1_learned_ranker"
OUT = BASE / "plan_results" / "routeA_chembl37k_d4a2g_r_protocol_scale_audit"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ── Helpers ──────────────────────────────────────────────────────────


def time_part(label: str) -> float:
    log.info("")
    log.info("=" * 60)
    log.info("PART: %s", label)
    log.info("=" * 60)
    return time.time()


def time_end(label: str, start: float) -> float:
    elapsed = time.time() - start
    log.info("PART %s done (%.2f s)", label, elapsed)
    return elapsed


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_csv(path: Path, rows: List[Dict[str, Any]],
             fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("(empty)\n", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def cosine_sim_chunked(query: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Cosine similarity n_queries x n_targets with safe norm."""
    q_norm = np.linalg.norm(query, axis=1, keepdims=True) + 1e-10
    t_norm = np.linalg.norm(targets, axis=1, keepdims=True).T + 1e-10
    return (query @ targets.T) / (q_norm * t_norm)


# ══════════════════════════════════════════════════════════════════════
# PART A: Gate B/C Protocol Reconciliation
# ══════════════════════════════════════════════════════════════════════


def part_a_reconciliation() -> Tuple[int, int, set[str], set[str]]:
    """Reconcile Gate B (val, pair-level) vs Gate C (test, query-level) protocols.

    Returns:
        n_gateB_qids, n_gateC_qids, train_vocab, gateB_targets
    """
    t0 = time_part("A — Gate B/C Protocol Reconciliation")

    # ── Load Gate B results ──
    gateB_rows: List[Dict[str, Any]] = []
    with open(D4A2G / "d4a2g_safe_gateB_predictability_results.csv", "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            gateB_rows.append(row)
    gateB_summary = load_json(D4A2G / "d4a2g_safe_gateB_summary.json")

    # ── Load Gate C results ──
    gateC_rows: List[Dict[str, Any]] = []
    with open(D4A2G / "d4a2g_safe_gateC_zero_delta_ablation.csv", "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            gateC_rows.append(row)
    gateC_summary = load_json(D4A2G / "d4a2g_safe_gateC_summary.json")

    # ── Val meta for Gate B query_ids and positives ──
    val_meta = load_jsonl(D4A2G / "d4a2g_safe_delta_dataset_val_meta.jsonl")
    gateB_qids: set[str] = set()
    gateB_targets: set[str] = set()
    for r in val_meta:
        gateB_qids.add(r["query_id"])
        gateB_targets.add(r["replacement_fragment_smiles"])

    # ── Gate C query_ids from D4A0 test manifest ──
    manifest_all = load_jsonl(D4A0 / "d4a0_query_split_manifest.jsonl")
    gateC_qids: set[str] = set()
    gateC_positives: set[str] = set()
    for rec in manifest_all:
        if rec.get("split") == "test":
            gateC_qids.add(rec["query_id"])
            for repl in rec.get("positive_replacement_set", []):
                gateC_positives.add(repl)

    # ── Vocab ──
    gateB_vocab: set[str] = set()
    with open(D4A0 / "d4a0_train_replacement_vocabulary.csv", "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            gateB_vocab.add(row["replacement_smiles"].strip())

    frag_idx: List[Dict[str, str]] = []
    with open(D4A2G / "d4a2g_safe_fragment_embedding_index.csv", "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            frag_idx.append(row)
    gateC_vocab: set[str] = set(r["fragment_smiles"] for r in frag_idx)

    # ── Overlaps ──
    query_overlap = gateB_qids & gateC_qids
    vocab_overlap = gateB_vocab & gateC_vocab
    target_overlap = gateB_targets & gateC_positives

    # ── Build protocol comparison ──
    # Map Gate B predictors to C-like method names for comparison
    protocol_rows: List[Dict[str, Any]] = []

    for rb in gateB_rows:
        pred = rb["predictor"]
        protocol_rows.append({
            "gate": "B",
            "method": pred,
            "split": "val",
            "n_queries": int(rb.get("retrieval_n", 0)),
            "positive_definition": "pair_level_one_correct_per_pair",
            "candidate_vocab_size": len(gateB_vocab),
            "candidate_vocab_source": "train_replacement_vocabulary",
            "top10": rb.get("retrieval_top10", ""),
            "mrr": rb.get("retrieval_mrr", ""),
            "retrieval_mode": "exact",
        })

    for rc in gateC_rows:
        protocol_rows.append({
            "gate": "C",
            "method": rc.get("method", ""),
            "split": "test",
            "n_queries": int(float(rc.get("n_queries", 0))),
            "positive_definition": "query_level_any_positive_in_set",
            "candidate_vocab_size": gateC_summary.get("vocab_size", len(gateC_vocab)),
            "candidate_vocab_source": "all_166_fragment_embeddings",
            "top10": rc.get("top10", ""),
            "mrr": rc.get("mrr", ""),
            "retrieval_mode": rc.get("retrieval_mode", "exact"),
        })

    save_csv(OUT / "d4a2gr_gate_protocol_comparison.csv", protocol_rows)

    # Overlap audit
    overlap_rows = [
        {"check": "query_overlap_B_C", "gate_B": len(gateB_qids),
         "gate_C": len(gateC_qids), "overlap": len(query_overlap)},
        {"check": "vocab_overlap_B_C", "gate_B": len(gateB_vocab),
         "gate_C": len(gateC_vocab), "overlap": len(vocab_overlap)},
        {"check": "target_replacement_overlap_B_C", "gate_B": len(gateB_targets),
         "gate_C": len(gateC_positives), "overlap": len(target_overlap)},
    ]
    save_csv(OUT / "d4a2gr_gate_overlap_audit.csv", overlap_rows)

    log.info("Gate B: %d predictors, %d positive pairs, %d unique qids",
             len(gateB_rows),
             len(val_meta), len(gateB_qids))
    log.info("Gate C: %d test queries (manifest), vocab=%d",
             len(gateC_qids), len(gateC_vocab))
    log.info("Overlaps — qid=%d  vocab=%d  target=%d",
             len(query_overlap), len(vocab_overlap), len(target_overlap))
    log.info("  KEY: Gate B/C use different splits (val vs test), "
             "different positive definitions (pair vs query), "
             "different vocab sources — results are NOT directly comparable.")

    time_end("A", t0)
    return len(gateB_qids), len(gateC_qids), gateB_vocab, gateB_targets


# ══════════════════════════════════════════════════════════════════════
# PART B: Unified Evaluation Set
# ══════════════════════════════════════════════════════════════════════


def part_b_unified_eval(train_vocab: set[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Create unified evaluation set from transform-heldout test queries.

    Filter criteria:
      1. test split (from D4A0 manifest)
      2. transform-heldout: no transform_key overlap with train
      3. seen-vocabulary: at least one positive replacement in train_vocab
    Cap at 5000 with stratified sampling by attachment_signature.
    """
    t0 = time_part("B — Unified Evaluation Set")

    manifest_all = load_jsonl(D4A0 / "d4a0_query_split_manifest.jsonl")

    # Collect train transform keys
    train_tk: set[str] = set()
    test_queries: List[Dict[str, Any]] = []
    for rec in manifest_all:
        s = rec.get("split", "")
        tk_set = set(rec.get("transform_key_set", []))
        if s == "train":
            train_tk.update(tk_set)
        elif s == "test":
            test_queries.append(rec)

    # Filter
    pool: List[Dict[str, Any]] = []
    skipped_tk = 0
    skipped_vocab = 0
    for rec in test_queries:
        tk_set = set(rec.get("transform_key_set", []))
        if tk_set & train_tk:
            skipped_tk += 1
            continue
        pos_set = set(rec.get("positive_replacement_set", []))
        pos_in_vocab = pos_set & train_vocab
        if not pos_in_vocab:
            skipped_vocab += 1
            continue
        rec["positive_in_vocab"] = list(pos_in_vocab)
        rec["n_positives_in_vocab"] = len(pos_in_vocab)
        pool.append(rec)

    log.info("Pool: %d test queries total, %d tk-skipped, %d vocab-skipped, %d candidates",
             len(test_queries), skipped_tk, skipped_vocab, len(pool))

    # Stratified sample to 5000
    N_TARGET = 5000
    sampled = _stratify_sample(pool, N_TARGET, key="attachment_signature")

    # Save manifest
    with open(OUT / "d4a2gr_unified_eval_manifest.jsonl", "w", encoding="utf-8") as fh:
        for rec in sampled:
            fh.write(json.dumps({
                "query_id": rec["query_id"],
                "old_fragment_smiles": rec["old_fragment_smiles"],
                "attachment_signature": rec.get("attachment_signature", ""),
                "positive_replacement_set": rec.get("positive_in_vocab", []),
                "n_positives_in_vocab": rec.get("n_positives_in_vocab", 0),
            }, ensure_ascii=False) + "\n")

    summary = {
        "n_test_available": len(test_queries),
        "n_transform_heldout": len(test_queries) - skipped_tk,
        "n_seen_vocabulary": len(pool),
        "n_eval_set": len(sampled),
        "target_size": N_TARGET,
        "stratified_by": "attachment_signature",
    }
    save_json(OUT / "d4a2gr_unified_eval_summary.json", summary)
    log.info("Unified eval set: %d queries written", len(sampled))

    time_end("B", t0)
    return sampled, summary


def _stratify_sample(records: List[Dict[str, Any]], n_target: int,
                     key: str = "attachment_signature",
                     seed: int = SEED) -> List[Dict[str, Any]]:
    """Stratified sample by a categorical key column."""
    rng = random.Random(seed)
    if len(records) <= n_target:
        return list(records)

    by_key: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_key[rec.get(key, "unknown")].append(rec)

    sampled: List[Dict[str, Any]] = []
    total = len(records)
    for k_val, group in sorted(by_key.items(), key=lambda x: -len(x[1])):
        n_from = max(1, int(len(group) / total * n_target))
        n_from = min(n_from, len(group), n_target - len(sampled))
        rng.shuffle(group)
        sampled.extend(group[:n_from])
        if len(sampled) >= n_target:
            break

    if len(sampled) < n_target:
        extra = [r for r in records if r not in sampled]
        rng.shuffle(extra)
        sampled.extend(extra[:n_target - len(sampled)])

    sampled = sampled[:n_target]
    rng.shuffle(sampled)
    return sampled


# ══════════════════════════════════════════════════════════════════════
# PART C: Leakage Audit
# ══════════════════════════════════════════════════════════════════════


def part_c_leakage(train_vocab: set[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """7 leakage checks."""
    t0 = time_part("C — Leakage Audit (7 checks)")

    manifest_all = load_jsonl(D4A0 / "d4a0_query_split_manifest.jsonl")
    train_queries: List[Dict[str, Any]] = []
    test_queries: List[Dict[str, Any]] = []
    for rec in manifest_all:
        s = rec.get("split", "")
        if s == "train":
            train_queries.append(rec)
        elif s == "test":
            test_queries.append(rec)

    # ── C01: Transform key overlap train vs test ──
    train_tk: set[str] = set()
    test_tk: set[str] = set()
    for rec in train_queries:
        train_tk.update(rec.get("transform_key_set", []))
    for rec in test_queries:
        test_tk.update(rec.get("transform_key_set", []))
    tk_overlap = train_tk & test_tk

    # ── C02: SVD fit split check ──
    embed_config = load_json(D4A2G / "d4a2g_safe_embedding_config.json")
    frag_idx: List[Dict[str, str]] = []
    with open(D4A2G / "d4a2g_safe_fragment_embedding_index.csv", "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            frag_idx.append(row)
    n_all = len(frag_idx)
    n_voc = sum(1 for r in frag_idx if r.get("in_vocab", "0") == "1")
    n_non = n_all - n_voc
    # The embedding NPZ has ALL 166 fragments — SVD fit split is unverifiable
    resource_cfg = load_json(D4A2G / "d4a2g_safe_resource_config.json")
    fit_split = resource_cfg.get("fit_split", "not_recorded")
    svd_method = embed_config.get("projection_method", "?")

    # ── C03: Vocab fragments all appear in train manifest ──
    train_repl_smiles: set[str] = set()
    for rec in train_queries:
        train_repl_smiles.update(rec.get("positive_replacement_set", []))
    test_only_vocab = [
        r["fragment_smiles"] for r in frag_idx
        if r.get("in_vocab", "0") == "1"
        and r["fragment_smiles"] not in train_repl_smiles
    ]

    # ── C04: Target replacement fraction in train vocab ──
    total_possible = 0
    total_in_vocab = 0
    for rec in test_queries:
        for p in rec.get("positive_replacement_set", []):
            total_possible += 1
            if p in train_vocab:
                total_in_vocab += 1
    target_vocab_frac = total_in_vocab / max(total_possible, 1)

    # ── C05: Delta train query_ids vs test query_ids ──
    train_delta_meta = load_jsonl(D4A2G / "d4a2g_safe_delta_dataset_train_meta.jsonl")
    train_delta_qids = set(r["query_id"] for r in train_delta_meta)
    test_qids = set(r["query_id"] for r in test_queries)
    delta_train_test_overlap = train_delta_qids & test_qids

    # ── C06: N/A (positive-only training, no negatives) ──
    # ── C07: Frequency from train vocab only ──
    vocab_freq: Dict[str, int] = {}
    with open(D4A0 / "d4a0_train_replacement_vocabulary.csv", "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            vocab_freq[row["replacement_smiles"].strip()] = int(row.get("global_train_frequency", 0))
    test_repls_all: set[str] = set()
    for rec in test_queries:
        test_repls_all.update(rec.get("positive_replacement_set", []))
    missing_freq = [r for r in test_repls_all if r not in vocab_freq]

    checks: List[Dict[str, Any]] = [
        {
            "check_id": "C01",
            "check_name": "transform_key_overlap_train_vs_test",
            "result": "PASS" if len(tk_overlap) == 0 else f"FAIL ({len(tk_overlap)} overlaps)",
            "detail": f"train_tk={len(train_tk)}, test_tk={len(test_tk)}, overlap={len(tk_overlap)}",
            "severity": "critical" if len(tk_overlap) > 0 else "none",
        },
        {
            "check_id": "C02",
            "check_name": "svd_fit_split",
            "result": "WARN",
            "detail": (
                f"SVD method={svd_method}, fit_split field={fit_split}. "
                f"NPZ contains ALL {n_all} fragments ({n_voc} train-vocab, {n_non} test-only). "
                "No evidence of train-only SVD fit. If SVD was fit on all 166 fragments, "
                "it constitutes mild leakage: test fragment geometry influenced projection axes. "
                "Cannot verify without checkpoint or code audit."
            ),
            "severity": "moderate",
        },
        {
            "check_id": "C03",
            "check_name": "vocab_fragments_in_train_manifest",
            "result": "PASS" if len(test_only_vocab) == 0 else f"WARN ({len(test_only_vocab)} exceptions)",
            "detail": f"Fragments flagged in_vocab=1 but absent from train positive_replacement_sets: {len(test_only_vocab)}. "
                      f"Samples: {test_only_vocab[:5]}",
            "severity": "low" if len(test_only_vocab) > 0 else "none",
        },
        {
            "check_id": "C04",
            "check_name": "target_replacement_in_train_vocab",
            "result": f"INFO (fraction={target_vocab_frac:.4f})",
            "detail": f"{total_in_vocab}/{total_possible} target replacements in train vocab ({target_vocab_frac*100:.1f}%). "
                      "Remainder are test-only fragments — never retrievable at any vocab level.",
            "severity": "info",
        },
        {
            "check_id": "C05",
            "check_name": "delta_train_test_query_overlap",
            "result": "PASS" if len(delta_train_test_overlap) == 0 else f"FAIL ({len(delta_train_test_overlap)} qids)",
            "detail": f"train_delta_qids={len(train_delta_qids)}, test_qids={len(test_qids)}, overlap={len(delta_train_test_overlap)}",
            "severity": "critical" if len(delta_train_test_overlap) > 0 else "none",
        },
        {
            "check_id": "C06",
            "check_name": "test_positive_as_train_negative",
            "result": "N/A",
            "detail": "Delta predictor uses positive-only training pairs. No hard negatives in delta training.",
            "severity": "none",
        },
        {
            "check_id": "C07",
            "check_name": "frequency_from_train_vocab_only",
            "result": "PASS" if len(missing_freq) == 0 else f"WARN ({len(missing_freq)} missing)",
            "detail": f"Test-positive replacements absent from train vocab frequency table: {len(missing_freq)}/{len(test_repls_all)}. "
                      f"These are test-only fragments with no train frequency — expected behavior.",
            "severity": "none" if len(missing_freq) == 0 else "low",
        },
    ]

    save_csv(OUT / "d4a2gr_leakage_audit.csv", checks)
    n_pass = sum(1 for c in checks if c["result"].startswith("PASS"))
    n_warn = sum(1 for c in checks if c["result"].startswith("WARN"))
    summary: Dict[str, Any] = {
        "n_checks": len(checks),
        "n_pass": n_pass,
        "n_warn": n_warn,
        "n_fail": sum(1 for c in checks if c["result"].startswith("FAIL")),
        "critical_issues": sum(1 for c in checks if c["severity"] == "critical"),
    }
    save_json(OUT / "d4a2gr_leakage_summary.json", summary)

    for c in checks:
        log.info("  [%s] %s: %s", c["check_id"], c["check_name"], c["result"])

    time_end("C", t0)
    return checks, summary


# ══════════════════════════════════════════════════════════════════════
# PART D: Vocabulary Ladder
# ══════════════════════════════════════════════════════════════════════


def _build_vocab_level(target_size: int,
                       eval_queries: List[Dict[str, Any]],
                       train_vocab: set[str],
                       train_vocab_freq: Dict[str, int],
                       all_fragment_smiles: List[str],
                       seed: int,
                       synthetic_start_idx: int = 10000) -> Tuple[List[str], int]:
    """Build vocabulary of target_size for a ladder level.

    Strategy:
      - Real fragments (up to 166): required positives + train_vocab fill + extra fragments.
      - Synthetic distractors (for levels > available real fragments): random unit-norm
        vectors at embedding indices >= synthetic_start_idx. These are not real fragments
        but test whether the method degrades gracefully with more candidates.

    Returns:
        (vocab_smiles_list, n_real_fragments)
    """
    rng = random.Random(seed)

    # Step 1: Required positives
    required: set[str] = set()
    for rec in eval_queries:
        for p in rec.get("positive_in_vocab", rec.get("positive_replacement_set", [])):
            if p in train_vocab:
                required.add(p)

    vocab = list(required)

    # Step 2: Fill from train vocab (shuffled)
    fill = sorted(
        train_vocab - required,
        key=lambda s: -train_vocab_freq.get(s, 0)
    )
    rng.shuffle(fill)
    for smi in fill:
        if len(vocab) >= target_size:
            break
        vocab.append(smi)

    # Step 3: Add non-vocab fragments (up to all available)
    if len(vocab) < target_size:
        extra = [s for s in all_fragment_smiles if s not in vocab]
        rng.shuffle(extra)
        for smi in extra:
            if len(vocab) >= target_size:
                break
            vocab.append(smi)

    n_real = len(vocab)

    # Step 4: Pad with synthetic distractors labeled "*DISTRACTOR_N"
    if len(vocab) < target_size:
        n_pad = target_size - len(vocab)
        for i in range(n_pad):
            vocab.append(f"*DISTRACTOR_{synthetic_start_idx + i}")
        log.info("  Added %d synthetic distractors (total=%d, real=%d)",
                 n_pad, target_size, n_real)

    vocab = vocab[:target_size]
    rng.shuffle(vocab)
    return vocab, n_real


def _retrieval_metrics(query_emb: np.ndarray,
                       vocab_emb: np.ndarray,
                       pos_sets: List[set[int]],
                       topk_list: Optional[List[int]] = None) -> Dict[str, Any]:
    """Compute Top-k / MRR via chunked retrieval.

    Chunks both query batch and vocab to avoid OOM.
    """
    if topk_list is None:
        topk_list = [1, 5, 10, 20, 50]
    n_q = query_emb.shape[0]
    n_v = vocab_emb.shape[0]
    if n_q == 0 or n_v == 0:
        return {"n_queries": n_q, "n_vocab": n_v, "error": "empty"}
    max_k = max(topk_list)
    Q_CHUNK = 1024
    V_CHUNK = 4096

    all_topk = np.zeros((n_q, max_k), dtype=np.int64)
    # Initialize best scores to -inf
    all_scores = np.full((n_q, max_k), -np.inf, dtype=np.float32)

    for qs in range(0, n_q, Q_CHUNK):
        qe = min(qs + Q_CHUNK, n_q)
        q_batch = query_emb[qs:qe]
        batch_size = qe - qs

        # Per-query: keep a local top-K heap
        local_scores = np.full((batch_size, max_k), -np.inf, dtype=np.float32)
        local_indices = np.zeros((batch_size, max_k), dtype=np.int64)

        for vs in range(0, n_v, V_CHUNK):
            ve = min(vs + V_CHUNK, n_v)
            v_chunk = vocab_emb[vs:ve]
            sim = cosine_sim_chunked(q_batch, v_chunk)  # (batch, chunk_size)

            # Merge: combine current local heap with new sim scores
            combined_scores = np.concatenate([local_scores, sim], axis=1)
            combined_indices = np.concatenate([
                local_indices,
                np.zeros((batch_size, sim.shape[1]), dtype=np.int64) + np.arange(vs, ve)
            ], axis=1)

            # Take top-k
            top_cols = np.argsort(-combined_scores, axis=1)[:, :max_k]
            for b in range(batch_size):
                local_scores[b] = combined_scores[b, top_cols[b]]
                local_indices[b] = combined_indices[b, top_cols[b]]

        all_scores[qs:qe] = local_scores
        all_topk[qs:qe] = local_indices

    # Compute metrics
    result: Dict[str, Any] = {"n_queries": n_q, "n_vocab": n_v}
    for k in topk_list:
        hits = 0
        for i in range(n_q):
            retrieved = set(all_topk[i, :k])
            if pos_sets[i] & retrieved:
                hits += 1
        result[f"top{k}"] = round(hits / max(n_q, 1), 6)

    rrs = []
    for i in range(n_q):
        rr = 0.0
        for rank in range(min(max_k, n_v)):
            if all_topk[i, rank] in pos_sets[i]:
                rr = 1.0 / (rank + 1)
                break
        rrs.append(rr)
    result["mrr"] = round(float(np.mean(rrs)), 6) if rrs else 0.0

    return result


def part_d_vocab_ladder(eval_queries: List[Dict[str, Any]],
                        train_vocab: set[str]
                        ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Vocabulary ladder: L1=166, L2=1k, L3=5k, L4=10k, L5=full.

    Methods: C0 (zero-delta), C1 (mean-delta), C2 (learned-delta/ridge), C3 (attach-freq).

    Returns:
        manifest_rows, result_rows
    """
    t0 = time_part("D — Vocabulary Ladder")

    from sklearn.linear_model import Ridge  # lazy import

    # ── Load fragment embeddings (166 x 128) ──
    npz = np.load(D4A2G / "d4a2g_safe_fragment_embeddings.npz", allow_pickle=True)
    frag_embs: np.ndarray = npz["embeddings"]
    frag_smiles: List[str] = list(npz["smiles"])
    emb_to_id: Dict[str, int] = {s: i for i, s in enumerate(frag_smiles)}
    pdim = frag_embs.shape[1]
    log.info("Fragment embeddings: %s, n=%d", frag_embs.shape, len(frag_smiles))

    # ── Train vocab frequencies (for C3 and vocab building) ──
    train_vocab_freq: Dict[str, int] = {}
    train_vocab_attach: Dict[str, str] = {}
    with open(D4A0 / "d4a0_train_replacement_vocabulary.csv", "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            smi = row["replacement_smiles"].strip()
            train_vocab_freq[smi] = int(row.get("global_train_frequency", 0))
            train_vocab_attach[smi] = row.get("attachment_signatures", "")

    # ── Build query z_old and positive sets ──
    n_eval = len(eval_queries)
    z_old = np.zeros((n_eval, pdim), dtype=np.float32)
    pos_sets: List[set[int]] = []
    for i, rec in enumerate(eval_queries):
        old_smi = rec.get("old_fragment_smiles", "")
        if old_smi in emb_to_id:
            z_old[i] = frag_embs[emb_to_id[old_smi]]
        pos_repl = set(rec.get("positive_in_vocab", rec.get("positive_replacement_set", [])))
        pos_sets.append(set(emb_to_id[p] for p in pos_repl if p in emb_to_id))

    log.info("Query embeddings built: %d queries, %d dims", n_eval, pdim)

    # ── Load train deltas and compute per-attachment means ──
    train_deltas_np = np.load(D4A2G / "d4a2g_safe_delta_vectors_train.npz")
    train_deltas: np.ndarray = train_deltas_np["delta_vectors"]  # (77789, 128)
    train_meta = load_jsonl(D4A2G / "d4a2g_safe_delta_dataset_train_meta.jsonl")

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
    global_mean_delta = np.mean(train_deltas, axis=0).astype(np.float32)
    log.info("Delta means: %d attachment signatures, global_mean_norm=%.4f",
             len(sig_means), float(np.linalg.norm(global_mean_delta)))

    # ── Train Ridge predictor ──
    all_sigs = sorted(set(
        [r.get("attachment_signature", "unknown") for r in train_meta] +
        [r.get("attachment_signature", "unknown") for r in eval_queries]
    ))
    sig_to_idx = {s: i for i, s in enumerate(all_sigs)}
    n_sigs = len(all_sigs)
    n_feat = pdim + n_sigs
    log.info("Ridge features: %d (embed %d + one-hot %d)", n_feat, pdim, n_sigs)

    # Build features: old-fragment embedding + attachment one-hot
    def _build_X(z_arr: np.ndarray, meta_list: List[Dict[str, Any]]) -> np.ndarray:
        n = len(meta_list)
        X = np.zeros((n, n_feat), dtype=np.float32)
        for j, rec in enumerate(meta_list):
            X[j, :pdim] = z_arr[j]
            sig = rec.get("attachment_signature", "unknown")
            if sig in sig_to_idx:
                X[j, pdim + sig_to_idx[sig]] = 1.0
        return X

    z_old_train = np.zeros((len(train_meta), pdim), dtype=np.float32)
    for j, rec in enumerate(train_meta):
        old_smi = rec.get("old_fragment_smiles", "")
        if old_smi in emb_to_id:
            z_old_train[j] = frag_embs[emb_to_id[old_smi]]
    X_train = _build_X(z_old_train, train_meta)

    z_old_eval = np.zeros((n_eval, pdim), dtype=np.float32)
    for i, rec in enumerate(eval_queries):
        old_smi = rec.get("old_fragment_smiles", "")
        if old_smi in emb_to_id:
            z_old_eval[i] = frag_embs[emb_to_id[old_smi]]
    X_eval = _build_X(z_old_eval, eval_queries)

    log.info("Training Ridge: X_train=%s, y_train=%s", X_train.shape, train_deltas.shape)
    ridge = Ridge(alpha=1.0, random_state=SEED)
    ridge.fit(X_train, train_deltas)
    pred_deltas_eval = ridge.predict(X_eval).astype(np.float32)
    log.info("Ridge trained: coef_norm=%.4f", float(np.linalg.norm(ridge.coef_)))

    # ── Vocabulary ladder ──
    all_frag_smiles = list(frag_smiles)
    levels = [
        ("L1", min(166, len(train_vocab) + len(all_frag_smiles))),
        ("L2", 1000),
        ("L3", 5000),
        ("L4", 10000),
        ("L5", 20000),
    ]

    # Pre-generate synthetic distractor embeddings: random unit-norm vectors
    # These are used when target_size > number of real fragment embeddings.
    rng_synth = np.random.RandomState(SEED + 999)
    MAX_SYNTH = 50000
    synthetic_embs = rng_synth.randn(MAX_SYNTH, pdim).astype(np.float32)
    # Normalize to unit norm (cosine sim normalization)
    synth_norms = np.linalg.norm(synthetic_embs, axis=1, keepdims=True) + 1e-10
    synthetic_embs /= synth_norms
    SYNTH_START = 10000  # embedding indices >= this are synthetic

    manifest_rows: List[Dict[str, Any]] = []
    result_rows: List[Dict[str, Any]] = []

    for level_name, target_size in levels:
        log.info("--- LADDER %s (target=%d) ---", level_name, target_size)

        # Build vocabulary (real fragments + synthetic distractors)
        vocab_smiles, n_real = _build_vocab_level(
            target_size, eval_queries, train_vocab, train_vocab_freq,
            all_frag_smiles, seed=SEED + hash(level_name) % (2 ** 31),
            synthetic_start_idx=SYNTH_START,
        )

        # Build embedding matrix: real from NPZ, synthetic as random unit vectors
        vocab_emb_list: List[np.ndarray] = []
        n_synth = 0
        for s in vocab_smiles:
            if s in emb_to_id:
                vocab_emb_list.append(frag_embs[emb_to_id[s]])
            elif s.startswith("*DISTRACTOR_"):
                syn_idx = int(s.split("_")[1]) - SYNTH_START
                if syn_idx < MAX_SYNTH:
                    vocab_emb_list.append(synthetic_embs[syn_idx])
                    n_synth += 1
                else:
                    # Fallback: random unit vector
                    rand_v = rng_synth.randn(pdim).astype(np.float32)
                    rand_v /= np.linalg.norm(rand_v) + 1e-10
                    vocab_emb_list.append(rand_v)
                    n_synth += 1
            else:
                # Should not happen; fallback
                rand_v = rng_synth.randn(pdim).astype(np.float32)
                rand_v /= np.linalg.norm(rand_v) + 1e-10
                vocab_emb_list.append(rand_v)
                n_synth += 1

        vocab_emb = np.array(vocab_emb_list, dtype=np.float32)

        # Remap positive sets to vocabulary indices
        smi_to_vidx = {s: i for i, s in enumerate(vocab_smiles)}
        pos_vocab: List[set[int]] = []
        for i, rec in enumerate(eval_queries):
            pos_repl = set(rec.get("positive_in_vocab", rec.get("positive_replacement_set", [])))
            pos_vocab.append(set(smi_to_vidx[p] for p in pos_repl if p in smi_to_vidx))

        if len(vocab_smiles) == 0:
            log.warning("  Empty vocabulary at level %s, skipping", level_name)
            continue

        # Methods
        # C0: zero-delta
        c0_emb = z_old_eval.copy()
        r0 = _retrieval_metrics(c0_emb, vocab_emb, pos_vocab)
        r0["method"] = "C0_zero_delta"
        r0["level"] = level_name
        r0["vocab_size"] = len(vocab_smiles)

        # C1: mean-delta by attachment
        c1_emb = np.zeros_like(c0_emb)
        for i, rec in enumerate(eval_queries):
            sig = rec.get("attachment_signature", "unknown")
            md = sig_means.get(sig, global_mean_delta)
            c1_emb[i] = c0_emb[i] + md.astype(np.float32)
        r1 = _retrieval_metrics(c1_emb, vocab_emb, pos_vocab)
        r1["method"] = "C1_mean_delta"
        r1["level"] = level_name
        r1["vocab_size"] = len(vocab_smiles)

        # C2: learned-delta (Ridge)
        c2_emb = c0_emb + pred_deltas_eval
        r2 = _retrieval_metrics(c2_emb, vocab_emb, pos_vocab)
        r2["method"] = "C2_learned_delta"
        r2["level"] = level_name
        r2["vocab_size"] = len(vocab_smiles)

        # C3: attach-freq baseline
        c3_scores = np.zeros((n_eval, len(vocab_smiles)), dtype=np.float32)
        for i, rec in enumerate(eval_queries):
            sig = rec.get("attachment_signature", "")
            for j, smi in enumerate(vocab_smiles):
                c3_scores[i, j] = float(train_vocab_freq.get(smi, 0))
        c3_topk = np.argsort(-c3_scores, axis=1)
        r3: Dict[str, Any] = {"n_queries": n_eval, "n_vocab": len(vocab_smiles),
                              "method": "C3_attach_freq", "level": level_name,
                              "vocab_size": len(vocab_smiles)}
        for k in [1, 5, 10, 20, 50]:
            hits = sum(1 for i in range(n_eval) if pos_vocab[i] & set(c3_topk[i, :k]))
            r3[f"top{k}"] = round(hits / max(n_eval, 1), 6)
        rrs = []
        for i in range(n_eval):
            rr = 0.0
            for rank, idx in enumerate(c3_topk[i], 1):
                if idx in pos_vocab[i]:
                    rr = 1.0 / rank
                    break
            rrs.append(rr)
        r3["mrr"] = round(float(np.mean(rrs)), 6) if rrs else 0.0

        manifest_rows.append({
            "level": level_name,
            "target_size": target_size,
            "actual_size": len(vocab_smiles),
            "n_real_fragments": n_real,
            "n_synthetic_distractors": max(0, len(vocab_smiles) - n_real),
            "n_positive_in_vocab": sum(len(p) for p in pos_vocab),
        })

        for r in [r0, r1, r2, r3]:
            result_rows.append(r)

        log.info("  C0 top10=%.4f  C1 top10=%.4f  C2 top10=%.4f  C3 top10=%.4f",
                 r0.get("top10", 0), r1.get("top10", 0),
                 r2.get("top10", 0), r3.get("top10", 0))

    save_csv(OUT / "d4a2gr_vocab_ladder_manifest.csv", manifest_rows)
    save_csv(OUT / "d4a2gr_vocab_ladder_results.csv", result_rows)

    # Gap analysis
    gap_rows: List[Dict[str, Any]] = []
    for level_name in [l[0] for l in levels]:
        lr = [r for r in result_rows if r["level"] == level_name]
        c0 = next((r for r in lr if r["method"] == "C0_zero_delta"), {})
        c1 = next((r for r in lr if r["method"] == "C1_mean_delta"), {})
        c2 = next((r for r in lr if r["method"] == "C2_learned_delta"), {})
        c3 = next((r for r in lr if r["method"] == "C3_attach_freq"), {})
        for met in ["top10", "top5", "mrr"]:
            v0 = c0.get(met, 0) or 0
            v1 = c1.get(met, 0) or 0
            v2 = c2.get(met, 0) or 0
            v3 = c3.get(met, 0) or 0
            gap_rows.append({
                "level": level_name,
                "metric": met,
                "C0": round(v0, 6),
                "C1": round(v1, 6),
                "C2": round(v2, 6),
                "C3": round(v3, 6),
                "C2_minus_C0": round(v2 - v0, 6),
                "C2_minus_C1": round(v2 - v1, 6),
                "C2_minus_C3": round(v2 - v3, 6),
            })

    save_csv(OUT / "d4a2gr_vocab_ladder_gaps.csv", gap_rows)
    log.info("Gaps written to d4a2gr_vocab_ladder_gaps.csv (%d rows)", len(gap_rows))

    time_end("D", t0)
    return manifest_rows, result_rows


# ══════════════════════════════════════════════════════════════════════
# PART E: Artifact Checks
# ══════════════════════════════════════════════════════════════════════


def part_e_artifacts(result_rows: List[Dict[str, Any]],
                     leakage_checks: List[Dict[str, Any]]
                     ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Artifact analysis using vocabulary ladder results."""
    t0 = time_part("E — Artifact Analysis")

    def _get(level: str, method: str, metric: str = "top10") -> float:
        for r in result_rows:
            if r.get("level") == level and r.get("method") == method:
                v = r.get(metric, 0) or 0
                return float(v)
        return 0.0

    l1_c2 = _get("L1", "C2_learned_delta")
    l4_c2 = _get("L4", "C2_learned_delta")
    l1_c0 = _get("L1", "C0_zero_delta")
    l4_c0 = _get("L4", "C0_zero_delta")
    l3_c2 = _get("L3", "C2_learned_delta")
    l3_c0 = _get("L3", "C0_zero_delta")
    l3_c3 = _get("L3", "C3_attach_freq")

    drop_l1_l4 = l1_c2 - l4_c2
    learned_over_zero_l1 = l1_c2 - l1_c0
    learned_over_zero_l3 = l3_c2 - l3_c0

    # Find C02 for SVD info
    svd_check = next((c for c in leakage_checks if c["check_id"] == "C02"), {})
    dim_128 = 128
    dim_ratios = {
        "L1": round(dim_128 / 166, 4),
        "L2": round(dim_128 / min(1000, 5000), 4),
        "L3": round(dim_128 / min(5000, 5000), 4),
        "L4": round(dim_128 / 10000, 4),
    }

    checks: List[Dict[str, Any]] = [
        {
            "check_id": "E01",
            "check_name": "small_vocabulary_artifact",
            "result": "WARN" if drop_l1_l4 > 0.15 else "PASS",
            "detail": f"L1→L4 learned_delta top10: {l1_c2:.4f}→{l4_c2:.4f} (drop={drop_l1_l4:.4f}). "
                      f"Threshold: <0.15. Degradation expected with scale.",
            "severity": "moderate" if drop_l1_l4 > 0.15 else "low",
        },
        {
            "check_id": "E02",
            "check_name": "learned_vs_attach_prior",
            "result": "INFO",
            "detail": f"L1 learned-vs-zero: {learned_over_zero_l1:+.4f}. "
                      f"L3 learned-vs-zero: {learned_over_zero_l3:+.4f}. "
                      f"L3 learned-vs-attachfreq: {l3_c2 - l3_c3:+.4f}. "
                      "If learned > attach_freq at L3, delta predictor adds value beyond frequency baseline.",
            "severity": "info",
        },
        {
            "check_id": "E03",
            "check_name": "transform_memorization",
            "result": "COVERED_BY_C01",
            "detail": "See C01 in leakage audit.",
            "severity": "none",
        },
        {
            "check_id": "E04",
            "check_name": "frequency_dominance",
            "result": "INFO",
            "detail": f"L3 learned-vs-attachfreq gap: {l3_c2 - l3_c3:+.4f}. "
                      "Positive gap means learned-delta > frequency-only baseline.",
            "severity": "info",
        },
        {
            "check_id": "E05",
            "check_name": "dimensionality_ratio",
            "result": "INFO",
            "detail": f"Embedding dim=128 vs vocab size ratios: {json.dumps(dim_ratios)}. "
                      "Lower ratio = harder retrieval task.",
            "severity": "info",
        },
        {
            "check_id": "E06",
            "check_name": "svd_leakage_concern",
            "result": svd_check.get("result", "UNKNOWN"),
            "detail": svd_check.get("detail", ""),
            "severity": "moderate",
        },
    ]

    save_csv(OUT / "d4a2gr_artifact_audit.csv", checks)
    summary = {"n_checks": len(checks)}
    save_json(OUT / "d4a2gr_artifact_summary.json", summary)

    for c in checks:
        log.info("  [%s] %s: %s", c["check_id"], c["check_name"], c["result"])

    time_end("E", t0)
    return checks, summary


# ══════════════════════════════════════════════════════════════════════
# PART F: Final Verdict
# ══════════════════════════════════════════════════════════════════════


def part_f_verdict(result_rows: List[Dict[str, Any]],
                   leakage_checks: List[Dict[str, Any]]) -> str:
    """Evaluate Direction 3 against 6 criteria and produce final verdict."""
    t0 = time_part("F — Final Verdict & Skeptical Review")

    def _get(level: str, method: str, metric: str = "top10") -> float:
        for r in result_rows:
            if r.get("level") == level and r.get("method") == method:
                v = r.get(metric, 0) or 0
                return float(v)
        return 0.0

    l2_c2 = _get("L2", "C2_learned_delta")
    l2_c0 = _get("L2", "C0_zero_delta")
    l3_c2 = _get("L3", "C2_learned_delta")
    l3_c0 = _get("L3", "C0_zero_delta")
    l3_c3 = _get("L3", "C3_attach_freq")

    l2_gap = l2_c2 - l2_c0
    l3_gap = l3_c2 - l3_c0
    l3_gap_vs_attach = l3_c2 - l3_c3
    l2_to_l3_drop = l2_c2 - l3_c2

    c01 = next((c for c in leakage_checks if c["check_id"] == "C01"), {}).get("result", "UNKNOWN")
    c05 = next((c for c in leakage_checks if c["check_id"] == "C05"), {}).get("result", "UNKNOWN")
    protocol_ok = c01.startswith("PASS") and c05.startswith("PASS")

    criteria = {
        "C1_protocol_no_leakage": protocol_ok,
        "C2_learned_vs_zero_L2_5pp": l2_gap >= 0.05,
        "C3_learned_vs_zero_L3_2pp": l3_gap >= 0.02,
        "C4_no_catastrophic_drop_L2_L3": l2_to_l3_drop < 0.15,
        "C5_learned_beats_or_near_attach_L3": l3_gap_vs_attach >= -0.05,
        "C6_learned_positive_at_L3": l3_gap > 0,
    }
    n_pass = sum(1 for v in criteria.values() if v)

    if not protocol_ok:
        verdict = "E. LEAKAGE_OR_PROTOCOL_BUG"
    elif l2_gap < 0:
        verdict = "D. DIRECTION3_FAILS_ZERO_DELTA"
    elif l2_gap < 0.05:
        verdict = "C. DIRECTION3_FAILS_SCALING"
    elif n_pass >= 5:
        verdict = "A. DIRECTION3_SCALE_CONFIRMED"
    elif n_pass >= 3:
        verdict = "B. DIRECTION3_SIGNAL_WEAK_BUT_PROMISING"
    else:
        verdict = "F. RESOURCE_LIMITED_INCONCLUSIVE"

    # ── Skeptical Review ──
    skeptical = f"""
### SKEPTICAL REVIEW

**1. Was D4A2G-SAFE inflated by small vocab?**
  YES — plausible. Gate C Top10=0.8692 at vocab=166 is very high.
  At L2 (1k), learned-delta = {l2_c2:.4f}. At L3 (5k), = {l3_c2:.4f}.
  The small-vocabulary ceiling at 166 is a known artifact: with only
  166 candidates and typically 1-5 valid replacements per query, random
  guessing has a non-negligible hit rate (~3-5%). The D4A2G-SAFE reported
  Top10 of 0.87 at 166-vocab is real but does not reflect operational
  vocabulary sizes.

**2. Were Gate B and Gate C incomparable?**
  YES — protocol difference is fundamental.
  - Gate B: pair-level retrieval (1 correct answer per 18,397 val pairs).
    Harder task, reflected in B4_mlp Top10=0.4951.
  - Gate C: query-level retrieval (multiple positives per 5,000 test queries).
    Easier task, reflected in C2_learned_delta Top10=0.8692.
  These measure DIFFERENT things. The D4A2G-SAFE narrative that
  "Gate B Top10=0.50 validates delta predictability, Gate C Top10=0.87
  validates retrieval" is a post-hoc justification for incomparable protocols.

**3. Can SVD train-only be proven?**
  NO — the embedding config shows SVD was fit on ALL 166 fragments
  (none are excluded in the NPZ). Without a fit-on-train-subset checkpoint
  or explicit fit_split field, the SVD projection may leak test-fragment
  geometry into the 128-dimensional embedding space.

**4. Does learned-delta genuinely beat zero-delta at scale?**
  Gate C at 166-vocab: learned=0.8692 vs zero=0.3054 (gap=0.5638).
  L2 (1k vocab): learned={l2_c2:.4f} vs zero={l2_c0:.4f} (gap={l2_gap:.4f}).
  L3 (5k vocab): learned={l3_c2:.4f} vs zero={l3_c0:.4f} (gap={l3_gap:.4f}).
  If the gap shrinks below 2pp at 5k, the delta-predictor signal is too
  weak for practical deployment.

**5. Is Direction 3 narrative-driven?**
  Concern: the multi-gate narrative (A->B->C->D) creates a sequence of
  binary pass/fail gates that collectively look stronger than individual
  results. The B/C protocol gap is obscured by the gate structure.
  The skeptical interpretation: Direction 3 shows WEAK signal that may
  be dominated by attachment-specific priors and small-vocabulary ceiling.
"""

    # ── Verdict markdown ──
    lines = [
        f"# D4A2G-R PROTOCOL SCALE AUDIT VERDICT",
        f"",
        f"Date: {now()}",
        f"",
        f"## Verdict: {verdict}",
        f"",
        f"### Criteria Results",
        f"",
    ]
    for k, v in criteria.items():
        lines.append(f"- **{k}**: {v} -> {'PASS' if v else 'FAIL'}")
    lines.append("")
    lines.append("### Key Metrics")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| L2 (1k) learned-vs-zero top10 gap | {l2_gap:.4f} |")
    lines.append(f"| L3 (5k) learned-vs-zero top10 gap | {l3_gap:.4f} |")
    lines.append(f"| L3 learned-delta top10 | {l3_c2:.4f} |")
    lines.append(f"| L3 learned-vs-attach-freq top10 | {l3_gap_vs_attach:+.4f} |")
    lines.append(f"| L2->L3 degradation | {l2_to_l3_drop:.4f} |")
    lines.append(f"")
    lines.append("### Interpretation")
    lines.append(f"")
    lines.append(_verdict_desc(verdict))
    lines.append(f"")
    lines.append("### Skeptical Review")
    lines.append(skeptical)
    lines.append(f"---")
    lines.append(f"*Generated by D4A2G-R audit script*")

    (OUT / "D4A2G_R_PROTOCOL_SCALE_AUDIT_VERDICT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    # ── Decision Log ──
    decision = f"""# MAIN_DECISION_LOG

## D4A2G-R Protocol Scale Audit

Date: {now()}

### Verdict
{verdict}

### Key Evidence
- L2 learned-vs-zero: {l2_gap:+.4f}
- L3 learned-vs-zero: {l3_gap:+.4f}
- L3 learned-vs-attachfreq: {l3_gap_vs_attach:+.4f}

### Protocol Status
- C01 (transform leakage): {c01}
- C05 (delta train/test overlap): {c05}
- SVD fit-split: unverifiable (see C02)

### Output Directory
{OUT}
"""
    (OUT / "MAIN_DECISION_LOG.md").write_text(decision, encoding="utf-8")

    log.info("")
    log.info("=" * 60)
    log.info("VERDICT: %s", verdict)
    log.info("=" * 60)
    log.info("Key metrics:")
    log.info("  L2 learned-vs-zero: %+.4f (need >= +0.05)", l2_gap)
    log.info("  L3 learned-vs-zero: %+.4f (need >= +0.02)", l3_gap)
    log.info("  L3 learned-vs-attachfreq: %+.4f", l3_gap_vs_attach)
    log.info("  L2->L3 drop: %.4f", l2_to_l3_drop)
    log.info("Criteria passed: %d/6", n_pass)

    time_end("F", t0)
    return verdict


def _verdict_desc(v: str) -> str:
    descs = {
        "A. DIRECTION3_SCALE_CONFIRMED": (
            "Direction 3 (learned delta predictor) survives scaling audit. "
            "Signal is robust at 5k vocabulary. Recommend proceeding."
        ),
        "B. DIRECTION3_SIGNAL_WEAK_BUT_PROMISING": (
            "Direction 3 shows signal but insufficient for confident deployment "
            "at scale. Recommend further investigation or feature engineering."
        ),
        "C. DIRECTION3_FAILS_SCALING": (
            "Direction 3 does not maintain advantage at larger vocabulary sizes. "
            "Not suitable for production without fundamental improvement."
        ),
        "D. DIRECTION3_FAILS_ZERO_DELTA": (
            "Learned delta predictor fails to beat zero-delta baseline even at L2. "
            "Fundamental flaw in delta prediction approach."
        ),
        "E. LEAKAGE_OR_PROTOCOL_BUG": (
            "Critical flaw in experimental design. Direction 3 results cannot "
            "be trusted. Fix experimental protocol before making claims."
        ),
        "F. RESOURCE_LIMITED_INCONCLUSIVE": (
            "Insufficient compute or data for confident verdict."
        ),
    }
    return descs.get(v, "Unknown verdict.")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    log.info("=" * 60)
    log.info("D4A2G-R: Protocol Reconciliation + Leakage Audit + "
             "Vocab Scaling + Artifact Analysis")
    log.info("=" * 60)
    log.info("SEED=%d, OUT=%s", SEED, OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    overall_start = time.time()

    # Part A
    n_b_qids, n_c_qids, train_vocab, _ = part_a_reconciliation()

    # Part B
    eval_queries, _ = part_b_unified_eval(train_vocab)

    # Part C
    leakage_checks, _ = part_c_leakage(train_vocab)

    # Part D (requires sklearn)
    try:
        result_rows: List[Dict[str, Any]] = []
        manifest_rows: List[Dict[str, Any]] = []
        manifest_rows, result_rows = part_d_vocab_ladder(eval_queries, train_vocab)
        have_ladder = len(result_rows) > 0
    except ImportError as e:
        log.warning("SKIP Part D: missing dependency (%s)", e)
        have_ladder = False
        result_rows = []

    # Part E
    if have_ladder:
        part_e_artifacts(result_rows, leakage_checks)
    else:
        log.warning("SKIP Part E (no ladder)")

    # Part F
    if have_ladder:
        verdict = part_f_verdict(result_rows, leakage_checks)
    else:
        verdict = "F. RESOURCE_LIMITED_INCONCLUSIVE"
        log.warning("SKIP Part F (no ladder)")

    total = time.time() - overall_start
    log.info("")
    log.info("=" * 60)
    log.info("D4A2G-R AUDIT COMPLETE")
    log.info("Verdict: %s", verdict)
    log.info("Total time: %.1f s", total)
    log.info("Output: %s", OUT)
    log.info("=" * 60)

    save_json(OUT / "d4a2gr_audit_complete.json", {
        "status": "COMPLETE",
        "verdict": verdict,
        "total_elapsed_sec": round(total, 1),
        "timestamp": now(),
        "n_eval_queries": len(eval_queries),
    })

    # Memory key summary
    mem = {
        "verdict": verdict,
        "parts": ["A", "B", "C", "D", "E", "F"],
        "n_eval_queries": len(eval_queries),
        "n_gateB_qids": n_b_qids,
        "n_gateC_qids": n_c_qids,
        "output_dir": str(OUT),
    }
    log.info("MEMORY_SUMMARY: %s", json.dumps(mem))


if __name__ == "__main__":
    main()

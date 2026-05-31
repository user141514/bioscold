#!/usr/bin/env python3
"""Route-A D4S0 SOTA opportunity decomposition and clean protocol lock."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

SEED = 20260525
TOPR = 50

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAN = PROJECT_ROOT / "plan_results"
OUT = PLAN / "routeA_chembl37k_d4s0_sota_opportunity"
OUT.mkdir(parents=True, exist_ok=True)

PHASE0 = PLAN / "routeA_chembl37k_d4p1_phase0_metric_lock"
PHASE1 = PLAN / "routeA_chembl37k_d4p1_phase1_subset_robustness"
PHASE2 = PLAN / "routeA_chembl37k_d4p1_phase2_component_contribution"
PHASE34 = PLAN / "routeA_chembl37k_d4p1_phase3_4_interpretability_cases"
D4A0 = PLAN / "routeA_chembl37k_d0d3_engineering_safe" / "07_d4a0_matrix_freeze"
D3R = PLAN / "routeA_chembl37k_d0d3_engineering_safe" / "06_d3r_exam_repair"
D4A1 = PLAN / "routeA_chembl37k_d4a1_learned_ranker"
D4A1R = PLAN / "routeA_chembl37k_d4a1r_ranker_audit"
D4A2D1_FULL = PLAN / "routeA_chembl37k_d4a2d1_full_gate"
D4A2D1R = PLAN / "routeA_chembl37k_d4a2d1r_dual_encoder_robustness"
D4A2D2 = PLAN / "routeA_chembl37k_d4a2d2_de_hgb_ensemble"
D4A3T = PLAN / "routeA_chembl37k_d4a3t_exploration_calibration"
D4A4 = PLAN / "routeA_chembl37k_d4a4_dual_mode_integration"

PHASE0_CANONICAL = PHASE0 / "d4p1_phase0_canonical_proposal_table.csv"
PHASE0_RECORDS = PHASE0 / "d4p1_phase0_extracted_metric_records.csv"
PHASE0_FIG = PHASE0 / "d4p1_phase0_fig_component_curve_data.csv"
PHASE0_VERDICT = PHASE0 / "D4P1_PHASE0_CANONICAL_METRIC_LOCK_VERDICT.md"
PHASE1_QUERY = PHASE1 / "d4p1_phase1_query_level_canonical_table.csv"
PHASE1_SUBSETS = PHASE1 / "d4p1_phase1_subset_definitions.csv"
PHASE1_METRICS = PHASE1 / "d4p1_phase1_subset_robustness_metrics.csv"
PHASE1_CLUSTERS = PHASE1 / "d4p1_phase1_old_fragment_cluster_metrics.csv"
PHASE1_VERDICT = PHASE1 / "D4P1_PHASE1_SUBSET_ROBUSTNESS_VERDICT.md"
PHASE2_COMPONENT = PHASE2 / "d4p1_phase2_component_contribution_metrics.csv"
PHASE2_SUBSET = PHASE2 / "d4p1_phase2_subset_component_matrix.csv"
PHASE2_HYP = PHASE2 / "d4p1_phase2_mechanism_hypothesis_tests.csv"
PHASE2_NEG = PHASE2 / "d4p1_phase2_negative_subspace_analysis.csv"
PHASE2_VERDICT = PHASE2 / "D4P1_PHASE2_COMPONENT_CONTRIBUTION_VERDICT.md"
PHASE2_INTERP = PHASE2 / "D4P1_PHASE2_COMPONENT_INTERPRETATION.md"
PHASE34_INPUT = PHASE34 / "d4p1_phase3_4_input_discovery.csv"
PHASE34_NEG = PHASE34 / "d4p1_phase3_negative_subspace_deep_dive.csv"
PHASE34_VERDICT = PHASE34 / "D4P1_PHASE3_4_INTERPRETABILITY_CASES_VERDICT.md"

MANIFEST = D4A0 / "d4a0_query_split_manifest.jsonl"
SPLIT_SUMMARY = D4A0 / "d4a0_split_summary.json"
VOCAB = D4A0 / "d4a0_train_replacement_vocabulary.csv"
TRAIN_DIR = D4A0 / "matrices" / "train"
VAL_DIR = D4A0 / "matrices" / "val"
TEST_DIR = D4A0 / "matrices" / "test"

DE_STD = D4A2D1R / "d4a2d1r_standardized_predictions.jsonl"
DE_QUERY_HITS = D4A2D1R / "d4a2d1r_query_level_hits.csv"
HGB_PREDS = D4A1 / "d4a1_test_predictions.jsonl"
HGB_TEST_METRICS = D4A1 / "d4a1_test_metrics.csv"
HGB_VAL_METRICS = D4A1 / "d4a1_validation_metrics.csv"
HGB_ARTIFACTS = D4A1 / "d4a1_model_artifacts_manifest.json"
QUERY_GATE_RESULTS = D4A2D2 / "d4a2d2_query_gate_results.csv"
QUERY_GATE_VAL = D4A2D2 / "d4a2d2_query_gate_val_results.csv"
RANK_FUSION_VAL_GRID = D4A2D2 / "d4a2d2_rank_fusion_val_grid.csv"
SCORE_FUSION_VAL_GRID = D4A2D2 / "d4a2d2_score_fusion_val_grid.csv"
VAL_DATA_NOTE = D4A2D2 / "d4a2d2_val_data_note.json"
VAL_METRICS_D4A2D1 = D4A2D1_FULL / "d4a2d1_full_val_metrics.csv"
VAL_SUBSET_D4A2D1 = D4A2D1_FULL / "d4a2d1_full_val_by_subset.csv"

SUBSET_ANNOT = D4A1R / "d4a1r_test_query_subset_annotations.csv"
D4A4_TIERS = D4A4 / "d4a4_candidate_final_tiers.csv"
D4A4_CANON = D4A4 / "d4a4_canonical_candidate_table.csv"
G1_LABELS = D4A3T / "d4a3t_candidate_labels_g1.csv"
PRIMARY_D3R_SPLIT = D3R / "d3r_query_split_transform_heldout_primary.jsonl"
CORE_D3R_SPLIT = D3R / "d3r_query_split_core_heldout_diagnostic.jsonl"
OLDFRAG_D3R_SPLIT = D3R / "d3r_query_split_old_fragment_heldout_diagnostic.jsonl"
REPL_D3R_SPLIT = D3R / "d3r_query_split_replacement_heldout_stress.jsonl"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_jsonl(path: Path) -> Iterable[dict]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def split_pipe(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(part) for part in value if str(part)]
    text = str(value)
    if text == "" or text.lower() == "nan":
        return []
    return [part for part in text.split("|") if part]


def pipe_join(values: Iterable[str]) -> str:
    ordered: List[str] = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        ordered.append(text)
        seen.add(text)
    return "|".join(ordered)


def rate(series: pd.Series) -> float:
    return float(series.astype(float).mean())


def pp(value: float) -> float:
    return 100.0 * float(value)


def load_manifest() -> pd.DataFrame:
    return pd.read_json(MANIFEST, lines=True)


def load_phase1_query() -> pd.DataFrame:
    df = pd.read_csv(PHASE1_QUERY)
    df["query_id"] = df["query_id"].astype(str)
    df["positive_replacements_exact_list"] = df["positive_replacements_exact"].apply(split_pipe)
    df["positive_replacements_exact_set"] = df["positive_replacements_exact_list"].apply(set)
    return df


def load_top50_standardized(canonical_qids: List[str]) -> pd.DataFrame:
    df = pd.read_json(DE_STD, lines=True)
    df = df.rename(
        columns={"q": "query_id", "m": "method", "r": "rank", "c": "candidate", "s": "score", "l": "is_pos"}
    )
    df["query_id"] = df["query_id"].astype(str)
    df = df[df["query_id"].isin(canonical_qids)].copy()
    df["rank"] = df["rank"].astype(int)
    df["is_pos"] = df["is_pos"].astype(int)
    return df


def list_shard_count(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    return len(list(path.glob("*.jsonl")))


def build_input_discovery() -> pd.DataFrame:
    specs = [
        (PHASE0_CANONICAL, "phase0", "canonical_table", "required", "proposal_table", "test"),
        (PHASE0_RECORDS, "phase0", "metric_records", "required", "metric_records", "test"),
        (PHASE0_FIG, "phase0", "component_curve_data", "required", "figure_data", "test"),
        (PHASE0_VERDICT, "phase0", "verdict", "required", "narrative", "test"),
        (PHASE1_QUERY, "phase1", "query_level_table", "required", "query_table", "test"),
        (PHASE1_SUBSETS, "phase1", "subset_definitions", "required", "subset_table", "test"),
        (PHASE1_METRICS, "phase1", "subset_metrics", "required", "subset_metrics", "test"),
        (PHASE1_CLUSTERS, "phase1", "cluster_metrics", "required", "cluster_metrics", "test"),
        (PHASE1_VERDICT, "phase1", "verdict", "required", "narrative", "test"),
        (PHASE2_COMPONENT, "phase2", "component_metrics", "required", "component_metrics", "test"),
        (PHASE2_SUBSET, "phase2", "subset_component_matrix", "required", "subset_metrics", "test"),
        (PHASE2_HYP, "phase2", "mechanism_hypotheses", "required", "diagnostic", "test"),
        (PHASE2_NEG, "phase2", "negative_subspace_analysis", "required", "diagnostic", "test"),
        (PHASE2_VERDICT, "phase2", "verdict", "required", "narrative", "test"),
        (PHASE2_INTERP, "phase2", "interpretation", "required", "narrative", "test"),
        (PHASE34_INPUT, "phase3_4", "input_discovery", "required", "diagnostic", "test"),
        (PHASE34_NEG, "phase3_4", "negative_subspace_deep_dive", "required", "diagnostic", "test"),
        (PHASE34_VERDICT, "phase3_4", "verdict", "required", "narrative", "test"),
        (MANIFEST, "d4a0", "query_manifest", "required", "metadata", "train/val/test"),
        (SPLIT_SUMMARY, "d4a0", "split_summary", "required", "metadata", "train/val/test"),
        (VOCAB, "d4a0", "train_vocab", "required", "metadata", "train"),
        (TRAIN_DIR, "d4a0", "train_feature_shards", "required", "raw_features", "train"),
        (VAL_DIR, "d4a0", "val_feature_shards", "required", "raw_features", "val"),
        (TEST_DIR, "d4a0", "test_feature_shards", "required", "raw_features", "test"),
        (DE_STD, "d4a2d1r", "standardized_predictions", "required", "DE+attach+HGB_top50", "test"),
        (DE_QUERY_HITS, "d4a2d1r", "query_hits", "optional", "DE+attach+HGB_top50", "test"),
        (HGB_PREDS, "d4a1", "test_predictions", "required", "HGB_scores", "test"),
        (HGB_TEST_METRICS, "d4a1", "test_metrics", "required", "HGB_metrics", "test"),
        (HGB_VAL_METRICS, "d4a1", "validation_metrics", "required", "HGB_metrics", "val"),
        (HGB_ARTIFACTS, "d4a1", "model_artifacts", "required", "HGB_features", "train/val/test"),
        (QUERY_GATE_RESULTS, "d4a2d2", "query_gate_results", "required", "Oracle_gate", "test"),
        (QUERY_GATE_VAL, "d4a2d2", "query_gate_val_results", "required", "Oracle_gate", "pseudo-val/test-exposed"),
        (RANK_FUSION_VAL_GRID, "d4a2d2", "rank_fusion_val_grid", "required", "fusion", "pseudo-val/test-exposed"),
        (SCORE_FUSION_VAL_GRID, "d4a2d2", "score_fusion_val_grid", "required", "fusion", "pseudo-val/test-exposed"),
        (VAL_DATA_NOTE, "d4a2d2", "val_data_note", "required", "fusion", "pseudo-val/test-exposed"),
        (VAL_METRICS_D4A2D1, "d4a2d1_full", "val_metrics", "required", "DE+attach", "val-aggregate"),
        (VAL_SUBSET_D4A2D1, "d4a2d1_full", "val_by_subset", "required", "DE+attach", "val-aggregate"),
        (SUBSET_ANNOT, "d4a1r", "subset_annotations", "optional", "subset_metadata", "test"),
        (D4A4_TIERS, "d4a4", "candidate_final_tiers", "optional", "A4C", "test-diagnostic"),
        (D4A4_CANON, "d4a4", "canonical_candidate_table", "optional", "A4C+provenance", "test-diagnostic"),
        (G1_LABELS, "d4a3t", "g1_labels", "optional", "A4C", "test-diagnostic"),
        (PRIMARY_D3R_SPLIT, "d3r", "primary_transform_heldout_split", "optional", "split_manifest", "historical"),
        (CORE_D3R_SPLIT, "d3r", "core_heldout_split", "optional", "split_manifest", "historical"),
        (OLDFRAG_D3R_SPLIT, "d3r", "old_fragment_heldout_split", "optional", "split_manifest", "historical"),
        (REPL_D3R_SPLIT, "d3r", "replacement_heldout_split", "optional", "split_manifest", "historical"),
    ]
    rows = []
    for path, stage, role, req, method, split in specs:
        row = {
            "file_path": str(path),
            "stage": stage,
            "role": role,
            "required_or_optional": req,
            "method": method,
            "split": split,
            "has_query_id": False,
            "has_positive_set": False,
            "has_rank": False,
            "has_score": False,
            "has_topK": False,
            "has_hit_flag": False,
            "has_candidate_smiles": False,
            "has_subset_fields": False,
            "has_a4c_fields": False,
            "status": "FOUND" if path.exists() else "MISSING",
            "notes": "",
        }
        if not path.exists():
            rows.append(row)
            continue
        if path.is_dir():
            row["notes"] = f"{list_shard_count(path)} jsonl shards"
            rows.append(row)
            continue
        suffix = path.suffix.lower()
        if suffix == ".csv":
            head = pd.read_csv(path, nrows=3)
            cols = set(head.columns)
            row["has_query_id"] = any(c in cols for c in ["query_id", "qid", "q"])
            row["has_positive_set"] = any(
                c in cols for c in ["positive_replacements_exact", "positive_replacement_set", "positive_replacement_smiles"]
            )
            row["has_rank"] = any(
                c in cols for c in ["rank", "rank_HGB", "rank_DE", "rank_Borda", "attach_best_rank", "DE_best_rank", "HGB_best_rank"]
            )
            row["has_score"] = any(c in cols for c in ["score", "score_DE", "score_HGB", "score_Borda", "metric_value"])
            row["has_topK"] = any(c in cols for c in ["Top10", "Attach_Top10", "DE_Top10", "HGB_Top10", "Borda_Top10"])
            row["has_hit_flag"] = any(c in cols for c in ["hit_10", "attach_hit10", "DE_hit10", "HGB_hit10", "Borda_hit10"])
            row["has_candidate_smiles"] = any(c in cols for c in ["candidate", "candidate_norm", "replacement_smiles"])
            row["has_subset_fields"] = any(c in cols for c in ["old_fragment_cluster_id", "target_replacement_frequency_bin", "attachment_signature"])
            row["has_a4c_fields"] = any(c in cols for c in ["final_action_tier", "a4c_review_bucket", "group_origin", "hard_alert_flag"])
        elif suffix == ".jsonl":
            first = next(load_jsonl(path))
            keys = set(first.keys())
            row["has_query_id"] = any(k in keys for k in ["query_id", "qid", "q"])
            row["has_positive_set"] = any(k in keys for k in ["positive_replacement_set", "label", "l"])
            row["has_rank"] = any(k in keys for k in ["rank", "r"])
            row["has_score"] = any(k in keys for k in ["score", "s", "attach_freq", "global_freq"])
            row["has_topK"] = any(k in keys for k in ["rank", "r"])
            row["has_hit_flag"] = any(k in keys for k in ["hit10", "h10"])
            row["has_candidate_smiles"] = any(k in keys for k in ["candidate", "c"])
            row["has_subset_fields"] = any(k in keys for k in ["split", "old_fragment_smiles", "attachment_signature"])
            row["has_a4c_fields"] = any(k in keys for k in ["final_action_tier", "a4c_review_bucket", "hard_alert_flag"])
        elif suffix == ".json":
            row["notes"] = "Metadata json"
        elif suffix == ".md":
            row["notes"] = "Narrative markdown"
        rows.append(row)
    return pd.DataFrame(rows)


def build_signal_availability() -> pd.DataFrame:
    rows = [
        {
            "signal": "attachment_frequency",
            "rank_available": True,
            "score_available": True,
            "full_vocab_rank_available": False,
            "topK_only": True,
            "val_available": False,
            "test_available": True,
            "train_available": False,
            "usable_for_reranker": True,
            "usable_for_fusion": True,
            "usable_for_guard": True,
            "notes": "Per-query artifact is top50-only in d4a2d1r_standardized_predictions; val has aggregate metrics only.",
        },
        {
            "signal": "DE",
            "rank_available": True,
            "score_available": True,
            "full_vocab_rank_available": False,
            "topK_only": True,
            "val_available": False,
            "test_available": True,
            "train_available": False,
            "usable_for_reranker": True,
            "usable_for_fusion": True,
            "usable_for_guard": True,
            "notes": "Clean canonical artifact is top50-only; existing val artifacts are pseudo-val/test-exposed.",
        },
        {
            "signal": "HGB",
            "rank_available": True,
            "score_available": True,
            "full_vocab_rank_available": False,
            "topK_only": False,
            "val_available": False,
            "test_available": True,
            "train_available": False,
            "usable_for_reranker": True,
            "usable_for_fusion": True,
            "usable_for_guard": True,
            "notes": "Test scores are available; current repo lacks per-query val predictions. Candidate rows exceed topK but are not emitted as a canonical 152-vocab table.",
        },
        {
            "signal": "Borda",
            "rank_available": True,
            "score_available": True,
            "full_vocab_rank_available": False,
            "topK_only": True,
            "val_available": False,
            "test_available": True,
            "train_available": False,
            "usable_for_reranker": True,
            "usable_for_fusion": False,
            "usable_for_guard": True,
            "notes": "Canonical Borda can be reconstructed on test from DE/HGB top50. Existing val fusion grids are pseudo-val diagnostics only.",
        },
        {
            "signal": "Oracle",
            "rank_available": False,
            "score_available": False,
            "full_vocab_rank_available": False,
            "topK_only": False,
            "val_available": False,
            "test_available": True,
            "train_available": False,
            "usable_for_reranker": False,
            "usable_for_fusion": False,
            "usable_for_guard": False,
            "notes": "Oracle gate uses test labels and is diagnostic only. It is not a deployable selector and not a strict upper bound for Borda.",
        },
        {
            "signal": "A4C",
            "rank_available": False,
            "score_available": False,
            "full_vocab_rank_available": False,
            "topK_only": False,
            "val_available": False,
            "test_available": True,
            "train_available": False,
            "usable_for_reranker": False,
            "usable_for_fusion": False,
            "usable_for_guard": False,
            "notes": "TopK/provenance coverage only; review proxy, not proposal-ground-truth signal.",
        },
        {
            "signal": "old_fragment_cluster",
            "rank_available": False,
            "score_available": False,
            "full_vocab_rank_available": False,
            "topK_only": False,
            "val_available": False,
            "test_available": True,
            "train_available": False,
            "usable_for_reranker": False,
            "usable_for_fusion": False,
            "usable_for_guard": False,
            "notes": "Current cluster labels were derived on the analyzed test set; re-clustering would be required for clean future use.",
        },
        {
            "signal": "replacement_frequency",
            "rank_available": False,
            "score_available": True,
            "full_vocab_rank_available": False,
            "topK_only": False,
            "val_available": True,
            "test_available": True,
            "train_available": True,
            "usable_for_reranker": True,
            "usable_for_fusion": True,
            "usable_for_guard": True,
            "notes": "Available from raw matrix frequencies or derivable from train-vocab counts; train-safe if recomputed without test labels.",
        },
        {
            "signal": "candidate_descriptors",
            "rank_available": False,
            "score_available": True,
            "full_vocab_rank_available": False,
            "topK_only": False,
            "val_available": True,
            "test_available": True,
            "train_available": True,
            "usable_for_reranker": True,
            "usable_for_fusion": False,
            "usable_for_guard": False,
            "notes": "Derivable from candidate SMILES and old-fragment SMILES; not emitted as a ready-made matrix but available to recompute.",
        },
        {
            "signal": "property_deltas",
            "rank_available": False,
            "score_available": True,
            "full_vocab_rank_available": False,
            "topK_only": False,
            "val_available": True,
            "test_available": True,
            "train_available": True,
            "usable_for_reranker": True,
            "usable_for_fusion": False,
            "usable_for_guard": False,
            "notes": "Derivable from SMILES; HGB artifact manifest confirms heavy_atom_delta and related descriptor families.",
        },
    ]
    return pd.DataFrame(rows)


def load_phase0_references() -> Dict[str, float]:
    table = pd.read_csv(PHASE0_CANONICAL)
    refs = {}
    for method, label in [
        ("Attachment_frequency", "Attachment_frequency"),
        ("DE", "DE"),
        ("HGB", "HGB"),
        ("Borda(DE,HGB)", "Borda(DE,HGB)"),
    ]:
        refs[label] = float(table.loc[table["method"] == method, "Top10"].iloc[0])
    phase2 = pd.read_csv(PHASE2_COMPONENT)
    refs["Oracle(DE,HGB)"] = float(phase2.loc[phase2["method_id"] == "M8", "Top10"].iloc[0])
    return refs


def _series_from_group(grouped: pd.Series, qindex: pd.Index, fill_value: float | int) -> pd.Series:
    series = grouped.reindex(qindex)
    if isinstance(fill_value, (int, np.integer)):
        return series.fillna(fill_value).astype(int)
    return series.fillna(fill_value).astype(float)


def reconstruct_core_metrics(top50: pd.DataFrame, canonical_qids: List[str]) -> dict:
    qindex = pd.Index(canonical_qids, name="query_id")
    results: dict = {"method_lists": {}, "hit10": {}, "best_rank": {}}
    method_map = {
        "M1_attach": "Attachment_frequency",
        "M2_DE": "DE",
        "M3_HGB": "HGB",
    }

    for raw_method, label in method_map.items():
        sub = top50[top50["method"] == raw_method].sort_values(["query_id", "rank"])
        results["method_lists"][label] = (
            sub.groupby("query_id")["candidate"].apply(list).reindex(qindex).apply(lambda x: x if isinstance(x, list) else []).to_dict()
        )
        hit10 = sub[sub["rank"] <= 10].groupby("query_id")["is_pos"].max()
        best_rank = sub[sub["is_pos"] == 1].groupby("query_id")["rank"].min()
        results["hit10"][label] = _series_from_group(hit10, qindex, 0)
        results["best_rank"][label] = _series_from_group(best_rank, qindex, math.inf)

    de = top50[top50["method"] == "M2_DE"][["query_id", "candidate", "rank", "is_pos"]].rename(columns={"rank": "de_rank"})
    hgb = top50[top50["method"] == "M3_HGB"][["query_id", "candidate", "rank"]].rename(columns={"rank": "hgb_rank"})
    borda = de.merge(hgb, on=["query_id", "candidate"], how="left")
    borda["hgb_rank"] = borda["hgb_rank"].fillna(TOPR + 1).astype(int)
    borda["borda_score"] = (TOPR - borda["de_rank"] + 1) + (TOPR - borda["hgb_rank"] + 1)
    borda = borda.sort_values(["query_id", "borda_score", "candidate"], ascending=[True, False, True], kind="mergesort").copy()
    borda["borda_rank"] = borda.groupby("query_id").cumcount() + 1
    results["borda_table"] = borda
    results["method_lists"]["Borda(DE,HGB)"] = (
        borda.groupby("query_id")["candidate"].apply(list).reindex(qindex).apply(lambda x: x if isinstance(x, list) else []).to_dict()
    )
    borda_hit10 = borda[borda["borda_rank"] <= 10].groupby("query_id")["is_pos"].max()
    borda_best = borda[borda["is_pos"] == 1].groupby("query_id")["borda_rank"].min()
    results["hit10"]["Borda(DE,HGB)"] = _series_from_group(borda_hit10, qindex, 0)
    results["best_rank"]["Borda(DE,HGB)"] = _series_from_group(borda_best, qindex, math.inf)

    oracle = np.maximum(results["hit10"]["DE"].to_numpy(), results["hit10"]["HGB"].to_numpy())
    results["hit10"]["Oracle(DE,HGB)"] = pd.Series(oracle, index=qindex, name="Oracle(DE,HGB)").astype(int)
    oracle_best = np.minimum(results["best_rank"]["DE"].to_numpy(), results["best_rank"]["HGB"].to_numpy())
    results["best_rank"]["Oracle(DE,HGB)"] = pd.Series(oracle_best, index=qindex, name="Oracle(DE,HGB)")
    return results


def build_metric_reconstruction(recon: dict, refs: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for label in ["Attachment_frequency", "DE", "HGB", "Borda(DE,HGB)", "Oracle(DE,HGB)"]:
        reconstructed = rate(recon["hit10"][label])
        reference = refs[label]
        abs_diff_pp = abs(pp(reconstructed - reference))
        rows.append(
            {
                "method": label,
                "reconstructed_top10": reconstructed,
                "reference_top10": reference,
                "abs_diff_pp": abs_diff_pp,
                "status": "PASS" if abs_diff_pp <= 0.5 else "FAIL",
                "reference_source": "phase0" if label != "Oracle(DE,HGB)" else "phase2_carry_through",
            }
        )
    return pd.DataFrame(rows)


def build_oracle_gap_tables(phase1: pd.DataFrame, recon: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    query = phase1.copy()
    query["split"] = "test"
    query["attach_hit10"] = query["attach_hit10"].astype(int)
    query["DE_hit10"] = query["DE_hit10"].astype(int)
    query["HGB_hit10"] = query["HGB_hit10"].astype(int)
    query["Borda_hit10"] = query["Borda_hit10"].astype(int)
    query["Oracle_hit10"] = query["Oracle_hit10_if_available"].astype(int)
    query["Oracle_best_rank"] = query["Oracle_best_rank_if_available"]
    query["num_positives"] = query["num_positive_replacements_total"].astype(int)
    query["rare_replacement_flag"] = query["target_replacement_frequency_bin"].eq("rare_replacement").astype(int)
    query["hard_top10_miss_flag"] = query["attach_hit10"].eq(0).astype(int)
    query["negative_subspace_c_o_flag"] = query["attachment_signature"].eq("C|O").astype(int)
    query["negative_subspace_cluster_05_flag"] = query["old_fragment_cluster_id"].eq("cluster_05").astype(int)
    query["negative_subspace_cluster_09_flag"] = query["old_fragment_cluster_id"].eq("cluster_09").astype(int)
    query["negative_subspace_flag"] = (
        (query["negative_subspace_c_o_flag"] == 1)
        | (query["negative_subspace_cluster_05_flag"] == 1)
        | (query["negative_subspace_cluster_09_flag"] == 1)
    ).astype(int)
    query["negative_subspace_flags"] = query.apply(
        lambda row: pipe_join(
            [
                "C|O" if row["negative_subspace_c_o_flag"] == 1 else "",
                "cluster_05" if row["negative_subspace_cluster_05_flag"] == 1 else "",
                "cluster_09" if row["negative_subspace_cluster_09_flag"] == 1 else "",
            ]
        ),
        axis=1,
    )
    query["C0_borda_hit"] = query["Borda_hit10"].eq(1).astype(int)
    query["C1_oracle_hit_borda_miss"] = ((query["Oracle_hit10"] == 1) & (query["Borda_hit10"] == 0)).astype(int)
    query["C2_de_hit_borda_miss"] = ((query["DE_hit10"] == 1) & (query["Borda_hit10"] == 0)).astype(int)
    query["C3_hgb_hit_borda_miss"] = ((query["HGB_hit10"] == 1) & (query["Borda_hit10"] == 0)).astype(int)
    query["C4_de_or_hgb_hit_borda_miss"] = (
        ((query["DE_hit10"] == 1) | (query["HGB_hit10"] == 1)) & (query["Borda_hit10"] == 0)
    ).astype(int)
    query["C5_de_and_hgb_hit_borda_miss"] = (
        (query["DE_hit10"] == 1) & (query["HGB_hit10"] == 1) & (query["Borda_hit10"] == 0)
    ).astype(int)
    query["C6_neither_de_nor_hgb_hit_but_oracle_hit"] = (
        (query["DE_hit10"] == 0) & (query["HGB_hit10"] == 0) & (query["Oracle_hit10"] == 1) & (query["Borda_hit10"] == 0)
    ).astype(int)
    query["C7_all_miss"] = (
        (query["DE_hit10"] == 0) & (query["HGB_hit10"] == 0) & (query["Borda_hit10"] == 0) & (query["Oracle_hit10"] == 0)
    ).astype(int)
    query["C8_negative_subspace_loss"] = (
        (query["negative_subspace_flag"] == 1) & (query["HGB_hit10"] == 1) & (query["Borda_hit10"] == 0)
    ).astype(int)
    query["recoverable_by_de_or_hgb_signal"] = query["C1_oracle_hit_borda_miss"]
    query["requires_new_signal"] = query["C6_neither_de_nor_hgb_hit_but_oracle_hit"]
    query["borda_only_vs_oracle_flag"] = ((query["Borda_hit10"] == 1) & (query["Oracle_hit10"] == 0)).astype(int)
    query["primary_category"] = np.select(
        [
            query["C0_borda_hit"] == 1,
            query["C1_oracle_hit_borda_miss"] == 1,
            query["C7_all_miss"] == 1,
        ],
        ["C0_borda_hit", "C1_oracle_hit_borda_miss", "C7_all_miss"],
        default="other_diagnostic_overlap",
    )

    total = float(len(query))
    oracle_only = int(query["C1_oracle_hit_borda_miss"].sum())
    borda_only_vs_oracle = int(query["borda_only_vs_oracle_flag"].sum())
    visible_gap = rate(query["Oracle_hit10"]) - rate(query["Borda_hit10"])
    summary_rows = [
        {"metric": "total_queries", "count": int(total), "rate": 1.0, "notes": "Canonical seen-vocab transform-heldout test queries."},
        {"metric": "Borda_hit_count", "count": int(query["Borda_hit10"].sum()), "rate": rate(query["Borda_hit10"]), "notes": "Canonical Borda(DE,HGB) Top10 hits."},
        {"metric": "Oracle_hit_count", "count": int(query["Oracle_hit10"].sum()), "rate": rate(query["Oracle_hit10"]), "notes": "Oracle(DE,HGB) query gate Top10 hits."},
        {
            "metric": "visible_oracle_gap_net",
            "count": int(query["Oracle_hit10"].sum() - query["Borda_hit10"].sum()),
            "rate": visible_gap,
            "notes": "Net Oracle minus Borda. Not a strict fusion upper bound.",
        },
        {
            "metric": "oracle_only_gap_queries",
            "count": oracle_only,
            "rate": oracle_only / total,
            "notes": "Oracle hits while Borda misses.",
        },
        {
            "metric": "borda_only_vs_oracle_queries",
            "count": borda_only_vs_oracle,
            "rate": borda_only_vs_oracle / total,
            "notes": "Borda hits while Oracle gate misses because Oracle only switches between DE/HGB top10 lists.",
        },
        {
            "metric": "fusion_recoverable_gap",
            "count": int(query["recoverable_by_de_or_hgb_signal"].sum()),
            "rate": rate(query["recoverable_by_de_or_hgb_signal"]),
            "notes": "Visible Oracle-only gap recoverable by a better DE/HGB selector over current Borda.",
        },
        {
            "metric": "new_signal_gap",
            "count": int(query["requires_new_signal"].sum()),
            "rate": rate(query["requires_new_signal"]),
            "notes": "Zero under the current Oracle definition; does not rule out deeper-rank gains from existing signals.",
        },
        {
            "metric": "negative_subspace_related_gap",
            "count": int(((query["C1_oracle_hit_borda_miss"] == 1) & (query["negative_subspace_flag"] == 1)).sum()),
            "rate": float(((query["C1_oracle_hit_borda_miss"] == 1) & (query["negative_subspace_flag"] == 1)).mean()),
            "notes": "Oracle-only gap inside C|O / cluster_05 / cluster_09.",
        },
        {
            "metric": "rare_related_gap",
            "count": int(((query["C1_oracle_hit_borda_miss"] == 1) & (query["rare_replacement_flag"] == 1)).sum()),
            "rate": float(((query["C1_oracle_hit_borda_miss"] == 1) & (query["rare_replacement_flag"] == 1)).mean()),
            "notes": "Oracle-only gap inside rare_replacement.",
        },
        {
            "metric": "hard_related_gap",
            "count": int(((query["C1_oracle_hit_borda_miss"] == 1) & (query["hard_top10_miss_flag"] == 1)).sum()),
            "rate": float(((query["C1_oracle_hit_borda_miss"] == 1) & (query["hard_top10_miss_flag"] == 1)).mean()),
            "notes": "Oracle-only gap inside attachment-frequency hard_top10_miss.",
        },
        {
            "metric": "C5_de_and_hgb_hit_borda_miss",
            "count": int(query["C5_de_and_hgb_hit_borda_miss"].sum()),
            "rate": rate(query["C5_de_and_hgb_hit_borda_miss"]),
            "notes": "Very small set where both base models hit but current Borda still misses.",
        },
        {
            "metric": "C6_neither_de_nor_hgb_hit_but_oracle_hit",
            "count": int(query["C6_neither_de_nor_hgb_hit_but_oracle_hit"].sum()),
            "rate": rate(query["C6_neither_de_nor_hgb_hit_but_oracle_hit"]),
            "notes": "Should be zero under the DE/HGB query-gate Oracle definition.",
        },
    ]
    return query, pd.DataFrame(summary_rows)


def parse_transform_keys(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x)]
    return split_pipe(value)


def build_protocol_audit(manifest: pd.DataFrame) -> tuple[pd.DataFrame, str, dict]:
    with open(SPLIT_SUMMARY, encoding="utf-8") as handle:
        split_summary = json.load(handle)
    with open(VAL_DATA_NOTE, encoding="utf-8") as handle:
        val_note = json.load(handle)
    transform_sets = {}
    for split, sub in manifest.groupby("split"):
        keys = set()
        for value in sub["transform_key_set"]:
            keys.update(parse_transform_keys(value))
        transform_sets[split] = keys
    overlaps = {
        "train_val": len(transform_sets.get("train", set()) & transform_sets.get("val", set())),
        "train_test": len(transform_sets.get("train", set()) & transform_sets.get("test", set())),
        "val_test": len(transform_sets.get("val", set()) & transform_sets.get("test", set())),
    }
    rows = [
        {
            "audit_item": "split_counts",
            "status": "PASS",
            "evidence": f"train={split_summary['train']}, val={split_summary['val']}, test={split_summary['test']}, seen_vocab_test={split_summary['seen_vocab_test']}",
            "implication": "Frozen train/val/test split exists.",
        },
        {
            "audit_item": "transform_key_overlap",
            "status": "PASS" if max(overlaps.values()) == 0 else "FAIL",
            "evidence": f"train_val={overlaps['train_val']}, train_test={overlaps['train_test']}, val_test={overlaps['val_test']}",
            "implication": "Current frozen split is clean at the transform-key level.",
        },
        {
            "audit_item": "per_query_val_predictions_DE",
            "status": "FAIL",
            "evidence": "Current repo has aggregate val metrics but not clean per-query DE val predictions for D4S fusion selection.",
            "implication": "Existing fusion selection path is not clean.",
        },
        {
            "audit_item": "per_query_val_predictions_HGB",
            "status": "FAIL",
            "evidence": val_note["hgb_val_top10"],
            "implication": "HGB was not run on val in the current artifact set.",
        },
        {
            "audit_item": "per_query_val_predictions_Borda",
            "status": "FAIL",
            "evidence": val_note["pseudo_val"],
            "implication": "Existing Borda val grids are pseudo-val/test-exposed diagnostics only.",
        },
        {
            "audit_item": "val_labels_available",
            "status": "PASS",
            "evidence": "Frozen D4A0 val matrix shards contain query_id, candidate, label, global_freq, attach_freq.",
            "implication": "A clean validation split exists at the raw feature level.",
        },
        {
            "audit_item": "val_subset_labels_available",
            "status": "PASS",
            "evidence": "Manifest contains old_fragment_smiles, attachment_signature, positive_replacement_set, num_positive_replacements, transform_key_set.",
            "implication": "Subset annotations are derivable on val without using test feedback.",
        },
        {
            "audit_item": "current_test_exposure",
            "status": "FAIL",
            "evidence": "D4P1 Phases 1/2/3/4 already mined the canonical test for robustness, mechanism, negative subspaces, and cases.",
            "implication": "New methods chosen from these patterns are post-hoc on the current test.",
        },
        {
            "audit_item": "secondary_blind_split_already_frozen",
            "status": "FAIL",
            "evidence": "No secondary blind split artifact exists under plan_results.",
            "implication": "Paper-main SOTA work would need an explicit new split freeze.",
        },
        {
            "audit_item": "secondary_blind_split_feasible",
            "status": "PASS",
            "evidence": f"Current manifest has train={len(transform_sets.get('train', set()))}, val={len(transform_sets.get('val', set()))}, test={len(transform_sets.get('test', set()))} unique transform-key sets with zero overlap, and historical D3R split manifests are present.",
            "implication": "A fresh secondary transform-heldout split is operationally feasible, but must be explicitly frozen.",
        },
        {
            "audit_item": "guard_selection_on_current_test_patterns",
            "status": "FAIL",
            "evidence": "C|O, cluster_05, and cluster_09 were discovered on the analyzed test set.",
            "implication": "Any guard keyed to those patterns is diagnostic until revalidated on a fresh split.",
        },
    ]
    classification = "C. NEW_BLIND_SPLIT_RECOMMENDED"
    extras = {
        "transform_counts": {split: len(keys) for split, keys in transform_sets.items()},
        "transform_overlaps": overlaps,
        "val_note": val_note,
        "split_summary": split_summary,
    }
    return pd.DataFrame(rows), classification, extras


def write_protocol_markdown(protocol_df: pd.DataFrame, classification: str, extras: dict) -> None:
    lines = [
        "# D4S0 Clean Evaluation Protocol",
        "",
        f"Protocol classification: **{classification}**",
        "",
        "## Audit Summary",
        "",
        f"- Frozen split counts: train={extras['split_summary']['train']}, val={extras['split_summary']['val']}, test={extras['split_summary']['test']}, seen-vocab-test={extras['split_summary']['seen_vocab_test']}.",
        f"- Transform-key counts: train={extras['transform_counts'].get('train', 0)}, val={extras['transform_counts'].get('val', 0)}, test={extras['transform_counts'].get('test', 0)}.",
        f"- Transform-key overlap: train/val={extras['transform_overlaps']['train_val']}, train/test={extras['transform_overlaps']['train_test']}, val/test={extras['transform_overlaps']['val_test']}.",
        f"- Current fusion val note: {extras['val_note']['note']}",
        f"- Pseudo-val note: {extras['val_note']['pseudo_val']}",
        "",
        "## Interpretation",
        "",
        "- The current canonical test has already been used for robustness, mechanism, negative-subspace, and case-study analysis in D4P1.",
        "- Existing DE/HGB/Borda selection artifacts do not provide a clean per-query validation path for new fusion or guard selection.",
        "- Frozen train/val raw matrices do exist, so clean development is still possible at the data level.",
        "- Because the negative subspaces were discovered on the canonical test, any guard targeting them is post-hoc unless validated on a new blind split.",
        "- A fresh secondary transform-heldout split is recommended before claiming any new paper-main SOTA gain beyond Borda.",
        "",
        "## Allowed Next-Step Regimes",
        "",
        "1. Diagnostic-only on current test:",
        "   - Report D4S1/D4S2 as post-hoc diagnostics with full transparency.",
        "2. Paper-main path:",
        "   - Freeze a new blind split.",
        "   - Select methods on clean val only.",
        "   - Evaluate the locked method once on the new blind test.",
        "",
        "## Fail-Closed Rules",
        "",
        "- Do not use C|O / cluster_05 / cluster_09 guards as paper-main claims on the current test.",
        "- Do not use pseudo-val fusion grids as clean validation evidence.",
        "- Do not treat Oracle(DE,HGB) as a deployable model or a strict upper bound for Borda.",
    ]
    (OUT / "D4S0_CLEAN_EVALUATION_PROTOCOL.md").write_text("\n".join(lines), encoding="utf-8")


def build_negative_subspace_opportunity(query: pd.DataFrame) -> pd.DataFrame:
    total_queries = float(len(query))
    subset_specs = [
        ("C|O", query["attachment_signature"].eq("C|O")),
        ("cluster_05", query["old_fragment_cluster_id"].eq("cluster_05")),
        ("cluster_09", query["old_fragment_cluster_id"].eq("cluster_09")),
        (
            "union_negative_subspaces",
            query["attachment_signature"].eq("C|O")
            | query["old_fragment_cluster_id"].eq("cluster_05")
            | query["old_fragment_cluster_id"].eq("cluster_09"),
        ),
    ]
    rows = []
    for name, mask in subset_specs:
        sub = query.loc[mask].copy()
        if sub.empty:
            continue
        switch_hgb_subset = rate(sub["HGB_hit10"]) - rate(sub["Borda_hit10"])
        switch_attach_subset = rate(sub["attach_hit10"]) - rate(sub["Borda_hit10"])
        rows.append(
            {
                "subset_name": name,
                "split": "test",
                "N": int(len(sub)),
                "Attach_Top10": rate(sub["attach_hit10"]),
                "DE_Top10": rate(sub["DE_hit10"]),
                "HGB_Top10": rate(sub["HGB_hit10"]),
                "Borda_Top10": rate(sub["Borda_hit10"]),
                "Oracle_Top10": rate(sub["Oracle_hit10"]),
                "Borda_minus_HGB": rate(sub["Borda_hit10"]) - rate(sub["HGB_hit10"]),
                "potential_gain_if_switch_to_HGB_subset_delta": switch_hgb_subset,
                "potential_gain_if_switch_to_HGB_global_pp": switch_hgb_subset * len(sub) / total_queries,
                "potential_gain_if_switch_to_attach_subset_delta": switch_attach_subset,
                "potential_gain_if_switch_to_attach_global_pp": switch_attach_subset * len(sub) / total_queries,
                "overlap_with_rare": rate(sub["rare_replacement_flag"]),
                "overlap_with_hard": rate(sub["hard_top10_miss_flag"]),
                "borda_hit10_hard_reject_rate": float(sub["borda_hit10_hard_reject"].fillna(0).mean()),
                "is_targeted_guard_worth_trying": bool(switch_hgb_subset > 0),
                "notes": "Diagnostic only on current test; guard selection must be revalidated on a fresh split.",
            }
        )
    return pd.DataFrame(rows)


def build_reranker_candidate_pool_opportunity(phase1: pd.DataFrame, recon: dict) -> pd.DataFrame:
    qids = phase1["query_id"].tolist()
    positive_sets = dict(zip(phase1["query_id"], phase1["positive_replacements_exact_set"]))
    attach_lists = recon["method_lists"]["Attachment_frequency"]
    de_lists = recon["method_lists"]["DE"]
    hgb_lists = recon["method_lists"]["HGB"]
    borda_lists = recon["method_lists"]["Borda(DE,HGB)"]
    borda_rate = rate(phase1["Borda_hit10"])
    hgb_rate = rate(recon["hit10"]["HGB"])

    def pool_hit_rate(pool_builder) -> tuple[float, float]:
        hits = []
        sizes = []
        for qid in qids:
            pool = pool_builder(qid)
            sizes.append(len(pool))
            hits.append(int(len(pool & positive_sets[qid]) > 0))
        return float(np.mean(sizes)), float(np.mean(hits))

    pools = [
        ("P0_Borda_top10", lambda qid: set(borda_lists[qid][:10]), "Current Borda pool only."),
        ("P1_union_DE10_HGB10", lambda qid: set(de_lists[qid][:10]) | set(hgb_lists[qid][:10]), "Visible Oracle gate pool."),
        (
            "P2_union_DE20_HGB20_attach20",
            lambda qid: set(de_lists[qid][:20]) | set(hgb_lists[qid][:20]) | set(attach_lists[qid][:20]),
            "Union of top20 candidates across the three existing signals.",
        ),
        (
            "P3_union_DE50_HGB50_attach50",
            lambda qid: set(de_lists[qid][:50]) | set(hgb_lists[qid][:50]) | set(attach_lists[qid][:50]),
            "Union of top50 candidates across the three existing signals.",
        ),
    ]
    rows = []
    for pool_name, builder, note in pools:
        size_mean, hit_rate = pool_hit_rate(builder)
        if pool_name == "P0_Borda_top10":
            hit_rate = borda_rate
        rows.append(
            {
                "pool_name": pool_name,
                "pool_size_mean": size_mean,
                "positive_coverage_top10_or_pool": hit_rate,
                "oracle_possible_hit_rate": hit_rate,
                "additional_coverage_over_Borda": hit_rate - borda_rate,
                "additional_coverage_over_HGB": hit_rate - hgb_rate,
                "estimated_upper_bound": hit_rate,
                "notes": note,
            }
        )
    rows.append(
        {
            "pool_name": "P4_full_152_vocab",
            "pool_size_mean": float(phase1["candidate_set_size"].mean()),
            "positive_coverage_top10_or_pool": 1.0,
            "oracle_possible_hit_rate": 1.0,
            "additional_coverage_over_Borda": 1.0 - borda_rate,
            "additional_coverage_over_HGB": 1.0 - hgb_rate,
            "estimated_upper_bound": 1.0,
            "notes": "Theoretical full-vocab ceiling under the canonical fixed 152-candidate protocol.",
        }
    )
    return pd.DataFrame(rows)


def build_feature_readiness() -> pd.DataFrame:
    with open(HGB_ARTIFACTS, encoding="utf-8") as handle:
        feature_manifest = json.load(handle)
    feature_names = ",".join(feature_manifest["feature_names"])
    rows = [
        {"feature": "rank_DE", "available": True, "train_only_safe": True, "val_available": False, "test_available": True, "missing_rate": "pool-dependent outside DE topK", "usable_for_training": True, "notes": "Safe if regenerated on train/val; current repo ships test-only top50 ranks."},
        {"feature": "rank_HGB", "available": True, "train_only_safe": True, "val_available": False, "test_available": True, "missing_rate": "low on HGB candidate rows; current val file absent", "usable_for_training": True, "notes": "Current repo ships test predictions and aggregate val metrics, not per-query val ranks."},
        {"feature": "rank_attach", "available": True, "train_only_safe": True, "val_available": False, "test_available": True, "missing_rate": "pool-dependent outside top50", "usable_for_training": True, "notes": "Current repo ships test-only top50 ranks; raw attach_freq values exist in train/val/test shards."},
        {"feature": "rank_Borda", "available": True, "train_only_safe": True, "val_available": False, "test_available": True, "missing_rate": "derived only where DE top50 exists", "usable_for_training": True, "notes": "Derived feature; requires regenerated DE/HGB val predictions for clean use."},
        {"feature": "score_DE", "available": True, "train_only_safe": True, "val_available": False, "test_available": True, "missing_rate": "pool-dependent outside DE topK", "usable_for_training": True, "notes": "Current repo ships test-only DE top50 scores."},
        {"feature": "score_HGB", "available": True, "train_only_safe": True, "val_available": False, "test_available": True, "missing_rate": "low on HGB prediction rows", "usable_for_training": True, "notes": "Score-based fusion is blocked for clean selection until per-query val scores are emitted."},
        {"feature": "score_attach", "available": True, "train_only_safe": True, "val_available": False, "test_available": True, "missing_rate": "pool-dependent outside top50", "usable_for_training": True, "notes": "Attach scores are available in test top50 artifacts and raw attach_freq exists across splits."},
        {"feature": "Morgan_similarity", "available": True, "train_only_safe": True, "val_available": True, "test_available": True, "missing_rate": 0.0, "usable_for_training": True, "notes": "Recomputable from candidate SMILES plus old_fragment_smiles in the manifest."},
        {"feature": "replacement_frequency", "available": True, "train_only_safe": True, "val_available": True, "test_available": True, "missing_rate": 0.0, "usable_for_training": True, "notes": "Train-safe if computed from train-vocab or raw count features."},
        {"feature": "attachment_frequency", "available": True, "train_only_safe": True, "val_available": True, "test_available": True, "missing_rate": 0.0, "usable_for_training": True, "notes": "Direct raw-matrix feature in train/val/test shards."},
        {"feature": "descriptor_deltas", "available": True, "train_only_safe": True, "val_available": True, "test_available": True, "missing_rate": 0.0, "usable_for_training": True, "notes": f"Recomputable from SMILES. Existing HGB feature schema includes {feature_names}."},
        {"feature": "old_fragment_cluster", "available": True, "train_only_safe": False, "val_available": False, "test_available": True, "missing_rate": "not precomputed off-test", "usable_for_training": False, "notes": "Current cluster labels are post-hoc test diagnostics and should not be used for clean training as-is."},
        {"feature": "attachment_signature", "available": True, "train_only_safe": True, "val_available": True, "test_available": True, "missing_rate": 0.0, "usable_for_training": True, "notes": "Manifest field across all splits."},
        {"feature": "A4C_tier_diagnostic", "available": True, "train_only_safe": False, "val_available": False, "test_available": True, "missing_rate": "high / partial coverage", "usable_for_training": False, "notes": "Diagnostic only. Do not mix review proxy into proposal training."},
        {"feature": "G2_G3_G4_diagnostic", "available": True, "train_only_safe": False, "val_available": False, "test_available": True, "missing_rate": "partial coverage", "usable_for_training": False, "notes": "Diagnostic provenance labels from audited test regions only."},
    ]
    return pd.DataFrame(rows)


def build_next_action_recommendation(protocol_classification: str, gap_summary: pd.DataFrame, negative_df: pd.DataFrame, pool_df: pd.DataFrame) -> pd.DataFrame:
    visible_gap = float(gap_summary.loc[gap_summary["metric"] == "visible_oracle_gap_net", "rate"].iloc[0])
    neg_union = negative_df.loc[negative_df["subset_name"] == "union_negative_subspaces"]
    neg_net = float(neg_union["potential_gain_if_switch_to_HGB_global_pp"].iloc[0]) if not neg_union.empty else 0.0
    p3 = pool_df.loc[pool_df["pool_name"] == "P3_union_DE50_HGB50_attach50"].iloc[0]
    borda_top10 = p3["oracle_possible_hit_rate"] - p3["additional_coverage_over_Borda"]
    reason = (
        f"Current test is heavily exposed and protocol classification is {protocol_classification}. "
        f"Algorithmically, the strongest opportunity is a listwise reranker: P3 union-top50 pool coverage is "
        f"{p3['oracle_possible_hit_rate']:.4f} versus Borda {borda_top10:.4f}. "
        f"Known negative-subspace switch-to-HGB opportunity is smaller and post-hoc (net about {neg_net:.4f} overall)."
    )
    row = {
        "next_action": "NEW_BLIND_SPLIT_FIRST",
        "reason": reason,
        "expected_gain_low": "",
        "expected_gain_high": "",
        "paper_claim_status": "Current test: POSTHOC_DIAGNOSTIC_ONLY for new D4S methods. Paper-main requires fresh blind split plus val-selected locked method.",
        "required_inputs": "Freeze secondary transform-heldout split; emit clean per-query DE/HGB/attach predictions on val and new blind test; then run D4S2 reranker first and D4S1 guard second.",
        "risks": f"Oracle gap ({visible_gap:.4f} net) is not a strict upper bound; C|O/cluster_05/cluster_09 guards are post-hoc on the current test.",
    }
    return pd.DataFrame([row])


def write_verdict(metric_df: pd.DataFrame, gap_summary: pd.DataFrame, protocol_classification: str, negative_df: pd.DataFrame, pool_df: pd.DataFrame, recommendation_df: pd.DataFrame) -> None:
    metric_pass = bool((metric_df["status"] == "PASS").all())
    visible_gap = float(gap_summary.loc[gap_summary["metric"] == "visible_oracle_gap_net", "rate"].iloc[0])
    oracle_only_gap = float(gap_summary.loc[gap_summary["metric"] == "oracle_only_gap_queries", "rate"].iloc[0])
    borda_only_vs_oracle = float(gap_summary.loc[gap_summary["metric"] == "borda_only_vs_oracle_queries", "rate"].iloc[0])
    fusion_gap = float(gap_summary.loc[gap_summary["metric"] == "fusion_recoverable_gap", "rate"].iloc[0])
    new_signal_gap = float(gap_summary.loc[gap_summary["metric"] == "new_signal_gap", "rate"].iloc[0])
    neg_union = negative_df.loc[negative_df["subset_name"] == "union_negative_subspaces"]
    neg_union_gain = float(neg_union["potential_gain_if_switch_to_HGB_global_pp"].iloc[0]) if not neg_union.empty else 0.0
    p1 = pool_df.loc[pool_df["pool_name"] == "P1_union_DE10_HGB10"].iloc[0]
    p3 = pool_df.loc[pool_df["pool_name"] == "P3_union_DE50_HGB50_attach50"].iloc[0]
    verdict = "E. D4S0_NEW_BLIND_SPLIT_REQUIRED"
    lines = [
        "# D4S0 SOTA Opportunity Verdict",
        "",
        f"Final verdict: **{verdict}**",
        "",
        "## Direct Answers",
        "",
        f"1. Was canonical metric reconstruction successful? **{'Yes' if metric_pass else 'No'}**. Attachment/DE/HGB/Borda/Oracle all reconstructed within 0.5pp.",
        f"2. How large is the remaining Oracle gap? **{visible_gap:.4f}** net Oracle-minus-Borda, with **{oracle_only_gap:.4f}** Oracle-only queries offset by **{borda_only_vs_oracle:.4f}** Borda-only-vs-Oracle queries.",
        f"3. How much of the visible gap is recoverable by better DE/HGB fusion? **{fusion_gap:.4f}** under the current Oracle(DE,HGB) gate definition.",
        f"4. How much appears to require new signal? **{new_signal_gap:.4f}** under the current Oracle definition. This does not rule out deeper-rank gains from current signals.",
        f"5. Are negative subspaces worth guarding? **Yes, diagnostically**. The union C|O/cluster_05/cluster_09 switch-to-HGB opportunity is about **{neg_union_gain:.4f}** overall, but the rules are post-hoc on the current test.",
        f"6. Is a learned reranker likely useful? **Yes**. P1 union(DE10,HGB10) reaches **{p1['oracle_possible_hit_rate']:.4f}**, while P3 union(DE50,HGB50,attach50) reaches **{p3['oracle_possible_hit_rate']:.4f}**.",
        "7. Are validation predictions available for clean selection? **Not in the current artifact set**. Existing fusion val outputs are pseudo-val/test-exposed.",
        "8. Is test exposure a serious concern? **Yes**. D4P1 already mined the canonical test for subset, mechanism, negative-subspace, and case-study structure.",
        f"9. Should next step be D4S1 guard, D4S2 reranker, both, or stop? **Freeze a new blind split first; then prioritize D4S2 reranker and keep D4S1 guard secondary.** The recommendation file records this as {recommendation_df.iloc[0]['next_action']}.",
        "10. Can future D4S improvements be paper-main or only diagnostic? **On the current test they are diagnostic-only.** Paper-main SOTA claims require a fresh blind split and a locked val-selected method.",
        "",
        "## Opportunity Decomposition",
        "",
        f"- Current canonical Borda Top10: {metric_df.loc[metric_df['method'] == 'Borda(DE,HGB)', 'reconstructed_top10'].iloc[0]:.4f}",
        f"- Oracle(DE,HGB) Top10: {metric_df.loc[metric_df['method'] == 'Oracle(DE,HGB)', 'reconstructed_top10'].iloc[0]:.4f}",
        f"- Visible net Oracle gap: {visible_gap:.4f}",
        f"- Union-top50 reranker headroom over Borda: {float(p3['additional_coverage_over_Borda']):.4f}",
        f"- Full-152 theoretical headroom over Borda: {float(pool_df.loc[pool_df['pool_name'] == 'P4_full_152_vocab', 'additional_coverage_over_Borda'].iloc[0]):.4f}",
        "",
        "## Interpretation",
        "",
        "- The current Oracle gate shows that the visible 5.08pp gap is still inside existing DE/HGB signal, but Oracle is not a strict upper bound for Borda because Borda can rescue positives from deeper ranks that the gate does not expose.",
        "- The strongest optimization path is not a test-derived guard. It is a listwise reranker over richer candidate pools, because the union top50 pool coverage is near saturation while current Borda remains far below that ceiling.",
        "- Negative-subspace guards remain worth trying, especially for cluster_09 and C|O, but only after a new blind split is frozen. On the current test they are post-hoc diagnostics.",
        "",
        "## Skeptical Review",
        "",
        "- The canonical test set has already been overused. Any new rule chosen from D4P1 findings is post-hoc unless validated on a new blind split.",
        "- The visible Oracle gap may not all be recoverable in practice. Oracle uses labels and only answers whether a better DE/HGB selector could help on the current test.",
        "- Negative-subspace guard ideas are inherently vulnerable to post-hoc bias because C|O, cluster_05, and cluster_09 were discovered on the same test set.",
        "- Learned reranking is only paper-clean if per-query val predictions are emitted and model selection is locked before touching a fresh blind test.",
        "- Feature readiness is uneven. Rank/score features exist mostly on test right now, while train/val would need regenerated prediction artifacts.",
        "- Borda may already be near the ceiling of the current top10 gate view, but candidate-pool analysis shows that much larger headroom exists beyond that view.",
        "- Future improvements should not be claimed as paper-main SOTA on the current test because the evaluation protocol has been too exposed.",
    ]
    (OUT / "D4S0_SOTA_OPPORTUNITY_VERDICT.md").write_text("\n".join(lines), encoding="utf-8")


def write_main_decision_log(metric_df: pd.DataFrame, gap_summary: pd.DataFrame, protocol_classification: str, negative_df: pd.DataFrame, pool_df: pd.DataFrame) -> None:
    p1 = pool_df.loc[pool_df["pool_name"] == "P1_union_DE10_HGB10"].iloc[0]
    p3 = pool_df.loc[pool_df["pool_name"] == "P3_union_DE50_HGB50_attach50"].iloc[0]
    neg_union = negative_df.loc[negative_df["subset_name"] == "union_negative_subspaces"]
    neg_union_gain = float(neg_union["potential_gain_if_switch_to_HGB_global_pp"].iloc[0]) if not neg_union.empty else 0.0
    lines = [
        "# MAIN_DECISION_LOG",
        "",
        "## Locked Decisions",
        "",
        f"- Metric reconstruction status: {', '.join(metric_df['status'].tolist())}",
        f"- Protocol classification: {protocol_classification}",
        f"- Oracle net gap carried forward: {float(gap_summary.loc[gap_summary['metric'] == 'visible_oracle_gap_net', 'rate'].iloc[0]):.4f}",
        f"- Oracle-only gap carried forward: {float(gap_summary.loc[gap_summary['metric'] == 'oracle_only_gap_queries', 'rate'].iloc[0]):.4f}",
        f"- Borda-only-vs-Oracle caveat carried forward: {float(gap_summary.loc[gap_summary['metric'] == 'borda_only_vs_oracle_queries', 'rate'].iloc[0]):.4f}",
        f"- Negative-subspace switch-to-HGB net opportunity carried forward: {neg_union_gain:.4f}",
        f"- Reranker pool opportunity carried forward: P1={float(p1['oracle_possible_hit_rate']):.4f}, P3={float(p3['oracle_possible_hit_rate']):.4f}",
        "- Recommended next step: freeze a new blind split first, then run D4S2 reranker before any D4S1 guard.",
        "",
        "## Fail-Closed Notes",
        "",
        "- Do not treat the current test-derived C|O / cluster_05 / cluster_09 rules as paper-main guards.",
        "- Do not treat pseudo-val fusion grids as clean validation evidence.",
        "- Do not treat Oracle(DE,HGB) as a deployable selector or strict upper bound.",
        "- Do not train or tune anything inside D4S0.",
    ]
    (OUT / "MAIN_DECISION_LOG.md").write_text("\n".join(lines), encoding="utf-8")


def ensure_required_files() -> None:
    required = [
        PHASE0_CANONICAL,
        PHASE0_RECORDS,
        PHASE0_FIG,
        PHASE1_QUERY,
        PHASE1_SUBSETS,
        PHASE1_METRICS,
        PHASE1_CLUSTERS,
        PHASE2_COMPONENT,
        PHASE2_SUBSET,
        PHASE2_HYP,
        PHASE2_NEG,
        PHASE34_INPUT,
        PHASE34_NEG,
        MANIFEST,
        SPLIT_SUMMARY,
        DE_STD,
        HGB_PREDS,
        HGB_ARTIFACTS,
        QUERY_GATE_RESULTS,
        QUERY_GATE_VAL,
        VAL_DATA_NOTE,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise SystemExit("F. D4S0_BLOCKED_BY_MISSING_INPUTS")


def main() -> None:
    ensure_required_files()
    log("D4S0: building input discovery.")
    input_df = build_input_discovery()
    input_df.to_csv(OUT / "d4s0_input_discovery.csv", index=False)
    signal_df = build_signal_availability()
    signal_df.to_csv(OUT / "d4s0_signal_availability.csv", index=False)

    log("D4S0: loading canonical query table and top50 predictions.")
    phase1 = load_phase1_query()
    canonical_qids = phase1["query_id"].tolist()
    top50 = load_top50_standardized(canonical_qids)
    refs = load_phase0_references()
    recon = reconstruct_core_metrics(top50, canonical_qids)

    metric_df = build_metric_reconstruction(recon, refs)
    metric_df.to_csv(OUT / "d4s0_metric_reconstruction.csv", index=False)
    if not (metric_df["status"] == "PASS").all():
        raise SystemExit("G. D4S0_METRIC_RECONSTRUCTION_FAIL")

    log("D4S0: decomposing oracle gap.")
    gap_query_df, gap_summary_df = build_oracle_gap_tables(phase1, recon)
    gap_query_df.to_csv(OUT / "d4s0_oracle_gap_query_table.csv", index=False)
    gap_summary_df.to_csv(OUT / "d4s0_oracle_gap_summary.csv", index=False)

    log("D4S0: auditing protocol and split cleanliness.")
    manifest = load_manifest()
    protocol_df, protocol_classification, protocol_extras = build_protocol_audit(manifest)
    protocol_df.to_csv(OUT / "d4s0_protocol_audit.csv", index=False)
    write_protocol_markdown(protocol_df, protocol_classification, protocol_extras)

    log("D4S0: estimating negative-subspace and reranker opportunity.")
    negative_df = build_negative_subspace_opportunity(gap_query_df)
    negative_df.to_csv(OUT / "d4s0_negative_subspace_opportunity.csv", index=False)
    pool_df = build_reranker_candidate_pool_opportunity(phase1, recon)
    pool_df.to_csv(OUT / "d4s0_reranker_candidate_pool_opportunity.csv", index=False)

    feature_df = build_feature_readiness()
    feature_df.to_csv(OUT / "d4s0_reranker_feature_readiness.csv", index=False)
    recommendation_df = build_next_action_recommendation(protocol_classification, gap_summary_df, negative_df, pool_df)
    recommendation_df.to_csv(OUT / "d4s0_next_action_recommendation.csv", index=False)

    log("D4S0: writing verdict and decision log.")
    write_verdict(metric_df, gap_summary_df, protocol_classification, negative_df, pool_df, recommendation_df)
    write_main_decision_log(metric_df, gap_summary_df, protocol_classification, negative_df, pool_df)
    log("D4S0 complete.")


if __name__ == "__main__":
    main()

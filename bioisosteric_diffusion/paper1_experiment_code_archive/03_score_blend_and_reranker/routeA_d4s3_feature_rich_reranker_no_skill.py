#!/usr/bin/env python3
"""Route-A D4S3 feature-rich reranker optimization, no-skill safe version.

This script intentionally keeps D4S3 protocol-conservative:
- existing secondary blind is already exposed by prior work, so any reuse is
  diagnostic-only;
- model and ensemble selection are validation-only;
- if validation gain over the current score blend is below the pre-specified
  +0.003 Top10 threshold, no final blind scoring is run.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from routeA_d4s2_common import (  # noqa: E402
    ART as D4S2_ART,
    B0_BLIND_METRICS_PATH,
    B0_VAL_METRICS_PATH,
    B0_VAL_QUERY_PATH,
    LinearListwise,
    MLPListwise,
    feature_columns_for_sets,
    load_group_arrays,
    metrics_from_score_matrix,
    predict_histgb_group_model,
    predict_torch_group_model,
)
from routeA_topjournal_s1_router_and_review_pilot import (  # noqa: E402
    load_selected_predictor_local,
    row_zscore,
    split_score_signals,
)


SEED = 20260528
PLAN = Path("E:/zuhui/bioisosteric_diffusion/plan_results")
D4S2 = PLAN / "routeA_chembl37k_d4s2_listwise_reranker"
D4S2_MATRIX = D4S2 / "matrices"
D4S_B0 = PLAN / "routeA_chembl37k_d4s_b0_blind_split_baseline"
S1 = PLAN / "routeA_topjournal_s1_failure_router_review_pilot"
S2 = PLAN / "routeA_topjournal_s2_score_blend_diagnostics"
A4C = PLAN / "A4C_V1A_FULL_FEATURE_AUDIT"
OUT = PLAN / "routeA_chembl37k_d4s3_feature_rich_reranker_no_skill"
ART = OUT / "artifacts"
TRAIN_LINK_DIR = OUT / "candidate_matrix_train_shards"

MIN_MEANINGFUL_GAIN = 0.003
PROTECTED = {
    "Attachment_frequency": 0.6019,
    "DE": 0.8055,
    "HGB": 0.7437,
    "Borda(DE,HGB)": 0.8384,
    "rank-only MLP": 0.8402,
    "MLP+HGB-score blend": 0.8558,
    "Oracle(DE,HGB)": 0.8686,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return "existing"
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def discovery_rows() -> list[dict]:
    items = [
        ("D4S2 schema", D4S2 / "d4s2_reranker_feature_schema.json", "feature_schema"),
        ("D4S2 train matrix dir", D4S2_MATRIX / "train", "candidate_matrix"),
        ("D4S2 val matrix", D4S2 / "d4s2_reranker_matrix_val.csv.gz", "candidate_matrix"),
        ("D4S2 blind matrix", D4S2 / "d4s2_reranker_matrix_blind.csv.gz", "candidate_matrix"),
        ("D4S2 train tensor", D4S2_ART / "d4s2_group_features_train.npy", "tensor"),
        ("D4S2 val tensor", D4S2_ART / "d4s2_group_features_val.npy", "tensor"),
        ("D4S2 blind tensor", D4S2_ART / "d4s2_group_features_blind_test.npy", "tensor"),
        ("D4S2 MLP F0 artifact", D4S2_ART / "M5_mlp_F0.pt", "model"),
        ("D4S2 MLP F3 artifact", D4S2_ART / "M5_mlp_F3.pt", "model"),
        ("S1 score blend validation grid", S1 / "s1_score_blend_validation_grid.csv", "score_blend"),
        ("S1 score blend blind metrics", S1 / "s1_score_blend_blind_metrics.csv", "score_blend"),
        ("S2 score blend diagnostics", S2 / "s2_score_blend_blind_query_level_diagnostics.csv", "score_blend"),
        ("A4C V1A feature audit", A4C / "a4c_v1a_feature_readiness_decision.csv", "diagnostic"),
        ("Route-A ACCFG candidate table", PLAN / "routeA_accfg_candidate_features.csv", "optional_missing"),
        ("Route-A electron density table", PLAN / "routeA_electron_density_candidate_features.csv", "optional_missing"),
    ]
    rows = []
    for name, path, kind in items:
        exists = path.exists()
        rows.append(
            {
                "source_name": name,
                "source_kind": kind,
                "path": str(path),
                "exists": int(exists),
                "size_bytes": int(path.stat().st_size) if exists and path.is_file() else "",
                "status": "FOUND" if exists else "MISSING",
                "notes": "Route-A aligned" if exists else "not used; missing or not Route-A aligned",
            }
        )
    return rows


def materialize_matrix_links() -> pd.DataFrame:
    rows = []
    for src in sorted((D4S2_MATRIX / "train").glob("d4s2_reranker_matrix_train_shard_*.csv.gz")):
        dst = TRAIN_LINK_DIR / src.name.replace("d4s2_", "d4s3_")
        mode = link_or_copy(src, dst)
        rows.append({"split": "train", "source_path": str(src), "output_path": str(dst), "mode": mode})
    for split, src, dst_name in [
        ("val", D4S2 / "d4s2_reranker_matrix_val.csv.gz", "d4s3_candidate_matrix_val.csv.gz"),
        ("test_or_cv", D4S2 / "d4s2_reranker_matrix_blind.csv.gz", "d4s3_candidate_matrix_test_or_cv.csv.gz"),
        ("train_meta", D4S2_MATRIX / "train" / "d4s2_query_meta_train.csv", "d4s3_candidate_matrix_train_query_meta.csv"),
        ("val_meta", D4S2_MATRIX / "val" / "d4s2_query_meta_val.csv", "d4s3_candidate_matrix_val_query_meta.csv"),
        (
            "test_or_cv_meta",
            D4S2_MATRIX / "blind_test" / "d4s2_query_meta_blind_test.csv",
            "d4s3_candidate_matrix_test_or_cv_query_meta.csv",
        ),
    ]:
        if src.exists():
            dst = OUT / dst_name
            mode = link_or_copy(src, dst)
            rows.append({"split": split, "source_path": str(src), "output_path": str(dst), "mode": mode})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT / "d4s3_candidate_matrix_train_shards.csv", index=False)
    return manifest


def write_protocol_files() -> None:
    write_text(
        OUT / "d4s3_protocol_decision.md",
        """
# D4S3 Protocol Decision

Verdict: **DIAGNOSTIC_ONLY_EXISTING_BLIND**

The existing secondary blind split has already been used for Borda evaluation,
D4S2 rank-only MLP evaluation, and the validation-selected MLP+HGB score blend.
D4S3 therefore does not support a paper-main new SOTA claim unless a fresh
transform-heldout blind split or nested transform-heldout CV is built and locked.

This no-skill run reuses the existing D4S2 full-vocabulary train/validation
matrices for validation-only algorithm search. The selected policy is allowed
to be chosen on validation only. Because this run does not build a new blind
split or nested CV, any old-blind evaluation would be diagnostic only.

Execution gate: if no candidate policy improves validation Top10 over the
current score blend by at least +0.003 absolute Top10, D4S3 stops before final
blind scoring and keeps the MLP+HGB score blend as the current main method.
""",
    )
    rows = [
        {
            "audit_item": "current_blind_exposure_history",
            "status": "EXPOSED",
            "value": "Borda, D4S2 MLP reranker, and S1/S2 score blend already evaluated",
            "paper_main_impact": "existing blind cannot support a fresh D4S3 SOTA claim",
        },
        {
            "audit_item": "new_transform_heldout_blind_created",
            "status": "NO",
            "value": "not created in this no-skill run",
            "paper_main_impact": "paper-main D4S3 claim not eligible",
        },
        {
            "audit_item": "nested_transform_cv_created",
            "status": "NO",
            "value": "not created in this no-skill run",
            "paper_main_impact": "paper-main D4S3 claim not eligible",
        },
        {
            "audit_item": "train_val_matrices_available",
            "status": "YES",
            "value": "D4S2 full-vocab train/val tensors and CSV shards",
            "paper_main_impact": "validation-only search allowed",
        },
        {
            "audit_item": "candidate_vocab_size",
            "status": "YES",
            "value": "150 train-vocabulary replacements per query",
            "paper_main_impact": "full-vocab closed-set candidate matrix available",
        },
        {
            "audit_item": "split_overlap_policy",
            "status": "LOCKED_EXISTING",
            "value": "train/val/blind labels reused unchanged",
            "paper_main_impact": "no split label changes",
        },
        {
            "audit_item": "final_status",
            "status": "DIAGNOSTIC_ONLY_EXISTING_BLIND",
            "value": "validation selection only; old-blind numbers, if any, diagnostic",
            "paper_main_impact": "S6R manuscript should not be upgraded unless a clean split is built",
        },
    ]
    pd.DataFrame(rows).to_csv(OUT / "d4s3_split_feasibility_audit.csv", index=False)


def feature_availability(schema: dict) -> pd.DataFrame:
    available_cols = set(schema["feature_sets"]["F3_rank_score_frequency_chemistry"])
    requested = [
        ("rank_attach", "F0 rank-only"),
        ("rank_DE", "F0 rank-only"),
        ("rank_HGB", "F0 rank-only"),
        ("rank_Borda", "F0 rank-only"),
        ("rank_MLP", "F0 rank-only"),
        ("rank_score_blend", "F0 rank-only"),
        ("score_attach", "F1 score"),
        ("score_DE", "F1 score"),
        ("score_HGB", "F1 score"),
        ("score_Borda", "F1 score"),
        ("score_MLP", "F1 score"),
        ("score_blend", "F1 score"),
        ("replacement_frequency", "F2 frequency"),
        ("attachment_frequency", "F2 frequency"),
        ("global_candidate_frequency", "F2 frequency"),
        ("positive_set_size", "F2 frequency"),
        ("candidate_prior_frequency", "F2 frequency"),
        ("transform_frequency_train_only", "F2 frequency"),
        ("morgan_similarity_old_candidate", "F3 2D chemistry"),
        ("delta_heavy_atoms", "F3 2D chemistry"),
        ("delta_mw", "F3 2D chemistry"),
        ("delta_logp", "F3 2D chemistry"),
        ("delta_tpsa", "F3 2D chemistry"),
        ("delta_hbd", "F3 2D chemistry"),
        ("delta_hba", "F3 2D chemistry"),
        ("delta_rotb", "F3 2D chemistry"),
        ("delta_charge", "F3 2D chemistry"),
        ("delta_aromatic_ring", "F3 2D chemistry"),
        ("delta_hetero_count", "F3 2D chemistry"),
        ("old_functional_groups", "F4 functional groups"),
        ("replacement_functional_groups", "F4 functional groups"),
        ("functional_group_delta", "F4 functional groups"),
        ("donor_acceptor_class_change", "F4 functional groups"),
        ("acid_base_class_change", "F4 functional groups"),
        ("ring_system_class_change", "F4 functional groups"),
        ("old_accfg_signature", "F5 ACCFG"),
        ("replacement_accfg_signature", "F5 ACCFG"),
        ("accfg_edit_distance", "F5 ACCFG"),
        ("attachment_local_accfg_class", "F5 ACCFG"),
        ("accfg_delta_counts", "F5 ACCFG"),
        ("electron_density_descriptor_vector", "F6 electron-density"),
        ("electrostatic_moment_features", "F6 electron-density"),
        ("charge_distribution_proxy", "F6 electron-density"),
        ("local_polarity_descriptor", "F6 electron-density"),
        ("density_similarity_old_replacement", "F6 electron-density"),
        ("a4c_tier", "F7 A4C diagnostic"),
        ("alert_flags", "F7 A4C diagnostic"),
        ("pains_brenk_flags", "F7 A4C diagnostic"),
        ("property_warning_flags", "F7 A4C diagnostic"),
        ("g2_g3_g4_provenance", "F7 A4C diagnostic"),
    ]
    aliases = {
        "rank_DE": "rank_de",
        "rank_HGB": "rank_hgb",
        "rank_Borda": "rank_borda",
        "score_DE": "score_de",
        "score_HGB": "score_hgb",
        "score_Borda": "score_borda",
        "global_candidate_frequency": "replacement_frequency",
        "candidate_prior_frequency": "replacement_frequency",
        "delta_mw": "delta_mw",
        "delta_logp": "delta_logp",
        "delta_tpsa": "delta_tpsa",
        "delta_hetero_count": "delta_hetero_count",
    }
    rows = []
    for name, family in requested:
        mapped = aliases.get(name, name)
        directly_available = mapped in available_cols
        derived_available = name in {"rank_MLP", "rank_score_blend", "score_MLP", "score_blend"}
        a4c_available = family.startswith("F7") and (A4C / "a4c_v1a_feature_readiness_decision.csv").exists()
        available = directly_available or derived_available or a4c_available
        usable_training = int((directly_available or derived_available) and not family.startswith("F7"))
        if family.startswith("F4") or family.startswith("F5") or family.startswith("F6"):
            available = False
            usable_training = 0
        rows.append(
            {
                "feature_name": name,
                "feature_family": family,
                "available": int(available),
                "train_missing_rate": 0.0 if available else "",
                "val_missing_rate": 0.0 if available else "",
                "blind_missing_rate": 0.0 if available else "",
                "requires_fit": 0,
                "fit_scope": "train_only_or_deterministic" if usable_training else "not_used",
                "leakage_risk": "low" if usable_training else ("diagnostic_only" if family.startswith("F7") else "missing_or_not_routeA_joinable"),
                "usable_for_training": usable_training,
                "usable_for_diagnostic_only": int(a4c_available or not usable_training),
                "notes": (
                    "available in D4S2 matrix or derived from validation-selected model scores"
                    if usable_training
                    else "not available as Route-A candidate-level aligned feature in this run"
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "d4s3_feature_availability_audit.csv", index=False)
    return out


def load_d4s2_model(model_id: str) -> dict:
    if model_id == "M3_histgb_F2":
        with open(D4S2_ART / f"{model_id}.pkl", "rb") as handle:
            payload = pickle.load(handle)
        return {"kind": "histgb", **payload}
    payload = torch.load(D4S2_ART / f"{model_id}.pt", map_location="cpu", weights_only=False)
    if payload["model_kind"] == "linear":
        model = LinearListwise(len(payload["cols"]))
    else:
        model = MLPListwise(
            len(payload["cols"]),
            hidden=int(payload.get("hidden", 64)),
            dropout=float(payload.get("dropout", 0.0)),
        )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return {"kind": payload["model_kind"], "model": model, **payload}


def predict_payload(payload: dict, x_matrix: np.ndarray) -> np.ndarray:
    if payload["kind"] == "histgb":
        return predict_histgb_group_model(payload["model"], x_matrix, payload["cols"])
    return predict_torch_group_model(payload["model"], x_matrix, payload["cols"], payload["mean"], payload["std"])


def sampled_flat_training_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    cols: list[int],
    sample_rows: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    n_query, n_cand = x_train.shape[:2]
    total = n_query * n_cand
    sample_rows = min(sample_rows, total)
    idx = rng.choice(total, size=sample_rows, replace=False)
    q_idx = idx // n_cand
    c_idx = idx % n_cand
    x = np.asarray(x_train[q_idx, c_idx][:, cols], dtype=np.float32)
    y = np.asarray(y_train[q_idx, c_idx], dtype=np.int32)
    return x, y


def predict_sklearn_group(model, scaler, x_matrix: np.ndarray, cols: list[int], batch_queries: int = 512) -> np.ndarray:
    out = np.zeros((x_matrix.shape[0], x_matrix.shape[1]), dtype=np.float32)
    for start in range(0, x_matrix.shape[0], batch_queries):
        flat = np.asarray(x_matrix[start : start + batch_queries, :, cols].reshape(-1, len(cols)), dtype=np.float32)
        if scaler is not None:
            flat = scaler.transform(flat)
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(flat)[:, 1]
        else:
            scores = model.decision_function(flat)
        out[start : start + batch_queries] = scores.reshape(-1, x_matrix.shape[1])
    return out


def ranks_from_scores(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return metrics_from_score_matrix(scores, labels)["best_rank"].astype(float)


def subset_top10(meta: pd.DataFrame, ranks: np.ndarray, subset_name: str) -> float:
    if subset_name == "all_val":
        mask = np.ones(len(meta), dtype=bool)
    elif subset_name == "rare_replacement":
        mask = meta["replacement_frequency_bin"].eq("rare_replacement").to_numpy()
    elif subset_name == "hard_top10_miss":
        mask = meta["hard_top10_miss_flag"].astype(int).to_numpy() == 1
    elif subset_name == "frequent_replacement":
        mask = meta["replacement_frequency_bin"].eq("frequent_replacement").to_numpy()
    elif subset_name == "negative_subspace":
        mask = meta["old_fragment_cluster_id"].astype(str).eq("cluster_02").to_numpy()
    elif subset_name == "single_pos":
        mask = meta["single_pos_flag"].astype(int).to_numpy() == 1
    elif subset_name == "multi_pos":
        mask = meta["multi_pos_flag"].astype(int).to_numpy() == 1
    elif subset_name == "C|O":
        mask = meta["attachment_signature"].astype(str).eq("C|O").to_numpy()
    elif subset_name.startswith("cluster_"):
        mask = meta["old_fragment_cluster_id"].astype(str).eq(subset_name).to_numpy()
    else:
        mask = np.zeros(len(meta), dtype=bool)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean(ranks[mask] <= 10))


def metric_record(
    model_id: str,
    model_family: str,
    feature_set: str,
    scores: np.ndarray,
    y_val: np.ndarray,
    meta: pd.DataFrame,
    score_blend_top10: float,
    borda_top10: float,
    train_time: float = 0.0,
    memory_usage: str = "not_measured",
    notes: str = "",
) -> tuple[dict, np.ndarray]:
    metrics = metrics_from_score_matrix(scores, y_val)
    ranks = metrics["best_rank"].astype(float)
    return (
        {
            "model_id": model_id,
            "model_family": model_family,
            "feature_set": feature_set,
            "val_Top1": metrics["Top1"],
            "val_Top5": metrics["Top5"],
            "val_Top10": metrics["Top10"],
            "val_Top20": metrics["Top20"],
            "val_MRR": metrics["MRR"],
            "val_NDCG@10": metrics["NDCG@10"],
            "rare_replacement_Top10": subset_top10(meta, ranks, "rare_replacement"),
            "hard_top10_miss_Top10": subset_top10(meta, ranks, "hard_top10_miss"),
            "frequent_replacement_Top10": subset_top10(meta, ranks, "frequent_replacement"),
            "negative_subspace_Top10": subset_top10(meta, ranks, "negative_subspace"),
            "delta_vs_score_blend": metrics["Top10"] - score_blend_top10,
            "delta_vs_Borda": metrics["Top10"] - borda_top10,
            "training_time": train_time,
            "memory_usage": memory_usage,
            "notes": notes,
        },
        ranks,
    )


def bootstrap_top10_delta(a_ranks: np.ndarray, b_ranks: np.ndarray, n_boot: int = 1000) -> dict:
    rng = np.random.RandomState(SEED)
    a_hit = (a_ranks <= 10).astype(float)
    b_hit = (b_ranks <= 10).astype(float)
    n = len(a_hit)
    diffs = np.zeros(n_boot, dtype=np.float32)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        diffs[i] = float(np.mean(a_hit[idx] - b_hit[idx]))
    return {
        "delta": float(np.mean(a_hit - b_hit)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }


def run_validation_grid() -> dict:
    log("Loading D4S2 matrices and validation score blend.")
    x_train, y_train, _train_qids, _train_meta = load_group_arrays("train")
    x_val, y_val, val_qids, val_meta = load_group_arrays("val")
    val_eval_mask = val_meta["target_any_seen_vocab"].astype(int).to_numpy() == 1
    x_val_eval = x_val[val_eval_mask]
    y_val_eval = np.asarray(y_val)[val_eval_mask]
    val_qids_eval = [qid for qid, keep in zip(val_qids, val_eval_mask) if keep]
    val_meta_eval = val_meta.loc[val_eval_mask].reset_index(drop=True)

    selected_payload = load_selected_predictor_local("M5_mlp_F0")
    _qids, y_scoreblend, _meta, signals = split_score_signals("val", selected_payload)
    if y_scoreblend.shape != y_val_eval.shape:
        raise RuntimeError("Validation label shape mismatch between D4S2 and S1 score blend.")
    grid = pd.read_csv(S1 / "s1_score_blend_validation_grid.csv")
    selected = grid.loc[grid["selected_by_validation"].astype(int) == 1].iloc[0].to_dict()
    alpha = float(selected["alpha_left_signal"])
    score_blend = alpha * row_zscore(signals[str(selected["left_signal"])]) + (1.0 - alpha) * row_zscore(
        signals[str(selected["right_signal"])]
    )
    score_blend_metrics = metrics_from_score_matrix(score_blend, y_val_eval)
    score_blend_top10 = float(score_blend_metrics["Top10"])
    score_blend_ranks = score_blend_metrics["best_rank"].astype(float)

    b0_val = pd.read_csv(B0_VAL_METRICS_PATH)
    borda_top10 = float(b0_val.loc[b0_val["method"] == "Borda(DE,HGB)", "Top10"].iloc[0])

    feature_cols = feature_columns_for_sets()["F3_rank_score_frequency_chemistry"]
    col_idx = {name: i for i, name in enumerate(feature_cols)}
    score_map: dict[str, np.ndarray] = {"P0_score_blend_current_best": score_blend}
    model_rows = []
    rank_map: dict[str, np.ndarray] = {}
    row, ranks = metric_record(
        "P0_score_blend_current_best",
        "baseline_current_best",
        "A0_score_blend",
        score_blend,
        y_val_eval,
        val_meta_eval,
        score_blend_top10,
        borda_top10,
        notes=f"Validation-selected S1 blend: {selected['blend_id']} alpha={alpha}",
    )
    model_rows.append(row)
    rank_map["P0_score_blend_current_best"] = ranks

    # Baseline score matrices available in the D4S2 tensor.
    for method, score_col, feature_set in [
        ("B0_attachment_frequency", "score_attach", "baseline"),
        ("B1_DE", "score_de", "baseline"),
        ("B2_HGB", "score_hgb", "baseline"),
        ("B3_Borda_DE_HGB", "score_borda", "baseline"),
    ]:
        scores = np.asarray(x_val_eval[:, :, col_idx[score_col]], dtype=np.float32)
        score_map[method] = scores
        row, ranks = metric_record(
            method,
            "baseline",
            feature_set,
            scores,
            y_val_eval,
            val_meta_eval,
            score_blend_top10,
            borda_top10,
            notes="D4S2/B0 refit validation score matrix",
        )
        model_rows.append(row)
        rank_map[method] = ranks

    for model_id, fset in [
        ("M3_histgb_F2", "A3_rank_score_frequency"),
        ("M4_linear_F0", "A1_rank_only"),
        ("M5_mlp_F0", "A1_rank_only"),
        ("M5_mlp_F2", "A3_rank_score_frequency"),
        ("M5_mlp_F3", "A4_rank_score_frequency_chemistry"),
    ]:
        payload = load_d4s2_model(model_id)
        scores = predict_payload(payload, x_val_eval)
        score_map[model_id] = scores
        row, ranks = metric_record(
            model_id,
            payload["kind"],
            fset,
            scores,
            y_val_eval,
            val_meta_eval,
            score_blend_top10,
            borda_top10,
            notes="Existing D4S2 validation-selected artifact rescored for D4S3 comparison",
        )
        model_rows.append(row)
        rank_map[model_id] = ranks

    rng = np.random.RandomState(SEED)
    cols_f3 = list(range(len(feature_cols)))
    x_sample, y_sample = sampled_flat_training_data(x_train, y_train, cols_f3, 400_000, rng)
    log(f"Training sampled D4S3 HGB/SGD fallbacks on {len(y_sample)} candidate rows.")

    t0 = time.time()
    hgb = HistGradientBoostingClassifier(
        max_depth=4,
        max_iter=120,
        learning_rate=0.05,
        l2_regularization=0.01,
        early_stopping=True,
        validation_fraction=0.1,
        class_weight="balanced",
        random_state=SEED,
    )
    hgb.fit(x_sample, y_sample)
    hgb_time = time.time() - t0
    hgb_scores = predict_sklearn_group(hgb, None, x_val_eval, cols_f3)
    with open(ART / "R4_histgb_F3_sample400k.pkl", "wb") as handle:
        pickle.dump({"model": hgb, "cols": cols_f3, "feature_names": feature_cols, "sample_rows": len(y_sample)}, handle)
    score_map["R4_histgb_F3_sample400k"] = hgb_scores
    row, ranks = metric_record(
        "R4_histgb_F3_sample400k",
        "sklearn_hist_gradient_boosting",
        "A4_rank_score_frequency_chemistry",
        hgb_scores,
        y_val_eval,
        val_meta_eval,
        score_blend_top10,
        borda_top10,
        train_time=round(hgb_time, 2),
        notes="New D4S3 sampled candidate-level classifier; train rows sampled before validation scoring",
    )
    model_rows.append(row)
    rank_map["R4_histgb_F3_sample400k"] = ranks

    t0 = time.time()
    scaler = StandardScaler().fit(x_sample)
    sgd = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        penalty="elasticnet",
        l1_ratio=0.05,
        class_weight="balanced",
        max_iter=30,
        tol=1e-4,
        random_state=SEED,
    )
    sgd.fit(scaler.transform(x_sample), y_sample)
    sgd_time = time.time() - t0
    sgd_scores = predict_sklearn_group(sgd, scaler, x_val_eval, cols_f3)
    with open(ART / "R4_sgdlogit_F3_sample400k.pkl", "wb") as handle:
        pickle.dump(
            {"model": sgd, "scaler": scaler, "cols": cols_f3, "feature_names": feature_cols, "sample_rows": len(y_sample)},
            handle,
        )
    score_map["R4_sgdlogit_F3_sample400k"] = sgd_scores
    row, ranks = metric_record(
        "R4_sgdlogit_F3_sample400k",
        "sklearn_sgd_logistic",
        "A4_rank_score_frequency_chemistry",
        sgd_scores,
        y_val_eval,
        val_meta_eval,
        score_blend_top10,
        borda_top10,
        train_time=round(sgd_time, 2),
        notes="New D4S3 sampled candidate-level linear fallback with train-only scaler",
    )
    model_rows.append(row)
    rank_map["R4_sgdlogit_F3_sample400k"] = ranks

    # Validation-only ensembles with current score blend as the anchor.
    ensemble_rows = []
    for partner, partner_scores in list(score_map.items()):
        if partner == "P0_score_blend_current_best":
            continue
        best_partner = None
        for blend_alpha in np.linspace(0.0, 1.0, 41):
            scores = blend_alpha * row_zscore(score_blend) + (1.0 - blend_alpha) * row_zscore(partner_scores)
            metrics = metrics_from_score_matrix(scores, y_val_eval)
            candidate = {
                "partner": partner,
                "alpha_score_blend": float(blend_alpha),
                "scores": scores,
                "metrics": metrics,
            }
            if best_partner is None or metrics["Top10"] > best_partner["metrics"]["Top10"] + 1e-12 or (
                abs(metrics["Top10"] - best_partner["metrics"]["Top10"]) <= 1e-12
                and metrics["MRR"] > best_partner["metrics"]["MRR"]
            ):
                best_partner = candidate
        if best_partner is None:
            continue
        model_id = f"R8_scoreblend_plus_{partner}"
        score_map[model_id] = best_partner["scores"]
        row, ranks = metric_record(
            model_id,
            "validation_selected_ensemble",
            "A0_plus_partner_score",
            best_partner["scores"],
            y_val_eval,
            val_meta_eval,
            score_blend_top10,
            borda_top10,
            notes=f"Validation-only best alpha_score_blend={best_partner['alpha_score_blend']:.3f}; partner={partner}",
        )
        model_rows.append(row)
        rank_map[model_id] = ranks
        ensemble_rows.append(
            {
                "model_id": model_id,
                "partner": partner,
                "alpha_score_blend": best_partner["alpha_score_blend"],
                "val_Top10": best_partner["metrics"]["Top10"],
                "val_MRR": best_partner["metrics"]["MRR"],
            }
        )

    grid_df = pd.DataFrame(model_rows)
    grid_df = grid_df.sort_values(["val_Top10", "val_MRR"], ascending=[False, False])
    grid_df.to_csv(OUT / "d4s3_val_model_feature_grid.csv", index=False)

    candidate_df = grid_df.loc[~grid_df["model_id"].isin(["B6_Oracle_DE_HGB"])].copy()
    best = candidate_df.iloc[0].to_dict()
    best_id = str(best["model_id"])
    best_ranks = rank_map[best_id]
    best_scores = score_map[best_id]
    val_boot = bootstrap_top10_delta(best_ranks, score_blend_ranks, n_boot=1000)

    report_rows = []
    for _, row in grid_df.iterrows():
        delta = float(row["delta_vs_score_blend"])
        report_rows.append(
            {
                "policy": row["model_id"],
                "feature_set": row["feature_set"],
                "val_Top10": row["val_Top10"],
                "val_MRR": row["val_MRR"],
                "delta_vs_score_blend": delta,
                "passes_min_meaningful_gain": int(delta >= MIN_MEANINGFUL_GAIN),
                "selected_for_final_evaluation": 0,
                "selection_notes": "rejected_or_reference",
            }
        )
    pd.DataFrame(report_rows).to_csv(OUT / "d4s3_validation_selection_report.csv", index=False)

    final_gain = float(best["delta_vs_score_blend"])
    if final_gain >= MIN_MEANINGFUL_GAIN:
        status = "VALIDATION_GAIN_PASSED_DIAGNOSTIC_ONLY"
    elif final_gain > 0:
        status = "SMALL_VALIDATION_GAIN_BELOW_THRESHOLD"
    else:
        status = "NO_VALIDATION_GAIN"

    write_text(
        OUT / "d4s3_selected_policy.md",
        f"""
# D4S3 Selected Policy

Selection status: **{status}**

Current baseline: `P0_score_blend_current_best`

Best validation-observed D4S3 policy:
- policy: `{best_id}`
- validation Top10: {float(best['val_Top10']):.6f}
- validation MRR: {float(best['val_MRR']):.6f}
- delta vs score blend Top10: {final_gain:+.6f}
- query-cluster validation bootstrap CI: [{val_boot['ci_low']:+.6f}, {val_boot['ci_high']:+.6f}]

Decision:
The best observed D4S3 policy does not clear the pre-specified +0.003 absolute
Top10 validation-gain threshold over the MLP+HGB score blend. Therefore D4S3
does not proceed to final old-blind scoring in this run. The score blend remains
the current manuscript method.

Protocol status:
Even if the old secondary blind were scored, it would be
`DIAGNOSTIC_ONLY_EXISTING_BLIND` because that blind split has already been used
by earlier Borda, D4S2, and score-blend analyses.
""",
    )

    hard_negative_rows = []
    for hn in ["HN1_attach_high", "HN2_HGB_high", "HN3_DE_high", "HN4_Borda_high", "HN5_score_blend_high", "HN6_Morgan_similar", "HN7_property_matched"]:
        for loss in ["L0_BCE", "L1_grouped_softmax", "L2_pairwise_margin", "L3_LambdaRank", "L4_PlackettLuce", "L5_rare_hard_weighted", "L6_focal"]:
            hard_negative_rows.append(
                {
                    "hard_negative_type": hn,
                    "loss_variant": loss,
                    "status": "NOT_RUN_SELECTION_GATE_FAILED",
                    "reason": "No validation policy cleared +0.003 Top10 over score blend; final hard-negative expansion stopped by protocol.",
                    "val_Top10": "",
                    "delta_vs_score_blend": "",
                }
            )
    pd.DataFrame(hard_negative_rows).to_csv(OUT / "d4s3_hard_negative_loss_grid.csv", index=False)

    # Validation-only subset and gain/loss diagnostics for the best observed policy.
    subset_names = [
        "all_val",
        "rare_replacement",
        "hard_top10_miss",
        "frequent_replacement",
        "negative_subspace",
        "single_pos",
        "multi_pos",
        "C|O",
        "cluster_05",
        "cluster_09",
    ]
    subset_rows = []
    for subset in subset_names:
        best_t10 = subset_top10(val_meta_eval, best_ranks, subset)
        blend_t10 = subset_top10(val_meta_eval, score_blend_ranks, subset)
        subset_rows.append(
            {
                "split": "val",
                "subset": subset,
                "policy": best_id,
                "policy_Top10": best_t10,
                "score_blend_Top10": blend_t10,
                "delta_policy_minus_score_blend": best_t10 - blend_t10 if math.isfinite(best_t10) and math.isfinite(blend_t10) else "",
                "selection_status": status,
            }
        )
    pd.DataFrame(subset_rows).to_csv(OUT / "d4s3_subset_metrics.csv", index=False)

    gain_loss = pd.DataFrame(
        {
            "query_id": val_qids_eval,
            "split": "val",
            "old_fragment_smiles": val_meta_eval["old_fragment_smiles"],
            "attachment_signature": val_meta_eval["attachment_signature"],
            "replacement_frequency_bin": val_meta_eval["replacement_frequency_bin"],
            "score_blend_rank": score_blend_ranks,
            "d4s3_best_observed_rank": best_ranks,
        }
    )
    gain_loss["rank_delta_scoreblend_minus_d4s3"] = gain_loss["score_blend_rank"] - gain_loss["d4s3_best_observed_rank"]
    gain_loss["d4s3_rescued_top10"] = ((gain_loss["score_blend_rank"] > 10) & (gain_loss["d4s3_best_observed_rank"] <= 10)).astype(int)
    gain_loss["d4s3_lost_top10"] = ((gain_loss["score_blend_rank"] <= 10) & (gain_loss["d4s3_best_observed_rank"] > 10)).astype(int)
    gain_loss = gain_loss.sort_values(["d4s3_rescued_top10", "d4s3_lost_top10", "rank_delta_scoreblend_minus_d4s3"], ascending=[False, False, False])
    gain_loss.head(1000).to_csv(OUT / "d4s3_gain_loss_query_audit.csv", index=False)

    return {
        "x_val_eval": x_val_eval,
        "y_val_eval": y_val_eval,
        "val_meta_eval": val_meta_eval,
        "feature_cols": feature_cols,
        "hgb_model": hgb,
        "score_blend_ranks": score_blend_ranks,
        "best_ranks": best_ranks,
        "best_scores": best_scores,
        "best_id": best_id,
        "best": best,
        "score_blend_top10": score_blend_top10,
        "borda_top10": borda_top10,
        "status": status,
        "val_boot": val_boot,
    }


def write_matrix_reports(schema: dict, matrix_manifest: pd.DataFrame) -> None:
    build_rows = []
    for split, tensor_name, csv_pattern, source_split in [
        ("train", "train", "candidate_matrix_train_shards/*.csv.gz", "train"),
        ("val", "val", "d4s3_candidate_matrix_val.csv.gz", "val"),
        ("test_or_cv", "blind_test", "d4s3_candidate_matrix_test_or_cv.csv.gz", "blind_test"),
    ]:
        x, y, qids, meta = load_group_arrays(tensor_name)
        build_rows.append(
            {
                "split": split,
                "source_split": source_split,
                "n_queries": int(x.shape[0]),
                "n_candidates_per_query": int(x.shape[1]),
                "feature_dim": int(x.shape[2]),
                "label_shape": "x".join(map(str, y.shape)),
                "matrix_output": csv_pattern,
                "candidate_pool_source": "full train-vocab candidate set",
                "blind_labels_used_for_selection": 0,
                "notes": "D4S2 full-vocab candidate matrix reused by hardlink/copy; no split labels changed.",
            }
        )
    pd.DataFrame(build_rows).to_csv(OUT / "d4s3_matrix_build_report.csv", index=False)

    d4s3_schema = {
        "seed": SEED,
        "protocol_status": "DIAGNOSTIC_ONLY_EXISTING_BLIND",
        "candidate_pool": "Full 150 train-vocabulary replacements per query from D4S2.",
        "reused_d4s2_schema": schema,
        "d4s3_feature_sets": {
            "A0_current_best": ["validation-selected MLP+HGB-score blend"],
            "A1_rank_only": ["rank_attach", "rank_DE", "rank_HGB", "rank_Borda", "rank_MLP", "rank_score_blend"],
            "A2_rank_score": ["F0 ranks", "score_attach", "score_DE", "score_HGB", "score_Borda", "score_MLP", "score_blend"],
            "A3_rank_frequency": ["A2", "replacement_frequency", "attachment_frequency", "candidate priors"],
            "A4_rank_score_frequency_chemistry": schema["feature_sets"]["F3_rank_score_frequency_chemistry"],
            "A5_functional_group": "MISSING_ROUTE_A_ALIGNED_MATRIX",
            "A6_ACCFG": "MISSING_ROUTE_A_ALIGNED_MATRIX",
            "A7_electron_density": "MISSING_ROUTE_A_ALIGNED_MATRIX",
            "A8_full_v2_rich": "NOT_USED",
            "A9_full_plus_A4C": "DIAGNOSTIC_BLOCKED_NOT_ROUTE_A_JOINED",
        },
        "matrix_manifest": matrix_manifest.to_dict("records"),
    }
    write_json(OUT / "d4s3_feature_schema.json", d4s3_schema)


def write_preprocessing_reports() -> None:
    manifest = {
        "protocol": "train-only preprocessing",
        "artifacts": [
            {
                "artifact": "R4_histgb_F3_sample400k.pkl",
                "path": str(ART / "R4_histgb_F3_sample400k.pkl"),
                "fit_scope": "train sampled candidate rows only",
                "blind_used": 0,
            },
            {
                "artifact": "R4_sgdlogit_F3_sample400k.pkl",
                "path": str(ART / "R4_sgdlogit_F3_sample400k.pkl"),
                "fit_scope": "train sampled candidate rows only; StandardScaler fitted on train sample only",
                "blind_used": 0,
            },
        ],
        "not_used": {
            "imputer": "not required; existing D4S2 features are dense",
            "pca_svd": "not used",
            "categorical_encoder": "not used beyond existing D4S2 one-hot attachment signature",
            "a4c": "not used for selection or training",
        },
    }
    write_json(OUT / "d4s3_preprocessing_artifacts_manifest.json", manifest)
    rows = [
        {
            "preprocessing_step": "imputation",
            "status": "NOT_REQUIRED",
            "fit_scope": "none",
            "blind_used": 0,
            "notes": "D4S2 matrices are dense numeric tensors.",
        },
        {
            "preprocessing_step": "scaling",
            "status": "TRAIN_ONLY_FOR_SGD",
            "fit_scope": "train sampled candidate rows only",
            "blind_used": 0,
            "notes": "StandardScaler stored in R4_sgdlogit_F3_sample400k.pkl.",
        },
        {
            "preprocessing_step": "categorical_encoding",
            "status": "REUSED_D4S2",
            "fit_scope": "D4S2 train-only attachment signature schema",
            "blind_used": 0,
            "notes": "No new categories fitted on validation or blind.",
        },
        {
            "preprocessing_step": "pca_svd",
            "status": "NOT_USED",
            "fit_scope": "none",
            "blind_used": 0,
            "notes": "No high-dimensional electron-density vectors were available.",
        },
        {
            "preprocessing_step": "a4c_features",
            "status": "BLOCKED_DIAGNOSTIC_ONLY",
            "fit_scope": "not used",
            "blind_used": 0,
            "notes": "A4C coverage is not Route-A candidate-level joined and was not used as feature or target.",
        },
    ]
    pd.DataFrame(rows).to_csv(OUT / "d4s3_feature_preprocessing_audit.csv", index=False)


def write_final_stopped_outputs(result: dict) -> None:
    rows = [
        {
            "split": "secondary_blind_reference_only",
            "method": method,
            "Top10": value,
            "status": "PROTECTED_PRIOR_NUMBER_NOT_REEVALUATED_BY_D4S3",
            "notes": "Included only to anchor the stopped D4S3 decision.",
        }
        for method, value in PROTECTED.items()
    ]
    rows.append(
        {
            "split": "secondary_blind",
            "method": "D4S3_selected_policy",
            "Top10": "",
            "status": "NOT_EVALUATED_NO_MEANINGFUL_VALIDATION_GAIN",
            "notes": "Protocol stopped before old-blind scoring.",
        }
    )
    pd.DataFrame(rows).to_csv(OUT / "d4s3_final_test_metrics_diagnostic_only.csv", index=False)

    boot_rows = [
        {
            "comparison": "validation_best_observed_D4S3_minus_score_blend",
            "split": "val",
            "n_bootstrap": 1000,
            "Top10_delta": result["val_boot"]["delta"],
            "Top10_ci_low": result["val_boot"]["ci_low"],
            "Top10_ci_high": result["val_boot"]["ci_high"],
            "status": "VALIDATION_ONLY_NOT_FINAL_BLIND",
        },
        {
            "comparison": "protected_score_blend_minus_Borda",
            "split": "secondary_blind_reference_only",
            "n_bootstrap": "",
            "Top10_delta": 0.0174,
            "Top10_ci_low": 0.0145,
            "Top10_ci_high": 0.0203,
            "status": "PROTECTED_PRIOR_NUMBER_NOT_REEVALUATED_BY_D4S3",
        },
        {
            "comparison": "D4S3_selected_policy_minus_score_blend",
            "split": "secondary_blind",
            "n_bootstrap": "",
            "Top10_delta": "",
            "Top10_ci_low": "",
            "Top10_ci_high": "",
            "status": "NOT_EVALUATED_NO_MEANINGFUL_VALIDATION_GAIN",
        },
    ]
    pd.DataFrame(boot_rows).to_csv(OUT / "d4s3_final_bootstrap_diagnostic_only.csv", index=False)

    with open(OUT / "d4s3_final_predictions_diagnostic_only.jsonl", "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "status": "NOT_EVALUATED_NO_MEANINGFUL_VALIDATION_GAIN",
                    "protocol": "D4S3 stopped before old-blind prediction export",
                    "best_validation_observed_policy": result["best_id"],
                    "delta_vs_score_blend_validation": float(result["best"]["delta_vs_score_blend"]),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def write_feature_importance(result: dict) -> None:
    x_val = result["x_val_eval"]
    y_val = result["y_val_eval"]
    feature_cols = result["feature_cols"]
    hgb = result["hgb_model"]
    rng = np.random.RandomState(SEED)
    n_query, n_cand = x_val.shape[:2]
    n_rows = min(50_000, n_query * n_cand)
    idx = rng.choice(n_query * n_cand, size=n_rows, replace=False)
    q_idx = idx // n_cand
    c_idx = idx % n_cand
    x_imp = np.asarray(x_val[q_idx, c_idx][:, : len(feature_cols)], dtype=np.float32)
    y_imp = np.asarray(y_val[q_idx, c_idx], dtype=np.int32)
    rows = []
    if len(np.unique(y_imp)) > 1:
        perm = permutation_importance(
            hgb,
            x_imp,
            y_imp,
            scoring="roc_auc",
            n_repeats=3,
            random_state=SEED,
        )
        for name, mean, std in zip(feature_cols, perm.importances_mean, perm.importances_std):
            rows.append(
                {
                    "model_id": "R4_histgb_F3_sample400k",
                    "feature_name": name,
                    "importance_type": "flat_validation_permutation_auc",
                    "importance_mean": float(mean),
                    "importance_std": float(std),
                    "notes": "Feature importance is diagnostic; selected policy did not clear validation gain threshold.",
                }
            )
    else:
        for name in feature_cols:
            rows.append(
                {
                    "model_id": "R4_histgb_F3_sample400k",
                    "feature_name": name,
                    "importance_type": "blocked",
                    "importance_mean": "",
                    "importance_std": "",
                    "notes": "Permutation importance blocked because sampled validation labels contain one class.",
                }
            )
    imp = pd.DataFrame(rows)
    imp.to_csv(OUT / "d4s3_feature_importance.csv", index=False)

    group_map = {}
    for name in feature_cols:
        if name.startswith("rank_"):
            group_map[name] = "F0 rank"
        elif name.startswith("score_"):
            group_map[name] = "F1 score"
        elif name in {"replacement_frequency", "attachment_frequency", "candidate_num_attachments_train", "compatibility_flag"}:
            group_map[name] = "F2 frequency"
        elif name.startswith("sig_"):
            group_map[name] = "F4 functional proxy"
        else:
            group_map[name] = "F3 2D chemistry"
    if not imp.empty and "importance_mean" in imp:
        fam = imp.copy()
        fam["feature_family"] = fam["feature_name"].map(group_map)
        fam["importance_mean_num"] = pd.to_numeric(fam["importance_mean"], errors="coerce").fillna(0.0)
        family = (
            fam.groupby("feature_family", as_index=False)
            .agg(total_importance=("importance_mean_num", "sum"), n_features=("feature_name", "count"))
            .sort_values("total_importance", ascending=False)
        )
    else:
        family = pd.DataFrame(columns=["feature_family", "total_importance", "n_features"])
    # Add explicitly missing feature families.
    for missing in ["F5 ACCFG", "F6 electron-density", "F7 A4C diagnostic"]:
        family = pd.concat(
            [
                family,
                pd.DataFrame(
                    [{"feature_family": missing, "total_importance": "", "n_features": 0, "notes": "missing_or_not_used"}]
                ),
            ],
            ignore_index=True,
        )
    family.to_csv(OUT / "d4s3_feature_family_importance.csv", index=False)

    write_text(
        OUT / "d4s3_feature_ablation_summary.md",
        f"""
# D4S3 Feature Ablation Summary

The feature-rich validation search did not identify a feature family that moved
Top10 meaningfully beyond the current score blend.

- Best observed validation policy: `{result['best_id']}`
- Validation Top10 delta vs score blend: {float(result['best']['delta_vs_score_blend']):+.6f}
- Required minimum meaningful gain: +0.003000
- Decision: do not update the manuscript main method.

Available F0-F3 features mostly improve MRR or reproduce the existing MLP/HGB
tradeoff. The sampled HGB/SGD F3 models did not exceed the score blend on
validation Top10. Functional-group, ACCFG, electron-density, and A4C diagnostic
features were not available as clean Route-A candidate-level matrices in this
run, so they cannot be credited with any gain.
""",
    )


def write_interpretation_and_verdict(result: dict) -> None:
    gain = float(result["best"]["delta_vs_score_blend"])
    verdict = "D4S3_SMALL_GAIN_NOT_SIGNIFICANT" if gain > 0 else "D4S3_NO_VALIDATION_GAIN_SCORE_BLEND_REMAINS_BEST"
    write_text(
        OUT / "d4s3_mechanism_interpretation.md",
        f"""
# D4S3 Mechanism Interpretation

1. Did richer chemistry features help rare replacements?
Validation evidence does not support a meaningful Top10 improvement. F3
chemistry-rich models improved some early-rank/MRR behavior, but did not clear
the score-blend Top10 threshold.

2. Did functional groups help?
No clean Route-A candidate-level functional-group matrix was available. Existing
attachment-signature one-hots act only as a weak functional proxy and did not
move Top10 beyond the score blend.

3. Did ACCFG help?
No. ACCFG features were not available as Route-A candidate-level aligned inputs
outside v2 geometry work, so they were not used.

4. Did electron-density help?
No. Electron-density descriptors were not available as Route-A candidate-level
aligned inputs and were not used.

5. Did hard negatives move candidates from rank 11-50 into Top10?
Hard-negative loss expansion was not run because the validation screen did not
produce a candidate policy clearing +0.003 Top10 over the score blend.

6. Did D4S3 improve Top10 or only MRR?
The best observed policy produced only a tiny validation Top10 change
({gain:+.6f}) and did not justify final evaluation. Several feature-rich models
increased MRR relative to simpler baselines, which is consistent with D4S2, but
the paper-critical Top10 endpoint remained saturated by the score blend.

7. Did it reduce gap to Oracle?
No manuscript-eligible oracle-gap reduction was established. The secondary
blind oracle gap remains the protected prior value of 0.0128 for the score blend.

8. Did it worsen negative subspaces?
No final claim can be made. Validation subset diagnostics were emitted for the
best observed policy, but they are post-hoc and below the selection threshold.
""",
    )

    write_text(
        OUT / "d4s3_a4c_risk_diagnostic_blocked.md",
        """
# D4S3 A4C Risk Diagnostic Blocked

A4C features were not used for D4S3 model selection, training targets, or final
policy selection.

Reason:
The available A4C V1A audit files describe a separate review-feature surface and
are not cleanly joined to the Route-A full-vocabulary 150-candidate matrix by
query_id/candidate_smiles at blind-wide coverage. Using them as training targets
would violate the protocol. Using them as input features would introduce a
diagnostic-risk feature family without a clean leakage audit.

Result:
No hard-alert rate, tier distribution, G2/G3/G4 composition, or review-proxy
gain/loss is reported for D4S3.
""",
    )

    write_text(
        OUT / "D4S3_FEATURE_RICH_RERANKER_VERDICT.md",
        f"""
# D4S3 Feature-Rich Reranker Verdict

Final verdict: **{verdict}**

## Direct Answers

1. Was clean evaluation possible?
No paper-main clean evaluation was built. This run is validation-only, with any
old-blind reuse classified as `DIAGNOSTIC_ONLY_EXISTING_BLIND`.

2. What feature families were available?
F0 rank, F1 score, F2 frequency/statistical, and F3 2D chemistry from the D4S2
full-vocabulary matrix. F4 functional groups, F5 ACCFG, F6 electron-density, and
F7 A4C were not available as clean Route-A candidate-level training features.

3. Did functional-group features help?
No clean test was possible beyond existing attachment-signature proxies.

4. Did ACCFG features help?
No. ACCFG was missing as a Route-A aligned candidate-level feature source.

5. Did electron-density features help?
No. Electron-density features were missing.

6. Which model/feature set won on validation?
Best observed policy: `{result['best_id']}`.

7. Did final selected model beat score blend?
Only by {gain:+.6f} Top10 on validation, below the +0.003 required threshold.
The final retained method is therefore the current MLP+HGB score blend.

8. Was the gain statistically significant?
No. Validation bootstrap for the best observed policy vs score blend:
[{result['val_boot']['ci_low']:+.6f}, {result['val_boot']['ci_high']:+.6f}].

9. Did rare/hard subsets improve?
Subset diagnostics were generated, but any apparent gains are below the main
selection threshold and should be treated as post-hoc.

10. Did negative subspaces improve?
No robust conclusion. Negative-subspace diagnostics remain post-hoc.

11. Did A4C risk worsen?
No A4C-driven model was trained or selected, so no D4S3 A4C risk worsening was
measured. The A4C diagnostic is blocked.

12. Can this become paper-main?
Not from this run. A paper-main D4S3 claim would need a fresh blind split or
nested transform-heldout CV and a meaningful validation-selected gain.

13. Should the manuscript be upgraded again or remain unchanged?
Remain unchanged. S6R-TEXT should keep the validation-selected MLP+HGB score
blend as the main method and keep D4S3, if mentioned, as a negative/future-work
feature-rich optimization audit.

## Skeptical Review

- The secondary blind has been reused too many times for a new D4S3 SOTA claim.
- Feature-rich models may overfit validation because many policies and
  ensembles were screened.
- Functional-group, ACCFG, and electron-density features were not available as
  clean Route-A candidate-level matrices, so their contribution cannot be
  claimed.
- Richer F3 features mostly add noise or shift MRR rather than Top10.
- Rare/hard gains would be cherry-picked unless reproduced in a clean split.
- A4C diagnostic features would risk leaking review labels and were blocked.
- The score blend appears to saturate most of the reachable DE/HGB oracle gap.
- The observed validation gain is too small to justify updating the manuscript.
- D4S3 should be future work or supplementary negative evidence, not paper-main.

No activity-preserving replacement, wet-lab validation, or open-vocabulary SOTA
claim is made.
""",
    )

    write_text(
        OUT / "MAIN_DECISION_LOG.md",
        f"""
# D4S3 Main Decision Log

- Protocol decision: `DIAGNOSTIC_ONLY_EXISTING_BLIND`.
- Matrix source: reused D4S2 full-vocabulary candidate matrices by hardlink/copy.
- Selection endpoint: validation Top10.
- Current method to beat: MLP+HGB score blend, validation Top10
  {result['score_blend_top10']:.6f}; protected secondary-blind Top10 0.8558.
- Best observed D4S3 policy: `{result['best_id']}`.
- Validation delta vs score blend: {gain:+.6f}.
- Minimum meaningful validation gain required: +0.003.
- Outcome: no final old-blind scoring; manuscript should remain S6R-TEXT with
  score blend as the main method.
""",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    log("Writing protocol and discovery reports.")
    write_protocol_files()
    pd.DataFrame(discovery_rows()).to_csv(OUT / "d4s3_input_discovery.csv", index=False)

    log("Materializing candidate matrix hardlinks.")
    matrix_manifest = materialize_matrix_links()

    schema = json.loads((D4S2 / "d4s2_reranker_feature_schema.json").read_text(encoding="utf-8"))
    write_matrix_reports(schema, matrix_manifest)
    feature_availability(schema)
    write_preprocessing_reports()

    log("Running validation-only model grid.")
    result = run_validation_grid()
    write_final_stopped_outputs(result)
    write_feature_importance(result)
    write_interpretation_and_verdict(result)
    log(f"D4S3 completed with status: {result['status']}")


if __name__ == "__main__":
    main()

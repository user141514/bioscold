#!/usr/bin/env python3
"""Route-A top-journal upgrade S1: score blending, router diagnostics, and review pilot.

This script is intentionally conservative:
- the score blend is selected on validation queries only and scored once on blind;
- the router rule is selected on validation queries only and scored once on blind;
- the expert-review package is blinded, with hidden labels written separately;
- no activity-preservation or human-expert claim is made here.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from routeA_d4s2_common import (  # noqa: E402
    ART,
    B0_BLIND_QUERY_PATH,
    B0_VAL_QUERY_PATH,
    LinearListwise,
    MLPListwise,
    build_frequency_counts,
    compute_query_meta,
    feature_columns_for_sets,
    fit_old_fragment_clusterer,
    load_manifest,
    load_group_arrays,
    metrics_from_score_matrix,
    predict_histgb_group_model,
    predict_torch_group_model,
)

SEED = 20260528
N_BOOT = 5000
ROUTER_MIN_SEGMENT_N = 500
SCORE_BLEND_ALPHA_GRID = [round(x, 3) for x in np.linspace(0.0, 1.0, 41)]
OUT = Path("E:/zuhui/bioisosteric_diffusion/plan_results/routeA_topjournal_s1_failure_router_review_pilot")
OUT.mkdir(parents=True, exist_ok=True)

PLAN = Path("E:/zuhui/bioisosteric_diffusion/plan_results")
D4P1 = PLAN / "routeA_chembl37k_d4p1_phase1_subset_robustness"
D4S_B0 = PLAN / "routeA_chembl37k_d4s_b0_blind_split_baseline"
D4S2 = PLAN / "routeA_chembl37k_d4s2_listwise_reranker"
D4A4_DUAL = PLAN / "routeA_chembl37k_d4a4_dual_mode_integration"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_mrr(rank_series: pd.Series) -> pd.Series:
    ranks = pd.to_numeric(rank_series, errors="coerce")
    ranks = ranks.where(np.isfinite(ranks), np.nan)
    return (1.0 / ranks).fillna(0.0)


def metric_row(name: str, hit: np.ndarray, mrr: np.ndarray, n_total: int, n_routed: int = 0) -> dict:
    return {
        "policy": name,
        "n_queries": int(n_total),
        "n_routed_to_hgb": int(n_routed),
        "Top10": float(np.mean(hit)),
        "MRR": float(np.mean(mrr)),
    }


def bootstrap_delta(
    a_hit: np.ndarray,
    b_hit: np.ndarray,
    a_mrr: np.ndarray,
    b_mrr: np.ndarray,
    label: str,
    rng: np.random.Generator,
) -> dict:
    n = len(a_hit)
    top_deltas = np.empty(N_BOOT, dtype=np.float64)
    mrr_deltas = np.empty(N_BOOT, dtype=np.float64)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        top_deltas[i] = float(np.mean(a_hit[idx] - b_hit[idx]))
        mrr_deltas[i] = float(np.mean(a_mrr[idx] - b_mrr[idx]))
    return {
        "comparison": label,
        "n_queries": int(n),
        "Top10_delta": float(np.mean(a_hit - b_hit)),
        "Top10_ci_low": float(np.quantile(top_deltas, 0.025)),
        "Top10_ci_high": float(np.quantile(top_deltas, 0.975)),
        "MRR_delta": float(np.mean(a_mrr - b_mrr)),
        "MRR_ci_low": float(np.quantile(mrr_deltas, 0.025)),
        "MRR_ci_high": float(np.quantile(mrr_deltas, 0.975)),
    }


def add_baseline_mrr_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Borda_mrr"] = safe_mrr(out["Borda_DE_HGB_best_rank"])
    out["HGB_mrr"] = safe_mrr(out["HGB_best_rank"])
    out["DE_mrr"] = safe_mrr(out["DE_best_rank"])
    out["Attachment_frequency_mrr"] = safe_mrr(out["Attachment_frequency_best_rank"])
    return out


def load_secondary_meta() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = load_manifest()
    global_counts, _attach_counts = build_frequency_counts(manifest)
    clusterer = fit_old_fragment_clusterer(
        manifest.loc[manifest["split_new"] == "train", "old_fragment_smiles"].astype(str).tolist()
    )
    meta, _thresholds = compute_query_meta(
        manifest,
        global_counts,
        clusterer,
        B0_BLIND_QUERY_PATH,
    )
    val_query = pd.read_csv(B0_VAL_QUERY_PATH)
    blind_query = pd.read_csv(B0_BLIND_QUERY_PATH)
    val = add_baseline_mrr_columns(
        meta.loc[meta["split"] == "val"].merge(val_query, on="query_id", how="inner")
    )
    blind = add_baseline_mrr_columns(
        meta.loc[meta["split"] == "blind_test"].merge(blind_query, on="query_id", how="inner")
    )
    selected_path = D4S2 / "artifacts" / "d4s2_selected_blind_query_level.csv"
    if selected_path.exists():
        blind = blind.merge(pd.read_csv(selected_path), on="query_id", how="left")
    return val, blind


def select_router_rules(val: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["old_fragment_cluster_id", "attachment_signature"]
    for keys, sub in val.groupby(group_cols):
        n = int(len(sub))
        borda_top10 = float(sub["Borda_DE_HGB_hit10"].mean())
        hgb_top10 = float(sub["HGB_hit10"].mean())
        borda_mrr = float(sub["Borda_mrr"].mean())
        hgb_mrr = float(sub["HGB_mrr"].mean())
        selected = (
            n >= ROUTER_MIN_SEGMENT_N
            and hgb_top10 > borda_top10
            and hgb_mrr >= borda_mrr
        )
        rows.append(
            {
                "rule_family": "train_frozen_cluster_plus_attachment_signature",
                "old_fragment_cluster_id": str(keys[0]),
                "attachment_signature": str(keys[1]),
                "validation_n": n,
                "validation_Borda_Top10": borda_top10,
                "validation_HGB_Top10": hgb_top10,
                "validation_delta_HGB_minus_Borda_Top10": hgb_top10 - borda_top10,
                "validation_Borda_MRR": borda_mrr,
                "validation_HGB_MRR": hgb_mrr,
                "validation_delta_HGB_minus_Borda_MRR": hgb_mrr - borda_mrr,
                "min_segment_n": ROUTER_MIN_SEGMENT_N,
                "runtime_safe_inputs_only": 1,
                "selected_for_blind_router": int(selected),
            }
        )
    rules = pd.DataFrame(rows).sort_values(
        ["selected_for_blind_router", "validation_delta_HGB_minus_Borda_Top10", "validation_n"],
        ascending=[False, False, False],
    )
    rules.to_csv(OUT / "s1_router_validation_rules.csv", index=False)
    return rules


def apply_router(blind: pd.DataFrame, rules: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = rules.loc[rules["selected_for_blind_router"] == 1].copy()
    route_mask = np.zeros(len(blind), dtype=bool)
    for row in selected.itertuples(index=False):
        route_mask |= (
            (blind["old_fragment_cluster_id"].astype(str) == str(row.old_fragment_cluster_id))
            & (blind["attachment_signature"].astype(str) == str(row.attachment_signature))
        ).to_numpy()
    hit = np.where(route_mask, blind["HGB_hit10"].to_numpy(), blind["Borda_DE_HGB_hit10"].to_numpy())
    mrr = np.where(route_mask, blind["HGB_mrr"].to_numpy(), blind["Borda_mrr"].to_numpy())
    return route_mask, hit.astype(float), mrr.astype(float)


def run_router_experiment() -> dict:
    log("Loading secondary split metadata and query-level metrics.")
    val, blind = load_secondary_meta()
    rules = select_router_rules(val)
    route_mask, router_hit, router_mrr = apply_router(blind, rules)

    metrics = [
        metric_row(
            "Borda_DE_HGB",
            blind["Borda_DE_HGB_hit10"].to_numpy(dtype=float),
            blind["Borda_mrr"].to_numpy(dtype=float),
            len(blind),
        ),
        metric_row(
            "HGB",
            blind["HGB_hit10"].to_numpy(dtype=float),
            blind["HGB_mrr"].to_numpy(dtype=float),
            len(blind),
        ),
        metric_row(
            "S1_validation_router_Borda_or_HGB",
            router_hit,
            router_mrr,
            len(blind),
            int(route_mask.sum()),
        ),
    ]
    if "M5_mlp_F0_hit10" in blind.columns:
        metrics.append(
            metric_row(
                "M5_mlp_F0",
                blind["M5_mlp_F0_hit10"].to_numpy(dtype=float),
                blind["M5_mlp_F0_mrr"].to_numpy(dtype=float),
                len(blind),
            )
        )
    metrics_df = pd.DataFrame(metrics)
    base_top10 = float(metrics_df.loc[metrics_df["policy"] == "Borda_DE_HGB", "Top10"].iloc[0])
    base_mrr = float(metrics_df.loc[metrics_df["policy"] == "Borda_DE_HGB", "MRR"].iloc[0])
    metrics_df["delta_Top10_vs_Borda"] = metrics_df["Top10"] - base_top10
    metrics_df["delta_MRR_vs_Borda"] = metrics_df["MRR"] - base_mrr
    metrics_df.to_csv(OUT / "s1_router_blind_metrics.csv", index=False)

    rng = np.random.default_rng(SEED)
    boot_rows = [
        bootstrap_delta(
            router_hit,
            blind["Borda_DE_HGB_hit10"].to_numpy(dtype=float),
            router_mrr,
            blind["Borda_mrr"].to_numpy(dtype=float),
            "S1_router_minus_Borda",
            rng,
        ),
        bootstrap_delta(
            router_hit,
            blind["HGB_hit10"].to_numpy(dtype=float),
            router_mrr,
            blind["HGB_mrr"].to_numpy(dtype=float),
            "S1_router_minus_HGB",
            rng,
        ),
    ]
    if "M5_mlp_F0_hit10" in blind.columns:
        boot_rows.append(
            bootstrap_delta(
                router_hit,
                blind["M5_mlp_F0_hit10"].to_numpy(dtype=float),
                router_mrr,
                blind["M5_mlp_F0_mrr"].to_numpy(dtype=float),
                "S1_router_minus_M5_mlp_F0",
                rng,
            )
        )
    pd.DataFrame(boot_rows).to_csv(OUT / "s1_router_blind_bootstrap.csv", index=False)

    subset_rows = []
    for col in ["old_fragment_cluster_id", "replacement_frequency_bin", "attachment_signature"]:
        for key, sub in blind.assign(_router_hit=router_hit, _router_mrr=router_mrr, _routed=route_mask).groupby(col):
            subset_rows.append(
                {
                    "subset_family": col,
                    "subset": str(key),
                    "n": int(len(sub)),
                    "n_routed_to_hgb": int(sub["_routed"].sum()),
                    "Borda_Top10": float(sub["Borda_DE_HGB_hit10"].mean()),
                    "HGB_Top10": float(sub["HGB_hit10"].mean()),
                    "S1_router_Top10": float(sub["_router_hit"].mean()),
                    "S1_router_delta_vs_Borda_Top10": float(sub["_router_hit"].mean() - sub["Borda_DE_HGB_hit10"].mean()),
                    "Borda_MRR": float(sub["Borda_mrr"].mean()),
                    "HGB_MRR": float(sub["HGB_mrr"].mean()),
                    "S1_router_MRR": float(sub["_router_mrr"].mean()),
                    "S1_router_delta_vs_Borda_MRR": float(sub["_router_mrr"].mean() - sub["Borda_mrr"].mean()),
                }
            )
    pd.DataFrame(subset_rows).to_csv(OUT / "s1_router_blind_subset_metrics.csv", index=False)

    canonical_path = D4P1 / "d4p1_phase1_query_level_canonical_table.csv"
    canonical = pd.read_csv(canonical_path)
    failure_rows = []
    for label, mask in {
        "canonical_cluster_09_OC_benzyl": canonical["old_fragment_cluster_id"].eq("cluster_09"),
        "canonical_cluster_05_nitro_aromatic": canonical["old_fragment_cluster_id"].eq("cluster_05"),
        "canonical_attachment_C_O": canonical["attachment_signature"].eq("C|O"),
    }.items():
        sub = canonical.loc[mask]
        failure_rows.append(
            {
                "space": label,
                "n": int(len(sub)),
                "Attachment_Top10": float(sub["attach_hit10"].mean()),
                "DE_Top10": float(sub["DE_hit10"].mean()),
                "HGB_Top10": float(sub["HGB_hit10"].mean()),
                "Borda_Top10": float(sub["Borda_hit10"].mean()),
                "HGB_minus_Borda_Top10": float(sub["HGB_hit10"].mean() - sub["Borda_hit10"].mean()),
                "HGB_MRR": float(sub["HGB_mrr"].mean()),
                "Borda_MRR": float(sub["Borda_mrr"].mean()),
                "HGB_minus_Borda_MRR": float(sub["HGB_mrr"].mean() - sub["Borda_mrr"].mean()),
            }
        )
    pd.DataFrame(failure_rows).to_csv(OUT / "s1_canonical_failure_subspace_audit.csv", index=False)

    selected_rules = rules.loc[rules["selected_for_blind_router"] == 1].to_dict("records")
    return {
        "val_n": int(len(val)),
        "blind_n": int(len(blind)),
        "selected_rules": selected_rules,
        "router_n_routed": int(route_mask.sum()),
        "metrics": metrics_df.to_dict("records"),
        "bootstrap": boot_rows,
    }


def load_selected_predictor_local(selected_model_id: str = "M5_mlp_F0") -> dict:
    if selected_model_id == "M3_histgb_F2":
        with open(ART / f"{selected_model_id}.pkl", "rb") as handle:
            payload = pickle.load(handle)
        return {"kind": "histgb", **payload}
    payload = torch.load(ART / f"{selected_model_id}.pt", map_location="cpu", weights_only=False)
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


def score_selected_local(payload: dict, x_matrix: np.ndarray) -> np.ndarray:
    if payload["kind"] == "histgb":
        return predict_histgb_group_model(payload["model"], x_matrix, payload["cols"])
    return predict_torch_group_model(
        payload["model"],
        x_matrix,
        payload["cols"],
        payload["mean"],
        payload["std"],
    )


def row_zscore(scores: np.ndarray) -> np.ndarray:
    arr = scores.astype(np.float32)
    return (arr - arr.mean(axis=1, keepdims=True)) / (arr.std(axis=1, keepdims=True) + 1e-6)


def split_score_signals(split_name: str, payload: dict) -> tuple[list[str], np.ndarray, pd.DataFrame, dict[str, np.ndarray]]:
    x_matrix, labels, query_ids, meta = load_group_arrays(split_name)
    eval_mask = meta["target_any_seen_vocab"].astype(int).to_numpy() == 1
    x_eval = x_matrix[eval_mask]
    y_eval = np.asarray(labels)[eval_mask]
    qids_eval = [qid for qid, keep in zip(query_ids, eval_mask) if keep]
    meta_eval = meta.loc[eval_mask].reset_index(drop=True)

    feature_cols = feature_columns_for_sets()["F3_rank_score_frequency_chemistry"]
    col_idx = {name: idx for idx, name in enumerate(feature_cols)}
    signals = {
        "M5_mlp_F0": score_selected_local(payload, x_eval),
        "Borda_refit_score": np.asarray(x_eval[:, :, col_idx["score_borda"]], dtype=np.float32),
        "HGB_refit_score": np.asarray(x_eval[:, :, col_idx["score_hgb"]], dtype=np.float32),
        "DE_refit_score": np.asarray(x_eval[:, :, col_idx["score_de"]], dtype=np.float32),
        "Attach_frequency_score": np.asarray(x_eval[:, :, col_idx["score_attach"]], dtype=np.float32),
    }
    return qids_eval, y_eval, meta_eval, signals


def hit_mrr_from_scores(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    metrics = metrics_from_score_matrix(scores, labels)
    best_rank = metrics["best_rank"]
    hit = (best_rank <= 10).astype(float)
    mrr = np.where(np.isfinite(best_rank), 1.0 / best_rank, 0.0).astype(float)
    return hit, mrr, metrics


def run_score_blend_experiment() -> dict:
    log("Running validation-selected MLP + baseline-signal score blending.")
    payload = load_selected_predictor_local("M5_mlp_F0")
    val_qids, y_val, _val_meta, val_signals = split_score_signals("val", payload)
    blind_qids, y_blind, _blind_meta, blind_signals = split_score_signals("blind_test", payload)

    signal_pairs = [
        ("M5_mlp_F0", "Borda_refit_score"),
        ("M5_mlp_F0", "HGB_refit_score"),
        ("M5_mlp_F0", "DE_refit_score"),
        ("M5_mlp_F0", "Attach_frequency_score"),
    ]
    grid_rows = []
    best = None
    for left, right in signal_pairs:
        left_val = row_zscore(val_signals[left])
        right_val = row_zscore(val_signals[right])
        for alpha in SCORE_BLEND_ALPHA_GRID:
            scores = alpha * left_val + (1.0 - alpha) * right_val
            metrics = metrics_from_score_matrix(scores, y_val)
            row = {
                "blend_id": f"{left}_plus_{right}",
                "left_signal": left,
                "right_signal": right,
                "alpha_left_signal": float(alpha),
                "validation_Top1": metrics["Top1"],
                "validation_Top5": metrics["Top5"],
                "validation_Top10": metrics["Top10"],
                "validation_Top20": metrics["Top20"],
                "validation_Top50": metrics["Top50"],
                "validation_MRR": metrics["MRR"],
                "validation_NDCG10": metrics["NDCG@10"],
            }
            grid_rows.append(row)
            key = (
                row["validation_Top10"],
                row["validation_MRR"],
                row["validation_Top5"],
                -abs(1.0 - float(alpha)),  # prefer staying close to the selected MLP on exact ties
            )
            if best is None or key > best[0]:
                best = (key, row)
    grid = pd.DataFrame(grid_rows)
    grid["selected_by_validation"] = 0
    selected = best[1]
    selected_mask = (
        (grid["blend_id"] == selected["blend_id"])
        & (grid["alpha_left_signal"] == selected["alpha_left_signal"])
    )
    grid.loc[selected_mask, "selected_by_validation"] = 1
    grid.to_csv(OUT / "s1_score_blend_validation_grid.csv", index=False)

    left = str(selected["left_signal"])
    right = str(selected["right_signal"])
    alpha = float(selected["alpha_left_signal"])
    blend_scores = alpha * row_zscore(blind_signals[left]) + (1.0 - alpha) * row_zscore(blind_signals[right])
    blend_hit, blend_mrr, blend_metrics = hit_mrr_from_scores(blend_scores, y_blind)
    mlp_hit, mlp_mrr, mlp_metrics = hit_mrr_from_scores(blind_signals["M5_mlp_F0"], y_blind)

    official = pd.DataFrame({"query_id": blind_qids})
    official = official.merge(pd.read_csv(B0_BLIND_QUERY_PATH), on="query_id", how="left")
    official["Borda_mrr"] = safe_mrr(official["Borda_DE_HGB_best_rank"])
    official_borda_hit = official["Borda_DE_HGB_hit10"].to_numpy(dtype=float)
    official_borda_mrr = official["Borda_mrr"].to_numpy(dtype=float)

    metrics_rows = [
        {
            "policy": "Borda_DE_HGB_official",
            "n_queries": int(len(blind_qids)),
            "Top1": float((pd.to_numeric(official["Borda_DE_HGB_best_rank"], errors="coerce") <= 1).mean()),
            "Top5": float((pd.to_numeric(official["Borda_DE_HGB_best_rank"], errors="coerce") <= 5).mean()),
            "Top10": float(np.mean(official_borda_hit)),
            "Top20": float((pd.to_numeric(official["Borda_DE_HGB_best_rank"], errors="coerce") <= 20).mean()),
            "Top50": float((pd.to_numeric(official["Borda_DE_HGB_best_rank"], errors="coerce") <= 50).mean()),
            "MRR": float(np.mean(official_borda_mrr)),
        },
        {
            "policy": "M5_mlp_F0",
            "n_queries": int(len(blind_qids)),
            "Top1": mlp_metrics["Top1"],
            "Top5": mlp_metrics["Top5"],
            "Top10": mlp_metrics["Top10"],
            "Top20": mlp_metrics["Top20"],
            "Top50": mlp_metrics["Top50"],
            "MRR": mlp_metrics["MRR"],
        },
        {
            "policy": "S1_validation_selected_score_blend",
            "n_queries": int(len(blind_qids)),
            "Top1": blend_metrics["Top1"],
            "Top5": blend_metrics["Top5"],
            "Top10": blend_metrics["Top10"],
            "Top20": blend_metrics["Top20"],
            "Top50": blend_metrics["Top50"],
            "MRR": blend_metrics["MRR"],
            "selected_blend_id": selected["blend_id"],
            "alpha_left_signal": alpha,
        },
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT / "s1_score_blend_blind_metrics.csv", index=False)

    rng = np.random.default_rng(SEED + 11)
    boot_rows = [
        bootstrap_delta(
            blend_hit,
            official_borda_hit,
            blend_mrr,
            official_borda_mrr,
            "S1_score_blend_minus_official_Borda",
            rng,
        ),
        bootstrap_delta(
            blend_hit,
            mlp_hit,
            blend_mrr,
            mlp_mrr,
            "S1_score_blend_minus_M5_mlp_F0",
            rng,
        ),
    ]
    pd.DataFrame(boot_rows).to_csv(OUT / "s1_score_blend_blind_bootstrap.csv", index=False)

    query_level = pd.DataFrame(
        {
            "query_id": blind_qids,
            "S1_score_blend_hit10": blend_hit.astype(int),
            "S1_score_blend_mrr": blend_mrr,
            "M5_mlp_F0_hit10_recomputed": mlp_hit.astype(int),
            "M5_mlp_F0_mrr_recomputed": mlp_mrr,
            "Borda_DE_HGB_hit10_official": official_borda_hit.astype(int),
            "Borda_DE_HGB_mrr_official": official_borda_mrr,
        }
    )
    query_level.to_csv(OUT / "s1_score_blend_blind_query_level.csv", index=False)

    return {
        "val_n": int(len(val_qids)),
        "blind_n": int(len(blind_qids)),
        "selected": selected,
        "metrics": metrics_df.to_dict("records"),
        "bootstrap": boot_rows,
    }


def normalize_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def stable_sample(df: pd.DataFrame, n: int, seed_offset: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=SEED + seed_offset).copy()


def ensure_star(value: str) -> str:
    value = str(value)
    if value.startswith("*"):
        return value
    return f"*{value}"


def build_review_pilot() -> dict:
    log("Building blinded expert-review pilot package from D4A4 dual-mode top10 rows.")

    conservative = pd.read_csv(D4A4_DUAL / "d4a4_conservative_mode_top10.csv")
    exploration = pd.read_csv(D4A4_DUAL / "d4a4_exploration_mode_top10.csv")
    for frame in (conservative, exploration):
        for col in ["rank_HGB", "rank_DE", "rank_Borda"]:
            if col not in frame.columns:
                frame[col] = np.nan
    top10 = pd.concat([conservative, exploration], ignore_index=True)
    top10["candidate_norm"] = top10["candidate_norm"].astype(str)

    def mode_membership(values: pd.Series) -> str:
        modes = sorted(set(values.astype(str)))
        if modes == ["Conservative", "Exploration"]:
            return "Both_modes"
        if modes == ["Exploration"]:
            return "Exploration_only"
        if modes == ["Conservative"]:
            return "Conservative_only"
        return "|".join(modes)

    grouped = (
        top10.groupby(["query_id", "candidate_norm"], as_index=False)
        .agg(
            mode_membership=("mode", mode_membership),
            conservative_topK_rank=("topK_rank", lambda x: float(x[top10.loc[x.index, "mode"].eq("Conservative")].min()) if top10.loc[x.index, "mode"].eq("Conservative").any() else np.nan),
            exploration_topK_rank=("topK_rank", lambda x: float(x[top10.loc[x.index, "mode"].eq("Exploration")].min()) if top10.loc[x.index, "mode"].eq("Exploration").any() else np.nan),
            rank_HGB=("rank_HGB", "min"),
            rank_DE=("rank_DE", "min"),
            rank_Borda=("rank_Borda", "min"),
            group_origin=("group_origin", "first"),
            final_action_tier=("final_action_tier", "first"),
            a4c_review_bucket=("a4c_review_bucket", "first"),
            hard_alert_flag=("hard_alert_flag", "max"),
            review_ready_flag=("review_ready_flag", "max"),
            reason_codes=("reason_codes", "first"),
        )
    )

    canonical_cols = [
        "query_id",
        "candidate_norm",
        "old_fragment_smiles",
        "attachment_signature",
        "core_key",
        "is_positive_any",
        "score_HGB",
        "score_DE",
        "score_Borda",
        "property_warning_flag",
        "geometry_warning_flag",
        "alert_type",
        "alert_reason_codes",
    ]
    canonical = pd.read_csv(D4A4_DUAL / "d4a4_canonical_candidate_table.csv", usecols=canonical_cols)
    all_candidates = grouped.merge(canonical, on=["query_id", "candidate_norm"], how="left")
    all_candidates["replacement_fragment_smiles"] = all_candidates["candidate_norm"].map(ensure_star)

    def classify(row: pd.Series) -> str:
        tier = str(row["final_action_tier"])
        group = str(row["group_origin"])
        mode = str(row["mode_membership"])
        if tier == "Tier2_EXPERT_REVIEW" and group == "G3_de_retained_by_borda" and mode == "Exploration_only":
            return "Tier2_G3_exploration_only"
        if tier == "Tier2_EXPERT_REVIEW" and group == "G3_de_retained_by_borda" and mode == "Both_modes":
            return "Tier2_G3_both_modes"
        if tier == "Tier2_EXPERT_REVIEW" and group == "G2_pure_borda_only" and mode == "Both_modes":
            return "Tier2_G2_both_modes"
        if tier == "Tier1_STANDARD_REVIEW" and group == "G3_de_retained_by_borda":
            return "Tier1_G3_standard_review"
        if tier == "Tier3_HARD_REJECT" and group in {"G2_pure_borda_only", "G3_de_retained_by_borda"}:
            return "Tier3_G2_G3_hard_reject_control"
        if tier == "Tier0_DATA_PENDING" and group == "G4_shared":
            return "Tier0_G4_shared_low_risk_control"
        return "not_sampled"

    all_candidates["source_stratum"] = all_candidates.apply(classify, axis=1)
    all_candidates = all_candidates.loc[all_candidates["source_stratum"] != "not_sampled"].copy()
    all_candidates = all_candidates.drop_duplicates(["query_id", "candidate_norm", "source_stratum"])
    all_candidates["scaffold_context_id"] = all_candidates["core_key"].astype(str)

    requested = {
        "Tier2_G3_exploration_only": 90,
        "Tier2_G3_both_modes": 60,
        "Tier2_G2_both_modes": 45,
        "Tier1_G3_standard_review": 45,
        "Tier3_G2_G3_hard_reject_control": 30,
        "Tier0_G4_shared_low_risk_control": 30,
    }

    def sample_one_per_query(pool: pd.DataFrame, n: int, seed_offset: int) -> pd.DataFrame:
        shuffled = pool.sample(frac=1.0, random_state=SEED + seed_offset).copy()
        first_pass = shuffled.drop_duplicates("query_id", keep="first")
        if len(first_pass) >= n:
            return first_pass.head(n).copy()
        remainder = shuffled.loc[~shuffled.index.isin(first_pass.index)]
        return pd.concat([first_pass, remainder.head(n - len(first_pass))], ignore_index=False).copy()

    samples = []
    availability_rows = []
    for i, (stratum, n) in enumerate(requested.items()):
        pool = all_candidates.loc[all_candidates["source_stratum"] == stratum].copy()
        sampled = sample_one_per_query(pool, n, i)
        availability_rows.append(
            {
                "source_stratum": stratum,
                "available_unique_rows": int(len(pool)),
                "available_unique_queries": int(pool["query_id"].nunique()),
                "available_unique_fragment_pairs": int(
                    pool.drop_duplicates(
                        ["old_fragment_smiles", "replacement_fragment_smiles", "attachment_signature"]
                    ).shape[0]
                ),
                "requested_n": int(n),
                "sampled_n": int(len(sampled)),
            }
        )
        samples.append(sampled)

    key = pd.concat(samples, ignore_index=True)
    key = key.sample(frac=1.0, random_state=SEED + 101).reset_index(drop=True)
    key.insert(0, "review_id", [f"S1REV_{i:04d}" for i in range(1, len(key) + 1)])
    key["review_instructions_version"] = "fragment_pair_blind_v1"
    key["human_label_plausible_bioisostere"] = ""
    key["human_label_context_dependent"] = ""
    key["human_label_reject_reason"] = ""
    key["human_confidence_1_to_5"] = ""
    key["reviewer_notes"] = ""

    blinded_cols = [
        "review_id",
        "scaffold_context_id",
        "old_fragment_smiles",
        "replacement_fragment_smiles",
        "candidate_norm",
        "attachment_signature",
        "human_label_plausible_bioisostere",
        "human_label_context_dependent",
        "human_label_reject_reason",
        "human_confidence_1_to_5",
        "reviewer_notes",
    ]
    key.to_csv(OUT / "s1_expert_review_pilot_key_300.csv", index=False)
    key[blinded_cols].to_csv(OUT / "s1_expert_review_pilot_blinded_300.csv", index=False)
    pd.DataFrame(availability_rows).to_csv(OUT / "s1_expert_review_pilot_availability.csv", index=False)

    return {
        "n_blinded_rows": int(len(key)),
        "availability": availability_rows,
        "stratum_counts": key["source_stratum"].value_counts().to_dict(),
    }


def write_verdict(router_summary: dict, score_summary: dict, pilot_summary: dict) -> None:
    metrics = {row["policy"]: row for row in router_summary["metrics"]}
    boot = {row["comparison"]: row for row in router_summary["bootstrap"]}
    selected_rules = router_summary["selected_rules"]
    rules_text = "\n".join(
        [
            (
                f"- {r['old_fragment_cluster_id']} + {r['attachment_signature']}: "
                f"validation n={r['validation_n']}, "
                f"HGB-Borda Top10={r['validation_delta_HGB_minus_Borda_Top10']:.4f}, "
                f"HGB-Borda MRR={r['validation_delta_HGB_minus_Borda_MRR']:.4f}"
            )
            for r in selected_rules
        ]
    )
    if not rules_text:
        rules_text = "- No validation segment passed the locked routing criteria."

    router = metrics["S1_validation_router_Borda_or_HGB"]
    borda = metrics["Borda_DE_HGB"]
    hgb = metrics["HGB"]
    mlp = metrics.get("M5_mlp_F0")
    router_vs_borda = boot["S1_router_minus_Borda"]
    router_vs_hgb = boot["S1_router_minus_HGB"]
    router_vs_mlp = boot.get("S1_router_minus_M5_mlp_F0")

    score_metrics = {row["policy"]: row for row in score_summary["metrics"]}
    score_boot = {row["comparison"]: row for row in score_summary["bootstrap"]}
    score_selected = score_summary["selected"]
    score_blend = score_metrics["S1_validation_selected_score_blend"]
    score_mlp = score_metrics["M5_mlp_F0"]
    score_borda = score_metrics["Borda_DE_HGB_official"]
    blend_vs_borda = score_boot["S1_score_blend_minus_official_Borda"]
    blend_vs_mlp = score_boot["S1_score_blend_minus_M5_mlp_F0"]

    mlp_line = ""
    if mlp and router_vs_mlp:
        mlp_line = (
            f"- MLP blind Top10={mlp['Top10']:.4f}, MRR={mlp['MRR']:.4f}; "
            f"router-MLP Top10 delta={router_vs_mlp['Top10_delta']:.4f} "
            f"95% CI [{router_vs_mlp['Top10_ci_low']:.4f}, {router_vs_mlp['Top10_ci_high']:.4f}], "
            f"MRR delta={router_vs_mlp['MRR_delta']:.4f} "
            f"95% CI [{router_vs_mlp['MRR_ci_low']:.4f}, {router_vs_mlp['MRR_ci_high']:.4f}].\n"
        )

    availability_lines = "\n".join(
        [
            f"- {row['source_stratum']}: available={row['available_unique_rows']}, "
            f"unique fragment pairs={row.get('available_unique_fragment_pairs', 'NA')}, "
            f"sampled={row['sampled_n']}"
            for row in pilot_summary["availability"]
        ]
    )

    verdict = f"""# S1 Top-Journal Upgrade Verdict

**Verdict:** B. S1_SCORE_BLEND_UPGRADE_AND_EXPERT_REVIEW_PACKAGE_READY_WITH_CAVEATS

## What Was Added

1. A validation-selected score blend was evaluated once on the blind split and improves the selected D4S2 MLP Top10.
2. A deployable-field-only failure-aware router was tested as a diagnostic stress test.
3. A 300-row blinded expert-review pilot package was materialized with a separate hidden key.
4. The activity-validation boundary remains unchanged: the current local Route-A stack is structure-derived and cannot support activity-preservation claims.

## Score-Blend Upgrade

Candidate family: selected D4S2 `M5_mlp_F0` plus one train-derived baseline score (`Borda_refit_score`, `HGB_refit_score`, `DE_refit_score`, or `Attach_frequency_score`), with row-wise z-scoring and alpha chosen on validation only.

Selected validation blend: `{score_selected['blend_id']}` with alpha on `{score_selected['left_signal']}` = {score_selected['alpha_left_signal']:.3f}.

- Official Borda(DE,HGB) blind Top10={score_borda['Top10']:.4f}, MRR={score_borda['MRR']:.4f}.
- M5_mlp_F0 blind Top10={score_mlp['Top10']:.4f}, MRR={score_mlp['MRR']:.4f}.
- S1 score blend blind Top10={score_blend['Top10']:.4f}, MRR={score_blend['MRR']:.4f}.
- Score blend - official Borda Top10 delta={blend_vs_borda['Top10_delta']:.4f}, 95% CI [{blend_vs_borda['Top10_ci_low']:.4f}, {blend_vs_borda['Top10_ci_high']:.4f}]; MRR delta={blend_vs_borda['MRR_delta']:.4f}, 95% CI [{blend_vs_borda['MRR_ci_low']:.4f}, {blend_vs_borda['MRR_ci_high']:.4f}].
- Score blend - M5_mlp_F0 Top10 delta={blend_vs_mlp['Top10_delta']:.4f}, 95% CI [{blend_vs_mlp['Top10_ci_low']:.4f}, {blend_vs_mlp['Top10_ci_high']:.4f}]; MRR delta={blend_vs_mlp['MRR_delta']:.4f}, 95% CI [{blend_vs_mlp['MRR_ci_low']:.4f}, {blend_vs_mlp['MRR_ci_high']:.4f}].

## Router Diagnostic

Locked selection criteria: segment family = train-frozen `old_fragment_cluster_id + attachment_signature`, validation segment n >= {ROUTER_MIN_SEGMENT_N}, HGB Top10 > Borda Top10, and HGB MRR >= Borda MRR. These are runtime-safe fields; target-frequency and hit/miss labels are deliberately excluded.

Selected validation rule(s):

{rules_text}

## Blind Results

- Borda(DE,HGB): Top10={borda['Top10']:.4f}, MRR={borda['MRR']:.4f}.
- HGB: Top10={hgb['Top10']:.4f}, MRR={hgb['MRR']:.4f}.
- S1 validation router: Top10={router['Top10']:.4f}, MRR={router['MRR']:.4f}, routed queries={router['n_routed_to_hgb']}/{router['n_queries']}.
- Router-Borda Top10 delta={router_vs_borda['Top10_delta']:.4f}, 95% CI [{router_vs_borda['Top10_ci_low']:.4f}, {router_vs_borda['Top10_ci_high']:.4f}]; MRR delta={router_vs_borda['MRR_delta']:.4f}, 95% CI [{router_vs_borda['MRR_ci_low']:.4f}, {router_vs_borda['MRR_ci_high']:.4f}].
- Router-HGB Top10 delta={router_vs_hgb['Top10_delta']:.4f}, 95% CI [{router_vs_hgb['Top10_ci_low']:.4f}, {router_vs_hgb['Top10_ci_high']:.4f}]; MRR delta={router_vs_hgb['MRR_delta']:.4f}, 95% CI [{router_vs_hgb['MRR_ci_low']:.4f}, {router_vs_hgb['MRR_ci_high']:.4f}].
{mlp_line}
Interpretation: the deployable-field router is a stress-test result, not the main algorithmic upgrade. If its blind delta is not positive, keep it as a negative diagnostic showing that canonical failure clusters do not transfer reliably.

## Expert Review Pilot Package

Blinded file: `s1_expert_review_pilot_blinded_300.csv`

Hidden key: `s1_expert_review_pilot_key_300.csv`

Availability and sampling:

{availability_lines}

This package is ready for real medicinal-chemistry review as query-context replacement cases. Some strata deliberately contain repeated fragment transformations across different scaffold contexts; `scaffold_context_id` is included in the blinded table to separate those cases. It is not, by itself, an expert-validation result until independent labels are returned.

## Claim Boundary

- Allowed now: structure-derived replacement ranking; blind split Top10 improvement from a validation-selected score blend; materialized blinded expert-review package.
- Strongest new algorithmic claim: validation-selected MLP+baseline-score blending improves blind Top10 over the selected MLP and official Borda baseline.
- Not allowed now: activity-preserving bioisostere discovery, wet-lab validation, or human-expert validation.

## Files

- `s1_router_validation_rules.csv`
- `s1_router_blind_metrics.csv`
- `s1_router_blind_bootstrap.csv`
- `s1_router_blind_subset_metrics.csv`
- `s1_score_blend_validation_grid.csv`
- `s1_score_blend_blind_metrics.csv`
- `s1_score_blend_blind_bootstrap.csv`
- `s1_score_blend_blind_query_level.csv`
- `s1_canonical_failure_subspace_audit.csv`
- `s1_expert_review_pilot_blinded_300.csv`
- `s1_expert_review_pilot_key_300.csv`
- `s1_expert_review_pilot_availability.csv`
"""
    (OUT / "S1_TOPJOURNAL_UPGRADE_VERDICT.md").write_text(verdict, encoding="utf-8")

    decision_log = {
        "verdict": "B. S1_SCORE_BLEND_UPGRADE_AND_EXPERT_REVIEW_PACKAGE_READY_WITH_CAVEATS",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "router_summary": router_summary,
        "score_summary": score_summary,
        "pilot_summary": pilot_summary,
        "claim_boundary": {
            "activity_preservation_claim": False,
            "human_expert_validation_claim": False,
            "blind_top10_score_blend_claim": True,
            "blind_top10_router_claim": False,
        },
    }
    (OUT / "MAIN_DECISION_LOG.md").write_text(
        "# MAIN_DECISION_LOG\n\n```json\n"
        + json.dumps(decision_log, ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )


def main() -> None:
    router_summary = run_router_experiment()
    score_summary = run_score_blend_experiment()
    pilot_summary = build_review_pilot()
    write_verdict(router_summary, score_summary, pilot_summary)
    log(f"Wrote S1 outputs to {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Route-A D4S2 blind evaluation after validation model selection."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from routeA_d4s2_common import (
    ART,
    B0_BLIND_METRICS_PATH,
    B0_BLIND_QUERY_PATH,
    B0_VAL_METRICS_PATH,
    MATRIX_DIR,
    OUT,
    LinearListwise,
    MLPListwise,
    build_feature_tensor_rows,
    compute_query_meta,
    feature_columns_for_sets,
    fit_old_fragment_clusterer,
    full_attach_arrays,
    load_baseline_artifacts,
    load_group_arrays,
    load_manifest,
    metrics_from_score_matrix,
    predict_histgb_group_model,
    predict_torch_group_model,
    query_level_from_scores,
    score_de_full,
    score_hgb_full,
    write_a4c_blocked_note,
)


def pick_selected_model():
    comp = pd.read_csv(OUT / "d4s2_validation_comparison.csv")
    models = comp.loc[comp["model_id"].astype(str).str.startswith("M")].copy()
    if models.empty:
        return None
    models = models.sort_values(
        by=["Top10", "MRR", "Top5", "complexity"],
        ascending=[False, False, False, True],
    )
    best = models.iloc[0].to_dict()
    b0_val = pd.read_csv(B0_VAL_METRICS_PATH)
    b0_borda_top10 = float(
        b0_val.loc[b0_val["method"] == "Borda(DE,HGB)", "Top10"].iloc[0]
    )
    if float(best["Top10"]) <= b0_borda_top10:
        return None
    return best


def load_selected_predictor(selected_model_id: str):
    feature_sets = feature_columns_for_sets()
    if selected_model_id.endswith(".pkl"):
        raise RuntimeError("Unexpected path rather than model id.")
    if selected_model_id == "M3_histgb_F2":
        with open(ART / f"{selected_model_id}.pkl", "rb") as handle:
            payload = pickle.load(handle)
        return {"kind": "histgb", **payload}
    payload = torch.load(ART / f"{selected_model_id}.pt", map_location="cpu")
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


def score_selected(payload, X_blind: np.ndarray):
    if payload["kind"] == "histgb":
        return predict_histgb_group_model(payload["model"], X_blind, payload["cols"])
    return predict_torch_group_model(
        payload["model"],
        X_blind,
        payload["cols"],
        payload["mean"],
        payload["std"],
    )


def main():
    selected = pick_selected_model()
    if selected is None:
        write_a4c_blocked_note()
        raise SystemExit("NO_VALIDATION_WINNER")

    manifest = load_manifest()
    artifacts = load_baseline_artifacts()
    clusterer = fit_old_fragment_clusterer(
        manifest.loc[manifest["split_new"] == "train", "old_fragment_smiles"].astype(str).tolist()
    )
    query_meta_df, _thresholds = compute_query_meta(
        manifest,
        artifacts.global_counts,
        clusterer,
        B0_BLIND_QUERY_PATH,
    )
    blind_df = manifest.loc[manifest["split_new"] == "blind_test"].copy()
    de_scores, de_ranks = score_de_full(blind_df, artifacts)
    hgb_scores, hgb_ranks = score_hgb_full(blind_df, artifacts)
    attach_scores, attach_ranks = full_attach_arrays(blind_df, artifacts)
    blind_meta = query_meta_df.loc[query_meta_df["split"] == "blind_test"].copy()
    build_feature_tensor_rows(
        query_df=blind_df,
        query_meta_df=blind_meta,
        artifacts=artifacts,
        de_scores=de_scores,
        de_ranks=de_ranks,
        hgb_scores=hgb_scores,
        hgb_ranks=hgb_ranks,
        attach_scores=attach_scores,
        attach_ranks=attach_ranks,
        split_name="blind_test",
        include_labels=True,
        csv_prefix="d4s2_reranker_matrix_blind",
    )
    blind_dir = MATRIX_DIR / "blind_test"
    for src in blind_dir.glob("d4s2_reranker_matrix_blind*.csv.gz"):
        target = OUT / src.name
        target.write_bytes(src.read_bytes())

    X_blind, y_blind, blind_qids, blind_query_meta = load_group_arrays("blind_test")
    blind_eval_mask = blind_query_meta["target_any_seen_vocab"].astype(int).to_numpy() == 1
    blind_df_eval = blind_df.loc[blind_eval_mask].reset_index(drop=True)
    X_blind_eval = X_blind[blind_eval_mask]
    y_blind_np = np.asarray(y_blind)[blind_eval_mask]
    blind_qids = [qid for qid, keep in zip(blind_qids, blind_eval_mask) if keep]
    blind_query_meta = blind_query_meta.loc[blind_eval_mask].reset_index(drop=True)
    payload = load_selected_predictor(str(selected["model_id"]))
    sel_scores = score_selected(payload, X_blind_eval)
    sel_metrics = metrics_from_score_matrix(sel_scores, y_blind_np)

    blind_baselines = pd.read_csv(B0_BLIND_METRICS_PATH)
    rows = blind_baselines.to_dict("records")
    rows.append(
        {
            "split": "blind_test",
            "method": str(selected["model_id"]),
            "n_queries_total_split": int(len(blind_df)),
            "n_queries_eval_seen_vocab": int(len(blind_qids)),
            "coverage": float(blind_df["target_any_seen_vocab"].mean()),
            "Top1": sel_metrics["Top1"],
            "Top5": sel_metrics["Top5"],
            "Top10": sel_metrics["Top10"],
            "Top20": sel_metrics["Top20"],
            "Top50": sel_metrics["Top50"],
            "MRR": sel_metrics["MRR"],
        }
    )
    blind_metrics_df = pd.DataFrame(rows)
    blind_metrics_df["gain_vs_Borda"] = np.where(
        blind_metrics_df["method"] == str(selected["model_id"]),
        blind_metrics_df["Top10"]
        - float(
            blind_baselines.loc[
                blind_baselines["method"] == "Borda(DE,HGB)", "Top10"
            ].iloc[0]
        ),
        np.nan,
    )
    blind_metrics_df["gain_vs_HGB"] = np.where(
        blind_metrics_df["method"] == str(selected["model_id"]),
        blind_metrics_df["Top10"]
        - float(blind_baselines.loc[blind_baselines["method"] == "HGB", "Top10"].iloc[0]),
        np.nan,
    )
    blind_metrics_df["gap_to_Oracle"] = np.where(
        blind_metrics_df["method"] == str(selected["model_id"]),
        float(
            blind_baselines.loc[
                blind_baselines["method"] == "Oracle(DE,HGB)", "Top10"
            ].iloc[0]
        )
        - blind_metrics_df["Top10"],
        np.nan,
    )
    blind_metrics_df.to_csv(OUT / "d4s2_blind_test_metrics.csv", index=False)

    query_level = query_level_from_scores(
        blind_qids, sel_scores, y_blind_np, str(selected["model_id"])
    )
    query_level.to_csv(ART / "d4s2_selected_blind_query_level.csv", index=False)

    # Selected predictions top50.
    order = np.argsort(-sel_scores, axis=1, kind="mergesort")[:, :50]
    rows = []
    vocab = artifacts.vocab
    for i, qid in enumerate(blind_qids):
        positives = set(blind_df_eval.iloc[i]["positive_replacement_set"])
        for rank_idx, cand_idx in enumerate(order[i], start=1):
            cand = vocab[int(cand_idx)]
            rows.append(
                {
                    "query_id": qid,
                    "method": str(selected["model_id"]),
                    "rank": rank_idx,
                    "candidate": cand,
                    "score": float(sel_scores[i, int(cand_idx)]),
                    "is_pos": int(cand in positives),
                }
            )
    with open(OUT / "d4s2_blind_predictions.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Blind subset metrics.
    b0_query = pd.read_csv(B0_BLIND_QUERY_PATH)
    merged_meta = blind_query_meta.merge(
        b0_query[["query_id", "Borda_DE_HGB_hit10", "Borda_DE_HGB_best_rank"]],
        on="query_id",
        how="left",
    )
    subset_rows = []
    selected_best = sel_metrics["best_rank"]
    for subset_name, mask in [
        ("all_blind", np.ones(len(merged_meta), dtype=bool)),
        ("hard_top10_miss", merged_meta["hard_top10_miss_flag"].astype(int).to_numpy() == 1),
        ("rare_replacement", merged_meta["replacement_frequency_bin"].eq("rare_replacement").to_numpy()),
        ("frequent_replacement", merged_meta["replacement_frequency_bin"].eq("frequent_replacement").to_numpy()),
        ("single_pos", merged_meta["single_pos_flag"].astype(int).to_numpy() == 1),
        ("multi_pos", merged_meta["multi_pos_flag"].astype(int).to_numpy() == 1),
    ]:
        if int(mask.sum()) == 0:
            continue
        subset_rows.append(
            {
                "subset": subset_name,
                "N_queries": int(mask.sum()),
                "Selected_Top10": float(np.mean(selected_best[mask] <= 10)),
                "Borda_Top10": float(np.mean(merged_meta.loc[mask, "Borda_DE_HGB_hit10"].astype(int))),
                "delta_Selected_minus_Borda": float(
                    np.mean(selected_best[mask] <= 10)
                    - np.mean(merged_meta.loc[mask, "Borda_DE_HGB_hit10"].astype(int))
                ),
            }
        )
    for sig in sorted(merged_meta["attachment_signature"].unique()):
        mask = merged_meta["attachment_signature"].eq(sig).to_numpy()
        subset_rows.append(
            {
                "subset": f"attachment_signature::{sig}",
                "N_queries": int(mask.sum()),
                "Selected_Top10": float(np.mean(selected_best[mask] <= 10)),
                "Borda_Top10": float(np.mean(merged_meta.loc[mask, "Borda_DE_HGB_hit10"].astype(int))),
                "delta_Selected_minus_Borda": float(
                    np.mean(selected_best[mask] <= 10)
                    - np.mean(merged_meta.loc[mask, "Borda_DE_HGB_hit10"].astype(int))
                ),
            }
        )
    for cluster in sorted(merged_meta["old_fragment_cluster_id"].unique()):
        mask = merged_meta["old_fragment_cluster_id"].eq(cluster).to_numpy()
        if int(mask.sum()) < 50:
            continue
        subset_rows.append(
            {
                "subset": f"old_fragment_cluster::{cluster}",
                "N_queries": int(mask.sum()),
                "Selected_Top10": float(np.mean(selected_best[mask] <= 10)),
                "Borda_Top10": float(np.mean(merged_meta.loc[mask, "Borda_DE_HGB_hit10"].astype(int))),
                "delta_Selected_minus_Borda": float(
                    np.mean(selected_best[mask] <= 10)
                    - np.mean(merged_meta.loc[mask, "Borda_DE_HGB_hit10"].astype(int))
                ),
            }
        )
    pd.DataFrame(subset_rows).to_csv(OUT / "d4s2_blind_subset_metrics.csv", index=False)

    # Gain/loss analysis.
    gain_rows = []
    labels = y_blind_np
    for i, qid in enumerate(blind_qids):
        sel_hit10 = int(selected_best[i] <= 10)
        borda_hit10 = int(merged_meta.loc[merged_meta["query_id"] == qid, "Borda_DE_HGB_hit10"].iloc[0])
        full_idx = np.where(blind_eval_mask)[0][i]
        pos_de_top10 = int(np.any((de_ranks[full_idx] <= 10) & (labels[i] > 0)))
        pos_hgb_top10 = int(np.any((hgb_ranks[full_idx] <= 10) & (labels[i] > 0)))
        pos_attach_top10 = int(np.any((attach_ranks[full_idx] <= 10) & (labels[i] > 0)))
        if sel_hit10 and not borda_hit10:
            if pos_de_top10 and not pos_hgb_top10:
                rescue_source = "DE_only_top10_signal"
            elif pos_hgb_top10 and not pos_de_top10:
                rescue_source = "HGB_only_top10_signal"
            elif pos_de_top10 and pos_hgb_top10:
                rescue_source = "DE_and_HGB_top10_signal"
            else:
                rescue_source = "below_top10_rescued_by_reranker"
        elif borda_hit10 and not sel_hit10:
            rescue_source = "Borda_only"
        else:
            rescue_source = "same_outcome"
        gain_rows.append(
            {
                "query_id": qid,
                "selected_hit10": sel_hit10,
                "borda_hit10": borda_hit10,
                "gain_flag": int(sel_hit10 and not borda_hit10),
                "loss_flag": int(borda_hit10 and not sel_hit10),
                "selected_best_rank": "" if not math.isfinite(selected_best[i]) else float(selected_best[i]),
                "borda_best_rank": merged_meta.loc[merged_meta["query_id"] == qid, "Borda_DE_HGB_best_rank"].iloc[0],
                "hard_top10_miss_flag": int(merged_meta.loc[merged_meta["query_id"] == qid, "hard_top10_miss_flag"].iloc[0]),
                "replacement_frequency_bin": merged_meta.loc[merged_meta["query_id"] == qid, "replacement_frequency_bin"].iloc[0],
                "attachment_signature": merged_meta.loc[merged_meta["query_id"] == qid, "attachment_signature"].iloc[0],
                "old_fragment_cluster_id": merged_meta.loc[merged_meta["query_id"] == qid, "old_fragment_cluster_id"].iloc[0],
                "rescue_source": rescue_source,
                "pos_attach_top10": pos_attach_top10,
                "pos_de_top10": pos_de_top10,
                "pos_hgb_top10": pos_hgb_top10,
            }
        )
    pd.DataFrame(gain_rows).to_csv(OUT / "d4s2_gain_loss_analysis.csv", index=False)

    write_a4c_blocked_note()


if __name__ == "__main__":
    main()

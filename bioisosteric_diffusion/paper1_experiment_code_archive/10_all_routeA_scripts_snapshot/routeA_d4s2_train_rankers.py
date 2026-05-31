#!/usr/bin/env python3
"""Route-A D4S2 train candidate rerankers and select on validation."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from routeA_d4s2_common import (
    B0_VAL_METRICS_PATH,
    OUT,
    ART,
    LinearListwise,
    MLPListwise,
    feature_columns_for_sets,
    fit_histgb_candidate_model,
    fit_torch_group_model,
    load_group_arrays,
    metrics_from_score_matrix,
    predict_histgb_group_model,
    predict_torch_group_model,
)


def subset_mask(meta: pd.DataFrame, subset_name: str):
    if subset_name == "all_val":
        return np.ones(len(meta), dtype=bool)
    if subset_name == "hard_top10_miss":
        return meta["hard_top10_miss_flag"].astype(int).to_numpy() == 1
    if subset_name == "rare_replacement":
        return meta["replacement_frequency_bin"].eq("rare_replacement").to_numpy()
    if subset_name == "frequent_replacement":
        return meta["replacement_frequency_bin"].eq("frequent_replacement").to_numpy()
    raise KeyError(subset_name)


def metric_rows_for_subset(model_id: str, feature_set: str, scores: np.ndarray, y_val: np.ndarray, meta: pd.DataFrame):
    rows = []
    for subset_name in ["all_val", "hard_top10_miss", "rare_replacement", "frequent_replacement"]:
        mask = subset_mask(meta, subset_name)
        if int(mask.sum()) == 0:
            continue
        metrics = metrics_from_score_matrix(scores[mask], y_val[mask])
        rows.append(
            {
                "model_id": model_id,
                "feature_set": feature_set,
                "subset": subset_name,
                "N_queries": int(mask.sum()),
                "Top1": metrics["Top1"],
                "Top5": metrics["Top5"],
                "Top10": metrics["Top10"],
                "Top20": metrics["Top20"],
                "Top50": metrics["Top50"],
                "MRR": metrics["MRR"],
                "NDCG@10": metrics["NDCG@10"],
            }
        )
    return rows


def main():
    X_train, y_train, _train_qids, _train_meta = load_group_arrays("train")
    X_val, y_val, val_qids, val_meta = load_group_arrays("val")
    val_eval_mask = val_meta["target_any_seen_vocab"].astype(int).to_numpy() == 1
    X_val_eval = X_val[val_eval_mask]
    y_val_np = np.asarray(y_val)[val_eval_mask]
    val_qids = [qid for qid, keep in zip(val_qids, val_eval_mask) if keep]
    val_meta = val_meta.loc[val_eval_mask].reset_index(drop=True)

    feature_sets = feature_columns_for_sets()
    all_cols = feature_sets["F3_rank_score_frequency_chemistry"]
    col_to_idx = {name: idx for idx, name in enumerate(all_cols)}

    b0_val = pd.read_csv(B0_VAL_METRICS_PATH)
    b0_borda_top10 = float(
        b0_val.loc[b0_val["method"] == "Borda(DE,HGB)", "Top10"].iloc[0]
    )
    b0_borda_mrr = float(
        b0_val.loc[b0_val["method"] == "Borda(DE,HGB)", "MRR"].iloc[0]
    )

    model_specs = [
        {
            "model_id": "M3_histgb_F2",
            "feature_set": "F2_rank_score_frequency",
            "kind": "histgb",
            "complexity": 2,
        },
        {
            "model_id": "M4_linear_F0",
            "feature_set": "F0_rank_only",
            "kind": "linear",
            "complexity": 0,
        },
        {
            "model_id": "M5_mlp_F0",
            "feature_set": "F0_rank_only",
            "kind": "mlp",
            "hidden": 32,
            "dropout": 0.0,
            "lr": 1e-3,
            "complexity": 1,
        },
        {
            "model_id": "M5_mlp_F2",
            "feature_set": "F2_rank_score_frequency",
            "kind": "mlp",
            "hidden": 64,
            "dropout": 0.05,
            "lr": 8e-4,
            "complexity": 2,
        },
        {
            "model_id": "M5_mlp_F3",
            "feature_set": "F3_rank_score_frequency_chemistry",
            "kind": "mlp",
            "hidden": 96,
            "dropout": 0.1,
            "lr": 8e-4,
            "complexity": 3,
        },
    ]

    training_rows = []
    val_rows = []
    comparison_rows = []
    manifest_rows = []

    best = None

    for spec in model_specs:
        cols = [col_to_idx[name] for name in feature_sets[spec["feature_set"]]]
        t0 = time.time()
        if spec["kind"] == "histgb":
            model = fit_histgb_candidate_model(X_train, y_train, cols)
            scores = predict_histgb_group_model(model, X_val_eval, cols)
            artifact_path = ART / f"{spec['model_id']}.pkl"
            with open(artifact_path, "wb") as handle:
                pickle.dump({"model": model, "cols": cols, "feature_names": feature_sets[spec["feature_set"]]}, handle)
            train_meta = {"artifact_type": "pickle_histgb"}
        elif spec["kind"] == "linear":
            model = LinearListwise(len(cols))
            model, mean, std, _ = fit_torch_group_model(
                model, X_train, y_train, X_val_eval, y_val_np, cols, lr=1e-3, epochs=8
            )
            scores = predict_torch_group_model(model, X_val_eval, cols, mean, std)
            artifact_path = ART / f"{spec['model_id']}.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "cols": cols,
                    "feature_names": feature_sets[spec["feature_set"]],
                    "mean": mean,
                    "std": std,
                    "model_kind": "linear",
                },
                artifact_path,
            )
            train_meta = {"artifact_type": "torch_linear"}
        else:
            model = MLPListwise(len(cols), hidden=spec["hidden"], dropout=spec["dropout"])
            model, mean, std, _ = fit_torch_group_model(
                model,
                X_train,
                y_train,
                X_val_eval,
                y_val_np,
                cols,
                lr=spec["lr"],
                epochs=8,
            )
            scores = predict_torch_group_model(model, X_val_eval, cols, mean, std)
            artifact_path = ART / f"{spec['model_id']}.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "cols": cols,
                    "feature_names": feature_sets[spec["feature_set"]],
                    "mean": mean,
                    "std": std,
                    "model_kind": "mlp",
                    "hidden": spec["hidden"],
                    "dropout": spec["dropout"],
                },
                artifact_path,
            )
            train_meta = {"artifact_type": "torch_mlp"}
        runtime = round(time.time() - t0, 2)
        metrics = metrics_from_score_matrix(scores, y_val_np)
        training_rows.append(
            {
                "model_id": spec["model_id"],
                "feature_set": spec["feature_set"],
                "model_kind": spec["kind"],
                "runtime_sec": runtime,
                "status": "PASS",
                "config": json.dumps(spec, ensure_ascii=False),
            }
        )
        val_rows.extend(metric_rows_for_subset(spec["model_id"], spec["feature_set"], scores, y_val_np, val_meta))
        comparison_rows.append(
            {
                "model_id": spec["model_id"],
                "feature_set": spec["feature_set"],
                "Top1": metrics["Top1"],
                "Top5": metrics["Top5"],
                "Top10": metrics["Top10"],
                "Top20": metrics["Top20"],
                "Top50": metrics["Top50"],
                "MRR": metrics["MRR"],
                "NDCG@10": metrics["NDCG@10"],
                "beats_b0_borda_top10": int(metrics["Top10"] > b0_borda_top10),
                "complexity": spec["complexity"],
            }
        )
        manifest_rows.append(
            {
                "model_id": spec["model_id"],
                "artifact_path": str(artifact_path),
                "feature_set": spec["feature_set"],
                "model_kind": spec["kind"],
                **train_meta,
            }
        )
        candidate = {
            "model_id": spec["model_id"],
            "feature_set": spec["feature_set"],
            "Top10": metrics["Top10"],
            "MRR": metrics["MRR"],
            "Top5": metrics["Top5"],
            "complexity": spec["complexity"],
        }
        if best is None:
            best = candidate
        else:
            better = False
            if candidate["Top10"] > best["Top10"] + 1e-12:
                better = True
            elif abs(candidate["Top10"] - best["Top10"]) <= 1e-12:
                if candidate["MRR"] > best["MRR"] + 1e-12:
                    better = True
                elif abs(candidate["MRR"] - best["MRR"]) <= 1e-12:
                    if candidate["Top5"] > best["Top5"] + 1e-12:
                        better = True
                    elif abs(candidate["Top5"] - best["Top5"]) <= 1e-12:
                        if candidate["complexity"] < best["complexity"]:
                            better = True
            if better:
                best = candidate

    baseline_rows = []
    for method in ["Attachment_frequency", "DE", "HGB", "Borda(DE,HGB)"]:
        sub = b0_val.loc[b0_val["method"] == method].iloc[0]
        baseline_rows.append(
            {
                "model_id": method,
                "feature_set": "baseline",
                "Top1": float(sub["Top1"]),
                "Top5": float(sub["Top5"]),
                "Top10": float(sub["Top10"]),
                "Top20": float(sub["Top20"]),
                "Top50": float(sub["Top50"]),
                "MRR": float(sub["MRR"]),
                "NDCG@10": np.nan,
                "beats_b0_borda_top10": int(float(sub["Top10"]) > b0_borda_top10),
                "complexity": -1,
            }
        )

    pd.DataFrame(training_rows).to_csv(OUT / "d4s2_model_training_summary.csv", index=False)
    pd.DataFrame(val_rows).to_csv(OUT / "d4s2_model_val_metrics.csv", index=False)
    dump = {"models": manifest_rows}
    with open(OUT / "d4s2_model_artifacts_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(dump, handle, ensure_ascii=False, indent=2)
    comp_df = pd.DataFrame(baseline_rows + comparison_rows)
    comp_df.to_csv(OUT / "d4s2_validation_comparison.csv", index=False)

    winner_beats_borda = best["Top10"] > b0_borda_top10
    selected_path = OUT / "d4s2_selected_model.md"
    if winner_beats_borda:
        selected_path.write_text(
            "\n".join(
                [
                    "# D4S2 Selected Model",
                    "",
                    f"Selected model: **{best['model_id']}**",
                    f"- Feature set: `{best['feature_set']}`",
                    f"- Validation Top10: `{best['Top10']:.6f}`",
                    f"- Validation MRR: `{best['MRR']:.6f}`",
                    f"- Validation Top5: `{best['Top5']:.6f}`",
                    f"- B0 Borda Top10 baseline: `{b0_borda_top10:.6f}`",
                    "",
                    "Selection rule:",
                    "- Primary: val Top10",
                    "- Tie-break 1: val MRR",
                    "- Tie-break 2: val Top5",
                    "- Tie-break 3: simpler model",
                ]
            ),
            encoding="utf-8",
        )
    else:
        selected_path.write_text(
            "\n".join(
                [
                    "# D4S2 Selected Model",
                    "",
                    "No candidate reranker beat the official B0 Borda baseline on validation.",
                    f"- Best candidate: `{best['model_id']}` with Top10 `{best['Top10']:.6f}`",
                    f"- Official B0 Borda Top10: `{b0_borda_top10:.6f}`",
                    f"- Official B0 Borda MRR: `{b0_borda_mrr:.6f}`",
                    "",
                    "Per protocol, blind evaluation as a new SOTA candidate should not proceed in this case.",
                ]
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

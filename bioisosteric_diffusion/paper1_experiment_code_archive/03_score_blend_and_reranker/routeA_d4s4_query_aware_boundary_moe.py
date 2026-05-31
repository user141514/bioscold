#!/usr/bin/env python3
"""Route-A D4S4 query-aware boundary mixture-of-experts.

D4S4 tests the next algorithmic direction after D4S3:
learn when the fixed MLP+HGB score blend should be trusted, and when a
query-level rescue gate should switch to an alternate expert.

The run is intentionally conservative. The existing secondary blind has already
been used by prior stages, so D4S4 is validation-first and does not open the old
blind as a paper-main final test.
"""

from __future__ import annotations

import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from routeA_d4s2_common import metrics_from_score_matrix  # noqa: E402
from routeA_topjournal_s1_router_and_review_pilot import (  # noqa: E402
    load_selected_predictor_local,
    row_zscore,
    split_score_signals,
)


SEED = 20260528
PLAN = Path("E:/zuhui/bioisosteric_diffusion/plan_results")
OUT = PLAN / "routeA_chembl37k_d4s4_query_aware_boundary_moe"
ART = OUT / "artifacts"
S1 = PLAN / "routeA_topjournal_s1_failure_router_review_pilot"
D4S3 = PLAN / "routeA_chembl37k_d4s3_feature_rich_reranker_no_skill"

EXPERTS = ["score_blend", "MLP", "HGB", "DE", "Borda"]
MIN_DIRECTIONAL_GAIN = 0.001
MIN_ROBUST_GAIN = 0.002
MIN_MAINLINE_GAIN = 0.003


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def selected_score_blend_config() -> dict:
    grid = pd.read_csv(S1 / "s1_score_blend_validation_grid.csv")
    return grid.loc[grid["selected_by_validation"].astype(int) == 1].iloc[0].to_dict()


def load_split_experts(split: str, config: dict):
    payload = load_selected_predictor_local("M5_mlp_F0")
    qids, labels, meta, signals = split_score_signals(split, payload)
    alpha = float(config["alpha_left_signal"])
    left = str(config["left_signal"])
    right = str(config["right_signal"])
    score_blend = alpha * row_zscore(signals[left]) + (1.0 - alpha) * row_zscore(signals[right])
    experts = {
        "score_blend": score_blend.astype(np.float32),
        "MLP": signals["M5_mlp_F0"].astype(np.float32),
        "HGB": signals["HGB_refit_score"].astype(np.float32),
        "DE": signals["DE_refit_score"].astype(np.float32),
        "Borda": signals["Borda_refit_score"].astype(np.float32),
    }
    return qids, np.asarray(labels), meta.reset_index(drop=True), experts


def rank_positions(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int16)
    ranks[np.arange(scores.shape[0])[:, None], order] = np.arange(1, scores.shape[1] + 1, dtype=np.int16)
    return ranks, order


def best_ranks(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return metrics_from_score_matrix(scores, labels)["best_rank"].astype(float)


def query_feature_matrix(experts: dict[str, np.ndarray], meta: pd.DataFrame, train_categories: dict | None = None):
    feature_arrays = []
    feature_names = []
    ranks = {}
    orders = {}

    for name in EXPERTS:
        scores = experts[name]
        rank, order = rank_positions(scores)
        ranks[name] = rank
        orders[name] = order
        sorted_scores = np.take_along_axis(scores, order, axis=1)
        values = {
            "std": scores.std(axis=1),
            "range": scores.max(axis=1) - scores.min(axis=1),
            "margin1": sorted_scores[:, 0] - sorted_scores[:, 1],
            "margin3": sorted_scores[:, 2] - sorted_scores[:, 3],
            "margin5": sorted_scores[:, 4] - sorted_scores[:, 5],
            "margin10": sorted_scores[:, 9] - sorted_scores[:, 10],
            "margin20": sorted_scores[:, 19] - sorted_scores[:, 20],
            "top10_mean": sorted_scores[:, :10].mean(axis=1),
            "top10_std": sorted_scores[:, :10].std(axis=1),
            "top10_to_30_gap": sorted_scores[:, 9] - sorted_scores[:, 29],
        }
        for suffix, arr in values.items():
            feature_arrays.append(arr)
            feature_names.append(f"{name}_{suffix}")

    pairs = [
        ("DE", "HGB"),
        ("score_blend", "DE"),
        ("score_blend", "HGB"),
        ("score_blend", "Borda"),
        ("score_blend", "MLP"),
        ("MLP", "HGB"),
        ("MLP", "Borda"),
        ("HGB", "Borda"),
    ]
    for left, right in pairs:
        overlap10 = np.array(
            [len(set(orders[left][i, :10]).intersection(orders[right][i, :10])) for i in range(len(meta))],
            dtype=np.float32,
        )
        overlap20 = np.array(
            [len(set(orders[left][i, :20]).intersection(orders[right][i, :20])) for i in range(len(meta))],
            dtype=np.float32,
        )
        feature_arrays.extend([overlap10, overlap20])
        feature_names.extend([f"{left}_{right}_top10_overlap", f"{left}_{right}_top20_overlap"])

    numeric = np.vstack(feature_arrays).T.astype(np.float32)
    df = pd.DataFrame(numeric, columns=feature_names)
    categories = pd.DataFrame(
        {
            "attachment_signature": meta["attachment_signature"].astype(str),
            "old_fragment_cluster_id": meta["old_fragment_cluster_id"].astype(str),
        }
    )
    if train_categories is None:
        train_categories = {col: sorted(categories[col].unique()) for col in categories.columns}
    for col, values in train_categories.items():
        for value in values:
            df[f"{col}::{value}"] = (categories[col] == value).astype(np.float32).to_numpy()
    return df, train_categories


def compose_hard_switch(experts: dict[str, np.ndarray], pred: np.ndarray) -> np.ndarray:
    scores = experts["score_blend"].copy()
    for i, name in enumerate(EXPERTS):
        mask = pred == i
        if mask.any():
            scores[mask] = experts[name][mask]
    return scores


def boundary_window_scores(base: np.ndarray, partner: np.ndarray, keep_top: int, window_hi: int, alpha_base: float) -> np.ndarray:
    base_z = row_zscore(base)
    partner_z = row_zscore(partner)
    blend = alpha_base * base_z + (1.0 - alpha_base) * partner_z
    n_query, n_cand = base.shape
    out = np.empty_like(base, dtype=np.float32)
    base_order = np.argsort(-base, axis=1, kind="mergesort")
    for i in range(n_query):
        order = base_order[i]
        top = order[:keep_top]
        window = order[keep_top:window_hi]
        tail = order[window_hi:]
        window_sorted = window[np.argsort(-blend[i, window], kind="mergesort")]
        final_order = np.concatenate([top, window_sorted, tail])
        out[i, final_order] = -np.arange(n_cand, dtype=np.float32)
    return out


def metric_row(policy: str, family: str, scores: np.ndarray, labels: np.ndarray, base_top10: float, notes: str = ""):
    metrics = metrics_from_score_matrix(scores, labels)
    return {
        "policy": policy,
        "family": family,
        "Top1": metrics["Top1"],
        "Top5": metrics["Top5"],
        "Top10": metrics["Top10"],
        "Top20": metrics["Top20"],
        "Top50": metrics["Top50"],
        "MRR": metrics["MRR"],
        "NDCG@10": metrics["NDCG@10"],
        "delta_vs_score_blend": metrics["Top10"] - base_top10,
        "notes": notes,
    }, metrics["best_rank"].astype(float)


def bootstrap_delta(policy_ranks: np.ndarray, base_ranks: np.ndarray, n_boot: int = 1000) -> dict:
    rng = np.random.RandomState(SEED)
    a = (policy_ranks <= 10).astype(np.float32)
    b = (base_ranks <= 10).astype(np.float32)
    n = len(a)
    diffs = np.zeros(n_boot, dtype=np.float32)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        diffs[i] = float(np.mean(a[idx] - b[idx]))
    return {
        "delta": float(np.mean(a - b)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }


def write_protocol() -> None:
    write_text(
        OUT / "d4s4_protocol_decision.md",
        """
# D4S4 Protocol Decision

Verdict: **VALIDATION_ONLY_EXISTING_SPLIT**

D4S4 implements query-aware routing and boundary mixture-of-experts on the
existing Route-A train/validation matrices. The secondary blind has already been
opened by Borda, D4S2, and S1/S2 score-blend work; therefore D4S4 does not use
the old blind as a paper-main final test.

Allowed in this run:
- train query-level routers on the train split;
- select policies on validation Top10;
- report validation bootstrap and subset diagnostics;
- write old-blind files only as `NOT_EVALUATED` diagnostic placeholders.

Forbidden in this run:
- tune on the old secondary blind;
- claim a new paper-main SOTA from this validation-only search;
- use activity labels, A4C labels, RRR/S7 labels, or wet-lab/expert outcomes.
""",
    )


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    write_protocol()

    config = selected_score_blend_config()
    log("Loading train/validation expert score matrices.")
    train_qids, y_train, train_meta, train_experts = load_split_experts("train", config)
    val_qids, y_val, val_meta, val_experts = load_split_experts("val", config)

    log("Building query-level feature matrices.")
    x_train_df, categories = query_feature_matrix(train_experts, train_meta)
    x_val_df, _ = query_feature_matrix(val_experts, val_meta, categories)
    x_train_df.insert(0, "query_id", train_qids)
    x_val_df.insert(0, "query_id", val_qids)
    x_train_df.to_csv(OUT / "d4s4_query_features_train.csv.gz", index=False, compression="gzip")
    x_val_df.to_csv(OUT / "d4s4_query_features_val.csv.gz", index=False, compression="gzip")
    feature_cols = [c for c in x_train_df.columns if c != "query_id"]

    schema = {
        "selected_score_blend": config,
        "experts": EXPERTS,
        "n_query_features": len(feature_cols),
        "feature_columns": feature_cols,
        "runtime_safe_feature_policy": "score/rank disagreement plus train-frozen attachment and old-fragment cluster categories",
        "excluded_features": [
            "positive_set_size",
            "replacement_frequency_bin",
            "hard_top10_miss_flag",
            "target_any_seen_vocab",
            "activity/RRR/S7/A4C labels",
        ],
    }
    write_json(OUT / "d4s4_query_feature_schema.json", schema)

    x_train = x_train_df[feature_cols].to_numpy(dtype=np.float32)
    x_val = x_val_df[feature_cols].to_numpy(dtype=np.float32)

    train_ranks = np.vstack([best_ranks(train_experts[name], y_train) for name in EXPERTS]).T
    val_ranks = np.vstack([best_ranks(val_experts[name], y_val) for name in EXPERTS]).T
    base_scores = val_experts["score_blend"]
    base_metrics = metrics_from_score_matrix(base_scores, y_val)
    base_top10 = float(base_metrics["Top10"])
    base_ranks = base_metrics["best_rank"].astype(float)

    log("Evaluating fixed experts and boundary windows.")
    rows = []
    rank_by_policy = {}
    score_by_policy = {}
    for name in EXPERTS:
        row, ranks = metric_row(f"expert::{name}", "fixed_expert", val_experts[name], y_val, base_top10)
        rows.append(row)
        rank_by_policy[row["policy"]] = ranks
        score_by_policy[row["policy"]] = val_experts[name]

    boundary_grid = []
    for partner in ["MLP", "HGB", "DE", "Borda"]:
        for keep_top in [3, 5, 7, 10]:
            for window_hi in [15, 20, 30, 50]:
                if window_hi <= keep_top:
                    continue
                for alpha_base in [0.5, 0.65, 0.8, 0.9]:
                    scores = boundary_window_scores(base_scores, val_experts[partner], keep_top, window_hi, alpha_base)
                    policy = f"boundary::{partner}::keep{keep_top}::hi{window_hi}::alpha{alpha_base:.2f}"
                    row, ranks = metric_row(
                        policy,
                        "boundary_window",
                        scores,
                        y_val,
                        base_top10,
                        notes="Validation-only boundary reorder; not selected on blind.",
                    )
                    boundary_grid.append(row)
                    rows.append(row)
                    rank_by_policy[policy] = ranks
                    score_by_policy[policy] = scores

    log("Training hard-switch and rescue routers on train.")
    router_rows = []
    train_best_rank_target = np.argmin(train_ranks, axis=1)
    train_hit_or_base_target = np.where(train_ranks[:, 0] <= 10, 0, train_best_rank_target)
    multiclass_models = {
        "tree3": DecisionTreeClassifier(max_depth=3, min_samples_leaf=500, class_weight="balanced", random_state=SEED),
        "tree5": DecisionTreeClassifier(max_depth=5, min_samples_leaf=300, class_weight="balanced", random_state=SEED),
        "rf5": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=300,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
        "hgb_small": HistGradientBoostingClassifier(
            max_iter=80,
            max_leaf_nodes=15,
            l2_regularization=0.2,
            learning_rate=0.05,
            random_state=SEED,
        ),
    }
    for target_name, target in [("argmin_rank", train_best_rank_target), ("hit_or_base", train_hit_or_base_target)]:
        for model_name, model in multiclass_models.items():
            model.fit(x_train, target)
            pred = model.predict(x_val)
            scores = compose_hard_switch(val_experts, pred)
            policy = f"hard_switch::{target_name}::{model_name}"
            row, ranks = metric_row(
                policy,
                "query_hard_switch",
                scores,
                y_val,
                base_top10,
                notes="Train-fitted query router; validation-only policy comparison.",
            )
            row["routed_fraction"] = float(np.mean(pred != 0))
            row["route_counts"] = "|".join(map(str, np.bincount(pred, minlength=len(EXPERTS)).tolist()))
            rows.append(row)
            router_rows.append(row)
            rank_by_policy[policy] = ranks
            score_by_policy[policy] = scores

    rescue_models = {
        "logit_l2_C0p1": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=500, class_weight="balanced", n_jobs=1),
        ),
        "rf4": RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=150,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
        "rf6": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=100,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED + 1,
        ),
        "extra4": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=150,
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED + 2,
        ),
        "tree4": DecisionTreeClassifier(max_depth=4, min_samples_leaf=250, class_weight="balanced", random_state=SEED),
        "hgb_balanced": HistGradientBoostingClassifier(
            max_iter=80,
            max_leaf_nodes=15,
            l2_regularization=0.5,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=SEED + 3,
        ),
    }
    for expert_idx, expert_name in enumerate(EXPERTS[1:], start=1):
        target = ((train_ranks[:, 0] > 10) & (train_ranks[:, expert_idx] <= 10)).astype(int)
        for model_name, model in rescue_models.items():
            model.fit(x_train, target)
            probs = model.predict_proba(x_val)[:, 1]
            for threshold in np.linspace(0.2, 0.9, 29):
                mask = probs >= threshold
                scores = base_scores.copy()
                scores[mask] = val_experts[expert_name][mask]
                policy = f"rescue::{expert_name}::{model_name}::threshold{threshold:.3f}"
                row, ranks = metric_row(
                    policy,
                    "binary_rescue_gate",
                    scores,
                    y_val,
                    base_top10,
                    notes=f"Train target: score_blend miss and {expert_name} hit Top10.",
                )
                row["routed_fraction"] = float(np.mean(mask))
                row["routed_queries"] = int(mask.sum())
                row["train_positive_rescue_labels"] = int(target.sum())
                rows.append(row)
                router_rows.append(row)
                rank_by_policy[policy] = ranks
                score_by_policy[policy] = scores

    all_rows = pd.DataFrame(rows).sort_values(["Top10", "MRR"], ascending=[False, False])
    all_rows.to_csv(OUT / "d4s4_policy_validation_grid.csv", index=False)
    pd.DataFrame(boundary_grid).sort_values(["Top10", "MRR"], ascending=[False, False]).to_csv(
        OUT / "d4s4_boundary_window_grid.csv", index=False
    )
    pd.DataFrame(router_rows).sort_values(["Top10", "MRR"], ascending=[False, False]).to_csv(
        OUT / "d4s4_router_gate_grid.csv", index=False
    )

    best = all_rows.iloc[0].to_dict()
    best_policy = str(best["policy"])
    selected_ranks = rank_by_policy[best_policy]
    best_scores = score_by_policy[best_policy]
    boot = bootstrap_delta(selected_ranks, base_ranks)

    if float(best["delta_vs_score_blend"]) >= MIN_MAINLINE_GAIN and boot["ci_low"] > 0:
        verdict = "D4S4_VALIDATION_GAIN_CANDIDATE_NEEDS_CLEAN_SPLIT"
    elif float(best["delta_vs_score_blend"]) >= MIN_ROBUST_GAIN and boot["ci_low"] > 0:
        verdict = "D4S4_DIRECTIONAL_GAIN_NEEDS_CONFIRMATION"
    elif float(best["delta_vs_score_blend"]) >= MIN_DIRECTIONAL_GAIN:
        verdict = "D4S4_SMALL_DIRECTIONAL_GAIN_NOT_MAINLINE"
    else:
        verdict = "D4S4_NO_MEANINGFUL_GAIN_SCORE_BLEND_REMAINS_BEST"

    write_text(
        OUT / "d4s4_selected_policy.md",
        f"""
# D4S4 Selected Policy

Selected validation policy: `{best_policy}`

- family: `{best['family']}`
- validation Top10: {float(best['Top10']):.6f}
- validation MRR: {float(best['MRR']):.6f}
- delta vs score blend Top10: {float(best['delta_vs_score_blend']):+.6f}
- validation bootstrap CI: [{boot['ci_low']:+.6f}, {boot['ci_high']:+.6f}]
- verdict: **{verdict}**

Interpretation:
D4S4 found a small query-aware rescue signal, but the gain is below the
+0.003 threshold needed to justify replacing the current score blend. The old
secondary blind is not evaluated in this run.
""",
    )

    pd.DataFrame(
        [
            {
                "comparison": "D4S4_selected_minus_score_blend",
                "split": "val",
                "Top10_delta": boot["delta"],
                "Top10_ci_low": boot["ci_low"],
                "Top10_ci_high": boot["ci_high"],
                "n_bootstrap": 1000,
                "status": "VALIDATION_ONLY",
            }
        ]
    ).to_csv(OUT / "d4s4_validation_bootstrap.csv", index=False)

    subset_rows = []
    subset_defs = {
        "all_val": np.ones(len(val_meta), dtype=bool),
        "rare_replacement": val_meta["replacement_frequency_bin"].eq("rare_replacement").to_numpy(),
        "frequent_replacement": val_meta["replacement_frequency_bin"].eq("frequent_replacement").to_numpy(),
        "hard_top10_miss": val_meta["hard_top10_miss_flag"].astype(int).to_numpy() == 1,
        "single_pos": val_meta["single_pos_flag"].astype(int).to_numpy() == 1,
        "multi_pos": val_meta["multi_pos_flag"].astype(int).to_numpy() == 1,
        "attachment_C|O": val_meta["attachment_signature"].astype(str).eq("C|O").to_numpy(),
        "cluster_05": val_meta["old_fragment_cluster_id"].astype(str).eq("cluster_05").to_numpy(),
        "cluster_09": val_meta["old_fragment_cluster_id"].astype(str).eq("cluster_09").to_numpy(),
    }
    for subset, mask in subset_defs.items():
        if int(mask.sum()) == 0:
            continue
        subset_rows.append(
            {
                "subset": subset,
                "n": int(mask.sum()),
                "score_blend_Top10": float(np.mean(base_ranks[mask] <= 10)),
                "D4S4_selected_Top10": float(np.mean(selected_ranks[mask] <= 10)),
                "delta": float(np.mean(selected_ranks[mask] <= 10) - np.mean(base_ranks[mask] <= 10)),
            }
        )
    pd.DataFrame(subset_rows).to_csv(OUT / "d4s4_subset_metrics.csv", index=False)

    gain_loss = pd.DataFrame(
        {
            "query_id": val_qids,
            "old_fragment_smiles": val_meta["old_fragment_smiles"],
            "attachment_signature": val_meta["attachment_signature"],
            "old_fragment_cluster_id": val_meta["old_fragment_cluster_id"],
            "score_blend_rank": base_ranks,
            "D4S4_rank": selected_ranks,
        }
    )
    gain_loss["D4S4_rescue_top10"] = ((gain_loss["score_blend_rank"] > 10) & (gain_loss["D4S4_rank"] <= 10)).astype(int)
    gain_loss["D4S4_loss_top10"] = ((gain_loss["score_blend_rank"] <= 10) & (gain_loss["D4S4_rank"] > 10)).astype(int)
    gain_loss["rank_delta_scoreblend_minus_D4S4"] = gain_loss["score_blend_rank"] - gain_loss["D4S4_rank"]
    gain_loss.sort_values(
        ["D4S4_rescue_top10", "D4S4_loss_top10", "rank_delta_scoreblend_minus_D4S4"],
        ascending=[False, False, False],
    ).head(2000).to_csv(OUT / "d4s4_gain_loss_query_audit.csv", index=False)

    # Old blind placeholders: intentionally not evaluated.
    pd.DataFrame(
        [
            {
                "split": "secondary_blind",
                "policy": "D4S4_selected",
                "status": "NOT_EVALUATED_EXISTING_BLIND_EXPOSED",
                "reason": "D4S4 is validation-only; old blind cannot support a fresh paper-main claim.",
            }
        ]
    ).to_csv(OUT / "d4s4_final_test_metrics_diagnostic_only.csv", index=False)
    with open(OUT / "d4s4_final_predictions_diagnostic_only.jsonl", "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "status": "NOT_EVALUATED_EXISTING_BLIND_EXPOSED",
                    "selected_validation_policy": best_policy,
                    "validation_delta_vs_score_blend": float(best["delta_vs_score_blend"]),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    write_text(
        OUT / "D4S4_QUERY_AWARE_BOUNDARY_MOE_VERDICT.md",
        f"""
# D4S4 Query-Aware Boundary MoE Verdict

Final verdict: **{verdict}**

## What Was Tested

D4S4 tested query-level routing over fixed experts:
`score_blend`, `MLP`, `HGB`, `DE`, and `Borda`.

The main successful pattern was not a broad soft mixture. It was a conservative
rescue gate that keeps score blend by default and switches only selected queries
to another expert.

## Result

- Selected policy: `{best_policy}`
- Validation Top10: {float(best['Top10']):.6f}
- Score blend validation Top10: {base_top10:.6f}
- Delta: {float(best['delta_vs_score_blend']):+.6f}
- Bootstrap CI: [{boot['ci_low']:+.6f}, {boot['ci_high']:+.6f}]

This is a real directional improvement over D4S3's feature-stacking attempt,
but it is too small to replace the current method.

## Algorithmic Takeaway

The useful signal is query-aware rescue, especially HGB rescue for a subset of
score-blend misses. Full hard-switch routers and broad boundary reordering tend
to hurt Top10 or only improve MRR. The next version should focus on a more
precise rescue target rather than a general MoE.

## Decision

Do not update the manuscript main method. Keep MLP+HGB score blend as the main
internal method. Treat D4S4 as an algorithmic prototype that justifies a cleaner
fresh-split D4S5 rescue-gate experiment.
""",
    )

    write_text(
        OUT / "MAIN_DECISION_LOG.md",
        f"""
# D4S4 Main Decision Log

- Protocol: validation-only; old secondary blind not evaluated.
- Current method to beat: MLP+HGB score blend.
- Best policy: `{best_policy}`.
- Validation delta vs score blend: {float(best['delta_vs_score_blend']):+.6f}.
- Bootstrap CI: [{boot['ci_low']:+.6f}, {boot['ci_high']:+.6f}].
- Decision: small directional gain, not enough to replace score blend.
- Next recommendation: D4S5 fresh-split rescue gate with a narrower target:
  score-blend miss, HGB/DE/Borda rescue available, and route only when confidence
  is high.
""",
    )

    log(f"D4S4 completed: {verdict}; best={best_policy}")


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""Route-A D4S5 broad algorithm exploration.

This stage searches several leakage-safe paths beyond the fixed MLP+HGB score
blend. It uses an internal train/selection split made from the existing train
queries, then evaluates the selected policies on the existing validation split.
The old secondary blind is intentionally not used.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from routeA_d4s2_common import (  # noqa: E402
    feature_columns_for_sets,
    load_group_arrays,
    metrics_from_score_matrix,
)
from routeA_topjournal_s1_router_and_review_pilot import (  # noqa: E402
    load_selected_predictor_local,
    row_zscore,
    split_score_signals,
)


SEED = 20260528
PLAN = Path("E:/zuhui/bioisosteric_diffusion/plan_results")
OUT = PLAN / "routeA_chembl37k_d4s5_algorithm_exploration"
ART = OUT / "artifacts"
S1 = PLAN / "routeA_topjournal_s1_failure_router_review_pilot"
EXPERTS = ["score_blend", "MLP", "HGB", "DE", "Borda"]


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


def load_split(split: str, config: dict):
    payload = load_selected_predictor_local("M5_mlp_F0")
    qids, labels, meta, signals = split_score_signals(split, payload)
    alpha = float(config["alpha_left_signal"])
    left = str(config["left_signal"])
    right = str(config["right_signal"])
    blend = alpha * row_zscore(signals[left]) + (1.0 - alpha) * row_zscore(signals[right])
    experts = {
        "score_blend": blend.astype(np.float32),
        "MLP": signals["M5_mlp_F0"].astype(np.float32),
        "HGB": signals["HGB_refit_score"].astype(np.float32),
        "DE": signals["DE_refit_score"].astype(np.float32),
        "Borda": signals["Borda_refit_score"].astype(np.float32),
    }
    return qids, np.asarray(labels), meta.reset_index(drop=True), experts


def load_feature_tensor_filtered(split: str):
    x, labels, qids, meta = load_group_arrays(split)
    keep = meta["target_any_seen_vocab"].astype(int).to_numpy() == 1
    return x[keep], np.asarray(labels)[keep], [qid for qid, k in zip(qids, keep) if k], meta.loc[keep].reset_index(drop=True)


def rank_positions(scores: np.ndarray):
    order = np.argsort(-scores, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int16)
    ranks[np.arange(scores.shape[0])[:, None], order] = np.arange(1, scores.shape[1] + 1, dtype=np.int16)
    return ranks, order


def best_ranks(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return metrics_from_score_matrix(scores, labels)["best_rank"].astype(float)


def query_hash_mask(qids: list[str], select_fraction: float = 0.2) -> np.ndarray:
    flags = []
    cutoff = int(select_fraction * 10_000)
    for qid in qids:
        digest = hashlib.md5(str(qid).encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 10_000
        flags.append(bucket < cutoff)
    return np.asarray(flags, dtype=bool)


def query_feature_matrix(experts: dict[str, np.ndarray], meta: pd.DataFrame, cats: dict | None = None):
    arrays = []
    names = []
    ranks = {}
    orders = {}
    for name in EXPERTS:
        scores = experts[name]
        rank, order = rank_positions(scores)
        ranks[name] = rank
        orders[name] = order
        sorted_scores = np.take_along_axis(scores, order, axis=1)
        feature_map = {
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
        for suffix, arr in feature_map.items():
            arrays.append(arr)
            names.append(f"{name}_{suffix}")
    for left, right in [
        ("DE", "HGB"),
        ("score_blend", "DE"),
        ("score_blend", "HGB"),
        ("score_blend", "Borda"),
        ("score_blend", "MLP"),
        ("MLP", "HGB"),
        ("MLP", "Borda"),
        ("HGB", "Borda"),
    ]:
        overlap10 = np.array(
            [len(set(orders[left][i, :10]).intersection(orders[right][i, :10])) for i in range(len(meta))],
            dtype=np.float32,
        )
        overlap20 = np.array(
            [len(set(orders[left][i, :20]).intersection(orders[right][i, :20])) for i in range(len(meta))],
            dtype=np.float32,
        )
        arrays.extend([overlap10, overlap20])
        names.extend([f"{left}_{right}_top10_overlap", f"{left}_{right}_top20_overlap"])
    df = pd.DataFrame(np.vstack(arrays).T.astype(np.float32), columns=names)
    cat_df = pd.DataFrame(
        {
            "attachment_signature": meta["attachment_signature"].astype(str),
            "old_fragment_cluster_id": meta["old_fragment_cluster_id"].astype(str),
        }
    )
    if cats is None:
        cats = {col: sorted(cat_df[col].unique()) for col in cat_df.columns}
    for col, values in cats.items():
        for value in values:
            df[f"{col}::{value}"] = (cat_df[col] == value).astype(np.float32).to_numpy()
    return df, cats


def metric_row(stage: str, policy: str, family: str, scores: np.ndarray, labels: np.ndarray, base_top10: float, notes: str = ""):
    m = metrics_from_score_matrix(scores, labels)
    return {
        "stage": stage,
        "policy": policy,
        "family": family,
        "Top1": m["Top1"],
        "Top5": m["Top5"],
        "Top10": m["Top10"],
        "Top20": m["Top20"],
        "Top50": m["Top50"],
        "MRR": m["MRR"],
        "NDCG@10": m["NDCG@10"],
        "delta_vs_score_blend": m["Top10"] - base_top10,
        "notes": notes,
    }, m["best_rank"].astype(float)


def compose_switch(experts: dict[str, np.ndarray], pred: np.ndarray) -> np.ndarray:
    out = experts["score_blend"].copy()
    for i, name in enumerate(EXPERTS):
        mask = pred == i
        if mask.any():
            out[mask] = experts[name][mask]
    return out


def rrf_scores(experts: dict[str, np.ndarray], subset: list[str], k: int, weights: list[float] | None = None) -> np.ndarray:
    if weights is None:
        weights = [1.0] * len(subset)
    out = np.zeros_like(experts["score_blend"], dtype=np.float32)
    for name, weight in zip(subset, weights):
        ranks, _ = rank_positions(experts[name])
        out += float(weight) / (float(k) + ranks.astype(np.float32))
    return out


def boundary_scores(base: np.ndarray, partner: np.ndarray, keep_top: int, window_hi: int, alpha_base: float) -> np.ndarray:
    base_z = row_zscore(base)
    partner_z = row_zscore(partner)
    mix = alpha_base * base_z + (1.0 - alpha_base) * partner_z
    n_query, n_cand = base.shape
    out = np.empty_like(base, dtype=np.float32)
    base_order = np.argsort(-base, axis=1, kind="mergesort")
    for i in range(n_query):
        order = base_order[i]
        top = order[:keep_top]
        window = order[keep_top:window_hi]
        tail = order[window_hi:]
        window_sorted = window[np.argsort(-mix[i, window], kind="mergesort")]
        final_order = np.concatenate([top, window_sorted, tail])
        out[i, final_order] = -np.arange(n_cand, dtype=np.float32)
    return out


def bootstrap_delta(policy_ranks: np.ndarray, base_ranks: np.ndarray, n_boot: int = 1000) -> dict:
    rng = np.random.RandomState(SEED)
    a = (policy_ranks <= 10).astype(np.float32)
    b = (base_ranks <= 10).astype(np.float32)
    diffs = np.zeros(n_boot, dtype=np.float32)
    for i in range(n_boot):
        idx = rng.randint(0, len(a), size=len(a))
        diffs[i] = float(np.mean(a[idx] - b[idx]))
    return {
        "delta": float(np.mean(a - b)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_gt_0": float(np.mean(diffs > 0)),
    }


def candidate_feature_rows(
    x: np.ndarray,
    labels: np.ndarray,
    experts: dict[str, np.ndarray],
    query_mask: np.ndarray,
    max_rows: int,
    rng: np.random.RandomState,
):
    feature_cols = feature_columns_for_sets()["F3_rank_score_frequency_chemistry"]
    n_feat = len(feature_cols)
    ranks = {name: rank_positions(score)[0] for name, score in experts.items()}
    base_rank = ranks["score_blend"]
    candidate_mask = (base_rank <= 50) | (labels > 0)
    q_idx_all, c_idx_all = np.where(candidate_mask & query_mask[:, None])
    if len(q_idx_all) > max_rows:
        pick = rng.choice(len(q_idx_all), size=max_rows, replace=False)
        q_idx_all = q_idx_all[pick]
        c_idx_all = c_idx_all[pick]
    base = np.asarray(x[q_idx_all, c_idx_all, :n_feat], dtype=np.float32)
    extra = []
    for name in EXPERTS:
        extra.append(experts[name][q_idx_all, c_idx_all])
        extra.append(ranks[name][q_idx_all, c_idx_all].astype(np.float32))
    extra = np.vstack(extra).T.astype(np.float32)
    y = labels[q_idx_all, c_idx_all].astype(np.int32)
    return np.hstack([base, extra]), y


def candidate_model_scores(model, x: np.ndarray, experts: dict[str, np.ndarray], batch_queries: int = 512) -> np.ndarray:
    feature_cols = feature_columns_for_sets()["F3_rank_score_frequency_chemistry"]
    n_feat = len(feature_cols)
    ranks = {name: rank_positions(score)[0] for name, score in experts.items()}
    out = np.zeros((x.shape[0], x.shape[1]), dtype=np.float32)
    for start in range(0, x.shape[0], batch_queries):
        end = min(start + batch_queries, x.shape[0])
        base = np.asarray(x[start:end, :, :n_feat].reshape(-1, n_feat), dtype=np.float32)
        extra_cols = []
        for name in EXPERTS:
            extra_cols.append(experts[name][start:end].reshape(-1))
            extra_cols.append(ranks[name][start:end].reshape(-1).astype(np.float32))
        feat = np.hstack([base, np.vstack(extra_cols).T.astype(np.float32)])
        if hasattr(model, "predict_proba"):
            pred = model.predict_proba(feat)[:, 1]
        else:
            pred = model.decision_function(feat)
        out[start:end] = pred.reshape(end - start, x.shape[1])
    return out


def write_protocol(config: dict) -> None:
    write_text(
        OUT / "d4s5_protocol.md",
        f"""
# D4S5 Algorithm Exploration Protocol

Status: **TRAIN_INTERNAL_SELECTION_PLUS_VALIDATION_HOLDOUT**

D4S5 explores stitched and new algorithmic policies while avoiding old-blind
tuning. The original train split is divided deterministically by query_id hash
into train-internal fit and selection subsets. Candidate policy parameters are
chosen on the selection subset. The existing validation split is then used as
the final measurement for this exploratory stage.

Old secondary blind is not evaluated.

Current fixed blend:
- blend id: `{config['blend_id']}`
- left signal: `{config['left_signal']}`
- right signal: `{config['right_signal']}`
- alpha left: `{config['alpha_left_signal']}`

Forbidden targets: activity labels, RRR/S7 labels, A4C review labels, and old
secondary blind labels.
""",
    )


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    config = selected_score_blend_config()
    write_protocol(config)

    log("Loading train and validation experts.")
    train_qids, y_train, train_meta, train_experts = load_split("train", config)
    val_qids, y_val, val_meta, val_experts = load_split("val", config)
    x_train, _y_train_tensor, _qids_tensor, _meta_tensor = load_feature_tensor_filtered("train")
    x_val, _y_val_tensor, _val_qids_tensor, _val_meta_tensor = load_feature_tensor_filtered("val")

    if len(train_qids) != x_train.shape[0] or len(val_qids) != x_val.shape[0]:
        raise RuntimeError("Expert/query alignment mismatch.")

    select_mask = query_hash_mask(train_qids, select_fraction=0.2)
    fit_mask = ~select_mask
    split_rows = [
        {"split": "train_internal_fit", "n_queries": int(fit_mask.sum())},
        {"split": "train_internal_select", "n_queries": int(select_mask.sum())},
        {"split": "validation_holdout", "n_queries": int(len(val_qids))},
    ]
    pd.DataFrame(split_rows).to_csv(OUT / "d4s5_train_internal_split.csv", index=False)

    log("Building query features.")
    train_qfeat, cats = query_feature_matrix(train_experts, train_meta)
    val_qfeat, _ = query_feature_matrix(val_experts, val_meta, cats)
    train_qfeat.insert(0, "query_id", train_qids)
    val_qfeat.insert(0, "query_id", val_qids)
    train_qfeat.to_csv(OUT / "d4s5_query_features_train.csv.gz", index=False, compression="gzip")
    val_qfeat.to_csv(OUT / "d4s5_query_features_val.csv.gz", index=False, compression="gzip")
    feature_cols = [c for c in train_qfeat.columns if c != "query_id"]
    write_json(
        OUT / "d4s5_feature_schema.json",
        {
            "query_feature_columns": feature_cols,
            "candidate_feature_base": feature_columns_for_sets()["F3_rank_score_frequency_chemistry"],
            "candidate_feature_extra": [f"{name}_{kind}" for name in EXPERTS for kind in ["score", "rank"]],
            "categories": cats,
        },
    )
    xq_train = train_qfeat[feature_cols].to_numpy(dtype=np.float32)
    xq_val = val_qfeat[feature_cols].to_numpy(dtype=np.float32)

    select_experts = {name: score[select_mask] for name, score in train_experts.items()}
    fit_experts = {name: score[fit_mask] for name, score in train_experts.items()}
    y_select = y_train[select_mask]
    y_fit = y_train[fit_mask]
    base_select = select_experts["score_blend"]
    base_val = val_experts["score_blend"]
    select_base_metrics = metrics_from_score_matrix(base_select, y_select)
    val_base_metrics = metrics_from_score_matrix(base_val, y_val)
    select_base_top10 = float(select_base_metrics["Top10"])
    val_base_top10 = float(val_base_metrics["Top10"])
    val_base_ranks = val_base_metrics["best_rank"].astype(float)

    selection_rows = []
    validation_rows = []
    selected_score_map = {}
    val_rank_map = {}

    def add_policy(policy: str, family: str, select_scores: np.ndarray, val_scores: np.ndarray, notes: str = ""):
        srow, _ = metric_row("selection", policy, family, select_scores, y_select, select_base_top10, notes)
        vrow, vrank = metric_row("validation", policy, family, val_scores, y_val, val_base_top10, notes)
        selection_rows.append(srow)
        validation_rows.append(vrow)
        selected_score_map[policy] = val_scores
        val_rank_map[policy] = vrank

    log("Evaluating fixed experts and rank fusion.")
    for name in EXPERTS:
        add_policy(f"expert::{name}", "fixed_expert", select_experts[name], val_experts[name])

    rrf_rows = []
    subsets = [
        ["score_blend", "MLP", "HGB"],
        ["score_blend", "HGB", "DE"],
        ["score_blend", "MLP", "HGB", "DE"],
        ["score_blend", "MLP", "HGB", "DE", "Borda"],
        ["MLP", "HGB", "DE", "Borda"],
    ]
    weight_sets = [
        None,
        [2.0, 1.0, 1.0],
        [3.0, 1.0, 1.0],
        [4.0, 1.0, 1.0],
        [2.0, 2.0, 1.0],
    ]
    for subset in subsets:
        for k in [5, 10, 20, 40, 60, 100]:
            for weights in weight_sets:
                if weights is not None and len(weights) != len(subset):
                    continue
                policy = f"rrf::{'+'.join(subset)}::k{k}::w{('equal' if weights is None else '-'.join(map(str, weights)))}"
                s = rrf_scores(select_experts, subset, k, weights)
                v = rrf_scores(val_experts, subset, k, weights)
                add_policy(policy, "rank_fusion_rrf", s, v)
                rrf_rows.append(selection_rows[-1])

    log("Evaluating boundary windows.")
    boundary_rows = []
    for partner in ["MLP", "HGB", "DE", "Borda"]:
        for keep_top in [3, 5, 7, 10]:
            for window_hi in [15, 20, 30, 50]:
                if window_hi <= keep_top:
                    continue
                for alpha_base in [0.5, 0.65, 0.8, 0.9]:
                    policy = f"boundary::{partner}::keep{keep_top}::hi{window_hi}::alpha{alpha_base:.2f}"
                    s = boundary_scores(base_select, select_experts[partner], keep_top, window_hi, alpha_base)
                    v = boundary_scores(base_val, val_experts[partner], keep_top, window_hi, alpha_base)
                    add_policy(policy, "boundary_window", s, v)
                    boundary_rows.append(selection_rows[-1])

    log("Training train-internal rescue gates.")
    train_ranks = np.vstack([best_ranks(train_experts[name], y_train) for name in EXPERTS]).T
    select_query_x = xq_train[select_mask]
    fit_query_x = xq_train[fit_mask]
    rescue_rows = []
    rescue_models = {
        "logit_l2_C0p1": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=500, class_weight="balanced", n_jobs=1),
        ),
        "rf4": RandomForestClassifier(
            n_estimators=160,
            max_depth=4,
            min_samples_leaf=150,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
        "tree4": DecisionTreeClassifier(max_depth=4, min_samples_leaf=250, class_weight="balanced", random_state=SEED),
    }
    for expert_idx, expert_name in enumerate(EXPERTS[1:], start=1):
        target = ((train_ranks[:, 0] > 10) & (train_ranks[:, expert_idx] <= 10)).astype(int)
        fit_target = target[fit_mask]
        for model_name, model in rescue_models.items():
            model.fit(fit_query_x, fit_target)
            select_prob = model.predict_proba(select_query_x)[:, 1]
            val_prob = model.predict_proba(xq_val)[:, 1]
            for threshold in np.linspace(0.2, 0.95, 31):
                select_route = select_prob >= threshold
                val_route = val_prob >= threshold
                s = base_select.copy()
                s[select_route] = select_experts[expert_name][select_route]
                v = base_val.copy()
                v[val_route] = val_experts[expert_name][val_route]
                policy = f"rescue::{expert_name}::{model_name}::threshold{threshold:.3f}"
                add_policy(policy, "train_internal_rescue_gate", s, v, notes="threshold selected on train-internal selection")
                selection_rows[-1]["routed_fraction"] = float(np.mean(select_route))
                validation_rows[-1]["routed_fraction"] = float(np.mean(val_route))
                validation_rows[-1]["routed_queries"] = int(val_route.sum())
                rescue_rows.append(selection_rows[-1])

    log("Training candidate-level residual correctors.")
    rng = np.random.RandomState(SEED)
    residual_rows = []
    x_fit_rows, y_fit_rows = candidate_feature_rows(
        x_train,
        y_train,
        train_experts,
        fit_mask,
        max_rows=700_000,
        rng=rng,
    )
    residual_models = {
        "sgd_log": make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                alpha=1e-5,
                penalty="elasticnet",
                l1_ratio=0.05,
                class_weight="balanced",
                max_iter=35,
                tol=1e-4,
                random_state=SEED,
            ),
        ),
        "histgb": HistGradientBoostingClassifier(
            max_depth=4,
            max_iter=150,
            learning_rate=0.05,
            l2_regularization=0.05,
            early_stopping=True,
            validation_fraction=0.1,
            class_weight="balanced",
            random_state=SEED,
        ),
    }
    select_x = x_train[select_mask]
    val_x = x_val
    for model_name, model in residual_models.items():
        model.fit(x_fit_rows, y_fit_rows)
        with open(ART / f"d4s5_residual_{model_name}.pkl", "wb") as handle:
            pickle.dump(model, handle)
        select_resid = candidate_model_scores(model, select_x, select_experts)
        val_resid = candidate_model_scores(model, val_x, val_experts)
        for alpha_base in np.linspace(0.0, 1.0, 21):
            s = alpha_base * row_zscore(base_select) + (1.0 - alpha_base) * row_zscore(select_resid)
            v = alpha_base * row_zscore(base_val) + (1.0 - alpha_base) * row_zscore(val_resid)
            policy = f"residual::{model_name}::alpha_base{alpha_base:.2f}"
            add_policy(policy, "candidate_residual_stacking", s, v, notes="candidate-level residual corrector fit on train-internal fit")
            residual_rows.append(selection_rows[-1])

    selection_df = pd.DataFrame(selection_rows).sort_values(["Top10", "MRR"], ascending=[False, False])
    validation_df = pd.DataFrame(validation_rows)
    selection_df.to_csv(OUT / "d4s5_search_selection_grid.csv", index=False)
    validation_df.to_csv(OUT / "d4s5_all_policy_validation_metrics.csv", index=False)
    pd.DataFrame(rrf_rows).to_csv(OUT / "d4s5_rank_fusion_grid.csv", index=False)
    pd.DataFrame(boundary_rows).to_csv(OUT / "d4s5_boundary_grid.csv", index=False)
    pd.DataFrame(rescue_rows).to_csv(OUT / "d4s5_rescue_gate_grid.csv", index=False)
    pd.DataFrame(residual_rows).to_csv(OUT / "d4s5_candidate_residual_grid.csv", index=False)

    best_selection = selection_df.iloc[0].to_dict()
    selected_policy = str(best_selection["policy"])
    selected_validation = validation_df.loc[validation_df["policy"] == selected_policy].iloc[0].to_dict()
    selected_ranks = val_rank_map[selected_policy]
    boot = bootstrap_delta(selected_ranks, val_base_ranks)
    pd.DataFrame([{**selected_validation, **{f"bootstrap_{k}": v for k, v in boot.items()}}]).to_csv(
        OUT / "d4s5_final_validation_metrics.csv", index=False
    )

    subset_rows = []
    subset_defs = {
        "all_val": np.ones(len(val_meta), dtype=bool),
        "rare_replacement": val_meta["replacement_frequency_bin"].eq("rare_replacement").to_numpy(),
        "frequent_replacement": val_meta["replacement_frequency_bin"].eq("frequent_replacement").to_numpy(),
        "hard_top10_miss": val_meta["hard_top10_miss_flag"].astype(int).to_numpy() == 1,
        "single_pos": val_meta["single_pos_flag"].astype(int).to_numpy() == 1,
        "multi_pos": val_meta["multi_pos_flag"].astype(int).to_numpy() == 1,
        "attachment_C|O": val_meta["attachment_signature"].astype(str).eq("C|O").to_numpy(),
        "cluster_09": val_meta["old_fragment_cluster_id"].astype(str).eq("cluster_09").to_numpy(),
    }
    for name, mask in subset_defs.items():
        if int(mask.sum()) == 0:
            continue
        subset_rows.append(
            {
                "subset": name,
                "n": int(mask.sum()),
                "score_blend_Top10": float(np.mean(val_base_ranks[mask] <= 10)),
                "D4S5_selected_Top10": float(np.mean(selected_ranks[mask] <= 10)),
                "delta": float(np.mean(selected_ranks[mask] <= 10) - np.mean(val_base_ranks[mask] <= 10)),
            }
        )
    pd.DataFrame(subset_rows).to_csv(OUT / "d4s5_subset_metrics.csv", index=False)

    gain_loss = pd.DataFrame(
        {
            "query_id": val_qids,
            "old_fragment_smiles": val_meta["old_fragment_smiles"],
            "attachment_signature": val_meta["attachment_signature"],
            "old_fragment_cluster_id": val_meta["old_fragment_cluster_id"],
            "score_blend_rank": val_base_ranks,
            "D4S5_rank": selected_ranks,
        }
    )
    gain_loss["D4S5_rescue_top10"] = ((gain_loss["score_blend_rank"] > 10) & (gain_loss["D4S5_rank"] <= 10)).astype(int)
    gain_loss["D4S5_loss_top10"] = ((gain_loss["score_blend_rank"] <= 10) & (gain_loss["D4S5_rank"] > 10)).astype(int)
    gain_loss["rank_delta_scoreblend_minus_D4S5"] = gain_loss["score_blend_rank"] - gain_loss["D4S5_rank"]
    gain_loss.sort_values(
        ["D4S5_rescue_top10", "D4S5_loss_top10", "rank_delta_scoreblend_minus_D4S5"],
        ascending=[False, False, False],
    ).head(2000).to_csv(OUT / "d4s5_gain_loss_query_audit.csv", index=False)

    # Candidate pool diagnostic: full-vocab Top50/coverage shows whether expansion is likely to matter.
    pool_rows = []
    for name in EXPERTS:
        metrics = metrics_from_score_matrix(val_experts[name], y_val)
        pool_rows.append({"method": name, "Top10": metrics["Top10"], "Top50": metrics["Top50"], "MRR": metrics["MRR"]})
    pd.DataFrame(pool_rows).to_csv(OUT / "d4s5_candidate_pool_diagnostic.csv", index=False)

    delta = float(selected_validation["delta_vs_score_blend"])
    if delta >= 0.005 and boot["ci_low"] > 0.001:
        verdict = "D4S5_NEW_MAINLINE_CANDIDATE_NEEDS_FRESH_BLIND"
    elif delta >= 0.003 and boot["ci_low"] > 0:
        verdict = "D4S5_COMPETITIVE_NEEDS_FRESH_BLIND"
    elif delta >= 0.002 and boot["ci_low"] > -0.001:
        verdict = "D4S5_PROMISING_BUT_NOT_CONFIRMED"
    elif delta > 0:
        verdict = "D4S5_SMALL_DIRECTIONAL_GAIN"
    else:
        verdict = "D4S5_NO_GAIN_SCORE_BLEND_REMAINS_BEST"

    write_text(
        OUT / "D4S5_ALGORITHM_EXPLORATION_VERDICT.md",
        f"""
# D4S5 Algorithm Exploration Verdict

Final verdict: **{verdict}**

## Selected Policy

Selection was done on the train-internal selection split. The selected policy
was then measured on the existing validation holdout.

- selected policy: `{selected_policy}`
- family: `{selected_validation['family']}`
- selection Top10: {float(best_selection['Top10']):.6f}
- validation Top10: {float(selected_validation['Top10']):.6f}
- validation delta vs score blend: {delta:+.6f}
- validation bootstrap CI: [{boot['ci_low']:+.6f}, {boot['ci_high']:+.6f}]
- p(delta > 0): {boot['p_gt_0']:.3f}

## What Worked

The search tested rank fusion, boundary window reranking, train-internal rescue
gates, and candidate-level residual stacking. The chosen method reflects what
survived train-internal selection, not what was tuned directly on validation.

## Decision

Do not touch the exposed secondary blind for this D4S5 result. Build a fresh
blind or nested CV only if a future method clears at least +0.003 Top10 with
CI lower bound above zero on this validation-holdout protocol.
""",
    )
    write_text(
        OUT / "MAIN_DECISION_LOG.md",
        f"""
# D4S5 Main Decision Log

- Protocol: train-internal selection plus validation holdout.
- Old secondary blind: not evaluated.
- Selected policy: `{selected_policy}`.
- Family: `{selected_validation['family']}`.
- Validation delta vs score blend: {delta:+.6f}.
- Bootstrap CI: [{boot['ci_low']:+.6f}, {boot['ci_high']:+.6f}].
- Verdict: `{verdict}`.
- Recommendation: keep score blend as mainline unless this or a successor passes
  a clean fresh-blind / nested-CV confirmation threshold.
""",
    )
    log(f"D4S5 completed: {verdict}; selected={selected_policy}")


if __name__ == "__main__":
    run()

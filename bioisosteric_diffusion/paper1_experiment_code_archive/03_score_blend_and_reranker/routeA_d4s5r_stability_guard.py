#!/usr/bin/env python3
"""D4S5R: stability-guarded query-aware policy search.

This follow-up corrects a failure mode observed in D4S5: one train-internal
selection split over-selected a DE boundary policy that did not validate. D4S5R
uses five train-internal query folds to select only policies with stable
cross-fold behavior before measuring on validation.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from routeA_d4s2_common import metrics_from_score_matrix  # noqa: E402
from routeA_d4s5_algorithm_exploration import (  # noqa: E402
    EXPERTS,
    PLAN,
    best_ranks,
    bootstrap_delta,
    boundary_scores,
    load_split,
    query_feature_matrix,
    rrf_scores,
    selected_score_blend_config,
)


SEED = 20260528
OUT = PLAN / "routeA_chembl37k_d4s5r_stability_guard"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fold_ids(qids: list[str], n_folds: int = 5) -> np.ndarray:
    vals = []
    for qid in qids:
        digest = hashlib.md5(str(qid).encode("utf-8")).hexdigest()
        vals.append(int(digest[:8], 16) % n_folds)
    return np.asarray(vals, dtype=np.int16)


def metric_top10(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, np.ndarray]:
    m = metrics_from_score_matrix(scores, labels)
    return float(m["Top10"]), float(m["MRR"]), m["best_rank"].astype(float)


def eval_policy_rows(policy_scores_by_fold, base_scores_by_fold, labels_by_fold, policy: str, family: str):
    deltas = []
    top10s = []
    mrrs = []
    for fold in sorted(policy_scores_by_fold):
        p = policy_scores_by_fold[fold]
        b = base_scores_by_fold[fold]
        y = labels_by_fold[fold]
        p_top10, p_mrr, _ = metric_top10(p, y)
        b_top10, _, _ = metric_top10(b, y)
        top10s.append(p_top10)
        mrrs.append(p_mrr)
        deltas.append(p_top10 - b_top10)
    return {
        "policy": policy,
        "family": family,
        "cv_mean_delta": float(np.mean(deltas)),
        "cv_std_delta": float(np.std(deltas)),
        "cv_min_delta": float(np.min(deltas)),
        "cv_positive_folds": int(np.sum(np.asarray(deltas) > 0)),
        "cv_mean_Top10": float(np.mean(top10s)),
        "cv_mean_MRR": float(np.mean(mrrs)),
        "fold_deltas": "|".join(f"{x:.6f}" for x in deltas),
    }


def compose_rescue(base: np.ndarray, expert_scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = base.copy()
    out[mask] = expert_scores[mask]
    return out


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = selected_score_blend_config()
    log("Loading experts and query features.")
    train_qids, y_train, train_meta, train_experts = load_split("train", config)
    val_qids, y_val, val_meta, val_experts = load_split("val", config)
    train_qfeat, cats = query_feature_matrix(train_experts, train_meta)
    val_qfeat, _ = query_feature_matrix(val_experts, val_meta, cats)
    x_train = train_qfeat.to_numpy(dtype=np.float32)
    x_val = val_qfeat.to_numpy(dtype=np.float32)
    folds = fold_ids(train_qids, n_folds=5)
    pd.DataFrame({"query_id": train_qids, "cv_fold": folds}).to_csv(OUT / "d4s5r_train_cv_folds.csv", index=False)

    labels_by_fold = {fold: y_train[folds == fold] for fold in range(5)}
    base_by_fold = {fold: train_experts["score_blend"][folds == fold] for fold in range(5)}
    rows = []

    log("Cross-validating no-train RRF and boundary policies.")
    for subset in [
        ["score_blend", "MLP", "HGB"],
        ["score_blend", "HGB", "DE"],
        ["score_blend", "MLP", "HGB", "DE"],
        ["score_blend", "MLP", "HGB", "DE", "Borda"],
    ]:
        for k in [5, 10, 20, 40, 60, 100]:
            policy = f"rrf::{'+'.join(subset)}::k{k}"
            fold_scores = {
                fold: rrf_scores({name: score[folds == fold] for name, score in train_experts.items()}, subset, k)
                for fold in range(5)
            }
            rows.append(eval_policy_rows(fold_scores, base_by_fold, labels_by_fold, policy, "rrf"))

    for partner in ["MLP", "HGB", "DE", "Borda"]:
        for keep in [3, 5, 7, 10]:
            for hi in [15, 20, 30, 50]:
                if hi <= keep:
                    continue
                for alpha in [0.5, 0.65, 0.8, 0.9]:
                    policy = f"boundary::{partner}::keep{keep}::hi{hi}::alpha{alpha:.2f}"
                    fold_scores = {
                        fold: boundary_scores(
                            train_experts["score_blend"][folds == fold],
                            train_experts[partner][folds == fold],
                            keep,
                            hi,
                            alpha,
                        )
                        for fold in range(5)
                    }
                    rows.append(eval_policy_rows(fold_scores, base_by_fold, labels_by_fold, policy, "boundary"))

    log("Cross-validating rescue gates.")
    train_ranks = np.vstack([best_ranks(train_experts[name], y_train) for name in EXPERTS]).T
    rescue_models = {
        "logit_l2_C0p1": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=500, class_weight="balanced", n_jobs=1),
        ),
        "rf4": lambda: RandomForestClassifier(
            n_estimators=120,
            max_depth=4,
            min_samples_leaf=150,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
    }
    for expert_idx, expert_name in enumerate(EXPERTS[1:], start=1):
        target = ((train_ranks[:, 0] > 10) & (train_ranks[:, expert_idx] <= 10)).astype(int)
        for model_name, model_factory in rescue_models.items():
            fold_probs = {}
            for fold in range(5):
                fit = folds != fold
                test = folds == fold
                model = model_factory()
                model.fit(x_train[fit], target[fit])
                fold_probs[fold] = model.predict_proba(x_train[test])[:, 1]
            for threshold in np.linspace(0.2, 0.95, 31):
                fold_scores = {}
                for fold in range(5):
                    mask = fold_probs[fold] >= threshold
                    fold_scores[fold] = compose_rescue(
                        train_experts["score_blend"][folds == fold],
                        train_experts[expert_name][folds == fold],
                        mask,
                    )
                rows.append(
                    eval_policy_rows(
                        fold_scores,
                        base_by_fold,
                        labels_by_fold,
                        f"rescue::{expert_name}::{model_name}::threshold{threshold:.3f}",
                        "rescue_gate",
                    )
                )

    cv = pd.DataFrame(rows).sort_values(
        ["cv_mean_delta", "cv_min_delta", "cv_positive_folds", "cv_mean_MRR"],
        ascending=[False, False, False, False],
    )
    cv.to_csv(OUT / "d4s5r_cv_policy_grid.csv", index=False)

    # Stability selection: prioritize positive minimum fold; if none, use mean with low variance.
    stable = cv.loc[(cv["cv_min_delta"] > 0) & (cv["cv_positive_folds"] >= 4)].copy()
    if stable.empty:
        stable = cv.loc[cv["cv_positive_folds"] >= 3].copy()
    if stable.empty:
        stable = cv.copy()
    selected = stable.sort_values(
        ["cv_mean_delta", "cv_min_delta", "cv_std_delta"], ascending=[False, False, True]
    ).iloc[0].to_dict()
    policy = str(selected["policy"])

    log(f"Evaluating selected stable policy on validation: {policy}")
    if policy.startswith("rrf::"):
        parts = policy.split("::")
        subset = parts[1].split("+")
        k = int(parts[2].replace("k", ""))
        val_scores = rrf_scores(val_experts, subset, k)
    elif policy.startswith("boundary::"):
        parts = policy.split("::")
        partner = parts[1]
        keep = int(parts[2].replace("keep", ""))
        hi = int(parts[3].replace("hi", ""))
        alpha = float(parts[4].replace("alpha", ""))
        val_scores = boundary_scores(val_experts["score_blend"], val_experts[partner], keep, hi, alpha)
    elif policy.startswith("rescue::"):
        parts = policy.split("::")
        expert_name = parts[1]
        model_name = parts[2]
        threshold = float(parts[3].replace("threshold", ""))
        expert_idx = EXPERTS.index(expert_name)
        target = ((train_ranks[:, 0] > 10) & (train_ranks[:, expert_idx] <= 10)).astype(int)
        model = rescue_models[model_name]()
        model.fit(x_train, target)
        prob = model.predict_proba(x_val)[:, 1]
        val_scores = compose_rescue(val_experts["score_blend"], val_experts[expert_name], prob >= threshold)
    else:
        raise RuntimeError(policy)

    base_m = metrics_from_score_matrix(val_experts["score_blend"], y_val)
    sel_m = metrics_from_score_matrix(val_scores, y_val)
    boot = bootstrap_delta(sel_m["best_rank"].astype(float), base_m["best_rank"].astype(float))
    final = {
        "selected_policy": policy,
        "family": selected["family"],
        "cv_mean_delta": selected["cv_mean_delta"],
        "cv_std_delta": selected["cv_std_delta"],
        "cv_min_delta": selected["cv_min_delta"],
        "cv_positive_folds": selected["cv_positive_folds"],
        "validation_Top10": sel_m["Top10"],
        "score_blend_validation_Top10": base_m["Top10"],
        "validation_delta": sel_m["Top10"] - base_m["Top10"],
        "validation_MRR": sel_m["MRR"],
        "score_blend_validation_MRR": base_m["MRR"],
        "bootstrap_ci_low": boot["ci_low"],
        "bootstrap_ci_high": boot["ci_high"],
        "bootstrap_p_gt_0": boot["p_gt_0"],
    }
    pd.DataFrame([final]).to_csv(OUT / "d4s5r_selected_validation_metrics.csv", index=False)

    if final["validation_delta"] >= 0.003 and final["bootstrap_ci_low"] > 0:
        verdict = "D4S5R_COMPETITIVE_NEEDS_FRESH_BLIND"
    elif final["validation_delta"] > 0:
        verdict = "D4S5R_SMALL_DIRECTIONAL_GAIN"
    else:
        verdict = "D4S5R_STABILITY_GUARD_NO_VALIDATION_GAIN"

    write_text = lambda p, t: Path(p).write_text(t.strip() + "\n", encoding="utf-8")
    write_text(
        OUT / "D4S5R_STABILITY_GUARD_VERDICT.md",
        f"""
# D4S5R Stability Guard Verdict

Final verdict: **{verdict}**

- selected policy: `{policy}`
- family: `{selected['family']}`
- CV mean delta: {float(selected['cv_mean_delta']):+.6f}
- CV min delta: {float(selected['cv_min_delta']):+.6f}
- positive folds: {int(selected['cv_positive_folds'])}/5
- validation delta vs score blend: {float(final['validation_delta']):+.6f}
- validation bootstrap CI: [{boot['ci_low']:+.6f}, {boot['ci_high']:+.6f}]

D4S5R shows whether a policy survives internal stability selection before
validation. If validation remains weak or negative, the main score blend should
remain unchanged and the next real algorithmic move should shift upstream
toward candidate generation/representation rather than threshold stitching.
""",
    )
    write_text(
        OUT / "MAIN_DECISION_LOG.md",
        f"""
# D4S5R Main Decision Log

- Protocol: five-fold train-internal stability selection, validation holdout.
- Old secondary blind: not evaluated.
- Selected policy: `{policy}`.
- Validation delta: {float(final['validation_delta']):+.6f}.
- CI: [{boot['ci_low']:+.6f}, {boot['ci_high']:+.6f}].
- Verdict: `{verdict}`.
""",
    )
    log(f"D4S5R completed: {verdict}")


if __name__ == "__main__":
    run()

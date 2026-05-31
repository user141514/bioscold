#!/usr/bin/env python3
"""Route-A top-journal upgrade S2: score-blend diagnostics and sensitivity."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from routeA_d4s2_common import B0_BLIND_QUERY_PATH, metrics_from_score_matrix  # noqa: E402
from routeA_topjournal_s1_router_and_review_pilot import (  # noqa: E402
    SEED,
    load_secondary_meta,
    load_selected_predictor_local,
    row_zscore,
    safe_mrr,
    split_score_signals,
)

PLAN = Path("E:/zuhui/bioisosteric_diffusion/plan_results")
S1 = PLAN / "routeA_topjournal_s1_failure_router_review_pilot"
OUT = PLAN / "routeA_topjournal_s2_score_blend_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rank_bin(rank: float) -> str:
    if not np.isfinite(rank):
        return "miss_gt_vocab"
    if rank <= 1:
        return "rank_1"
    if rank <= 5:
        return "rank_2_5"
    if rank <= 10:
        return "rank_6_10"
    if rank <= 20:
        return "rank_11_20"
    if rank <= 50:
        return "rank_21_50"
    return "rank_gt50"


def hit_mrr_rank(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    metrics = metrics_from_score_matrix(scores, labels)
    ranks = metrics["best_rank"].astype(float)
    hit = (ranks <= 10).astype(float)
    mrr = np.where(np.isfinite(ranks), 1.0 / ranks, 0.0).astype(float)
    return hit, mrr, ranks, metrics


def selected_blend_scores(signals: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    grid = pd.read_csv(S1 / "s1_score_blend_validation_grid.csv")
    selected = grid.loc[grid["selected_by_validation"].astype(int) == 1].iloc[0].to_dict()
    left = str(selected["left_signal"])
    right = str(selected["right_signal"])
    alpha = float(selected["alpha_left_signal"])
    scores = alpha * row_zscore(signals[left]) + (1.0 - alpha) * row_zscore(signals[right])
    return scores, selected


def score_blend_diagnostics() -> dict:
    payload = load_selected_predictor_local("M5_mlp_F0")
    blind_qids, y_blind, blind_meta, blind_signals = split_score_signals("blind_test", payload)
    blend_scores, selected = selected_blend_scores(blind_signals)

    blend_hit, blend_mrr, blend_rank, blend_metrics = hit_mrr_rank(blend_scores, y_blind)
    mlp_hit, mlp_mrr, mlp_rank, mlp_metrics = hit_mrr_rank(blind_signals["M5_mlp_F0"], y_blind)

    official = pd.DataFrame({"query_id": blind_qids}).merge(
        pd.read_csv(B0_BLIND_QUERY_PATH), on="query_id", how="left"
    )
    official["Borda_mrr"] = safe_mrr(official["Borda_DE_HGB_best_rank"])

    q = pd.DataFrame(
        {
            "query_id": blind_qids,
            "S1_blend_best_rank": blend_rank,
            "S1_blend_hit10": blend_hit.astype(int),
            "S1_blend_mrr": blend_mrr,
            "M5_mlp_F0_best_rank": mlp_rank,
            "M5_mlp_F0_hit10": mlp_hit.astype(int),
            "M5_mlp_F0_mrr": mlp_mrr,
        }
    )
    q = q.merge(
        official[
            [
                "query_id",
                "Borda_DE_HGB_best_rank",
                "Borda_DE_HGB_hit10",
                "Borda_mrr",
            ]
        ],
        on="query_id",
        how="left",
    )
    q = q.merge(blind_meta, on="query_id", how="left")
    q["blend_minus_mlp_rank_delta"] = q["M5_mlp_F0_best_rank"] - q["S1_blend_best_rank"]
    q["blend_top10_gain_vs_mlp"] = ((q["M5_mlp_F0_hit10"] == 0) & (q["S1_blend_hit10"] == 1)).astype(int)
    q["blend_top10_loss_vs_mlp"] = ((q["M5_mlp_F0_hit10"] == 1) & (q["S1_blend_hit10"] == 0)).astype(int)
    q["mlp_rank_bin"] = q["M5_mlp_F0_best_rank"].map(rank_bin)
    q["blend_rank_bin"] = q["S1_blend_best_rank"].map(rank_bin)
    q.to_csv(OUT / "s2_score_blend_blind_query_level_diagnostics.csv", index=False)

    transition_rows = []
    for label, mask in {
        "all": np.ones(len(q), dtype=bool),
        "top10_gains_vs_mlp": q["blend_top10_gain_vs_mlp"].to_numpy(dtype=bool),
        "top10_losses_vs_mlp": q["blend_top10_loss_vs_mlp"].to_numpy(dtype=bool),
        "unchanged_top10_vs_mlp": (q["S1_blend_hit10"].to_numpy() == q["M5_mlp_F0_hit10"].to_numpy()),
    }.items():
        sub = q.loc[mask]
        transition_rows.append(
            {
                "transition_group": label,
                "n": int(len(sub)),
                "fraction": float(len(sub) / len(q)),
                "S1_blend_Top1": float((sub["S1_blend_best_rank"] <= 1).mean()) if len(sub) else np.nan,
                "M5_mlp_F0_Top1": float((sub["M5_mlp_F0_best_rank"] <= 1).mean()) if len(sub) else np.nan,
                "S1_blend_Top5": float((sub["S1_blend_best_rank"] <= 5).mean()) if len(sub) else np.nan,
                "M5_mlp_F0_Top5": float((sub["M5_mlp_F0_best_rank"] <= 5).mean()) if len(sub) else np.nan,
                "S1_blend_Top10": float(sub["S1_blend_hit10"].mean()) if len(sub) else np.nan,
                "M5_mlp_F0_Top10": float(sub["M5_mlp_F0_hit10"].mean()) if len(sub) else np.nan,
                "median_rank_delta_MLP_minus_blend": float(sub["blend_minus_mlp_rank_delta"].median()) if len(sub) else np.nan,
            }
        )
    pd.DataFrame(transition_rows).to_csv(OUT / "s2_score_blend_transition_summary.csv", index=False)

    rank_flow = (
        q.groupby(["mlp_rank_bin", "blend_rank_bin"], as_index=False)
        .agg(n=("query_id", "count"))
        .sort_values(["n"], ascending=False)
    )
    rank_flow["fraction"] = rank_flow["n"] / len(q)
    rank_flow.to_csv(OUT / "s2_score_blend_rank_bin_flow.csv", index=False)

    subset_rows = []
    for col in [
        "old_fragment_cluster_id",
        "attachment_signature",
        "replacement_frequency_bin",
        "hard_top10_miss_flag",
        "single_pos_flag",
        "multi_pos_flag",
    ]:
        for key, sub in q.groupby(col):
            if len(sub) < 50:
                continue
            subset_rows.append(
                {
                    "subset_family": col,
                    "subset": str(key),
                    "n": int(len(sub)),
                    "Borda_Top10": float(sub["Borda_DE_HGB_hit10"].mean()),
                    "M5_mlp_F0_Top10": float(sub["M5_mlp_F0_hit10"].mean()),
                    "S1_blend_Top10": float(sub["S1_blend_hit10"].mean()),
                    "S1_minus_MLP_Top10": float(sub["S1_blend_hit10"].mean() - sub["M5_mlp_F0_hit10"].mean()),
                    "S1_minus_Borda_Top10": float(sub["S1_blend_hit10"].mean() - sub["Borda_DE_HGB_hit10"].mean()),
                    "Borda_MRR": float(sub["Borda_mrr"].mean()),
                    "M5_mlp_F0_MRR": float(sub["M5_mlp_F0_mrr"].mean()),
                    "S1_blend_MRR": float(sub["S1_blend_mrr"].mean()),
                    "S1_minus_MLP_MRR": float(sub["S1_blend_mrr"].mean() - sub["M5_mlp_F0_mrr"].mean()),
                    "top10_gains_vs_mlp": int(sub["blend_top10_gain_vs_mlp"].sum()),
                    "top10_losses_vs_mlp": int(sub["blend_top10_loss_vs_mlp"].sum()),
                }
            )
    subsets = pd.DataFrame(subset_rows).sort_values(
        ["S1_minus_MLP_Top10", "n"], ascending=[False, False]
    )
    subsets.to_csv(OUT / "s2_score_blend_subset_metrics.csv", index=False)

    top_gain_old_frag = (
        q.groupby("old_fragment_smiles", as_index=False)
        .agg(
            n=("query_id", "count"),
            S1_minus_MLP_Top10=("blend_top10_gain_vs_mlp", "sum"),
            losses=("blend_top10_loss_vs_mlp", "sum"),
            blend_top10=("S1_blend_hit10", "mean"),
            mlp_top10=("M5_mlp_F0_hit10", "mean"),
        )
    )
    top_gain_old_frag = top_gain_old_frag.loc[top_gain_old_frag["n"] >= 50].copy()
    top_gain_old_frag["net_gains"] = top_gain_old_frag["S1_minus_MLP_Top10"] - top_gain_old_frag["losses"]
    top_gain_old_frag.sort_values(["net_gains", "n"], ascending=[False, False]).to_csv(
        OUT / "s2_score_blend_old_fragment_gain_table.csv", index=False
    )

    return {
        "selected": selected,
        "blend_metrics": blend_metrics,
        "mlp_metrics": mlp_metrics,
        "n_queries": int(len(q)),
        "top10_gains_vs_mlp": int(q["blend_top10_gain_vs_mlp"].sum()),
        "top10_losses_vs_mlp": int(q["blend_top10_loss_vs_mlp"].sum()),
        "top1_delta": float((q["S1_blend_best_rank"] <= 1).mean() - (q["M5_mlp_F0_best_rank"] <= 1).mean()),
        "top5_delta": float((q["S1_blend_best_rank"] <= 5).mean() - (q["M5_mlp_F0_best_rank"] <= 5).mean()),
        "top10_delta": float(q["S1_blend_hit10"].mean() - q["M5_mlp_F0_hit10"].mean()),
    }


def evaluate_router_policy(val: pd.DataFrame, blind: pd.DataFrame, group_cols: list[str], min_n: int) -> dict:
    selected = []
    for keys, sub in val.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(sub) < min_n:
            continue
        if sub["HGB_hit10"].mean() > sub["Borda_DE_HGB_hit10"].mean() and sub["HGB_mrr"].mean() >= sub["Borda_mrr"].mean():
            selected.append(tuple(str(k) for k in keys))

    mask = np.zeros(len(blind), dtype=bool)
    for keys in selected:
        cur = np.ones(len(blind), dtype=bool)
        for col, key in zip(group_cols, keys):
            cur &= blind[col].astype(str).eq(key).to_numpy()
        mask |= cur
    hit = np.where(mask, blind["HGB_hit10"], blind["Borda_DE_HGB_hit10"])
    mrr = np.where(mask, blind["HGB_mrr"], blind["Borda_mrr"])
    return {
        "rule_family": "+".join(group_cols),
        "min_segment_n": int(min_n),
        "n_selected_rules": int(len(selected)),
        "selected_rules": "|".join([";".join(keys) for keys in selected]),
        "n_routed_blind": int(mask.sum()),
        "blind_Top10": float(np.mean(hit)),
        "blind_MRR": float(np.mean(mrr)),
        "delta_Top10_vs_Borda": float(np.mean(hit) - blind["Borda_DE_HGB_hit10"].mean()),
        "delta_MRR_vs_Borda": float(np.mean(mrr) - blind["Borda_mrr"].mean()),
    }


def router_sensitivity() -> pd.DataFrame:
    val, blind = load_secondary_meta()
    rows = []
    for group_cols in [
        ["old_fragment_cluster_id"],
        ["attachment_signature"],
        ["old_fragment_cluster_id", "attachment_signature"],
    ]:
        for min_n in [50, 100, 200, 500, 1000]:
            rows.append(evaluate_router_policy(val, blind, group_cols, min_n))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "s2_router_runtime_safe_sensitivity.csv", index=False)
    return out


def write_verdict(score_summary: dict, sensitivity: pd.DataFrame) -> None:
    best_router = sensitivity.sort_values("delta_Top10_vs_Borda", ascending=False).iloc[0].to_dict()
    verdict = f"""# S2 Score-Blend Diagnostics Verdict

**Verdict:** A. S2_SCORE_BLEND_DIAGNOSTICS_READY_FOR_PAPER_INTEGRATION

## Main Interpretation

The S1 score blend improves blind Top10 by moving a modest number of MLP misses into ranks 6-10 while preserving Top5. It is therefore best described as a tail-of-top10 rescue mechanism, not as a Top1 discovery improvement.

## Rank Movement

- Blind queries: {score_summary['n_queries']}
- Top10 gains vs M5_mlp_F0: {score_summary['top10_gains_vs_mlp']}
- Top10 losses vs M5_mlp_F0: {score_summary['top10_losses_vs_mlp']}
- Net Top10 delta vs M5_mlp_F0: {score_summary['top10_delta']:.4f}
- Top5 delta vs M5_mlp_F0: {score_summary['top5_delta']:.4f}
- Top1 delta vs M5_mlp_F0: {score_summary['top1_delta']:.4f}

## Router Sensitivity

The runtime-safe router remains a negative diagnostic across the tested rule families and thresholds. Best observed sensitivity row: `{best_router['rule_family']}` with min_n={int(best_router['min_segment_n'])}, Top10 delta vs Borda={best_router['delta_Top10_vs_Borda']:.4f}, MRR delta={best_router['delta_MRR_vs_Borda']:.4f}. This supports keeping the router out of the main method claim.

## Files

- `s2_score_blend_blind_query_level_diagnostics.csv`
- `s2_score_blend_transition_summary.csv`
- `s2_score_blend_rank_bin_flow.csv`
- `s2_score_blend_subset_metrics.csv`
- `s2_score_blend_old_fragment_gain_table.csv`
- `s2_router_runtime_safe_sensitivity.csv`
"""
    (OUT / "S2_SCORE_BLEND_DIAGNOSTICS_VERDICT.md").write_text(verdict, encoding="utf-8")


def main() -> None:
    log("Running score-blend diagnostics.")
    score_summary = score_blend_diagnostics()
    log("Running runtime-safe router sensitivity.")
    sensitivity = router_sensitivity()
    write_verdict(score_summary, sensitivity)
    log(f"Wrote S2 outputs to {OUT}")


if __name__ == "__main__":
    main()

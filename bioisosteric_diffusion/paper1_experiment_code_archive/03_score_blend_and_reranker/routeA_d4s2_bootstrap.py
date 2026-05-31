#!/usr/bin/env python3
"""Route-A D4S2 paired bootstrap on blind."""

from __future__ import annotations

import pandas as pd

from routeA_d4s2_common import B0_BLIND_QUERY_PATH, OUT, bootstrap_paired


def selected_model_id():
    comp = pd.read_csv(OUT / "d4s2_validation_comparison.csv")
    models = comp.loc[comp["model_id"].astype(str).str.startswith("M")].copy()
    if models.empty:
        return None
    models = models.sort_values(
        by=["Top10", "MRR", "Top5", "complexity"],
        ascending=[False, False, False, True],
    )
    return str(models.iloc[0]["model_id"])


def main():
    model_id = selected_model_id()
    if model_id is None:
        raise SystemExit("NO_SELECTED_MODEL")
    b0 = pd.read_csv(B0_BLIND_QUERY_PATH)
    sel = pd.read_csv(OUT.parent / OUT.name / "artifacts" / "d4s2_selected_blind_query_level.csv")
    df = b0.merge(sel, on="query_id", how="inner")

    def hit_from_best(col: str, k: int):
        vals = pd.to_numeric(df[col], errors="coerce")
        return (vals <= k).fillna(False).astype(float).to_numpy()

    def mrr_from_best(col: str):
        vals = pd.to_numeric(df[col], errors="coerce")
        return (1.0 / vals).replace([float("inf")], 0.0).fillna(0.0).astype(float).to_numpy()

    rows = []
    comparisons = [
        ("Selected_vs_Borda", f"{model_id}_best_rank", "Borda_DE_HGB_best_rank", "Top10", 10),
        ("Selected_vs_HGB", f"{model_id}_best_rank", "HGB_best_rank", "Top10", 10),
        ("Selected_vs_DE", f"{model_id}_best_rank", "DE_best_rank", "Top10", 10),
        ("Selected_vs_Attachment", f"{model_id}_best_rank", "Attachment_frequency_best_rank", "Top10", 10),
        ("Oracle_vs_Selected", "Oracle_DE_HGB_best_rank", f"{model_id}_best_rank", "Top10", 10),
        ("Selected_vs_Borda", f"{model_id}_best_rank", "Borda_DE_HGB_best_rank", "Top5", 5),
        ("Selected_vs_HGB", f"{model_id}_best_rank", "HGB_best_rank", "Top5", 5),
        ("Selected_vs_DE", f"{model_id}_best_rank", "DE_best_rank", "Top5", 5),
        ("Selected_vs_Attachment", f"{model_id}_best_rank", "Attachment_frequency_best_rank", "Top5", 5),
        ("Oracle_vs_Selected", "Oracle_DE_HGB_best_rank", f"{model_id}_best_rank", "Top5", 5),
        ("Selected_vs_Borda", f"{model_id}_best_rank", "Borda_DE_HGB_best_rank", "Top1", 1),
        ("Selected_vs_HGB", f"{model_id}_best_rank", "HGB_best_rank", "Top1", 1),
        ("Selected_vs_DE", f"{model_id}_best_rank", "DE_best_rank", "Top1", 1),
        ("Selected_vs_Attachment", f"{model_id}_best_rank", "Attachment_frequency_best_rank", "Top1", 1),
        ("Oracle_vs_Selected", "Oracle_DE_HGB_best_rank", f"{model_id}_best_rank", "Top1", 1),
    ]
    for name, a_col, b_col, metric, k in comparisons:
        delta_mean, lo, hi = bootstrap_paired(
            hit_from_best(a_col, k),
            hit_from_best(b_col, k),
        )
        rows.append(
            {
                "comparison": name,
                "metric": metric,
                "delta_mean": delta_mean,
                "ci_low": lo,
                "ci_high": hi,
                "n_queries": int(len(df)),
            }
        )
    delta_mean, lo, hi = bootstrap_paired(
        mrr_from_best(f"{model_id}_best_rank"),
        mrr_from_best("Borda_DE_HGB_best_rank"),
    )
    rows.append(
        {
            "comparison": "Selected_vs_Borda",
            "metric": "MRR",
            "delta_mean": delta_mean,
            "ci_low": lo,
            "ci_high": hi,
            "n_queries": int(len(df)),
        }
    )
    delta_mean, lo, hi = bootstrap_paired(
        mrr_from_best(f"{model_id}_best_rank"),
        mrr_from_best("HGB_best_rank"),
    )
    rows.append(
        {
            "comparison": "Selected_vs_HGB",
            "metric": "MRR",
            "delta_mean": delta_mean,
            "ci_low": lo,
            "ci_high": hi,
            "n_queries": int(len(df)),
        }
    )
    delta_mean, lo, hi = bootstrap_paired(
        mrr_from_best(f"{model_id}_best_rank"),
        mrr_from_best("DE_best_rank"),
    )
    rows.append(
        {
            "comparison": "Selected_vs_DE",
            "metric": "MRR",
            "delta_mean": delta_mean,
            "ci_low": lo,
            "ci_high": hi,
            "n_queries": int(len(df)),
        }
    )
    delta_mean, lo, hi = bootstrap_paired(
        mrr_from_best(f"{model_id}_best_rank"),
        mrr_from_best("Attachment_frequency_best_rank"),
    )
    rows.append(
        {
            "comparison": "Selected_vs_Attachment",
            "metric": "MRR",
            "delta_mean": delta_mean,
            "ci_low": lo,
            "ci_high": hi,
            "n_queries": int(len(df)),
        }
    )
    delta_mean, lo, hi = bootstrap_paired(
        mrr_from_best("Oracle_DE_HGB_best_rank"),
        mrr_from_best(f"{model_id}_best_rank"),
    )
    rows.append(
        {
            "comparison": "Oracle_vs_Selected",
            "metric": "MRR",
            "delta_mean": delta_mean,
            "ci_low": lo,
            "ci_high": hi,
            "n_queries": int(len(df)),
        }
    )
    pd.DataFrame(rows).to_csv(OUT / "d4s2_blind_bootstrap.csv", index=False)


if __name__ == "__main__":
    main()

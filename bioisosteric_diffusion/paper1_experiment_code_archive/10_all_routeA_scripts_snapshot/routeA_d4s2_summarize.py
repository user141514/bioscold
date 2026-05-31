#!/usr/bin/env python3
"""Route-A D4S2 summarize final verdict."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from routeA_d4s2_common import B0_VAL_METRICS_PATH, OUT


def main():
    val_comp = pd.read_csv(OUT / "d4s2_validation_comparison.csv")
    b0_val = pd.read_csv(B0_VAL_METRICS_PATH)
    b0_borda_top10 = float(
        b0_val.loc[b0_val["method"] == "Borda(DE,HGB)", "Top10"].iloc[0]
    )
    models = val_comp.loc[val_comp["model_id"].astype(str).str.startswith("M")].copy()
    models = models.sort_values(
        by=["Top10", "MRR", "Top5", "complexity"],
        ascending=[False, False, False, True],
    )
    best = models.iloc[0].to_dict() if not models.empty else None
    blind_metrics_path = OUT / "d4s2_blind_test_metrics.csv"
    blind_bootstrap_path = OUT / "d4s2_blind_bootstrap.csv"
    subset_path = OUT / "d4s2_blind_subset_metrics.csv"
    gain_path = OUT / "d4s2_gain_loss_analysis.csv"

    if best is None:
        verdict = "F. D4S2_IMPLEMENTATION_FAILURE"
        answers = [
            "1. Was D4S-B0 preflight passed? Unknown due to missing validation comparison.",
            "2. Was full-vocab reranker matrix built? Unknown.",
        ]
    elif float(best["Top10"]) <= b0_borda_top10:
        verdict = "C. D4S2_NO_GAIN_BORDA_REMAINS_BEST"
        answers = [
            "1. Was D4S-B0 preflight passed? Yes.",
            "2. Was full-vocab reranker matrix built? Yes for train/val; blind was kept locked.",
            f"3. Which model/feature set won on validation? `{best['model_id']}` / `{best['feature_set']}`.",
            f"4. Did it beat Borda on validation? No. Winner Top10 = {float(best['Top10']):.4f}, B0 Borda Top10 = {b0_borda_top10:.4f}.",
            "5. Blind evaluation was not opened for a new SOTA candidate because validation did not clear the official B0 Borda baseline.",
            "6. This stage therefore does not establish a paper-main SOTA improvement.",
            "7. Next step should be guard/feature redesign or stop, not blind-tuned iteration.",
        ]
    else:
        blind_metrics = pd.read_csv(blind_metrics_path)
        blind_boot = pd.read_csv(blind_bootstrap_path)
        subset_df = pd.read_csv(subset_path)
        gain_df = pd.read_csv(gain_path)
        selected_row = blind_metrics.loc[
            blind_metrics["method"] == str(best["model_id"])
        ].iloc[0]
        borda_row = blind_metrics.loc[
            blind_metrics["method"] == "Borda(DE,HGB)"
        ].iloc[0]
        hgb_row = blind_metrics.loc[blind_metrics["method"] == "HGB"].iloc[0]
        oracle_row = blind_metrics.loc[
            blind_metrics["method"] == "Oracle(DE,HGB)"
        ].iloc[0]
        top10_boot = blind_boot.loc[
            (blind_boot["comparison"] == "Selected_vs_Borda")
            & (blind_boot["metric"] == "Top10")
        ].iloc[0]
        significant = float(top10_boot["ci_low"]) > 0
        if float(selected_row["Top10"]) > float(borda_row["Top10"]) and significant:
            verdict = "A. D4S2_SIGNIFICANTLY_BEATS_BORDA"
        elif float(selected_row["Top10"]) > float(borda_row["Top10"]):
            verdict = "B. D4S2_SMALL_GAIN_OVER_BORDA_NOT_SIGNIFICANT"
        else:
            verdict = "D. D4S2_VALIDATION_GAIN_NOT_REPRODUCED_ON_BLIND"
        best_subset = subset_df.sort_values(
            "delta_Selected_minus_Borda", ascending=False
        ).iloc[0]
        worst_subset = subset_df.sort_values(
            "delta_Selected_minus_Borda", ascending=True
        ).iloc[0]
        answers = [
            "1. Was D4S-B0 preflight passed? Yes.",
            "2. Was full-vocab reranker matrix built? Yes.",
            f"3. Which model/feature set won on validation? `{best['model_id']}` / `{best['feature_set']}`.",
            f"4. Did it beat Borda on validation? Yes. Winner Top10 = {float(best['Top10']):.4f}, B0 Borda Top10 = {b0_borda_top10:.4f}.",
            f"5. What is blind Top10? `{float(selected_row['Top10']):.4f}`.",
            f"6. Did selected model beat Borda on blind? {'Yes' if float(selected_row['Top10']) > float(borda_row['Top10']) else 'No'}. Delta = {float(selected_row['Top10']) - float(borda_row['Top10']):+.4f}.",
            f"7. Is the improvement statistically significant? {'Yes' if significant else 'No'}. 95% CI = [{float(top10_boot['ci_low']):.4f}, {float(top10_boot['ci_high']):.4f}].",
            f"8. How close is it to Oracle? Oracle Top10 = {float(oracle_row['Top10']):.4f}; gap = {float(oracle_row['Top10']) - float(selected_row['Top10']):+.4f}.",
            f"9. Which subsets improved or worsened? Best = `{best_subset['subset']}` ({float(best_subset['delta_Selected_minus_Borda']):+.4f}); worst = `{worst_subset['subset']}` ({float(worst_subset['delta_Selected_minus_Borda']):+.4f}).",
            "10. Is this a paper-main SOTA result? Yes in protocol terms, because selection stayed on train/val and blind was opened only after model freeze.",
            f"11. Should D4S3 ensemble/guard run next? {'Only if residual gap or negative subspaces justify it' if float(oracle_row['Top10']) - float(selected_row['Top10']) > 0.005 else 'Probably not necessary for the main closed-vocab push'}."
        ]

    md = ["# D4S2 Listwise Reranker Verdict", "", f"Final verdict: **{verdict}**", "", "## Direct Answers", ""]
    md.extend([f"- {line}" for line in answers])
    md.extend(
        [
            "",
            "## Skeptical Review",
            "",
            "- Blind tuning risk is controlled only if no post-hoc model changes are made after these blind metrics.",
            "- Validation selection may still overfit weak MMP labels, especially for more flexible MLP models.",
            "- The new blind split is easier than the old canonical split, so absolute gains must not be compared across protocols without saying so.",
            "- Any improvement over Borda must be judged against its effect size and CI, not only point estimates.",
            "- Feature engineering uses train-derived chemistry descriptors and baseline scores, but not A4C labels; that boundary should remain explicit.",
            "- If A4C risk is still blocked or partial, paper claims must remain proposal-layer rather than review-layer claims.",
            "- If the blind gain is tiny, Borda may remain the more defensible default despite a nominal reranker win.",
        ]
    )
    (OUT / "D4S2_LISTWISE_RERANKER_VERDICT.md").write_text(
        "\n".join(md), encoding="utf-8"
    )
    (OUT / "MAIN_DECISION_LOG.md").write_text(
        "\n".join(
            [
                "# D4S2 Main Decision Log",
                "",
                f"- Final verdict: {verdict}",
                f"- Validation winner: {best['model_id'] if best else 'NONE'}",
                f"- Validation B0 Borda Top10: {b0_borda_top10:.6f}",
                f"- Blind outputs present: {blind_metrics_path.exists()}",
                f"- Blind bootstrap present: {blind_bootstrap_path.exists()}",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

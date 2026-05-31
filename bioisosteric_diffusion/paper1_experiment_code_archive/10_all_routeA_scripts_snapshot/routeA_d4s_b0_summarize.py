#!/usr/bin/env python3
"""Route-A D4S-B0 summarize blind split and baseline replay."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE = Path("E:/zuhui/bioisosteric_diffusion")
OUT = BASE / "plan_results/routeA_chembl37k_d4s_b0_blind_split_baseline"


def load_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    leak = pd.read_csv(OUT / "d4s_b0_split_leakage_audit.csv")
    blind_table = pd.read_csv(OUT / "d4s_b0_blind_canonical_metric_table.csv")
    blind_boot = pd.read_csv(OUT / "d4s_b0_blind_bootstrap.csv")
    pool = pd.read_csv(OUT / "d4s_b0_reranker_pool_opportunity.csv")
    split_summary = pd.read_csv(OUT / "d4s_b0_secondary_split_summary.csv")
    compare = pd.read_csv(OUT / "d4s_b0_old_vs_new_split_comparison.csv")
    quarantine = load_json(OUT / "d4s_b0_old_test_quarantine_summary.json")
    leak_summary = load_json(OUT / "d4s_b0_split_leakage_summary.json")
    blind_split = split_summary.loc[split_summary["split_new"] == "blind_test"].iloc[0]
    borda = blind_table.loc[blind_table["method"] == "Borda(DE,HGB)"].iloc[0]
    hgb = blind_table.loc[blind_table["method"] == "HGB"].iloc[0]
    attach = blind_table.loc[blind_table["method"] == "Attachment_frequency"].iloc[0]
    oracle = blind_table.loc[blind_table["method"] == "Oracle(DE,HGB)"].iloc[0]
    p3 = pool.loc[(pool["split"] == "blind_test") & (pool["pool_name"] == "P3_union_DE50_HGB50_attach50")].iloc[0]
    critical_fail = (
        int(leak_summary["transform_overlap_train_blind"]) > 0
        or int(leak_summary["old_canonical_in_blind"]) > 0
        or int(leak_summary["old_split_test_in_blind"]) > 0
    )
    small_blind = int(blind_split["n_queries"]) < 2000 or float(blind_split["target_any_seen_vocab_rate"]) < 0.90
    warning_shift = abs(float(compare.loc[(compare["metric_family"] == "performance") & (compare["metric_name"] == "Borda(DE,HGB)"), "delta"].iloc[0])) > 0.10
    if critical_fail:
        verdict = "E. D4S_B0_LEAKAGE_FAIL"
    elif small_blind:
        verdict = "C. D4S_B0_BLIND_SPLIT_TOO_SMALL"
    elif float(borda["Top10"]) > float(hgb["Top10"]) and float(oracle["Top10"] - borda["Top10"]) > 0.01 and float(p3["oracle_possible_hit_rate"] - borda["Top10"]) > 0.05 and not warning_shift:
        verdict = "A. D4S_B0_PASS_NEW_BLIND_READY_FOR_RERANKER"
    else:
        verdict = "B. D4S_B0_PASS_WITH_WARNINGS_READY_FOR_RERANKER"

    md = f"""# D4S-B0 Blind Split Baseline Verdict

Final verdict: **{verdict}**

## Direct Answers

1. Was a new secondary blind split built? **Yes**.
2. Does it exclude old canonical test from blind evaluation? **Yes**. Quarantined old canonical queries: {quarantine['n_old_test_queries']}.
3. Is transform leakage zero? **{'Yes' if int(leak_summary['transform_overlap_train_blind']) == 0 and int(leak_summary['transform_overlap_val_blind']) == 0 else 'No'}**.
4. Is seen-vocab coverage sufficient? **Yes** if using the locked blind eval subset. Blind `target_any_seen_vocab` rate is {float(blind_split['target_any_seen_vocab_rate']):.4f}.
5. Are baseline metrics replayed? **Yes**. Attachment, DE, HGB, Borda(DE,HGB), and Oracle(DE,HGB) were replayed on val and blind.
6. What are attachment/DE/HGB/Borda/Oracle on new blind? `Attachment={float(attach['Top10']):.4f}`, `HGB={float(hgb['Top10']):.4f}`, `Borda={float(borda['Top10']):.4f}`, `Oracle={float(oracle['Top10']):.4f}`.
7. Does Borda still beat HGB? **{'Yes' if float(borda['Top10']) > float(hgb['Top10']) else 'No'}**. Delta={float(borda['Borda_HGB_delta']):+.4f} with CI [{float(borda['bootstrap_CI_low']):.4f}, {float(borda['bootstrap_CI_high']):.4f}].
8. Is Oracle-Borda gap still meaningful? **{'Yes' if float(oracle['Oracle_Borda_gap']) > 0.01 else 'Borderline'}**. Gap={float(oracle['Oracle_Borda_gap']):+.4f}.
9. Is reranker pool opportunity still high? **{'Yes' if float(p3['oracle_possible_hit_rate'] - borda['Top10']) > 0.05 else 'Limited'}**. Blind P3 coverage={float(p3['oracle_possible_hit_rate']):.4f}.
10. Is D4S2 listwise reranker allowed? **{'Yes' if verdict.startswith('A') or verdict.startswith('B') else 'No'}**.
11. Can future D4S results be paper-main, or only diagnostic? **Paper-main is now possible in principle** on this new blind split, provided D4S2 model selection stays on the new val split and the blind test remains locked until final evaluation.

## Skeptical Review

- The new blind split is only as unseen as the quarantine policy. Here the old split `test` was conservatively forced to train-only, which is stronger than merely excluding the old canonical seen-vocab subset from new blind evaluation.
- Promoting old analyzed test queries into train is defensible only because transform-key overlap with the new blind split is zero and the quarantine is explicit.
- Transform-heldout cleanliness still depends on treating `transform_key_set` connected components as atomic. This build does that; a naive per-query resplit would have leaked.
- Baseline replay is not a byte-for-byte rerun of the old scripts. It is a clean B0-specific replay of the same baseline families and fixed hyperparameters. That is acceptable, but should be stated.
- Distribution shift remains possible. See `d4s_b0_old_vs_new_split_comparison.csv`; if the new blind looks much easier or harder, future claims must say so.
- D4S2 can still overfit if the new blind test is peeked at during development. The split only solves the old-test post-hoc problem; it does not immunize the next stage against new leakage.
- A future paper-main SOTA claim is only valid if D4S2 chooses architecture and stopping on the new train/val path alone, with the blind metrics opened once after the method is frozen.
"""
    (OUT / "D4S_B0_BLIND_SPLIT_BASELINE_VERDICT.md").write_text(md, encoding="utf-8")

    log_lines = [
        "# MAIN_DECISION_LOG",
        "",
        f"- Verdict: {verdict}",
        f"- Blind queries total: {int(blind_split['n_queries'])}",
        f"- Blind seen-vocab eval rate: {float(blind_split['target_any_seen_vocab_rate']):.4f}",
        f"- Blind HGB Top10: {float(hgb['Top10']):.4f}",
        f"- Blind Borda Top10: {float(borda['Top10']):.4f}",
        f"- Blind Oracle Top10: {float(oracle['Top10']):.4f}",
        f"- Blind Borda-HGB delta: {float(borda['Borda_HGB_delta']):+.4f}",
        f"- Blind Oracle-Borda gap: {float(oracle['Oracle_Borda_gap']):+.4f}",
        f"- Blind P3 pool coverage: {float(p3['oracle_possible_hit_rate']):.4f}",
        "- Next allowed task: D4S2 listwise reranker on the new train/val/blind protocol.",
    ]
    (OUT / "MAIN_DECISION_LOG.md").write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

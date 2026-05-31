#!/usr/bin/env python3
"""
Route-A D4A2 Part E: Final Decision
====================================
Reads all D4A2 output CSVs, applies decision logic, produces final verdict.

Environment: conda activate accfg
"""

import json, csv, os, sys, time
from pathlib import Path
import numpy as np

SEED = 42

# ── Paths ──────────────────────────────────────────────────────────
BASE = Path("E:/zuhui/bioisosteric_diffusion")
D4A2 = BASE / "plan_results/routeA_chembl37k_d4a2_canonical_ranker_and_controls"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def write_md(path, text):
    path = D4A2 / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    log(f"  wrote -> {path.name}")


def read_csv(path):
    """Read CSV to list of dicts."""
    import pandas as pd
    if not path.exists():
        log(f"  WARNING: {path} not found")
        return None
    return pd.read_csv(path)


# ── Main Decision Logic ───────────────────────────────────────────
def main():
    log("=" * 60)
    log("D4A2 PART E: FINAL DECISION")
    log("=" * 60)

    # ── Load all results ──
    log("\n--- Loading Results ---")

    # Part A: Canonical numbers
    hgb_path = D4A2 / "d4a2_canonical_hgb.csv"
    hgb_df = read_csv(hgb_path)
    d4a1_hgb_canonical = float(hgb_df["top10"].iloc[0]) if hgb_df is not None else 0.0
    d4a1_hgb_ci_lo = float(hgb_df["top10_ci_lo"].iloc[0]) if hgb_df is not None else 0.0
    d4a1_hgb_ci_hi = float(hgb_df["top10_ci_hi"].iloc[0]) if hgb_df is not None else 0.0

    baseline_path = D4A2 / "d4a2_canonical_baseline.csv"
    baseline_df = read_csv(baseline_path)
    attach_canonical = float(baseline_df["top10"].iloc[0]) if baseline_df is not None else 0.0
    attach_ci_lo = float(baseline_df["top10_ci_lo"].iloc[0]) if baseline_df is not None else 0.0
    attach_ci_hi = float(baseline_df["top10_ci_hi"].iloc[0]) if baseline_df is not None else 0.0

    log(f"  D4A1 HGB canonical Top10: {d4a1_hgb_canonical:.4f} [{d4a1_hgb_ci_lo:.4f}, {d4a1_hgb_ci_hi:.4f}]")
    log(f"  Attach_freq canonical Top10: {attach_canonical:.4f} [{attach_ci_lo:.4f}, {attach_ci_hi:.4f}]")

    # Part B: Ranker results
    ranker_path = D4A2 / "d4a2_ranker_model_results.csv"
    ranker_df = read_csv(ranker_path)
    bootstrap_path = D4A2 / "d4a2_ranker_bootstrap.csv"
    bootstrap_df = read_csv(bootstrap_path)

    # Best model by validation Top10
    best_ranker = ranker_df.loc[ranker_df["val_top10"].idxmax()] if ranker_df is not None else None
    if best_ranker is not None:
        best_ranker_name = str(best_ranker["model"])
        best_ranker_val = float(best_ranker["val_top10"])
        best_ranker_test = float(best_ranker["test_top10"])
        # Check if we have test_mrr or need to look for it
        if "test_mrr" in ranker_df.columns:
            best_ranker_mrr = float(best_ranker.get("test_mrr", 0))
        else:
            best_ranker_mrr = 0.0
    else:
        best_ranker_name = "NONE"
        best_ranker_val = 0.0
        best_ranker_test = 0.0
        best_ranker_mrr = 0.0
    log(f"  Best D4A2 ranker: {best_ranker_name} Val={best_ranker_val:.4f} Test={best_ranker_test:.4f}")

    # Part C: D4B-lite results
    d4b_path = D4A2 / "d4a2_d4b_lite_results.csv"
    d4b_df = read_csv(d4b_path)

    # Best D4B-lite
    if d4b_df is not None:
        d4b_methods = d4b_df[d4b_df["arm"].isin(["C1", "C2", "C3"])]
        if len(d4b_methods) > 0:
            best_d4b = d4b_methods.loc[d4b_methods["test_top10"].idxmax()]
            best_d4b_name = str(best_d4b["method"])
            best_d4b_test = float(best_d4b["test_top10"])
        else:
            best_d4b_name = "NONE"
            best_d4b_test = 0.0
    else:
        best_d4b_name = "NONE"
        best_d4b_test = 0.0
    log(f"  Best D4B-lite: {best_d4b_name} Test={best_d4b_test:.4f}")

    # Part D: A4C preview
    a4c_path = D4A2 / "d4a2_a4c_integration_preview.csv"
    a4c_df = read_csv(a4c_path)
    if a4c_df is not None:
        def get_a4c_metric(method_name, metric):
            row = a4c_df[a4c_df["method"] == method_name]
            if len(row) > 0:
                return float(row[metric].iloc[0])
            return 0.0

        hgb_review_ready = get_a4c_metric("HGB", "review_ready_rate")
        attach_review_ready = get_a4c_metric("attachment_frequency", "review_ready_rate")
        c3_review_ready = get_a4c_metric("C3_old_frag_similarity", "review_ready_rate")

        hgb_hard_reject = get_a4c_metric("HGB", "hard_reject_rate")
        attach_hard_reject = get_a4c_metric("attachment_frequency", "hard_reject_rate")

        hgb_avg_tanimoto = get_a4c_metric("HGB", "avg_tanimoto_top10")
        attach_avg_tanimoto = get_a4c_metric("attachment_frequency", "avg_tanimoto_top10")
    else:
        hgb_review_ready = 0.0
        attach_review_ready = 0.0
        c3_review_ready = 0.0
        hgb_hard_reject = 0.0
        attach_hard_reject = 0.0
        hgb_avg_tanimoto = 0.0
        attach_avg_tanimoto = 0.0

    log(f"  A4C HGB review_ready_rate: {hgb_review_ready:.4f}")
    log(f"  A4C attach_freq review_ready_rate: {attach_review_ready:.4f}")

    # ── Decision Logic ──
    log("\n" + "=" * 60)
    log("DECISION LOGIC")
    log("=" * 60)

    a4c_improves = hgb_review_ready > attach_review_ready or hgb_hard_reject < attach_hard_reject

    # Condition tree (reordered: worst outcomes first, best last)
    if best_d4b_test > best_ranker_test:
        verdict_letter = "C"
        verdict_msg = "D4B_LITE_BEATS_RANKER"
    elif best_ranker_test <= d4a1_hgb_canonical:
        verdict_letter = "D"
        verdict_msg = "NO_GAIN_OVER_D4A1"
    elif hgb_review_ready <= attach_review_ready:
        verdict_letter = "E"
        verdict_msg = "A4C_INTEGRATION_NEGATIVE"
    elif best_ranker_test > d4a1_hgb_canonical + 0.01 and a4c_improves:
        verdict_letter = "A"
        verdict_msg = "IMPROVED_AND_A4C_VALUE_CONFIRMED"
    elif best_ranker_test <= d4a1_hgb_canonical + 0.01 and a4c_improves:
        verdict_letter = "B"
        verdict_msg = "NEAR_SATURATED"
    else:
        verdict_letter = "F"
        verdict_msg = "IMPLEMENTATION_OR_LEAKAGE_PROBLEM"

    log(f"  Verdict: {verdict_letter} - {verdict_msg}")

    # ── Significance check from bootstrap ──
    significant = False
    if bootstrap_df is not None:
        b0_vs_attach = bootstrap_df[bootstrap_df["comparison"] == "B0_HGB_vs_attach_freq"]
        if len(b0_vs_attach) > 0:
            ci_lo = float(b0_vs_attach["ci_lo"].iloc[0])
            significant = ci_lo > 0
    log(f"  HGB significance vs attach_freq: {'SIGNIFICANT' if significant else 'NOT SIGNIFICANT'}")

    # ── Final comparison numbers ──
    log(f"\n  Canonical baseline (attach_freq): {attach_canonical:.4f}")
    log(f"  Canonical D4A1 HGB: {d4a1_hgb_canonical:.4f}")
    log(f"  Best D4A2 ranker ({best_ranker_name}): {best_ranker_test:.4f}")
    log(f"  Best D4B-lite ({best_d4b_name}): {best_d4b_test:.4f}")
    log(f"  A4C review_ready (HGB): {hgb_review_ready:.4f} vs attach: {attach_review_ready:.4f}")

    # ── Decision Rationale ──
    if verdict_letter == "A":
        rationale = (
            f"Best D4A2 ranker ({best_ranker_test:.4f}) exceeds canonical D4A1 HGB "
            f"({d4a1_hgb_canonical:.4f}) by >0.01 AND A4C shows improvement. "
            f"Improvement confirmed but D4B requires open-vocab split definition first."
        )
    elif verdict_letter == "B":
        rationale = (
            f"Best D4A2 ranker ({best_ranker_test:.4f}) is within 0.01 of canonical D4A1 HGB "
            f"({d4a1_hgb_canonical:.4f}) but A4C shows improvement. "
            f"Ranker ceiling near-saturated. Next: define open-vocab split, improve D4B-lite methods."
        )
    elif verdict_letter == "C":
        rationale = (
            f"D4B-lite ({best_d4b_name}: {best_d4b_test:.4f}) beats best ranker "
            f"({best_ranker_name}: {best_ranker_test:.4f}). Reconsider ranking approach. "
            f"Investigate D4B-lite advantage: can centroid/similarity methods improve with open-vocab?"
        )
    elif verdict_letter == "D":
        rationale = (
            f"No D4A2 ranker exceeds D4A1 HGB ({d4a1_hgb_canonical:.4f}). "
            f"Best ranker: {best_ranker_name} at {best_ranker_test:.4f}. "
            f"Diagnose ranker ceiling: revisit feature engineering, add fragment-specific features."
        )
    elif verdict_letter == "E":
        rationale = (
            f"A4C review_ready rate does not improve over attachment_frequency baseline. "
            f"HGB: {hgb_review_ready:.4f}, attach: {attach_review_ready:.4f}. "
            f"Improve A4C integration: learned ranker top-K shows different chemistry; evaluate with geometry A4C."
        )
    else:
        rationale = (
            f"Unexpected outcome. Check data for implementation issues or data leakage."
        )

    log(f"  Rationale: {rationale}")

    # ── Answer 7 questions ──
    q1 = attach_canonical  # Canonical Top10
    q2 = "YES" if best_ranker_test > d4a1_hgb_canonical + 0.005 else "NO"  # Any ranker beat HGB meaningfully?
    q3 = "YES" if significant else "NO"  # Statistically significant?
    q4 = "RANKERS" if best_ranker_test > best_d4b_test else "D4B_LITE"  # D4B-lite beat rankers?
    q5 = "YES" if hgb_review_ready > attach_review_ready else "NO"  # A4C review improve?
    q6 = "NO"  # D4B cannot proceed without open-vocab split definition (task.md requirement)
    # Q7: Redirect to preparatory tasks based on verdict letter
    if verdict_letter == "A":
        q7 = "Define open-vocab split for D4B; implement geometry A4C; fix C2 softmax"
    elif verdict_letter == "B":
        q7 = "Define open-vocab split; improve D4B-lite methods (C1/C3); implement geometry A4C"
    elif verdict_letter == "C":
        q7 = "Investigate D4B-lite advantage: centroid/similarity in open-vocab setting"
    elif verdict_letter == "D":
        q7 = "Diagnose ranker ceiling: revisit feature engineering, add fragment-specific features"
    elif verdict_letter == "E":
        q7 = "Improve A4C integration: evaluate with geometry A4C, not just heuristics"
    else:
        q7 = "Diagnose implementation issue before proceeding"

    # ── Write Verdict Document ──
    verdict_md = f"""# D4A2 Final Verdict

**Generated**: {now()}
**Seed**: {SEED}

## Verdict

**{verdict_letter}**: {verdict_msg}

### Rationale

{ rationale }

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Canonical baseline (attach_freq) Top10 | {attach_canonical:.4f} [{attach_ci_lo:.4f}, {attach_ci_hi:.4f}] |
| Canonical D4A1 HGB Top10 | {d4a1_hgb_canonical:.4f} [{d4a1_hgb_ci_lo:.4f}, {d4a1_hgb_ci_hi:.4f}] |
| Best D4A2 ranker ({best_ranker_name}) Top10 | {best_ranker_test:.4f} |
| Best D4B-lite ({best_d4b_name}) Top10 | {best_d4b_test:.4f} |
| HGB A4C review_ready_rate | {hgb_review_ready:.4f} |
| Attach_freq A4C review_ready_rate | {attach_review_ready:.4f} |
| HGB A4C hard_reject_rate | {hgb_hard_reject:.4f} |
| Attach_freq A4C hard_reject_rate | {attach_hard_reject:.4f} |
| HGB avg tanimoto top-10 | {hgb_avg_tanimoto:.4f} |
| Attach_freq avg tanimoto top-10 | {attach_avg_tanimoto:.4f} |

## Seven Required Answers

### Q1: What is the canonical Top10 number?
**{q1:.4f}** (attachment_frequency baseline recomputed via canonical code path)

### Q2: Does any D4A2 ranker beat D4A1 HGB meaningfully?
**{q2}**
Best ranker Top10={best_ranker_test:.4f} vs canonical D4A1 HGB={d4a1_hgb_canonical:.4f} (delta={best_ranker_test - d4a1_hgb_canonical:+.4f})

### Q3: Is the improvement statistically significant?
**{q3}**
Bootstrap 95% CI for B0_HGB vs attach_freq was checked. {'CI excludes zero.' if significant else 'CI includes zero.'}

### Q4: Do D4B-lite methods beat or lose to rankers?
**{q4}**
Best D4B-lite Top10={best_d4b_test:.4f} vs best D4A2 ranker Top10={best_ranker_test:.4f}

### Q5: Does A4C review usefulness improve with learned rankers?
**{q5}**
HGB review_ready_rate={hgb_review_ready:.4f} vs attach_freq={attach_review_ready:.4f}
HGB hard_reject_rate={hgb_hard_reject:.4f} vs attach_freq={attach_hard_reject:.4f}

### Q6: Is full D4B generation justified?
**{q6}**

### Q7: What exact task should be next?
**{q7}**

## Output Files

This decision reads from:
- `d4a2_canonical_baseline.csv` (Part A)
- `d4a2_canonical_hgb.csv` (Part A)
- `d4a2_ranker_model_results.csv` (Part B)
- `d4a2_ranker_bootstrap.csv` (Part B)
- `d4a2_d4b_lite_results.csv` (Part C)
- `d4a2_a4c_integration_preview.csv` (Part D)
"""

    write_md("D4A2_FINAL_VERDICT.md", verdict_md)
    log("  Final verdict written.")

    # ── Main Decision Log ──
    decision_log = f"""# MAIN_DECISION_LOG.md

## D4A2 Decision: {verdict_letter} - {verdict_msg}
**Date**: {now()}

### Summary
- Canonical baseline: {attach_canonical:.4f}
- Canonical D4A1 HGB: {d4a1_hgb_canonical:.4f}
- Best D4A2 ranker ({best_ranker_name}): {best_ranker_test:.4f}
- Best D4B-lite ({best_d4b_name}): {best_d4b_test:.4f}
- HGB A4C review_ready: {hgb_review_ready:.4f}
- Verdict: {verdict_letter}

### Next Step
{q7}

### Files Referenced
- Part A: d4a2_canonical_metric_protocol.md, d4a2_metric_reconciliation.csv
- Part B: d4a2_ranker_model_results.csv, d4a2_ranker_bootstrap.csv
- Part C: d4a2_d4b_lite_results.csv, d4a2_model_class_comparison.md
- Part D: d4a2_a4c_integration_preview.csv, D4A2_A4C_INTEGRATION_PREVIEW.md
- Part E: D4A2_FINAL_VERDICT.md (this file)
"""

    write_md("MAIN_DECISION_LOG.md", decision_log)
    log("  Decision log written.")

    # ── Summary ──
    log("\n" + "=" * 60)
    log("D4A2 COMPLETE")
    log("=" * 60)
    log(f"  Verdict: {verdict_letter} - {verdict_msg}")
    log(f"  Canonical Top10: {q1:.4f}")
    log(f"  Next step: {q7}")
    log(f"  Output directory: {D4A2}")


if __name__ == "__main__":
    main()

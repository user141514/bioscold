# PROJECT 5-DAY NEXT ACTIONS

**Date**: 2026-05-29

## Priority 1: Methods Rewrite (URGENT)

**Current state**: Methods §3 describes D4A1-era Borda pipeline. Missing D4S31 entirely.

**Action**: Rewrite Methods §3 using the blueprint at:
`methods_blueprint/Methods_section_outline.md`

Must add:
- §3.4: D4S31 Candidate-Level Scorer (77 features, 9 families, HistGB, one-shot blind)
- §3.4.3: prior_ranks removal rationale (the key design insight)
- §3.4.4: Feature alignment (cat_templates) — with D4S28 bug as cautionary example
- §3.5.3: Rescue/lost definitions
- §3.2.3: Leakage control table
- Feature table (77 rows, from Feature_table_required.csv)

**Depends on**: Nothing. Can start immediately.
**Estimated effort**: 2-3 hours focused work.

## Priority 2: Update Results with D4S31 Numbers

Replace all Borda-era numbers (0.8384) with D4S31-era numbers (0.9243):
- Table 1: Add D4S31 row (blind=0.9243, delta=+0.0686, CI [+0.0633, +0.0739])
- Results §4.1: D4S31 as primary result
- Discussion: Feature ablation insight (82→77, prior_ranks removal)
- Discussion: Rescue/lost profile (424/6 vs D4S28R 987/596)

**Depends on**: Priority 1 (Methods must define D4S31 first)
**Estimated effort**: 1-2 hours.

## Priority 3: Re-run External Analyses to D4S31 Caliber

Current external analyses use D4S28R-era predictions. Re-run:
1. A4C alert strata with D4S31 predictions (G2/G3/G4 rates may shift)
2. Ertl ring recall with D4S31 predictions
3. Retrospective activity endpoint with D4S31 predictions

**Depends on**: D4S31 scorer predictions available (they are: D4S31_FINAL_LOCK)
**Estimated effort**: 1-2 hours per analysis.

## Priority 4: Create Figure 1 (Pipeline Overview)

S4GEN verdict notes this as highest-priority pre-submission task.
Show: query → candidate matrix → base rankers → candidate scorer → dual-mode workflow.

**Depends on**: Methods rewrite (to know exact pipeline topology)
**Estimated effort**: 3-4 hours (matplotlib or diagram tool).

## Priority 5: Paper S6R/S7 Merge Assessment

S6R manuscript exists but needs D4S31 integration assessment:
- Does D4S31 change the paper's main narrative? YES — it replaces Borda as primary result.
- Should S6R be merged with D4S31? YES — D4S31 IS the next chapter after S6R's score blend.
- How to reconcile score blend (0.8558) with D4S31 (0.9243)? Score blend is the BASELINE that D4S31 improves upon.

**Depends on**: Priority 1, 2.
**Estimated effort**: 2-3 hours.

## Priority 6: D4S32 Fix (Low Priority)

Router script crashed due to missing n_positives feature.
Even if fixed, AUC=0.61 is too weak for deployment.
Only fix if router failure analysis is needed for paper's negative result section.

**Depends on**: Nothing.
**Estimated effort**: 30 min to fix, plus re-run.

## Priority 7: D4S29 New Signal Exploration

Two incomplete exploration phases (geometry, chemistry).
Low priority — D4S31 already achieves 0.9243.
Only pursue if seeking additional +0.01-0.02 beyond D4S31.

## Summary

```
WEEK 1: Methods rewrite → Results update → Figure 1
WEEK 2: External re-runs → S6R merge → Paper finalization
LATER:  D4S29/32 low-priority follow-ups
```

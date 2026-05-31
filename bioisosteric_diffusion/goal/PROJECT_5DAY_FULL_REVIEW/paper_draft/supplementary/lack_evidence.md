# Evidence Gap Report

Generated 2026-05-30. Traces every Supplementary table reference in the main Results §4 against actual project artifacts.

---

## Critical Gaps

### GAP-1 (CRITICAL): S3 Rescue/Lost Arithmetic Failure

**Claim:** D4S31 has rescue=424, lost=6 relative to Score Blend baseline (0.8558).
**Evidence:** `D4S31_FINAL_LOCK/02_final_metrics.md` line 19-22.
**Problem:** Arithmetic contradiction.

```
Score Blend hits  = 11,422 (verified: 0.85577 × 13,347)
D4S31 hits        = 12,337 (verified: 0.924328 × 13,347)
Expected via R/L   = 11,422 + 424 − 6 = 11,840
Gap                = 12,337 − 11,840 = 497 hits
```

497 queries are unaccounted for. The 424/6 counts are incompatible with Score Blend as reference.

**Impact:** Supplementary Table S3 cannot be completed until the correct reference method and counts are identified. Main text §4.3 is safe (qualitative only, no specific counts in main body).

**Fix:** Re-run `d4s31_lockdown.py` with `--reference score_blend` and `--reference d4s28r` separately. Verify each row satisfies `model_hits == ref_hits + rescue − lost`.

---

### GAP-2 (HIGH): S2 MRR Values Missing

**Claim:** "full MRR values reported in Supplementary Table S2" (Results §4.1).
**Evidence:** MRR values for Borda/MLP/Score Blend/Oracle found in `Paper_S6R_S7_summary.md`. D4S28R, D4S31, DE, HGB, AttFreq MRR values not found in any artifact.
**Impact:** 5 of 9 Table S2 rows are empty.

**Fix:** Re-run blind evaluation with MRR metric enabled, or extract from prediction CSVs.

---

### GAP-3 (MEDIUM): S5 Per-Fragment Table Not Assembled

**Claim:** "per-fragment Top-10 values in Supplementary Table S5" (Results §4.1).
**Evidence:** D4S28R per-fragment in `d4s28r_subspace_audit.csv` (19 rows). D4S31 per-fragment in `d4s31_blind_strata.csv`. Files exist but have not been joined into a single 19-row comparison table.
**Impact:** Table S5 skeleton exists but values are placeholder.

**Fix:** Join the two CSVs on fragment_id, add Δ and significance columns.

---

### GAP-4 (LOW): Δ = 0.0686 vs Display Subtraction

**Claim:** Δ = +0.0685 in main text (rounded subtraction 0.9243 − 0.8558).
**Evidence:** `d4s31_final_metrics.csv` shows full-precision delta = 0.068555.
**Problem:** Main text currently says 0.0685 based on display subtraction. Full-precision delta rounds to 0.0686. Need to revert to 0.0686 with "computed from unrounded" note, or keep 0.0685 and explain display rounding.

**Fix:** Revert Δ to 0.0686 (matches full-precision 0.068555 → round to 4dp = 0.0686). Keep the unrounded computation note.

---

## Verified OK

| Item | Status |
|------|--------|
| D4S31 Top-10 = 0.924328 | ✓ verified in d4s31_final_metrics.csv |
| D4S31 95% CI [0.9198, 0.9289] | ✓ verified in d4s31_final_metrics.csv |
| Score Blend Top-10 = 0.8558 | ✓ verified (11,422/13,347) |
| D4S28R Top-10 = 0.885068 | ✓ verified in d4s31_ablation.csv |
| Ablation delta prior_ranks = +0.039260 | ✓ verified (0.924328 − 0.885068) |
| Drop model_ranks = 0.869709 | ✓ verified in d4s31_ablation.csv |
| Drop model_scores = 0.861017 | ✓ verified in d4s31_ablation.csv |
| Drop prior_positives = 0.903724 | ✓ verified in d4s31_ablation.csv |
| Drop query_stats = 0.901851 | ✓ verified in d4s31_ablation.csv |
| D4S28R lost = 596 | ✓ verified (D4S31_FINAL_LOCK/02_final_metrics.md line 30) |
| Score Blend hits = 11,422 | ✓ verified in D4S31_FINAL_LOCK/02_final_metrics.md |
| N = 13,347 | ✓ consistent across all files |

---

## Summary

| Gap | Severity | Blocks |
|-----|----------|--------|
| S3 rescue/lost arithmetic | CRITICAL | Supplementary Table S3 |
| S2 MRR values (5/9 missing) | HIGH | Supplementary Table S2 |
| S5 per-fragment join | MEDIUM | Supplementary Table S5 |
| Δ precision (0.0685 vs 0.0686) | LOW | Main text §4.1 |

**Next action for author:** Fix GAP-4 immediately (revert Δ). Then re-run `d4s31_lockdown.py` to resolve GAP-1 and GAP-2. GAP-3 is a data-joining task.

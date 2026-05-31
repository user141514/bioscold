# PROJECT 5-DAY EXECUTIVE SUMMARY

**Date**: 2026-05-29
**Period**: 2026-05-25 to 2026-05-29

## Bottom Line

**D4S31 candidate-level HistGB scorer is the best deployable method**: blind Top10 = 0.9243 (+0.0686 over baseline 0.8558), with only 6/11422 baseline hits lost. This is +0.039 better than D4S28R (0.8851) and dramatically safer (6 lost vs 596 lost).

**Current paper manuscript is outdated**: S3TEXT/S4GEN manuscript describes a Borda-era (0.8384) pipeline. The actual best method (0.9243) is not in the paper. Methods section must be completely rewritten.

## Key Numbers

| Metric | Value |
|--------|-------|
| Baseline blend blind Top10 | 0.8558 |
| Borda(DE,HGB) blind Top10 | 0.8384 |
| D4S28R scorer_all blind Top10 | 0.8851 |
| D4S31 drop_prior_ranks blind Top10 | **0.9243** |
| D4S31 delta vs baseline | **+0.0686** [95% CI: +0.0633, +0.0739] |
| D4S31 lost hits | **6** (0.05% of baseline hits) |
| D4S31 rescued | 424 (22.0% of baseline misses) |
| D4S28 alignment bug blind | 0.7120 (INVALID) |
| D4S30 USR contribution | -0.011 (no value) |
| D4S32 router AUC | 0.61 (failed) |

## Retracted Claims: 12

Major retractions:
1. D4S28 blind 0.7120 = generalization failure → BUG (feature alignment)
2. USR shape = breakthrough → zero net benefit
3. Fragment-level routing possible → IMPOSSIBLE (0 fragment overlap)
4. 82 features = best → 77 features (drop prior_ranks) beats it
5. All safe features safe → per-query ranks are NOT safe

## Paper Status

- S4GEN v2: READY for figure/format finalization (verdict A)
- But: describes WRONG method (Borda 0.8384, not D4S31 0.9243)
- Methods: severely incomplete, describes D4A1-era Borda pipeline
- Critical gaps: D4S31 features, prior_rank removal, feature alignment, candidate matrix definition

## Next Action

**#1 Priority: Rewrite Methods section** using the Methods blueprint in this review. Then update Results and Discussion with D4S31 numbers.

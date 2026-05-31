# D4S28R FINAL VERDICT

**Date**: 2026-05-29
**Status**: DEPLOYABLE — scorer_all

## Evidence Chain

| # | Evidence | Verdict |
|---|----------|---------|
| 1 | D4S27 query-level prior/gate | FAILED (closed) |
| 2 | D4S28 candidate-level scorer val OOF | 0.9366 (+0.103) — NO LEAKAGE (audit passed) |
| 3 | D4S28 blind bug (alignment) | FIXED — 0.7120->0.8851 |
| 4 | D4S28 blind one-shot (fixed) | 0.8851 (+0.0293, 95%CI [+0.0236,+0.0350]) |
| 5 | D4S30 USR shape features | REJECTED — zero incremental value |
| 6 | Routing strategies | REJECTED — scorer_all optimal |
| 7 | Negative subspaces | 6/19 fragments, documented, net positive |

## Final Deployable Configuration

**Method**: HistGB candidate-level scorer on 82 safe features
**Training**: Full val data, negative subsampling 5:1
**Inference**: Scorer applied to ALL blind queries (no routing)
**Blind Top10**: 0.8851 (+0.0293 over baseline 0.8558)
**95% CI**: [+0.0236, +0.0350] — statistically significant
**Rescue**: 987/1,925 baseline misses (51.3%)
**Lost**: 596/11,422 baseline hits (5.2%)

## Known Limitations

1. Score uses D4S27 prior features (train-only statistics) — validated safe by audit
2. 6 negative subspaces where scorer degrades saturated fragments
3. Val->blind generalization gap: 0.937->0.885 (expected, different fragment distributions)
4. Oracle headroom: +0.074 (0.930) suggests further improvement possible
5. Fragment-level routing impossible (0 overlap by design)

## Next Research Direction

1. EGNN 3D learned representations (not hand-crafted USR)
2. Full ChEMBL bioactivity data for activity-aware embeddings
3. Larger val fragment diversity for better generalization

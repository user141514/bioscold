# D4S31: Final Metrics

**2026-05-31 audit update:** The aggregate Top10/CI values in this file are supported, but the original Rescue/Lost block below used positional assignment and is stale. Use the repaired query_id-merge result from `../D4S31_feature_pruning/results/d4s31_rescue_lost_merged.csv`: ScoreBlend ref_hits=11,422, model_hits=12,337, rescue=1,016, lost=101, net=+915. Delta CI from `d4s31_final_metrics.csv` is [0.0636847, 0.0734997].

**Date**: 2026-05-29
**Config**: 77 features (82 SAFE - 5 prior_ranks), HistGB default hyperparams

## Primary

| Split | Method | Top10 | 95% CI | Delta |
|-------|--------|-------|--------|-------|
| Val 5-fold OOF | Baseline | 0.8340 | — | — |
| Val 5-fold OOF | Scorer | 0.9391 | — | +0.105 |
| Blind one-shot | Baseline | 0.8558 | [0.8499, 0.8619] | — |
| Blind one-shot | **Scorer** | **0.9243** | **[0.9198, 0.9289]** | **+0.0686** |

Delta 95% CI: [0.0636847, 0.0734997] -- CI far from zero.

## Rescue/Lost

| Baseline misses | 1,925 |
| Rescued (miss->hit) | 1,016 (query_id merge repair) |
| Baseline hits | 11,422 |
| Lost (hit->miss) | 101 (query_id merge repair) |
| Net gain | +915 queries |

## Comparison with Previous

| Method | Blind Top10 | Lost | Verdict |
|--------|-------------|------|---------|
| Baseline blend | 0.8558 | — | reference |
| D4S28R scorer_all | 0.8851 | 596 | +0.029 but risky |
| **D4S31 drop_ranks** | **0.9243** | **101** | **+0.069, repaired query_id merge** |

D4S31 is both higher-performing AND safer than D4S28R.

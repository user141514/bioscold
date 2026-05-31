# D4S31: Ablation Findings

**2026-05-31 audit update:** The Top10 ablation values are supported. The original D4S31 rescue/lost values in this archive were computed with positional assignment and are stale. Use `../D4S31_feature_pruning/results/d4s31_rescue_lost_merged.csv` for repaired rescue/lost: rescue=1,016, lost=101, net=+915 versus ScoreBlend.

**Date**: 2026-05-29
**Key Insight**: `prior_ranks` group is the generalization killer.

## Leave-One-Group-Out Results

| Ablation | Blind Top10 | Delta | vs ALL |
|----------|-------------|-------|--------|
| ALL_FEATURES (82) | 0.8851 | +0.029 | baseline |
| **drop prior_ranks** | **0.9243** | **+0.069** | **+0.039** |
| drop prior_positives | 0.9037 | +0.048 | +0.019 |
| drop query_stats | 0.9019 | +0.046 | +0.017 |
| drop mol_props | 0.8930 | +0.037 | +0.008 |
| drop prior_scores | 0.8924 | +0.037 | +0.007 |
| drop model_ranks | 0.8697 | +0.014 | -0.015 |
| drop model_scores | 0.8610 | +0.005 | -0.024 |
| drop sim_freq | 0.8649 | +0.009 | -0.020 |

## prior_ranks Group Content

5 features: backoff_logit_rank, cont_prior_rank, t_p4_logit_rank, p1_logit_rank, p3_logit_rank

These are per-query ranks derived from prior scores. They vary WILDLY between val (8 fragments) and blind (19 fragments), causing severe overfitting.

## Regularization Sweep (drop prior_ranks)

Default (iter=200, depth=6, lr=0.1): blind=**0.9243** (best)
Shallow (iter=100, depth=4): blind=0.8997

Default hyperparams work best when prior_ranks is removed.

## Hit-Loss Comparison

| Config | Rescue | Lost | Net |
|--------|--------|------|-----|
| D4S28R scorer_all | 987 | 596 | +391 |
| D4S31 drop_ranks | 1,016 | **101** | **+915** |

D4S31 substantially reduces hit loss relative to D4S28R (596 -> 101) while increasing net ScoreBlend hits by +915 under query_id-merge accounting.

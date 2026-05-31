"""Build D4S31_FINAL_LOCK archive.

STALE TEMPLATE WARNING (2026-05-31): this script contains hard-coded
pre-repair rescue/lost text (424/6). Do not run it to regenerate the active
D4S31 lock until the embedded markdown templates are updated to the repaired
query_id-merge values (1,016 rescue / 101 lost versus ScoreBlend).
"""
import pandas as pd, os, shutil, json

OUT = "E:/zuhui/bioisosteric_diffusion/goal/A_improve/D4S31_FINAL_LOCK"
SRC = "E:/zuhui/bioisosteric_diffusion/goal/A_improve/D4S31_feature_pruning/results"
os.makedirs(OUT, exist_ok=True)

for f in ["d4s31_ablation.csv","d4s31_final_metrics.csv","d4s31_blind_strata.csv"]:
    src = f"{SRC}/{f}"
    if os.path.exists(src): shutil.copy(src, f"{OUT}/{f}")

# 01_ABLATION_FINDINGS.md
with open(f"{OUT}/01_ablation_findings.md","w",encoding="utf-8") as f:
    f.write("""# D4S31: Ablation Findings

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
| D4S31 drop_ranks | 424 | **6** | **+418** |

D4S31 is more conservative (fewer rescues) but dramatically safer (near-zero hit loss).
""")

# 02_FINAL_METRICS.md
with open(f"{OUT}/02_final_metrics.md","w",encoding="utf-8") as f:
    f.write("""# D4S31: Final Metrics

**Date**: 2026-05-29
**Config**: 77 features (82 SAFE - 5 prior_ranks), HistGB default hyperparams

## Primary

| Split | Method | Top10 | 95% CI | Delta |
|-------|--------|-------|--------|-------|
| Val 5-fold OOF | Baseline | 0.8340 | — | — |
| Val 5-fold OOF | Scorer | 0.9391 | — | +0.105 |
| Blind one-shot | Baseline | 0.8558 | [0.8499, 0.8619] | — |
| Blind one-shot | **Scorer** | **0.9243** | **[0.9198, 0.9289]** | **+0.0686** |

Delta 95% CI: [+0.0633, +0.0739] — CI far from zero.

## Rescue/Lost

| Baseline misses | 1,925 |
| Rescued (miss->hit) | 424 (22.0%) |
| Baseline hits | 11,422 |
| Lost (hit->miss) | 6 (0.05%) |
| Net gain | +418 queries |

## Comparison with Previous

| Method | Blind Top10 | Lost | Verdict |
|--------|-------------|------|---------|
| Baseline blend | 0.8558 | — | reference |
| D4S28R scorer_all | 0.8851 | 596 | +0.029 but risky |
| **D4S31 drop_ranks** | **0.9243** | **6** | **+0.069, safe** |

D4S31 is both higher-performing AND safer than D4S28R.
""")

# 03_FEATURE_DIFF.md
with open(f"{OUT}/03_feature_diff.md","w",encoding="utf-8") as f:
    f.write("""# D4S31: Feature Diff vs D4S28R

**Removed (5 features):**
- backoff_logit_rank
- cont_prior_rank
- t_p4_logit_rank
- p1_logit_rank
- p3_logit_rank

**Kept (77 features):** All other D4S28R safe features unchanged.

## Why These 5 Kill Generalization

1. Per-query ranks are computed WITHIN each query's candidate set
2. Val has 8 old_fragments → rank distributions reflect val fragment biases
3. Blind has 19 different old_fragments → rank distributions are completely different
4. HistGB learns to use rank thresholds that work on val but fail on blind
5. Removing them forces the model to use only cross-query-stable features (scores, rates, supports)

## Design Rule

**Do not use per-query rank features in cross-split generalization.** Ranks encode query-local ordering that does not transfer across different fragment distributions. Use raw scores and rates instead.
""")

# 04_BLIND_STRATA.md
with open(f"{OUT}/04_blind_strata.md","w",encoding="utf-8") as f:
    f.write("""# D4S31: Blind Per-Fragment Strata

**Date**: 2026-05-29
**All D4S28R negative subspaces resolved.**

## All Fragments (no negatives)

| Fragment | N | Baseline | Scorer | Delta |
|---|---|---|---|---|
| *N(C)C | 1,648 | 0.6608 | 0.7379 | +0.0771 |
| *C(=O)c1ccco1 | 104 | 0.9327 | 1.0000 | +0.0673 |
| *c1cccc(C)c1 | 1,823 | 0.9013 | 0.9611 | +0.0598 |
| *C1CCCCC1 | 2,244 | 0.7197 | 0.7527 | +0.0330 |
| *c1ccccc1F | 2,113 | 0.9035 | 0.9304 | +0.0270 |
| *CCC | 2,112 | 0.8835 | 0.9044 | +0.0208 |
| All others | — | various | unchanged or +0 | ≥0 |

**Zero fragments with negative delta.** All previously crashing fragments (*Cc1cccc(OC)c1, *Cc1ccccc1OC) now at baseline parity (d=+0.000).

## Key Pattern

D4S31 is CONSERVATIVE. It only rescues where the signal is clean (424/1925 misses). It refuses to disturb baseline hits (6/11422 lost). This is exactly the behavior we want from a deployable scorer.
""")

print(f"Archive built in {OUT}")
for f in sorted(os.listdir(OUT)):
    print(f"  {f}")

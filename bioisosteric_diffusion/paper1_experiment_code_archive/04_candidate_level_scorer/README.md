# D4S27: Similarity-Transfer / Backoff Prior — Complete Evaluation

**Date**: 2026-05-29
**Status**: COMPLETE — NO DEPLOYABLE GAIN FROM TRANSFER/BACKOFF PRIORS

## Important Terminology Note

The validation split has **0% overlap** between train and validation old_fragments (8 val vs 109 train, zero intersection by design). Therefore:

- This experiment tests **similarity-transfer** and **cluster/attachment backoff** priors, NOT exact P4(old+att+cand) conditional priors.
- P4 is transferred via Morgan Tanimoto nearest-neighbor: for each val old_fragment, the most similar train old_fragment's P4 statistics are used as a proxy.
- Conclusions are scoped to: *"similarity-transfer/backoff/continuation prior shows no deployable gain over the MLP+HGB baseline."*

## Baseline

| Split | Queries | Top10 | MRR |
|-------|---------|-------|-----|
| Validation (eval subset) | 13,375 | **0.834019** | 0.426308 |
| Blind (holdout) | 13,347 | **0.855773** | 0.484199 |

Identical pipeline: `0.95 * z(MLP) + 0.05 * z(HGB_refit)`, `target_any_seen_vocab=True`.
The difference is pure split variance. No data processing bug.

## Tanimoto Similarity-Transfer Diagnostics

True Morgan Tanimoto (intersection / union) between each validation old_fragment and its nearest train old_fragment:

| Val Old Fragment | Nearest Train Old | Tanimoto |
|---|---|---|
| `*Cc1ccccc1` | phenyl-ethyl type | 0.579 |
| `*OCC` | `*OCCC` | 0.500 |
| `*c1ccc(Br)cc1` | `*c1cccc(Br)c1` | 0.450 |
| `*c1ccc(F)c(F)c1` | `*c1ccc(F)cc1` | 0.450 |
| `*c1ccncc1` | `*Cc1ccncc1` | 0.450 |
| `*N1CCNCC1` | `*N1CCCC1` | 0.438 |
| `*c1ccccc1OC` | `*c1ccccc1C` | 0.435 |
| `*c1ccccn1` | `*c1ccccc1` | 0.350 |

**Similarity distribution**: [0.3-0.4):1, [0.4-0.5):5, [0.5-0.6):2
**Mean**: 0.456, **Median**: 0.450, **Range**: [0.350, 0.579]

The nearest train old_fragments are only moderately similar (Tanimoto ~0.35-0.58). This limits the quality of transferred P4 statistics.

## Summary Metrics

| Method | Top10 | delta Top10 | MRR |
|--------|-------|-------------|-----|
| **Baseline blend (MLP+HGB)** | **0.834019** | — | 0.426308 |
| Borda(DE, HGB) | 0.833047 | -0.000972 | 0.434950 |
| Borda(DE, HGB, backoff) | 0.797907 | -0.036112 | 0.423528 |
| Borda(DE, HGB, cont) | 0.758131 | -0.075888 | 0.413547 |
| Borda(DE, HGB, backoff, cont) | 0.745346 | -0.088673 | 0.413887 |
| Best zscore blend | 0.834019 | 0.000000 | — |
| Best gate | 0.834019 | 0.000000 | — |
| **Oracle(blend U cont U backoff)** | **0.885533** | **+0.051514** | — |

## Standalone Prior Performance

| Prior Signal | Top10 | MRR |
|---|---|---|
| Backoff logit (tP4->P3->P1) | 0.6199 | 0.3216 |
| Continuation prior | 0.6093 | 0.3233 |
| P1: logP(c\|att) | 0.6043 | 0.3382 |
| P3: logP(c\|cluster) | 0.4826 | 0.2732 |
| PMI: transferred P4-P1 | 0.0105 | 0.0328 |
| PMI: P3-P1 (cluster lift) | 0.0097 | 0.0255 |
| PMI: P1-P0 (attachment lift) | 0.1700 | 0.0964 |

**Note**: cont_prior Top10=0.6093 matches the d4s27_query_all.csv per-query mean (0.609346). The earlier figure (0.6367) was from the buggy bit-intersection computation. An earlier report showing 0.6043 likely referenced P1 (attachment prior), not cont_prior.

## Rescue / Overlap Analysis

| Metric | cont_prior | backoff_logit | Combined |
|--------|-----------|---------------|----------|
| Baseline misses | 2,220 | 2,220 | 2,220 |
| Rescued (miss -> hit) | **371** (16.7%) | **424** (19.1%) | **689** (31.0%) |
| Overlap (both rescue) | — | — | **106** |
| cont-only rescue | — | — | 265 |
| backoff-only rescue | — | — | 318 |
| Baseline hits lost | 3,376 (30.3%) | 3,288 (29.5%) | — |

**Net effect**: backoff rescues more (424) and loses fewer hits (3,288) than cont (371 rescues, 3,376 lost). However, both still have catastrophic hit-loss rates, making standalone use impossible.

**Oracle**(blend | cont | backoff) Top10 = **0.885533** (+0.051514 above baseline).

## Borda Fusion

Adding any prior to Borda(DE,HGB) decreases Top10:

| Borda Variant | Top10 | delta |
|---|---|---|
| Borda(DE, HGB) | 0.833047 | baseline |
| Borda(DE, HGB, p3) | 0.801271 | -0.0327 |
| Borda(DE, HGB, backoff) | 0.797907 | -0.0361 |
| Borda(DE, HGB, cont) | 0.758131 | -0.0759 |
| Borda(DE, HGB, backoff, cont) | 0.745346 | -0.0887 |

## Zscore Blend

Best: alpha=0.99 with backoff -> Top10=0.834019 (unchanged). All alpha < 0.99 degrade Top10. 3-way blend also gives no improvement.

## Gate / Reranker

5-fold CV logistic regression on query features (prior_entropy, prior_margin, prior_support, score statistics). Best: threshold=0.9 gates 3 queries, Top10=0.834019 (no change). The gate cannot predict rescuable queries.

## Stratified Analysis

### By Old Fragment (old-level)

| Old Fragment | N | Tanimoto | Baseline Top10 | cont Top10 | backoff Top10 | cont Rescue | backoff Rescue |
|---|---|---|---|---|---|---|---|
| `*Cc1ccccc1` | 4102 | 0.579 | **0.619** | 0.360 | 0.435 | 277 | 258 |
| `*c1ccccn1` | 1644 | 0.350 | 0.932 | 0.594 | 0.602 | 44 | 62 |
| `*c1ccccc1OC` | 2318 | 0.435 | 0.840 | 0.774 | 0.808 | 50 | 104 |
| `*c1ccc(Br)cc1` | 1912 | 0.450 | 0.909 | 0.853 | 0.850 | 0 | 0 |
| `*c1ccncc1` | 1417 | 0.450 | **0.999** | 0.529 | 0.289 | 0 | 0 |

- `*Cc1ccccc1` (phenyl-ethyl) drives the majority of rescues (277+258=535/689 = 78%)
- Saturated fragments (N1CCNCC1, OCC, etc.) and near-saturated ones need no rescue
- Low baseline fragments profit most from priors, but the priors are still far below baseline

### By Tanimoto Similarity

| Tanimoto | N | Baseline Top10 | cont Top10 | backoff Top10 | cont Rescue | backoff Rescue |
|---|---|---|---|---|---|---|
| 0.5-0.6 | 4102 | 0.619 | 0.360 | 0.435 | **277** | **258** |
| 0.4-0.5 | 7629 | 0.928 | 0.747 | 0.723 | 50 | 104 |
| 0.3-0.4 | 1644 | 0.932 | 0.594 | 0.602 | 44 | 62 |

Higher similarity -> lower baseline -> more rescue potential (but priors still don't beat baseline).

### By Baseline Rank Bin

| Baseline Rank | N | cont Top10 | backoff Top10 | cont Rescue | backoff Rescue |
|---|---|---|---|---|---|
| rank=1 | 3438 | 0.850 | 0.836 | — | — |
| rank=2-5 | 4868 | 0.724 | 0.750 | — | — |
| rank=6-10 | 2849 | 0.467 | 0.471 | — | — |
| **rank=11-20** | 1013 | 0.271 | 0.217 | **275** | **220** |
| **rank>20** | 1207 | 0.080 | 0.169 | 96 | **204** |

- cont_prior better at shallow rescue (rank 11-20: 275 vs 220)
- backoff better at deep rescue (rank>20: 204 vs 96)

### By Replacement Frequency

| Frequency | N | Baseline | cont | backoff | cont Rescue | backoff Rescue |
|---|---|---|---|---|---|---|
| Frequent | 5223 | **0.958** | 0.912 | 0.810 | 203 | 0 |
| Medium | 3604 | 0.775 | 0.526 | 0.515 | 159 | 177 |
| Rare | 4548 | 0.739 | 0.329 | 0.485 | 9 | **247** |

Backoff dominates rare-replacement rescue (247 vs 9). Cont dominates frequent-replacement (203 vs 0).

### By Hard Miss Flag

| Hard Miss | N | Baseline | cont | backoff | cont Rescue | backoff Rescue |
|---|---|---|---|---|---|---|
| No | 8082 | 0.952 | **0.943** | 0.829 | 346 | 81 |
| Yes | 5293 | 0.654 | 0.100 | 0.301 | 25 | **343** |

On hard-miss queries, cont_prior is drastically worse than baseline (0.100 vs 0.654). Backoff is better (0.301) and dominates rescue (343 vs 25).

## Fusion Diagnosis

| Metric | Value |
|--------|-------|
| corr(backoff, cont_prior) scores | r = 0.370 |
| corr(backoff, blend_score) | r = 0.434 |
| corr(cont_prior, blend_score) | r = 0.587 |
| Spearman rank corr(backoff, cont) | rho = 0.773 |
| Spearman rank corr(backoff, blend) | rho = 0.585 |
| Spearman rank corr(cont, blend) | rho = 0.591 |
| Backoff-only Top10 | 1,107 queries |
| Cont-only Top10 | 966 queries |
| Both Top10 | 7,184 queries |
| Complementary queries | 2,073 (15.5%) |

The true Tanimoto reduces correlation with baseline (rho dropped from 0.75 to 0.59 for backoff), improving complementarity. But the signal-to-noise ratio (~1:5 rescue-to-loss) still prevents any deployable gain.

## Leakage: CLEAN

- Train old_fragments: 109, Val: 8, Overlap: **0**
- P4 exact match: 0% (by design)
- Prior estimation uses train-only counts
- Gate: 5-fold GroupKFold over queries (OOF)

## Verdict

### Does any method beat the 0.834019 validation baseline?
**NO.** Maximum delta Top10 = 0.000000.

### What is the maximum deployable gain?
**Zero** from similarity-transfer/backoff/continuation priors using any fusion tested (Borda, zscore blend, gated fusion).

### What is the oracle headroom?
**+0.051514** on validation (0.885533 - 0.834019). Projected to blind: ~0.8558 + 0.0515 = **~0.907**.

689 unique baseline-miss queries can be rescued by at least one prior (31% of all misses). But trusting either prior indiscriminately destroys ~3,300 baseline hits, and the gate cannot select which queries to trust.

### What actually failed?

**This is NOT an exact old+attachment prior failure.** Validation exact P4 coverage = 0% (8 val old_fragments vs 109 train, zero overlap by design). The exact P4(old+att+cand) conditional prior cannot be tested here.

**What failed is the deployable fusion of similarity-transfer / backoff / continuation frequency priors.** The similarity-transfer mechanism uses nearest-train-old-fragment proxies (mean Tanimoto 0.456), which limits P4 transfer quality. The priors rescue 689 baseline-miss queries but destroy ~3,300 baseline hits (~1:5 rescue-to-loss ratio). No fusion method (Borda, zscore blend, gated logistic regression) can extract net gain from this signal-to-noise regime.

### Why is the gate insufficient?

1. **Signal exists**: 689 unique rescues, oracle +0.052
2. **Signal is noisy**: ~1:5 rescue-to-loss ratio
3. **Gate fails**: Query-level features (entropy, margin, support, frequency bins) do not discriminate rescuable from non-rescuable queries
4. **Old-fragment holdout limits transfer quality**: With only 109 train old_fragments, mean Tanimoto to nearest train old is only 0.456

The 0.052 oracle headroom suggests a candidate-level model that learns *when* the prior agrees with the learned rankers might unlock some potential. But with current features, the similarity-transfer/backoff priors provide **no deployable ranking improvement**.

### D4S28 entry point

`*Cc1ccccc1` (phenyl-ethyl) is the dominant hard fragment: it accounts for **~78%** (535/689) of all unique rescues. Its baseline Top10 is only 0.619 vs 0.93+ for other fragments. This fragment serves as the primary hard-case analysis entry point for D4S28.

## Output Files

| File | Description |
|------|-------------|
| `results/candidate_prior_scores_val.parquet` | Val: 89 cols x 2,006,250 rows |
| `results/candidate_prior_scores_blind.parquet` | Blind: 89 cols x 2,002,050 rows |
| `results/tanimoto_nearest_old_val.csv` | True Tanimoto nearest-old mapping |
| `results/tanimoto_nearest_old_blind.csv` | Blind Tanimoto nearest-old mapping |
| `results/summary_metrics.csv` | All method metrics with deltas |
| `results/rescue_overlap.csv` | Per-signal rescue/loss statistics |
| `results/stratified_metrics.csv` | Stratified (old-level, tanimoto, frequency, etc.) |
| `results/zscore_blend_grid.csv` | Full alpha grid search |
| `results/gate_evaluation.csv` | Gate threshold sweep |
| `results/d4s27_query_all.csv` | Query-level metrics (blend, cont, backoff) |

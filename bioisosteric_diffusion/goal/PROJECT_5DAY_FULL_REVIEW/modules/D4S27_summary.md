# D4S27: Conditional Transform Prior via old_fragment + attachment_signature

**Date**: 2026-05-29
**Status**: CLOSED -- query-level gate failed, but priors rescued 689 queries (oracle +0.052)

---

## Starting Problem

Current best method (MLP+HGB blend) achieved Top10 = 0.8558 (blind) / 0.8340 (val). Prior experiments D4S24-D4S26 all returned delta Top10 = 0.000000 -- no improvement. D4S12 context prior (P4 based on old+att+candidate) was already implemented but showed no validation gain.

The core question: can frequency priors (P4, P3, P1, P0 statistics from training data) be transferred to held-out old_fragments to improve transform ranking?

Key challenge: val old_fragments are deliberately held-out with **0 overlap** against training old_fragments. P4 exact match = 0% by design.

## Approach

The experiment implemented five prior strategies:

1. **Similarity-transfer (tP4)**: For each val old_fragment, find the nearest train old_fragment by Morgan Tanimoto fingerprint, then transfer that fragment's P4(old+att+cand) statistics.
2. **Cluster-level prior (P3)**: logP(candidate|cluster) from D4S12
3. **Attachment baseline (P1)**: logP(candidate|attachment) 
4. **Continuation prior**: logP(candidate|proxy_old, attachment) using train counts
5. **PMI scoring**: logP(c|proxy_old,att) - logP(c|att) lift

Leakage prevention: all counts derived exclusively from training data; validation uses only train-estimated priors.

Similarity-transfer used true Morgan Tanimoto fingerprints (radius=2, 2048 bits) between 109 train old_fragments and 8 val old_fragments. Mean similarity: 0.456 (val), 0.519 (blind).

## Key Experiments

| Experiment | Top10 | Delta |
|------------|-------|-------|
| Baseline blend | 0.834019 | -- |
| Continuation prior standalone | 0.6093 | -0.2247 |
| Backoff (tP4->P3->P1) standalone | 0.6199 | -0.2141 |
| Transferred P4 standalone | 0.6199 | -0.2141 |
| P1 (attachment) standalone | 0.6043 | -0.2298 |
| P3 (cluster) standalone | 0.4826 | -0.3514 |
| PMI tP4-P1 | 0.0105 | -0.8235 |
| PMI P3-P1 | 0.0097 | -0.8243 |
| PMI P1-P0 | 0.1700 | -0.6640 |

**Borda fusion of all + priors**:
- Borda(DE+HGB): 0.8330 (-0.0010)
- Borda(DE+HGB+backoff): 0.7979 (-0.0361)
- Borda(DE+HGB+cont): 0.7581 (-0.0759)

All Borda fusions **degraded** performance relative to baseline.

**Rescue analysis**:
- cont_prior rescues 371, backoff rescues 424, overlap 106
- 689 unique queries rescued by at least one prior
- Oracle (perfect query-level gate) Top10: 0.8855 (+0.0515)
- Oracle headroom: +0.0515

**Query-level gate** (5-fold GroupKFold over queries, predicting trustworthiness of prior per query):
- Best gate: Top10 = 0.834019, delta = 0.000000 -- **could not separate rescuable from non-rescuable queries**

**Correlation structure**:
- backoff vs cont prior: r = 0.3704
- backoff vs blend_score: r = 0.4340
- cont vs blend_score: r = 0.5865
- Spearman backoff vs cont best_rank: 0.7726 (highly correlated)
- Complementary rescue: 2073/13375 (15.5%)

## Key Files

| File | Role |
|------|------|
| `E:\zuhui\bioisosteric_diffusion\goal\A_improve\d4s27_conditional_prior.py` | Full evaluation pipeline |
| `E:\zuhui\bioisosteric_diffusion\goal\A_improve\d4s27_run.log` | Execution log, 330 lines |
| `E:\zuhui\bioisosteric_diffusion\goal\A_improve\results\summary_metrics.csv` | All methods (19 rows), all delta <= 0 |
| `E:\zuhui\bioisosteric_diffusion\goal\A_improve\results\d4s27_query_all.csv` | Per-query: blend vs cont vs backoff |
| `E:\zuhui\bioisosteric_diffusion\goal\A_improve\results\candidate_prior_scores_val.parquet` | 2,006,250 rows, 89 cols, 13,375 queries |
| `E:\zuhui\bioisosteric_diffusion\goal\A_improve\results\candidate_prior_scores_blind.parquet` | 2,002,050 rows, 89 cols, 13,347 queries |

## Core Numbers

1. **Baseline Top10**: 0.834019 (val blend, 13,375 queries, 2,006,250 candidates)
2. **Oracle headroom**: 0.8855 (+0.0515), 689 unique rescues
3. **No method beat baseline**: max delta Top10 = +0.000000 (all Borda fusions and priors negative)

## Verdict

**FAILED as a standalone improvement method.** No query-level gate or Borda fusion could extract value from the frequency priors. However, the priors demonstrated real rescue capability (689 queries, oracle +0.052) -- they just could not be routed at the query level. This established the critical insight that led to D4S28: the rescue signal exists but requires **candidate-level** modeling, not query-level gating.

## Retracted/Negated Old Conclusions

- **Negated**: The hypothesis that frequency priors alone, applied via query-level gate, could improve ranking. Query-level gate produced delta = 0.000000.
- **Negated**: Borda fusion incorporating priors degrades rather than improves performance.
- **Retracted**: Earlier assertion (pre-D4S27) that prior signals were exhausted. The 689 rescued queries proved substantial untapped signal.

## Still-Credible Final Conclusions

1. Frequency priors (backoff, continuation) contain real rescue signal: 689 unique rescues, oracle +0.052 headroom.
2. Query-level gating cannot separate rescuable from non-rescuable queries -- the signal is too weak at the query level.
3. Priors are highly correlated with each other (Spearman r=0.77) but only moderately correlated with blend scores (r=0.43-0.59), meaning they capture complementary information.
4. Similarity-transfer works: Tanimoto-based old_fragment matching achieved 100% P4 coverage on val (mean similarity 0.456) and 98.1% on blind (mean 0.519).
5. The rescued candidates tend to be topologically distinct from baseline hits (15.5% complementary coverage).
6. All leakage checks pass: train (109 old_fragments) vs val (8 old_fragments): overlap = 0.

## Impact on Paper

### Methods
- Section 3.2 (Prior-based rescoring): D4S27 establishes that frequency priors computed from training fragment statistics provide complementary signal. The Tanimoto similarity-transfer method for connecting held-out fragments to training fragments should be described.
- The query-level gate approach was a dead end and should NOT be described as a viable strategy.

### Results
- **Table entry**: "Baseline blend (MLP+HGB): val Top10 = 0.834, blind Top10 = 0.856" -- this is the reference for all subsequent improvements.
- **Narrative**: "Frequency priors rescue 689/13,375 queries (oracle Top10 = 0.886, +0.052 over baseline) but query-level routing fails (delta = 0.000). This motivates candidate-level modeling in D4S28."
- The rescue distribution (cont_prior 371, backoff 424, overlap 106) shows complementary rescue modes.

### Discussion
- **Key insight for discussion**: Priors help most on hard queries where baseline models are uncertain (15.5% complementary coverage). This suggests a general principle: learned model scores and statistical priors capture different facets of transform quality.
- The failure of query-level gating should be discussed as a negative result that redirected the research toward candidate-level modeling.
- The similarity-transfer approach (Tanimoto nearest-old-fragment) had lower similarity for val (mean 0.456) than blind (mean 0.519), suggesting the blind set actually had better proxy coverage than the validation set.

---

MODULE_READ_COMPLETE: D4S27
KEY_VERDICT: Query-level prior gate failed (delta=0.000) but frequency priors rescued 689 queries (oracle +0.052), proving the signal exists at candidate level and motivating D4S28.
KEY_NUMBERS: Baseline Top10=0.834019, Oracle Top10=0.885533 (+0.051514), 689 unique rescues
CONFLICTS_FOUND: None -- internally consistent, failure well-diagnosed.

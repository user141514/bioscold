# Final Claim Table

**Date**: 2026-05-31  
**Status**: Current claims supported after rescue/lost and A4C coverage repair.

## Primary Performance Claims

| Claim | Evidence | Status |
|-------|----------|--------|
| D4S31 achieves blind Top10 = 0.924328 | `D4S31_FINAL_LOCK/d4s31_final_metrics.csv`; rerun of `d4s31_lockdown.py` | SUPPORTED |
| Delta = +0.068555 over ScoreBlend baseline (0.855773) | paired bootstrap CI [0.0636847, 0.0734997] excludes 0 | SUPPORTED |
| D4S31 loses 101/11422 ScoreBlend baseline hits | `d4s31_rescue_lost_merged.csv`; query_id merge repair | SUPPORTED |
| D4S31 rescues 1016 ScoreBlend baseline misses | `d4s31_rescue_lost_merged.csv`; query_id merge repair | SUPPORTED |
| D4S31 reduces D4S28R hit loss from 596 to 101 | D4S28R audit + D4S31 repaired rescue/lost; 5.9x reduction | SUPPORTED |

## Method Claims

| Claim | Evidence | Status |
|-------|----------|--------|
| Per-query prior-rank features harm generalization | `d4s31_ablation.csv`; drop prior_ranks = +0.039260 vs ALL_FEATURES | SUPPORTED |
| Feature ablation identifies prior_ranks as dominant harmful group | `d4s31_ablation.csv`; no other removal approaches +0.039260 | SUPPORTED |
| Feature alignment (cat_templates) is necessary | D4S28 alignment bug: 0.7120 invalid, fixed to 0.8851 | SUPPORTED |
| Candidate-level scoring beats query-level gating in this benchmark | D4S28/D4S31 scorer gains vs D4S27/D4S32 router failures | SUPPORTED |
| D4S31 improves old-fragment point-estimate robustness | all 19 old-fragment strata have non-negative point-estimate deltas | SUPPORTED |

## Benchmark Claims

| Claim | Evidence | Status |
|-------|----------|--------|
| Transform-heldout split prevents transform-identity memorization | verified zero train/blind transform overlap | SUPPORTED |
| Secondary blind protocol is stricter than random split | transform identities held out; shared closed training vocabulary disclosed | SUPPORTED |
| ScoreBlend is stronger than Borda on blind | 0.8558 vs 0.8384; delta = +0.0174 | SUPPORTED |

## Negative Results Preserved for Paper

| Claim | Evidence | Status |
|-------|----------|--------|
| Query-level prior gate cannot separate rescue queries | D4S27 delta = 0.000000 | SUPPORTED |
| USR shape features provide no net benefit | USR contribution = -0.011 vs safe-only | SUPPORTED |
| Fragment-level routing is unreliable under transform-heldout evaluation | val/blind routing mismatch and negative transfer | SUPPORTED |
| Cluster-level routing unreliable | val-blind delta correlation r = -0.7371 | SUPPORTED |
| Learned query router too weak | AUC approximately 0.61; blind deployment rejected | SUPPORTED |

## A4C / Review Proxy Claims

| Claim | Evidence | Status |
|-------|----------|--------|
| G2 exploratory candidates have high covered alert rate (46.85%) | D4A3T risk decomposition; coverage=100% | SUPPORTED |
| G3 DE-retained candidates have moderate covered alert rate (9.67%) | D4A3T risk decomposition; coverage=100% | SUPPORTED |
| G4 shared candidates have total lower-bound alert rate = 0.99% | D4A3T risk decomposition; coverage=5.63%, unknown=94.37%, covered alert=17.60% | SUPPORTED_WITH_CAVEAT |
| A4C is computational proxy, not truth standard | rule-based filters; no expert or wet-lab validation | SUPPORTED |
| A4C is not tied to D4S31 primary Top10 evaluation | workflow strata use Borda/HGB/DE provenance groups | SUPPORTED_WITH_CAVEAT |

## Forbidden Claims

- Activity-preserving bioisostere prediction: NOT CLAIMED.
- Wet-lab validated replacement: NOT CLAIMED.
- Review-safe production system: NOT CLAIMED.
- MLP Top10 SOTA: NOT CLAIMED; MLP is secondary/MRR diagnostic evidence.
- Universal improvement: NOT CLAIMED; D4S31 old-fragment point estimates are non-negative, without per-fragment significance claim.

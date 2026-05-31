# Current Paper Problem

**Date**: 2026-05-29
**Source**: Full 5-day project review

## 1. Current Paper Main Character

**The paper currently has no clear main character.**

- Current manuscript (S3TEXT/S4GEN): Borda(DE,HGB) = 0.8384 Top10 is the primary result
- Actual best deployable: D4S31 drop_prior_ranks = 0.9243 Top10 (+0.0686 over baseline)
- The gap: manuscript's "main result" (0.8384) is outperformed by a method that's NOT in the paper

**Root cause**: D4S31 was discovered AFTER the S3TEXT manuscript was finalized. The paper was written in the "Borda era" (D4A2 era) and never updated with D4S series findings.

## 2. Borda's Role

Borda(DE,HGB) Top10 = 0.8384 is a fixed rank-fusion baseline, NOT the final deployable method.

Correct role:
- Borda demonstrates DE-HGB complementarity under leakage control
- Borda is parameter-free (important for transform-heldout constraint)
- Borda is NOT the best scorer — it's a fusion baseline that proves the concept

## 3. Score Blend's Role

Blend (0.95*MLP + 0.05*HGB) Top10 = 0.8558 is the CURRENT baseline.

Correct role:
- Blend is the strongest pre-D4S era method
- It serves as the baseline against which all D4S improvements are measured
- It should NOT be the paper's final claim

## 4. D4S31's Role

D4S31 drop_prior_ranks Top10 = 0.9243 (+0.0686, lost=6) is the BEST deployable scorer.

Correct role:
- PRIMARY result: candidate-level HistGB scorer with 77 features
- Key ablation insight: dropping 5 per-query rank features improves blind +0.039
- Rescue/lost profile (424/6) is the strongest safety argument

## 5. D4S27/D4S32 Routing Failure

D4S27 query-level gate: FAILED (delta=0.000)
D4S32 query router: FAILED (AUC=0.61, crashed on blind)

How to write:
- These are NEGATIVE RESULTS that motivate the candidate-level approach
- D4S27 proved the signal EXISTS but query-level routing can't exploit it
- D4S32 confirmed that learned routing is too hard with limited fragment diversity
- Together they justify WHY we moved to candidate-level scoring (D4S28→D4S31)

## 6. D4S30 USR No Value

How to write:
- USR shape descriptors added zero net benefit
- The finding is: 2D Morgan similarity already captures the relevant structural signal
- This is a negative ablation result that validates the feature engineering approach

## 7. External Ertl / Activity / A4C

**Should be re-run to D4S31 caliber**, but this is lower priority than fixing Methods.

- Ertl ring recall: needs re-evaluation with D4S31 predictions
- Retrospective activity endpoint: needs D4S31 scorer predictions as input
- A4C alert strata: already computed for D4S28R era, needs D4S31 update

## 8. Methods Must Be Rewritten

Current Methods (S3TEXT §3) is a 2-page skeleton written in the "Borda era". It describes:
- Task definition (OK)
- Benchmark construction (OK)
- Models: attach-freq, DE, HGB, Borda, MLP (Borda-era only)
- Evaluation protocol (minimal)
- A4C proxy (minimal)

It completely MISSES:
- Candidate matrix definition (what is each row?)
- How base rankers produce scores (DE architecture, HGB features)
- D4S31 candidate scorer pipeline (HistGB, 77 features, 5-fold CV, one-shot blind)
- Feature family documentation (9 families, 77 features, which 5 were dropped and WHY)
- prior_ranks removal rationale (the key design insight)
- Bootstrap CI methodology (5000 replicates)
- Rescue/lost definitions
- Leakage control details (cat_templates alignment, train-only statistics, transform-heldout verification)
- Negative subspaces handling
- Pipeline topology (how components connect)

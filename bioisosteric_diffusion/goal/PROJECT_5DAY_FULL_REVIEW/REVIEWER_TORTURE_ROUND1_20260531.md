# Reviewer Torture Round 1 - 2026-05-31

## Active Manuscript

- `goal/PROJECT_5DAY_FULL_REVIEW/paper_draft/combined_draft_clean.md`

## Team Setup

The review team was organized as three adversarial roles:

1. **Reviewer A: claim and novelty discipline**
   - Focus: title, abstract, main contribution, post-audit versus prospective claims.
2. **Reviewer B: methods and reproducibility**
   - Focus: leakage control, train-only feature construction, script-to-claim traceability, P3/P4 prior definitions.
3. **Reviewer C: statistics and medicinal-chemistry boundary**
   - Focus: ablation uncertainty, rescue/lost interpretation, A4C overclaiming, activity-preservation boundaries.

Native Codex sub-agent spawning was attempted after the manuscript edits, but the current thread had reached the collaboration agent limit. Ruflo swarm status was checked and was healthy/running. The review below records the completed Claude-CLI reviewer round and the coordinator revisions applied locally.

## Main Reviewer Attacks

### Critical

1. **Post-audit selection risk**
   - The 77-feature D4S31 result was obtained after blind-set diagnostics identified the prior_ranks problem.
   - The manuscript must not present D4S31 as a fully prospective blind feature-selection result.

2. **"Feature-regularized" was misleading**
   - No explicit regularization penalty is used.
   - The more accurate description is post-audit feature-pruned / shortcut-ablated candidate scoring.

3. **A4C overclaiming**
   - A4C is not expert validation, wet-lab validation, safety validation, or activity validation.
   - G4 has only 5.63% A4C coverage; the 0.99% value is a total-candidate lower bound, not a calibrated low-risk rate.

4. **Oracle terminology**
   - "Oracle(DE,HGB)" can be misread as a true upper bound.
   - It is only a Best-of-DE+HGB diagnostic for per-query selection between two base rankers.

5. **Ablation uncertainty**
   - The +0.0392 prior_ranks removal effect lacks its own paired uncertainty interval.
   - It must be described as a descriptive post-audit ablation unless a separate bootstrap is added.

## Applied Revisions

1. Changed title from:
   - `Feature-Regularized Candidate Scoring...`
   to:
   - `Post-Audit Feature-Pruned Candidate Scoring...`

2. Rewrote the abstract to state:
   - D4S31 is a locked post-selection finding.
   - The 77-feature result is not a fully prospective feature-selection result.
   - A4C strata are unvalidated computational triage signals.

3. Added an explicit method caveat:
   - HistGB classifier architecture itself is not claimed as novel.
   - The contribution is the leakage-controlled ranking setup, audited feature design, and protocol separation.

4. Renamed `Oracle(DE,HGB)` to:
   - `Best-of-DE+HGB diagnostic`
   and clarified that it is not a ceiling for candidate-level models using additional features.

5. Reframed the prior_ranks ablation:
   - The +0.0392 effect is descriptive because it was motivated by post-audit blind diagnostics.
   - Rescue/lost counts are descriptive query-level audit counts, not bootstrapped uncertainty estimates.

6. Removed A4C action-language:
   - Replaced "actionable risk structure" and recommendation language with "coverage-limited computational triage structure" and "illustrative provenance strata."

7. Strengthened limitations:
   - Transform-heldout controls transform-identity leakage, not every possible form of chemical relatedness.
   - Fully prospective confirmation would require a fresh blind split or independent replication.

8. Added reproducibility trace:
   - `goal/A_improve/D4S31_feature_pruning/d4s31_lockdown.py`
   - `goal/A_improve/D4S31_feature_pruning/d4s31_rescue_lost_repair.py`
   - `core/scripts/routeA_paper_p0_newblind_metric_ablation_lock.py`

## Remaining Reviewer Risks

1. **P3/P4 prior definitions remain compressed**
   - The paper summarizes these priors but does not fully formalize similarity thresholds, clustering, and smoothing details in the main text.
   - Recommendation: add a Supplementary Methods table or appendix with implementation-level formulas.

2. **D4S31 paper-main eligibility is still delicate**
   - The manuscript is transparent, but a strict reviewer may still argue that D4S31 should be treated as post-hoc.
   - Defensive stance: primary claim should remain "post-audit locked result requiring prospective replication."

3. **Ablation multiple-comparison control is not provided**
   - Current fix is claim-downgrading, not new statistics.
   - Recommendation: if time allows, add paired bootstrap CIs for key ablations, especially prior_ranks removal.

4. **A4C is still conceptually fragile**
   - Even with caveats, readers may overinterpret provenance strata.
   - Defensive stance: keep A4C out of the abstract's contribution framing and never call it validation.

5. **Independent external comparison remains weak**
   - The paper distinguishes itself from NeBULA, DeepBioisostere, and GraphBioisostere by task definition, not by direct benchmark superiority.
   - Avoid SOTA claims across the broader bioisostere discovery literature.

## Current Verdict

The manuscript is materially safer after this round. The strongest version is now:

> D4S31 is a high-performing post-audit feature-pruned scorer on a transform-heldout closed-vocabulary ranking benchmark, with transparent protocol caveats and computational-only triage support.

It should not be framed as:

> a fully prospective blind-selected SOTA bioisostere discovery method, activity-preserving predictor, or medicinal-chemistry validated workflow.


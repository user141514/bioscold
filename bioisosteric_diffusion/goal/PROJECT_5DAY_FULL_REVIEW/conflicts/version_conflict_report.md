# Version Conflict Report

**Date**: 2026-05-29

## Conflict 1: D4S28 Blind — Bug vs Generalization Failure

| Version | Claim | Source | Resolution |
|---------|-------|--------|------------|
| D4S28 README (original) | Blind 0.7120 = generalization failure due to 8 vs 19 fragments | D4S28_candidate_transform_scorer/README.md | INVALID |
| D4S30 audit | 0.7120 = cat_templates alignment bug | D4S30_shape_augmented/D4S30_AUDIT_VERDICT.md | CORRECT |
| D4S28R final | Blind 0.8851 after fixing alignment | D4S28R_FINAL_LOCK/02_scorer_all_metrics.md | FINAL |

**Resolution**: The 0.7120 was a one-hot encoding column mismatch. val has 8 old_fragment one-hot columns, blind has 19. Without cat_templates alignment, HistGB weights were applied to wrong feature indices. Fixed by extracting categories from val and mapping blind to the same column order.

## Conflict 2: D4S30 USR — Breakthrough vs Zero Value

| Version | Claim | Source | Resolution |
|---------|-------|--------|------------|
| D4S30 initial | USR shape features improve blind, breakthrough | (implicit in experiment design) | INVALID |
| D4S30 audit | Safe features alone = 0.906. Safe+USR = 0.895. USR contribution = -0.011 | D4S30_AUDIT_VERDICT.md | FINAL |

**Resolution**: The initial apparent improvement came from fixing the cat_templates alignment bug, NOT from USR features. Safe features alone achieve 0.906. Adding USR reduces to 0.895. USR provides zero net benefit.

## Conflict 3: Routing — Possible vs Impossible

| Version | Claim | Source | Resolution |
|---------|-------|--------|------------|
| Hypothesis | Fragment-level routing can improve scorer safety by avoiding negative subspaces | D4S28R experiment design | INVALID |
| Route audit | Fragment-level routing IMPOSSIBLE (0 val/blind overlap). Cluster routing: r=-0.7371 (NEGATIVE). Attachment routing: worse than scorer_all | D4S28R_FINAL_LOCK/03_route_rejection.md | FINAL |

## Conflict 4: Feature Set Size — 82 vs 77

| Version | Claim | Source | Resolution |
|---------|-------|--------|------------|
| D4S28R final | 82 safe features = best | D4S28R_FINAL_LOCK/07_FINAL_VERDICT.md | SUPERSEDED |
| D4S31 ablation | 77 features (drop prior_ranks) = better | D4S31_FINAL_LOCK/01_ablation_findings.md | FINAL |

**Resolution**: D4S31 is strictly better: +0.039 blind improvement, 596→6 lost hits, all negative subspaces resolved.

## Conflict 5: Paper Main Method — Borda vs Score Blend vs D4S31

| Version | Main Method | Top10 | Source |
|---------|------------|-------|--------|
| S3TEXT | Borda(DE,HGB) | 0.8384 | routeA_paper_s3text_journal_facing |
| S4GEN | Borda(DE,HGB) | 0.8384 | routeA_paper_s4gen_general_journal_polish |
| S6R | Score blend (MLP+HGB) | 0.8558 | routeA_paper_s6r_merge_claim_denoised_upgrade |
| THIS REVIEW | D4S31 drop_prior_ranks | 0.9243 | goal/A_improve/D4S31_FINAL_LOCK |

**Resolution**: Paper needs complete update. Borda is Chapter 1 (fusion concept proof). Score blend is the pre-D4S31 baseline. D4S31 is the actual best method and should be the paper's primary result.

## Conflict 6: D4S27 — Signal Exists vs Can Be Used

| Version | Claim | Source | Resolution |
|---------|-------|--------|------------|
| D4S27 analysis | Priors rescue 689 queries, oracle +0.052 | A_improve/report.md | CORRECT — signal EXISTS |
| D4S27 conclusion | No deployable gain from priors | A_improve/report.md | CORRECT — signal cannot be USED at query level |
| D4S28 hypothesis | Candidate-level model can use this signal | D4S28/README.md | CORRECT — D4S31 proves it (+0.0686) |

**Resolution**: All versions are CORRECT but scoped differently. D4S27 correctly found the signal exists but query-level gate can't use it. D4S28/D4S31 correctly showed candidate-level model CAN use it.

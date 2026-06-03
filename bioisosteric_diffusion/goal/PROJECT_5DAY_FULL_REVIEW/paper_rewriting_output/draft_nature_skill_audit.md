# Nature-Skill Draft Audit

**Audit target:** `revised_main_manuscript.md`  
**Date:** 2026-06-02  
**Mode:** Direct draft audit only. Final LaTeX conversion issues are excluded except where they affect submission readiness.  
**Skill basis:** Yuan1z0825/nature-skills `nature-writing` paper-review checklist plus local Nature-style polishing guardrails.

## Overall verdict

The draft has a strong and defensible manuscript spine: leakage-controlled benchmark, candidate-level scorer, and post-audit prior_ranks shortcut deletion. The strongest feature is not wording but claim discipline: the draft repeatedly states that the 77-feature model is post-audit/post-selection, that labels are structure-derived, and that A4C is unvalidated triage. This is exactly the right posture for JCIM reviewers.

The main remaining risk is not overclaiming. It is evidence packaging: several claims are believable but under-supported in the main text unless the supplementary tables and routing evidence are present and easy to inspect. The manuscript can move from "good internal draft" to "submission-ready" by tightening three reviewer-facing weak points.

## Priority findings

### P1. Training/evaluation split wording may confuse reviewers

**Location:** lines 43, 138-139, 319, 325-327.

The draft says the candidate vocabulary is constructed from the training split, the development/calibration set is drawn from the training partition, and the final secondary-blind scorer is trained on the "full development set." This may be correct in your internal protocol, but the terminology is easy to misread as: "the model was trained only on the development set," or "development set statistics define both training and frozen schema." A reviewer may ask what happened to the 109,985-query train matrix in Table M1.

**Fix:** add a short paragraph or schematic sentence that explicitly names which split is used for each purpose:

- train partition: prior/count/statistic construction and candidate vocabulary;
- development/calibration: model selection, schema template, hyperparameters, and final scorer fitting if that is true;
- secondary blind: one-shot evaluation only.

If the scorer is truly trained only on the 13,375-query development/calibration set, say why. If it is trained on a combined train/development matrix, revise line 327.

### P1. Query-level routing claim needs a small evidence table or demotion

**Location:** lines 181, 612-614, 646.

The draft elevates "candidate-level > query-level routing" into a complementary finding, but the visible evidence is only one sentence with "AUC about 0.61" and a qualitative statement that neither routing approach improved. Nature-style review logic requires this to be either evidenced or demoted.

**Fix:** add a compact table in Supplementary or Results with attempted routing strategies, selection target, AUC or validation metric, blind Top-10, and failure mode. If you do not want another table, soften line 646 from a "complementary finding" to "development observation."

### P1. Per-fragment robustness claim is important but too dependent on Supplementary Table S5

**Location:** lines 529, 572-574.

The "82-feature degraded five old-fragment strata; 77-feature has no negative point-estimate delta across 19 strata" claim is central to Results 4.1/4.3 and to the prior_ranks mechanism. Right now it is text-only in the main draft and relies on Supplementary Table S5.

**Fix:** add Figure S3 or a small supplementary lollipop plot. This is exactly the figure you described: per old_fragment, show 82-feature delta, 77-feature delta, and N. The caption should say "point estimates only; no per-fragment significance test." This will make the stratum-level claim reviewable.

### P2. Related-work comparisons need one guard sentence against hidden direct-comparison expectations

**Location:** lines 29-31, 632-634.

The draft correctly says NeBULA, DeepBioisostere, GraphBioisostere, CReM, and SwissBioisostere solve related but different tasks. However, because Table 1 contains only internal baselines, reviewers may still ask why there is no direct external-method comparison.

**Fix:** add one explicit sentence in Related Work or Limitations: direct comparison would require adapting open-vocabulary generators/database retrieval systems to the closed 150-candidate ranking matrix, so the current paper compares against internal rankers under a controlled benchmark and leaves external-method adaptation as a separate benchmark extension.

### P2. The abstract is defensible but dense

**Location:** lines 5-7.

The abstract is accurate and well-scoped, but it carries benchmark, model, two deltas, post-selection caveat, A4C, and activity-label caveat in two paragraphs. That is scientifically safe, but the reader has to work.

**Fix:** keep the numbers, but consider splitting the second paragraph into: (1) prior_ranks deletion and post-audit caveat; (2) A4C and label boundary. This is a readability improvement, not a scientific correction.

### P2. Data/software availability still promises future deposition

**Location:** line 656.

The draft says large processed matrices "will be deposited or documented separately before publication" and DOI "will be provided upon acceptance." That may be acceptable before final submission only if ACS policy allows placeholder language, but it is a reviewer/reproducibility vulnerability.

**Fix:** before submission, replace future-tense availability with concrete artifact locations, or state exactly which artifacts are currently public and which are restricted by ChEMBL redistribution terms.

## Strengths to preserve

- The post-audit caveat is consistently present in Abstract, Introduction, Methods, Results, Discussion, and Conclusion.
- The task boundary is clear: structure-derived labels, not activity preservation.
- Table 3 has the right explanatory caption: it prevents confusion between candidate-matrix score columns and standalone Table 1 base-ranker rows.
- The A4C section is appropriately humble: unvalidated computational screening signal, not medicinal-chemistry validation.
- The Borda discussion is useful because it explains complementarity rather than just reporting another baseline.

## Claim-evidence map

| Claim | Evidence in draft | Status |
|---|---|---|
| Transform-heldout benchmark prevents transform-identity leakage | Table M2, lines 162-173 | Supported |
| 77-feature scorer improves over Score Blend | Table 1, paired CI in lines 503-509 | Supported, post-audit caveat preserved |
| prior_ranks deletion is dominant shortcut-removal finding | Table 2, lines 531-552 | Supported, but mechanism remains interpretive |
| 77-feature scorer removes negative old-fragment point estimates | Lines 529, 574 plus Supplementary Table S5 | Supported if S5/Figure S3 is present; needs visual packaging |
| Candidate-level scoring succeeds where query-level routing fails | Lines 181, 612-614 | Under-evidenced in main draft |
| A4C provides provenance-stratified triage | Table 4, lines 584-600 | Supported as workflow signal only |

## Recommended next actions

1. Add Figure S3 for per-fragment blind deltas.
2. Add a tiny routing-failure table or demote the routing claim.
3. Clarify the train/development/final-fit protocol in Section 3.4.3.
4. Ensure Supplementary Tables S3-S5 are packaged with the submission.
5. Before final LaTeX conversion, migrate the Markdown references into the final bibliography and preserve Table 3 caption detail.


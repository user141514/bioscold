# Strict Nature-Skill Audit After Revision

**Audit target:** `manuscript_en.md` and `manuscript_zh.md`  
**Date:** 2026-06-02  
**Audit stance:** skeptical reviewer, evidence first, no invented claims.

## Verdict

The revised natureskill version is substantially stronger than the original draft. The story is now explicit:

1. **Prospective thread:** transform-heldout benchmark plus the initial 82-feature candidate-level scorer.
2. **Post-audit mechanism thread:** prior_ranks deletion produces the locked 77-feature model and explains the main safety improvement.
3. **Workflow thread:** A4C provides provenance-stratified computational triage only.

This separation is the manuscript's main defense against reviewer attack. It prevents the post-audit 77-feature result from being misrepresented as a fully prospective model-selection result while still allowing the deletion result to be scientifically meaningful.

## What Was Fixed

### 1. Evidence hierarchy is now explicit

The abstract, introduction, methods preamble, results, discussion, and conclusion now distinguish the prospective 82-feature scorer from the post-audit 77-feature scorer. This avoids the most serious framing risk: presenting a blind-diagnostic feature deletion as if it were pre-registered.

### 2. Split roles are clearer

The revised Methods now states that:

- the training partition defines the candidate vocabulary and train-derived priors/statistics;
- the development/calibration matrix is used for scorer configuration, schema construction, and final fitting;
- the secondary blind set is used for one-shot evaluation and post-audit diagnosis.

This directly addresses the reviewer confusion risk around Table M1 versus "trained on the full development set."

### 3. External-method comparison is better bounded

The revised Related Work and Discussion now explain why NeBULA, CReM, DeepBioisostere, and GraphBioisostere are not directly compared in Table 1: they solve retrieval, generation, or pair-prediction tasks rather than returning scores over the same fixed 150-candidate ranking matrix.

### 4. Routing is demoted to the right evidentiary level

The query-level routing observation is no longer framed as a co-equal major finding. It is now a diagnostic development observation that supports a design interpretation.

### 5. Per-fragment claim is more cautious

The revised text states that the 19 old-fragment strata are point-estimate audits, not independent significance tests. It also flags that this result should be visualized as Supplementary Fig. S3.

## Remaining Submission Risks

### P1. Figure S3 is still needed

The manuscript still depends on the stratum-level statement that the 82-feature scorer degraded five old-fragment strata while the 77-feature scorer has no negative point-estimate delta across 19 strata. This should not remain text-only.

**Required fix:** create Supplementary Fig. S3 as a horizontal lollipop / paired-dot plot with 82-feature delta, 77-feature delta, and N for each old_fragment.

### P1. Supplementary Tables S3-S5 must ship with the draft

The main text relies on S3 for 82-feature rescue/lost, S4 for full ablation, and S5 for per-fragment strata. Missing supplementary material would make the core claims feel under-supported.

**Required fix:** include a clean supplementary package or an SI draft in `natureskill_paper`.

### P1. Data availability still has a final-submission gap

The revised availability statement is more honest, but it still says large matrices require separate deposition/access documentation. This is acceptable as a pre-submission draft state, not as the final submission state.

**Required fix:** before submission, provide concrete artifact locations, restricted-data notes, or deposition instructions.

### P2. Routing evidence remains thin

The routing claim is now correctly demoted. If the authors want to keep it prominent, they need a small routing-failure table.

**Optional fix:** add a supplementary table listing attempted routing methods, training signal, AUC/validation metric, blind Top-10, and failure mode.

### P2. Citation metadata should be rechecked before final LaTeX

The Markdown draft has references, but final LaTeX conversion previously dropped the bibliography. Also, 2025-2026 references and preprint status should be verified before submission.

**Required fix at typesetting stage:** migrate all 20 references into the final bibliography and verify DOI/preprint status.

## Reviewer Attack Simulation

| Reviewer question | Current defense | Residual risk |
|---|---|---|
| Was the 77-feature model selected after looking at blind data? | Yes, explicitly disclosed as post-audit/post-selection. | Low if caveat remains everywhere. |
| Why is the 82-feature result not the final prospective model? | It is now named as the prospective candidate-level result; 77-feature is post-audit locked. | Low. |
| Are labels activity-preserving? | No, structure-derived only; stated repeatedly. | Low. |
| Why no direct comparison to DeepBioisostere/NeBULA? | Different task formulation; fixed 150-candidate ranking matrix would require adaptation. | Medium but acceptable. |
| Does every old-fragment stratum improve significantly? | No significance claim; point estimates only. | Low if Figure S3 caption says this clearly. |
| Is A4C a safety validation? | No, computational triage only. | Low. |
| Can results be reproduced? | Code archive and commits cited. | Medium until data/deposition details are complete. |

## Recommendation

Proceed with this natureskill draft as the working manuscript. The next high-value work is not more polishing. It is evidence packaging:

1. draw Figure S3;
2. package Supplementary Tables S3-S5;
3. fix final LaTeX bibliography;
4. finalize data availability.


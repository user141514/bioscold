# Targeted Prior-Family Text Integration Audit

Date: 2026-06-02

## Scope

This audit records the manuscript text integration of the targeted prior-family ablation and support-bin diagnostic.

## Main-Text Integration

- Results 4.2 now retains a single compact paragraph for the targeted prior-family ablation and support-bin audit.
- The paragraph is 121 words, within the requested 120--160 word target.
- The paragraph reports only the core targeted-ablation and support-bin values:
  - Drop prior_ranks: +0.0393, 95% CI [0.0354, 0.0432]
  - Drop all non-rank prior families: +0.0174
  - Drop prior statistics: +0.0151
  - Drop prior scores: +0.0073
  - Drop PMI/conditional-prior contrasts: -0.0037, 95% CI [-0.0064, -0.0010]
  - Low/medium/high support-bin gains: +0.0488, +0.0344, +0.0346
- Old-fragment block details were removed from the Results narrative and left to the Supplementary tables plus a bounded Discussion statement.

## Discussion Integration

- Discussion 5.3 now uses the bounded interpretation:
  - query-relative prior-rank features behaved as non-transferable shortcuts under secondary blind fragment shift;
  - the largest targeted ablation gain came from removing prior ranks;
  - the gain was strongest in low train-derived support regimes;
  - fragment-level regressions were reduced after pruning.
- The text explicitly avoids claiming that all raw prior features are beneficial, that all rank features are harmful, or that the post-audit pruning was fully prospective.

## Cleanup Checks

- Removed unsupported "not significant" wording from the MLP/Borda Top-10 comparison.
- Replaced the main-text internal code `D4S-B0` with `locked standalone base rankers`.
- Replaced the Methods wording "feature set ... was finalized" with post-audit ablation language.
- Replaced the conclusion slogan about "most impactful feature-engineering decision" with bounded shortcut-audit language.
- Corrected the DeepBioisostere author list to `Kim, H.; Moon, S.; Zhung, W.; Lim, J.; Kim, W. Y.`

## Synchronization

- `jcim_submission_candidate/main_manuscript.md` was updated.
- `paper_draft/combined_draft_clean.md` was synchronized from the candidate manuscript.

## Remaining Figure Note

Supplementary Table S5 and the main manuscript use the updated old-fragment boundary:
the post-audit 77-feature scorer reduces negative old-fragment point estimates relative to Score Blend from five to two, and 17 of 19 old-fragment strata are non-negative relative to the initial 82-feature scorer.

The legacy Supplementary Figure S3 asset/caption still contains older `0/19` wording and should be regenerated or replaced before final submission if Supplementary Figure S3 remains in the package.

# Reviewer Attack Response

## 1. Is the 77-feature result post hoc?

Yes. The manuscript states that the prior_ranks removal was prompted by blind diagnostics. The 77-feature scorer is reported as a locked post-selection finding, while the initial 82-feature scorer is the prospective candidate-level blind result. The claim is not that feature selection was fully prospective; the claim is that the audit identifies a reproducible shortcut failure and motivates prospective replication.

## 2. Did the work reuse the blind set too many times?

The final draft separates evidentiary roles. The secondary blind set supports the locked reported performance and the post-audit diagnostic finding. The text explicitly states that a fresh blind split or independent replication would be required to elevate the 77-feature configuration to a fully prospective feature-selection result.

## 3. Does A4C validate medicinal chemistry decisions?

No. A4C is described as a computational review proxy and provenance/risk stratification layer. It does not replace medicinal chemistry expert review, experimental profiling, toxicity assessment, or synthetic-accessibility assessment.

## 4. Do structure-derived MMP labels prove activity-preserving bioisosteric replacement?

No. The labels record observed structural substitutions from ChEMBL-derived matched molecular pairs. The manuscript repeatedly states that these labels do not establish activity preservation.

## 5. Does the 77-feature scorer invalidate the Borda mechanism story?

No. Borda is no longer the final best method, but it remains the parameter-free mechanism anchor showing DE/HGB complementarity. The candidate-level scorer extends that complementarity by integrating base-ranker signals with molecular descriptors, training-derived priors, and frozen categorical features.

## 6. Is the public repository enough for reproducibility?

The repository provides analysis scripts, feature-schema documentation, query-level audit files, bootstrap summaries, and small evidence tables. Large processed matrices and redistribution-restricted ChEMBL-derived artifacts are not bundled directly and must be deposited or documented separately before publication, subject to ChEMBL-derived data terms.

## 7. Are the supplementary ablations overclaimed?

No. Supplementary Table S4 states that only the prior_ranks deletion has a paired confidence interval in the current evidence lock. Other leave-one-family-out rows are point estimates.

## 8. Are the per-fragment results cherry-picked?

The main text uses the per-fragment analysis only as a reliability profile and explicitly avoids separate per-fragment significance claims. Supplementary Table S5 reports all 19 old-fragment strata.

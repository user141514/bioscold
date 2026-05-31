# JCIM Review Integration Plan - 2026-05-31

## Final Target-Journal Verdict

**JCIM remains the best primary target**, but the manuscript needs a major JCIM-facing revision before submission.

The current science is suitable for JCIM if framed as:

> leakage-controlled closed-vocabulary MMP-derived fragment replacement ranking with candidate-level scoring and post-audit shortcut analysis.

It is not ready to be framed as:

> validated bioisosteric discovery, activity-preserving prediction, or prospective medicinal chemistry workflow.

## Top 10 Paper-Blocking Edits

1. **Resolve D4S31 evidence hierarchy.**
   - State clearly that D4S28R is the fully prospective blind model and D4S31 is post-audit locked, unless a fresh blind split is created.

2. **Add CI for prior_ranks ablation.**
   - Paired query bootstrap for D4S28R vs D4S31.

3. **Add dataset summary table.**
   - MMP count, split sizes, candidate vocabulary size, candidates/query, positives/query, old fragments/split, attachment signatures/split.

4. **Define P0/P1/P3/P4 formally.**
   - Include smoothing, clustering, similarity, and transfer formulas.

5. **Define attachment compatibility.**
   - Formalize attachment signature and valence/bond-order compatibility predicate.

6. **Fix A4C circularity.**
   - PAINS/Brenk pre-filter means A4C alert rates are conditional on pre-cleaned data.

7. **Add Data and Software Availability.**
   - Repository, commit hash, environment file, data source, processed artifacts, license.

8. **Add figures and chemical examples.**
   - Split/pipeline schematic, Top-K curves, chemical examples, ablation/importance plot.

9. **Reframe as MMP-derived ranking, not bioisostere validation.**
   - Use "structure-derived replacement ranking" as the core phrase.

10. **Rename internal model IDs in prose.**
   - Keep D4S31/D4S28R in parentheses or supplement; use descriptive names in main text.

## Safe Title Candidates

1. **Leakage-Controlled Closed-Vocabulary Fragment Replacement Ranking with Candidate-Level Scoring**
2. **A Transform-Heldout Benchmark for MMP-Derived Fragment Replacement Ranking**
3. **Candidate-Level Scoring for Leakage-Controlled Scaffold-Conditioned Fragment Replacement**
4. **Shortcut-Aware Candidate Scoring for Closed-Vocabulary Fragment Replacement Ranking**

Recommended:

> Leakage-Controlled Closed-Vocabulary Fragment Replacement Ranking with Candidate-Level Scoring

## Revised Abstract Strategy

Use five sentences:

1. Problem: fragment replacement ranking is useful but random splits leak transform identity.
2. Method: transform-heldout benchmark and candidate-level scorer.
3. Prospective result: initial 82-feature scorer vs Score Blend.
4. Post-audit result: prior_ranks removal gives D4S31 0.9243, explicitly post-selection.
5. Scope: structure-derived labels, no activity preservation, A4C only computational triage.

## Required Supplementary Tables

1. Dataset statistics by split.
2. P0/P1/P3/P4 prior definitions.
3. Full Top-K/MRR metrics with CIs.
4. Full ablation grid with paired CIs where possible.
5. Rescue/lost bootstrap summary.
6. Per-fragment strata table with uncertainty or explicit point-estimate-only label.
7. A4C coverage and alert table with pre-filter caveat.

## Code/Data Availability Requirements

Add a section before References:

- ChEMBL source/version.
- Processed data artifact location or DOI.
- Code repository path/URL.
- Commit hash or archive ID.
- Conda environment or `requirements.txt`.
- Main scripts:
  - `goal/A_improve/D4S31_feature_pruning/d4s31_lockdown.py`
  - `goal/A_improve/D4S31_feature_pruning/d4s31_rescue_lost_repair.py`
  - `core/scripts/routeA_paper_p0_newblind_metric_ablation_lock.py`
- License.
- Reproducibility command examples.

## Claims to Remove or Downgrade

- Remove any implication of activity preservation.
- Remove any implication that A4C validates medicinal chemistry decisions.
- Downgrade D4S31 from fully prospective blind method to post-audit locked method.
- Downgrade prior_ranks ablation until CI is added.
- Downgrade G4 alert estimate because 94.37% status is unknown.
- Avoid superiority claims over NeBULA, DeepBioisostere, GraphBioisostere, or open-vocabulary methods.

## Experiments or Reruns That Are Genuinely Necessary

1. Paired bootstrap CI for D4S28R vs D4S31.
2. Full secondary metrics: Top-1/5/10/20/50 and MRR with CIs.
3. Dataset summary counts from existing artifacts.
4. A4C pre-filter caveat support: confirm whether PAINS/Brenk filtering was applied before A4C.
5. Optional but valuable: random-drop-5 feature negative control.

## Writing-Only Edits

1. Title revision.
2. Abstract rewrite.
3. Contribution hierarchy rewrite.
4. Related work repositioning.
5. Remove internal-code style model names.
6. Move script paths to Data/Software Availability.
7. Consolidate repeated prior_ranks narrative.

## Go/No-Go Recommendation

**No-go for immediate JCIM submission.**

**Go after major revision** if the paper adds:

- ablation CI;
- dataset statistics;
- reproducible prior definitions;
- Data/Software Availability;
- figures and chemical examples;
- stronger prospective/post-audit hierarchy.


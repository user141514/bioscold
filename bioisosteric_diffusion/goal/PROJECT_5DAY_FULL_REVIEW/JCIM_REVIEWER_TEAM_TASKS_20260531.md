# JCIM Reviewer Team Tasks - 2026-05-31

## Target Journal

Primary target: **Journal of Chemical Information and Modeling (JCIM)**

Active manuscript:

- `goal/PROJECT_5DAY_FULL_REVIEW/paper_draft/combined_draft_clean.md`

Decision file:

- `goal/PROJECT_5DAY_FULL_REVIEW/TARGET_JOURNAL_DECISION_20260531.md`

## Global Rules for All Reviewers

Do not suggest new claims unless they are already supported by data.

Do not allow:

- activity-preserving bioisostere discovery;
- wet-lab validation;
- expert validation;
- review-safe production claims;
- superiority over open-vocabulary generative methods;
- fully prospective feature-selection claims for D4S31;
- A4C as medicinal-chemistry validation.

All reviewers should separate:

1. **paper-blocking issues**;
2. **major revision issues**;
3. **minor polish issues**;
4. **recommended manuscript edits**.

## Reviewer 1: JCIM Methods Reviewer

### Mission

Attack the manuscript as a JCIM methods reviewer.

### Focus

- Is the ranking task defined precisely enough?
- Are candidate sets, positives, negatives, and multi-positive queries clear?
- Are train/development/blind splits described well enough?
- Are all features train-only where required?
- Are P0/P1/P3/P4 prior definitions reproducible?
- Are categorical schema alignment and query grouping sufficiently clear?
- Are script paths sufficient for reproducibility?
- Is the post-audit timeline honest enough?

### Output

Create:

- `goal/PROJECT_5DAY_FULL_REVIEW/jcim_reviews/reviewer1_methods_report.md`

## Reviewer 2: JCIM Cheminformatics Reviewer

### Mission

Attack chemical plausibility and cheminformatics framing.

### Focus

- MMP extraction and fragmentation rules.
- Attachment signature definition.
- Closed-vocabulary replacement realism.
- Fragment vocabulary and candidate compatibility.
- Decoy construction and non-cognate scaffold assumptions.
- Bioisostere terminology boundaries.
- Whether ChEMBL-derived structure labels are enough for a JCIM method paper.
- Whether A4C alert filters are described correctly and not overinterpreted.

### Output

Create:

- `goal/PROJECT_5DAY_FULL_REVIEW/jcim_reviews/reviewer2_cheminformatics_report.md`

## Reviewer 3: JCIM ML and Statistics Reviewer

### Mission

Attack model selection, uncertainty, baselines, and statistical claims.

### Focus

- D4S31 post-audit selection and whether wording is safe.
- Bootstrap by query_id.
- Whether ablation +0.0392 needs CI or downgrade.
- Rescue/lost counts as descriptive audit only.
- Multiple comparisons in leave-one-family-out ablation.
- Baseline fairness: Attachment, HGB, DE, Borda, MLP, Score Blend, Best-of-DE+HGB.
- Whether "strongest result" wording overclaims.
- Whether MRR and secondary metrics are underreported.

### Output

Create:

- `goal/PROJECT_5DAY_FULL_REVIEW/jcim_reviews/reviewer3_ml_statistics_report.md`

## Reviewer 4: JCIM Editorial Fit Reviewer

### Mission

Judge whether the manuscript is framed for JCIM rather than Nature Communications, Bioinformatics, or Journal of Cheminformatics.

### Focus

- Title fit.
- Abstract density.
- Contribution hierarchy.
- Related work positioning versus NeBULA, DeepBioisostere, GraphBioisostere, SwissBioisostere, and CReM.
- Whether the paper should be sold as algorithmic chemical information modeling, not computational biology endpoint prediction.
- Whether figures/tables should be reorganized for JCIM.
- Whether a Data and Code Availability section is missing.

### Output

Create:

- `goal/PROJECT_5DAY_FULL_REVIEW/jcim_reviews/reviewer4_editorial_fit_report.md`

## Coordinator Integration Task

After all four reports are generated, create:

- `goal/PROJECT_5DAY_FULL_REVIEW/JCIM_REVIEW_INTEGRATION_PLAN_20260531.md`

The integration plan must include:

1. final target-journal verdict;
2. top 10 paper-blocking edits;
3. safe title candidates;
4. revised abstract strategy;
5. required supplementary tables;
6. code/data availability requirements;
7. claims to remove or downgrade;
8. experiments or reruns that are genuinely necessary;
9. edits that are writing-only;
10. go/no-go recommendation for JCIM submission.


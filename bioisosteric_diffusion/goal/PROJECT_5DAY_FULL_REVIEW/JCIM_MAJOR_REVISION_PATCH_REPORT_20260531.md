# JCIM Major Revision Patch Report

Date: 2026-05-31

## Scope

This patch targets the first JCIM-facing revision pass for `paper_draft/combined_draft_clean.md`.

Primary actions:

- Added dataset statistics for train, development/calibration, and secondary blind splits.
- Recomputed query-level bootstrap confidence intervals for the main secondary blind metrics.
- Added paired bootstrap confidence intervals for the post-audit 77-feature scorer versus the initial 82-feature scorer and Score Blend baseline.
- Added rescue/lost bootstrap uncertainty.
- Rewrote the title and abstract around leakage-controlled closed-vocabulary fragment replacement ranking.
- Added a journal-facing Data and Software Availability section.
- Removed remaining internal model identifiers from manuscript prose.
- Tightened A4C wording to prevent safety, expert-validation, or medicinal-chemistry decision claims.

## Evidence Files

Evidence directory:

`E:\zuhui\bioisosteric_diffusion\goal\PROJECT_5DAY_FULL_REVIEW\jcim_major_revision\`

Generated files:

- `jcim_dataset_statistics.csv`
- `jcim_full_secondary_metrics_with_ci.csv`
- `jcim_paired_bootstrap_key_deltas.csv`
- `jcim_rescue_lost_bootstrap.csv`
- `jcim_query_level_82_77_scoreblend_audit.csv`
- `jcim_major_revision_evidence_manifest.json`

Rerun script:

`E:\zuhui\bioisosteric_diffusion\goal\PROJECT_5DAY_FULL_REVIEW\scripts\jcim_major_revision_patch.py`

The script reuses locked score columns and fixed model definitions. It does not tune hyperparameters, select features, or change split labels.

## Dataset Statistics Added

| Split | Candidate rows | Queries | Positive rows | Unique candidates | Unique old fragments | Unique attachment signatures | Unique transforms | Single-positive queries | Multi-positive queries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 16,497,750 | 109,985 | 181,483 | 150 | 109 | 5 | 351 | 73,870 | 36,115 |
| Development/calibration | 2,006,250 | 13,375 | 21,408 | 150 | 8 | 5 | 27 | 8,686 | 4,689 |
| Secondary blind | 2,002,050 | 13,347 | 20,082 | 150 | 19 | 5 | 63 | 9,377 | 3,970 |

All splits use 150 candidates per query. Median positive set size is 1 in all three splits.

## Main Metric Evidence

Secondary blind Top-10:

- Attachment-Frequency: 0.6019
- DE: 0.8055
- HGB: 0.8624
- Borda(DE,HGB): 0.8456
- Score Blend: 0.8558
- Initial 82-feature scorer: 0.8851, 95% CI [0.8796, 0.8903]
- Post-audit 77-feature scorer: 0.9243, 95% CI [0.9199, 0.9287]

Important caveat:

The initial 82-feature scorer is the prospective candidate-level blind result. The 77-feature scorer is a locked post-audit result because prior-rank feature removal was prompted by blind diagnostics.

## Paired Bootstrap Evidence

Post-audit 77-feature scorer versus initial 82-feature scorer:

- Top-10 delta = +0.0393
- 95% CI = [+0.0354, +0.0432]

Post-audit 77-feature scorer versus Score Blend:

- Top-10 delta = +0.0686
- 95% CI = [+0.0638, +0.0733]

Bootstrap unit:

- query-level rows
- not candidate rows

## Rescue/Lost Evidence

77-feature scorer versus Score Blend:

- Rescue = 1,016
- Lost = 101
- Net = +915
- Net per query = +0.0686
- 95% CI = [+0.0638, +0.0733]
- Rescue/lost ratio = 10.06
- Ratio CI = [8.25, 12.52]

77-feature scorer versus initial 82-feature scorer:

- Rescue = 626
- Lost = 102
- Net = +524
- Net per query = +0.0393
- 95% CI = [+0.0354, +0.0432]
- Rescue/lost ratio = 6.14
- Ratio CI = [5.03, 7.65]

## Manuscript Edits

Updated file:

`E:\zuhui\bioisosteric_diffusion\goal\PROJECT_5DAY_FULL_REVIEW\paper_draft\combined_draft_clean.md`

Major text changes:

- Title changed to `Leakage-Controlled Closed-Vocabulary Fragment Replacement Ranking with Candidate-Level Scoring`.
- Abstract rewritten to center the MMP-derived closed-vocabulary ranking task.
- Contributions reframed around benchmark, candidate-level scoring, evidence separation, and computational triage caveats.
- Dataset summary table added as Table M1.
- Leakage table renamed to Table M2.
- Feature-family table renumbered to Table M3.
- Evaluation protocol table renumbered to Table M4.
- Provenance-group table renumbered to Table M5.
- Results 4.1 rewritten to separate prospective 82-feature evidence from post-audit 77-feature evidence.
- Table 1 updated with 82-feature and 77-feature scorer rows.
- Table 2 updated with CI for the prior-rank ablation.
- Table 3 expanded with rescue/lost comparison against both Score Blend and the 82-feature scorer.
- A4C text revised to emphasize coverage-limited computational triage on a PAINS/Brenk pre-filtered vocabulary.
- Data and Software Availability section added before References.
- Data and Software Availability revised to use definite deposition language rather than placeholder `should be deposited` wording.

## Claim Safety Status

Passed checks:

- No activity-preserving prediction claim.
- No wet-lab validation claim.
- No expert validation claim.
- No review-safe or production-safe claim.
- No claim that A4C is a validated medicinal-chemistry decision rule.
- No claim that the 77-feature scorer is fully prospective.
- No old `+0.0392` ablation delta remains.
- No old CI strings `[0.0637, 0.0735]`, `[0.9198, 0.9289]`, or `0.99%` remain.
- Internal model identifiers were removed from manuscript prose.

Remaining caveats to preserve:

- 77-feature scorer is post-audit/locked, not fully prospective model selection.
- The benchmark uses structure-derived labels from ChEMBL MMPs.
- Activity preservation is not established.
- A4C is an unvalidated computational proxy.
- G4 A4C coverage is sparse and cannot support a group-wide alert estimate.

## Remaining Reviewer-Risk Items

- Add journal-quality figures and chemical examples.
- Replace manuscript-package availability text with a real repository URL or DOI before submission.
- Decide whether to keep the 77-feature scorer in the main claim or foreground the 82-feature scorer as the only prospective candidate-level result.
- Consider adding a concise schematic of the split protocol and evidence hierarchy.
- Verify citation formatting against JCIM requirements.

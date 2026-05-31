# Algorithm Code Git Archive

Date: 2026-05-31

Purpose:

Archive the code and small evidence artifacts that support the current JCIM-facing manuscript draft for Route-A closed-vocabulary fragment replacement ranking.

## Included Scope

This archive is intended to capture the algorithmic trail behind the current manuscript:

- D4S27 conditional-prior/query routing baseline.
- D4S28 candidate-level transform scorer and audit scripts.
- D4S28R initial 82-feature scorer lock and small audit artifacts.
- D4S30 shape-augmented negative result.
- D4S31 post-audit feature-pruned 77-feature scorer scripts and lock artifacts.
- D4S32 query-router negative result.
- Best-of-DE+HGB diagnostic recomputation script.
- JCIM major-revision evidence script and small CSV/JSON evidence tables.
- Current combined manuscript draft and JCIM review/patch reports.

## Excluded Scope

The archive intentionally excludes large or environment-specific artifacts:

- model weights (`*.pth`, `*.pt`, `*.onnx`)
- pickled datasets and knowledge bases (`*.pkl`, `*.joblib`)
- database dumps (`*.db`, `*.sqlite`)
- parquet/matrix/data shards
- compressed external datasets
- cache folders, logs, and local agent state

These are excluded to keep the git archive reviewable and to avoid accidentally publishing bulky or redistributable third-party-derived data. The manuscript's Data and Software Availability section separately states that processed candidate matrices, split labels, feature schemas, query-level predictions, bootstrap scripts, and audit tables will be deposited in a public repository before publication, with a frozen commit hash and archival DOI provided upon acceptance, subject to the redistribution terms of ChEMBL-derived data.

## Current Number-Lock Files

The JCIM revision evidence is stored under:

`goal/PROJECT_5DAY_FULL_REVIEW/jcim_major_revision/`

Key files:

- `jcim_dataset_statistics.csv`
- `jcim_full_secondary_metrics_with_ci.csv`
- `jcim_paired_bootstrap_key_deltas.csv`
- `jcim_rescue_lost_bootstrap.csv`
- `jcim_query_level_82_77_scoreblend_audit.csv`
- `jcim_major_revision_evidence_manifest.json`

Authoritative paired-bootstrap values:

- Post-audit 77-feature scorer vs initial 82-feature scorer: Top-10 Delta = +0.0393, 95% CI [+0.0354, +0.0432].
- Post-audit 77-feature scorer vs Score Blend: Top-10 Delta = +0.0686, 95% CI [+0.0638, +0.0733].

## Repository Note

At archive creation time, the local git repository root is `E:\zuhui`, while the project lives in `E:\zuhui\bioisosteric_diffusion`. Files are staged selectively from the project subtree to avoid committing unrelated sibling projects and large local artifacts.

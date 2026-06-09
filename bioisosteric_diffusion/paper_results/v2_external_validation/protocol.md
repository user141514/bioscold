# V2 External Validation Protocol

Date: 2026-06-09
Status: pre-specified, no external numerical claims until result files are locked

## Objective

Add an external evidence tier for LBC-Ranker without weakening the current ChEMBL 36 result lock.

The external package has three roles:

1. **Main external benchmark:** a BindingDB-derived MMP replacement benchmark.
2. **Auxiliary validation:** a ChEMBL temporal split.
3. **External replacement reference:** SwissBioisostere overlap and sanity checks.

The current ChEMBL 36 10-seed OF-level split remains the internal development benchmark. BindingDB is the main external dataset once its standardized matrix and result files are generated and audited.

## Source Policy

### BindingDB-Derived MMP Replacement Benchmark

Primary external source:

- BindingDB curated article TSV release, archived or version-pinned.
- Official download page: https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp
- Prefer `BindingDB_BindingDB_Articles_YYYYMM_tsv.zip` for ChEMBL-independent evidence.
- If the full BindingDB dump is used, remove rows whose source is ChEMBL before MMP construction.
- Record download URL, release date, file MD5, row count before and after filtering, and curation-date coverage.

Inclusion rules:

- Keep ligand-target affinity rows with parseable canonical SMILES.
- Keep quantitative Ki, Kd, IC50, or EC50 values with interpretable units.
- Convert nM affinity values to pActivity as `9 - log10(value_nM)`.
- Treat pKi, pKd, pIC50, and pEC50 as separate endpoint families unless an assay-level merge is explicitly documented.
- Use `pActivity >= 6` as the active threshold unless a stricter threshold is pre-registered before running.
- Define comparable compound pairs within the same target and endpoint family, and within the same assay/reaction-set context when assay identifiers are available.

MMP construction:

- Apply the same Hussain-Rea acyclic-single-bond fragmentation rules used for the ChEMBL archive.
- Canonicalize old fragment and candidate fragment SMILES with the same RDKit settings as the internal pipeline.
- Label a candidate replacement positive if at least one comparable BindingDB compound pair supports the replacement under the activity threshold.
- Treat unlabeled candidate rows as missing-negative candidates for ranking, matching the current ChEMBL convention.

Evaluation:

- Main external BindingDB benchmark: repeated OF-level 70/30 split within BindingDB, using the same locked model family, fixed feature set, and locked baseline definitions.
- The first easy-negative BindingDB matrix is a feasibility check only if CA reaches near-ceiling performance. The candidate main benchmark must use a pre-registered hard-negative policy, preferably same-target/same-endpoint/attachment-compatible candidates ranked by CA-matched similarity.
- Cross-source transfer diagnostic: train on the locked ChEMBL internal matrix and evaluate on BindingDB without any BindingDB label-derived tuning. Candidate frequency must come only from the training source.
- Report OF-macro Hit@10 as primary and query-weighted Hit@10 as secondary.
- Report dataset size, positive rate, OF coverage, candidate coverage, and zero-positive OF exclusions.

### ChEMBL Temporal Split

Purpose: auxiliary validation of time robustness, not the main external benchmark.

Construction:

- Use versioned ChEMBL releases or activity publication/curation dates when available.
- Official download documentation: https://chembl.gitbook.io/chembl-interface-documentation/downloads
- Train on earlier evidence and evaluate on later evidence.
- Exclude exact query-side transform keys and old-fragment/candidate positives seen before the temporal boundary.
- Keep the same feature definitions and evaluation code.

Minimum reporting:

- temporal boundary date or release pair,
- number of train/test OFs,
- number of held-out queries,
- candidate coverage,
- OF-macro and query-weighted Hit@10.

### SwissBioisostere Overlap / Sanity Check

Purpose: external replacement-reference sanity check, not a performance benchmark.

Reference source: http://www.swissbioisostere.ch/

Use only for:

- overlap rate between top-ranked LBC candidates and SwissBioisostere replacement records,
- enrichment of known replacement records in LBC top-K versus lower-ranked candidates,
- examples where LBC top candidates are supported by known replacement records,
- examples where no SwissBioisostere support exists, preserving the boundary that absence from SwissBioisostere is not evidence of invalidity.

Do not use SwissBioisostere for:

- model training,
- hyperparameter tuning,
- label construction for the main BindingDB benchmark,
- claims of activity preservation or wet-lab validation.

## Standardized Matrix Contract

External evaluators consume a standardized candidate matrix with one row per query-candidate pair.

Required columns:

| Column | Type | Description |
|---|---|---|
| `dataset` | string | Dataset id, e.g. `bindingdb_mmp_202606` |
| `query_id` | string | Stable query identifier |
| `old_fragment_smiles` | string | Query old fragment |
| `candidate_smiles` | string | Candidate replacement fragment |
| `label` | int | 1 for supported positive, 0 for in-matrix unlabeled/negative |

Optional precomputed feature columns:

| Column | Description |
|---|---|
| `morgan` | Candidate-OF Morgan Tanimoto |
| `bit_corr` | Candidate-OF binary fingerprint correlation |
| `dHeavy` | Absolute heavy atom difference |
| `dRings` | Absolute ring-count difference |
| `dMW` | Normalized molecular-weight difference |
| `dLogP` | Normalized logP difference |
| `dTPSA` | Normalized TPSA difference |

If optional features are absent, the evaluator computes them with RDKit. For large matrices, precomputing features is recommended.

## Commands

Smoke test:

```powershell
python paper_results/v2_external_validation/run_external_validation.py --self-test --out-dir $env:TEMP\lbc_external_selftest
```

BindingDB repeated OF split after matrix construction:

```powershell
E:\Anaconda3\envs\accfg\python.exe paper_results/v2_external_validation/build_bindingdb_mmp_matrix.py `
  --matrix-name candidate_matrix.csv.gz `
  --write-fragment-records
```

```powershell
python paper_results/v2_external_validation/run_external_validation.py `
  --matrix paper_results/v2_external_validation/bindingdb_mmp/candidate_matrix.csv.gz `
  --dataset-name bindingdb_mmp `
  --protocol repeated_of_split `
  --n-seeds 10 `
  --out-dir paper_results/v2_external_validation/bindingdb_mmp/results
```

BindingDB hard-negative diagnostic:

```powershell
E:\Anaconda3\envs\accfg\python.exe paper_results/v2_external_validation/build_bindingdb_mmp_matrix.py `
  --max-rows 10000 `
  --negative-mode ca_matched `
  --hard-negative-pool-size 5000 `
  --matrix-name candidate_matrix_10k_hard.csv.gz `
  --manifest-name source_manifest_10k_hard.json `
  --audit-name matrix_audit_10k_hard.md
```

```powershell
E:\Anaconda3\envs\accfg\python.exe paper_results/v2_external_validation/run_external_validation.py `
  --matrix paper_results/v2_external_validation/bindingdb_mmp/candidate_matrix_10k_hard.csv.gz `
  --dataset-name bindingdb_mmp_10k_hard `
  --protocol repeated_of_split `
  --n-seeds 3 `
  --out-dir paper_results/v2_external_validation/bindingdb_mmp/results_10k_hard
```

Diagnostic v2 decision rule:

- If `LBC > FullBlend > Frequency / CA` on the 10k hard-negative diagnostic, proceed to the full hard-negative 10-seed benchmark as an LBC external superiority test.
- If `Frequency ~= LBC`, retain the full benchmark but frame the claim as transferable candidate support dominating BindingDB-hard ranking.
- If `FullBlend ~= LBC` and both exceed Frequency/CA, do not claim LBC-specific superiority; frame the result as learned transferable content support dominating hand-crafted CA and raw frequency.

Current large-scale feasibility check:

```powershell
E:\Anaconda3\envs\accfg\python.exe paper_results/v2_external_validation/run_external_validation.py `
  --matrix paper_results/v2_external_validation/bindingdb_mmp/candidate_matrix.csv.gz `
  --dataset-name bindingdb_mmp_full_3seed `
  --protocol repeated_of_split `
  --n-seeds 3 `
  --out-dir paper_results/v2_external_validation/bindingdb_mmp/results_full_3seed
```

ChEMBL-to-BindingDB transfer diagnostic:

```powershell
python paper_results/v2_external_validation/run_external_validation.py `
  --train-matrix paper_results/v2_external_validation/chembl_internal_train/candidate_matrix.csv.gz `
  --matrix paper_results/v2_external_validation/bindingdb_mmp/candidate_matrix.csv.gz `
  --dataset-name chembl_to_bindingdb `
  --protocol external_holdout `
  --out-dir paper_results/v2_external_validation/chembl_to_bindingdb/results
```

## Success Criteria

Before any external result enters the manuscript:

- Dataset source URL, release date, and MD5 are recorded.
- Matrix construction script and standardized matrix are archived.
- `run_external_validation.py` completes and writes summary/detail CSV files.
- At least one independent audit confirms no ChEMBL-derived BindingDB rows leaked into the main BindingDB external benchmark.
- Manuscript claims quote only locked CSV results.

## Claim Policy

Allowed before results are locked:

- "We pre-specify BindingDB-derived MMPs as the main external benchmark."
- "ChEMBL temporal splitting and SwissBioisostere overlap are auxiliary checks."

Not allowed before results are locked:

- Any BindingDB Hit@10 value.
- Any statement that LBC-Ranker generalizes externally.
- Any claim that SwissBioisostere overlap validates activity preservation.

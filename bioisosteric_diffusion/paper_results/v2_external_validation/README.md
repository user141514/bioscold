# External Validation Package

This directory is the entry point for the new external evidence tier.

## Evidence Roles

| Role | Dataset | Claim boundary |
|---|---|---|
| Main external benchmark | BindingDB-derived MMP replacement benchmark | Performance benchmark after matrix/result/audit lock |
| Auxiliary validation | ChEMBL temporal split | Time robustness within ChEMBL source family |
| External replacement reference | SwissBioisostere overlap | Sanity/overlap only, not training or activity validation |

## Source Pointers

- BindingDB downloads: https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp
- ChEMBL downloads: https://chembl.gitbook.io/chembl-interface-documentation/downloads
- SwissBioisostere: http://www.swissbioisostere.ch/

## Current Files

- `protocol.md`: pre-specified source policy, matrix contract, commands, and claim boundaries.
- `run_external_validation.py`: locked evaluator for standardized candidate matrices.
- `build_bindingdb_mmp_matrix.py`: BindingDB curated-article TSV to target-conditioned MMP candidate matrix builder.
- `bindingdb_mmp/source_manifest.json`: latest local BindingDB build manifest.
- `bindingdb_mmp/matrix_audit.md`: latest local BindingDB matrix audit.

## Required Next Files

These are intentionally not fabricated. Add them only after the corresponding raw source has been downloaded, hashed, and transformed.

```text
bindingdb_mmp/
  source_manifest.json
  candidate_matrix.csv.gz
  matrix_audit.md
  results/
    bindingdb_mmp_summary.csv
    bindingdb_mmp_detail.csv

chembl_temporal/
  source_manifest.json
  candidate_matrix.csv.gz
  temporal_split_audit.md
  results/
    chembl_temporal_summary.csv
    chembl_temporal_detail.csv

swissbioisostere_overlap/
  source_manifest.json
  overlap_summary.csv
  overlap_examples.csv
  overlap_audit.md
```

## Smoke Test

```powershell
python paper_results/v2_external_validation/run_external_validation.py --self-test --out-dir $env:TEMP\lbc_external_selftest
```

The smoke test uses a synthetic matrix and verifies that the evaluator can split by old fragment, train LBC-Ranker, score the locked baselines, and write summary/detail outputs.

## BindingDB Build

Use a Python environment with RDKit, pandas, and scikit-learn. On this machine the verified interpreter is:

```powershell
E:\Anaconda3\envs\accfg\python.exe
```

Full curated-article matrix build:

```powershell
E:\Anaconda3\envs\accfg\python.exe paper_results/v2_external_validation/build_bindingdb_mmp_matrix.py `
  --matrix-name candidate_matrix.csv.gz `
  --write-fragment-records
```

Full 3-seed external check:

```powershell
E:\Anaconda3\envs\accfg\python.exe paper_results/v2_external_validation/run_external_validation.py `
  --matrix paper_results/v2_external_validation/bindingdb_mmp/candidate_matrix.csv.gz `
  --dataset-name bindingdb_mmp_full_3seed `
  --protocol repeated_of_split `
  --n-seeds 3 `
  --out-dir paper_results/v2_external_validation/bindingdb_mmp/results_full_3seed
```

Latest full local build, generated from 86,108 BindingDB curated-article rows:

| Metric | Value |
|---|---:|
| Active fragment records | 232,855 |
| Targets | 1,179 |
| Eligible target/endpoint/core groups | 18,715 |
| Queries | 81,839 |
| Candidate rows | 5,402,220 |
| Positive rows | 842,108 |
| Candidate fragments | 24,037 |

The large generated raw files, matrices, and result detail directories are ignored by `.gitignore`; keep or archive them as local artifacts unless a release package explicitly requires them.

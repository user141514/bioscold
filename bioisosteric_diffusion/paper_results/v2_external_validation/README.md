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
- `candidate_heldout_content_calibration_paper_design.md`: design seed for the separate candidate-heldout content-calibration paper.
- `run_external_validation.py`: locked evaluator for standardized candidate matrices.
- `build_bindingdb_mmp_matrix.py`: BindingDB curated-article TSV to target-conditioned MMP candidate matrix builder.
- `bindingdb_mmp/source_manifest.json`: latest local BindingDB build manifest.
- `bindingdb_mmp/matrix_audit.md`: latest local BindingDB matrix audit.
- `bindingdb_mmp/diagnostic_v4_cf_c3_summary.md`: first CF-C3 10k algorithm-gate result.

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

Hard-negative diagnostic build:

```powershell
E:\Anaconda3\envs\accfg\python.exe paper_results/v2_external_validation/build_bindingdb_mmp_matrix.py `
  --max-rows 10000 `
  --negative-mode ca_matched `
  --hard-negative-pool-size 5000 `
  --matrix-name candidate_matrix_10k_hard.csv.gz `
  --manifest-name source_manifest_10k_hard.json `
  --audit-name matrix_audit_10k_hard.md
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

## Benchmark Difficulty Note

The initial BindingDB full matrix uses easy unlabeled negatives and should be treated as a feasibility check. A 10k-source-row diagnostic showed that CA reached near-ceiling performance on easy negatives, but dropped from 0.994591 to 0.665755 query Hit@10 when `--negative-mode ca_matched` was used. The hard-negative design is therefore the better candidate for the main external benchmark.

Diagnostic v2 adds FullBlend and LBC no-freq. On the 10k hard-negative matrix, FullBlend slightly exceeded LBC and LBC no-freq was essentially tied with LBC on OF-macro Hit@10. Therefore the full 10-seed BindingDB-hard run should not be framed as an LBC-superiority benchmark unless a later pre-registered diagnostic reverses that ordering.

Diagnostic v3 adds candidate-heldout, strict candidate-heldout, core-heldout, stable Hit@K tie-breaking, matched inner FullBlend tuning, and candidate-heldout metrics beyond Hit@10. After audit, weak candidate-heldout removes positive candidate frequency support, but FullBlend selects pure learned content (`alpha=0`) and remains a strong baseline. Strict candidate-heldout has zero candidate-frequency signal, but raw Hit@10 is near the random expectation because the candidate pool is small; use Hit@1, Hit@5, MRR, NDCG@10, random-normalized Hit@10, and enrichment. CTCR is only marginally above LBC no-freq and near FullBlend, so the evidence supports transferable learned-content support rather than a CTCR-specific superiority claim.

Diagnostic v4 implements CF-C3, a counterfactual context-calibrated content ranker. On the 10k BindingDB-hard weak candidate-heldout diagnostic, CF-C3 exceeds the strongest learned baseline by +0.095154 OF-macro MRR and +0.117359 OF-macro NDCG@10. This passes the first algorithm gate; the next required diagnostics are candidate-cluster-heldout and CF-C3 ablations before any full huge run.

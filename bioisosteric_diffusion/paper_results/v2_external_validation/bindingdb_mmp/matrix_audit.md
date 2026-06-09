# BindingDB MMP Matrix Audit

Generated: 2026-06-09T13:39:29.584446+00:00
Dataset: `bindingdb_mmp_202606_articles`

## Source

- Download page: https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp
- Direct zip: https://www.bindingdb.org/rwd/bind/downloads/BindingDB_BindingDB_Articles_202606_tsv.zip
- Direct md5: https://www.bindingdb.org/rwd/bind/downloads/BindingDB_BindingDB_Articles_202606_tsv.md5
- Zip MD5 match: True

## Parse Counts

- `active_fragment_records`: 232855
- `active_measurements`: 43684
- `active_rows_with_fragments`: 41580
- `active_rows_without_fragments`: 2067
- `rows_read`: 86108
- `rows_with_invalid_smiles`: 359
- `rows_without_active_measurement`: 42102
- `unique_canonical_ligands_fragmented`: 26442
- `unique_context_groups`: 143040
- `unique_fragments`: 24037
- `unique_targets`: 1179

## Matrix Counts

- `candidate_fragments`: 24037
- `context_groups`: 143040
- `eligible_groups`: 18715
- `input_fragment_records`: 232855
- `negative_rows`: 4560112
- `old_fragments`: 20709
- `positive_rows`: 842108
- `queries`: 81839
- `random_seed`: 20260609
- `rows`: 5402220
- `skipped_queries_without_negatives`: 0

## Label Semantics

Positive labels mean that active BindingDB compounds support an active-active single-cut BRICS replacement within the same target, endpoint, core, and attachment signature. Zero labels are in-matrix unlabeled candidates sampled from other compatible fragment contexts; they are not asserted inactive.

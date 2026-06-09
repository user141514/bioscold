# BindingDB Full 3-Seed Feasibility Summary

Date: 2026-06-09

This is a feasibility check on the full BindingDB curated-article MMP matrix, not the final 10-seed result lock.

## Matrix Scale

| Metric | Value |
|---|---:|
| BindingDB rows read | 86,108 |
| Active fragment records | 232,855 |
| Unique targets | 1,179 |
| Eligible target/endpoint/core groups | 18,715 |
| Queries | 81,839 |
| Candidate rows | 5,402,220 |
| Positive rows | 842,108 |
| Candidate fragments | 24,037 |

## Repeated OF Split, 3 Seeds

| Method | Query Hit@10 mean | Query Hit@10 sd | OF-macro Hit@10 mean | OF-macro Hit@10 sd |
|---|---:|---:|---:|---:|
| Frequency | 0.825433 | 0.004639 | 0.649452 | 0.002142 |
| Content-aware | 0.991022 | 0.001101 | 0.995270 | 0.000363 |
| Blend | 0.996513 | 0.000363 | 0.997034 | 0.000308 |
| LBC | 0.997418 | 0.000619 | 0.997698 | 0.000525 |

Mean evaluation coverage: 24,714.3 queries and 6,213 old fragments per seed.

## Boundary

These values establish that the BindingDB-derived external matrix is large, non-empty, and evaluable with the locked scorer. Manuscript claims should still wait for the pre-specified 10-seed lock and an independent leakage/source audit.

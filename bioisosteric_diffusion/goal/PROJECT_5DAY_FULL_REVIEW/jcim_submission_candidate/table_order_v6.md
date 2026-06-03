# V6 Table Order and Captions

## Submission-Facing Table Order

The manuscript no longer uses internal `Table M*` labels. Main-text tables are numbered consecutively:

1. **Table 1**: Fresh Blind2 full-161 candidate matrix (primary prospective)
2. **Table 2**: Original secondary-blind diagnostic candidate-matrix statistics
3. **Table 3**: Leakage and overlap verification across split pairs
4. **Table 4**: Feature families of the candidate-level scorer with audit trail
5. **Table 5**: Evaluation protocol separation
6. **Table 6**: Fresh Blind2 primary prospective ranking performance
7. **Table 7**: Fresh Blind2 paired deltas and grouped uncertainty
8. **Table 8**: Secondary-blind DE/HGB Top-10 hit overlap diagnostic

## Table Captions

### Table 1. Fresh Blind2 full-161 candidate matrix used for the primary prospective evaluation.

| Split | Queries | Candidates/query | Candidate rows | Positive-set size sum | Query-side transform keys | Old fragments | Attachment signatures | Zero in-matrix-positive queries | Vocabulary size |

Train2: 93,083 queries, 161 candidates/query, 14,986,363 candidate rows.
Dev2/calibration: 27,415 queries, 161 candidates/query, 4,413,815 candidate rows.
Blind2: 17,058 queries, 161 candidates/query, 2,746,338 candidate rows.
Blind2 random expected Top-10: 0.0987.

### Table 2. Original secondary-blind diagnostic candidate-matrix statistics (retained as diagnostic history).

| Split | Candidate rows | Queries | Positive rows | Unique old fragments | Transform identities | Single-positive queries | Multi-positive queries |

Train: 109,985 queries; Development/calibration: 13,375; Secondary blind: 13,347.
Each query has 150 candidate rows.

**Note**: This is diagnostic history / original benchmark. It is not the primary prospective evaluation. The old no-prior-rank 77-feature result (0.9243 Top-10) was identified after secondary-blind diagnostics and is therefore post-audit only.

### Table 3. Leakage and overlap verification across split pairs.

| Split Pair | Query-side (fold, sigma) overlap | Old-fragment overlap | Full old-to-replacement overlap | Attachment-signature overlap |

All split pairs: query-side transform-key overlap = 0.
Fresh Blind2 old-fragment overlaps: Train2/Dev2=53, Train2/Blind2=46, Dev2/Blind2=17.
Fresh Blind2 full old-to-replacement overlaps: Train2/Dev2=393, Train2/Blind2=430, Dev2/Blind2=188.
Attachment-signature overlap: 5 for all Fresh Blind2 split pairs.
Original split pairs: old-fragment/full-overlap/attachment marked as not computed with current conventions.

### Table 4. Feature families of the candidate-level scorer with audit trail.

F1-F9 families (77 retained features), categorical (22 one-hot), prior_ranks (5 removed).
82-feature variant includes prior_ranks; 77-feature variant excludes them.

### Table 5. Evaluation protocol separation.

| Protocol | Queries | Role |
|----------|---------|------|
| Development/calibration | varies | Architecture, feature, hyperparameter, and scorer fitting |
| Original secondary blind | 13,347 | Diagnostic history, post-audit evidence, and stress-test context |
| Fresh Blind2 | 17,058 | **Primary prospective evaluation** |
| Canonical analysis | 21,052 | Robustness and mechanism support only |
| A4C / activity diagnostics | varies | Boundary and workflow diagnostics only |

### Table 6. Fresh Blind2 primary prospective ranking performance.

Primary performance table for the full 161-candidate Train2 vocabulary.

### Table 7. Fresh Blind2 paired deltas and grouped uncertainty.

Query-level, transform-key-level, old-fragment-level, attachment-signature-level, and cluster-level uncertainty are reported in main text and Supplementary Table S7.

### Table 8. Secondary-blind DE/HGB Top-10 hit overlap diagnostic.

Secondary-blind diagnostic only; not Fresh Blind2 primary evidence.

## Notes on Table Ordering

- **Table 1 (Fresh Blind2) is placed before Table 2 (original secondary-blind)** to reflect the primary prospective evidence hierarchy.
- **Table 3 (leakage/overlap)** is expanded with old-fragment, full old-to-replacement, and attachment-signature overlap columns.
- **Table 2 is labeled as diagnostic history**, with the old no-prior-rank 77-feature result demoted to Supplementary Table S1.
- This file records numbering for the markdown manuscript; journal-side production may move some methods tables to Supplementary Information during typesetting.

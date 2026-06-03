# Reviewer Major Concerns Patch V7 Response Log

Patch date: 2026-06-03.

Scope: targeted reviewer-concern patch only. No new experiments were added, the main conclusion was not changed, and HistGB77 was not re-promoted.

| # | Concern | Action taken | Files / location | Status |
|---:|---|---|---|---|
| 1 | Table 3 logical contradiction: nonzero overlap cannot be true full triple overlap when query-side `(f_old, sigma)` overlap is zero. | Renamed the nonzero column to old-to-replacement pair overlap `(f_old, c)`, excluding attachment signature. Removed main-text and SI wording that treated the nonzero value as full triple `(f_old, sigma, c)` overlap. | `manuscript_v7_reviewer_patch.md`, Table 3 and leakage text; `supporting_information.md`, SM5a. | Fixed |
| 2 | Add small closed-vocabulary boundary sentence. | Added the exact boundary sentence to the Abstract. | `manuscript_v7_reviewer_patch.md`, Abstract. | Fixed |
| 3 | Add base-ranker / feature-generation provenance table. | Added a provenance table covering DE, HGB, MLP rank-only, Score Blend, HistGB82, and HistGB77. Clarified that Dev2 base-ranker score columns are fitted-model scores, not out-of-fold predictions unless explicitly stated. | `manuscript_v7_reviewer_patch.md`, Section 3.3.5. | Fixed |
| 4 | Modify Abstract performance sentence. | Changed performance sentence to "query-level Top-10 = 0.8480" and added wider group-level Top-10 uncertainty. | `manuscript_v7_reviewer_patch.md`, Abstract. | Fixed |
| 5 | Remove LambdaRank from Abstract or add evidence table; prefer Abstract removal / future-work framing. | Kept LambdaRank out of the Abstract and removed the Results 4.7 LambdaRank evidence subsection. LambdaRank remains only as a Discussion/Future Work direction. | `manuscript_v7_reviewer_patch.md`, Abstract, Results, Discussion. | Fixed with option A |
| 6 | Add exact random baseline formula to Supplementary Methods. | Added query-level multi-positive random expected Hit@K formula: `E[Hit@K_i] = 1 - C(n_i - p_i, K) / C(n_i, K)`. | `supporting_information.md`, SM1. | Fixed |
| 7 | Change "65 Dev2 zero-positive query groups" to "65 Dev2 zero-positive queries" unless literal grouping. | Reworded to "65 Dev2 zero-positive queries." | `manuscript_v7_reviewer_patch.md`, Section 3.4.3. | Fixed |
| 8 | Do not change main conclusion. | Conclusion text and final benchmark-audit framing were preserved. | `manuscript_v7_reviewer_patch.md`, Section 6. | Preserved |
| 9 | Do not add new experiments. | No new metrics, experiments, or performance claims were introduced. Patch only clarified provenance, table semantics, and formula definitions. | All V7 patch files. | Preserved |
| 10 | Output requested files. | Created `manuscript_v7_reviewer_patch.md`, this response log, and `remaining_blockers.md`. | `jcim_submission_candidate/` | Complete |

## Notes

- The Table 3 change deliberately treats the nonzero overlap as `(f_old, c)` pair overlap. If a future script reports true full triple `(f_old, sigma, c)` overlap, it should be recomputed and should be zero whenever query-side `(f_old, sigma)` overlap is zero.
- The base-ranker provenance table is left unnumbered as a "Provenance table" to avoid renumbering all existing manuscript tables.
- LambdaRank remains mentioned in Discussion because it is a future algorithmic direction, not current main evidence.

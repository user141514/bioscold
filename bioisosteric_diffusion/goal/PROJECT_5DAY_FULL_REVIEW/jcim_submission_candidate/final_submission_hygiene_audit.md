# Final Submission Hygiene Audit

Audit date: 2026-06-03.

Scope: non-Git manuscript-package hygiene after the V6 Fresh Blind2 evidence hierarchy was aligned. This audit does not reopen the scientific spine, add experiments, or promote HistGB77.

## Verdict

| Area | Status | Notes |
|---|---|---|
| Scientific spine | Pass | Fresh Blind2 remains the primary prospective evaluation; HistGB82 remains the main Top-10 HistGB result; HistGB77 remains a no-prior-rank diagnostic comparator. |
| Feature-table semantics | Pass | Main-text Table 4 is now a HistGB82/HistGB77 feature-family comparison rather than a retained-77 table. |
| Supplement cross-references | Pass | SM1--SM6, Tables S1--S11, and Figure S4 resolve in `supporting_information.md`; activity uses S10 and calibration uses S11. |
| Citation metadata | Pass | 2025/2026 references were audited; DeepBioisostere author/year metadata was corrected and documented in `citation_audit_2025_2026.md`. |
| Abstract density | Pass | The third abstract paragraph was compressed to avoid an evidence-inventory feel while retaining group-uncertainty and boundary caveats. |
| Forbidden-claim scan | Pass | No residual old 77-centered wording, probability/likelihood overclaims, internal project labels, or internal evidence-patch labels were found in the manuscript/SI files checked. |
| Main-text table burden | Accept with caveat | Tables 1--8 are internally consistent. For journal production, Tables 2, 4, 5, and 8 can be moved to SI if space pressure arises, but this is a layout decision rather than a scientific blocker. |
| Public archive/tag | External blocker | The manuscript wording is submission-facing, but public tag visibility still must be verified outside this non-Git pass. See `final_blockers.md`. |

## Main-Text Reference Map

| Main-text reference | Supplement item | Exists? | Content matches? | Fix needed? |
|---|---|---:|---:|---|
| ChEMBL37K curation and vocabulary construction | SM1 | Yes | Yes | No |
| A4C workflow annotations | SM2 / Table S9 | Yes | Yes | No |
| Activity-comparable reranking boundary diagnostic | SM3 / Table S10 | Yes | Yes | No |
| Calibration sanity boundary diagnostic | SM4 / Table S11 | Yes | Yes | No |
| Candidate-matrix sensitivity | SM5 / Table S8 | Yes | Yes | No |
| Frozen categorical schema alignment | SM6 / Figure S4 | Yes | Yes | No |
| Original secondary-blind 82/77 diagnostic history | Table S1 | Yes | Yes | No |
| Original rescue/lost audit | Table S3 | Yes | Yes | No |
| Fresh Blind2 grouped uncertainty | Table S7 | Yes | Yes | No |

## Table Strategy

The current main manuscript keeps eight numbered tables because it is still a benchmark-audit draft with Methods-level reproducibility tables embedded in the main file. This is internally consistent:

1. Table 1: Fresh Blind2 candidate matrix.
2. Table 2: original secondary-blind diagnostic matrix.
3. Table 3: leakage and overlap verification.
4. Table 4: HistGB82/HistGB77 feature-family comparison.
5. Table 5: protocol separation.
6. Table 6: Fresh Blind2 primary performance.
7. Table 7: grouped uncertainty.
8. Table 8: secondary-blind DE/HGB overlap diagnostic.

Recommended production option if the journal asks for fewer main tables: keep Tables 1, 6, and 7 in the main text; move Tables 2, 4, 5, and 8 to SI; keep Table 3 either main text or SI depending on Methods length.

## Final Non-Git Action Items

- No manuscript-science blocker remains from this hygiene pass.
- Do not re-expand A4C, LambdaRankCat, activity diagnostics, or prior_ranks in the main text.
- Before final submission, perform a production-format pass after the journal template is applied, because table placement may change.

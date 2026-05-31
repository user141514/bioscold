# REFERENCE BIBTEX LOCK VERDICT

**Date:** 2026-05-31  
**Verdict:** **READY_WITH_MINOR_NOTES** - 18 manuscript references synchronized with `combined_draft_clean.md`. 0 critical unresolved. 1 medium note.

---

## Citation Key Scan

| Metric | Count |
|--------|-------|
| Total numbered references in `combined_draft_clean.md` | 18 |
| BibTeX entries generated | 18 |
| Fully verified / standard references | 17 |
| Preprint retained with explicit caveat | 1 (Kim2025) |
| Unresolved critical references | 0 |

## Reference Positioning Audit

All citation claim mappings were checked against the current manuscript text.

| Risk Level | Count | Status |
|------------|-------|--------|
| HIGH risk if miscited | 1 (Helmke2025) | SAFE - generic off-target-risk support only; no unsupported specific numbers |
| MEDIUM risk | 3 (Huang2025, Kim2025, Masunaga2026) | SAFE - positioned as platform/generative/GNN context, not direct closed-vocabulary baselines |
| LOW risk | 14 | SAFE - standard data, methods, software, or background references |

## 2025/2026 References

| Ref | Year | Type | Status |
|-----|------|------|--------|
| Huang2025 (NeBULA) | 2025 | Journal | DOI locked as `10.1016/j.medidd.2025.100231` |
| Kim2025 (DeepBioisostere) | 2025 | arXiv preprint | Explicitly marked preprint / not peer reviewed |
| Masunaga2026 (GraphBioisostere) | 2026 | Journal | DOI locked as `10.1007/s11227-026-08232-y` |
| Helmke2025 | 2025 | Journal | DOI locked as `10.1039/d5md00686d` |
| Landrum2026 (RDKit) | 2026 accessed | Software | Included because `combined_draft_clean.md` cites RDKit as reference 18 |

## Forbidden Patterns - All Safe

- Helmke2025: no unsupported CHRM2, 10x, or 88-target numeric claim.
- Kim2025: not positioned as a closed-vocabulary ranking baseline.
- Masunaga2026: not positioned as a same-task direct comparison.
- Huang2025: positioned as a resource/platform, not a learning baseline.
- RDKit: used only as software support for molecular descriptors.
- No fabricated BibTeX entries.

## Remaining Author Actions

1. **Kim2025:** Check before submission whether a peer-reviewed version has appeared. If yes, replace the arXiv entry.
2. **Numbered references:** If the manuscript is converted to LaTeX, replace manual numeric references with BibTeX keys to avoid renumbering drift.
3. **Year-key conventions:** Zdrazil2023 and Wirth2012 retain database-release-year keys while the publication years are 2024 and 2013 respectively; this is acceptable but should remain internally consistent.

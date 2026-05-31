# Final Automatic Check Report

Date: 2026-05-31

Files checked:

- `combined_draft_clean.md`
- `supplementary/supplementary_tables.md`

Result: 41 PASS, 0 FAIL.

## Checked Items

- Numeric locks: 82-feature Top-10 and CI, 77-feature Top-10 and CI, Score Blend paired gain, paired CIs, rescue/lost counts, and A4C G2/G3/G4 values.
- Supplementary synchronization: main text cites Supplementary Tables S3, S4, and S5; the supplementary file contains all three tables.
- Table numbering: Methods tables are uniquely numbered M1-M5; main-text tables are uniquely numbered 1-4.
- Citation numbering: references are numbered 1-18; all numeric citations resolve; no listed reference is uncited.
- Formatting residue: no `[extract]`, author-action placeholders, markdown strike-through, mojibake artifacts, or `should be deposited` language remains in the checked files.
- Claim safety: no activity-preserving prediction, activity-preserving replacement, validated medicinal replacement, wet-lab validation, expert validation, or review-safe production claim was detected.
- Data availability: the public repository URL and frozen commit hash are present.

## Notes

- The 77-feature scorer remains described as a post-audit locked result, not a fully prospective feature-selection result.
- Supplementary Table S4 reports paired CI only for the main prior_ranks deletion; other leave-one-family-out rows remain point estimates unless a separate paired bootstrap is generated.
- Supplementary Table S5 reports per-fragment point estimates and explicitly avoids per-fragment significance claims.

# Final Claim Audit, 2026-06-02

## Scope

This audit records the post-review integration of random/permissive split stress testing, targeted prior-family ablations, support-bin diagnostics, old-fragment block diagnostics, and query-level routing negative results.

Updated Markdown sources:

- `jcim_submission_candidate/main_manuscript.md`
- `jcim_submission_candidate/supporting_information.md`
- `paper_draft/combined_draft_clean.md`
- `paper_draft/supplementary/supplementary_tables.md`
- `paper_draft/paper_version2/claim_to_evidence_map.md`

Note: `main_manuscript.docx`, `main_manuscript.pdf`, `supporting_information.docx`, and `supporting_information.pdf` were not regenerated in this pass.

## Main-Text Integrations

| Item | Location | Claim boundary |
|---|---|---|
| Random/permissive split stress test | Results 4.1; Supplementary Table S6 | D4S-B0 base-ranker diagnostic only; not a competing benchmark |
| Targeted prior-family ablation | Results 4.2; Discussion 5.3; Supplementary Figure S5 | Post-audit mechanism diagnostic; not prospective feature selection |
| Support-bin audit | Results 4.2; Discussion 5.3; Supplementary Figure S5 | Support/sparsity diagnostic; not causal proof |
| Old-fragment block update | Results 4.1/4.3; Supplementary Table S5 | Point-estimate block diagnostic; not per-fragment significance |
| Query-level routing | Discussion 5.2 | Negative diagnostic only; not a contribution |

## Locked Diagnostic Values Added to Claim Map

| Quantity | Value | Status |
|---|---:|---|
| Drop all non-rank prior families | +0.0174 | Post-audit diagnostic |
| Drop prior statistics | +0.0151 | Post-audit diagnostic |
| Drop prior scores | +0.0073 | Post-audit diagnostic |
| Drop PMI/conditional-prior contrasts | -0.0037, CI [-0.0064, -0.0010] | Post-audit diagnostic |
| Low-support 77-minus-82 gain | +0.0488 | Support-bin diagnostic |
| Medium-support 77-minus-82 gain | +0.0344 | Support-bin diagnostic |
| High-support 77-minus-82 gain | +0.0346 | Support-bin diagnostic |
| Old-fragment strata non-negative vs 82-feature scorer | 17/19 | Point-estimate block diagnostic |
| Random/permissive train-blind transform overlap | 840/840 | Split-protocol diagnostic |
| Random/permissive DE Top-10 | 0.8784, Delta +0.0729 | Base-ranker diagnostic |
| Random/permissive HGB Top-10 | 0.8191, Delta +0.0755 | Base-ranker diagnostic |
| Random/permissive Borda Top-10 | 0.8699, Delta +0.0316 | Base-ranker diagnostic |
| Random/permissive Attachment-Frequency Top-10 | 0.5727, Delta -0.0292 | Boundary diagnostic |
| Runtime-safe query router vs Borda | Delta 0.0000, CI [-0.0008, 0.0008] | Negative diagnostic |
| Routed blind queries | 884/13,347 | Negative diagnostic |
| Learned query-router development AUC | about 0.61 | Negative diagnostic |

## Claim-Safety Checks

- PASS: No zero-of-19 or all-fragment-positive old-fragment claim remains in the Markdown manuscript/SI/claim map.
- PASS: Random/permissive split is restricted to D4S-B0 base-ranker diagnostics.
- PASS: Score Blend, 82-feature scorer, and 77-feature scorer are not claimed under random/permissive split.
- PASS: Prior-family results are marked post-audit/diagnostic and do not claim all prior information is harmful.
- PASS: Support-bin results are framed as diagnostics, not causal proof.
- PASS: Query-level routing is downgraded to a negative diagnostic.
- PASS: A4C remains computational triage only.
- PASS: No activity-preserving or wet-lab validation claim was introduced.

## Remaining Packaging Step

Regenerate DOCX/PDF files from the updated Markdown before freezing a new submission package.

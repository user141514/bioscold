# Consistency Repair Log

## P0 Repairs

- Removed journal-targeting language from the Introduction.
  - Old risk: `This paper addresses the JCIM evaluation question directly`.
  - Current wording: `We address this evaluation problem directly`.

- Unified HGB-refit diagnostic value.
  - Current manuscript value: HGB-refit candidate score Top-10 = 0.8623.
  - Main Table 1 HGB remains the standalone base ranker with Top-10 = 0.7437.

- Added claim-level evidence inventory for secondary diagnostics.
  - New file: `claim_to_evidence_map.md`.
  - Covered diagnostics: Top-1, MRR, old-fragment macro, attachment-signature macro, old-fragment/attachment identity, and 2/19 negative old-fragment blocks.

- Removed internal experiment label from the main text.
  - Old risk: `D4S-B0 blind replay`.
  - Current wording: `standalone base ranker`.

- Replaced old feature-finalization wording.
  - Old risk: `feature set ... finalized through systematic leave-one-family-out ablation`.
  - Current wording: `We used leave-one-family-out ablation to identify the post-audit feature-pruned configuration`.

## P1 Repairs

- Reduced A4C prominence in the Introduction.
  - Introduction now says only that A4C provenance strata are coverage-limited computational triage.
  - G2/G3/G4 numerical details remain in Results 4.5 and Discussion 5.1.

- Operationalized the signal-transfer framing in Discussion 5.3.
  - Four signal roles are now explicit: transform-memory signal, transferable candidate-level signal, query-relative shortcut signal, and workflow triage signal.

- Clarified Top-10 scope in the Abstract.
  - The abstract now says the 77-feature scorer improved Top-10 recovery over Score Blend.

## P2 Repairs

- Corrected the DeepBioisostere author list in reference 9.
  - Current author list: Kim, H.; Moon, S.; Zhung, W.; Lim, J.; Kim, W. Y.
  - Checked against arXiv: https://arxiv.org/abs/2403.02706.

- Added a scope note to `spine_audit.md`.
  - The spine audit now states that it is a framing audit, not a substitute for data, reference, table, or figure-caption verification.


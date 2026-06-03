# Spine Change Log

## Edited Sections

- Abstract: rewritten into three compact paragraphs following problem/benchmark, scoring result, audit finding and boundaries.
- Introduction opening paragraph: reframed the task around closed-vocabulary scaffold-conditioned ranking and structure-derived recovery, not activity preservation.
- Introduction evaluation-risk paragraph: made transform-level memorization and shortcut features under fragment distribution shift the core evaluation problem.
- Introduction contribution paragraph: reordered contributions as benchmark, candidate-level scoring, prior-rank shortcut diagnosis, with A4C as workflow-only.
- Introduction spine paragraph: added the central signal-transfer question explicitly.
- Introduction evidence-hierarchy paragraph: preserved locked values while distinguishing prospective 82-feature evidence from post-audit 77-feature evidence.
- Introduction organization paragraph: organized the manuscript by evidence role: prospective evaluation, post-audit mechanism, workflow triage.
- Discussion 5.1: reduced principal findings to three linked contributions and demoted query-level routing to a secondary design observation.
- Discussion 5.3: strengthened the signal-provenance theory under distribution shift and retained alternative explanations.
- Conclusion: rewritten into three short paragraphs answering pain point, benchmark/scoring result, audit finding, and remaining limitations.
- Reviewer-consistency repair: removed journal-targeting language, replaced internal experiment label `D4S-B0`, unified HGB-refit diagnostic value to 0.8623, removed the old feature-finalization wording, and corrected the DeepBioisostere author list.
- Evidence repair: added `claim_to_evidence_map.md` to cover Top-1, MRR, block-level diagnostics, baseline naming, and workflow-only A4C claims.

## What Did Not Change

- No locked numerical value was changed.
- No Methods section was rewritten wholesale.
- No Results section was rewritten wholesale.
- No equations were changed.
- References were not structurally reordered; reference 9 author list was corrected after a targeted reference check.
- No tables were modified.

## Framing Changes

- The paper is now framed as a leakage-controlled benchmark and audit framework.
- Candidate-level scoring is the scoring contribution, not a generic HistGB claim.
- prior_ranks pruning is an audit finding, not the central innovation by itself.
- A4C is coverage-limited computational triage, not validation or safety scoring.
- Query-level routing failure is secondary discussion, not a core contribution.

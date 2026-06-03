# Reviewer Packet

## Two-Sentence Thesis

This paper frames closed-vocabulary fragment replacement ranking as a benchmark-audit problem: first remove transform memorization, then test which empirical, structural, and candidate-level signals still transfer under a locked secondary blind protocol. Under that framing, candidate-level scoring improves Top-10 recovery, while the post-audit prior_ranks diagnosis shows that sparse query-relative empirical ranks can behave as non-transferable shortcuts rather than robust medicinal-chemistry evidence.

## Contribution Hierarchy

1. **Benchmark necessity.** The transform-heldout secondary blind protocol removes repeated fragment-attachment transform identities before ranking claims are made.
2. **Candidate-level scoring evidence.** The HistGB candidate-level scorer tests signal transfer at the query-candidate level and improves secondary blind Top-10 recovery over Score Blend.
3. **Prior_ranks shortcut mechanism.** Post-audit ablations identify prior_ranks as the dominant unstable feature family, with the largest gains in low train-derived support regimes.
4. **Mechanism anchors and boundaries.** Borda(DE,HGB) remains a mechanism anchor for DE/HGB complementarity; Best-of-DE+HGB is diagnostic only; A4C is workflow-only computational triage.
5. **Limitations.** The 77-feature scorer is a locked post-selection result, labels are structure-derived rather than activity-preserving, and no wet-lab or expert validation is claimed.

## Three Core Figures

| Core figure | Manuscript status | Reviewer-facing message |
|---|---|---|
| Permissive vs transform-heldout stress test | Figure 1 plus Results 4.1 / Supplementary Table S6 | Random/permissive splits can leak transform identities; transform-heldout secondary blind evaluation is the necessary benchmark frame. |
| Secondary blind Top-10 performance | Main Figure 3 | Candidate-level scoring improves Top-10 recovery under transform-heldout secondary blind evaluation; 77-feature is post-audit locked, not fully prospective. |
| Prior_ranks mechanism audit | Main Figure 4 plus Supplementary Figure S5 | Removing prior ranks is the dominant positive post-audit ablation, reduces lost queries, and is strongest in low-support regimes; the interpretation is shortcut diagnosis, not universal proof about prior information. |

## Four Questions for Expert Reviewers

1. Does the split stress test and transform-heldout protocol convincingly justify treating secondary blind evaluation as the primary benchmark?
2. Are the boundaries between the prospective 82-feature scorer and the locked post-audit 77-feature scorer clear enough to avoid overclaiming feature selection?
3. Does the full-metric profile support a Top-10 recovery claim without implying broad ranking dominance across Top-1, Top-50, and MRR?
4. Should A4C provenance triage remain in the main manuscript as workflow-only support, or should it be moved further into Supporting Information to keep the paper centered on benchmark audit and signal transfer?

# Discussion Material: Claim Boundary and Post-Audit Status

Moved from Results §4.7. Integrate into Discussion §5 under a subsection like "Claim boundaries and post-audit status" or "Limitations."

---

The prior_ranks ablation was motivated by post-audit analysis after the initial 82-feature blind evaluation revealed unexpected per-fragment degradation (Section 3.5). The mechanistic explanation---shortcut learning driven by non-transferable per-query rank features---is independently testable: the 8-to-19 fragment identity shift, the query-relative nature of rank features, and the threshold-transfer failure mode are each observable properties of the data and model, not post-hoc rationalizations.

We note two caveats. First, the specific performance gain of +0.0392 from prior_ranks removal may partially reflect the post-hoc nature of the finding, since the removal decision was informed by the same blind evaluation that produced the baseline 0.8851. Second, the design rule---"per-query rank features are a generalization hazard"---is presented as an analytical finding emerging from this study, not as a pre-registered hypothesis. Independent replication on structurally distinct fragment replacement benchmarks would strengthen its generality.

---

**Integration notes:**
- This material pairs naturally with the limitation that structure-derived labels do not establish activity preservation
- Can be combined with a broader "Limitations" subsection covering: post-hoc feature selection, structure-derived labels, A4C proxy scope, single-dataset evaluation
- The Results §4.2 now contains a one-sentence pointer: "Because this ablation was motivated by post-audit analysis, we treat the design rule as an analytical finding rather than a pre-registered hypothesis; the full claim-boundary discussion appears in Section 5."

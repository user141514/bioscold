# Core Figure Standalone Audit

Audit objective: force the visual evidence into one benchmark-audit argument: remove transform memorization first, then test which ranking signals still transfer.

## Figure 1: Permissive vs Transform-Heldout Stress Test

| Item | Audit |
|---|---|
| Current visual artifact | Active Figure 1 (`figures/main/frozen_jcim_style_v4/Figure_1_harmonized.*`) explains the closed-vocabulary task, random split leakage risk, and transform-heldout protocol. The numeric stress test is currently reported in Results 4.1 and Supplementary Table S6 rather than as a dedicated plotted figure. |
| Standalone conclusion | Permissive/random splitting can place the same fragment-attachment transform in training and evaluation, so transform-heldout evaluation is necessary before ranking signals can be trusted. |
| Supports conclusion without text? | Partly. The schematic clearly explains the leakage mechanism and the transform-heldout cure. It does not visually show the measured stress-test result that all 840/840 permissive blind transform identities overlap training or that learned base-ranker performance inflates under this condition. |
| Needed title/caption change | Title is suitable. Caption should explicitly mention that Results 4.1 and Supplementary Table S6 provide the numeric stress test, while the figure defines the leakage-control rationale. If a stricter core-figure sequence is desired, add a small main-text inset or new compact panel showing "permissive overlap: 840/840" and "transform-heldout overlap: 0". |
| Main or supplementary? | Main. This is the benchmark necessity figure. If space allows, it should become the first core evidence figure by incorporating a tiny numeric stress-test inset; otherwise keep Figure 1 main and cite Supplementary Table S6 immediately in Results 4.1. |
| Risk | Without a numeric inset, a skeptical reader may see the figure as protocol explanation rather than direct evidence. The manuscript text must compensate by making the stress-test numbers prominent in Results 4.1. |

## Figure 3: Secondary Blind Top-10 Performance

| Item | Audit |
|---|---|
| Current visual artifact | Active Figure 3 (`figures/main/frozen_jcim_style_v4/Figure_3_harmonized.*`) is a single-panel horizontal forest/dot plot with query-bootstrap CIs. |
| Standalone conclusion | Under transform-heldout secondary blind evaluation, candidate-level scoring improves Top-10 recovery beyond the strongest pre-D4S baseline, with the post-audit 77-feature scorer giving the largest Top-10 result. |
| Supports conclusion without text? | Yes for Top-10 hierarchy. The method order, CIs, and weak evidence tags make the hierarchy readable without extra prose. The figure correctly keeps Best-of-DE+HGB diagnostic secondary. |
| Needed title/caption change | Caption should keep the evidence boundary: 82-feature = prospective candidate-level blind result; 77-feature = locked post-audit feature-pruned result; Best-of-DE+HGB = diagnostic bound for per-query DE/HGB selection, not a ceiling. Title can stay "Secondary blind Top-10 performance"; "transform-heldout" may be added in the caption rather than title to avoid clutter. |
| Main or supplementary? | Main. This is the primary candidate-level scoring evidence figure. |
| Risk | The figure only shows Top-10. Supplementary Table S2 must remain visible because the full metrics show that the 77-feature scorer does not dominate every ranking metric, especially Top-1 and MRR. |

## Figure 4 / Supplementary Figure S5: Prior_Ranks Mechanism Audit

| Item | Audit |
|---|---|
| Current visual artifact | Active Figure 4 (`figures/main/frozen_jcim_style_v4/Figure_4_harmonized.*`) shows the prior-rank deletion ablation and reliability/rescue-lost audit. Supplementary Figure S5 (`figures/supp/Figure_S5_prior_ranks_mechanism_audit.*`) adds targeted prior-family ablations and train-derived support-bin diagnostics. |
| Standalone conclusion | The post-audit 77-feature improvement is linked to removal of query-relative prior-rank features, which behaved as non-transferable shortcuts under secondary blind fragment shift; the strongest gain appears in low-support regimes. |
| Supports conclusion without text? | Figure 4 supports the broad mechanism: prior-rank deletion is the dominant positive family-level ablation and lost queries fall sharply. Supplementary Figure S5 supports the narrower mechanistic claim that the unstable component is rank-specific rather than prior information in general. |
| Needed title/caption change | Figure 4 caption should retain "post-audit" and "locked post-selection" wording. Supplementary Figure S5 caption already states "post-audit targeted mechanism diagnostic, not prospective feature selection"; this wording should remain. |
| Main or supplementary? | Figure 4 should remain main. Supplementary Figure S5 should remain supplementary unless the manuscript narrows further around prior-rank mechanism; if the core reviewer concern becomes "why prior ranks specifically?", then S5 can be promoted or summarized as a small Figure 4 inset. |
| Risk | Overclaiming would be easy here. The safe conclusion is not "all prior information is harmful"; it is that sparse query-relative prior ranks were the dominant unstable feature family in this benchmark audit. |

## Core Figure Verdict

The current figure sequence mostly supports the requested argument. The only gap is that the "permissive vs transform-heldout stress test" exists as text/table evidence rather than as a fully standalone plotted figure. Figure 3 and Figure 4 are main-figure ready for the performance and mechanism portions; Supplementary Figure S5 is the reviewer-defense figure for the targeted prior-family interpretation.

# Motivation Options After Research

Cross-referencing: Scene Analyst (JCIM venue), Exemplar Learner (structural patterns), SOTA Mapper (gap analysis).

---

## How to Read These Options

Each option represents a different controlling motivation for the paper's narrative arc. All four are valid — they differ in emphasis, not in truth. Choose the one that best matches what you want the paper to communicate.

---

| # | One-Sentence Motivation | Core Innovation | Why It Is Not Overbroad | Required Evidence | Best-Fit Paper Arc |
|---|---|---|---|---|---|
| **A** | Fragment replacement ranking needs leakage-controlled evaluation because random splits enable shortcut memorization, and we show that a candidate-level scorer with audited features achieves strong blind performance (Top-10 = 0.9243) while a post-audit shortcut removal reveals where generalization fails. | Transform-heldout benchmark as evaluation standard | Evaluation protocol is concrete and reproducible — not claiming a universal method, claiming a specific improvement under a specific protocol | Top-10 results (Table 1), leakage verification (Table M2), rescue/lost (Table 3) | **JCIM benchmark paper.** Gap-narrowing intro (leakage is a known confound → no existing benchmark controls for it → we build one), benchmark-first results, ablation as mechanism, limitations-aware close. **Current manuscript already follows this arc.** |
| **B** | The most impactful feature-engineering decision in molecular ranking can be a deletion, not an addition: removing five per-query prior-rank features eliminates a non-transferable shortcut, improving blind Top-10 by +0.0393 with a 5.9× reduction in baseline hit loss. | Shortcut learning in fragment ranking — an audit methodology, not a model | Finding is specific (prior_ranks family), mechanistic (8-to-19 fragment shift), and testable (reproducible with released code) | Table 2 ablation, Section 3.5.2 mechanism, Figure showing per-fragment degradation (if exists), Supplementary Table S4 full ablation | **Mechanism-discovery paper.** Intro frames shortcut learning gap (Geirhos 2020), Methods describe the unexpected finding, Results center on ablation, Discussion generalizes to design rules. **Current draft has strong elements of this arc (Section 3.5, 4.2) but subordinates it to the benchmark arc A.** |
| **C** | We provide a complete fragment replacement ranking system combining leakage-controlled evaluation, candidate-level scoring, and a dual-mode provenance-stratified workflow that communicates computational alert risk transparently. | Full pipeline: benchmark + scorer + workflow with risk communication | System is practical, not claiming algorithmic novelty in each component — explicitly a "framework" paper | Table 1 (scoring), Table 4 (A4C alerts), dual-mode definitions (Section 3.7), code archive | **Systems/application paper.** Broader framing, intro opens with practical need (medicinal chemists need ranking tools with risk labels), Methods covers entire pipeline, Results show each component, Discussion emphasizes practical utility. **Current draft's A4C section is the weakest component for this arc (sparse G4 coverage, unvalidated alerts).** |
| **D** | For molecular ranking tasks where the predictive signal is distributed across individual query-candidate pairs, candidate-level scoring consistently outperforms query-level routing, and we demonstrate why in a leakage-controlled fragment replacement benchmark. | Ranking architecture principle (candidate-level > query-level) | Claim is about architecture design, not about bioisosterism specifically — testable transfer to other ranking domains | Table 1 (candidate-level vs. base rankers), Section 5.2 analysis (routing failures), Best-of-DE+HGB diagnostic | **Architecture insight paper.** Intro frames the routing vs. scoring design question, Methods tests both architectures, Results show candidate-level wins, Discussion generalizes to design principle. **Current draft's Section 5.2 is strong but brief — would need expansion to carry the full paper.**

---

## Recommendation

**Option A** (benchmark paper) is the best fit for JCIM because:
1. JCIM regularly publishes benchmark-driven methods papers — this is the journal's core genre
2. The current manuscript already follows this arc closely
3. The transform-heldout split is the paper's most distinctive contribution (no other bioisostere paper does this)
4. Options B, C, and D can serve as **secondary narrative threads** within arc A — they don't replace it

**Suggested hybrid:** Arc A as the primary spine, with arc B as the strongest secondary thread (the prior_ranks deletion is the paper's most memorable single finding). Relegate C (A4C) to a supporting role as currently structured. Elevate D (candidate-level > query-level) from Discussion Section 5.2 to a more prominent Results subsection if space allows.

---

## User Decision Required

Choose, revise, or write your own motivation. After confirmation, `confirmed_motivation.md` will be created and the rewrite can proceed.

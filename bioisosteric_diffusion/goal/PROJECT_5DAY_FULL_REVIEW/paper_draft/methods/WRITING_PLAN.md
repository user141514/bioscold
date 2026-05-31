# Methods Section -- Writing Order Plan

## Paper Title
A Dual-Mode Workflow for Scaffold-Conditioned Fragment Replacement Proposal via Neural-Empirical Rank Fusion

## Recommended Writing Order (Dependency-Driven)

```
3.1 Task Formulation ────────────────────────────────────────────── (no deps)
       │
       v
3.2 Benchmark Construction ──────────────────────────────────────── (needs 3.1 notation)
       │
       v
3.3 Base Ranking Methods ────────────────────────────────────────── (needs 3.1 metrics, 3.2 splits; provides inputs to 3.4)
       │
       v
3.4 Candidate-Level Scorer ──────────────────────────────────────── (needs 3.3 base ranker outputs; provides model to 3.5)
       │
       v
3.5 Ablation: Prior_Ranks Removal ───────────────────────────────── (needs 3.4 feature set; provides final model to 3.6)
       │
       v
3.6 Evaluation Protocol ─────────────────────────────────────────── (needs results from 3.3-3.5 for context; describes metrics already defined in 3.1)
       │
       v
3.7 Computational Review Proxy (A4C) ────────────────────────────── (needs 3.3 base rankers for fusion modes)
```

## Dependency Rationale

| # | Module | Depends On | Provides | Writer Skill |
|---|--------|-----------|----------|--------------|
| 1 | 3.1 Task Formulation | None | Notation (Q, C_i, P_i, rho, hit@K) | Mathematical formulation |
| 2 | 3.2 Benchmark | 3.1 notation | Data splits, leakage control | Dataset construction |
| 3 | 3.3 Base Rankers | 3.1, 3.2 | Five methods + performance table | ML methods exposition |
| 4 | 3.4 Candidate Scorer | 3.3 | 77-feature scorer architecture | Feature engineering + ML |
| 5 | 3.5 Ablation | 3.4 | prior_ranks removal, shortcut learning | Experimental analysis |
| 6 | 3.6 Evaluation | 3.3-3.5* | Bootstrap CI, rescue/lost, protocol separation | Statistical methodology |
| 7 | 3.7 A4C Proxy | 3.3 | Dual-mode workflow, provenance labels | Workflow/risk exposition |

\* 3.6 references results from 3.3-3.5 but can be drafted with placeholder numbers.

## Cross-Cutting Concerns

1. **Notation consistency** -- All modules must use the same symbols: $q_i = (f_i^{\text{old}}, \sigma_i)$, $\mathcal{C}_i$, $\mathcal{P}_i$, $\rho_\theta(i,c)$, $\operatorname{Top@K}$. The notation is established in 3.1 and must be propagated.

2. **Table/figure numbering** -- Tables in the unified draft are numbered 1-7 across all sections. The full draft renumbers for each section. Decide on global vs. per-section numbering before final assembly.

3. **Reference to Section 2.2** -- The provenance levels (P0, P1, P3, P4) used in 3.4 for prior features are defined in Section 2.2 (Related Work / Preliminaries). Ensure that section exists and defines these.

4. **Metric definitions** -- Top@K and MRR are defined in 3.1. The 3.6 section describes bootstrap CI on these metrics. Avoid re-defining; cross-reference.

5. **Results overlap** -- The full draft currently places results (Section 4) inline with methods in some places (e.g., per-ranker performance numbers in 3.3). The Methods section should describe *how* evaluation is done, while *what* the numbers are belongs in Section 4. However, base ranker performance (Table 2) serves dual purpose as both method characterization and result. Keep base ranker table in 3.3, move scorer main results to Section 4.

## Current Draft Status Summary

| Module | Unified Draft | Full Draft (math_polished_B) | Gaps |
|--------|--------------|------------------------------|------|
| 3.1 | Complete, with slight notation differences | Complete, more formal | Unify notation: use $\mathcal{P}_i$ vs $P_q$, $\mathcal{V}$ vs $V$ |
| 3.2 | Complete | Complete | Minor: include 1:5 diagnostic set table |
| 3.3 | Complete, LaTeX-formatted | Complete, Markdown with math | Dual versions; align Oracle ceiling description |
| 3.4 | Complete | Complete | Check feature counts (82 vs 77 narrative flow) |
| 3.5 | Complete | Complete | Rescue/lost in 3.5 (ablation) overlaps with 3.6.3 (method). Consolidate: keep definitions in 3.6, results in 3.5. |
| 3.6 | Complete | Has redundant rescue/lost equations (3.6.3 vs 3.5) | Deduplicate. 3.6 should define the *methodology* of rescue/lost; 3.5 shows *results* of the ablation. |
| 3.7 | Complete | Complete | Add the A4C alert-rate table consistently |

## Estimated Writing Effort

| Module | Est. Words | Key Challenge |
|--------|-----------|---------------|
| 3.1 | ~600 | Getting notation exactly right; tie-breaking rank definition |
| 3.2 | ~900 | Clear split design motivation; leakage verification |
| 3.3 | ~1500 | Balancing detail for 5 methods; Oracle ceiling framing |
| 3.4 | ~1800 | Feature table clarity; cat_templates explanation; training protocol |
| 3.5 | ~1000 | Shortcut learning narrative; rescue/lost mechanism |
| 3.6 | ~900 | Not repeating 3.1 metrics; protocol separation table |
| 3.7 | ~700 | G2/G3/G4 provenance clarity; scope limitations |
| **Total** | **~7400** | |

# Module 9 — Paper-S6R / S7 / D4A2 Task Decomposition Summary

## Starting Problem

The S6R merge attempted to upgrade the S4GEN manuscript by replacing Borda as the main method with a stronger, validation-selected MLP+HGB-score blend that achieves higher blind Top10. The S6R line includes: (1) a validation-selected score blend as the primary internal blind method, (2) external curated ring-replacement recall (RRR) using Ertl templates, (3) retrospective activity-retained support from ChEMBL (S7), and (4) A4C computational risk stratification. The D4A2 task decomposition defined the original plan for canonical metric freeze, ranker ceiling testing, D4B-lite control, and A4C preview. Separately, an extensive algorithm optimization synthesis (D4S7 through D4S26/c) tested whether any post-hoc score fusion, rescue, boundary-expert, or feature-expansion approach could improve beyond the score blend.

---

## Key Findings

### S6R Merge Claim-Denoised Upgrade

- **Adoption verdict**: B. S6R_ADOPT_WITH_CAVEATS (not unconditional A).
- **Score blend is the new main method**: MLP+HGB-score blend achieves Top10=0.8558, MRR=0.4842 on blind (13,347 queries).
- **Score blend vs Borda**: delta=+0.0174, 95% CI [0.0145, 0.0203], significant.
- **Borda is demoted** from final method to "strong fixed rank-fusion baseline and mechanism anchor."
- **Score blend selected on validation**, evaluated once on blind — no blind labels used for selection.
- **RRR external test**: 36 templates, 243 directed pairs. PairTop10 gain = +0.0741, PairMRR gain = +0.0319. Query Top10 gain = +0.0556 but 95% CI [-0.1667, 0.2778] — directional only.
- **S7 retrospective activity**: 11,520 same-target same-assay comparable pairs; 9,614 activity-retained. Score blend and Borda are TIED on this endpoint: score blend T10=0.7480 vs Borda T10=0.7481, delta=-0.0001, CI [-0.0024, 0.0023].
- **Why not verdict A (unconditional adoption)**: RRR query-level CI crosses zero; publication-weighted metrics carry bias risk; S7 does not favor score blend over Borda; A4C is not expert validation.

### S6R Journal-Facing Draft

- **Verdict**: S6R_TEXT_READY_AS_JOURNAL_FACING_MAIN_MANUSCRIPT_DRAFT.
- **Primary claim**: "The validation-selected MLP+HGB-score blend is the best internal blind Top10 method."
- **Borda role**: "Strong fixed baseline and mechanism anchor."
- **RRR role**: "Bounded curated ring-replacement recall, not prospective bioisostere discovery."
- **S7 role**: "Retrospective endpoint support, not model superiority."
- **A4C role**: "Computational risk stratification, not expert validation."
- Internal stage labels removed. Report-style evidence hierarchy replaced with journal-facing narrative flow. Table 1 compressed to manuscript-facing columns.

### D4A2 Task Decomposition

- Defines 5-part execution order: A (canonical metric freeze) -> B (ranker ceiling) -> C (D4B-lite model-class control) -> D (A4C integration) -> E (final decision).
- **Five numbers to reconcile**: 60.61 (D4A0 baseline), 62.42 (D4A1 recomputed), 70.08 (D4A1 bootstrap), 72.17 (D4A1 test_metrics), 74.66 (D4A1R retrained diagnostic).
- Recommended canonical: **72.17** for D4A1 learned model, **60.61** for D4A0 frozen baseline.
- Ranker ceiling tests: B0 (HGB reproduction), B1 (LambdaRank), B2 (XGBoost pairwise), B3 (pairwise logistic), B4 (listwise softmax), B5 (two-stage reranker), B6 (Dual Encoder optional).

### Algorithm Optimization Synthesis (D4S7 through D4S26)

- **Current best method**: `0.95 * z(M5 rank-only MLP score) + 0.05 * z(HGB refit score)`. Validation Top10=0.834019, blind Top10=0.8558.
- **All post-hoc approaches rejected for Top10 upgrade**:
  - D4S7: Boundary classifiers (AUC=0.944-0.954 but ranking policies failed).
  - D4S8: Direct ranking (LambdaRank F2 best=0.827514, below baseline 0.834019).
  - D4S9: Tier-1 SMILES/2D feature reranker (best delta Top10=+0.000075).
  - D4S10: Near-miss boundary audit (rank 11-20 near misses=1,013 queries, but limited headroom).
  - D4S11: Signal-supported override (395 policies evaluated, no policy reached +0.003).
  - D4S12: Train-only exact context prior (557 policies, max S2 delta=0.0).
  - D4S13: Old-fragment KNN prior (best delta=+0.003065, but gain/loss ratio=1.02 — nearly equal gains and losses).
  - D4S13R: Precision gates (9,070 policies, 0 passing).
  - D4S14/D4S14R: KNN trust router / transparent gates (best delta=+0.003738, but rule selected on validation).
  - D4S15: Stress test (locked gate delta=-0.004682 under heldout — NEGATIVE).
  - D4S16: Loss-aware guards (best delta=-0.000082, CI includes zero).
  - D4S17: Fragment-context compatibility (best delta=+0.001047 on val, +0.000200 on heldout).
  - D4S18/D4S19: Top10-preserving early-rank refinement (MRR gain on val, but old-heldout unstable).
  - **D4S20: Stopline** — locked score blend as primary method. Stop post-hoc fusion/reranking search.
  - D4S21: Candidate-pool audit (pool failure rate only 3.07%, ranking failure dominates).
  - D4S22: Scoring-failure anatomy (oracle near-only 11-20: 993 queries; deep miss >50: 45 queries).
  - D4S23: Rare-replacement de-bias (best delta=+0.000473 — too small).
  - D4S24: Rare-weighted boundary specialist (delta=0.000000 — zero boundary movement).
  - D4S25: Orthogonal feature repair (new features 0.021 importance sum vs base 0.302; best delta=0.000000 on val, -0.003801 on heldout).
  - D4S26: Loss-aware candidate-frequency guard (best delta=0.000000 — no-op).
- **Final algorithmic verdict**: The current feature/rank/signal stack is saturated for Top10. Next improvement requires new base representation (ACCFG, pharmacophore, learned embeddings, 3D/electrostatic), not another fusion/guard.

---

## Key Files (S6R Merge)

| File | Path |
|------|------|
| Adoption Verdict | `plan_results/routeA_paper_s6r_merge_claim_denoised_upgrade/PAPER_S6R_MERGE_ADOPTION_VERDICT.md` |
| Decision Log | `plan_results/routeA_paper_s6r_merge_claim_denoised_upgrade/MAIN_DECISION_LOG.md` |
| Manuscript Candidate | `plan_results/routeA_paper_s6r_merge_claim_denoised_upgrade/paper_s6r_merge_manuscript_candidate.md` |
| Claim Safety Report | `plan_results/routeA_paper_s6r_merge_claim_denoised_upgrade/paper_s6r_merge_claim_safety_report.md` |
| Updated Table 1 | `plan_results/routeA_paper_s6r_merge_claim_denoised_upgrade/paper_s6r_merge_table1_updated.csv` |
| RRR Claim Control | `plan_results/routeA_paper_s6r_merge_claim_denoised_upgrade/paper_s6r_merge_rrr_claim_control.md` |
| S7 Claim Control | `plan_results/routeA_paper_s6r_merge_claim_denoised_upgrade/paper_s6r_merge_s7_claim_control.md` |
| S6R Journal-Facing Verdict | `plan_results/routeA_paper_s6r_text_journal_facing/PAPER_S6R_TEXT_VERDICT.md` |
| S6R Journal-Facing Manuscript | `plan_results/routeA_paper_s6r_text_journal_facing/paper_s6r_text_journal_facing_main_manuscript.md` |
| D4A2 Task Decomposition | `task/d4a2_task_decomposition.md` |
| Algorithm Optimization Synthesis | `plan_results/routeA_algorithm_optimization_synthesis_20260528.md` |
| Algorithm Optimization Verdict | `plan_results/ROUTEA_ALGORITHM_OPTIMIZATION_CURRENT_VERDICT.md` |

---

## Core Numbers

1. **Score blend blind Top10 = 0.8558**, MRR = 0.4842 (current best internal method).
2. **Score blend vs Borda**: delta = +0.0174, 95% CI [0.0145, 0.0203], significant.
3. **RRR external test**: PairTop10 gain = +0.0741, PairMRR = +0.0319. Query Top10 gain = +0.0556, 95% CI [-0.1667, 0.2778] (directional only).
4. **S7 retained pairs**: 9,614 activity-retained. Score blend T10 = 0.7480 vs Borda T10 = 0.7481 (TIED).
5. **D4S31 blind Top10 = 0.9243** (with prior_ranks removed, valid) — see separate module for details.
6. **D4S28R blind Top10 = 0.8851** (aligned scorer_all).
7. **Algorithm stopline confirmed**: 20+ post-hoc modules (D4S7 through D4S26) all failed to produce robust Top10 gain.
8. **Candidate-pool failure rate**: 3.07%. Ranking/scoring failure dominates (16.6% seen-vocab miss rate).

---

## Verdict

**S6R is the better high-upside manuscript line (vs S4GEN), adopting the validation-selected MLP+HGB-score blend as the primary internal blind Top10 method (T10=0.8558) while preserving Borda as a mechanism anchor. The upgrade is strong but requires caveats: RRR query-level gain is directional, S7 score blend and Borda are tied, and A4C is not expert validation. The algorithm optimization synthesis confirms that the current feature stack is saturated for Top10 improvement.**

---

## Retracted/Negated Old Conclusions

- **Borda was originally the primary method in S4GEN**. In S6R, Borda is demoted to "strong fixed baseline and mechanism anchor," replaced by the validation-selected score blend.
- **RRR was initially framed as a stronger external validation**. After audit, caveats were added: 36-template query-level test is underpowered, publication-weighted metrics carry bias risk, CI crosses zero for query-level Top10.
- **S7 was initially considered evidence supporting score blend superiority**. The analysis showed score blend and Borda are tied on activity-retained pairs (delta = -0.0001, CI includes zero). S7 supports endpoint relevance, not model superiority.
- **Multiple post-hoc approaches (D4S8 direct ranking, D4S13 KNN prior, D4S17 fragment-context compatibility, D4S25 orthogonal features) were briefly considered as potential Top10 upgrades**. All were rejected in D4S20 stopline or subsequent stress tests. The score blend remains saturated.
- **Rare-replacement boundary specialist (D4S24) was expected to improve Top10** but produced exactly zero delta (0.000000) — no boundary movement at all.
- **D4S25 orthogonal features showed promise on rare fragments** but old-heldout stress showed significant degradation (-0.003801) driven by frequent-subspace damage.

---

## Still-Credible Final Conclusions

1. **Score blend (MLP+HGB) is the best internal blind Top10 method** at T10=0.8558.
2. **Borda is a strong fixed baseline** that exposes DE-HGB complementarity (parameter-free, no validation tuning).
3. **RRR adds bounded medicinal-chemistry-facing support** at the pair level, not query level.
4. **S7 supports retrospective activity-retained context** but does not distinguish methods.
5. **The algorithm optimization stack is saturated** — all post-hoc fusion/reranking/feature-expansion paths failed under old-heldout stress.
6. **Next improvement requires new base representation** (ACCFG, pharmacophore, learned embeddings, 3D/electrostatic), not another post-hoc adapter.
7. **The D4A2 canonical metric freeze is complete** with 72.17 as the canonical HGB number and 60.61 as the baseline.

---

## Impact on Paper

- **S6R supersedes S4GEN** as the manuscript line if adopted, with score blend replacing Borda as the primary method.
- **Borda is preserved** in a supporting role as mechanism anchor — essential for explaining complementarity.
- **RRR and S7 are included only as bounded support**, not as primary validation. Their caveats must remain visible.
- **The algorithm optimization synthesis** should be cited in Discussion to justify stopping further post-hoc search.
- **Methods must describe**: score blend construction, validation selection procedure, blind evaluation protocol.
- **The oracle gap (0.8686 vs 0.8558)** indicates residual headroom that should be mentioned as future work.
- **The candidate-pool failure rate (3.07%)** should be mentioned to clarify that most misses are ranking failures, not vocabulary gaps.

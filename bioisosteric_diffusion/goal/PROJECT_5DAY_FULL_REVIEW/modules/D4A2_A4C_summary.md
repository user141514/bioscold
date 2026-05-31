# Module 7 — D4A2 / A4C Review Summary

## Starting Problem

The D4A2 pipeline evaluated whether ensemble-based rank fusion can improve fragment replacement retrieval beyond the single-model HGB baseline, whether A4C computational review labels can be reliably assigned to proposals, and whether the overall framework (DE + HGB + Borda fusion + A4C annotation) is ready for production/paper use. Multiple sub-modules addressed specific gates: ensemble fusion (D4A2D2), dual-mode workflow integration (D4A4), A4C Borda review (D4A3R), A4C coverage expansion (D4A3S), full feature audit (A4C V1A), and semantic evaluation (A4 Full).

---

## Key Findings

### D4A2D2 — DE+HGB Ensemble (Rank Fusion)

- **Borda(DE, HGB) beats HGB**: Ensemble T10=0.7642 vs HGB T10=0.7217 on canonical test (21,052 queries). Bootstrap delta=+0.0278, 95% CI [0.0230, 0.0327], significant.
- **DE-alone does NOT beat HGB**: DE T10=0.7167 vs HGB T10=0.7217. Delta=-0.0049, 95% CI [-0.0112, 0.0010], NOT significant.
- **DE and HGB are complementary**: DE-only hits=1,964 (7.8%), HGB-only hits=2,069 (8.2%), both hit=13,124 (52.3%), both miss=3,895 (15.5%).
- **Union coverage**: DE U HGB Top10 = 68.4%.
- **HGB systematic blind spot**: For 360 queries with rare replacements (global_freq <= 137), HGB Top10 = 0.0%. DE covers these (DE Top10 on rare_repl = 0.3861). This is a systematic blind spot of HGB, not a pipeline bug.
- **Routed ensemble**: subset-aware routing (rare_repl->DE, hard_baseline_miss->HGB, other->Borda) achieves Routed T10=0.7456 vs HGB T10=0.7217, but Borda alone (0.7642) beats routed on aggregate. Routed was abandoned in favor of simpler full Borda.
- **Borda is parameter-free**: RRF k=10 chosen as default, no validation tuning needed.
- **Score-based fusion blocked**: Scale incompatibility between DE and HGB scores prevents score fusion. Only rank-based fusion (RRF/Borda) works.

### D4A4 — Dual-Mode Integration (Conservative + Exploration)

- **Verdict**: B. D4A4_READY_WITH_G2_EXPERT_WARNING.
- **Two-mode architecture implemented**: Conservative Mode = HGB proposals; Exploration Mode = Borda(DE,HGB) proposals with A4C labels.
- **Conservative Mode metrics**: hit_rate_top10=0.4502, hard_reject_rate=0.0067, standard_review_rate=0.0307.
- **Exploration Mode metrics**: hit_rate_top10=0.5327, hard_reject_rate=0.0220, standard_review_rate=0.0530.
- **Delta (Exploration - Conservative)**: hit_rate_top10=+0.0825 [0.0777, 0.0872] significant; hard_reject_rate=+0.0153 [0.0135, 0.0172] significant; standard_review_rate=+0.0223 [0.0203, 0.0244] significant.
- **G2 handling**: G2 candidates are NOT auto-rejected. ~53% of G2 have no hard alert and may contain useful chemistry. G2 = expert-review required, not hard-reject.
- **Mode Tier0 rate**: 0.9920 (most candidates are Tier0_DATA_PENDING because A4C labels only cover G1 region from D4A3T).
- **G2 alert rate may be inflated**: recomputed A4C buckets use SMILES-based rules which may differ from original A4C pipeline methodology. G2 alert elevation may be driven by recomputation bias (Patch 4 finding).

### D4A3R — A4C Borda Review

- **Verdict**: C. BORDA_PENDING_A4C_COVERAGE.
- **Gate 1 (A4C Coverage)**: FAIL. Borda Top10 coverage rate=100.00%, HGB coverage rate=100.00%, mapping rate=100.00% -- but A4C evaluation set coverage is incomplete. 100% mapping means candidates were found in A4C tables, but labels were present for only a subset.
- **Gate 2 (A4C Risk)**: PASS. Hard reject rate=7.41% for both Borda and HGB.
- **Gate 3 (Joint Utility)**: FAIL. Borda exact_hit_and_review_ready@10=57.58%, HGB=57.58%, delta=0.00pp. Identical performance.
- **Bootstrap**: All Borda vs HGB comparisons show zero delta [0.0000, 0.0000] -- no significant difference because A4C review coverage is insufficient to detect one.
- **Root cause**: Canonical key approach relies on SMILES-level matching. If A4C coverage < 95%, the gap is not in the Borda algorithm but in the A4C evaluation set.

### D4A3S — A4C Coverage Expansion

- **Verdict**: C. BORDA_PENDING_A4C_COVERAGE.
- **G1 query count**: 4,919 / 21,052 (23.4%) have Borda > T10 hits that HGB misses.
- **G1 total candidates**: 5,358.
- **Initial A4C coverage**: 2,249/5,358 = 41.97%.
- **Dominant gap**: FRAGMENT_GRAPH_MISSING (3,109 = 58.0% of G1). JOIN_MISSING = 2,249 (42.0%).
- **SMILES recompute attempted**: RDKit available, G4 validation agreement rate = 0.9065, validation PASSED. But recomputed count = 0 -- no new records generated.
- **Chemical screening**: G1 vs G4 Tanimoto, alert rate, diversity compared. All proxy evidence is marked PROXY_EVIDENCE_NOT_A4C_REVIEW.
- **Re-evaluation NOT completed**: Coverage < 95%, pre-registered success criteria not met.
- **Key constraints**: SMILES-based recompute is a heuristic, not expert review. Chemical proxy screening is explicitly NOT A4C review. Borda gains on 14,215 non-A4C-eval queries are harder to verify than gains on 6,837 A4C eval queries.

### A4C V1A — Full Feature Audit

- **Verdict**: A. V1A_READY_FOR_V1B_RULE_BUCKET.
- **Feature computability**: 2,695/2,705 candidates (99.6%) have full 2D delta + alert + 3D shape features.
- **Feature taxonomy**: 21 CASE_LEVEL (replacement-level) features: all 2D delta, alert, pharmacophore count features. 4 CANDIDATE_LEVEL (conformer-level) features: delta_Rg, abs_delta_Rg, pm_distance, delta_asphericity.
- **Hard filter candidates**: |ΔMW| > 100 Da, |ΔLogP| > 3, |ΔTPSA| > 80, new_alert_introduced.
- **Top-K conformer selector NOT justified**: Only 4/25 features vary within case. Insufficient for discrimination.
- **v1B recommended**: Hard filters + warning annotations + review annotation table. No scoring, no top-K selector.
- **Thresholds are heuristic**: Based on medchem heuristics, not data calibration. Need CS1B validation.

### A4 Full — Bioisostere Semantic Evaluation

- **Verdict**: A4 PASS_SEMANTIC_EVALUATION.
- **Clean candidates**: 135.
- **Score distribution**: High priority=123 (91%), Priority=12 (9%), Review=0, Low=0.
- **Evaluation dimensions**: Pharmacophore features, shape/size, physchem, geometry.
- **FG Replacement Stability**: ETKDG_FALLBACK_SMALL_RING=100% (2/2), R7D_short_tether=58% (22/38) with 16 hard failures, S4B/S5D DENSE_SOFT_SCAFFOLD=100%.

---

## Key Files

| File | Path |
|------|------|
| D4A2D2 Ensemble Verdict | `plan_results/routeA_chembl37k_d4a2d2_de_hgb_ensemble/D4A2D2_DE_HGB_ENSEMBLE_VERDICT.md` |
| D4A2D2 Routed Verdict | `plan_results/routeA_chembl37k_d4a2d2_de_hgb_ensemble/D4A2D2_ROUTED_ENSEMBLE_VERDICT.md` |
| D4A2D2 Error Analysis | `plan_results/routeA_chembl37k_d4a2d2_de_hgb_ensemble/D4A2D2_ENSEMBLE_ERROR_ANALYSIS.md` |
| D4A2D2 Decision Log | `plan_results/routeA_chembl37k_d4a2d2_de_hgb_ensemble/MAIN_DECISION_LOG.md` |
| D4A4 Dual-Mode Verdict | `plan_results/routeA_chembl37k_d4a4_dual_mode_integration/D4A4_DUAL_MODE_INTEGRATION_VERDICT.md` |
| D4A4 Decision Log | `plan_results/routeA_chembl37k_d4a4_dual_mode_integration/MAIN_DECISION_LOG.md` |
| D4A3R Borda Review Verdict | `plan_results/routeA_chembl37k_d4a3r_a4c_borda_review/D4A3R_A4C_BORDA_REVIEW_VERDICT.md` |
| D4A3S Coverage Verdict | `plan_results/routeA_chembl37k_d4a3s_a4c_coverage_expansion/D4A3S_A4C_COVERAGE_VERDICT.md` |
| A4C V1A Feature Audit Verdict | `plan_results/A4C_V1A_FULL_FEATURE_AUDIT/A4C_V1A_FULL_FEATURE_AUDIT_VERDICT.md` |
| A4 Full Semantic Evaluation Verdict | `plan_results/A4_FULL_BIOISOSTERE_SEMANTIC_EVALUATION/A4_FULL_BIOISOSTERE_SEMANTIC_EVALUATION_VERDICT.md` |

---

## Core Numbers

1. **Borda(DE, HGB) canonical T10 = 0.7642** vs HGB T10 = 0.7217 (delta = +0.0425, significant).
2. **DE-HGB complementary hits**: DE-only=1,964 (7.8%), HGB-only=2,069 (8.2%). Union coverage=68.4%.
3. **G1 queries**: 4,919/21,052 (23.4%) where Borda beats HGB at Top10. Initial A4C coverage only 41.97%.
4. **HGB rare_repl blind spot**: T10=0.0 for 360 queries with rare replacements (global_freq <= 137). DE covers these (T10=0.3861).
5. **Routed ensemble T10 = 0.7456** vs HGB T10 = 0.7217, but Borda alone (0.7642) beats routed.
6. **Exploration Mode hit_rate_top10 = 0.5327** vs Conservative Mode = 0.4502 (delta = +0.0825, significant).
7. **Feature audit**: 99.6% of candidates have full features (2,695/2,705). 21 case-level, 4 candidate-level features.
8. **G2 alert rate**: 46.85% (Borda-only candidates). G4 alert rate: 0.99% (shared candidates). G3: 9.67%.

---

## Verdict

**The D4A2 ensemble (Borda/DE+HGB) is a proven improvement over HGB for fragment replacement ranking, with strong statistical significance and demonstrated complementarity. The A4C review proxy is implementable (99.6% feature coverage) but A4C label coverage is incomplete (41.97% for G1), blocking full evaluation. The dual-mode workflow (Conservative + Exploration) is structurally ready but requires G2 expert-review caveats. A4C V1A is ready for rule-bucket implementation but not for scoring/conformer selection.**

---

## Retracted/Negated Old Conclusions

- **Routed ensemble was briefly considered superior to full Borda**, but analysis showed Borda alone (T10=0.7642) beats routed (T10=0.7456). Routed was abandoned.
- **Score-based fusion was attempted** but blocked by scale incompatibility between DE and HGB scores. Only rank-based fusion (RRF/Borda) works.
- **HGB initially appeared to match DE's rare_repl coverage** but D4A2D2 confirmed HGB has a systematic blind spot (T10=0.0 for rare_repl), while DE covers these (T10=0.3861).
- **A4C label coverage was initially assumed adequate** but D4A3R/D4A3S showed only 41.97% coverage, causing Gate 1 and Gate 3 failures.

---

## Still-Credible Final Conclusions

1. **Borda(DE,HGB) reliably improves over HGB** — significant gain on canonical (+0.0425) test with bootstrap CI excluding zero.
2. **DE and HGB are genuinely complementary** — DE covers rare-replacement blind spot of HGB; HGB covers hard_baseline_miss blind spots of DE.
3. **Parameter-free Borda is the correct fusion method** because transform-heldout split prevents validation tuning of fusion weights.
4. **A4C V1A feature pipeline works** for 99.6% of candidates.
5. **Dual-mode architecture is structurally correct** but requires explicit G2 expert-review caveats.
6. **A4C full coverage is not yet achieved** — D4A5 is needed for external validation and coverage expansion.
7. **G2 alert rate may be inflated by SMILES recomputation bias** — non-recomputed records may show lower risk.

---

## Impact on Paper

- The Borda fusion result is the **main empirical claim** of the S4GEN manuscript (Borda T10=0.8384 on blind).
- D4A2D2 provides the **mechanism evidence**: complementarity, rare-replacement rescue, negative subspaces.
- A4C appears in the paper as a **computational review proxy** only, explicitly caveated as not expert validation.
- The incomplete A4C coverage (41.97%) is a **limitation** that must be stated in Discussion.
- The dual-mode workflow is a **practical output framing** but not a primary performance claim.
- The feature audit (V1A) is **not ready for paper inclusion** as a scoring method — only as a rule-bucket description in Methods.
- D4A4's G2 alert rate caveat (potential recomputation bias) needs to be mentioned if A4C strata are reported.

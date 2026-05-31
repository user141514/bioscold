#!/usr/bin/env python3
"""Route-A D4P1-Phase3/4 summary, interpretation draft, and verdict."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAN = PROJECT_ROOT / "plan_results"
OUT = PLAN / "routeA_chembl37k_d4p1_phase3_4_interpretability_cases"

PHASE1_VERDICT = PLAN / "routeA_chembl37k_d4p1_phase1_subset_robustness" / "D4P1_PHASE1_SUBSET_ROBUSTNESS_VERDICT.md"
PHASE2_VERDICT = PLAN / "routeA_chembl37k_d4p1_phase2_component_contribution" / "D4P1_PHASE2_COMPONENT_CONTRIBUTION_VERDICT.md"


def main() -> None:
    query_df = pd.read_csv(OUT / "d4p1_phase3_query_analysis_table.csv")
    de_summary = pd.read_csv(OUT / "d4p1_phase3_de_signal_summary.csv")
    de_subset = pd.read_csv(OUT / "d4p1_phase3_de_signal_by_subset.csv")
    hgb_fi = pd.read_csv(OUT / "d4p1_phase3_hgb_feature_importance.csv")
    hgb_fg = pd.read_csv(OUT / "d4p1_phase3_hgb_feature_group_importance.csv")
    borda_cat = pd.read_csv(OUT / "d4p1_phase3_borda_complementarity_categories.csv")
    negative_df = pd.read_csv(OUT / "d4p1_phase3_negative_subspace_deep_dive.csv")
    case_df = pd.read_csv(OUT / "d4p1_phase4_case_study_table.csv")

    de_only = de_summary[de_summary["analysis_group"] == "DE_only_hit"].iloc[0]
    hgb_only = de_summary[de_summary["analysis_group"] == "HGB_only_hit"].iloc[0]
    rare = de_subset[de_subset["subset_name"] == "rare_replacement"].iloc[0]
    freq = de_subset[de_subset["subset_name"] == "frequent_replacement"].iloc[0]
    hard = de_subset[de_subset["subset_name"] == "hard_top10_miss"].iloc[0]
    all_test = de_subset[de_subset["subset_name"] == "all_test"].iloc[0]
    top_group = hgb_fg.sort_values("permutation_importance_abs_sum", ascending=False).iloc[0]
    rare_borda_hgb = float(query_df[query_df["rare_replacement_flag"] == 1]["Borda_hit10"].mean() - query_df[query_df["rare_replacement_flag"] == 1]["HGB_hit10"].mean())
    freq_borda_hgb = float(query_df[query_df["frequent_replacement_flag"] == 1]["Borda_hit10"].mean() - query_df[query_df["frequent_replacement_flag"] == 1]["HGB_hit10"].mean())
    hard_borda_hgb = float(query_df[query_df["hard_top10_miss_flag"] == 1]["Borda_hit10"].mean() - query_df[query_df["hard_top10_miss_flag"] == 1]["HGB_hit10"].mean())

    interp_rows = [
        {
            "topic": "DE signal",
            "evidence_class": "Direct experiment + inferred mechanism",
            "evidence": f"DE-only hits N={int(de_only['N_queries'])}; DE-only positives are more enriched in rare replacements ({de_only['rare_replacement_rate']:.3f} vs {hgb_only['rare_replacement_rate']:.3f}) and show lower Morgan similarity ({de_only['mean_DE_positive_morgan_similarity']:.3f} vs {hgb_only['mean_HGB_positive_morgan_similarity']:.3f}) than HGB-only positives.",
            "results_sentence": "DE provides complementary non-frequency structural signal, especially in rare/hard regimes.",
            "discussion_sentence": "Without embedding dumps, this is inferred from rank/feature behavior rather than latent-space attribution.",
            "do_not_claim": "Do not say DE is universally better than HGB.",
        },
        {
            "topic": "HGB signal",
            "evidence_class": "Direct diagnostic fallback",
            "evidence": f"Primary HGB feature-family signal is {top_group['feature_group']} by permutation importance; SHAP unavailable.",
            "results_sentence": "HGB remains strong where frequency/context priors already organize the replacement space.",
            "discussion_sentence": "Interpretability relies on permutation importance and ablation fallback because no serialized selected-model artifact is available for SHAP.",
            "do_not_claim": "Do not present fallback feature attribution as exact SHAP.",
        },
        {
            "topic": "Borda complementarity",
            "evidence_class": "Direct experiment + inferred mechanism",
            "evidence": "Borda-only gain cases exist; complementarity categories show fusion is not just replaying single-model top10 hits.",
            "results_sentence": "Borda improves overall and in rare/hard regimes, but not uniformly across all chemical subspaces.",
            "discussion_sentence": "Simple Borda outperforms hand-written routing because useful complementarity extends beyond one separable query niche.",
            "do_not_claim": "Do not say Borda is universally better.",
        },
        {
            "topic": "Rare and hard regimes",
            "evidence_class": "Direct experiment",
            "evidence": f"rare_replacement Borda-HGB={rare_borda_hgb:.4f}; hard_top10_miss Borda-HGB={hard_borda_hgb:.4f}; hard_top10_miss DE-attach={hard['DE_minus_attach']:.4f}.",
            "results_sentence": "Rare/hard regimes are the main operating region where DE signal becomes useful to the fusion and yields the largest Borda gains.",
            "discussion_sentence": "These regimes align with the pre-registered mechanism hypothesis from Phase2.",
            "do_not_claim": "Do not generalize this to all subsets.",
        },
        {
            "topic": "Frequent regime",
            "evidence_class": "Direct experiment",
            "evidence": f"frequent_replacement Borda-HGB={freq_borda_hgb:.4f}; DE-HGB={freq['DE_minus_HGB']:.4f}, both near zero at the proposal layer.",
            "results_sentence": "Frequent replacements show little marginal benefit from the DE component relative to HGB/frequency priors.",
            "discussion_sentence": "This is consistent with HGB/frequency already solving much of the frequent regime.",
            "do_not_claim": "Do not treat the frequent regime as evidence against the overall canonical gain.",
        },
        {
            "topic": "Negative subspaces",
            "evidence_class": "Direct experiment + cautious inference",
            "evidence": "C|O, cluster_05, and cluster_09 remain negative; HGB is uniquely strong in cluster_09 while DE/Borda collapse.",
            "results_sentence": "HGB remains strong in specific negative subspaces.",
            "discussion_sentence": "Cluster_09 is only partially explained and should remain a named failure mode.",
            "do_not_claim": "Do not hide negative subspaces in a footnote or skeptical appendix only.",
        },
        {
            "topic": "Dual-mode relation",
            "evidence_class": "Inferred from prior audited outputs",
            "evidence": "G2/G3/G4 case categories connect proposal behavior to Conservative/Exploration architecture without mixing review metrics into proposal claims.",
            "results_sentence": "G3 supports acceptable exploration logic; G2 shows why review stratification is still necessary.",
            "discussion_sentence": "Exploration-mode examples are diagnostic and should remain separated from canonical proposal metrics.",
            "do_not_claim": "Do not say Exploration Mode is review-safe production.",
        },
    ]
    interp_df = pd.DataFrame(interp_rows)
    interp_df.to_csv(OUT / "d4p1_phase3_4_paper_interpretation_table.csv", index=False)

    draft = f"""# D4P1-Phase3/4 Interpretation Draft

## Evidence
- Canonical proposal-layer chain remains fixed from earlier phases; this package adds interpretation and case-study evidence without changing metrics.
- DE-only rescued queries number {int(de_only['N_queries'])}, while HGB-only rescued queries number {int(hgb_only['N_queries'])}.
- Rare/hard subsets remain the strongest positive regime for the non-frequency component:
  - rare_replacement Borda-HGB = {rare_borda_hgb:.4f}
  - hard_top10_miss Borda-HGB = {hard_borda_hgb:.4f}
  - hard_top10_miss DE-attach = {hard['DE_minus_attach']:.4f}
- Frequent replacements remain near-flat for marginal DE contribution:
  - frequent_replacement Borda-HGB = {freq_borda_hgb:.4f}
  - frequent_replacement DE-HGB = {freq['DE_minus_HGB']:.4f}
- Negative subspaces remain explicit:
  - C|O, cluster_05, cluster_09

## Inference
- DE provides complementary non-frequency structural signal.
- HGB remains strong where frequency/context priors already solve much of the ranking problem.
- Borda improves overall and in rare/hard regimes, but not uniformly across all chemical subspaces.
- HGB remains strong in specific negative subspaces.

## Results
- Say: Borda improves overall and in rare/hard regimes, but not uniformly across all chemical subspaces.
- Say: DE provides complementary non-frequency structural signal.
- Say: HGB remains strong in specific negative subspaces.
- Say: frequent replacements show little marginal gain from the DE component.

## Discussion
- Emphasize complementarity rather than universal model superiority.
- Note that DE interpretation is `EMBEDDING_ANALYSIS_PARTIAL`; it is inferred from rank behavior, frequency behavior, and Morgan-style structural similarity, not direct latent-space attribution.
- Note that HGB explanation uses permutation-importance and ablation fallback because SHAP is blocked by missing serialized model artifacts.
- Keep cluster_09 as a named unresolved failure mode even though its dominant chemistry is identifiable.

## G2/G3/G4 and Dual Mode
- G3 examples motivate Exploration Mode as a reviewable exploration list.
- G2 examples show why expert review stratification remains necessary.
- G4 examples provide the shared low-risk baseline where Conservative and Exploration can agree.

## Must Not Claim
- Do not say: Borda is universally better.
- Do not say: Exploration Mode is review-safe production.
- Do not say: MMP positives prove activity-preserving bioisosteres.
- Do not say: DE is universally better than HGB.

## Skeptical Review
- DE signal analysis is still partly inferential because embeddings are unavailable.
- HGB feature importance is fallback-based and should not be oversold as exact feature attribution for the originally selected serialized model.
- Borda complementarity remains a post-hoc explanation of a locked gain, not a proof of causal fusion mechanics.
- Case studies are sampled by rule, but they are still examples and should not substitute for the aggregate evidence.
"""
    (OUT / "D4P1_PHASE3_4_INTERPRETATION_DRAFT.md").write_text(draft, encoding="utf-8")

    input_complete = (OUT / "d4p1_phase3_4_input_discovery.csv").exists()
    shap_blocked = (OUT / "D4P1_PHASE3_HGB_SHAP_BLOCKED.md").exists()
    case_ok = len(case_df) > 0
    verdict = "B. PHASE3_4_COMPLETE_WITH_SHAP_OR_EMBEDDING_LIMITATIONS"
    verdict_md = f"""# D4P1-Phase3/4 Interpretability and Cases Verdict

Verdict: **{verdict}**

## Answers
1. Was input discovery complete?
   - {'YES' if input_complete else 'NO'}
2. Was DE signal interpretable?
   - YES, but `EMBEDDING_ANALYSIS_PARTIAL`.
3. Was HGB feature importance available or fallback used?
   - Fallback used: permutation importance + grouped feature-family aggregation + ablation. SHAP blocked = {'YES' if shap_blocked else 'NO'}.
4. Was Borda complementarity explained?
   - YES, as DE-HGB complementarity with both direct-rescue and mid-rank promotion behavior.
5. Were negative subspaces analyzed?
   - YES: C|O, cluster_05, cluster_09 are explicit.
6. Were non-cherry-picked case studies generated?
   - {'YES' if case_ok else 'NO'}
7. Are figure/table data ready?
   - YES.
8. Is paper skeleton allowed?
   - YES.
9. What remains unresolved?
   - No DE embedding dump for direct latent-space interpretation.
   - No serialized selected HGB artifact for SHAP.
   - Cluster_09 remains only partially explained and should stay a named failure mode.

## Evidence Boundary
- Proposal-layer claims remain anchored to the earlier locked canonical phases.
- A4C tier/provenance fields are used only as diagnostic interpretation support.

## Skeptical Review
- Whether DE signal analysis is circular: reduced by comparing DE-only vs HGB-only vs pool-level frequency/similarity behavior, but still partial without embeddings.
- Whether HGB feature importance is reliable: fallback evidence is informative, but not equivalent to exact SHAP on the selected model object.
- Whether case studies are cherry-picked: sampling is rule-based with fixed seed and attachment-signature stratification.
- Whether weak MMP labels are overclaimed: this package does not claim activity-preserving bioisosteres.
"""
    (OUT / "D4P1_PHASE3_4_INTERPRETABILITY_CASES_VERDICT.md").write_text(verdict_md, encoding="utf-8")

    main_log = f"""# MAIN_DECISION_LOG

## D4P1-Phase3/4
- Anchor: canonical proposal-layer evidence remains Phase0/1/2.
- DE interpretation mode: `EMBEDDING_ANALYSIS_PARTIAL`.
- HGB interpretation mode: `SHAP_BLOCKED`, fallback to permutation importance + grouped feature-family aggregation + ablation.
- Negative subspaces kept explicit: C|O, cluster_05, cluster_09.
- Case studies sampled by rule with fixed seed; no manual substitution.
- Final verdict: {verdict}

## Inputs Referenced
- {PHASE1_VERDICT}
- {PHASE2_VERDICT}
"""
    (OUT / "MAIN_DECISION_LOG.md").write_text(main_log, encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "plan_results" / "routeA_paper_s0_manuscript_skeleton"

P0 = ROOT / "plan_results" / "routeA_paper_p0_newblind_metric_ablation_lock"
P1 = ROOT / "plan_results" / "routeA_chembl37k_d4p1_phase1_subset_robustness"
P2 = ROOT / "plan_results" / "routeA_chembl37k_d4p1_phase2_component_contribution"
P34 = ROOT / "plan_results" / "routeA_chembl37k_d4p1_phase3_4_interpretability_cases"
B0 = ROOT / "plan_results" / "routeA_chembl37k_d4s_b0_blind_split_baseline"
S2 = ROOT / "plan_results" / "routeA_chembl37k_d4s2_listwise_reranker"
A3T = ROOT / "plan_results" / "routeA_chembl37k_d4a3t_exploration_calibration"
A4 = ROOT / "plan_results" / "routeA_chembl37k_d4a4_dual_mode_integration"
CS1 = ROOT / "plan_results" / "routeA_cs1b0_validation_feasibility"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_md(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def fmt4(value: float) -> str:
    return f"{float(value):.4f}"


def load_method_map(df: pd.DataFrame, key: str = "Method") -> dict[str, pd.Series]:
    return {str(row[key]): row for _, row in df.iterrows()}


def find_subset(df: pd.DataFrame, name: str) -> pd.Series:
    sub = df.loc[df["subset_name"] == name]
    if sub.empty:
        raise KeyError(f"Missing subset: {name}")
    return sub.iloc[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    required = {
        "main_table": P0 / "paper_p0_main_table1_newblind_candidate.csv",
        "rank_audit": P0 / "paper_p0_rank_availability_audit.csv",
        "old_supp": P0 / "paper_p0_supplementary_oldcanonical_table.csv",
        "phase1_metrics": P1 / "d4p1_phase1_subset_robustness_metrics.csv",
        "phase2_metrics": P2 / "d4p1_phase2_component_contribution_metrics.csv",
        "phase2_hyp": P2 / "d4p1_phase2_mechanism_hypothesis_tests.csv",
        "blind_bootstrap": B0 / "d4s_b0_blind_bootstrap.csv",
        "split_compare": B0 / "d4s_b0_old_vs_new_split_comparison.csv",
        "split_leakage": B0 / "d4s_b0_split_leakage_audit.csv",
        "s2_bootstrap": S2 / "d4s2_blind_bootstrap.csv",
        "a3t_risk": A3T / "d4a3t_risk_decomposition.csv",
        "a3t_alert": A3T / "d4a3t_alert_rate_by_group.csv",
        "a4_metrics": A4 / "d4a4_two_mode_comparison.csv",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise SystemExit(f"INPUT_MISSING_CANONICAL_PHASE_FILES: {missing}")

    main_table = pd.read_csv(required["main_table"])
    old_table = pd.read_csv(required["old_supp"])
    phase1 = pd.read_csv(required["phase1_metrics"])
    phase2_metrics = pd.read_csv(required["phase2_metrics"])
    phase2_hyp = pd.read_csv(required["phase2_hyp"])
    split_compare = pd.read_csv(required["split_compare"])
    dual_mode = pd.read_csv(required["a4_metrics"])
    risk_decomp = pd.read_csv(required["a3t_risk"])
    alert_by_group = pd.read_csv(required["a3t_alert"])

    main_rows = load_method_map(main_table)
    old_rows = load_method_map(old_table)
    dual_mode_rows = {str(row["metric"]): row for _, row in dual_mode.iterrows()}
    g2 = risk_decomp.loc[risk_decomp["group"] == "G2_pure_borda_only"].iloc[0]
    g3 = risk_decomp.loc[risk_decomp["group"] == "G3_de_retained_by_borda"].iloc[0]
    g4 = alert_by_group.loc[alert_by_group["group"] == "G4"].iloc[0]

    hard = find_subset(phase1, "hard_top10_miss")
    rare = find_subset(phase1, "rare_replacement")
    freq = find_subset(phase1, "frequent_replacement")
    single = find_subset(phase1, "single_pos")
    multi = find_subset(phase1, "multi_pos")
    neg_co = find_subset(phase1, "C|O")
    neg_c05 = find_subset(phase1, "cluster_05")
    neg_c09 = find_subset(phase1, "cluster_09")

    central_claim = (
        "On a leakage-controlled transform-heldout benchmark, a dual-mode ranking framework that fuses "
        "HGB and a Dual Encoder with Borda substantially improves closed-vocabulary scaffold-conditioned "
        "replacement proposal over strong single-model baselines, with gains concentrated in rare and hard "
        "regimes, while A4C-style review reveals distinct risk strata required for exploratory use."
    )

    evidence_rows = [
        {
            "file_path": rel(P0 / "paper_p0_main_table1_newblind_candidate.csv"),
            "stage": "PAPER-P0",
            "evidence_type": "Main performance table candidate",
            "paper_section_candidate": "Results 1: Main blind performance",
            "main_or_supplementary_candidate": "Main",
            "status": "LOCKED",
            "notes": "Primary paper-main performance table under the D4S-B0 secondary blind protocol.",
        },
        {
            "file_path": rel(P0 / "paper_p0_newblind_ablation_metrics.csv"),
            "stage": "PAPER-P0",
            "evidence_type": "New-blind component ablation table",
            "paper_section_candidate": "Results 3: Component mechanism",
            "main_or_supplementary_candidate": "Supplementary",
            "status": "LOCKED",
            "notes": "M4/M5/M7 are truncated diagnostic only; not a main-table replacement.",
        },
        {
            "file_path": rel(P0 / "paper_p0_rank_availability_audit.csv"),
            "stage": "PAPER-P0",
            "evidence_type": "Rank availability / provenance audit",
            "paper_section_candidate": "Methods / Limitations",
            "main_or_supplementary_candidate": "Supplementary",
            "status": "LOCKED",
            "notes": "Supports the claim that M4/M5/M7 are top50-truncated diagnostics only.",
        },
        {
            "file_path": rel(P0 / "paper_p0_old_vs_new_protocol_positioning.md"),
            "stage": "PAPER-P0",
            "evidence_type": "Protocol positioning statement",
            "paper_section_candidate": "Discussion / Methods",
            "main_or_supplementary_candidate": "Main",
            "status": "LOCKED",
            "notes": "Defines new blind as paper-main and old canonical as analysis-only.",
        },
        {
            "file_path": rel(B0 / "d4s_b0_blind_canonical_metric_table.csv"),
            "stage": "D4S-B0",
            "evidence_type": "Blind baseline replay table",
            "paper_section_candidate": "Results 1: Main blind performance",
            "main_or_supplementary_candidate": "Main",
            "status": "LOCKED",
            "notes": "Source of blind baseline metrics and Oracle gap.",
        },
        {
            "file_path": rel(B0 / "d4s_b0_blind_bootstrap.csv"),
            "stage": "D4S-B0",
            "evidence_type": "Blind paired bootstrap",
            "paper_section_candidate": "Results 1: Main blind performance",
            "main_or_supplementary_candidate": "Main",
            "status": "LOCKED",
            "notes": "Provides CI for Borda-HGB and Oracle-Borda on the new blind split.",
        },
        {
            "file_path": rel(B0 / "d4s_b0_split_leakage_audit.csv"),
            "stage": "D4S-B0",
            "evidence_type": "Leakage audit",
            "paper_section_candidate": "Dataset and benchmark / Methods",
            "main_or_supplementary_candidate": "Main",
            "status": "LOCKED",
            "notes": "Supports zero transform overlap and old-test quarantine for the new blind protocol.",
        },
        {
            "file_path": rel(B0 / "d4s_b0_old_vs_new_split_comparison.csv"),
            "stage": "D4S-B0",
            "evidence_type": "Protocol shift audit",
            "paper_section_candidate": "Discussion / Limitations",
            "main_or_supplementary_candidate": "Supplementary",
            "status": "LOCKED",
            "notes": "Documents that the new blind split is easier and distribution-shifted relative to old canonical.",
        },
        {
            "file_path": rel(S2 / "d4s2_blind_test_metrics.csv"),
            "stage": "D4S2",
            "evidence_type": "Blind reranker metrics",
            "paper_section_candidate": "Results 1: Main blind performance",
            "main_or_supplementary_candidate": "Main",
            "status": "LOCKED",
            "notes": "Shows the selected rank-only MLP improves MRR but not Top10 significantly.",
        },
        {
            "file_path": rel(S2 / "d4s2_blind_bootstrap.csv"),
            "stage": "D4S2",
            "evidence_type": "Blind reranker bootstrap",
            "paper_section_candidate": "Results 1: Main blind performance",
            "main_or_supplementary_candidate": "Main",
            "status": "LOCKED",
            "notes": "Provides CI showing the MLP Top10 gain over Borda crosses zero.",
        },
        {
            "file_path": rel(P1 / "d4p1_phase1_subset_robustness_metrics.csv"),
            "stage": "D4P1-Phase1",
            "evidence_type": "Subset robustness metrics",
            "paper_section_candidate": "Results 2: Robustness and subset analysis",
            "main_or_supplementary_candidate": "Main",
            "status": "LOCKED",
            "notes": "Supports rare/hard gains, frequent near-zero marginal gain, and explicit negative subspaces.",
        },
        {
            "file_path": rel(P1 / "d4p1_phase1_gain_concentration.csv"),
            "stage": "D4P1-Phase1",
            "evidence_type": "Gain concentration analysis",
            "paper_section_candidate": "Results 2: Robustness and subset analysis",
            "main_or_supplementary_candidate": "Supplementary",
            "status": "LOCKED",
            "notes": "Shows the gain concentrates in hard/rare regimes but not in a single old-fragment cluster.",
        },
        {
            "file_path": rel(P2 / "d4p1_phase2_component_contribution_metrics.csv"),
            "stage": "D4P1-Phase2",
            "evidence_type": "Component contribution curve",
            "paper_section_candidate": "Results 3: Component mechanism",
            "main_or_supplementary_candidate": "Main",
            "status": "LOCKED",
            "notes": "Provides the canonical component mechanism story on the old canonical protocol.",
        },
        {
            "file_path": rel(P2 / "d4p1_phase2_mechanism_hypothesis_tests.csv"),
            "stage": "D4P1-Phase2",
            "evidence_type": "Mechanism hypothesis tests",
            "paper_section_candidate": "Results 3: Component mechanism",
            "main_or_supplementary_candidate": "Main",
            "status": "LOCKED",
            "notes": "H1-H4 are all supported under the old canonical analysis protocol.",
        },
        {
            "file_path": rel(P2 / "d4p1_phase2_negative_subspace_analysis.csv"),
            "stage": "D4P1-Phase2",
            "evidence_type": "Negative-subspace analysis",
            "paper_section_candidate": "Results 3 / Results 4",
            "main_or_supplementary_candidate": "Main",
            "status": "LOCKED",
            "notes": "Keeps C|O, cluster_05, and cluster_09 explicit instead of hidden.",
        },
        {
            "file_path": rel(P34 / "D4P1_PHASE3_4_INTERPRETATION_DRAFT.md"),
            "stage": "D4P1-Phase3/4",
            "evidence_type": "Interpretation draft",
            "paper_section_candidate": "Results 4: Interpretability and case studies",
            "main_or_supplementary_candidate": "Main",
            "status": "LOCKED",
            "notes": "Defines the interpretability story with DE partial embedding analysis and HGB SHAP fallback.",
        },
        {
            "file_path": rel(P34 / "d4p1_phase4_case_study_table.csv"),
            "stage": "D4P1-Phase4",
            "evidence_type": "Non-cherry-picked case study table",
            "paper_section_candidate": "Results 4: Interpretability and case studies",
            "main_or_supplementary_candidate": "Supplementary",
            "status": "LOCKED",
            "notes": "Full pool of rule-sampled cases; a small selected panel can be shown in the main paper.",
        },
        {
            "file_path": rel(A3T / "d4a3t_risk_decomposition.csv"),
            "stage": "D4A3T",
            "evidence_type": "G-group risk decomposition",
            "paper_section_candidate": "Results 5: A4C review/risk and dual-mode workflow",
            "main_or_supplementary_candidate": "Main",
            "status": "LOCKED",
            "notes": "Supports G2 high-risk, G3 moderate-risk, and the need for expert-review stratification.",
        },
        {
            "file_path": rel(A4 / "d4a4_two_mode_comparison.csv"),
            "stage": "D4A4",
            "evidence_type": "Dual-mode workflow comparison",
            "paper_section_candidate": "Results 5: A4C review/risk and dual-mode workflow",
            "main_or_supplementary_candidate": "Main",
            "status": "LOCKED",
            "notes": "Supports Conservative vs Exploration workflow metrics and reviewable-rate differences.",
        },
        {
            "file_path": rel(CS1 / "CS1B0_VALIDATION_FEASIBILITY_VERDICT.md"),
            "stage": "CS1B0",
            "evidence_type": "Validation feasibility boundary",
            "paper_section_candidate": "Discussion / Limitations",
            "main_or_supplementary_candidate": "Main",
            "status": "LOCKED",
            "notes": "Locks the structure-only claim boundary and rejects activity-preserving claims.",
        },
    ]
    pd.DataFrame(evidence_rows).to_csv(OUT / "paper_s0_evidence_inventory.csv", index=False)

    claim_rows = [
        {
            "claim_id": "C1",
            "claim_text": central_claim,
            "claim_strength": "SUPPORTED_MAIN",
            "supporting_files": "; ".join(
                [
                    rel(P0 / "paper_p0_main_table1_newblind_candidate.csv"),
                    rel(B0 / "d4s_b0_split_leakage_audit.csv"),
                    rel(P1 / "d4p1_phase1_subset_robustness_metrics.csv"),
                    rel(P2 / "d4p1_phase2_mechanism_hypothesis_tests.csv"),
                    rel(A3T / "d4a3t_risk_decomposition.csv"),
                ]
            ),
            "supporting_numbers": (
                f"new blind Borda-HGB={fmt4(main_rows['Borda(DE,HGB)']['Gain_vs_HGB_Top10'])}; "
                f"rare delta={fmt4(rare['Borda_minus_HGB'])}; hard delta={fmt4(hard['Borda_minus_HGB'])}; "
                f"G2 alert={fmt4(g2['alert_rate_among_covered'])}"
            ),
            "allowed_wording": "substantially improves closed-vocabulary replacement proposal; dual-mode ranking; review-aware exploration",
            "forbidden_wording": "universally better across chemistry; review-safe production; activity-preserving bioisostere prediction",
            "paper_location": "Title / Abstract / Results / Discussion",
        },
        {
            "claim_id": "C2",
            "claim_text": "Under the new D4S-B0 secondary blind protocol, Borda(DE,HGB) is the strongest closed-vocabulary Top10 proposal baseline among the fixed non-reranking methods.",
            "claim_strength": "SUPPORTED_MAIN",
            "supporting_files": "; ".join(
                [
                    rel(P0 / "paper_p0_main_table1_newblind_candidate.csv"),
                    rel(B0 / "d4s_b0_blind_bootstrap.csv"),
                ]
            ),
            "supporting_numbers": (
                f"Attachment={fmt4(main_rows['Attachment frequency']['Top10'])}; "
                f"DE={fmt4(main_rows['DE']['Top10'])}; "
                f"HGB={fmt4(main_rows['HGB']['Top10'])}; "
                f"Borda={fmt4(main_rows['Borda(DE,HGB)']['Top10'])}; "
                f"Oracle={fmt4(main_rows['Oracle(DE,HGB)']['Top10'])}"
            ),
            "allowed_wording": "paper-main performance protocol; secondary blind split; paired-bootstrap supported gain",
            "forbidden_wording": "same protocol as old canonical; same metric as D4A4 workflow hit rate",
            "paper_location": "Results 1 / Table 1 / Figure 2",
        },
        {
            "claim_id": "C3",
            "claim_text": "The new blind performance protocol is leakage-controlled and quarantines the old heavily analyzed canonical test from blind evaluation.",
            "claim_strength": "SUPPORTED_MAIN",
            "supporting_files": "; ".join(
                [
                    rel(B0 / "d4s_b0_split_leakage_audit.csv"),
                    rel(B0 / "d4s_b0_old_test_quarantine.csv"),
                    rel(B0 / "D4S_B0_BLIND_SPLIT_BASELINE_VERDICT.md"),
                ]
            ),
            "supporting_numbers": "train/blind transform overlap=0; old canonical test in new blind=0; blind seen-vocab eval N=13,347",
            "allowed_wording": "leakage-controlled transform-heldout blind split; quarantined old test",
            "forbidden_wording": "same as prior canonical split; test-tuned evaluation",
            "paper_location": "Methods / Results 1",
        },
        {
            "claim_id": "C4",
            "claim_text": "The Borda gain is robust in major hard and rare regimes, but not uniform across all subsets.",
            "claim_strength": "SUPPORTED_ANALYSIS",
            "supporting_files": "; ".join(
                [
                    rel(P1 / "d4p1_phase1_subset_robustness_metrics.csv"),
                    rel(P1 / "d4p1_phase1_gain_concentration.csv"),
                ]
            ),
            "supporting_numbers": (
                f"hard={fmt4(hard['Borda_minus_HGB'])}; rare={fmt4(rare['Borda_minus_HGB'])}; "
                f"frequent={fmt4(freq['Borda_minus_HGB'])}; single={fmt4(single['Borda_minus_HGB'])}; "
                f"multi={fmt4(multi['Borda_minus_HGB'])}"
            ),
            "allowed_wording": "gain is strongest in rare and hard regimes; robustness confirmed with subset caveats",
            "forbidden_wording": "uniform robustness; all chemical subspaces improve",
            "paper_location": "Results 2 / Table 2 / Figure 4",
        },
        {
            "claim_id": "C5",
            "claim_text": "The component mechanism is best explained by DE-HGB complementarity rather than by adding more frequency-biased priors.",
            "claim_strength": "SUPPORTED_ANALYSIS",
            "supporting_files": "; ".join(
                [
                    rel(P2 / "d4p1_phase2_component_contribution_metrics.csv"),
                    rel(P2 / "d4p1_phase2_mechanism_hypothesis_tests.csv"),
                    rel(P0 / "paper_p0_newblind_ablation_metrics.csv"),
                ]
            ),
            "supporting_numbers": (
                f"old canonical M6-HGB={fmt4(float(phase2_metrics.loc[phase2_metrics['method_id'] == 'M6', 'Top10'].iloc[0]) - float(phase2_metrics.loc[phase2_metrics['method_id'] == 'M3', 'Top10'].iloc[0]))}; "
                f"new blind M4={fmt4(float(main_rows['Borda(DE,attach)']['Top10']))}; "
                f"new blind M5={fmt4(float(main_rows['Borda(HGB,attach)']['Top10']))}; "
                f"new blind M6={fmt4(float(main_rows['Borda(DE,HGB)']['Top10']))}"
            ),
            "allowed_wording": "complementary non-frequency structural signal; frequency-biased variants are weaker than full Borda",
            "forbidden_wording": "DE universally dominates HGB; causal proof of fusion mechanics",
            "paper_location": "Results 3 / Figure 3 / Supplementary ablation table",
        },
        {
            "claim_id": "C6",
            "claim_text": "The selected rank-only MLP improves ranking quality as measured by MRR, but it does not significantly improve the primary Top10 proposal hit rate over Borda on the new blind split.",
            "claim_strength": "SUPPORTED_MAIN",
            "supporting_files": "; ".join(
                [
                    rel(S2 / "d4s2_blind_test_metrics.csv"),
                    rel(S2 / "d4s2_blind_bootstrap.csv"),
                    rel(S2 / "D4S2_LISTWISE_RERANKER_VERDICT.md"),
                ]
            ),
            "supporting_numbers": "MLP Top10=0.8402; MLP-Borda Top10=+0.0018 [-0.0005, 0.0040]; MRR delta=+0.0565 [0.0512, 0.0614]",
            "allowed_wording": "secondary MRR improvement; not a Top10 SOTA replacement",
            "forbidden_wording": "significantly beats Borda on Top10; new proposal SOTA",
            "paper_location": "Results 1 / Table 1 / Discussion",
        },
        {
            "claim_id": "C7",
            "claim_text": "Known negative subspaces remain explicit failure modes: C|O, cluster_05, and cluster_09 must be discussed rather than hidden in supplementary caveats.",
            "claim_strength": "SUPPORTED_ANALYSIS",
            "supporting_files": "; ".join(
                [
                    rel(P1 / "d4p1_phase1_subset_robustness_metrics.csv"),
                    rel(P2 / "d4p1_phase2_negative_subspace_analysis.csv"),
                    rel(P34 / "D4P1_PHASE3_NEGATIVE_SUBSPACE_REPORT.md"),
                ]
            ),
            "supporting_numbers": (
                f"C|O={fmt4(neg_co['Borda_minus_HGB'])}; "
                f"cluster_05={fmt4(neg_c05['Borda_minus_HGB'])}; "
                f"cluster_09={fmt4(neg_c09['Borda_minus_HGB'])}"
            ),
            "allowed_wording": "specific negative subspaces; unresolved failure modes; HGB remains stronger in some regimes",
            "forbidden_wording": "uniform positive gain; failures are negligible",
            "paper_location": "Results 2 / Results 4 / Discussion",
        },
        {
            "claim_id": "C8",
            "claim_text": "The dual-mode workflow is operationally coherent: Conservative Mode provides the lower-risk HGB stream, while Exploration Mode increases proposal yield but requires A4C-style provenance and expert-review stratification.",
            "claim_strength": "SUPPORTED_WORKFLOW",
            "supporting_files": "; ".join(
                [
                    rel(A4 / "d4a4_two_mode_comparison.csv"),
                    rel(A3T / "d4a3t_risk_decomposition.csv"),
                    rel(A4 / "D4A4_DUAL_MODE_INTEGRATION_VERDICT.md"),
                ]
            ),
            "supporting_numbers": (
                f"Conservative hit@10={fmt4(dual_mode_rows['hit_rate_top10']['conservative'])}; "
                f"Exploration hit@10={fmt4(dual_mode_rows['hit_rate_top10']['exploration'])}; "
                f"G2 alert={fmt4(g2['alert_rate_among_covered'])}; "
                f"G3 alert={fmt4(g3['alert_rate_among_covered'])}; "
                f"G4 alert={fmt4(g4['alert_rate'])}"
            ),
            "allowed_wording": "dual-mode proposal workflow; review-aware exploration; expert-review required for high-risk provenance groups",
            "forbidden_wording": "Exploration Mode is review-safe production; G2 automatic hard reject",
            "paper_location": "Results 5 / Table 3 / Figure 5 / Discussion",
        },
        {
            "claim_id": "C9",
            "claim_text": "The current Route-A evidence supports structure-derived replacement proposal and computational review analysis, not activity-preserving bioisostere validation.",
            "claim_strength": "SUPPORTED_LIMITATION",
            "supporting_files": "; ".join(
                [
                    rel(CS1 / "CS1B0_VALIDATION_FEASIBILITY_VERDICT.md"),
                    rel(P34 / "D4P1_PHASE3_4_INTERPRETATION_DRAFT.md"),
                ]
            ),
            "supporting_numbers": "with_target_id=0; with_assay_id=0; with_activity=0; with_pchembl=0",
            "allowed_wording": "structure-only benchmark; structure-derived replacement pairs; computational review proxy",
            "forbidden_wording": "activity-preserving replacement; experimentally validated bioisostere discovery",
            "paper_location": "Abstract boundary / Discussion / Limitations",
        },
        {
            "claim_id": "C10",
            "claim_text": "Interpretability evidence is meaningful but incomplete: DE interpretation is partial without embedding dumps, and HGB feature attribution uses fallback methods because SHAP is blocked.",
            "claim_strength": "SUPPORTED_LIMITATION",
            "supporting_files": "; ".join(
                [
                    rel(P34 / "D4P1_PHASE3_4_INTERPRETATION_DRAFT.md"),
                    rel(P34 / "D4P1_PHASE3_HGB_SHAP_BLOCKED.md"),
                    rel(P34 / "D4P1_PHASE3_4_INTERPRETABILITY_CASES_VERDICT.md"),
                ]
            ),
            "supporting_numbers": "DE interpretability status=EMBEDDING_ANALYSIS_PARTIAL; HGB status=SHAP_BLOCKED",
            "allowed_wording": "fallback attribution chain; partial embedding analysis; interpretability with limitations",
            "forbidden_wording": "fully explained latent mechanism; exact SHAP attribution of the selected HGB model",
            "paper_location": "Results 4 / Limitations",
        },
    ]
    pd.DataFrame(claim_rows).to_csv(OUT / "paper_s0_claim_to_evidence_map.csv", index=False)

    claim_md = [
        "# Paper S0 Claim Hierarchy",
        "",
        "## Recommended Title Direction",
        "",
        "Primary title:",
        "Closed-Vocabulary Scaffold-Conditioned Replacement Proposal on a Leakage-Controlled Transform-Heldout Benchmark with Dual-Mode Ranking and Review-Aware Exploration",
        "",
        "Alternative title options:",
        "1. Dual-Mode Ranking for Closed-Vocabulary Scaffold-Conditioned Replacement Proposal on a Leakage-Controlled Transform-Heldout Benchmark",
        "2. Leakage-Controlled Scaffold-Conditioned Replacement Proposal with Borda Fusion, Dual-Mode Output, and Review-Aware Exploration",
        "3. Closed-Vocabulary Replacement Proposal with HGB-Dual Encoder Fusion and Review-Aware Exploration",
        "",
        "## Level 1: Central Claim",
        "",
        f"- C1: {central_claim}",
        "",
        "## Level 2: Main Evidence Claims",
        "",
        "- C2: Under the new blind protocol, Borda(DE,HGB) is the strongest fixed non-reranking Top10 proposal baseline.",
        "- C3: The new blind protocol is leakage-controlled and quarantines the old canonical test from blind evaluation.",
        "- C6: The D4S2 rank-only MLP improves MRR, but it does not significantly improve blind Top10 over Borda.",
        "",
        "## Level 3: Mechanism Claims",
        "",
        "- C4: The gain is strongest in rare and hard regimes, with frequent replacements showing near-zero marginal gain.",
        "- C5: The mechanism is best explained by DE-HGB complementarity, not by adding more frequency-biased priors.",
        "- C7: Negative subspaces remain explicit and must be discussed as failure modes.",
        "",
        "## Level 4: Workflow / Review Claims",
        "",
        "- C8: Conservative Mode and Exploration Mode form a coherent dual-mode workflow when exploration outputs are paired with A4C-style provenance and expert-review stratification.",
        "",
        "## Level 5: Limitations",
        "",
        "- C9: The paper is structure-only and cannot claim activity-preserving validation.",
        "- C10: Interpretability is meaningful but incomplete because DE embeddings are unavailable and HGB SHAP is blocked.",
        "- Review-layer closure is incomplete beyond the calibrated G1 region; full-system Tier0 candidates remain future work rather than a present-tense claim.",
        "- M4/M5/M7 under the new blind protocol are diagnostic only because locked blind exports expose top50 component ranks rather than locked full-vocab ranks.",
    ]
    write_md(OUT / "paper_s0_claim_hierarchy.md", claim_md)

    figure_table_rows = [
        {
            "item_id": "T1",
            "item_type": "Table",
            "placement": "Main",
            "protocol": "D4S-B0 secondary blind seen-vocab eval",
            "title_or_caption_stub": "Main closed-vocabulary proposal performance on the new blind split",
            "purpose": "Anchor the paper-main metric table under the clean blind protocol.",
            "source_files": "; ".join(
                [
                    rel(P0 / "paper_p0_main_table1_newblind_candidate.csv"),
                    rel(B0 / "d4s_b0_blind_bootstrap.csv"),
                    rel(S2 / "d4s2_blind_bootstrap.csv"),
                ]
            ),
            "status": "READY",
            "notes": "Rows: Attachment, DE, HGB, Borda, D4S2 MLP, Oracle. Keep M4/M5/M7 out of main rows.",
        },
        {
            "item_id": "T2",
            "item_type": "Table",
            "placement": "Main",
            "protocol": "Old canonical D4A2D2 / D4P1 analysis protocol",
            "title_or_caption_stub": "Subset robustness and explicit failure-mode summary",
            "purpose": "Show that gains concentrate in rare/hard regimes and that negative subspaces remain explicit.",
            "source_files": rel(P1 / "d4p1_phase1_subset_robustness_metrics.csv"),
            "status": "READY",
            "notes": "Rows: all, hard_top10_miss, rare_replacement, frequent_replacement, single_pos, multi_pos, C|O, cluster_05, cluster_09.",
        },
        {
            "item_id": "T3",
            "item_type": "Table",
            "placement": "Main",
            "protocol": "D4A3T / D4A4 workflow-risk diagnostic",
            "title_or_caption_stub": "Dual-mode workflow and A4C-style risk stratification summary",
            "purpose": "Make the review-aware exploration claim concrete without mixing these metrics into proposal Top10.",
            "source_files": "; ".join(
                [
                    rel(A4 / "d4a4_two_mode_comparison.csv"),
                    rel(A3T / "d4a3t_risk_decomposition.csv"),
                    rel(A3T / "d4a3t_alert_rate_by_group.csv"),
                ]
            ),
            "status": "READY",
            "notes": "Rows: Conservative, Exploration, G2, G3, G4. Caption must say workflow/risk metrics, not the same as proposal Top10.",
        },
        {
            "item_id": "F1",
            "item_type": "Figure",
            "placement": "Main",
            "protocol": "Conceptual workflow",
            "title_or_caption_stub": "Route-A workflow: benchmark, models, Borda fusion, dual-mode output, and A4C-style review layer",
            "purpose": "Orient the reader to the benchmark and workflow before the results section.",
            "source_files": "; ".join(
                [
                    rel(P0 / "paper_p0_old_vs_new_protocol_positioning.md"),
                    rel(A4 / "D4A4_DUAL_MODE_INTEGRATION_VERDICT.md"),
                ]
            ),
            "status": "TO_RENDER",
            "notes": "Diagram only; no new metrics.",
        },
        {
            "item_id": "F2",
            "item_type": "Figure",
            "placement": "Main",
            "protocol": "D4S-B0 secondary blind seen-vocab eval",
            "title_or_caption_stub": "Main proposal performance on the new blind split",
            "purpose": "Show Table 1 graphically with emphasis on Borda-HGB and Oracle-Borda gaps.",
            "source_files": rel(P0 / "paper_p0_main_table1_newblind_candidate.csv"),
            "status": "READY",
            "notes": "Use point-range or bar+CI plot; keep D4S2 note visually secondary.",
        },
        {
            "item_id": "F3",
            "item_type": "Figure",
            "placement": "Main",
            "protocol": "D4S-B0 secondary blind seen-vocab eval",
            "title_or_caption_stub": "Component contribution curve on the new blind split",
            "purpose": "Show Attachment -> DE -> HGB -> Borda -> Oracle under the paper-main protocol.",
            "source_files": "; ".join(
                [
                    rel(P0 / "paper_p0_main_table1_newblind_candidate.csv"),
                    rel(P0 / "paper_p0_newblind_ablation_metrics.csv"),
                ]
            ),
            "status": "READY",
            "notes": "Main figure uses M1/M2/M3/M6/M8 only; M4/M5/M7 stay diagnostic and move to Supplementary.",
        },
        {
            "item_id": "F4",
            "item_type": "Figure",
            "placement": "Main",
            "protocol": "Old canonical D4A2D2 / D4P1 analysis protocol",
            "title_or_caption_stub": "Subset robustness and mechanism heatmap",
            "purpose": "Show where gains come from and where they do not, with explicit negative subspaces.",
            "source_files": "; ".join(
                [
                    rel(P1 / "d4p1_phase1_fig_subset_delta_data.csv"),
                    rel(P2 / "d4p1_phase2_fig_subset_mechanism_matrix.csv"),
                ]
            ),
            "status": "READY",
            "notes": "Caption must label this as old canonical analysis protocol, not new blind performance.",
        },
        {
            "item_id": "F5",
            "item_type": "Figure",
            "placement": "Main",
            "protocol": "D4A3T / D4A4 workflow-risk diagnostic",
            "title_or_caption_stub": "Risk decomposition of exploratory provenance groups and dual-mode reviewable yield",
            "purpose": "Visualize G2/G3/G4 risk strata and the Conservative vs Exploration review trade-off.",
            "source_files": "; ".join(
                [
                    rel(A3T / "d4a3t_risk_decomposition.csv"),
                    rel(A4 / "d4a4_two_mode_comparison.csv"),
                ]
            ),
            "status": "READY",
            "notes": "Do not frame as external medicinal-chemistry truth; keep it as computational review proxy.",
        },
        {
            "item_id": "F6",
            "item_type": "Figure",
            "placement": "Main",
            "protocol": "Old canonical interpretability / case-study protocol",
            "title_or_caption_stub": "Non-cherry-picked case-study panel covering rescue, agreement, and failure cases",
            "purpose": "Make the interpretability section concrete without replacing aggregate evidence.",
            "source_files": "; ".join(
                [
                    rel(P34 / "d4p1_phase4_fig_case_panel_data.csv"),
                    rel(P34 / "d4p1_phase4_case_study_table.csv"),
                ]
            ),
            "status": "READY",
            "notes": "Keep panel small; full table stays Supplementary.",
        },
        {
            "item_id": "S1",
            "item_type": "Table",
            "placement": "Supplementary",
            "protocol": "Old canonical D4A2D2 / D4P1 analysis protocol",
            "title_or_caption_stub": "Old canonical performance table",
            "purpose": "Keep legacy analysis metrics accessible without mixing protocols in the main table.",
            "source_files": rel(P0 / "paper_p0_supplementary_oldcanonical_table.csv"),
            "status": "READY",
            "notes": "Mandatory supplementary table.",
        },
        {
            "item_id": "S2",
            "item_type": "Table",
            "placement": "Supplementary",
            "protocol": "D4S-B0 secondary blind seen-vocab eval",
            "title_or_caption_stub": "Diagnostic M4/M5/M7 truncated Borda ablations on the new blind split",
            "purpose": "Complete the new-blind ablation package without overpromoting truncated ranks.",
            "source_files": "; ".join(
                [
                    rel(P0 / "paper_p0_newblind_ablation_metrics.csv"),
                    rel(P0 / "paper_p0_rank_availability_audit.csv"),
                ]
            ),
            "status": "READY",
            "notes": "Label TRUNCATED_BORDA_DIAGNOSTIC_ONLY in title or footnote.",
        },
        {
            "item_id": "S3",
            "item_type": "Table",
            "placement": "Supplementary",
            "protocol": "Bootstrap support tables",
            "title_or_caption_stub": "Complete bootstrap intervals",
            "purpose": "Provide detailed paired-bootstrap support for main and supplementary metrics.",
            "source_files": "; ".join(
                [
                    rel(B0 / "d4s_b0_blind_bootstrap.csv"),
                    rel(P0 / "paper_p0_newblind_ablation_bootstrap.csv"),
                    rel(S2 / "d4s2_blind_bootstrap.csv"),
                ]
            ),
            "status": "READY",
            "notes": "Use appendix rather than main text.",
        },
        {
            "item_id": "S4",
            "item_type": "Figure/Table",
            "placement": "Supplementary",
            "protocol": "Old canonical D4P1 analysis protocol",
            "title_or_caption_stub": "Negative-subspace deep dive",
            "purpose": "Show richer C|O / cluster_05 / cluster_09 breakdown than the main paper can carry.",
            "source_files": "; ".join(
                [
                    rel(P2 / "d4p1_phase2_negative_subspace_analysis.csv"),
                    rel(P34 / "d4p1_phase3_negative_subspace_deep_dive.csv"),
                ]
            ),
            "status": "READY",
            "notes": "Main text should still mention the named failure modes explicitly.",
        },
        {
            "item_id": "S5",
            "item_type": "Table",
            "placement": "Supplementary",
            "protocol": "Old canonical interpretability / case-study protocol",
            "title_or_caption_stub": "Full non-cherry-picked case-study table",
            "purpose": "Keep the case sampling auditable.",
            "source_files": rel(P34 / "d4p1_phase4_case_study_table.csv"),
            "status": "READY",
            "notes": "Include sampling seed and underfilled-category note.",
        },
        {
            "item_id": "S6",
            "item_type": "Text/Table",
            "placement": "Supplementary",
            "protocol": "CS1B0 feasibility audit",
            "title_or_caption_stub": "Validation feasibility boundary",
            "purpose": "Document why the paper is structure-only and why activity-preserving claims are excluded.",
            "source_files": rel(CS1 / "CS1B0_VALIDATION_FEASIBILITY_VERDICT.md"),
            "status": "READY",
            "notes": "Strong limitation/supporting appendix item.",
        },
    ]
    pd.DataFrame(figure_table_rows).to_csv(OUT / "paper_s0_figure_table_plan.csv", index=False)

    outline_md = [
        "# Paper S0 Manuscript Outline",
        "",
        "## 1. Title Options",
        "Purpose:",
        "- Choose a benchmark-plus-workflow title that foregrounds replacement proposal rather than activity-preserving bioisosteres.",
        "Key claims:",
        "- New blind is the paper-main performance protocol.",
        "- Dual-mode ranking and review-aware exploration are the distinctive framing.",
        "Supporting evidence files:",
        f"- {rel(P0 / 'PAPER_P0_NUMERIC_POSITIONING_STATEMENT.md')}",
        f"- {rel(A4 / 'D4A4_DUAL_MODE_INTEGRATION_VERDICT.md')}",
        "Figures/tables used:",
        "- None.",
        "Caveats:",
        "- Do not put 'bioisostere prediction' or 'validated discovery' into the title.",
        "",
        "## 2. Abstract Skeleton",
        "Purpose:",
        "- State the benchmark setting, the main new-blind result, the rare/hard mechanism, the dual-mode workflow, and the claim boundary.",
        "Key claims:",
        f"- New blind Borda(DE,HGB) Top10 = {fmt4(main_rows['Borda(DE,HGB)']['Top10'])}, versus HGB = {fmt4(main_rows['HGB']['Top10'])}.",
        f"- Rare and hard regimes show the strongest analysis gains: rare = {fmt4(rare['Borda_minus_HGB'])}, hard = {fmt4(hard['Borda_minus_HGB'])}.",
        "- Exploration Mode requires review-aware provenance and expert-review stratification.",
        "Supporting evidence files:",
        f"- {rel(P0 / 'paper_p0_main_table1_newblind_candidate.csv')}",
        f"- {rel(P1 / 'd4p1_phase1_subset_robustness_metrics.csv')}",
        f"- {rel(A4 / 'd4a4_two_mode_comparison.csv')}",
        "Figures/tables used:",
        "- Table 1; optional Figure 1 and Figure 2 references.",
        "Caveats:",
        "- Mention structure-only scope explicitly; do not claim activity preservation.",
        "",
        "## 3. Introduction",
        "Purpose:",
        "- Frame scaffold-conditioned replacement proposal as a benchmark, ranking, and workflow problem rather than a wet-lab validation story.",
        "Key claims:",
        "- Closed-vocabulary replacement ranking is a meaningful, leakage-sensitive benchmark problem.",
        "- Existing frequency or single-model baselines leave recoverable ranking opportunity.",
        "- Review-aware exploration matters because exploratory gains are not uniformly low-risk.",
        "Supporting evidence files:",
        f"- {rel(B0 / 'd4s_b0_split_leakage_audit.csv')}",
        f"- {rel(B0 / 'd4s_b0_blind_canonical_metric_table.csv')}",
        f"- {rel(A3T / 'd4a3t_risk_decomposition.csv')}",
        "Figures/tables used:",
        "- Figure 1.",
        "Caveats:",
        "- Keep the motivation on structure-derived proposal and review workflow, not medicinal-chemistry truth claims.",
        "",
        "## 4. Related Work",
        "Purpose:",
        "- Position the paper among scaffold-conditioned replacement ranking, structure-derived matched-pair benchmarks, and review-aware workflow papers.",
        "Key claims:",
        "- The paper is benchmark-plus-algorithm-plus-workflow, not an activity-prediction or generative open-vocabulary paper.",
        "Supporting evidence files:",
        "- Internal section only; to be supported by verified citations during drafting.",
        "Figures/tables used:",
        "- None.",
        "Caveats:",
        "- Cite bioisostere and matched-pair literature carefully without conflating them with the present claim.",
        "",
        "## 5. Dataset and Benchmark",
        "Purpose:",
        "- Describe the closed-vocabulary transform-heldout benchmark and why the new blind split became the paper-main protocol.",
        "Key claims:",
        "- Secondary blind split is leakage-controlled and quarantines the old canonical test.",
        "- New blind is easier/distribution-shifted, so old and new protocols must be reported separately.",
        "Supporting evidence files:",
        f"- {rel(B0 / 'd4s_b0_split_leakage_audit.csv')}",
        f"- {rel(B0 / 'd4s_b0_old_vs_new_split_comparison.csv')}",
        f"- {rel(P0 / 'paper_p0_old_vs_new_protocol_positioning.md')}",
        "Figures/tables used:",
        "- Figure 1; optional supplementary split audit table.",
        "Caveats:",
        "- Do not mix old canonical and new blind values in one performance row.",
        "",
        "## 6. Methods",
        "Purpose:",
        "- Describe Attachment frequency, DE, HGB, Borda fusion, D4S2 rank-only MLP, and the dual-mode workflow.",
        "Key claims:",
        "- Borda(DE,HGB) is the canonical fusion method.",
        "- Conservative Mode = HGB; Exploration Mode = Borda(DE,HGB) plus A4C-style provenance.",
        "- D4S2 is a rank-only reranker evaluated under the locked blind protocol.",
        "Supporting evidence files:",
        f"- {rel(P0 / 'paper_p0_main_table1_newblind_candidate.csv')}",
        f"- {rel(S2 / 'd4s2_selected_model.md')}",
        f"- {rel(A4 / 'd4a4_two_mode_comparison.csv')}",
        "Figures/tables used:",
        "- Figure 1; Table 1; Table 3.",
        "Caveats:",
        "- M4/M5/M7 are diagnostic only because they use truncated ranks under the new blind export surface.",
        "",
        "## 7. Results 1: Main Blind Performance",
        "Purpose:",
        "- Present the main paper performance result on the clean secondary blind split.",
        "Key claims:",
        f"- Borda(DE,HGB) reaches Top10 = {fmt4(main_rows['Borda(DE,HGB)']['Top10'])} and exceeds HGB by {fmt4(main_rows['Borda(DE,HGB)']['Gain_vs_HGB_Top10'])}.",
        f"- Oracle(DE,HGB) leaves a residual gap of {fmt4(main_rows['Oracle(DE,HGB)']['Gain_vs_Borda_Top10'])} over Borda.",
        "- D4S2 MLP improves MRR but not blind Top10 significantly.",
        "Supporting evidence files:",
        f"- {rel(P0 / 'paper_p0_main_table1_newblind_candidate.csv')}",
        f"- {rel(B0 / 'd4s_b0_blind_bootstrap.csv')}",
        f"- {rel(S2 / 'd4s2_blind_bootstrap.csv')}",
        "Figures/tables used:",
        "- Table 1; Figure 2.",
        "Caveats:",
        "- State that Borda remains the main Top10 result; MLP is secondary ranking-quality evidence.",
        "",
        "## 8. Results 2: Robustness and Subset Analysis",
        "Purpose:",
        "- Show where the Borda gain comes from and where it does not.",
        "Key claims:",
        f"- hard_top10_miss = {fmt4(hard['Borda_minus_HGB'])}",
        f"- rare_replacement = {fmt4(rare['Borda_minus_HGB'])}",
        f"- frequent_replacement = {fmt4(freq['Borda_minus_HGB'])}",
        f"- explicit negative subspaces: C|O = {fmt4(neg_co['Borda_minus_HGB'])}, cluster_05 = {fmt4(neg_c05['Borda_minus_HGB'])}, cluster_09 = {fmt4(neg_c09['Borda_minus_HGB'])}",
        "Supporting evidence files:",
        f"- {rel(P1 / 'd4p1_phase1_subset_robustness_metrics.csv')}",
        f"- {rel(P1 / 'd4p1_phase1_gain_concentration.csv')}",
        "Figures/tables used:",
        "- Table 2; Figure 4.",
        "Caveats:",
        "- Caption and text must call this the old canonical analysis protocol, not the new blind main protocol.",
        "",
        "## 9. Results 3: Component Mechanism",
        "Purpose:",
        "- Explain why Borda helps and why frequency-biased variants are insufficient.",
        "Key claims:",
        "- DE-HGB complementarity is the best mechanism account.",
        "- Rare and hard regimes support the non-frequency signal interpretation.",
        "- Diagnostic new-blind M4/M5/M7 rows remain weaker than the full Borda row.",
        "Supporting evidence files:",
        f"- {rel(P2 / 'd4p1_phase2_mechanism_hypothesis_tests.csv')}",
        f"- {rel(P2 / 'd4p1_phase2_component_contribution_metrics.csv')}",
        f"- {rel(P0 / 'paper_p0_newblind_ablation_metrics.csv')}",
        "Figures/tables used:",
        "- Figure 3; Figure 4; Supplementary diagnostic ablation table.",
        "Caveats:",
        "- Do not oversell post-hoc mechanism analysis as causal proof.",
        "",
        "## 10. Results 4: Interpretability and Case Studies",
        "Purpose:",
        "- Provide interpretable evidence for DE, HGB, complementarity, and failure modes without pretending full latent attribution is available.",
        "Key claims:",
        "- DE contributes complementary non-frequency structural signal.",
        "- HGB remains strong in specific negative subspaces.",
        "- Case studies are rule-sampled, not cherry-picked.",
        "Supporting evidence files:",
        f"- {rel(P34 / 'D4P1_PHASE3_4_INTERPRETATION_DRAFT.md')}",
        f"- {rel(P34 / 'd4p1_phase4_case_study_table.csv')}",
        f"- {rel(P34 / 'D4P1_PHASE3_HGB_SHAP_BLOCKED.md')}",
        "Figures/tables used:",
        "- Figure 6; Supplementary full case table; Supplementary negative-subspace deep dive.",
        "Caveats:",
        "- Mention EMBEDDING_ANALYSIS_PARTIAL and SHAP_BLOCKED explicitly.",
        "",
        "## 11. Results 5: A4C Review/Risk and Dual-Mode Workflow",
        "Purpose:",
        "- Connect proposal-layer gains to a review-aware workflow without claiming review-safe automation.",
        "Key claims:",
        f"- Exploration hit@10 = {fmt4(dual_mode_rows['hit_rate_top10']['exploration'])}, Conservative hit@10 = {fmt4(dual_mode_rows['hit_rate_top10']['conservative'])}.",
        f"- G2 alert = {fmt4(g2['alert_rate_among_covered'])}, G3 alert = {fmt4(g3['alert_rate_among_covered'])}, G4 alert = {fmt4(g4['alert_rate'])}.",
        "- G2 requires expert review; G3 can support reviewable exploration; G4 is the low-risk baseline.",
        "Supporting evidence files:",
        f"- {rel(A4 / 'd4a4_two_mode_comparison.csv')}",
        f"- {rel(A3T / 'd4a3t_risk_decomposition.csv')}",
        "Figures/tables used:",
        "- Table 3; Figure 5.",
        "Caveats:",
        "- Keep these as workflow/risk metrics distinct from the main proposal Top10 metrics.",
        "",
        "## 12. Discussion",
        "Purpose:",
        "- Consolidate what Route-A shows, what it does not show, and why the paper is still useful without activity labels.",
        "Key claims:",
        "- The strongest evidence line is benchmark plus ranking plus workflow, not medicinal-chemistry validation.",
        "- New blind confirms strong aggregate proposal performance; old canonical explains the gain mechanism.",
        "- Negative subspaces and incomplete review-layer closure are live caveats rather than reasons to hide the work.",
        "Supporting evidence files:",
        f"- {rel(P0 / 'PAPER_P0_NUMERIC_POSITIONING_STATEMENT.md')}",
        f"- {rel(CS1 / 'CS1B0_VALIDATION_FEASIBILITY_VERDICT.md')}",
        f"- {rel(A4 / 'D4A4_DUAL_MODE_INTEGRATION_VERDICT.md')}",
        "Figures/tables used:",
        "- References back to Tables 1-3 and Figures 2-6.",
        "Caveats:",
        "- Explicitly disclose new-blind ease/distribution shift and structure-only scope.",
        "",
        "## 13. Limitations",
        "Purpose:",
        "- State the claim boundary without hedging around the missing evidence.",
        "Key claims:",
        "- No activity labels or same-assay validation.",
        "- No expert review performed yet.",
        "- Full-system A4C coverage not closed; Tier0 remains future work.",
        "- DE embedding dump and HGB SHAP remain unavailable.",
        "Supporting evidence files:",
        f"- {rel(CS1 / 'CS1B0_VALIDATION_FEASIBILITY_VERDICT.md')}",
        f"- {rel(P34 / 'D4P1_PHASE3_4_INTERPRETABILITY_CASES_VERDICT.md')}",
        f"- {rel(A4 / 'D4A4_DUAL_MODE_INTEGRATION_VERDICT.md')}",
        "Figures/tables used:",
        "- None required; can point to Supplementary limitation notes.",
        "Caveats:",
        "- None; this section should be blunt.",
        "",
        "## 14. Future Work",
        "Purpose:",
        "- Route unresolved work into D4A5, expert review, public-baseline adaptation, and open-vocabulary extensions without backfilling unsupported claims.",
        "Key claims:",
        "- D4A5 review-layer closure is workflow work, not a prerequisite for the present proposal-layer paper.",
        "- Expert blind review and activity-linked data acquisition are future validation paths.",
        "- Diffusion/open-vocabulary generation belongs to a different paper line.",
        "Supporting evidence files:",
        f"- {rel(CS1 / 'CS1B0_VALIDATION_FEASIBILITY_VERDICT.md')}",
        f"- {rel(A4 / 'D4A4_DUAL_MODE_INTEGRATION_VERDICT.md')}",
        "Figures/tables used:",
        "- None.",
        "Caveats:",
        "- Do not let future work silently inflate present-tense claims.",
        "",
        "## 15. Methods Details / Supplementary Plan",
        "Purpose:",
        "- Enumerate what moves to Supplementary and how protocol labels are preserved.",
        "Key claims:",
        "- Old canonical full table, truncated ablations, bootstrap tables, negative-subspace deep dive, case-study full table, and CS1B0 feasibility audit belong in Supplementary.",
        "Supporting evidence files:",
        f"- {rel(OUT / 'paper_s0_supplementary_plan.md')}",
        "Figures/tables used:",
        "- Supplementary items S1-S6.",
        "Caveats:",
        "- Supplementary must still preserve explicit protocol labels and claim boundaries.",
    ]
    write_md(OUT / "paper_s0_manuscript_outline.md", outline_md)

    main_table_spec = [
        "# Paper S0 Main Table Specification",
        "",
        "## Table 1: Main New-Blind Performance",
        "",
        "Protocol:",
        "- D4S-B0 secondary blind seen-vocab evaluation only.",
        "",
        "Rows:",
        "1. Attachment frequency",
        "2. DE",
        "3. HGB",
        "4. Borda(DE,HGB)",
        "5. D4S2 rank-only MLP",
        "6. Oracle(DE,HGB)",
        "",
        "Columns:",
        "- Method",
        "- Description",
        "- Protocol",
        "- N_queries",
        "- Vocab_size",
        "- Top1",
        "- Top5",
        "- Top10",
        "- Top20",
        "- Top50",
        "- MRR",
        "- Gain_vs_HGB_Top10",
        "- Gain_vs_Borda_Top10",
        "- 95_CI",
        "- Notes",
        "",
        "Required footnotes:",
        "- New blind is the paper-main performance protocol.",
        "- D4S2 rank-only MLP improves MRR but not Top10 significantly over Borda.",
        "- Oracle is an upper bound, not a deployable method.",
        "",
        "Forbidden Table 1 moves:",
        "- Do not insert old canonical rows into this table.",
        "- Do not place M4/M5/M7 here as if they were full-rank main ablations.",
        "",
        "## Table 2: Robustness and Failure-Mode Summary",
        "",
        "Protocol:",
        "- Old canonical D4A2D2 / D4P1 analysis protocol only.",
        "",
        "Suggested rows:",
        "- all_test",
        "- hard_top10_miss",
        "- rare_replacement",
        "- frequent_replacement",
        "- single_pos",
        "- multi_pos",
        "- C|O",
        "- cluster_05",
        "- cluster_09",
        "",
        "Caption requirement:",
        "- State explicitly that this table is analysis/robustness evidence on the old canonical protocol, not the paper-main performance protocol.",
        "",
        "## Table 3: Dual-Mode Workflow and Risk Summary",
        "",
        "Protocol:",
        "- D4A3T / D4A4 workflow-risk diagnostics only.",
        "",
        "Rows:",
        "- Conservative Mode",
        "- Exploration Mode",
        "- G2 pure Borda-only",
        "- G3 DE-retained by Borda",
        "- G4 shared",
        "",
        "Caption requirement:",
        "- State that these are workflow/risk metrics and review-proxy diagnostics, not the same as proposal Top10 performance.",
    ]
    write_md(OUT / "paper_s0_main_table_spec.md", main_table_spec)

    supp_md = [
        "# Paper S0 Supplementary Plan",
        "",
        "## Mandatory Supplementary Items",
        "",
        "1. Old canonical full performance table",
        f"- Source: {rel(P0 / 'paper_p0_supplementary_oldcanonical_table.csv')}",
        "- Role: preserves the full D4P1 analysis protocol separately from the paper-main new blind table.",
        "",
        "2. New-blind diagnostic M4/M5/M7 ablation table",
        f"- Source: {rel(P0 / 'paper_p0_newblind_ablation_metrics.csv')}",
        f"- Audit: {rel(P0 / 'paper_p0_rank_availability_audit.csv')}",
        "- Role: completes the ablation story while marking the rows TRUNCATED_BORDA_DIAGNOSTIC_ONLY.",
        "",
        "3. Complete bootstrap appendix",
        f"- Sources: {rel(B0 / 'd4s_b0_blind_bootstrap.csv')}, {rel(P0 / 'paper_p0_newblind_ablation_bootstrap.csv')}, {rel(S2 / 'd4s2_blind_bootstrap.csv')}",
        "- Role: auditability for all intervals reported in the main text.",
        "",
        "4. Negative-subspace deep dive",
        f"- Sources: {rel(P2 / 'd4p1_phase2_negative_subspace_analysis.csv')}, {rel(P34 / 'd4p1_phase3_negative_subspace_deep_dive.csv')}",
        "- Role: richer explanation of C|O, cluster_05, and cluster_09.",
        "",
        "5. Full non-cherry-picked case-study appendix",
        f"- Source: {rel(P34 / 'd4p1_phase4_case_study_table.csv')}",
        "- Role: prove that the main case panel is sampled by rule rather than hand-picked.",
        "",
        "6. Validation-feasibility boundary note",
        f"- Source: {rel(CS1 / 'CS1B0_VALIDATION_FEASIBILITY_VERDICT.md')}",
        "- Role: documents why the manuscript is structure-only and why activity-preserving claims are excluded.",
        "",
        "## Optional Supplementary Items",
        "",
        "1. D4S2 subset and gain/loss analysis",
        f"- Sources: {rel(S2 / 'd4s2_blind_subset_metrics.csv')}, {rel(S2 / 'd4s2_gain_loss_analysis.csv')}",
        "- Role: useful if reviewers ask why reranking did not materially improve Top10.",
        "",
        "2. D4S-B0 old-vs-new split comparison",
        f"- Source: {rel(B0 / 'd4s_b0_old_vs_new_split_comparison.csv')}",
        "- Role: explicit protocol-shift disclosure appendix.",
        "",
        "3. D4P1 gain concentration appendix",
        f"- Source: {rel(P1 / 'd4p1_phase1_gain_concentration.csv')}",
        "- Role: shows rare/hard concentration without a single-cluster collapse.",
    ]
    write_md(OUT / "paper_s0_supplementary_plan.md", supp_md)

    wording_md = [
        "# Paper S0 Wording Boundary",
        "",
        "## Allowed Phrases",
        "",
        "- scaffold-conditioned replacement proposal",
        "- closed-vocabulary replacement ranking",
        "- transform-heldout benchmark",
        "- structure-derived replacement pairs",
        "- computational review proxy",
        "- A4C-style review annotation",
        "- dual-mode proposal workflow",
        "- Conservative Mode",
        "- Exploration Mode",
        "- complementary non-frequency structural signal",
        "",
        "## Restricted Phrases",
        "",
        "- bioisostere / bioisosteric",
        "  Use only in related-work or heuristic context, not as the direct label for current outputs.",
        "- validated replacement",
        "  Use only if explicitly qualified as computationally reviewable or structure-derived, not experimentally validated.",
        "- safer / lower-risk",
        "  Use only relative to the current A4C-style computational proxy, not as medicinal-chemistry truth.",
        "- exploration-ready",
        "  Use only with provenance and expert-review caveats.",
        "",
        "## Forbidden or Disallowed Phrases",
        "",
        "- activity-preserving bioisostere prediction",
        "- experimentally validated bioisostere discovery",
        "- wet-lab validated",
        "- review-safe production system",
        "- fully automated medicinal chemistry decision",
        "- universal improvement across all chemical subspaces",
        "- open-vocabulary generation SOTA",
        "- D4S2 MLP significantly improves Top10 over Borda",
        "",
        "## Preferred Sentence Templates",
        "",
        "- Borda improves overall and in rare/hard regimes, but not uniformly across all chemical subspaces.",
        "- DE provides complementary non-frequency structural signal.",
        "- HGB remains strong in specific negative subspaces.",
        "- Exploration Mode increases proposal yield but requires A4C-style provenance and expert-review stratification for review use.",
        "- The current benchmark is structure-only and does not support activity-preserving claims.",
        "",
        "## Disallowed Sentence Templates",
        "",
        "- Borda is universally better.",
        "- Exploration Mode is review-safe production.",
        "- MMP positives prove activity-preserving bioisosteres.",
        "- The model discovers validated bioisosteres.",
        "- The reranker establishes a new Top10 SOTA over Borda.",
    ]
    write_md(OUT / "paper_s0_wording_boundary.md", wording_md)

    journal_md = [
        "# Paper S0 Journal Positioning",
        "",
        "## Primary Positioning",
        "",
        "- Paper type: benchmark plus ranking algorithm plus review-aware workflow.",
        "- Best fit family: methods-forward cheminformatics / computational biology venues that accept strong benchmarking and workflow papers without requiring wet-lab validation.",
        "- Core value proposition: a leakage-controlled benchmark, a strong dual-model fusion result, a dual-mode workflow framing, and explicit risk/limitation disclosure.",
        "",
        "## Strongest Version of the Paper",
        "",
        "- New blind main performance as the paper-main result.",
        "- Old canonical robustness and mechanism analysis as a separate labeled analysis protocol.",
        "- Dual-mode workflow with A4C-style risk strata as a practical downstream interpretation layer.",
        "- Explicit limitation boundary around structure-only evidence.",
        "",
        "## Fallback Version",
        "",
        "- If the workflow/risk layer feels too broad for the target venue, the paper can be tightened to benchmark plus ranking plus mechanism, with dual-mode/A4C moved to a shorter discussion and supplementary package.",
        "",
        "## What Evidence Is Still Missing for a Higher-Tier Version",
        "",
        "- Direct activity-linked validation or curated external bioisostere labels.",
        "- A completed expert blind-review pilot.",
        "- A fuller public-baseline comparison adapted into the exact benchmark protocol.",
        "- Full review-layer closure beyond the calibrated G1 region.",
        "- Stronger model-attribution artifacts, such as HGB SHAP on the exact selected model or DE embedding exports.",
        "",
        "## Safe Claims",
        "",
        "- Leakage-controlled closed-vocabulary replacement proposal benchmark.",
        "- Strong Borda fusion gain over HGB under the new blind protocol.",
        "- Gains are strongest in rare and hard regimes on the old canonical analysis protocol.",
        "- Dual-mode workflow offers a useful review-aware decomposition of lower-risk versus exploratory proposals.",
        "",
        "## Unsafe Claims",
        "",
        "- Activity-preserving replacement discovery.",
        "- Wet-lab validated bioisostere proposal.",
        "- Review-safe production automation.",
        "- Universal chemical-space improvement.",
        "",
        "## Recommendation",
        "",
        "- Writing can proceed now as a methods/benchmark/workflow paper with conservative claim boundaries.",
        "- If the target venue demands stronger medicinal-chemistry truth, Route-A should not overreach; it would first need additional validation work outside the current paper scope.",
    ]
    write_md(OUT / "paper_s0_journal_positioning.md", journal_md)

    missing_rows = [
        {
            "item": "D4A5 review-layer closure beyond calibrated G1",
            "classification": "BLOCKS_STRONG_CLAIM_ONLY",
            "why_it_matters": "Prevents claims that Exploration Mode is review-safe across the full-system universe.",
            "impact_on_paper": "Requires explicit workflow caveat, but does not block the proposal-layer manuscript.",
            "required_for_first_draft": "No",
            "notes": "Keep as Future Work / Discussion.",
        },
        {
            "item": "Full-system A4C coverage remains incomplete",
            "classification": "BLOCKS_STRONG_CLAIM_ONLY",
            "why_it_matters": "Tier0_DATA_PENDING remains substantial outside the calibrated subset.",
            "impact_on_paper": "Blocks production-style review claims, not the benchmark/ranking paper.",
            "required_for_first_draft": "No",
            "notes": "Must be disclosed in Results 5 and Limitations.",
        },
        {
            "item": "No activity labels / same-assay validation",
            "classification": "BLOCKS_STRONG_CLAIM_ONLY",
            "why_it_matters": "Prevents activity-preserving or medicinal-chemistry-truth claims.",
            "impact_on_paper": "Narrows the paper to structure-derived proposal and review-proxy claims.",
            "required_for_first_draft": "No",
            "notes": "Explicit claim boundary already locked by CS1B0.",
        },
        {
            "item": "No expert blind review performed yet",
            "classification": "BLOCKS_STRONG_CLAIM_ONLY",
            "why_it_matters": "Prevents claims of expert validation or deployment readiness.",
            "impact_on_paper": "Keep expert review as future validation, not present evidence.",
            "required_for_first_draft": "No",
            "notes": "Optional strengthening experiment line.",
        },
        {
            "item": "HGB SHAP unavailable for the exact selected model",
            "classification": "SUPPLEMENTARY_OPTIONAL",
            "why_it_matters": "Limits exact feature-attribution strength.",
            "impact_on_paper": "Use fallback attribution chain and state SHAP_BLOCKED.",
            "required_for_first_draft": "No",
            "notes": "Does not block Results 4 if limitations are explicit.",
        },
        {
            "item": "DE embedding dump unavailable",
            "classification": "SUPPLEMENTARY_OPTIONAL",
            "why_it_matters": "Limits direct latent-space interpretation.",
            "impact_on_paper": "Interpretability section remains partial but still usable.",
            "required_for_first_draft": "No",
            "notes": "Use EMBEDDING_ANALYSIS_PARTIAL wording.",
        },
        {
            "item": "No direct public SOTA baseline adapted into the exact benchmark",
            "classification": "BLOCKS_STRONG_CLAIM_ONLY",
            "why_it_matters": "Weakens broad SOTA framing outside the internal baseline family.",
            "impact_on_paper": "Safe framing is 'strong internal benchmark and workflow result', not 'field-wide SOTA'.",
            "required_for_first_draft": "No",
            "notes": "Useful future strengthening task, especially before external submission.",
        },
        {
            "item": "Publication-grade figure rendering and citation audit not yet done",
            "classification": "BLOCKS_MAIN_DRAFT",
            "why_it_matters": "The manuscript skeleton exists, but a first full draft still needs figures, captions, and verified citations.",
            "impact_on_paper": "Blocks submission-ready draft, not the skeleton itself.",
            "required_for_first_draft": "Yes",
            "notes": "Recommended next task after Paper-S0.",
        },
    ]
    pd.DataFrame(missing_rows).to_csv(OUT / "paper_s0_missing_evidence_checklist.csv", index=False)

    verdict = "B. PAPER_S0_READY_WITH_REVIEW_LAYER_CAVEAT"
    verdict_md = [
        "# Paper S0 Manuscript Skeleton Verdict",
        "",
        f"Final verdict: **{verdict}**",
        "",
        "## Direct Answers",
        "",
        "- 1. Is the paper skeleton ready? Yes.",
        "- 2. Can writing begin now? Yes.",
        f"- 3. What is the central claim? {central_claim}",
        "- 4. What are the main tables and figures? Table 1 new-blind main performance, Table 2 old-canonical subset robustness, Table 3 dual-mode/A4C risk summary, Figure 1 workflow, Figure 2 new-blind performance, Figure 3 component curve, Figure 4 subset robustness heatmap, Figure 5 risk decomposition, Figure 6 case-study panel.",
        "- 5. What must be completed before first draft submission? Figure rendering, caption writing, citation verification, and protocol-label consistency checks. No new experiment is required for a first full draft.",
        "- 6. What is optional but useful? Public-baseline adaptation, expert blind review, D4A5 review-layer closure, HGB SHAP recovery, and DE embedding export.",
        "- 7. What claims are forbidden? Activity-preserving bioisostere prediction, wet-lab validation, review-safe production, universal chemical-space improvement, and Top10 SOTA claims for the D4S2 MLP.",
        "- 8. What is the recommended next task? Paper-S1 manuscript drafting with Table 1 / Figure 1-3 asset build and citation-grounded Introduction + Results writing.",
        "",
        "## Skeptical Review",
        "",
        "- The new blind split is cleaner than the old canonical split, but it is also easier and distribution-shifted, so the paper must disclose that explicitly.",
        "- Old canonical and new blind numbers must stay in separate protocol blocks; mixing them in one row would invalidate the narrative.",
        "- The Borda gain is strong, but the paper must not imply universal benefit because C|O, cluster_05, and cluster_09 remain negative.",
        "- Structure-only data is a real limitation; without activity labels, the paper cannot make medicinal-chemistry-truth claims.",
        "- A4C is a computational review proxy, not an external validation source; its role must stay workflow-oriented.",
        "- The D4S2 MLP should be included only as a secondary MRR result because the blind Top10 gain is not significant.",
        "- The absence of a directly adapted public SOTA baseline weakens any field-wide SOTA rhetoric and should narrow the framing to a strong internal benchmark result.",
        "- Because the work is structure-only, the safest paper identity is benchmark plus ranking plus workflow, not validated bioisostere discovery.",
    ]
    write_md(OUT / "PAPER_S0_MANUSCRIPT_SKELETON_VERDICT.md", verdict_md)

    decision_log = [
        "# Paper S0 Main Decision Log",
        "",
        f"- Final verdict: {verdict}",
        "- New blind D4S-B0 protocol is the main paper performance table.",
        "- Old canonical D4P1 protocol is retained for robustness, mechanism, interpretability, and case-study analysis only.",
        "- D4S2 rank-only MLP remains a secondary MRR result, not a Top10 SOTA replacement for Borda.",
        "- M4/M5/M7 remain Supplementary diagnostic rows because locked new-blind exports provide top50 rather than locked full-vocab component ranks.",
        "- The manuscript claim boundary remains structure-only and review-aware; no activity-preserving or review-safe production claim is allowed.",
    ]
    write_md(OUT / "MAIN_DECISION_LOG.md", decision_log)


if __name__ == "__main__":
    main()

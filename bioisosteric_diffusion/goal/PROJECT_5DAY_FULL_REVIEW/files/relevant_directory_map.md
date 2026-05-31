# Relevant Directory Map
Generated: 2026-05-29 23:08:37
Review Period: Last 5 days

## Summary
- Total directories with recent changes: 102
- Total relevant directories: 88

---

## A_improve Module Directories
### /goal/A_improve/ (root)
Purpose: Top-level orchestration for D4S27-D4S32 scoring pipeline
- Key files: README.md, TASK.md, report.md, d4s27_conditional_prior.py, evaluate_and_fuse.py, build_candidate_scores.py, summary_metrics.csv
### D4S27_signal_exploration
Purpose: Conditional prior signal exploration (D4S27)
- Status: No recent changes in this 5-day window
### D4S28_candidate_transform_scorer
Purpose: D4S28 candidate transform scoring, blind evaluation, subspace audit, final lockdown
- Key scripts: d4s28_scorer.py, d4s28_audit.py, d4s28_blind.py, d4s28_blind_fixed.py, d4s28r_final_lockdown.py, d4s28r_lockdown_v2.py, d4s28r_subspace_audit.py
- Key docs: README.md, README_AUDIT.md
- Results: audit_decision.json, d4s28_summary_metrics.csv, d4s28r_final_verdict.json, d4s28r_subspace_audit.csv
### D4S28R_FINAL_LOCK
Purpose: Locked audit artifacts for D4S28 revisit
- Key docs: 01-07 analysis md files, FINAL_VERDICT.md, audit_decision.json
- Key results: d4s28_summary_metrics.csv, d4s28r_final_verdict.json, feature_provenance.csv
### D4S29_new_signal_exploration
Purpose: New signal exploration (geometry phase 1 + chemistry phase 3)
- Key scripts: d4s29_phase1_geometry.py, d4s29_phase3_chemistry.py
- Key docs: D4S29_EXPLORATION_SUMMARY.md
- Large datasets: d4s29_geometry_sample.csv (6.9MB), d4s29_chemistry_sample.csv (2.2MB)
### D4S30_shape_augmented
Purpose: Shape-augmented scoring (D4S30 audit)
- Key scripts: d4s30_shape_scorer.py, d4s30_audit.py
- Key docs: D4S30_AUDIT_VERDICT.md
- Results: d4s30_summary.csv, d4s30_blind_strata_audit.csv
### D4S31_feature_pruning
Purpose: Feature ablation and lockdown (D4S31)
- Key scripts: d4s31_ablation.py, d4s31_lockdown.py
- Results: d4s31_ablation.csv, d4s31_blind_strata.csv, d4s31_final_metrics.csv
### D4S31_FINAL_LOCK
Purpose: Locked D4S31 final artifacts
- Key docs: ablation_findings.md, final_metrics.md, feature_diff.md, blind_strata.md
- Key results: d4s31_final_metrics.csv, d4s31_blind_strata.csv
### D4S32_query_router
Purpose: Query router (D4S32) - latest module
- Key scripts: d4s32_router.py
- Logs: d4s32_router.log
### results/ (under A_improve)
Purpose: D4S27 evaluation results
- Large results: candidate_prior_scores_blind.csv.gz (408MB), candidate_prior_scores_val.csv.gz (410MB)
- Key outputs: summary_metrics.csv, stratified_metrics.csv, d4s27_query_all.csv, gate_evaluation.csv

---

## Paper Writing Directories (plan_results/routeA_paper*)
### routeA_paper_p0_newblind_metric_ablation_lock
Purpose: Phase 0: New blind metric ablation and numerical positioning
- Recent files: 12
- Key files: MAIN_DECISION_LOG.md, PAPER_P0_NEWBLIND_METRIC_ABLATION_LOCK_VERDICT.md
### routeA_paper_skill0_research_writing_skill_adaptation
Purpose: Skill 0: Research writing skill adaptation, central story, reviewer attack prep
- Recent files: 11
- Key files: MAIN_DECISION_LOG.md, PAPER_SKILL0_RESEARCH_WRITING_SKILL_ADAPTATION_VERDICT.md
### routeA_paper_s0_manuscript_skeleton
Purpose: Stage 0: Manuscript skeleton, claim hierarchy, evidence inventory
- Recent files: 12
- Key files: MAIN_DECISION_LOG.md, paper_s0_manuscript_outline.md, PAPER_S0_MANUSCRIPT_SKELETON_VERDICT.md
### routeA_paper_s1_methods_results_draft
Purpose: Stage 1: Methods and results draft with figure/table specs
- Recent files: 18
- Key files: MAIN_DECISION_LOG.md, PAPER_S1_METHODS_RESULTS_DRAFT_VERDICT.md
### routeA_paper_s2_full_draft
Purpose: Stage 2: Full manuscript draft with intro, related work, discussion
- Recent files: 7
- Key files: MAIN_DECISION_LOG.md, PAPER_S2_FULL_DRAFT_VERDICT.md, paper_s2_full_manuscript.md
### routeA_paper_s2r_full_manuscript_merge
Purpose: Stage 2R: Full manuscript merge, redundancy check, number consistency
- Recent files: 10
- Key files: paper_s2r_full_manuscript_merged.md, PAPER_S2R_FULL_MANUSCRIPT_VERDICT.md, paper_s2r_manuscript_cleanliness_check.md
### routeA_paper_s3_citations_figures
Purpose: Stage 3: Citations and figures integration (initial pass)
- Recent files: 25
- Key files: paper_s3_full_manuscript_cited.md, paper_s3_full_manuscript_cited_clean.md
### routeA_paper_s3fig_figure_table_integration
Purpose: Stage 3FIG: Figure and table integration, visual assets
- Recent files: 29
- Key files: MAIN_DECISION_LOG.md, PAPER_S3FIG_FIGURE_TABLE_INTEGRATION_VERDICT.md, paper_s3fig_manuscript_with_figure_calls.md
### routeA_paper_s3plot_deterministic_figures
Purpose: Stage 3PLOT: Deterministic R-script generated figures
- Recent files: 30
- Key files: MAIN_DECISION_LOG.md, PAPER_S3FIG_FIGURE_TABLE_INTEGRATION_VERDICT.md, paper_s3fig_manuscript_with_figure_calls.md
### routeA_paper_s3cite_citation_resolution
Purpose: Stage 3CITE: Citation resolution, reference list, related work revision
- Recent files: 12
- Key files: MAIN_DECISION_LOG.md, PAPER_S3CITE_CITATION_RESOLUTION_VERDICT.md, paper_s3cite_full_manuscript_with_citations.md
### routeA_paper_s3text_journal_facing
Purpose: Stage 3TEXT: Journal-facing text rewrite, claim safety report
- Recent files: 5
- Key files: paper_s3text_full_manuscript_journal_facing.md, PAPER_S3TEXT_JOURNAL_FACING_VERDICT.md
### routeA_paper_s4_polish
Purpose: Stage 4: Nature journal polish, NMI style manuscript
- Recent files: 4
- Key files: paper_s4_nature_nmi_polished_manuscript.md, PAPER_S4_POLISH_VERDICT.md
### routeA_paper_s4gen_general_journal_polish
Purpose: Stage 4GEN: General journal polish, claim safety, section flow, reviewer risk memo
- Recent files: 13
- Key files: MAIN_DECISION_LOG.md, PAPER_S4GEN_GENERAL_JOURNAL_POLISH_VERDICT.md, paper_s4gen_polished_manuscript.md, paper_s4gen_polished_manuscript_zh.md, paper_s6r_text_journal_facing_main_manuscript.md
### routeA_paper_s5_topjournal_upgrade
Purpose: Stage 5: Top journal upgrade manuscript
- Recent files: 2
- Key files: paper_s5_topjournal_upgrade_manuscript.md, S5_TOPJOURNAL_UPGRADE_VERDICT.md
### routeA_paper_s6_curated_recall_upgrade
Purpose: Stage 6: Curated recall upgrade with verdict
- Recent files: 2
- Key files: paper_s6_curated_recall_manuscript.md, S6_CURATED_RECALL_UPGRADE_VERDICT.md
### routeA_paper_s6r_claim_audit
Purpose: Stage 6R: Claim hierarchy audit and score blend audit
- Recent files: 3
- Key files: --
### routeA_paper_s6r_merge_claim_denoised_upgrade
Purpose: Stage 6R Merge: Merged claim denoised upgrade with evidence hierarchy
- Recent files: 15
- Key files: MAIN_DECISION_LOG.md, PAPER_S6R_MERGE_ADOPTION_VERDICT.md, paper_s6r_merge_manuscript_candidate.md
### routeA_paper_s6r_text_journal_facing
Purpose: Stage 6R Text: Journal-facing main manuscript, number consistency check
- Recent files: 7
- Key files: MAIN_DECISION_LOG.md, paper_s6r_text_journal_facing_main_manuscript.md, PAPER_S6R_TEXT_VERDICT.md

---

## D4S Algorithm Directories (plan_results/routeA_chembl37k_d4s*)
### routeA_chembl37k_d4s_b0_blind_split_baseline
Purpose: B0: Blind split baseline
- Recent files: 24
- Key verdict/manifesto: D4S_B0_BLIND_SPLIT_BASELINE_VERDICT.md
### routeA_chembl37k_d4s0_sota_opportunity
Purpose: S0: SOTA opportunity analysis
- Recent files: 13
- Key verdict/manifesto: D4S0_SOTA_OPPORTUNITY_VERDICT.md
### routeA_chembl37k_d4s2_listwise_reranker
Purpose: S2: Listwise reranker
- Recent files: 21
- Key verdict/manifesto: D4S2_LISTWISE_RERANKER_VERDICT.md
### routeA_chembl37k_d4s3_feature_rich_reranker_no_skill
Purpose: S3: Feature-rich reranker
- Recent files: 31
- Key verdict/manifesto: D4S3_FEATURE_RICH_RERANKER_VERDICT.md
### routeA_chembl37k_d4s4_query_aware_boundary_moe
Purpose: S4: Query-aware boundary MoE
- Recent files: 16
- Key verdict/manifesto: D4S4_QUERY_AWARE_BOUNDARY_MOE_VERDICT.md
### routeA_chembl37k_d4s5_algorithm_exploration
Purpose: S5: Algorithm exploration
- Recent files: 19
- Key verdict/manifesto: D4S5_ALGORITHM_EXPLORATION_VERDICT.md
### routeA_chembl37k_d4s5r_stability_guard
Purpose: S5R: Stability guard extension
- Recent files: 5
- Key verdict/manifesto: D4S5R_STABILITY_GUARD_VERDICT.md
### routeA_chembl37k_d4s6_algorithm_rescue_search
Purpose: S6: Algorithm rescue search
- Recent files: 10
- Key verdict/manifesto: D4S6_PHASE3_ALGORITHM_RESCUE_VERDICT.md
### routeA_chembl37k_d4s6_boundary_expert
Purpose: S6: Boundary expert
- Recent files: 8
- Key verdict/manifesto: D4S6_BOUNDARY_EXPERT_VERDICT.md
### routeA_chembl37k_d4s6_candidate_matrix_recovery
Purpose: S6: Candidate matrix recovery (multi-phase)
- Recent files: 61
- Key verdict/manifesto: D4S6_MATRIX_RECOVERY_PHASE1_VERDICT.md, D4S6_PHASE1B_MLP_HGB_RECONSTRUCTION_VERDICT.md, D4S6_PHASE1C_SAME_NAMESPACE_SOURCE_VERDICT.md, D4S6_PHASE2B_REPAIR_VERDICT.md, D4S6_PHASE2C_VALIDATION_MATRIX_VERDICT.md, D4S6_PHASE2_BLEND_REPRODUCTION_VERDICT.md, _d4s6_phase2c_fix_verdict.py, _d4s6_phase2c_fix_verdict2.py
### routeA_chembl37k_d4s6_clean_eval_foundation
Purpose: S6: Clean eval foundation
- Recent files: 7
- Key verdict/manifesto: d4s6_execution_verdict.md
### routeA_chembl37k_d4s6_de_representation_audit
Purpose: S6: DE representation audit
- Recent files: 7
- Key verdict/manifesto: D4S6_DE_REPRESENTATION_VERDICT.md
### routeA_chembl37k_d4s6_evidence_repair
Purpose: S6: Evidence repair
- Recent files: 11
- Key verdict/manifesto: d4s6_repair_clean_eval_verdict.md
### routeA_chembl37k_d4s6_hgb_rescue_predeclared
Purpose: S6: HGB rescue predeclared
- Recent files: 7
- Key verdict/manifesto: D4S6_HGB_RESCUE_VERDICT.md
### routeA_chembl37k_d4s6_multimodel_ensemble
Purpose: S6: Multimodel ensemble
- Recent files: 27
- Key verdict/manifesto: D4S6_PHASE5B_MULTIMODEL_REPAIR_VERDICT.md, D4S6_PHASE5C_EVAL_SUBSET_VERDICT.md, D4S6_PHASE5_MULTIMODEL_ENSEMBLE_VERDICT.md
### routeA_chembl37k_d4s6_robust_guard_selection
Purpose: S6: Robust guard selection
- Recent files: 9
- Key verdict/manifesto: D4S6_PHASE4_ROBUST_GUARD_VERDICT.md
### routeA_chembl37k_d4s6_swarm_dispatch
Purpose: S6: Swarm dispatch
- Recent files: 4
- Key verdict/manifesto: --
### routeA_chembl37k_d4s7_boundary_listwise_expert
Purpose: S7: Boundary listwise expert
- Recent files: 20
- Key verdict/manifesto: D4S7_PHASE6_BOUNDARY_LISTWISE_VERDICT.md
### routeA_chembl37k_d4s8_direct_ranking_objective
Purpose: S8: Direct ranking objective
- Recent files: 37
- Key verdict/manifesto: D4S8B_RANKER_BLEND_AUDIT_VERDICT.md, D4S8_DIRECT_RANKING_OBJECTIVE_VERDICT.md
### routeA_chembl37k_d4s9_rich_feature_alignment
Purpose: S9: Rich feature alignment
- Recent files: 9
- Key verdict/manifesto: D4S9_PHASE1_RICH_FEATURE_ALIGNMENT_VERDICT.md
### routeA_chembl37k_d4s9_tier1_smiles_features
Purpose: S9: Tier1 SMILES features
- Recent files: 52
- Key verdict/manifesto: D4S9_PHASE2_TIER1_SMILES_FEATURE_VERDICT.md, D4S9_PHASE2_V3_BASELINE_GATE_VERDICT.md, D4S9_PHASE2_V3_VERDICT.md
### routeA_chembl37k_d4s10_nearmiss_boundary_audit
Purpose: S10: Near-miss boundary audit
- Recent files: 15
- Key verdict/manifesto: D4S10_NEARMISS_BOUNDARY_AUDIT_VERDICT.md
### routeA_chembl37k_d4s11_signal_supported_override
Purpose: S11: Signal-supported override
- Recent files: 16
- Key verdict/manifesto: D4S11_SIGNAL_SUPPORTED_OVERRIDE_VERDICT.md
### routeA_chembl37k_d4s12_context_prior_adapter
Purpose: S12: Context prior adapter
- Recent files: 35
- Key verdict/manifesto: D4S12_CONTEXT_PRIOR_ADAPTER_VERDICT.md
### routeA_chembl37k_d4s13_old_similarity_prior
Purpose: S13: Old similarity prior
- Recent files: 21
- Key verdict/manifesto: D4S13R_PRECISION_GATE_VERDICT.md, D4S13_OLD_SIMILARITY_PRIOR_VERDICT.md
### routeA_chembl37k_d4s14_knn_trust_router
Purpose: S14: KNN trust router
- Recent files: 24
- Key verdict/manifesto: D4S14R_TRANSPARENT_GATE_VERDICT.md, D4S14_KNN_TRUST_ROUTER_VERDICT.md
### routeA_chembl37k_d4s15_locked_gate_clean_eval
Purpose: S15: Locked gate clean eval
- Recent files: 18
- Key verdict/manifesto: D4S15_LOCKED_GATE_STRESS_TEST_VERDICT.md
### routeA_chembl37k_d4s16_loss_aware_knn_guard
Purpose: S16: Loss-aware KNN guard
- Recent files: 9
- Key verdict/manifesto: D4S16_LOSS_AWARE_KNN_GUARD_VERDICT.md
### routeA_chembl37k_d4s17_fragment_context_compatibility
Purpose: S17: Fragment context compatibility
- Recent files: 12
- Key verdict/manifesto: D4S17_FRAGMENT_CONTEXT_COMPATIBILITY_VERDICT.md
### routeA_chembl37k_d4s18_top10_preserving_early_rank
Purpose: S18: Top-10 preserving early rank
- Recent files: 7
- Key verdict/manifesto: D4S18_TOP10_PRESERVING_EARLY_RANK_VERDICT.md
### routeA_chembl37k_d4s19_top10_preserving_stability_audit
Purpose: S19: Top-10 preserving stability audit
- Recent files: 8
- Key verdict/manifesto: D4S19_TOP10_PRESERVING_STABILITY_VERDICT.md
### routeA_chembl37k_d4s20_algorithm_stopline_and_next_steps
Purpose: S20: Algorithm stopline and next steps
- Recent files: 5
- Key verdict/manifesto: D4S20_ALGORITHM_STOPLINE_VERDICT.md
### routeA_chembl37k_d4s21_candidate_pool_and_label_headroom
Purpose: S21: Candidate pool and label headroom
- Recent files: 9
- Key verdict/manifesto: D4S21_CANDIDATE_POOL_HEADROOM_VERDICT.md
### routeA_chembl37k_d4s22_scoring_failure_anatomy
Purpose: S22: Scoring failure anatomy
- Recent files: 9
- Key verdict/manifesto: D4S22_SCORING_FAILURE_ANATOMY_VERDICT.md
### routeA_chembl37k_d4s23_rare_boundary_debias
Purpose: S23: Rare boundary debias
- Recent files: 8
- Key verdict/manifesto: D4S23_RARE_BOUNDARY_DEBIAS_VERDICT.md
### routeA_chembl37k_d4s24_rare_weighted_boundary_specialist
Purpose: S24: Rare weighted boundary specialist
- Recent files: 16
- Key verdict/manifesto: D4S24_RARE_WEIGHTED_BOUNDARY_SPECIALIST_VERDICT.md
### routeA_chembl37k_d4s25_orthogonal_feature_repair
Purpose: S25: Orthogonal feature repair
- Recent files: 21
- Key verdict/manifesto: D4S25_ORTHOGONAL_FEATURE_REPAIR_VERDICT.md
### routeA_chembl37k_d4s26_loss_aware_frequency_guard
Purpose: S26: Loss-aware frequency guard
- Recent files: 13
- Key verdict/manifesto: D4S26_LOSS_AWARE_FREQUENCY_GUARD_VERDICT.md

---

## Task Directory

### /task/
Purpose: Project-level task orchestration
- Key file: task.md (7067 bytes, modified May 29 23:04)


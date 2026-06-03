TASK: MANUSCRIPT_V6_FINAL_ALIGNMENT_PASS

You are a Claude CLI worker with Ruflo MCP tools available. Use Ruflo/MCP review or orchestration tools if useful, but keep the work bounded and do not add experiments.

Workspace:
E:\zuhui\bioisosteric_diffusion

Primary input:
goal/PROJECT_5DAY_FULL_REVIEW/jcim_submission_candidate/main_manuscript.md

Existing context files:
- goal/PROJECT_5DAY_FULL_REVIEW/jcim_submission_candidate/CLAIM_MAP_V5.md
- goal/PROJECT_5DAY_FULL_REVIEW/jcim_submission_candidate/V5_SURGICAL_REVISION_LOG.md
- goal/PROJECT_5DAY_FULL_REVIEW/jcim_submission_candidate/V5_METHODS_RESULTS_ALIGNMENT_LOG.md

Core thesis:
Fresh Blind2 prospectively supports candidate-level HistGB scoring: HistGB82 is the primary Top-10 configuration. The earlier no-prior-rank 77-feature gain does not replicate as a Top-10 improvement and should be treated as an instability diagnostic, not as a validated model-selection improvement.

Hard constraints:
- Do not add new experiments.
- Do not open a new algorithm track.
- Do not re-promote 77-feature.
- Do not make LambdaRankCat the main method.
- Do not expand A4C.
- Do not invent new evidence.

Required V6 deliverables, all under:
goal/PROJECT_5DAY_FULL_REVIEW/jcim_submission_candidate/

1. manuscript_v6_aligned.md
   A clean submission-facing V6 manuscript derived from main_manuscript.md.

2. abstract_v6.md
   Abstract only.

3. methods_v6_patch.md
   Methods patch / replacement snippets documenting what changed.

4. results_v6_patch.md
   Results patch / replacement snippets documenting what changed.

5. discussion_conclusion_v6.md
   Discussion and Conclusion replacement text.

6. table_order_v6.md
   Table order and captions:
   - Table M1: Fresh Blind2 primary split and candidate matrix.
   - Table M2: Original secondary-blind diagnostic matrix.
   - Table M3: Leakage / overlap verification.
   Include note that old secondary blind is diagnostic history / original benchmark, not primary prospective evaluation.

7. data_availability_v6.md
   Updated Data and Software Availability wording.

8. final_alignment_checklist.md
   Yes/No checklist with 14 items:
   1. Is Fresh Blind2 clearly the primary prospective evaluation?
   2. Is HistGB82 clearly the main prospective Top-10 HistGB result?
   3. Is HistGB77 clearly diagnostic, not main method?
   4. Is old secondary blind demoted to diagnostic history?
   5. Is prior_ranks deletion described as non-replicated for Top-10?
   6. Is C_i defined consistently with the full-vocabulary policy?
   7. Is Fresh Blind2 candidate matrix table placed before old secondary-blind table?
   8. Does leakage table quantify residual relatedness beyond query-side key overlap?
   9. Is A4C supplementary-only?
   10. Is LambdaRankCat future direction only?
   11. Are activity/calibration diagnostics explicitly non-validating?
   12. Does Data Availability point to V5/Fresh Blind2 evidence?
   13. Are all internal project labels removed from main text?
   14. Does the conclusion end with benchmark-audit framing?

9. final_blockers.md only if any checklist answer is No.

Required edits / alignment:

1. Recenter paper around Fresh Blind2
- Abstract, Introduction, Results, Discussion, Conclusion must state:
  Fresh Blind2 is the primary prospective evaluation.
  HistGB82 is the main prospective Top-10 HistGB result.
  HistGB77 is a no-prior-rank diagnostic comparator.
  Old secondary-blind 77-feature result is diagnostic history only.
  Prior-rank deletion does not prospectively replicate as a Top-10 improvement.
- Remove wording implying 77-feature is main method, prior_ranks deletion improves transfer generally, old secondary blind remains primary benchmark, or old 0.9243 is headline.

2. Promote Fresh Blind2 candidate/split table
- Table M1 must be Fresh Blind2 primary split and candidate matrix.
- Table M2 must be original secondary-blind diagnostic matrix.
- Table M3 must be leakage / overlap verification.
- Fresh Blind2 table includes Train2 / Dev2 / Blind2, queries, candidates/query, candidate rows, positive-set size sum, query-side transform keys, old fragments, attachment signatures, zero in-matrix-positive queries, vocabulary size, and random expected Top-10 for Blind2 = 0.0987.

3. Expand leakage-overlap table
- Add columns for query-side transform-key overlap, full old-to-replacement key overlap, old-fragment overlap, attachment-signature overlap.
- For Fresh Blind2 use available local evidence:
  Train2/Dev2: query-side overlap 0, old-fragment overlap 53, full old-to-replacement overlap 393.
  Train2/Blind2: query-side overlap 0, old-fragment overlap 46, full old-to-replacement overlap 430.
  Dev2/Blind2: query-side overlap 0, old-fragment overlap 17, full old-to-replacement overlap 188.
  Attachment-signature overlap is 5 for these split pairs unless a local evidence file proves otherwise.
- Use wording:
  “The split removes exact query-side old-fragment/attachment-context overlap. It does not remove all old-fragment, replacement-triple, or attachment-signature relatedness.”
- Do not say “eliminates leakage”, “chemically independent”, or “fully held out chemical space”.

4. Align task formulation with full-vocabulary policy
- Keep C_i = V_train^(p).
- Fresh Blind2 ranks each query against full Train2-derived 161-fragment vocabulary.
- Attachment signature is query context / feature input, not primary hard candidate filter in full-161.
- Attachment-filtered subsets are sensitivity analyses only.

5. Compress A4C
- Main Methods: one short paragraph only.
- “A4C annotations are reported only in Supplementary Information as workflow-level diagnostics. They are not used for fitting, feature selection, ranking claims, activity validation, safety scoring, or medicinal-chemistry validation.”
- Remove detailed A4C workflow, G2/G3/G4 table, alert-rate discussion, extensive PAINS/Brenk framing from main manuscript.

6. LambdaRankCat boundary
- Results 4.7 and Discussion 5.4 should say:
  “Categorical-aware LambdaRank was evaluated as a post-hoc architecture diagnostic after the HistGB evidence hierarchy was established. It robustly improves over Score Blend, but its comparison against HistGB is heterogeneous at group level. It is therefore a future pre-specified architecture path, not the current main method.”

7. Merge boundary diagnostics
- Section title: “Boundary diagnostics: activity-comparable reranking and calibration sanity.”
- Main text only states:
  activity-comparable positives can be linked and ranked;
  degraded activity pairs can also receive high ranks;
  no activity-preservation or biological-validation claim is supported;
  HistGB outputs are ranking scores, not calibrated probabilities.

8. Prior_ranks language
- Section 3.5 title:
  “Prior_Ranks as a Post-Audit Instability Diagnostic”
- Safe language only:
  prior_ranks were suspected unstable after original secondary-blind audit;
  no-prior-rank configuration did not replicate Top-10 advantage on Fresh Blind2;
  sparse query-relative prior-rank transforms should be audited before being treated as transferable ranking signals.

9. Results structure
- 4.1 Fresh Blind2 benchmark and candidate matrix
- 4.2 Fresh Blind2 primary prospective performance
- 4.3 Query-level and grouped uncertainty
- 4.4 Prior-rank deletion does not prospectively replicate
- 4.5 DE/HGB complementarity as secondary-blind diagnostic
- 4.6 Candidate-matrix sensitivity
- 4.7 Learning-to-rank diagnostics as future architecture path
- 4.8 Boundary diagnostics: activity-comparable reranking and calibration sanity

10. Data Availability
- Replace old archive language with V5/Fresh Blind2 wording.
- Mention Fresh Blind2 split artifacts, full-161 candidate matrix, HistGB82/77 scorer outputs, grouped uncertainty audit, candidate-matrix sensitivity, LambdaRank diagnostic outputs, Claim Map V5, exact code tag / commit for V5 evidence lock if present locally.
- If no exact V5 code tag/commit can be found locally, write “V5 evidence-lock commit/tag to be inserted before submission” and list it as a blocker.
- Remove outdated labels if they point only to old secondary-blind paper.

11. Cleanup search
Search and fix:
- secondary blind is primary
- final model
- final scorer
- 77-feature main result
- valid replacement
- candidate likelihood
- calibrated probability
- activity preservation
- safety scoring
- medicinal chemistry validation
- A4C main result
- old-test quarantine
- D4S
- JCIM-facing evidence patch
- internal run labels in main text

Return a concise summary of files changed and any blockers.

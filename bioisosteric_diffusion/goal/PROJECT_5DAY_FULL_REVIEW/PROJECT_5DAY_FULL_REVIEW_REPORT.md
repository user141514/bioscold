# PROJECT 5-DAY FULL REVIEW REPORT

**Date**: 2026-05-29
**Period**: 2026-05-25 to 2026-05-29
**Scope**: D4S27 through D4S32, D4S28R, D4A2/A4C, Paper-S4GEN, Paper-S6R/S7

---

## 1. Five-Day Timeline

| Date | Event | Significance |
|------|-------|-------------|
| 05-25 | D4S27 similarity-transfer prior evaluation | FAILED — no deployable gain |
| 05-25 | D4S-B0 blind split baseline established | Baseline 0.8558 locked |
| 05-25-26 | Paper-S0 through S3TEXT manuscript pipeline | Journal-facing draft created |
| 05-26 | Paper-S3CITE citation resolution | 14 real citations integrated |
| 05-26-27 | Paper-S3FIG/S3PLOT figures and tables | Visual assets created |
| 05-27 | Paper-S4 Nature/NMI polish | Later rejected for S4GEN |
| 05-28 | D4S28 candidate-level scorer val result | 0.9374 OOF — breakthrough |
| 05-28 | D4S28 blind = 0.7120 | Initially interpreted as generalization failure |
| 05-28 | D4S8-D4S26 algorithm optimization | All 20+ attempts failed (delta=0 or negative) |
| 05-29 | D4S28 alignment bug discovered | 0.7120 → 0.8851 after cat_templates fix |
| 05-29 | D4S30 USR shape audit | USR = zero net benefit |
| 05-29 | D4S28R route audit | All routing strategies rejected |
| 05-29 | D4S31 feature ablation | drop prior_ranks → 0.9243, lost=6 |
| 05-29 | D4S32 query router | Failed — AUC=0.61, blind crashed |
| 05-29 | Paper-S4GEN general journal polish | Completed, verdict A |
| 05-29 | Paper-S6R merge adoption | Score blend adopted as primary (0.8558) |
| 05-29 | THIS REVIEW | Full synthesis and gap analysis |

## 2. Per-Experiment Final Verdicts

### D4S27: Conditional Prior via Similarity-Transfer
**Verdict**: FAILED for deployment. Oracle headroom +0.052 proven.
**Key number**: Max delta = 0.000000. 689 unique rescues, ~3300 hit losses.
**Significance**: Proved the SIGNAL exists but query-level gate cannot exploit it. Motivated D4S28.

### D4S28: Candidate-Level HistGB Scorer
**Verdict**: VALID in principle, blind initially invalid (bug).
**Key numbers**: Val OOF = 0.9374 (+0.103). Blind bugged = 0.7120. Blind fixed = 0.8851 (+0.029).
**Significance**: First method to exceed baseline on blind. Alignment bug was a critical lesson.

### D4S28R: Aligned Scorer + Route Audit
**Verdict**: scorer_all = 0.8851 deployable. All routing REJECTED.
**Key numbers**: 987 rescued, 596 lost, net +391. No route beats scorer_all.
**Significance**: Fragment-level routing impossible (0 fragment overlap). Val-blind delta r = -0.7371.

### D4S30: USR Shape Augmented
**Verdict**: USR = zero value. Safe features alone = 0.906.
**Key numbers**: USR contribution = -0.011. Safe-only blind = 0.9059.
**Significance**: 2D Morgan similarity already captures shape-relevant signal.

### D4S31: Feature Pruning / Drop prior_ranks
**Verdict**: BEST DEPLOYABLE. 0.9243 with lost=6.
**Key numbers**: +0.0686 over baseline. 424 rescued, 6 lost. All negative subspaces resolved.
**Significance**: The key insight — per-query ranks are a generalization hazard. This is a publishable finding.

### D4S32: Query Router
**Verdict**: FAILED. AUC=0.61. Blind crashed.
**Significance**: Confirms routing is too hard with limited fragment diversity. Negative result that supports the "scorer on all queries" approach.

### D4A2/A4C: Canonical Ranker + Review Proxy
**Verdict**: Borda beats HGB (+0.0425). A4C coverage 41.97% (incomplete).
**Significance**: DE-HGB complementarity proven. A4C needs coverage expansion.

### Paper-S4GEN: General Journal Polish
**Verdict**: A — READY for figure/format finalization.
**Significance**: Clean manuscript but describes wrong (Borda-era) method.

### Paper-S6R/S7: Score Blend Adoption
**Verdict**: Score blend (0.8558) adopted as primary. Borda demoted to baseline.
**Significance**: S6R is still pre-D4S31. D4S31 needs to be integrated.

## 3. Retracted Claims (12 total)

See `conflicts/retracted_claims_table.csv` for full list.

Most impactful retractions:
1. D4S28 0.7120 = generalization failure → alignment BUG
2. USR breakthrough → zero value
3. Fragment routing possible → impossible
4. 82 features = best → 77 features (drop prior_ranks) better
5. All safe features safe → prior_ranks NOT safe

## 4. Final Credible Conclusions

1. **D4S31 is the best deployable scorer**: blind Top10 = 0.9243, lost = 6
2. **Per-query rank features are a generalization hazard**: design rule for learning-to-rank
3. **Feature ablation methodology works**: leave-one-group-out identifies prior_ranks as the problem
4. **DE-HGB complementarity is real**: Borda improves over both constituents
5. **Candidate-level scoring beats query-level gating**: D4S28 >> D4S27
6. **USR shape descriptors are redundant**: 2D Morgan similarity already captures this signal
7. **Routing strategies fail under transform-heldout**: fragment-level impossible, cluster-level unreliable
8. **Score blend (0.8558) is the pre-D4S31 baseline**, not the final method

## 5. Current Best Method

**D4S31 drop_prior_ranks HistGB scorer**
- 77 features (82 SAFE - 5 prior_ranks)
- Blind Top10: 0.9243 [0.9198, 0.9289]
- Delta vs baseline: +0.0686 [0.0633, 0.0739]
- Rescue/Lost: 424/6 (22.0% rescue rate, 0.05% loss rate)
- All fragments at baseline parity or better

## 6. Current Paper's Biggest Problem

**The paper describes the wrong method.**

S3TEXT/S4GEN presents Borda(DE,HGB) = 0.8384 as the primary result.
Actual best method is D4S31 = 0.9243 (not in the paper).

Methods section is a D4A1-era skeleton missing:
- D4S31 feature families (9 families, 77 features)
- prior_ranks removal (the key design insight)
- Feature alignment (cat_templates)
- Candidate matrix definition
- Rescue/lost methodology
- Pipeline topology

## 7. Why Methods Is Not Adequate

Current Methods (§3.1-3.6, ~3 pages):
- Task definition: adequate but imprecise
- Benchmark construction: adequate
- Models: describes Borda-era (DE, HGB, Borda, MLP) only
- Misses: the ENTIRE D4S28→D4S31 pipeline (candidate scorer, features, ablation, alignment)
- Evaluation protocol: mentions bootstrap CI but not rescue/lost definitions
- A4C: minimal, lacks provenance label definitions

A reader cannot understand what D4S31 does or how it was built from the current Methods.

## 8. Next Action: Methods Rewrite First

The most rational next step is to write the Methods section from scratch using the blueprint at:
`methods_blueprint/Methods_section_outline.md`

This must precede any Results/Discussion updates because you cannot report numbers for a method that hasn't been defined.

---

## Files Produced by This Review

```
PROJECT_5DAY_FULL_REVIEW/
├── files/
│   ├── recent_file_inventory.csv        (1663 files)
│   ├── relevant_directory_map.md        (97 directories)
│   └── read_priority_list.md
├── modules/
│   ├── D4S27_summary.md
│   ├── D4S28_summary.md
│   ├── D4S28R_summary.md
│   ├── D4S30_summary.md
│   ├── D4S31_summary.md
│   ├── D4S32_summary.md
│   ├── D4A2_A4C_summary.md
│   └── Paper_S4GEN_summary.md
├── conflicts/
│   ├── retracted_claims_table.csv       (12 claims)
│   └── version_conflict_report.md
├── facts/
│   ├── final_metric_table.csv           (24 methods)
│   ├── final_claim_table.md
│   └── final_method_status_table.md
├── paper_strategy/
│   ├── current_paper_problem.md
│   ├── recommended_final_storyline.md
│   ├── methods_required_content.md
│   ├── results_required_tables.md
│   └── discussion_required_points.md
├── methods_blueprint/
│   ├── Methods_section_outline.md
│   ├── Feature_table_required.csv       (77 features documented)
│   ├── Data_to_candidate_matrix_pipeline.md
│   ├── Base_rankers_pipeline.md
│   ├── D4S31_candidate_scorer_pipeline.md
│   ├── Leakage_control_required_table.csv
│   └── Evaluation_protocol_required.md
├── PROJECT_5DAY_FULL_REVIEW_REPORT.md   (this file)
├── PROJECT_5DAY_EXECUTIVE_SUMMARY.md
├── PROJECT_5DAY_NEXT_ACTIONS.md
└── TASK.md
```

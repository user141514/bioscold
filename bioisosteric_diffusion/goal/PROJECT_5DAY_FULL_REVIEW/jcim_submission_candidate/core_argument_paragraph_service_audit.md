# Core Argument Paragraph Service Audit

Audit objective: force the manuscript into one benchmark-audit argument: remove transform memorization first, then test which ranking signals still transfer.

Allowed paragraph roles:

- benchmark necessity
- candidate-level scoring evidence
- prior_ranks shortcut mechanism
- workflow-only A4C
- limitation
- delete/downweight

## Introduction

| Paragraph | Role | Verdict | Rationale |
|---|---|---|---|
| I-1 | benchmark necessity | Keep | Defines the closed-vocabulary fragment replacement task and states that it is structure-derived rather than activity-preserving. |
| I-2 | benchmark necessity | Keep | Explains why leakage control is needed: transform leakage, strong frequency baselines, and difficult signal combination. |
| I-3 | benchmark necessity / candidate-level scoring evidence / prior_ranks shortcut mechanism | Keep | States the three-layer contribution hierarchy and explicitly downweights Borda, A4C, and routing. |
| I-4 | candidate-level scoring evidence / prior_ranks shortcut mechanism | Keep, but monitor | Reports core secondary blind values and post-audit boundary. A4C is framed as workflow-only, not final takeaway. |
| I-5 | benchmark necessity | Keep | Restates hierarchy and prevents Borda/A4C/routing from becoming parallel contributions. |

Flagged Introduction paragraphs: none.

## Results

| Paragraph | Role | Verdict | Rationale |
|---|---|---|---|
| R-1 | benchmark necessity | Keep | Announces Results order: split stress test first, scorer second, prior_ranks audit third. |
| R-2 | benchmark necessity | Keep | Reports permissive split overlap and base-ranker inflation; this is the benchmark necessity evidence. |
| R-3 | benchmark necessity | Keep | Adds the critical caveat that Attachment-Frequency decreases and primary claims use transform-heldout secondary blind only. |
| R-4 | candidate-level scoring evidence | Keep | Reports 82-feature prospective result and 77-feature post-audit result under secondary blind. |
| R-5 | candidate-level scoring evidence | Keep | Transitional sentence to Table 1. |
| R-6 | candidate-level scoring evidence | Keep | Table 1 caption clarifies evidence status and diagnostic bound. |
| R-7 | candidate-level scoring evidence | Keep | Explains base-ranker hierarchy, Borda complementarity, Score Blend baseline, and candidate-level upgrade. |
| R-8 | candidate-level scoring evidence | Keep | Keeps Best-of-DE+HGB diagnostic bounded; prevents ceiling/oracle interpretation. |
| R-9 | prior_ranks shortcut mechanism | Keep | Reports main leave-one-family-out prior_ranks deletion. |
| R-10 | prior_ranks shortcut mechanism | Keep | Table 2 caption explains post-audit deletion and CI availability. |
| R-11 | prior_ranks shortcut mechanism | Keep | Mechanistic explanation of query-relative prior rank instability. |
| R-12 | prior_ranks shortcut mechanism | Keep | Targeted prior-family and support-bin audit; supports rank-specific shortcut interpretation. |
| R-13 | prior_ranks shortcut mechanism / limitation | Keep | Explicitly marks the finding as post-audit analytical rather than pre-registered. |
| R-14 | candidate-level scoring evidence / prior_ranks shortcut mechanism | Keep | Rescue/lost decomposition supports why 77-feature is preferable to 82-feature. |
| R-15 | candidate-level scoring evidence | Keep | Table 3 caption defines aligned rescue/lost arithmetic. |
| R-16 | candidate-level scoring evidence / prior_ranks shortcut mechanism | Keep | Reports lost-query reduction and points per-fragment details to Supplementary tables. |
| R-17 | limitation | Keep as audit note | Categorical schema repair is an implementation audit, not a model contribution. |
| R-18 | workflow-only A4C | Keep, but clearly secondary | Introduces A4C only after primary benchmark/scoring/mechanism results. |
| R-19 | workflow-only A4C | Keep | Table 4 caption emphasizes coverage limitation. |
| R-20 | workflow-only A4C | Keep | Maintains computational-triage boundary and avoids validation language. |

Flagged Results paragraphs: none for deletion. R-17 and R-18--R-20 are secondary/supporting and should not be promoted in Abstract or Conclusion.

## Discussion

| Paragraph | Role | Verdict | Rationale |
|---|---|---|---|
| D-1 | candidate-level scoring evidence / prior_ranks shortcut mechanism / workflow-only A4C | Keep, but watch hierarchy | Summarizes three empirical findings; A4C is third and explicitly triage-only. |
| D-2 | prior_ranks shortcut mechanism / limitation | Keep | States post-audit boundary and need for prospective replication. |
| D-3 | candidate-level scoring evidence | Keep | Query-level routing failure supports candidate-level scoring design; negative diagnostic only. |
| D-4 | prior_ranks shortcut mechanism | Keep | Uses bounded shortcut language and points to Supplementary evidence. |
| D-5 | prior_ranks shortcut mechanism / limitation | Keep | Avoids universal claims about rank features or prior information. |
| D-6 | candidate-level scoring evidence | Keep | Borda is explicitly mechanism anchor, not final method. |
| D-7 | workflow-only A4C | Keep, secondary | Describes provenance strata as workflow language, not validation. |
| D-8 | workflow-only A4C / limitation | Keep | A4C limitations are explicit and protective. |
| D-9 | limitation | Keep | Positions closed-vocabulary ranking relative to database/generative methods. |
| D-10 | limitation | Keep | States closed-vocabulary scope and future open-vocabulary direction. |
| D-11 | limitation | Keep | Lists core limitations: single data source, structure-derived labels, post-audit pruning, A4C proxy. |
| D-12 | limitation | Keep | Future work: independent replication, activity endpoints, GNN representations, expert review calibration. |

Flagged Discussion paragraphs: D-1 should not expand further; if space is tight, compress the A4C clause so the final Discussion emphasis stays on benchmark-audit and signal transfer.

## Overall Audit Verdict

The current Introduction, Results, and Discussion now mostly serve the requested benchmark-audit argument. No paragraph is structurally orphaned. The only downweight risk is A4C: it remains correctly framed as workflow-only, but it should not be moved into the Abstract ending, Conclusion ending, or core figure sequence.

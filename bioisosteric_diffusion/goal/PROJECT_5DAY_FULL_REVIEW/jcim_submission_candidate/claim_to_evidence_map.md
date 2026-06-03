# Claim-to-Evidence Map

This map records the current JCIM submission-candidate claim hierarchy after the manuscript identity repair. The manuscript should be read as three linked evidentiary layers: transform-heldout benchmark, candidate-level scoring under secondary blind evaluation, and post-audit prior_ranks shortcut diagnosis.

## Layer 1: Transform-Heldout Benchmark

| Claim | Manuscript Location | Evidence | Status / Boundary |
|---|---|---|---|
| The benchmark uses a transform-heldout secondary blind protocol with 13,347 queries. | Abstract; Introduction; Methods 3.2; Results 4 | Candidate-matrix split statistics and leakage verification in Tables M1-M2 | Primary benchmark claim; controls identical fragment-attachment transform overlap but not every form of chemical similarity. |
| Permissive random splitting changes learned base-ranker transfer estimates. | Results 4.1; Supplementary Table S6 | 840/840 train-blind transform-key overlap; DE +0.0729 [0.0643, 0.0814], HGB +0.0755 [0.0655, 0.0853], Borda +0.0316 [0.0230, 0.0398] under permissive split | Split-protocol diagnostic for locked standalone base rankers only; Score Blend and candidate-level scorers were not rebuilt under this split. |
| Random/permissive splitting does not inflate every baseline. | Results 4.1; Supplementary Table S6 | Attachment-Frequency decreases from 0.6019 to 0.5727 under the permissive split | Boundary claim; prevents overstatement of the split stress test. |

## Layer 2: Candidate-Level Scoring under Secondary Blind Evaluation

| Claim | Manuscript Location | Evidence | Status / Boundary |
|---|---|---|---|
| Score Blend is the strongest pre-D4S baseline using base-ranker signals. | Results 4.2; Table 1 | Score Blend Top-10 = 0.8558, 95% CI [0.8495, 0.8621] | Baseline hierarchy claim. |
| The initial 82-feature scorer is the prospective candidate-level blind result. | Abstract; Introduction; Results 4.2; Table 1 | Top-10 = 0.8851, 95% CI [0.8796, 0.8903] | Prospective candidate-level result; later post-audit diagnostics show more regressions than the 77-feature scorer. |
| The post-audit 77-feature scorer is the locked post-selection result. | Abstract; Introduction; Results 4.2; Table 1; Discussion 5.1 | Top-10 = 0.9243, 95% CI [0.9199, 0.9287]; Delta vs Score Blend = +0.0686, 95% CI [+0.0638, +0.0733] | Strongest locked result in this draft; not a fully prospective feature-selection result. |
| Borda(DE,HGB) is a complementarity/mechanism anchor. | Results 4.2; Discussion 5.3 | Borda Top-10 = 0.8384 and exceeds either individual base ranker under transform-heldout evaluation | Borda is not the final best method. |
| Query-level routing is a negative diagnostic. | Discussion 5.2 | Runtime-safe Borda/HGB router: Delta Top-10 = 0.0000, 95% CI [-0.0008, 0.0008]; learned router about 0.61 validation AUC | Not a manuscript contribution; supports candidate-level rather than query-level decision design. |

## Layer 3: Prior_Ranks Shortcut Diagnosis

| Claim | Manuscript Location | Evidence | Status / Boundary |
|---|---|---|---|
| Removing prior_ranks improves transfer. | Results 4.3; Table 2 | Drop prior_ranks: Delta Top-10 = +0.0393, 95% CI [+0.0354, +0.0432] vs initial 82-feature scorer | Post-audit analytical finding; not pre-registered feature selection. |
| The unstable prior component is rank-specific rather than prior information in general. | Results 4.3; Discussion 5.3; Supplementary Figure S5 | Drop all non-rank prior families +0.0174; drop prior statistics +0.0151; drop prior scores +0.0073; drop PMI/conditional-prior contrasts -0.0037, 95% CI [-0.0064, -0.0010] | Bounded shortcut interpretation; does not imply all raw prior features are beneficial or all rank features are harmful. |
| The 77-feature gain is largest in low train-derived support queries. | Results 4.3; Discussion 5.3; Supplementary Figure S5 | Low support +0.0488; medium +0.0344; high +0.0346 | Support-bin diagnostic; not causal proof. |
| Prior-rank pruning reduces regression while preserving rescue. | Results 4.4; Supplementary Tables S3/S5 | Lost vs Score Blend decreases from 596 to 101; 77-feature rescues 1,016 Score Blend misses and loses 101 hits; per-fragment table shows two negative old-fragment point estimates vs Score Blend and 17/19 non-negative relative to the 82-feature scorer | Reliability and block-level point-estimate diagnostics; not per-fragment significance testing. |

## Supporting Workflow Only

| Claim | Manuscript Location | Evidence | Status / Boundary |
|---|---|---|---|
| A4C provenance strata provide coverage-limited triage signals. | Methods 3.7; Results 4.6; Table 4; Discussion 5.4 | G2 alert rate 46.85%; G3 9.67%; G4 coverage 5.63%, with 94.37% unknown | Workflow-only computational proxy; not expert validation, toxicity prediction, activity preservation, or review-safe production. |
| Structure-derived MMP labels support replacement-pattern recovery only. | Methods 3.1-3.2; Results 4 | ChEMBL MMP-derived positives and query-level Top-10 metrics | Does not prove activity-preserving bioisostere discovery. |

## Figure/Supplement Linkage

| Artifact | Claim Supported | Status |
|---|---|---|
| Supplementary Figure S5. Prior-ranks mechanism audit | Targeted prior-family ablation and support-bin diagnostic | Active supplementary mechanism figure. |
| Supplementary Table S4 | Leave-one-family-out ablation values | Only prior_ranks deletion has paired CI in the current lock. |
| Supplementary Table S6 | Random/permissive split stress test | Locked standalone base-ranker diagnostic only. |

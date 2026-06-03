# P0 Manuscript Identity Completion Audit

Date: 2026-06-02

## Scope

This audit verifies the completion of the P0 manuscript-identity repair tasks:

1. restore the three-layer contribution hierarchy;
2. move the random/permissive split stress test to the beginning of Results;
3. integrate the prior_ranks mechanism evidence into the manuscript and supplement;
4. remove old risk phrases and internal-code residue.

The authoritative text files checked are:

- `jcim_submission_candidate/main_manuscript.md`
- `paper_draft/combined_draft_clean.md`
- `jcim_submission_candidate/supporting_information.md`
- `paper_draft/supplementary/supplementary_tables.md`
- `jcim_submission_candidate/claim_boundary_matrix.md`
- `jcim_submission_candidate/claim_to_evidence_map.md`
- `paper_draft/paper_version2/claim_to_evidence_map.md`
- `jcim_submission_candidate/figures/main/ACTIVE_FIGURE_SET.md`

## Requirement-by-Requirement Verification

| Requirement | Evidence | Status |
|---|---|---|
| Contribution 1 = transform-heldout benchmark | Introduction states the first evidentiary layer is the transform-heldout benchmark; Results 4.1 opens with permissive split stress test. | PASS |
| Contribution 2 = candidate-level scoring under secondary blind | Introduction states candidate-level scoring as the second evidentiary layer; Results 4.2 reports Table 1 secondary blind performance. | PASS |
| Contribution 3 = prior_ranks shortcut diagnosis | Introduction states prior_ranks shortcut diagnosis as the third evidentiary layer; Results 4.3 and Discussion 5.3 contain the targeted ablation/support-bin interpretation. | PASS |
| A4C is supporting workflow only | Introduction says A4C is a supporting computational triage workflow; Methods 3.7 and Results 4.6 retain coverage-limited computational-proxy caveats. | PASS |
| Borda is complementarity/mechanism anchor | Introduction and Discussion 5.3 state Borda's mechanism-anchor role; Table 1 no longer frames Borda as final method. | PASS |
| Query routing is negative diagnostic only | Introduction and Discussion 5.2 state query-level routing is a negative diagnostic, not a contribution. | PASS |
| Results 4.1 = permissive split stress test | Section title is `4.1 Permissive Splits Inflate Learned Ranking Signals`; reports 840/840 transform-key overlap and DE/HGB/Borda inflation with Attachment-Frequency decrease caveat. | PASS |
| Results 4.2 = candidate-level scoring | Section title is `4.2 Candidate-Level Scoring Improves Top-10 Recovery under Transform-Heldout Evaluation`; Table 1 follows this section. | PASS |
| Results 4.3 = prior_ranks transfer mechanism | Section title is `4.3 Removing Prior_Ranks Improves Transfer`; targeted prior-family ablation paragraph is included. | PASS |
| Claim-to-evidence map updated | New candidate file `jcim_submission_candidate/claim_to_evidence_map.md` records benchmark, scorer, prior_ranks, routing, and A4C evidence boundaries; old `paper_version2` map synchronized. | PASS |
| Mechanism figure listed as Supplementary Figure | Supporting Information figure directory and `ACTIVE_FIGURE_SET.md` list `Supplementary Figure S5. Prior-ranks mechanism audit`. | PASS |
| Discussion 5.3 uses bounded shortcut interpretation | Discussion 5.3 says query-relative prior-rank features behaved as non-transferable shortcuts; it explicitly avoids universal claims about rank features or prior information. | PASS |
| Remove unsupported `not significant` wording | `rg` check across authoritative text files found no `not significant`. | PASS |
| Remove main-text internal code `D4S-B0` | `rg` check across authoritative manuscript/SI/claim-map files found no `D4S-B0`; SI now uses `locked standalone base rankers`. | PASS |
| Replace `feature set finalized` wording | Methods 3.5 now says leave-one-family-out ablation identified the post-audit feature-pruned configuration. | PASS |
| Replace abstract slogan about deletion | Discussion 5.1 and Conclusion use bounded audit wording: query-relative prior-rank features behave as non-transferable shortcuts under secondary blind fragment shift. | PASS |
| Correct DeepBioisostere authors | Reference 9 reads `Kim, H.; Moon, S.; Zhung, W.; Lim, J.; Kim, W. Y.` | PASS |

## Verification Commands

The following checks were run on the authoritative text files:

```powershell
rg -n "two main contributions|two supporting safeguards|Our contributions are as follows|Fourth, we|not significant|D4S-B0|feature set in Section 3\.4 was finalized|most impactful feature-engineering decision|central methodological finding is that the most impactful|Kim, S\.|### 4\.1 Candidate|### 4\.2 Removing|### 4\.3 Feature Pruning|### 4\.4 Categorical|### 4\.5 Dual-Mode|Results 4\.5; Discussion|Section 4\.5 reports A4C|Alert rates are reported in Section 4\.5" ...
```

Result: no matches.

```powershell
rg -n "We organize the study around three evidentiary layers|benchmark first, candidate-level scoring second, and prior_ranks shortcut diagnosis third|A4C is reported only as a supporting computational triage workflow|query-level routing is treated as a negative diagnostic|### 4\.1 Permissive Splits Inflate Learned Ranking Signals|### 4\.2 Candidate-Level Scoring Improves Top-10 Recovery under Transform-Heldout Evaluation|### 4\.3 Removing Prior_Ranks Improves Transfer|Supplementary Figure S5|Kim, H.; Moon, S.; Zhung, W.; Lim, J.; Kim, W. Y." ...
```

Result: required current-state anchors found.

```powershell
git diff --check -- [authoritative files]
```

Result: only CRLF normalization warnings; no whitespace errors.

## Notes

- This audit covers Markdown manuscript/SI/claim-map source files. Existing DOCX/PDF exports were not regenerated in this pass.
- Supplementary Figure S3 remains a separate figure-asset issue if the figure itself is kept in the final package; the textual P0 objective is complete.

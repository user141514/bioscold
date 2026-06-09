# Data Availability Audit

Audit date: 2026-06-03.

| Requirement | Audit result | Status |
|---|---|---|
| Public repository URL | Branch URL resolves: `https://github.com/user141514/paper1/tree/codex/jcim-algorithm-archive` | Pass |
| Evidence-lock manifest path | Branch raw manifest resolves at `bioisosteric_diffusion/goal/PROJECT_5DAY_FULL_REVIEW/jcim_submission_candidate/V5_FRESH_BLIND2_EVIDENCE_LOCK_MANIFEST.md` | Pass |
| Exact tag | Raw URL for `v5-fresh-blind2-evidence-lock-20260603` returned 404 during this audit | Fail |
| SHA256 availability | Local manifest contains SHA256 rows for Claim Map V5, Fresh Blind2 metrics, grouped uncertainty, candidate-matrix sensitivity, LTR diagnostics, activity/calibration diagnostics, and manuscript/SI package | Pass |
| Claim Map V5 | Manifest includes `CLAIM_MAP_V5.md` | Pass |
| Fresh Blind2 split artifacts | Manifest includes Blind2 manifest and leakage audit CSV rows | Pass |
| Full-161 candidate matrix definition | Manifest and manuscript Methods identify full Train2-derived 161-fragment vocabulary | Pass |
| HistGB82/77 outputs | Manifest includes `blind2_82_77_final_metrics.csv` and bootstrap outputs | Pass |
| Grouped uncertainty tables | Manifest includes `blind2_grouped_uncertainty.csv` | Pass |
| Candidate-matrix sensitivity | Manifest includes E2 candidate-compatibility sensitivity report, coverage, and metrics | Pass |
| Boundary diagnostics | Manifest includes activity reranking and calibration reports/metrics | Pass |
| V7 manuscript/SI package | Public branch contains V7 manuscript; local SI has been updated for conservative freeze but the manifest hash is not yet refreshed for this final freeze pass | Partial |

Local SHA256 after this pass:

| File | SHA256 |
|---|---|
| `manuscript_final_conservative_freeze.md` | `206DFCDE698CB96066030697599C206A1E8717EAB708865B892D4FD14495D288` |
| `supporting_information.md` | `E272161DB3A7DD126F376CB65BE143B5364203C6893A1FED4460ED3382C7D836` |
| `V5_FRESH_BLIND2_EVIDENCE_LOCK_MANIFEST.md` | `7FF72EE07866C66A43A18C8CA7126AFF326ACED14B3E371645C2747BEF72EE6A` |

Verdict: NOT FULLY FROZEN. The public branch is accessible, but the exact tag required by the freeze criteria did not resolve, and the manifest has not yet been refreshed for `manuscript_final_conservative_freeze.md` and the updated SI hash.

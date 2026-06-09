# Banned Phrase Audit

Audit date: 2026-06-03.

Scope: `manuscript_final_conservative_freeze.md` and `supporting_information.md`.

| Phrase class | Result | Interpretation |
|---|---|---|
| `77-feature main method` | Not found | Pass |
| `final model` / `final scorer` | Not found | Pass |
| `prior_ranks improves transfer` | Not found | Pass |
| `calibrated probability` / `calibrated probabilities` | Not found after rewrite | Pass |
| `candidate likelihood` | Not found | Pass |
| `valid replacement` | Not found | Pass |
| `activity preservation` | Not found after rewrite | Pass |
| `biological validation` | Not found after rewrite | Pass |
| `medicinal chemistry validation` | Not found after rewrite | Pass |
| `safety scoring` | Not found after rewrite | Pass |
| `uniform improvement` / `group-level improvement` | Not found | Pass |
| `full old-to-replacement triple` for nonzero overlap | Not found | Pass |
| Internal labels: `D4S`, `old-test quarantine`, `JCIM-facing patch` | Not found in final manuscript/SI | Pass |
| Old secondary-blind `0.9243` | Not found in main manuscript; present only in Supplementary diagnostic-history tables | Pass |

Verdict: PASS. Remaining historical numbers in SI are explicitly labeled diagnostic history and do not re-promote HistGB77.

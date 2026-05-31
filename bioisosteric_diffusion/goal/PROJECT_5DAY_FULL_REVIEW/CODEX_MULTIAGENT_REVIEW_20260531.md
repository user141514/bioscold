# Codex Multi-Agent Review - 2026-05-31

## Scope

Current manuscript line: `paper_draft/combined_draft_clean.md`.

Review goal: adversarial claim/evidence audit across the project, with code reruns where manuscript data support was questionable.

## Agents

| Agent | Role | Output |
|-------|------|--------|
| Lorentz | Manuscript claim audit | Identified unsupported/over-strong claims: per-fragment significance, D4S28R 6/19 wording, MLP MRR p-value, ablation-family overstatement, A4C caveat needs. |
| Ohm | Evidence-chain audit | Built key-number-to-artifact map and flagged rescue/lost, delta CI, ScoreBlend source, Oracle discrepancy, A4C G4 coverage. |
| Nietzsche | Code reproduction path audit | Located rerun scripts and verified `accfg_migrated` environment plus key dependencies and parquet readability. |
| Volta | Minimal text repair review | Proposed conservative manuscript wording changes. |
| Avicenna | A4C-specific audit | Confirmed G4 0.99% is total lower bound with 5.63% coverage and 94.37% unknown status. |
| Poincare | Evidence-lock sync strategy | Prioritized facts/status/lock-file synchronization to prevent stale 424/6 claims from leaking back. |

## Commands Rerun

```powershell
D:\anaconda0\envs\accfg_migrated\python.exe goal\A_improve\D4S31_feature_pruning\d4s31_lockdown.py
D:\anaconda0\envs\accfg_migrated\python.exe goal\A_improve\D4S31_feature_pruning\d4s31_rescue_lost_repair.py
D:\anaconda0\envs\accfg_migrated\python.exe core\scripts\routeA_paper_p0_newblind_metric_ablation_lock.py
```

## Rerun Results

| Quantity | Rerun result | Status |
|----------|--------------|--------|
| D4S31 Top10 | 0.924328 | Supported |
| D4S31 Top10 CI | [0.919830, 0.928898] | Supported |
| Delta vs ScoreBlend | +0.068555 | Supported |
| Delta CI | [0.0636847, 0.0734997] | Supported; manuscript now rounds to [0.0637, 0.0735] |
| D4S31 rescue/lost vs ScoreBlend | 1,016 / 101 | Supported by query_id merge repair |
| D4S31 net gain vs ScoreBlend | +915 | Supported |
| D4S31 old-fragment strata | 19/19 non-negative point-estimate deltas | Supported; not a significance claim |
| Borda(DE,HGB) | 0.8383906 | Reproduced by Paper-P0 lock script |
| Oracle(DE,HGB) | 0.8685847 | Reproduced by Paper-P0 lock script; use as diagnostic bound |

## Main Corrections Applied

### Manuscript

File: `paper_draft/combined_draft_clean.md`

- Delta CI changed from `[0.0633, 0.0739]` to `[0.0637, 0.0735]`.
- Removed unsupported MLP MRR p-value claim; reframed MRR as secondary diagnostic.
- Reframed D4S28R degradation as five old-fragment strata plus one ATT:C|S attachment stratum.
- Reframed D4S31 per-fragment result as non-negative point-estimate deltas, not statistically significant non-degradation.
- Reframed ablation interpretation: prior_ranks is the dominant harmful group; not every retained family is individually beneficial.
- Reframed A4C G4 as coverage-limited lower-bound evidence, not a near-zero low-risk region.
- Softened broad generalizability language to testable design principle in this benchmark / related ranking tasks.

### Evidence Tables

Updated:

- `facts/final_metric_table.csv`
- `facts/final_claim_table.md`
- `facts/final_method_status_table.md`
- `PAPER_STATUS_SUMMARY.md`
- `goal/A_improve/D4S31_FINAL_LOCK/01_ablation_findings.md`
- `goal/A_improve/D4S31_FINAL_LOCK/02_final_metrics.md`
- `goal/A_improve/D4S31_FINAL_LOCK/04_blind_strata.md`
- `goal/A_improve/D4S31_FINAL_LOCK/_build_archive.py` warning banner

## Important Caveats Remaining

1. `goal/A_improve/D4S31_FINAL_LOCK/_build_archive.py` still contains stale hard-coded 424/6 text inside its templates. A warning banner was added. Do not rerun it until the templates are fully repaired.
2. Paper-P0 source CSV still describes Oracle(DE,HGB) as an "upper bound"; the manuscript should keep the safer wording "diagnostic bound, not a ceiling for feature-augmented models."
3. ScoreBlend full precision is still mostly reconstructed from 11,422/13,347 and D4S scripts. This is adequate for current claims but should eventually be locked as a standalone CSV row.
4. D4S28R lost=596 is supported by audit markdown/archive text, but not by a D4S31-style repaired query_id-merge CSV.
5. A4C G4 is coverage-limited: coverage=5.63%, covered alert rate=17.60%, total lower-bound=0.99%, unknown=94.37%.
6. Old complete drafts and Chinese drafts still contain stale 424/6 and "Oracle ceiling" language. Treat them as archived, not active sources.

## Verdict

The main D4S31 performance claim is now supported by rerun evidence. The manuscript remains viable, but only with the repaired claim hierarchy:

- Primary claim: D4S31 feature-regularized candidate scoring improves internal blind Top10 over ScoreBlend.
- Mechanistic claim: removing prior_ranks avoids a non-transferable shortcut under fragment distribution shift.
- Reliability claim: D4S31 has non-negative old-fragment point-estimate deltas and repaired 1,016/101 rescue/lost versus ScoreBlend.
- Workflow claim: A4C provenance strata are computational triage signals, with explicit coverage caveats.
- Forbidden: activity preservation, wet-lab validation, expert validation, universal safety, and Oracle-as-ceiling language.

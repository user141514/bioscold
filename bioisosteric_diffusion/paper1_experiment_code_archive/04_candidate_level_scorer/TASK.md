# D4S27: Conditional Transform Prior via old_fragment + attachment_signature

See TASK.md at /goal/A_improve/TASK.md (Git path) or this file.

## Goal
Validate old_fragment + attachment_signature conditional prior / transform representation learning.

## Background
- Current best: MLP+HGB blend, Top10=0.8558 (blind), 0.8340 (val eval subset)
- D4S24-D4S26: all delta Top10 = 0.000000
- D4S12 context prior: already implemented P4(oid+att+cand), no val gain
- This experiment adds: PMI/lift computation, continuation prior, exhaustive stratified eval

## Data
- Priors: D4S12 cached P4 (old+att+cand), P1 (att+cand), P0 (cand only)
- Val matrix: D4S6 corrected (2.07M rows, blend_score included)
- Train counts from D4S12 — no leakage

## Key scripts
- `d4s27_conditional_prior.py`: Full evaluation pipeline
- `d4s27_run.log`: Execution log
- `d4s27_all_metrics.csv`: All method metrics
- `d4s27_stratified_query.csv`: Per-query comparison

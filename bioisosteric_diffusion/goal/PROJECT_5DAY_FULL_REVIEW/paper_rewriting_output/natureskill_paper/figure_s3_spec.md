# Figure S3 Specification: Per-Fragment Blind Deltas

## Purpose

Support the stratum-level claim in Results 4.1 and 4.3:

- the 82-feature scorer degraded five old-fragment strata;
- the 77-feature scorer has no negative point-estimate delta across all 19 old-fragment strata.

## Data

Use the locked tables:

- 82-feature scorer strata: `goal/A_improve/D4S28R_FINAL_LOCK/d4s28_blind_fixed_strata.csv`
- 77-feature scorer strata: `goal/A_improve/D4S31_FINAL_LOCK/d4s31_blind_strata.csv`

Derived fields:

- `delta_82 = scorer_Top10 - blend_Top10`
- `delta_77 = sc_Top10 - bl_Top10`
- `N = n`

## Figure Type

Horizontal paired-dot / lollipop plot.

Each row is one `old_fragment`, sorted by `delta_82` ascending or by `delta_77 - delta_82` descending.

Visual encoding:

- grey vertical zero line;
- muted red/orange dot for 82-feature delta;
- blue/green dot for 77-feature delta;
- thin connector between the two dots;
- right-side `N` label;
- highlight the five degraded 82-feature strata.

## Caption Boundary

Caption must state:

> Values are stratum-level point estimates. No per-fragment bootstrap confidence intervals or per-fragment significance tests are reported.


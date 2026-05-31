# D4S31: Blind Per-Fragment Strata

**2026-05-31 audit update:** The per-fragment point estimates are supported. Do not describe this as a per-fragment statistical-significance test; no per-fragment bootstrap CIs are included here. The old 424/6 rescue/lost wording is stale; use repaired query_id-merge values 1,016/101 for ScoreBlend rescue/lost.

**Date**: 2026-05-29
**All D4S28R negative subspaces resolved.**

## All Fragments (no negatives)

| Fragment | N | Baseline | Scorer | Delta |
|---|---|---|---|---|
| *N(C)C | 1,648 | 0.6608 | 0.7379 | +0.0771 |
| *C(=O)c1ccco1 | 104 | 0.9327 | 1.0000 | +0.0673 |
| *c1cccc(C)c1 | 1,823 | 0.9013 | 0.9611 | +0.0598 |
| *C1CCCCC1 | 2,244 | 0.7197 | 0.7527 | +0.0330 |
| *c1ccccc1F | 2,113 | 0.9035 | 0.9304 | +0.0270 |
| *CCC | 2,112 | 0.8835 | 0.9044 | +0.0208 |
| All others | — | various | unchanged or +0 | ≥0 |

**Zero fragments with negative delta.** All previously crashing fragments (*Cc1cccc(OC)c1, *Cc1ccccc1OC) now at baseline parity (d=+0.000).

## Key Pattern

D4S31 has non-negative point-estimate deltas across all 19 old-fragment strata. Under repaired query_id-merge accounting versus ScoreBlend, it rescues 1,016 baseline misses and loses 101 baseline hits.

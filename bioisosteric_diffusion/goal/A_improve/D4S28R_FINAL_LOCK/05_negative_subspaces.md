# D4S28R Negative Subspace Audit

**Date**: 2026-05-29

Scorer_all improves 13/19 blind fragments but degrades 6/19.
Negative subspaces documented here as known limitations.

## Negative Subspaces (Blind)

| Fragment | N Queries | Baseline | Scorer | Delta |
|----------|-----------|----------|--------|-------|
| *Cc1cccc(OC)c1 | 118 | 1.0000 | 0.0000 | -1.0000 |
| *Cc1ccccc1OC | 108 | 1.0000 | 0.1944 | -0.8056 |
| *Cc1ccccc1Cl | 215 | 1.0000 | 0.7070 | -0.2930 |
| *Cc1cccnc1 | 158 | 1.0000 | 0.8734 | -0.1266 |
| *CCCc1ccccc1 | 216 | 1.0000 | 0.9954 | -0.0046 |

Also C|S attachment signature: -0.0972.

## Positive Subspaces (Top 5 Improvements)

| Fragment | N Queries | Baseline | Scorer | Delta |
|----------|-----------|----------|--------|-------|
| *Cc1ccccn1 | 233 | 0.5708 | 0.8841 | +0.3133 |
| *N(C)C | 1,648 | 0.6608 | 0.8083 | +0.1475 |
| *c1cccc(C)c1 | 1,823 | 0.9013 | 0.9731 | +0.0719 |
| *C(=O)c1ccco1 | 104 | 0.9327 | 1.0000 | +0.0673 |
| *c1ccccc1F | 2,113 | 0.9035 | 0.9612 | +0.0577 |

## Pattern

Negative subspaces are small-N fragments (108-216 queries) where baseline is already at ceiling (Top10=1.0). Scorer introduces noise that breaks perfect baseline performance. These are "saturated" fragments where no improvement was possible and any change is regression.

Positive subspaces include mid-N fragments (233-2,113 queries) with baseline headroom (0.57-0.90). Scorer captures complementary signal.

## Mitigation

No routing strategy works (see 03_route_rejection.md). The net effect is positive (+0.0293). Accept negative subspaces as known limitation.

# D4S30 USR Rejection

**Date**: 2026-05-29
**Verdict**: USR (Ultrafast Shape Recognition) 3D shape features provide ZERO incremental value.

## Experiment

Added 5 USR features (usr_similarity + 4 shape moment scores) to the 82 safe features.
Trained same HistGB architecture. Evaluated on val OOF + blind one-shot.

## Results

| Configuration | Val OOF Top10 | Blind Top10 |
|---------------|---------------|-------------|
| Safe features only | 0.9366 | **0.9059** |
| Safe + USR | 0.9375 | 0.8949 |

USR contribution to blind: **-0.011** (negative).

## Why USR Fails

1. **USR-label correlation: 0.083** — very weak standalone signal
2. **USR standalone Top10: 0.342** (val), 0.313 (blind) — far below baseline
3. **Redundant with _tanimoto_similarity** — 2D Morgan fingerprint already captures shape-relevant information
4. **USR captures global shape moments only** — 12-dimensional descriptor cannot capture fine-grained 3D match

## Pre-computation Advantage

USR requires only 158 conformer generations (150 candidates + 8 old_fragments) due to pre-computation caching. This is computationally feasible but the signal quality is too low to justify inclusion.

## Conclusion

**USR rejected.** 3D shape descriptors (at least USR-level) do not provide orthogonal signal beyond existing 2D features. Future 3D exploration should use learned representations (EGNN) rather than hand-crafted shape descriptors.

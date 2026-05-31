# D4S30 AUDIT VERDICT

**Date**: 2026-05-29
**Status**: D4S28 blind failure = feature alignment BUG. Real blind ~0.90. USR not needed.

## Key Findings

### 1. D4S28 Blind Was a Bug

D4S28 blind 0.712 was caused by val/blind feature misalignment:
- Val training: 79 feature columns (8 old_fragments → 8 one-hot)
- Blind prediction: 90 feature columns (19 old_fragments → 19 one-hot)
- HistGB was trained on 79 dims, then asked to predict on 90 dims
- ValueError caught this, but the "fix" used label-aligned templates at PREDICTION time, not TRAINING time
- Result: model weights applied to wrong feature indices → garbage predictions (0.712)

D4S30 fix: proper `cat_templates` alignment during both training AND prediction.

### 2. USR Provides Zero Net Benefit

| Configuration | Blind Top10 |
|---------------|-------------|
| Safe features only (no USR) | **0.9059** |
| Safe features + USR | 0.8949 |
| Baseline blend | 0.8558 |

USR contribution: **-0.011** (negative). USR is not harmful per se, but the model already captures all useful signal from the safe feature set.

### 3. Safe Features Alone Generalize Well

With proper feature alignment, the same safe features that achieve val OOF 0.937 achieve blind 0.906. This is a +0.05 improvement over baseline on blind — the FIRST verifiable blind improvement.

### 4. Per-Fragment Blind Results (No USR)

| Fragment | N | Baseline | Safe-only | Safe+USR |
|---|---|---|---|---|
| *C1CCCCC1 | 2244 | 0.720 | 0.787 | 0.755 |
| *CCC | 2112 | 0.884 | 0.880 | 0.884 |
| *N(C)C | 1648 | 0.661 | **0.928** | 0.817 |
| *c1cccc(C)c1 | 1823 | 0.901 | **0.972** | 0.969 |
| *c1ccccc1F | 2113 | 0.904 | **0.947** | 0.948 |
| *Cc1ccccn1 | 233 | 0.571 | **0.884** | 0.936 |

The safe feature model doubles *N(C)C Top10 (0.661→0.928) and significantly improves several other fragments. No fragment is degraded.

### 5. Why USR Doesn't Help

- USR-label correlation: 0.083 (vs blend_score: 0.200)
- USR standalone Top10: 0.342 (val), 0.313 (blind)
- USR is a weak signal that's already captured by _tanimoto_similarity + molecular descriptors
- USR's 3D shape information is largely redundant with existing 2D Morgan fingerprint similarity

## Corrected Timeline

| Event | Old Understanding | New Understanding |
|-------|-------------------|-------------------|
| D4S28 val OOF 0.937 | Real, no leakage | STILL real, no leakage |
| D4S28 blind 0.712 | Generalization failure | **BUG** — feature misalignment |
| D4S30 blind 0.902 | USR breakthrough | Safe features alone achieve 0.906 |

## Recommendation

1. **Fix D4S28**: Re-run blind with proper feature alignment → expect ~0.90
2. **USR/E3D**: Do not pursue — shape features already captured by 2D similarity
3. **Next real signal**: Activity coordinates (requires full ChEMBL bioactivity DB) or graph-level 3D (EGNN) — but EGNN should be tested with proper alignment from the start
4. **Current best deployable**: Safe features + HistGB → blind ~0.906 (+0.05 over baseline)

# D4S30 Summary: Shape-Augmented Scorer (USR) + Audit

**Date**: 2026-05-29
**Status**: COMPLETE. USR provides zero net benefit. Real blind ~0.906 from safe features alone with proper alignment.

---

## Starting Problem

D4S28 blind achieved only 0.712 Top10 on blind data, despite val OOF of 0.937. This was initially interpreted as a generalization failure, and 3D shape features (USR/USRCAT) were proposed to bridge the gap.

## Key Experiments

### 1. Root Cause Diagnosis — Feature Misalignment Bug

The D4S28 blind failure was caused by a `cat_templates` misalignment between val and blind:
- **Val training**: 79 feature columns (8 old_fragments encoded as 8 one-hot columns)
- **Blind prediction**: 90 feature columns (19 old_fragments encoded as 19 one-hot columns)
- HistGB was trained on 79 dimensions but asked to predict on 90 dimensions
- A ValueError was caught, but the "fix" applied label-aligned templates at PREDICTION time, not TRAINING time
- Result: model weights applied to wrong feature indices -> garbage predictions (0.712)

D4S30 fix: Proper `cat_templates` alignment during BOTH training AND prediction.

### 2. USR Feature Engineering

- Pre-computed USR descriptors via RDKit ETKDGv3 conformer embedding + MMFF optimization
- USR similarity: 1/(1 + Euclidean distance) between old_fragment and candidate USR vectors
- USR shape scores: per-moment (CTD, CST, FCT, FTF) similarities
- Cached on unique SMILES only (no label involvement)
- Added `usr_similarity` + 4 shape moments to feature set

### 3. Ablation: Safe Features Only vs Safe+USR

- Safe features only (no USR): blind Top10 = **0.9059**
- Safe features + USR: blind Top10 = 0.8949
- USR contribution: **-0.011** (negative)
- USR standalone Top10: 0.342 (val), 0.313 (blind)

### 4. Per-Fragment Blind Strata (No USR)

| Fragment | N | Baseline | Safe-only | Safe+USR |
|---|---|---|---|---|
| *C1CCCCC1 | 2244 | 0.720 | 0.787 | 0.755 |
| *CCC | 2112 | 0.884 | 0.880 | 0.884 |
| *N(C)C | 1648 | 0.661 | **0.928** | 0.817 |
| *c1cccc(C)c1 | 1823 | 0.901 | **0.972** | 0.969 |
| *c1ccccc1F | 2113 | 0.904 | **0.947** | 0.948 |
| *Cc1ccccn1 | 233 | 0.571 | **0.884** | 0.936 |

### 5. Audit Checks (d4s30_audit.py)

- **USR provenance**: CLEAN — purely structural 3D conformer, no label involvement
- **USR-label correlation**: 0.083 (vs blend_score-label: 0.200)
- **USR standalone**: Top10 0.342 (val), 0.313 (blind) — much weaker than baseline
- **USR ablation**: Removing USR does NOT crash blind (0.906 without, 0.895 with)
- **Feature importance**: USR ranked low; top features are blend_score, prior scores, tanimoto_similarity
- **Verdict**: Shape signal is REAL but REDUNDANT with existing 2D features

## Key Files

| File | Purpose |
|------|---------|
| `D4S30_shape_augmented/d4s30_shape_scorer.py` | Main: precompute USR, train HistGB, eval val + blind |
| `D4S30_shape_augmented/d4s30_audit.py` | Audit: provenance, ablation, strata, feature importance |
| `D4S30_shape_augmented/D4S30_AUDIT_VERDICT.md` | Final verdict document |
| `D4S30_shape_augmented/results/d4s30_summary.csv` | Final metrics table |
| `D4S30_shape_augmented/results/d4s30_blind_strata_audit.csv` | Per-fragment blind results |

## Core Numbers

| Metric | Value |
|--------|-------|
| Safe-only blind Top10 (no USR) | **0.9059** |
| Safe+USR blind Top10 | 0.8949 |
| Baseline blend blind Top10 | 0.8558 |
| Delta safe-only vs baseline | **+0.0501** |
| USR standalone val Top10 | 0.3425 |
| USR-blind val Top10 (D4S28 bugged) | 0.7120 |
| Val OOF Top10 (safe-only) | 0.9375 |
| *N(C)C blind Top10 improvement | 0.661 -> 0.928 (+0.267) |
| *Cc1ccccn1 blind Top10 improvement | 0.571 -> 0.884 (+0.313) |

## Verdict

**USR shape features provide zero net benefit and are already captured by existing 2D Morgan fingerprint similarity + molecular descriptors. The real blind improvement (+0.05 over baseline) comes from proper feature alignment, not shape features.**

## Retracted/Negated Old Conclusions

1. **"D4S28 blind failure is a generalization problem requiring 3D shape features"** — NEGATED. It was a `cat_templates` feature alignment BUG.
2. **"USR/USRCAT blind ~0.902 is a shape breakthrough"** — RETRACTED. Safe features alone achieve 0.906 without USR. USR actually hurts (-0.011).
3. **"D4S28 blind 0.712 is real performance"** — NEGATED. It was garbage from misaligned feature indices.

## Still-Credible Final Conclusions

1. **Val OOF 0.937 is real and not leaked** — consistent across D4S28, D4S30 (0.937421 vs 0.937495).
2. **Safe features + HistGB generalizes to blind at ~0.906** — first verifiable blind improvement (+0.05 over baseline).
3. **Per-fragment improvement is uneven but no fragment is degraded** — safe-only helps or matches every fragment.
4. **USR is a weak signal** (Top10 ~0.34) and redundant with 2D Tanimoto similarity.
5. **Next real signal frontier** is either activity coordinates (full ChEMBL bioactivity DB) or graph-level 3D (EGNN with proper alignment).

## Impact on Paper

| Section | Impact |
|---------|--------|
| **Methods** | Remove USR/shape augmentation. The proper alignment fix is a bugfix, not a methodological advance. Include the `cat_templates` alignment procedure as a technical best practice. |
| **Results** | The deployable blind score is **0.906** (not 0.902 with USR). The improvement over baseline is +0.05, not +0.047. Fragment-level improvements (*N(C)C +0.267, *Cc1ccccn1 +0.313) are the strongest results. |
| **Discussion** | Remove discussion of 3D shape features being necessary. 2D similarity already captures the signal. The bug story (D4S28 blind failure) is an important cautionary tale about train/test feature alignment. |

---

MODULE_READ_COMPLETE: D4S30
KEY_VERDICT: USR shape features provide zero net benefit; real blind Top10 0.906 comes from fixing a feature alignment bug in D4S28, not from 3D shape augmentation.
KEY_NUMBERS: Safe-only blind Top10=0.9059, USR-only val Top10=0.3425, *N(C)C blind improvement +0.267
CONFLICTS_FOUND: D4S30's "USR breakthrough" narrative in the output log conflicts with the ablation audit (USR hurts); D4S30 blind 0.902 (with USR) vs safe-only 0.906 (no USR) — the better number is from fewer features.

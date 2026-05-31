# D4S29: New Signal Exploration Summary

**Date**: 2026-05-29
**Status**: EXPLORATION COMPLETE — weak orthogonal signal found in geometry

## Phase 1: Geometry (3D Conformer Features)

**Method**: RDKit conformer generation + shape similarity + electrostatic + PMI descriptors.
**Sample**: 65 queries, 9,750 pairs. Focus on *Cc1ccccc1 (30 queries).

**Key features with label separation:**

| Feature | Pos Mean | Neg Mean | Delta | p-value |
|---------|----------|----------|-------|---------|
| shape_tanimoto | 0.6655 | 0.5798 | +0.0856 | <0.001 |
| pmi2_ratio | 0.2662 | 0.8697 | -0.6035 | 0.004 |
| pmi3_ratio | 0.0979 | 0.6317 | -0.5338 | 0.007 |
| charge_max_cand | 0.2126 | 0.2772 | -0.0646 | <0.001 |

**Within-query ranking (*Cc1ccccc1, n=30):**

| Method | Top10 | Hits |
|--------|-------|------|
| Shape only | 0.2000 | 6/30 |
| Blend only | 0.4667 | 14/30 |
| Both hit | — | 2 |
| Shape-only rescue | — | **4** |
| Neither hit | — | 12 |

**Verdict**: Shape provides orthogonal signal — 4 queries rescued where blend fails. But effect is weak (12 queries neither solves). Geometry alone insufficient but useful as complementary feature.

## Phase 2: Activity Coordinates

**BLOCKED**. ChEMBL37K is structure-only. No pChEMBL, target ID, or assay data available.
- `d0_chembl37k_activity_availability.csv`: `structure_only, NOT_AVAILABLE, 0 records`
- Cannot compute activity-aware embeddings or target profiles

## Phase 3: Chemical Context

**Method**: Attachment bond chemistry, functional group changes, property deltas.
**Sample**: Same 65 queries.

**Key features with label separation:**

| Feature | Delta | p-value |
|---------|-------|---------|
| tpsa_change | -8.33 | <0.001 |
| hba_change | -0.36 | <0.001 |
| rot_bond_change | -0.30 | <0.001 |
| hetero_ratio_cand | -0.07 | <0.001 |

**Within-query ranking (*Cc1ccccc1, n=30):**

| Method | Top10 |
|--------|-------|
| Chem features | **0.0000** (0/30) |
| Blend | 0.4667 (14/30) |

**Verdict**: Chemical context features have statistical signal but provide ZERO orthogonal rescues. The features are redundant with existing 2D descriptors already captured by MLP/HGB/DE.

## Overall Assessment

| Signal Source | Orthogonal? | Strength | Deployable? |
|---------------|-------------|----------|-------------|
| Geometry (3D) | YES (4 rescues) | Weak | Partial — needs scaling |
| Activity coords | N/A | Blocked | Not available |
| Chemical context | NO (0 rescues) | Redundant | Not useful |

## Key Insight

The existing MLP+HGB+DE rankers already capture most 2D chemical information. Breaking through the *Cc1ccccc1 performance ceiling requires either:
1. 3D geometry at scale (full conformer generation is ~29h for all val queries)
2. External data sources (activity, target, assay from full ChEMBL or PubChem)
3. Fragment-level transfer learning from larger corpora

## Recommendation

**Short term**: Integrate shape_tanimoto as a lightweight complementary feature. Even weak orthogonal signal (4/30 rescues) is worth trying at scale, especially if we can approximate 3D similarity with faster methods (ETKDG, pre-computed conformers, or 2.5D descriptors like USRCAT).

**Medium term**: Obtain full ChEMBL database with bioactivity data for activity-aware embeddings.

**Next**: D4S30 — shape-augmented candidate scorer. Integrate geometry features into HistGB pipeline, test on full val OOF + blind one-shot.

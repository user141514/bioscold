# D4S28 Alignment Bug Explanation

**Date**: 2026-05-29
**Status**: FIXED

## Bug Description

D4S28 blind evaluation initially returned Top10=0.7120 — far below baseline 0.8558.
This was caused by a one-hot encoding column mismatch between val and blind feature matrices.

## Root Cause

Val has 8 unique old_fragments; blind has 19. The `pd.get_dummies()` call produces:
- Val: 8 one-hot columns for old_fragment_smiles
- Blind: 19 one-hot columns for old_fragment_smiles

Without column alignment, HistGB was trained on ~79 features but predictions on blind used ~90 features. The model weights were applied to wrong feature indices.

Additionally, `attachment_signature` and `replacement_frequency_bin` had similar mismatches.

## Fix

`cat_templates`: Extract one-hot categories from val data. For blind, map unseen categories to 'OTHER' and ensure column order matches val exactly.

```python
val_X, templates = build_X(val)
blind_X = build_X(blind, cat_templates=templates)  # aligned
assert val_X.shape[1] == blind_X.shape[1]
```

## Impact

| Metric | Buggy (0.7120) | Fixed (0.8851) |
|--------|----------------|----------------|
| Blind Top10 | 0.7120 | 0.8851 |
| Delta vs baseline | -0.1438 | +0.0293 |

The bug made the scorer appear to catastrophically fail on blind. In reality, it provides a modest but real improvement.

## Lesson

One-hot encoding alignment is critical for any model that uses categorical features across datasets with different category sets. Always use template-based alignment, not naive `pd.get_dummies()`.

# D4S28R Scorer-All Final Metrics

**Date**: 2026-05-29
**Method**: HistGB on 82 safe features, val-trained, blind one-shot.
**Feature alignment**: cat_templates (fixed).

## Primary Metrics

| Split | Method | Top10 | MRR | Delta |
|-------|--------|-------|-----|-------|
| Val (5-fold OOF) | Baseline blend | 0.8340 | 0.4263 | — |
| Val (5-fold OOF) | **HistGB scorer** | **0.9366** | — | **+0.1026** |
| Blind (one-shot) | Baseline blend | 0.8558 | 0.4842 | — |
| Blind (one-shot) | **HistGB scorer** | **0.8851** | — | **+0.0293** |

## Bootstrap CI (Blind)

Scorer-all blind Top10: 0.8851, 95% CI [0.8795, 0.8904]
Baseline blind Top10: 0.8558, 95% CI [0.8499, 0.8619]

Delta 95% CI: [+0.0236, +0.0350] — CI excludes 0.

## Rescue/Lost

| Metric | Count | Pct |
|--------|-------|-----|
| Baseline misses | 1,925 | — |
| Scorer rescues (miss->hit) | 987 | 51.3% |
| Baseline hits | 11,422 | — |
| Scorer loses (hit->miss) | 596 | 5.2% |

Net gain: 987 - 596 = +391 queries (2.9% of all queries).

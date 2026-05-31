# Results Required Tables

## Table 1: Full Method Ranking (Primary)
All methods on secondary blind, ranked by Top-10.
- Attachment-Frequency, HGB, DE, Borda(DE,HGB), MLP(rank-only), Score Blend
- D4S28R (82-feature scorer), D4S31 (77-feature scorer)
- Oracle(DE,HGB) diagnostic
- With 95% bootstrap CIs

## Table 2: Leave-One-Family-Out Ablation
- Full 82-feature (D4S28R) = 0.8851 baseline
- Drop prior_ranks (D4S31) = 0.9243 (+0.0392)
- Drop prior_positives, query_stats, model_ranks, model_scores
- Remaining families: negative deltas −0.003 to −0.027

## Table 3: Rescue/Lost Analysis
- 82-feature: 987 rescued, 596 lost, +391 net
- 77-feature: 424 rescued, 6 lost, +418 net
- Relative to best-base diagnostic

## Table 4: A4C Alert Stratification
- G4 (shared): 0.99%
- G3 (DE-elevated): 9.67%
- G2 (Borda-emergent): 46.85%

## Supplementary
- S1: Full feature definitions (77 features + 5 removed)
- S2: MRR values for all methods
- S3: Per-fragment Top-10 breakdown (19 blind fragments)
- S4: Canonical analysis results

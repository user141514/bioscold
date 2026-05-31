# Final Method Status Table

| Method ID | Description | Blind Top-10 | Status | Notes |
|-----------|-------------|-------------|--------|-------|
| AttFreq | Attachment-frequency baseline | 0.6019 | Baseline | Lower bound; no learned parameters |
| HGB | HistGB feature-based ranker | 0.7437 | Baseline | Conservative Mode source |
| DE | Dual Encoder similarity ranker | 0.8055 | Baseline | Learned structural compatibility |
| Borda(DE,HGB) | Parameter-free rank fusion | 0.8384 | Baseline | DE-HGB complementarity evidence |
| MLP(rank-only) | Learned rank aggregator | 0.8402 | Baseline | Marginal Top-10 gain; MRR diagnostic |
| Score Blend | MLP + HGB-refit (alpha=0.95) | 0.8558 | Strongest pre-D4S baseline | Primary comparison target |
| D4S27 | Conditional prior / query-level gate | -- | Rejected | Routing unreliable; val-blind transfer negative |
| D4S28 (bugged) | Candidate-level scorer, no alignment | 0.7120 | Invalid | One-hot schema mismatch bug |
| D4S28R | Aligned 82-feature scorer | 0.8851 | Superseded | 596 lost hits vs ScoreBlend; five old-fragment and one ATT:C\|S degradation strata |
| D4S30 | USR shape features | -- | Rejected | USR contributed no value |
| **D4S31** | **77-feature scorer (no prior_ranks)** | **0.9243** | **Final locked** | Post-audit; lost=101 vs ScoreBlend by query_id merge |
| D4S32 | Query router | AUC approximately 0.61 | Rejected | Routed Top-10 below scorer_all / deployment rejected |
| D4A2/A4C | Computational review proxy | -- | Workflow layer | G2/G3 covered alert rates: 46.85%/9.67%; G4 0.99% total lower bound with 5.63% coverage |
| Oracle(DE,HGB) | Diagnostic bound | 0.8686 | Reference | Not a ceiling for feature-augmented models |

## Rejected Methods

- D4S27: query-level gate; unreliable routing.
- D4S30: USR shape features; no value.
- D4S32: query router; AUC too low for reliable deployment.

## Current Best Deployable

D4S31: 77-feature HistGB scorer, Top-10 = 0.9243, lost = 101 vs ScoreBlend by query_id merge.

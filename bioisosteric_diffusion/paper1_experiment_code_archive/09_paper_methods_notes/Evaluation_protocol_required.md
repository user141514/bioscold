# Evaluation Protocol

## Metrics
- Primary: Top-10 accuracy (query-level)
- Secondary: Top-1, Top-5, Top-20, Top-50, MRR
- Multi-positive: hit if ANY positive in top-K

## Bootstrap
- Nonparametric, B=5,000 replicates
- Query-level resampling (all candidates of sampled query retained)
- Paired delta bootstrap for pairwise comparisons
- CI_95% = [θ*(0.025), θ*(0.975)]

## Rescue/Lost Analysis
- Rescue: baseline r_i+ > 10 AND model r_i+ ≤ 10
- Lost: baseline r_i+ ≤ 10 AND model r_i+ > 10
- Net gain G = R − L

## Protocol Separation
| Protocol | Queries | Purpose |
|----------|---------|---------|
| Development/calibration | Held-out train | Architecture/hyperparameter selection |
| Secondary blind | 13,347 | Primary performance claim |
| Canonical analysis | 21,052 | Robustness/mechanism only |
| Dual-mode (§3.7) | -- | Workflow interpretation only |

## Leakage Controls
1. Transform-heldout: zero (f_old,σ) overlap
2. Feature-level: train-only statistics
3. Categorical: frozen schema alignment
4. Overlap: verified zero all pairs
5. Label-free: 77 features, no ground-truth dependence

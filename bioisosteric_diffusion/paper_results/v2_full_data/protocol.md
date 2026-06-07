# V2 Full-Data Experiment Protocol

Date: 2026-06-06
Data: 123 OFs from all 301 label shards (247 train + 54 test)

## Primary Metric

OF-macro Hit@10 over outer-test OFs.

## Delta Definition

Δ = LBC-LR - max(Frequency, CA-tuned, C3F-tuned)
HGB excluded from comparator (same feature family, high-capacity control).

## Claim Thresholds

| Tier | Δmacro | Δquery | Seed wins |
|------|--------|--------|-----------|
| Strong | ≥ +0.15 | ≥ +0.05 | ≥ 9/10 |
| Medium | ≥ +0.10 | ≥ 0 | — |
| Weak | < +0.10 | — | — |

## Tuning Grids

### CA-tuned
λ ∈ {0, 0.25, 0.5, 0.75, 1.0}
score = λ × Morgan + (1 − λ) × PhysChem

### C3F-tuned
K ∈ {3, 5, 7, 10}
α ∈ {0.1, 0.2, 0.3, 0.5, 0.7}

### HGB-tuned
max_depth ∈ {2, 3, 5}
max_iter ∈ {100, 200}
learning_rate ∈ {0.05, 0.1}
l2_regularization = 0.1

### LBC-LR
C = 1.0, L2 penalty, class_weight = None, full 8-feature set.

## Inner CV

3-fold GroupKFold over OFs within outer-train. Seed = outer_seed.
Selection by mean inner OF-macro Hit@10.
Tiebreaker (within 0.005): lower complexity → smaller K → lower α → lower max_depth → lower max_iter → deterministic order.

## Outer Evaluation

10 seeds, 70/30 OF split.
For each seed: tune on inner 3-fold, retrain best config on full outer-train, evaluate on outer-test.
Ablation: 8 one-drop models, same outer split, no inner tuning (use same LR C=1.0).

## HGB vs LR Branch

| Gap | Narrative |
|-----|-----------|
| LR ≥ HGB or |gap| < 0.01 | LR is primary; HGB is capacity control |
| HGB > LR by 0.01–0.03 | LR is interpretable implementation; HGB is performance upper bound |
| HGB > LR by >0.03 | Primary = learned compatibility family; LR is interpretable readout |

# D4S31: Feature Diff vs D4S28R

**Removed (5 features):**
- backoff_logit_rank
- cont_prior_rank
- t_p4_logit_rank
- p1_logit_rank
- p3_logit_rank

**Kept (77 features):** All other D4S28R safe features unchanged.

## Why These 5 Kill Generalization

1. Per-query ranks are computed WITHIN each query's candidate set
2. Val has 8 old_fragments → rank distributions reflect val fragment biases
3. Blind has 19 different old_fragments → rank distributions are completely different
4. HistGB learns to use rank thresholds that work on val but fail on blind
5. Removing them forces the model to use only cross-query-stable features (scores, rates, supports)

## Design Rule

**Do not use per-query rank features in cross-split generalization.** Ranks encode query-local ordering that does not transfer across different fragment distributions. Use raw scores and rates instead.

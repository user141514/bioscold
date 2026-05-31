# D4S31 Candidate-Level Scorer Pipeline

## Status
Final locked model. Post-audit feature-regularized (not fully prospective pre-registered).

## Architecture
- HistGradientBoostingClassifier (sklearn)
- 200 iterations, max_depth=6, lr=0.1, class_weight='balanced'
- 5-fold GroupKFold (by query_id)
- Negative subsampling 5:1
- Hyperparameters fixed on development set

## Feature Set: 77 dimensions in 9 families

| Family | Count | Source |
|--------|-------|--------|
| F1: Model Scores | 5 | z-normalized base ranker outputs |
| F2: Model Raw Scores | 4 | Raw base ranker outputs |
| F3: Model Ranks | 6 | Per-query ranks from base rankers |
| F4: Top-K Flags | 4 | Binary top-10 membership |
| F5: Prior Scores | 5 | Logit-transformed P0/P1/P3/P4 |
| F6: Prior Statistics | 12 | Smoothed rates, support/positive counts |
| F7: PMI Contrast Scores | 6 | Pointwise mutual information across prior levels |
| F8: Molecular Descriptors | 10 | RDKit properties + deltas |
| F9: Similarity & Frequency | 3 | Tanimoto, global/attachment frequency |
| Categorical (one-hot) | 22 | Fragment ID, attachment sig, frequency bin |

## Removed Features
Prior_Ranks (5): backoff_logit_rank, cont_prior_rank, t_p4_logit_rank, p1_logit_rank, p3_logit_rank
- Removed post-audit: identified as shortcut learning hazard
- Per-query rank of prior scores → non-transferable across fragment distributions

## Key Implementation Detail
Frozen categorical schema alignment (cat_templates)
- Fit one-hot encoder on development set
- Unseen categories → OTHER
- Enforce identical column order via shape assertion
- Without: 0.7120; With: 0.8851

## Training Protocol
1. Train on full development set (one-shot)
2. Apply once to secondary blind
3. No blind labels used for feature values or model parameters
4. Feature standardization from development set statistics

## Final Performance
- Blind Top-10: 0.9243 [0.9198, 0.9289]
- Δ over Score Blend: +0.0686 [0.0633, 0.0739]
- Zero fragments with significant degradation (vs 6/19 for 82-feature)

# 3.4 Candidate-Level Scorer -- Spec

## Coverage
Describes the primary methodological contribution: 77-feature candidate-level HistGB scorer. Covers motivation (gap beyond base rankers), architecture overview, 9 feature families in detail, HistGB model configuration and training protocol (5-fold GroupKFold, one-shot blind evaluation), and cat_templates feature alignment.

## Key Equations / Notation

Feature vector:
$$
\Phi(q_i, c) \in \mathbb{R}^{77}
$$

Scorer output:
$$
\widehat{p}_{ic} = F_\Theta(\Phi(q_i, c)), \quad 0 \le \widehat{p}_{ic} \le 1
$$

Weighted binary log loss:
$$
\mathcal{L}_{\text{HistGB}}(\Theta) = -\sum_i \sum_{c\in\mathcal{C}_i} w_{ic}[y_{ic}\log\widehat{p}_{ic} + (1-y_{ic})\log(1-\widehat{p}_{ic})]
$$

## Feature Family Summary Table

| Family | Count | Key Features |
|--------|-------|-------------|
| F1: Model Scores | 5 | blend_score, mlp_z, hgb_z, mlp_score, hgb_refit_score |
| F2: Model Raw Scores | 4 | score_DE, score_HGB, score_Borda, score_attach |
| F3: Model Ranks | 6 | mlp_rank, blend_rank, rank_DE, rank_HGB, rank_Borda, rank_attach |
| F4: Top-K Flags | 4 | candidate_in_{attach,DE,HGB,Borda}_top10 |
| F5: Prior Scores | 5 | cont_prior_score, backoff_logit_score, t_p4_logit_score, p1_logit_score, p3_logit_score |
| F6: Prior Statistics | 12 | smoothed rates, support counts, positive counts for t_p4, p1, p3, backoff |
| F7: PMI Scores & Ranks | 6 | t_pmi_p4_p1, pmi_p3_p1, pmi_p1_p0 (scores + ranks) |
| F8: Molecular Descriptors | 10 | delta_heavy_atoms, delta_ring_count, delta_hetero_count, delta_mw, delta_logp, delta_tpsa, candidate_heavy_atoms, candidate_mw, candidate_logp, candidate_tpsa |
| F9: Similarity & Freq | 3 | tanimoto_similarity, replacement_frequency, attachment_frequency |
| Categorical (one-hot) | ~19 | old_fragment_smiles, attachment_signature, replacement_frequency_bin |

HistGB hyperparameters: max_iter=200, max_depth=6, lr=0.1, class_weight=balanced, neg_subsample=5:1.

## Figures / Tables Needed
- **Table 3**: Feature families summary (the 9 families + categorical, with counts and chemical rationale)
- **Supplementary Table S1**: Complete per-feature definitions (deferred -- not in main text)
- **Figure**: Feature computation pipeline diagram (optional)

## Dependencies
- Requires base ranker definitions from 3.3 (F1-F4 feature families)
- Requires provenance level definitions (P0, P1, P3, P4) -- defined in Section 2.2
- Trained model is analyzed in 3.5 (ablation)
- Evaluation results go in Section 4

## What Needs Updating from Existing Draft

### Issues (full draft lines 308-396, unified draft lines 190-255)
1. **82-feature vs. 77-feature framing**: The section describes the 77-feature model but the ablation (3.5) starts from 82 features. The narrative flow is: "we designed 82 features; ablation showed 5 harm generalization; final model uses 77." This section should present the *final* 77-feature set but acknowledge the 82-feature starting point. The unified draft handles this well by describing the 77 features and noting 5 were removed.
2. **Feature family table**: Both drafts have this table. Ensure counts match the feature schema (77 total = 5+4+6+4+5+12+6+10+3+~19). The unified draft Table 3 has clearer chemical rationale column. Keep that.
3. **Model-derived features (F1-F4)**: Good prose. However, note that F3 (model ranks) includes 6 rank features from base rankers, but the ablation removes *prior rank* features (different from model ranks). Clarify: F3 ranks are from base ranker *outputs*; prior_ranks (removed in 3.5) are from *prior scores*. These are different families.
4. **Prior features (F5-F7)**: The provenance levels (P0, P1, P3, P4) need to be defined. They are referenced from Section 2.2. If Section 2.2 does not exist yet, define them compactly here.
5. **Training protocol**: 5-fold GroupKFold by query_id. Negative subsampling 5:1. One-shot blind evaluation. All well-described in both drafts. Unify the language -- the unified draft is slightly cleaner.
6. **cat_templates section (3.4.4)**: Good -- this is an important practical contribution. Keep the dramatic "without: 0.7120, with: 0.8851" comparison.
7. **Feature count verification**: Both drafts say "77" but the unified draft Table 3 lists counts that add to different totals depending on one-hot encoding. Provide a precise breakdown: "77 non-constant features plus variable-dimension one-hot encoding (typically 19 columns) for a total of ~96 model inputs."
8. **HistGB justification**: Both drafts give 3 reasons (heterogeneous features, nonlinear interactions, built-in importance). Keep all three.

### Final Output Requirements
- ~1500-1800 words (second-longest section)
- Complete feature family table
- Training protocol clearly specified for reproducibility

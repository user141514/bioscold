# D4S28R Feature Schema Lock

**Date**: 2026-05-29
**Frozen schema**: 82 features (69 numeric + 13 one-hot)

## Numeric Features (69)

### Model Scores (5)
blend_score, mlp_z, hgb_z, mlp_score, hgb_refit_score

### Model Raw Scores (4)
score_DE, score_HGB, score_Borda, score_attach

### Model Ranks (6)
mlp_rank, blend_rank, rank_DE, rank_HGB, rank_Borda, rank_attach

### Top-K Flags (4)
candidate_in_attach_top10, candidate_in_DE_top10, candidate_in_HGB_top10, candidate_in_Borda_top10

### Prior Scores (5)
cont_prior_score, backoff_logit_score, t_p4_logit_score, p1_logit_score, p3_logit_score

### Prior Smoothed Rates (4)
t_p4_smoothed_rate, p1_smoothed_rate, p3_smoothed_rate, backoff_smoothed_rate

### Prior Support Counts (5)
t_p4_support, p1_support, p3_support, p0_support, backoff_support

### Prior Positive Counts (3)
t_p4_positive, p1_positive, p3_positive

### Prior Ranks (5)
backoff_logit_rank, cont_prior_rank, t_p4_logit_rank, p1_logit_rank, p3_logit_rank

### PMI Scores (3)
t_pmi_p4_p1, pmi_p3_p1, pmi_p1_p0

### PMI Ranks (3)
t_pmi_p4_p1_rank, pmi_p3_p1_rank, pmi_p1_p0_rank

### Molecular Descriptors (10)
delta_heavy_atoms, delta_ring_count, delta_hetero_count, delta_mw, delta_logp, delta_tpsa
candidate_heavy_atoms, candidate_mw, candidate_logp, candidate_tpsa

### Similarity & Frequency (3)
_tanimoto_similarity, replacement_frequency, attachment_frequency

### Query-Level Prior Stats (3)
query_prior_entropy, query_prior_margin, query_prior_max_support

## Categorical Features (3 -> 13 one-hot)

- old_fragment_smiles: 8 val categories + OTHER
- attachment_signature: 5 val categories + OTHER
- replacement_frequency_bin: 3 val categories + OTHER

## Excluded Features

UNSAFE_LABEL_DERIVED: label, positive_set_size, num_positives, single_pos_flag, multi_pos_flag, hard_top10_miss_flag

SUSPICIOUS: replacement_frequency_mean, target_any_seen_vocab, target_all_seen_vocab, morgan_similarity_old_candidate_if_available, compatibility_flag, a4c_tier_if_available_diagnostic_only

METADATA: query_id, split, candidate_id, candidate_smiles, old_fragment_cluster_*

USR_FEATURES: usr_similarity, usr_shape_* (REJECTED by D4S30)

## Model Hyperparameters

- Model: sklearn HistGradientBoostingClassifier
- max_iter: 200, max_depth: 6, learning_rate: 0.1
- class_weight: balanced
- Negative subsampling: 5:1 ratio
- random_state: 42

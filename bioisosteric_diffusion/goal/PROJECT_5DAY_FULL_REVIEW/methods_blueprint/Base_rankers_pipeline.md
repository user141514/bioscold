# Base Rankers Pipeline

## Methods (5 total)

### 1. Attachment-Frequency
- Score: p̂_att(c|σ) = n_train(c,σ) / Σ n_train(c',σ)
- No learned parameters
- Blind Top-10: 0.6019

### 2. Dual Encoder (DE)
- Query encoder: Morgan FP(f_i^old) + learned σ embedding → MLP(d=128) → h_q
- Candidate encoder: Morgan FP(c) → MLP(d=128) → h_c
- Score: cosine(h_q, h_c)
- Training: margin ranking loss, K=20 negatives, δ=0.3
- Blind Top-10: 0.8055

### 3. HistGB Ranker (HGB)
- 4 feature groups: frequency, attachment sig, molecular descriptors, fingerprint similarity
- HistGradientBoostingClassifier: 200 iter, depth 6, lr 0.1
- Interpretable → Conservative Mode source
- Blind Top-10: 0.7437

### 4. Borda Fusion (DE + HGB)
- S_Borda(i,c) = Σ_m (|C_i| + 1 - ρ_m(i,c))
- Parameter-free (transform-heldout prevents validation tuning)
- Blind Top-10: 0.8384

### 5. Score Blend (MLP + HGB)
- Rank-only MLP(d=32) on DE/HGB/AttFreq ranks
- z-score normalization within query
- s_blend = 0.95·z(MLP) + 0.05·z(HGB-refit)
- Blind Top-10: 0.8558 (strongest baseline)

## Oracle(DE,HGB) Diagnostic
- 0.8686: fraction of queries where DE or HGB places correct candidate in top-10
- Diagnostic upper bound for per-query DE/HGB selection
- NOT a ceiling for feature-augmented models

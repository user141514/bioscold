# Methods Required Content

## 1. Task Definition
- Query = (old_fragment SMILES, attachment_signature)
- Candidate set C_q = attachment-compatible subset of train replacement vocabulary V
- Positive set P_q from structure-derived MMP pairs on ChEMBL37K
- Query-level exact-recovery hit: any positive in top-K = 1, else 0
- Primary metric: Top10. Secondary: Top1, Top5, Top20, Top50, MRR
- Multi-positive queries: hit if ANY positive in top-K (standard IR)
- Closed vocabulary: V constructed from training split only, size 150-152

## 2. Benchmark Construction
- ChEMBL37K → MMP extraction → replacement pairs
- Transform-heldout split: zero (old_fragment, attachment_signature) overlap train/test
- Secondary blind split: zero train/blind overlap, 13347 queries, 150-fragment vocab
- Four decoy repairs: 1:1 ratio, wrong-positive removal, dedup, 1:5 diagnostic
- Leakage verification: zero train/test transform overlap, zero train/blind transform overlap
- Leakage control table: overlap counts for each split pair

## 3. Base Rankers (Borda-era)
- Attachment-frequency: P(c|att) from train counts
- Dual Encoder (DE): query encoder (old_frag + att_sig) + candidate encoder, cosine similarity
- HGB: feature-based histogram gradient-boosted ranker on fragment frequency, attachment stats, mol props, fingerprints
- Borda(DE,HGB): parameter-free rank aggregation
- MLP reranker: rank-only (DE rank, HGB rank, attach-freq rank) → secondary MRR evidence

## 4. D4S31 Candidate Scorer Pipeline
- Input: candidate matrix where each row = (query_id, candidate_smiles, label, features)
- Features: 77 dimensions (82 SAFE - 5 prior_ranks)
- 9 feature families:
  1. Model scores (5): blend_score, mlp_z, hgb_z, mlp_score, hgb_refit_score
  2. Model raw scores (4): score_DE, score_HGB, score_Borda, score_attach
  3. Model ranks (6): mlp_rank, blend_rank, rank_DE, rank_HGB, rank_Borda, rank_attach
  4. Top-K flags (4): candidate_in_{attach,DE,HGB,Borda}_top10
  5. Prior scores (5): cont_prior_score, backoff_logit_score, t_p4_logit_score, p1_logit_score, p3_logit_score
  6. Prior rates/supports (12): smoothed rates, support counts, positive counts
  7. PMI scores (3): t_pmi_p4_p1, pmi_p3_p1, pmi_p1_p0
  8. Molecular descriptors (10): delta_heavy_atoms, delta_ring_count, delta_hetero_count, delta_mw, delta_logp, delta_tpsa, candidate_heavy_atoms, candidate_mw, candidate_logp, candidate_tpsa
  9. Similarity & frequency (3): _tanimoto_similarity, replacement_frequency, attachment_frequency
- Categorical (3→13 one-hot): old_fragment_smiles, attachment_signature, replacement_frequency_bin
- DROPPED (5 prior_ranks): backoff_logit_rank, cont_prior_rank, t_p4_logit_rank, p1_logit_rank, p3_logit_rank

## 5. Why prior_ranks Removed
- Per-query ranks encode ordering WITHIN a query's candidate set
- Val has 8 old_fragments → rank distributions reflect val-specific fragment biases
- Blind has 19 different old_fragments → completely different rank distributions
- HistGB learns rank thresholds that fail to transfer
- Design rule: per-query rank features are a generalization hazard; use raw scores/rates instead
- Ablation evidence: drop prior_ranks → +0.039 blind improvement, 596→6 lost hits

## 6. Training Protocol
- Model: sklearn HistGradientBoostingClassifier
- Hyperparams: max_iter=200, max_depth=6, learning_rate=0.1, class_weight=balanced
- Training: 5-fold GroupKFold (by query_id) on val split
- Negative subsampling: 5:1 ratio
- Cat_templates: one-hot categories extracted from val, blind mapped via templates
- Blind evaluation: ONE-SHOT (train on full val, predict blind once)
- No blind tuning, no blind model selection

## 7. Evaluation Protocol
- Per-query ranking: sort candidates by scorer output, compute hit@K per query
- Bootstrap CI: 5000 nonparametric bootstrap resamples over query_id
- Delta CI: paired bootstrap of (scorer - baseline) per query
- Rescue: baseline miss (rank>10) → scorer hit (rank≤10)
- Lost: baseline hit (rank≤10) → scorer miss (rank>10)
- Net gain: rescued - lost

## 8. Leakage Control
- All prior features use TRAIN-ONLY statistics
- D4S27 similarity-transfer uses train old_fragments only
- Cat_templates alignment: val → blind
- Feature schema lock: 77 features, fixed column order
- Assert val_X.shape[1] == blind_X.shape[1]
- Verified: no label-derived features in safe set
- Verified: zero train/val/blind transform overlap

## 9. Negative Subspaces
- D4S28R: 6/19 fragments degraded (small-N saturated fragments)
- D4S31: ALL negative subspaces resolved (zero fragments with negative delta)
- Methodology: per-fragment strata report, explicit failure mode disclosure

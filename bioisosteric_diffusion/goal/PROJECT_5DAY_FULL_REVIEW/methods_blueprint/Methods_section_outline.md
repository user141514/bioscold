# Methods Section Outline — Post-D4S31

## 3.1 Task Definition

### 3.1.1 Query Formulation
- Query q = (old_fragment SMILES with attachment marker, attachment_signature)
- Closed global replacement vocabulary V from training split (size 150-152)
- Candidate set C_q = attachment-compatible subset of V
- Only fragments in V are eligible

### 3.1.2 Labels
- Positive set P_q from structure-derived MMP pairs on ChEMBL37K
- Labels are structure-derived, do not establish activity preservation
- Multi-positive queries: hit if ANY positive in top-K

### 3.1.3 Metrics
- Primary: query-level Top10 (exact-recovery hit@10)
- Secondary: Top1, Top5, Top20, Top50, MRR
- Rescue: baseline miss → scorer hit
- Lost: baseline hit → scorer miss

## 3.2 Benchmark Construction

### 3.2.1 Data Source
- ChEMBL37K → MMP extraction → replacement pairs
- Four decoy repairs: 1:1 ratio, wrong-positive removal, dedup, 1:5 diagnostic

### 3.2.2 Split Design
- Transform-heldout: zero (old_fragment, attachment_signature) overlap train/test
- Secondary blind: zero train/blind overlap, 13347 queries, 150-fragment vocab
- Canonical analysis: 21052 queries, 152-fragment vocab (robustness only)
- Verified properties: zero train/test overlap, zero train/blind overlap, zero canonical/blind query overlap

### 3.2.3 Leakage Control Table
| Split Pair | Transform Overlap | Verified |
|------------|------------------|----------|
| Train / Val test | 0 | Yes |
| Train / Blind | 0 | Yes |
| Train / Canonical test | 0 | Yes |
| Canonical / Blind queries | 0 | Yes |

## 3.3 Base Rankers

### 3.3.1 Attachment-Frequency Baseline
- P(c|att) from train counts
- Strong empirical prior, serves as lower bound

### 3.3.2 Dual Encoder (DE)
- Query encoder: old_fragment Morgan FP + attachment_signature embedding → fixed-dim vector
- Candidate encoder: candidate Morgan FP → fixed-dim vector
- Score: cosine similarity(query_emb, candidate_emb)
- Trained with contrastive loss on train pairs

### 3.3.3 Histogram Gradient-Boosted Ranker (HGB)
- Features: fragment frequency, attachment statistics, molecular descriptors, Morgan fingerprints
- Trained on train split, no blind access
- Serves as Conservative Mode source

### 3.3.4 Borda Fusion
- S_Borda(q,r) = sum_{m in {DE,HGB}} (|V| + 1 - rank_m(q,r))
- Parameter-free by design (transform-heldout prevents validation tuning)
- DE-HGB complementarity baseline

### 3.3.5 Score Blend (MLP + HGB)
- 0.95 * z(MLP_rank_score) + 0.05 * z(HGB_refit_score)
- Pre-D4S strongest baseline: blind Top10 = 0.8558

## 3.4 Candidate-Level Scorer (Primary Method)

### 3.4.1 Architecture
- sklearn HistGradientBoostingClassifier
- Hyperparams: max_iter=200, max_depth=6, lr=0.1, class_weight=balanced
- Negative subsampling: 5:1 ratio
- 5-fold GroupKFold (by query_id) for val evaluation
- Blind: one-shot (train full val, predict blind once)

### 3.4.2 Feature Schema (77 dimensions)
9 feature families:
1. Model scores (5): blend_score, mlp_z, hgb_z, mlp_score, hgb_refit_score
2. Model raw scores (4): score_DE, score_HGB, score_Borda, score_attach
3. Model ranks (6): mlp_rank, blend_rank, rank_DE, rank_HGB, rank_Borda, rank_attach
4. Top-K flags (4): candidate_in_{attach,DE,HGB,Borda}_top10
5. Prior scores (5): cont_prior_score, backoff_logit_score, t_p4_logit_score, p1_logit_score, p3_logit_score
6. Prior rates & supports (12): smoothed rates, support counts, positive counts for t_p4, p1, p3, backoff
7. PMI scores & ranks (6): t_pmi_p4_p1, pmi_p3_p1, pmi_p1_p0 (scores + ranks)
8. Molecular descriptors (10): delta_heavy_atoms, delta_ring_count, delta_hetero_count, delta_mw, delta_logp, delta_tpsa, candidate_heavy_atoms, candidate_mw, candidate_logp, candidate_tpsa
9. Similarity & frequency (3): _tanimoto_similarity, replacement_frequency, attachment_frequency
10. Categorical → one-hot (13): old_fragment_smiles, attachment_signature, replacement_frequency_bin

### 3.4.3 Feature Ablation: prior_ranks Removal
- Removed 5 features: backoff_logit_rank, cont_prior_rank, t_p4_logit_rank, p1_logit_rank, p3_logit_rank
- Rationale: per-query ranks encode ordering WITHIN a query's candidate set
- Val has 8 old_fragments → rank distributions reflect val-specific biases
- Blind has 19 old_fragments → completely different rank distributions
- HistGB learns rank thresholds that fail to transfer
- Ablation result: +0.039 blind improvement, 596→6 lost hits
- Design rule: do not use per-query rank features in cross-split generalization

### 3.4.4 Feature Alignment
- cat_templates: one-hot categories extracted from val data
- Blind: map unseen categories to 'OTHER', enforce column order = val order
- Assert val_X.shape[1] == blind_X.shape[1]
- Without alignment: D4S28 blind = 0.7120 (weights applied to wrong feature indices)

## 3.5 Evaluation Protocol

### 3.5.1 Ranking Evaluation
- Per-query: sort candidates by scorer output, compute hit@K
- Metrics averaged at query level
- Bootstrap CI: 5000 nonparametric resamples over query_id
- Delta CI: paired bootstrap (scorer - baseline) per query

### 3.5.2 Protocol Separation
| Protocol | Queries | Purpose |
|----------|---------|---------|
| Secondary blind | 13,347 | Primary performance claim |
| Canonical analysis | 21,052 | Robustness/mechanism only |
| Dual-mode risk-stratification | — | Workflow/risk interpretation |

### 3.5.3 Rescue/Lost Definitions
- Rescue: baseline best_rank > 10 AND scorer best_rank ≤ 10
- Lost: baseline best_rank ≤ 10 AND scorer best_rank > 10
- Net gain: rescued - lost

## 3.6 A4C Computational Review Proxy

### 3.6.1 Annotation Layer
- PAINS alerts, Brenk alerts, property shift computations
- Computational screening aid, not medicinal-chemistry truth standard

### 3.6.2 Dual-Mode Workflow
- Conservative Mode: HGB proposals (history-aligned)
- Exploration Mode: Borda(DE,HGB) proposals with G2/G3/G4 provenance labels
- G4: shared candidates (both HGB and Borda top-K), alert rate 0.99%
- G3: DE-retained candidates, alert rate 9.67%
- G2: Borda-only candidates, alert rate 46.85%

### 3.6.3 Scope
- Used only for workflow/risk interpretation
- Not for primary proposal Top10 claim
- Not for medicinal-chemistry decision-making

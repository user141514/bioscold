# LBC-Ranker: Manuscript V2 Outline + Evidence Map

Target: JCIM (Journal of Chemical Information and Modeling)
Date: 2026-06-07
Protocol: `paper_results/v2_full_data/protocol.md`
Data: 123 OFs, 10-seed repeated OF split, inner 3-fold CV for tuning

---

## 1. Abstract

### Claim stack
1. Fragment replacement Top10 ranking requires balancing replacement track record with substructure match
2. LBC-Ranker: compact supervised ranker (LR, 8 features) combining train-only candidate support + binary-fingerprint compatibility
3. Across 123 old fragments, 10 repeated OF-level splits: Δ macro Hit@10 +0.130 over best tuned non-LBC baseline
4. Ablation: freq (-0.193) > bit_corr (-0.155) > morgan (-0.109)
5. LR ≈ HGB — feature representation matters more than model capacity
6. C3F viable at sufficient OF coverage (0.72) but below LBC

### Evidence
- `e1_main_10seed_summary.csv`: LBC 0.8521, C3F 0.7163, CA 0.6614, Freq 0.4624
- `e1_main_paired_tests.csv`: p=0.00195, all diffs positive
- `e2_ablation_full_summary.csv`: freq -0.193, bit_corr -0.155, morgan -0.109
- `e6_hgb_vs_lr.csv`: LR 0.8521, HGB 0.8505, Δ -0.0016

---

## 2. Introduction

### Para 1: Problem
- Bioisosteric replacement = fundamental lead optimization strategy
- Ranking candidate fragments for a query OF is core computational task
- Practical challenge: most OFs have few labeled positives; candidate pool large

### Para 2: Existing approaches
- Fingerprint similarity (Morgan Tanimoto) — treats all bits equally
- Physicochemical filtering — necessary but insufficient for bioisosterism
- Collaborative filtering (C3F, Humbeck et al. 2017) — relies on cross-OF co-occurrence, requires sufficient OF coverage
- Hand-crafted scoring (Content-Aware) — fixed weights, cannot adapt to data

### Para 3: Gap
- No existing method learns the balance between candidate replacement track record and structural compatibility
- Retrieval-based methods ignore compatibility features; similarity methods ignore candidate priors
- Full-model evaluation under OF-level protocols is lacking

### Para 4: Our approach
- LBC-Ranker: logistic regression with 8 features
- Three design principles:
  1. Train-only candidate frequency as replacement prior
  2. Bit-level fingerprint correlation (bit_corr) as standardized structural readout
  3. Learned feature combination rather than hand-tuned weights

### Para 5: Contributions
1. LBC-Ranker: compact supervised ranking model for fragment replacement
2. OF-level evaluation protocol with tuned baselines (CA, C3F, HGB)
3. Evidence that learned compatibility scoring improves over both retrieval and hand-crafted similarity
4. Feature ablation showing freq + bit_corr + morgan each contribute independently

### Evidence
- No table needed here; results referenced qualitatively

---

## 3. Methods

### 3.1 Problem Formulation
- OF f, query q, candidate pool C_q, binary labels y
- Ranking by scoring function s(c|f)
- Evaluation: Hit@K (fraction of queries with positive in top K)

### 3.2 Dataset
- ChEMBL 36-derived D4A0 archive
- 137,556 query rows, 137 unique OF strings
- 123 OFs with positive-label support; 152 candidate fragments
- 14,950,719 candidate pair rows; 1.33% positive rate
- OF-level train/test partitioning, zero query-level leakage

**Evidence**: `e4_coldstart_audit.csv`

### 3.3 Feature Design (8 features)
| # | Feature | Description | Source |
|---|---------|-------------|--------|
| x1 | Morgan Tanimoto | 2048-bit radius-2 Morgan fingerprint similarity | RDKit |
| x2 | bit_corr | Pearson correlation of L1-normalized FP bit vectors | RDKit |
| x3-x7 | ΔHeavy, ΔRings, ΔMW, ΔlogP, ΔTPSA | Absolute physicochemical deltas (normalized) | RDKit |
| x8 | freq(c) | Candidate frequency among training positive labels (min-max normalized) | Train set |

### 3.4 LBC-Ranker Model
- Logistic regression: s(c|f) = σ(w0 + Σ wi xi)
- L2 regularization (C=1.0), standardized features
- Compact: 9 parameters, deterministic, directly interpretable

### 3.5 Baselines

| Baseline | Description | Tuning |
|----------|-------------|--------|
| Frequency | Rank by train-positive candidate count | None |
| CA (tuned) | λ × Morgan + (1-λ) × PhysChem, λ ∈ {0, 0.25, 0.5, 0.75, 1.0} | Inner 3-fold CV |
| C3F (tuned) | K-nearest OF collaborative filtering, K ∈ {3,5,7,10}, α ∈ {0.1,0.2,0.3,0.5,0.7} | Inner 3-fold CV |
| HGB | HistGradientBoostingClassifier on same 8 features, depth ∈ {3,5}, iter=100, lr=0.05 | Inner 3-fold CV |

### 3.6 Evaluation Protocol
- 10 independent 70/30 OF-level random splits
- Per split: inner 3-fold GroupKFold CV for hyperparameter selection
- Primary metric: OF-macro Hit@10
- Paired sign-flip exact test for LBC vs each baseline
- Feature ablation: one-drop retraining, same splits
- Tuning ledger records all inner-val configurations

**Evidence**: `protocol.md`, `e3_tuning_ledger.csv`

---

## 4. Results

### 4.1 Primary Ranking Performance

**Table 1**: 10-seed OF-macro Hit@10 (values with mean ± std where applicable)

| Method | Macro Hit@10 | Paired Δ vs LBC | p (two-sided) |
|--------|-------------|-----------------|---------------|
| LBC-Ranker | 0.852 ± 0.032 | — | — |
| HGB | 0.851 | -0.002 | — |
| C3F (tuned) | 0.716 | +0.136 | 0.00195 |
| CA (tuned) | 0.661 | +0.191 | 0.00195 |
| Frequency | 0.462 | +0.390 | 0.00195 |

LBC wins in 10/10 seeds. Best non-LBC baseline is C3F in 7/10 seeds, CA in 3/10.

**Evidence**: `e1_main_10seed_summary.csv`, `e1_main_paired_tests.csv`

### 4.2 Model Capacity: LR vs HGB

**Table 2**: Per-seed LR vs HGB comparison

LR (0.852) ≈ HGB (0.851), Δ = -0.002.
HGB wins 6/10 seeds, LR wins 4/10.

Interpretation: the 8-feature representation drives performance, not model capacity. LR retained for interpretability.

**Evidence**: `e6_hgb_vs_lr.csv`

### 4.3 Feature Ablation

**Table 3**: One-drop ablation, OF-macro Hit@10 delta

| Feature Dropped | Δ Hit@10 | Std |
|----------------|---------|------|
| frequency | -0.193 | 0.044 |
| bit_corr | -0.155 | 0.059 |
| morgan | -0.109 | 0.055 |
| dLogP | -0.091 | 0.041 |
| dTPSA | -0.085 | 0.041 |
| dRings | -0.079 | 0.045 |
| dMW | -0.077 | 0.045 |
| dHeavy | -0.070 | 0.045 |

Three features contribute substantively: freq > bit_corr > morgan. Physicochemical deltas contribute modestly but consistently.

**Evidence**: `e2_ablation_full_summary.csv`, `e2_ablation_full_detail.csv`

### 4.4 Per-OF Breakdown

- Distribution of per-OF Hit@10 across methods (boxplot data)
- Rescue/loss counts vs best baseline
- Query-weighted Hit@10: Δ = +0.093

**Evidence**: `e1_main_10seed_detail.csv`, `e5_rank_diag.csv`

### 4.5 Baseline Tuning Behavior

- CA best λ: predominantly 0.75
- C3F best K: 3-10, best α: predominantly 0.1
- HGB best: predominantly depth=5
- Full ledger in supplement

**Evidence**: `e3_tuning_ledger.csv`

---

## 5. Discussion

### 5.1 Why Learned Compatibility Works
- Frequency alone (0.462) insufficient; pure structural similarity (Morgan=0.62) insufficient
- Combining both through learned weights recovers stronger signal
- bit_corr provides standardized, scale-invariant structural readout

### 5.2 Retrieval vs Compatibility
- C3F (0.716) = viable retrieval baseline at sufficient OF coverage
- LBC-C3F gap (0.136) comes from direct feature-based compatibility, not neighbor aggregation
- Retrieval useful but incomplete; learning directly from candidate–fragment features adds independent signal

### 5.3 Simplicity and Interpretability
- 9-parameter LR matches higher-capacity HGB
- Each weight directly interpretable as feature contribution
- Suitable for practical deployment (CPU, no GPU, deterministic)

### 5.4 Limitations
- 123 OFs from single data source (ChEMBL 36)
- 152 candidates in pool; real-world pools larger
- No 3D conformational or electrostatic features
- Label imbalance (1.33% positive rate); ranking formulation partially mitigates
- OF coverage still incomplete (14 OFs zero positive support)

---

## 6. Conclusion

LBC-Ranker learns a balance between candidate replacement track record and binary-fingerprint substructure match, achieving +0.130 macro Hit@10 improvement over the strongest tuned non-LBC baseline across 123 old fragments. Feature ablation confirms three complementary signals: candidate frequency, bit-level correlation, and Morgan similarity. LR matches HGB, supporting the sufficiency of the 8-feature compatibility representation. The model is compact, interpretable, and practical.

---

## 7. Data and Software Availability

- Code: [repository URL]
- Precomputed features: [URL]
- ChEMBL 36: https://www.ebi.ac.uk/chembl/
- All experiment outputs: `paper_results/v2_full_data/`
- Protocol: `paper_results/v2_full_data/protocol.md`

---

## Evidence Map

| Claim | Evidence File | Values |
|-------|--------------|--------|
| LBC > C3F | `e1_main_paired_tests.csv` | Δ +0.136, p=0.00195 |
| LBC > CA | `e1_main_paired_tests.csv` | Δ +0.191, p=0.00195 |
| LBC 10/10 wins | `e1_main_10seed_summary.csv` | δ_macro > 0 in all rows |
| freq > bit_corr > morgan | `e2_ablation_full_summary.csv` | -0.193 > -0.155 > -0.109 |
| LR ≈ HGB | `e6_hgb_vs_lr.csv` | Δ -0.002 |
| 123 OFs, 15M pairs | `e4_coldstart_audit.csv` | — |
| CA best λ=0.75 | `e3_tuning_ledger.csv` | — |

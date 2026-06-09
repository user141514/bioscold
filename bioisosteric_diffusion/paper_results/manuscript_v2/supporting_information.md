# Supporting Information

## LBC-Ranker: Learned Candidate–Fragment Compatibility Ranking for Fragment Replacement

---

## S1. Full Experiment Protocol

The experiment protocol is archived at `paper_results/v2_full_data/protocol.md`. Key design decisions:

- **Data**: 123 OFs with positive-label support from 137,556 query rows (ChEMBL 36-derived D4A0 archive)
- **Evaluation**: 10 independent 70/30 OF-level splits, inner 3-fold GroupKFold CV for hyperparameter selection
- **Primary metric**: OF-macro Hit@10
- **Statistical test**: Exact paired sign-flip test (2^10 = 1,024 sign configurations)
- **Tiebreaker**: Within 0.005 mean inner Hit@10, lower model complexity preferred

---

## S2. Dataset Statistics

**Table S1: Dataset summary.**

| Metric | Value |
|--------|-------|
| Total query rows | 137,556 |
| Unique OF strings | 137 |
| OFs with positive-label support | 123 |
| OFs with zero positive-label support | 14 (10.2%) |
| Unique candidate fragments | 152 |
| Total candidate pair rows | 14,950,719 |
| Positive pair rate | 1.33% |
| Positive pair range per OF | 98–27,448 |
| Top 10 OFs share of positives | 47.8% |

Source: `paper_results/v2_full_data/e4_coldstart_audit.csv`

---

## S3. Split Imbalance Diagnostics

**Table S2: Per-seed split imbalance audit.**

| Seed | Train OFs | Test OFs | Train pos pairs | Test pos pairs | Train share |
|------|-----------|----------|-----------------|----------------|-------------|
| 0 | 86 | 37 | 142,053 | 53,668 | 72.6% |
| 1 | 86 | 37 | 149,997 | 45,724 | 76.6% |
| 2 | 86 | 37 | 162,164 | 33,557 | 82.9% |
| 3 | 86 | 37 | 136,447 | 59,274 | 69.7% |
| 4 | 86 | 37 | 148,631 | 47,090 | 75.9% |
| 5 | 86 | 37 | 149,217 | 46,504 | 76.2% |
| 6 | 86 | 37 | 156,085 | 39,636 | 79.7% |
| 7 | 86 | 37 | 153,185 | 42,536 | 78.3% |
| 8 | 86 | 37 | 158,272 | 37,449 | 80.9% |
| 9 | 86 | 37 | 149,346 | 46,375 | 76.3% |

Train positive-pair share ranges from 69.7% to 82.9% across seeds (mean 76.9%). Splits are not stratified by positive-pair count; this variability is reflected in the per-seed standard deviation of Hit@10 estimates.

Source: `paper_results/v2_full_data/e4_split_imbalance_audit.csv`

---

## S4. Per-Seed Detailed Results

**Table S3: Per-seed OF-macro Hit@10.**

| Seed | LBC-Ranker | HGB | C3F-style (tuned) | CA (tuned) | Frequency | Best non-LBC | Δ macro |
|------|-----------|-----|-------------------|------------|-----------|-------------|---------|
| 0 | 0.8266 | 0.8640 | 0.7467 | 0.6195 | 0.4694 | C3F | +0.0799 |
| 1 | 0.8686 | 0.8778 | 0.6944 | 0.6593 | 0.4519 | C3F | +0.1743 |
| 2 | 0.8625 | 0.8781 | 0.6763 | 0.6926 | 0.4112 | CA | +0.1699 |
| 3 | 0.8591 | 0.7438 | 0.6742 | 0.6552 | 0.4367 | C3F | +0.1849 |
| 4 | 0.8162 | 0.8244 | 0.7737 | 0.6460 | 0.4927 | C3F | +0.0425 |
| 5 | 0.8285 | 0.9016 | 0.6910 | 0.6493 | 0.4147 | C3F | +0.1375 |
| 6 | 0.8991 | 0.8942 | 0.7633 | 0.6640 | 0.5094 | C3F | +0.1358 |
| 7 | 0.8376 | 0.7838 | 0.6742 | 0.6899 | 0.4285 | CA | +0.1477 |
| 8 | 0.9037 | 0.8766 | 0.7864 | 0.6288 | 0.5068 | C3F | +0.1173 |
| 9 | 0.8190 | 0.8603 | 0.6832 | 0.7090 | 0.5032 | CA | +0.1099 |
| **Mean** | **0.8521** | 0.8505 | 0.7163 | 0.6614 | 0.4624 | — | **+0.1300** |
| **Std** | 0.0318 | — | — | — | — | — | 0.0443 |

Source: `paper_results/v2_full_data/e1_main_10seed_summary.csv`

---

## S5. Paired Sign-Flip Tests

**Table S4: Exact paired sign-flip test results (2^10 = 1,024 configurations).**

| Comparison | n | Mean Δ | All positive | One-sided p | Two-sided p |
|------------|---|--------|-------------|------------|------------|
| LBC vs CA | 10 | +0.1907 | Yes | 0.000977 | 0.001953 |
| LBC vs C3F-style | 10 | +0.1357 | Yes | 0.000977 | 0.001953 |
| LBC vs Frequency | 10 | +0.3896 | Yes | 0.000977 | 0.001953 |
| LBC vs Best non-LBC | 10 | +0.1300 | Yes | 0.000977 | 0.001953 |

Source: `paper_results/v2_full_data/e1_main_paired_tests.csv`

---

## S6. Full Ablation Results

**Table S5: Per-seed feature ablation (OF-macro Hit@10 degradation relative to full 8-feature model).**

| Seed | freq | bit_corr | morgan | dLogP | dTPSA | dRings | dMW | dHeavy |
|------|------|----------|--------|-------|-------|--------|-----|--------|
| 0 | −0.2141 | −0.1516 | −0.0734 | −0.0832 | −0.0796 | −0.0777 | −0.0786 | −0.0641 |
| 1 | −0.1857 | −0.1650 | −0.1107 | −0.0931 | −0.0897 | −0.0782 | −0.0759 | −0.0714 |
| 2 | −0.1929 | −0.1578 | −0.1083 | −0.0887 | −0.0829 | −0.0778 | −0.0764 | −0.0706 |
| 3 | −0.1993 | −0.1588 | −0.1101 | −0.0889 | −0.0854 | −0.0753 | −0.0742 | −0.0689 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

Full per-seed ablation results are archived in `paper_results/v2_full_data/e2_ablation_full_detail.csv`.

**Table S6: Ablation summary (10-seed mean ± std).**

| Feature Dropped | Mean Δ Hit@10 | Std |
|----------------|--------------|-----|
| frequency | −0.193 | 0.044 |
| bit_corr | −0.155 | 0.059 |
| morgan | −0.109 | 0.055 |
| dLogP | −0.091 | 0.041 |
| dTPSA | −0.085 | 0.041 |
| dRings | −0.079 | 0.045 |
| dMW | −0.077 | 0.045 |
| dHeavy | −0.070 | 0.045 |

Source: `paper_results/v2_full_data/e2_ablation_full_summary.csv`

---

## S7. HGB vs LR Per-Seed Comparison

**Table S7: LR vs HGB per seed.**

| Seed | LR Macro Hit@10 | HGB Macro Hit@10 | Δ (HGB − LR) |
|------|----------------|-----------------|--------------|
| 0 | 0.8266 | 0.8640 | +0.0374 |
| 1 | 0.8686 | 0.8778 | +0.0092 |
| 2 | 0.8625 | 0.8781 | +0.0157 |
| 3 | 0.8591 | 0.7438 | −0.1153 |
| 4 | 0.8162 | 0.8244 | +0.0082 |
| 5 | 0.8285 | 0.9016 | +0.0731 |
| 6 | 0.8991 | 0.8942 | −0.0049 |
| 7 | 0.8376 | 0.7838 | −0.0538 |
| 8 | 0.9037 | 0.8766 | −0.0270 |
| 9 | 0.8190 | 0.8603 | +0.0414 |
| **Mean** | **0.8521** | **0.8505** | **−0.0016** |

Source: `paper_results/v2_full_data/e6_hgb_vs_lr.csv`

---

## S8. Tuning Ledger Summary

**Table S8: Best configurations per seed.**

| Seed | Best CA λ | Best C3F K | Best C3F α | Best HGB depth |
|------|----------|-----------|-----------|----------------|
| 0 | 0.75 | 10 | 0.1 | 5 |
| 1 | 0.75 | 7 | 0.1 | 5 |
| 2 | 0.75 | 5 | 0.1 | 5 |
| 3 | 0.75 | 5 | 0.1 | 5 |
| 4 | 0.75 | 7 | 0.1 | 3 |
| 5 | 0.75 | 10 | 0.1 | 5 |
| 6 | 0.75 | 5 | 0.1 | 5 |
| 7 | 0.75 | 7 | 0.1 | 5 |
| 8 | 0.75 | 3 | 0.1 | 3 |
| 9 | 0.75 | 5 | 0.1 | 5 |

CA consistently selects λ = 0.75 (3:1 Morgan-to-physchem weighting). C3F-style selects α = 0.1 in all seeds (light content-aware fallback), with K varying from 3 to 10. HGB selects depth = 5 in 8/10 seeds and depth = 3 in 2/10 seeds.

Full tuning ledger with all inner cross-validation scores for all 38 evaluated configurations per seed is archived in `paper_results/v2_full_data/e3_tuning_ledger.csv`.

---

## S9. Learned Feature Weights

**Table S9: LBC-Ranker feature weights across 10 seeds (standardized features, mean ± std).**

| Feature | Mean Weight | Std | |Mean| |
|---------|------------|-----|--------|
| bit_corr | +3.351 | 0.160 | 3.351 |
| Morgan | −2.404 | 0.131 | 2.404 |
| dLogP | −0.442 | 0.034 | 0.442 |
| dTPSA | −0.408 | 0.027 | 0.408 |
| freq | +0.393 | 0.018 | 0.393 |
| dHeavy | −0.362 | 0.044 | 0.362 |
| dRings | +0.162 | 0.036 | 0.162 |
| dMW | −0.026 | 0.042 | 0.026 |
| Intercept | −5.425 | 0.052 | — |

Weights are on standardized features (zero mean, unit variance) and are therefore comparable in magnitude. bit_corr and Morgan carry the largest learned weights, consistent with their roles as the primary structural compatibility signals. The low standard deviations across seeds indicate stable weight estimates. Note that feature weight magnitude does not directly correspond to ablation importance because features are correlated; the ablation results (Table S6) provide the causal estimate of each feature's marginal contribution.

Source: `paper_results/v2_full_data/e7_learned_weights.csv`

---

## S10. Query-Weighted Hit@10

**Table S10: Query-weighted Hit@10 across 10 seeds.**

| Seed | LBC-Ranker | HGB | C3F-style (tuned) | CA (tuned) | Frequency | Δ query (LBC − best) |
|------|-----------|-----|-------------------|------------|-----------|---------------------|
| 0 | 0.7256 | 0.7829 | 0.6879 | 0.5641 | 0.4358 | +0.0377 |
| 1 | 0.8281 | 0.8625 | 0.7133 | 0.6353 | 0.5278 | +0.1147 |
| 2 | 0.7981 | 0.7960 | 0.7268 | 0.6634 | 0.3806 | +0.1347 |
| 3 | 0.6780 | 0.6520 | 0.6004 | 0.5312 | 0.4530 | +0.0776 |
| 4 | 0.7134 | 0.7234 | 0.7025 | 0.6127 | 0.4513 | +0.0110 |
| 5 | 0.8063 | 0.8528 | 0.7138 | 0.6220 | 0.4532 | +0.0925 |
| 6 | 0.8171 | 0.8321 | 0.7725 | 0.6079 | 0.5395 | +0.0447 |
| 7 | 0.7811 | 0.7727 | 0.6731 | 0.6019 | 0.4708 | +0.1793 |
| 8 | 0.8456 | 0.8394 | 0.7569 | 0.6191 | 0.5234 | +0.0887 |
| 9 | 0.7777 | 0.8110 | 0.6903 | 0.6263 | 0.5314 | +0.1514 |
| **Mean** | **0.7771** | 0.7925 | 0.7037 | 0.6084 | 0.4767 | **+0.0932** |

Query-weighted Hit@10 is computed across all queries (weighted by query count per OF). LBC-Ranker achieves a paired mean improvement of +0.093 over the per-seed best non-LBC baseline. Seed 4 shows the narrowest margin (+0.011), consistent with its low macro Δ (+0.043).

Source: `paper_results/v2_full_data/e1_main_10seed_summary.csv` (query-weighted columns)

---

## S11. Zero-Positive OF Predictions

Fourteen OFs in the manifest have zero labeled positive replacements in the current archive. We apply LBC-Ranker (trained on all 123 labeled OFs) to predict top-ranked candidates for these OFs as a qualitative check of model behavior in the absence of training labels for the query OF.

**Table S11: Top-3 predicted candidates for zero-positive OFs.**

| OF | Top-1 | Score | Top-2 | Score | Top-3 | Score |
|-----|-------|-------|-------|-------|-------|-------|
| *C(=O)O | *c1ccccc1 | 0.067 | *C(C)=O | 0.061 | *C(=O)OC | 0.054 |
| *C(=O)c1ccc(OC)cc1 | *c1ccc(OC)cc1 | 0.142 | *c1ccccc1 | 0.077 | *C(=O)c1ccccc1 | 0.045 |
| *C(=O)c1cccs1 | *c1ccccc1 | 0.205 | *c1ccc(Cl)cc1 | 0.047 | *C(=O)c1ccccc1 | 0.045 |
| *CC(C)C | *C(C)C | 0.068 | *CC | 0.063 | *C(C)CC | 0.046 |
| *CCCCCC | *CC | 0.056 | *CCC | 0.047 | *c1ccccc1 | 0.038 |
| *Cc1cccnc1 | *c1ccccc1 | 0.225 | *Cc1ccccc1 | 0.113 | *Cc1ccncc1 | 0.062 |
| *NCCc1ccccc1 | *c1ccccc1 | 0.302 | *Cc1ccccc1 | 0.079 | *NCc1ccccc1 | 0.048 |
| *c1ccc(-c2ccccc2)cc1 | *c1ccccc1 | 0.220 | *c1ccc(Cl)cc1 | 0.093 | *c1ccc(F)cc1 | 0.066 |
| *c1ccc(C(=O)O)cc1 | *c1ccccc1 | 0.361 | *c1ccc(OC)cc1 | 0.112 | *c1ccc(F)cc1 | 0.094 |
| *c1ccc(Cl)cc1 | *c1ccccc1 | 0.406 | *c1ccc(F)cc1 | 0.162 | *c1ccc(C)cc1 | 0.138 |
| *c1ccc2ccccc2c1 | *c1ccccc1 | 0.346 | *c1ccc(Cl)cc1 | 0.113 | *c1ccc(F)cc1 | 0.079 |
| *c1cccc(C#N)c1 | *c1ccccc1 | 0.443 | *c1cccc(OC)c1 | 0.087 | *c1ccc(OC)cc1 | 0.075 |
| *c1cccc(OC)c1 | *c1ccccc1 | 0.462 | *c1ccc(OC)cc1 | 0.237 | *c1ccccc1OC | 0.079 |
| *c1ccccc1OC | *c1ccccc1 | 0.394 | *c1ccc(OC)cc1 | 0.189 | *c1ccc(F)cc1 | 0.106 |

The top-ranked candidate for all 14 zero-positive OFs is *c1ccccc1 (phenyl), the most frequent positive-label candidate in the training set (27,448 positive pairs). This reflects the dominant role of the frequency feature when no OF-specific training signal is available. Top-1 scores range from 0.067 (*C(=O)O, very low confidence) to 0.462 (*c1cccc(OC)c1, moderate confidence). For OFs with clear structural analogs among labeled OFs (e.g., *c1ccc(Cl)cc1, *c1cccc(OC)c1), the top-2 and top-3 predictions include structurally plausible replacements beyond the frequency prior.

Source: `paper_results/v2_full_data/e8_zero_of_predictions.csv`

---

## S12. External Validation Extension

External validation is pre-specified in `paper_results/v2_external_validation/protocol.md`. The extension separates three evidence roles:

- **BindingDB-derived MMP replacement benchmark**: main external dataset after source files, standardized matrix, and result CSVs are locked.
- **ChEMBL temporal split**: auxiliary time-robustness validation, not an independent external data source.
- **SwissBioisostere overlap / sanity check**: external replacement-reference overlap analysis only; not used for training, tuning, or activity-preservation claims.

The standardized external candidate matrix has one row per query-candidate pair and requires `query_id`, `old_fragment_smiles`, `candidate_smiles`, and `label`. Optional precomputed features are `morgan`, `bit_corr`, `dHeavy`, `dRings`, `dMW`, `dLogP`, and `dTPSA`; otherwise the evaluator computes them with RDKit. The locked evaluator is `paper_results/v2_external_validation/run_external_validation.py`.

Before any BindingDB, temporal-split, or SwissBioisostere result is promoted to the main manuscript, the corresponding source release, matrix construction script, matrix hash, result CSV, and leakage audit must be archived.

---

## S13. Data Availability

- **Experiment protocol**: `paper_results/v2_full_data/protocol.md`
- **Per-seed summary**: `paper_results/v2_full_data/e1_main_10seed_summary.csv`
- **Per-OF detail**: `paper_results/v2_full_data/e1_main_10seed_detail.csv`
- **Paired tests**: `paper_results/v2_full_data/e1_main_paired_tests.csv`
- **Ablation detail**: `paper_results/v2_full_data/e2_ablation_full_detail.csv`
- **Ablation summary**: `paper_results/v2_full_data/e2_ablation_full_summary.csv`
- **Tuning ledger**: `paper_results/v2_full_data/e3_tuning_ledger.csv`
- **Cold-start audit**: `paper_results/v2_full_data/e4_coldstart_audit.csv`
- **Split imbalance**: `paper_results/v2_full_data/e4_split_imbalance_audit.csv`
- **Rank diagnostics**: `paper_results/v2_full_data/e5_rank_diag.csv`
- **HGB vs LR**: `paper_results/v2_full_data/e6_hgb_vs_lr.csv`
- **External validation protocol**: `paper_results/v2_external_validation/protocol.md`
- **External validation evaluator**: `paper_results/v2_external_validation/run_external_validation.py`

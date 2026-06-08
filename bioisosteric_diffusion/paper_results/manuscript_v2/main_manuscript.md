# Learned Candidate–Fragment Compatibility Ranking for Fragment Replacement

## Abstract

Ranking candidate replacements for a query old fragment (OF) is a central task in bioisosteric design. Effective scoring must balance two signals: whether a candidate has a reliable replacement track record, and whether its substructure pattern matches the OF. Existing methods address only one side of this balance. Hand-crafted content similarity ignores candidate priors, while retrieval-based approaches aggregate cross-OF replacement patterns without modeling direct structural compatibility between a specific candidate and the query OF. We introduce the Learned Binary-fingerprint Compatibility Ranker (LBC-Ranker). LBC-Ranker is a compact logistic regression model that learns this balance from labeled replacement pairs. It uses eight features: Morgan Tanimoto similarity, bit-level fingerprint correlation, five physicochemical property deltas, and train-only candidate frequency. Across 123 old fragments under a 10-seed repeated OF-level split protocol with inner 3-fold cross-validated baseline tuning, LBC-Ranker achieves an OF-macro Hit@10 of 0.852 ± 0.032. It improves over the strongest tuned non-LBC baseline by a paired mean difference of +0.130 ± 0.044 (two-sided exact p = 0.00195), winning in all 10 seeds. LBC-Ranker matches a higher-capacity gradient-boosted tree (HGB, 0.851) on the same features; under the current feature set and HGB hyperparameter grid, increasing model capacity beyond a linear model did not yield measurable improvement. Feature ablation identifies three independently contributing signals: candidate frequency (Δ = −0.193), bit-level correlation (Δ = −0.155), and Morgan similarity (Δ = −0.109). The model is compact (9 parameters), deterministic, and directly interpretable, making it suitable for practical deployment in fragment replacement workflows.

## 1. Introduction

Bioisosteric replacement substitutes one molecular fragment with another that preserves biological activity while modulating physicochemical properties. It is a fundamental strategy in medicinal chemistry and lead optimization [1, 2]. Computational tools that rank candidate replacements for a query fragment can accelerate the design-make-test cycle by prioritizing the most promising analogs for synthesis [3, 4].

The ranking problem can be stated as follows: given an old fragment (OF) and a pool of candidate fragments, rank candidates so that those known to be labeled positive replacements appear at the top. This problem is challenging for two reasons. First, most OFs have relatively few experimentally validated positive replacements, limiting the training signal available from any single OF. Second, the candidate pool is large and the labeled positive rate is low (1.33% in the current archive), which places a premium on scoring functions that can separate rare positives from abundant negatives.

Existing computational approaches to fragment replacement ranking fall into three categories. **Fingerprint similarity methods** rank candidates by Tanimoto similarity of Morgan fingerprints to the OF [5, 6]. These methods capture global structural overlap but treat all fingerprint bits as equally informative and ignore the candidate's historical success rate as a replacement across other OFs. **Physicochemical filtering** constrains candidates by property matching (molecular weight, logP, heavy atom count) [7, 8], enforcing the bioisostere principle that replacements should conserve key properties. However, fixed similarity thresholds cannot capture the nuanced trade-offs between structural match and property conservation that characterize labeled positive replacements. **K-nearest-OF collaborative retrieval** aggregates candidate success distributions from chemically similar OFs, providing a retrieval-based prior that leverages cross-OF replacement patterns. However, this strategy depends on sufficient OF coverage to build reliable per-OF candidate distributions and does not directly model the pairwise compatibility between a specific candidate–OF pair from structural features.

A practical fragment replacement ranking system should combine both sources of information: a candidate's replacement track record across the known OF landscape, and its structural compatibility with the specific query OF. No existing method learns this balance from data.

We propose the **Learned Binary-fingerprint Compatibility Ranker (LBC-Ranker)**, a supervised ranking model that addresses this gap through three design choices:

1. **Learned feature combination**: Rather than hand-tuning the relative weights of similarity, physicochemical, and frequency signals, LBC-Ranker learns these weights via logistic regression trained on labeled replacement pairs.

2. **Bit-level correlation feature**: Beyond global Tanimoto similarity, we compute the Pearson correlation between L1-normalized candidate and OF fingerprint bit vectors. This provides a readout of substructure pattern agreement that complements global Tanimoto similarity: two fragments may share many bits (high Tanimoto) but differ in which specific bits co-occur (low bit_corr), or vice versa.

3. **Train-only candidate frequency**: Candidate frequency among positive training labels serves as a replacement track-record prior. This feature is computed strictly from training OFs to prevent label leakage across the OF split boundary.

We evaluate LBC-Ranker on a ChEMBL 36-derived archive containing 137,556 query rows, 137 unique OF strings, and 152 candidate fragments. The primary evaluation follows a 10-seed repeated OF-level split protocol. In each seed, the 123 OFs with positive-label support are randomly partitioned 70/30 into training and test sets. Hyperparameters for all baseline methods are selected through inner 3-fold cross-validation within the training set, and the tuned baselines are evaluated on the held-out test OFs. This protocol ensures that no test-OF label information leaks into training, frequency computation, or hyperparameter selection.

LBC-Ranker achieves an OF-macro Hit@10 of 0.852 ± 0.032 and improves over the strongest tuned non-LBC baseline by a paired mean difference of +0.130 (p = 0.00195, two-sided exact sign-flip test), winning in all 10 seeds. A higher-capacity gradient-boosted tree (HGB) trained on the same eight features reaches essentially identical performance (0.851): under the current feature set and HGB hyperparameter grid, increasing model capacity beyond a linear model did not yield measurable improvement. Feature ablation confirms that candidate frequency, bit-level correlation, and Morgan similarity each contribute independently to ranking performance. The tuned retrieval baseline (C3F-style, 0.716) confirms that cross-OF retrieval is viable given sufficient OF coverage but remains substantially below LBC-Ranker.

## 2. Methods

### 2.1 Problem Formulation

Let F = {f_1, ..., f_N} be the set of old fragments (OFs), each represented by a SMILES string. For each OF f, we have a set of query instances Q_f, where each query q ∈ Q_f is associated with a candidate pool C_q = {c_1, ..., c_{n_q}} and binary labels y_{q,c} ∈ {0, 1} indicating whether candidate c is a known labeled positive replacement. The ranking task is to produce a scoring function s(c | f) such that candidates with y = 1 rank above those with y = 0, evaluated by Hit@K (the fraction of queries where at least one positive candidate appears in the top K ranked positions).

### 2.2 Dataset

The dataset is derived from ChEMBL 36 [9] through the D4A0 processing pipeline and contains 137,556 query rows, 137 unique OF strings, and 152 unique candidate fragments in the ranking pool. Labeled positive replacements are defined as candidate–OF pairs where a compound containing the candidate fragment and a compound containing the OF were both tested against the same protein target and both met a bioactivity threshold (pIC50 ≥ 6 or equivalent). The D4A0 pipeline performs OF-level train/test partitioning; no query-level information crosses the OF split boundary.

Of the 137 unique OFs, 123 have positive-label support (at least one labeled positive replacement pair in the scanned label shards), and 14 (10.2%) have zero positive-label support in the current archive. The overall positive pair rate across 14,950,719 candidate pair rows is 1.33%. For the 123 labeled OFs, positive support mass (total positive-labeled query instances) ranges across several orders of magnitude, with the top 10 OFs accounting for the majority of positive pairs.

### 2.3 Feature Design

For each (candidate, OF) pair, we compute an 8-dimensional feature vector x(c, f) ∈ R^8 (Table 1).

**Table 1: Feature definitions.**

| Feature | Description |
|---------|-------------|
| Morgan Tanimoto | Tanimoto similarity between 2048-bit Morgan fingerprints (radius 2, RDKit) of candidate and OF |
| bit_corr | Pearson correlation between L1-normalized candidate and OF fingerprint bit vectors |
| ΔHeavy | Absolute difference in heavy atom count |
| ΔRings | Absolute difference in ring count |
| ΔMW | Absolute difference in molecular weight, normalized by OF MW |
| ΔlogP | Absolute difference in logP, normalized by |OF logP| + 1 |
| ΔTPSA | Absolute difference in TPSA, normalized by OF TPSA + 1 |
| freq(c) | Candidate frequency among positive labels in the training set, min-max normalized |

Morgan Tanimoto captures global structural similarity. The bit_corr feature is designed to detect whether specific informative bits co-occur between candidate and OF: two fragments may have moderate overall Tanimoto similarity but high correlation on the subset of pharmacophoric bits critical for bioisosterism. For binary fingerprints, bit_corr is related to but distinct from Tanimoto similarity: it measures the linear association of normalized bit intensities rather than the fractional overlap of set bits, capturing a different aspect of fingerprint agreement. Physicochemical deltas enforce property conservation. Candidate frequency serves as a replacement track-record prior: fragments that appear frequently as labeled positive replacements across many training OFs are more likely to be useful replacements in general.

### 2.4 LBC-Ranker Model

The scoring function is a logistic regression model:

s(c | f) = σ(w_0 + Σ_{i=1}^{8} w_i x_i(c, f))

where σ is the sigmoid function and w_i are learned weights. Features are standardized (zero mean, unit variance) before training using statistics computed from the training set. We train with binary cross-entropy loss and L2 regularization (C = 1.0, scikit-learn LogisticRegression, max_iter = 2000).

For capacity comparison, we also train a HistGradientBoostingClassifier (HGB) on the same eight features. HGB hyperparameters are selected via the same inner 3-fold cross-validation protocol used for all baselines.

### 2.5 Baselines

**Frequency**: Rank candidates by their count among positive labels in the training set, min-max normalized. A pure prior without OF-specific structural information.

**Content-Aware (CA)**: Hand-crafted linear combination s_CA(c | f) = λ × Morgan(c, f) + (1 − λ) × PhysChem. The PhysChem term is 1/(1 + ΣΔphyschem). The weight λ is tuned via inner cross-validation from {0, 0.25, 0.5, 0.75, 1.0}.

**C3F-style retrieval baseline**: K-nearest-OF collaborative retrieval with weighted candidate success aggregation and a content-aware fallback for OFs with weak collaborative signal. For each test OF, we find the K most Morgan-similar training OFs, aggregate their per-OF normalized candidate success distributions (each computed strictly from training labels) with similarity-based weighting, and blend with content similarity via a tunable α parameter. K ∈ {3, 5, 7, 10} and α ∈ {0.1, 0.2, 0.3, 0.5, 0.7} are selected via inner cross-validation. This baseline represents the retrieval-based strategy of using cross-OF replacement patterns without learning direct candidate–OF compatibility features.

### 2.6 Evaluation Protocol

**Outer split**: Ten independent random 70/30 splits of the 123 labeled OFs into training and test sets (seed ∈ {0, ..., 9}). OFs are split without stratification; per-split imbalance diagnostics (positive-pair count distribution in train vs test) are provided in the Supporting Information.

**Inner tuning**: For each outer split, hyperparameters for CA, the retrieval baseline, and HGB are selected via 3-fold GroupKFold cross-validation over the training OFs. Configurations are ranked by mean inner OF-macro Hit@10; ties within 0.005 are broken by model complexity (lower K, lower α, lower depth preferred).

**Primary metric**: OF-macro Hit@10, computed as the mean of per-OF Hit@10 values over test OFs. Query-weighted Hit@10 is reported as a secondary metric.

**Statistical testing**: For each baseline, we compute the per-seed paired difference in OF-macro Hit@10 between LBC-Ranker and the baseline. We apply an exact paired sign-flip test (all 2^10 sign configurations) to obtain two-sided p-values.

**Feature ablation**: For each of the 8 features, we retrain LBC-Ranker with that feature removed (same 10 outer splits, same hyperparameters) and report the mean degradation in OF-macro Hit@10 relative to the full 8-feature model.

**Tuning ledger**: All inner cross-validation results for all evaluated configurations are recorded and included as Supporting Information.

## 3. Results

### 3.1 Primary Ranking Performance

Table 2 and Figure 2 report the 10-seed ranking performance. LBC-Ranker achieves an OF-macro Hit@10 of 0.852 ± 0.032. The strongest tuned non-LBC baseline is C3F-style retrieval in 7 of 10 seeds and CA in 3 of 10 seeds. The per-seed best non-LBC baseline achieves a mean Hit@10 of 0.723, and the paired mean difference between LBC-Ranker and this per-seed best is +0.130 ± 0.044 (two-sided p = 0.00195, exact sign-flip test). LBC-Ranker outperforms the best non-LBC baseline in all 10 seeds.

**Table 2: 10-seed OF-macro ranking performance (123 OFs).**

| Method | Macro Hit@10 | Δ (LBC − method) | Two-sided p |
|--------|-------------|-------------------|-------------|
| **LBC-Ranker** | **0.852 ± 0.032** | — | — |
| Best non-LBC (per-seed) | 0.723 | +0.130 ± 0.044 | 0.00195 |
| HGB | 0.851 | +0.002 | — |
| C3F-style retrieval (tuned) | 0.716 | +0.136 | 0.00195 |
| CA (tuned) | 0.661 | +0.191 | 0.00195 |
| Frequency | 0.462 | +0.390 | 0.00195 |

*Best non-LBC (per-seed): Mean across seeds of the per-seed best non-LBC baseline Hit@10 (0.723). The best baseline is C3F-style retrieval in 7 of 10 seeds and CA in 3 of 10 seeds. The paired Δ is LBC minus the per-seed best baseline. HGB achieves Hit@10 within 0.002 of LBC-Ranker; the difference is not statistically distinguishable under the current HGB hyperparameter grid.*

### 3.2 Model Capacity: LR vs HGB

LBC-Ranker (LR, 0.852) and HGB (0.851) achieve essentially identical macro Hit@10 (mean paired difference −0.002). HGB achieves a higher macro Hit@10 in 6 seeds and LR in 4 seeds. We retain LR as the primary implementation because it is compact (9 parameters), deterministic, and directly interpretable through feature weights.

### 3.3 Feature Ablation

Table 3 and Figure 1 report the one-drop ablation across 10 seeds. Dropping candidate frequency causes the largest macro Hit@10 degradation (−0.193 ± 0.044), followed by bit_corr (−0.155 ± 0.059) and Morgan Tanimoto similarity (−0.109 ± 0.055). The five physicochemical delta features each contribute modestly (Δ ranging from −0.070 to −0.091).

**Table 3: Feature ablation (10-seed mean OF-macro Hit@10 degradation).**

| Feature Dropped | Δ Hit@10 | Std |
|----------------|---------|------|
| frequency | −0.193 | 0.044 |
| bit_corr | −0.155 | 0.059 |
| morgan | −0.109 | 0.055 |
| dLogP | −0.091 | 0.041 |
| dTPSA | −0.085 | 0.041 |
| dRings | −0.079 | 0.045 |
| dMW | −0.077 | 0.045 |
| dHeavy | −0.070 | 0.045 |

### 3.4 Baseline Tuning Behavior

CA tuning consistently selects λ = 0.75 across all seeds (3:1 Morgan-to-physchem weighting). C3F-style retrieval tuning selects K ∈ {3, 5, 7, 10} with α = 0.1 in all 10 seeds. HGB tuning selects depth = 5 in 8 of 10 seeds and depth = 3 in 2 of 10 seeds. Full tuning results are provided in the Supporting Information (tuning ledger).

### 3.5 Query-Weighted and Per-OF Analysis

Under query-weighted Hit@10, LBC-Ranker achieves a paired improvement of +0.093 over the best tuned non-LBC baseline. The full per-OF Hit@10 distribution across all seeds is provided in the Supporting Information (Table S3).

## 4. Discussion

### 4.1 Why Learned Compatibility Improves Ranking

The ablation results clarify why learning the balance between candidate support and structural compatibility matters. Frequency alone achieves a Hit@10 of 0.462, and tuned Morgan-based content similarity reaches only 0.661. Combining both through learned weights, with bit_corr providing an independent structural readout, recovers a substantially stronger ranking signal (0.852). The gap between tuned CA (0.661) and LBC-Ranker (0.852) shows that fixed or hand-tuned content similarity is insufficient; learning the balance from data is necessary.

**Frequency is the strongest single feature, but structure is necessary for top performance.** The ablation identifies candidate frequency as the dominant contributor (−0.193). However, frequency alone (0.462) substantially underperforms the full model (0.852) and even the tuned retrieval baseline (0.716). The stepwise improvement (from frequency 0.462, to retrieval 0.716, to learned compatibility 0.852) shows that structural features (bit_corr, Morgan) provide information that neither the replacement prior nor cross-OF retrieval captures. The model does not merely apply a weighted frequency baseline; it learns how structural compatibility modifies the ranking beyond what track record alone predicts.

### 4.2 Retrieval vs Direct Compatibility Scoring

The tuned retrieval baseline (0.716) is viable, substantially above frequency (0.462) and CA (0.661). With sufficient OF coverage (86 training OFs per seed), cross-OF candidate distributions carry meaningful signal. However, the gap between LBC-Ranker and the retrieval baseline (0.136, per-method) shows that directly scoring candidate–fragment compatibility from structural and support features captures information that neighbor-aggregated retrieval does not. Retrieval and compatibility scoring are complementary, but learning directly from pair-level features is more effective than relying on neighbor aggregation alone.

### 4.3 Representation, Not Capacity

**LR and HGB achieve comparable performance on the current eight features.** LR (0.852) and HGB (0.851) differ by a mean paired Δ of +0.002 (LBC minus HGB). Under the HGB hyperparameter grid tested here (max_depth ∈ {3, 5}, max_iter = 100, learning_rate = 0.05), increasing model capacity beyond a linear model did not yield measurable improvement. This does not imply that model capacity is irrelevant in general: with substantially more training OFs or additional feature dimensions, higher-capacity models might outperform linear ones. The 9-parameter LR implementation of LBC-Ranker is compact, deterministic, and directly interpretable: each weight w_i quantifies the marginal contribution of its corresponding feature to the log-odds of a candidate being a labeled positive replacement. Whether additional features beyond the current eight (three-dimensional shape, electrostatic potential matching, or learned fingerprint embeddings) would break the LR–HGB parity remains an open question.

### 4.4 Practical Implications

For practical deployment, the LR implementation of LBC-Ranker offers several advantages beyond its ranking accuracy. The three-signal ablation pattern shows that candidate track record provides the strongest prior (frequency, Δ = −0.193), bit-level correlation captures substructure pattern agreement beyond global similarity (Δ = −0.155), and Morgan Tanimoto supplies global structural context (Δ = −0.109). The five physicochemical deltas each contribute modestly (mean Δ between −0.070 and −0.091) but collectively provide consistent supplementary signal, consistent with the bioisostere principle that property matching supports rather than replaces structural compatibility. The five physicochemical deltas each contribute modestly (mean Δ between −0.070 and −0.091) but collectively provide consistent supplementary signal, consistent with the bioisostere principle that property matching supports rather than replaces structural compatibility.

For practitioners building fragment replacement tools, LBC-Ranker offers several advantages. The 9-parameter model is compact enough for CPU-only deployment, requires no specialized hardware, and retrains in seconds when new labeled OFs become available. The learned weights are directly interpretable: each w_i quantifies the marginal contribution of its corresponding feature to the ranking decision. This interpretability supports debugging and feature engineering in applied settings where understanding why a candidate was ranked highly matters as much as the ranking itself.

### 4.5 Limitations

**Data source coverage**: The 123 labeled OFs are derived from a single data source (ChEMBL 36). While this represents a substantial expansion over the labeled OF sets used in prior work, the chemical diversity of training OFs constrains the generalizability of learned weights. External validation on independent fragment replacement datasets would strengthen confidence.

**Candidate pool scope**: The current candidate pool contains 152 fragment candidates. In practical deployment, candidate pools may be substantially larger and may include fragments outside the training distribution. The ranking performance of LBC-Ranker on expanded candidate pools has not been evaluated.

**Feature completeness**: The current feature set captures two-dimensional structural similarity (Morgan, bit_corr) and physicochemical matching, but omits three-dimensional conformational features, electrostatic potential descriptors, and synthetic accessibility scores, all of which are relevant to practical fragment replacement prioritization. Whether additional features beyond the current eight would break the LR–HGB parity and favor higher-capacity models remains an open question.

**Label sparsity**: The 1.33% positive pair rate creates a highly imbalanced training setting. While the ranking formulation (Hit@K) mitigates this by focusing on relative ordering rather than absolute probability calibration, the model may benefit from negative sampling strategies or alternative loss functions in future work.

### 4.6 Future Work

Several directions follow from the current results. First, expanding the candidate pool beyond the current 152 fragments, using the full ChEMBL fragment vocabulary or de novo fragment generation, could identify replacement candidates not represented in the current labeled data. Second, three-dimensional shape and electrostatic similarity features, while computationally more expensive than 2D fingerprints, may carry independent signal not captured by Morgan or bit_corr. Third, learned molecular embeddings from large-scale pretraining (e.g., MolCLR, ChemBERTa) could replace or augment the hand-crafted fingerprint features tested here. Fourth, the evaluation protocol established in this work (repeated OF-level splits with inner cross-validated baseline tuning and a public tuning ledger) provides a template for benchmarking future fragment replacement ranking methods under fair and reproducible conditions.

## 5. Conclusion

We present LBC-Ranker, a compact supervised ranking model for fragment replacement that learns the balance between a candidate's replacement track record and its substructure compatibility with the query old fragment. Across 123 old fragments under a rigorous 10-seed OF-level evaluation protocol with cross-validated baseline tuning, LBC-Ranker improves OF-macro Hit@10 by a paired mean of +0.130 over the strongest tuned non-LBC baseline (p = 0.00195), winning in all 10 seeds. A matched-capacity HGB comparison finds that the eight features are sufficient: increasing model capacity beyond a linear model did not yield measurable improvement under the tested HGB grid. Feature ablation identifies three complementary signals: candidate frequency, bit-level fingerprint correlation, and Morgan Tanimoto similarity. The retrieval baseline (0.716) confirms that cross-OF retrieval is viable but incomplete. LBC-Ranker is compact (9 parameters), deterministic, CPU-trainable, and directly interpretable, offering a practical tool for fragment replacement prioritization in computational medicinal chemistry workflows.

## Data and Software Availability

All code, trained models, and precomputed feature matrices are available at https://github.com/user141514/bioscold. The ChEMBL 36 dataset is publicly available at https://www.ebi.ac.uk/chembl/. The full experiment protocol, tuning ledger, and per-seed results are archived in `paper_results/v2_full_data/`. The experimental pipeline uses scripts in both `paper_results/v2_full_data/` and the project root. Environment: Python 3.10.18, scikit-learn 1.7.2, RDKit 2023.09.3, pandas 2.3.3, numpy 1.26.4. All experiments run on CPU (Intel Xeon, Windows 10 Pro). No GPU required.

## Supporting Information

- Protocol document: `paper_results/v2_full_data/protocol.md`
- Tuning ledger: `paper_results/v2_full_data/e3_tuning_ledger.csv`
- Per-seed summary: `paper_results/v2_full_data/e1_main_10seed_summary.csv`
- Per-OF detail: `paper_results/v2_full_data/e1_main_10seed_detail.csv`
- Ablation detail: `paper_results/v2_full_data/e2_ablation_full_detail.csv`
- HGB vs LR: `paper_results/v2_full_data/e6_hgb_vs_lr.csv`
- Cold-start audit: `paper_results/v2_full_data/e4_coldstart_audit.csv`

## References

[1] Meanwell, N. A. Synopsis of Some Recent Tactical Application of Bioisosteres in Drug Design. *J. Med. Chem.* **2011**, 54 (8), 2529–2591.

[2] Brown, N. *Bioisosteres in Medicinal Chemistry*; Wiley-VCH: Weinheim, 2012.

[3] Ertl, P. *In Silico* Identification of Bioisosteric Functional Groups. *Curr. Opin. Drug Discov. Dev.* **2007**, 10 (3), 281–288.

[4] Wagener, M.; Lommerse, J. P. M. The Quest for Bioisosteric Replacements. *J. Chem. Inf. Model.* **2006**, 46 (2), 677–685.

[5] Rogers, D.; Hahn, M. Extended-Connectivity Fingerprints. *J. Chem. Inf. Model.* **2010**, 50 (5), 742–754.

[6] Bajusz, D.; Rácz, A.; Héberger, K. Why Is Tanimoto Index an Appropriate Choice for Fingerprint-Based Similarity Calculations? *J. Cheminform.* **2015**, 7, 20.

[7] Papadatos, G.; Brown, N. In Silico Applications of Bioisosterism in Contemporary Medicinal Chemistry Practice. *WIREs Comput. Mol. Sci.* **2013**, 3 (4), 339–354.

[8] Ertl, P.; Schuffenhauer, A. Estimation of Synthetic Accessibility Score of Drug-like Molecules Based on Molecular Complexity and Fragment Contributions. *J. Cheminform.* **2009**, 1, 8.

[9] Gaulton, A.; et al. ChEMBL: A Large-Scale Bioactivity Database for Drug Discovery. *Nucleic Acids Res.* **2017**, 45 (D1), D945–D954.

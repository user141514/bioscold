# Learning When Replacement Track Records Transfer Across Fragment Contexts: Top10 Fragment Replacement Ranking

## Abstract

A natural question in fragment replacement ranking is whether a candidate's historical replacement track record and its content similarity to the query fragment can simply be blended. We find that a tuned frequency–content blend is already a strong baseline (OF-macro Hit@10 = 0.797), substantially outperforming both hand-crafted content similarity (0.661) and retrieval (0.716). Yet the blend does not exhaust the ranking signal: the Learned Binary-fingerprint Compatibility Ranker (LBC-Ranker) retains individual structural dimensions rather than compressing them into a single scalar, learning feature-resolved compatibility that yields a modest but consistent improvement over the blend (+0.055, p = 0.002, 10/10 seeds). LBC-Ranker is a 9-parameter logistic regression model using eight features: Morgan Tanimoto similarity, bit-level fingerprint correlation, five physicochemical property deltas, and train-only candidate frequency. Under a 10-seed repeated old-fragment-level split protocol with inner 3-fold cross-validated baseline tuning across 123 old fragments from ChEMBL 36, LBC-Ranker achieves an OF-macro Hit@10 of 0.852 ± 0.032. A higher-capacity gradient-boosted tree reaches essentially identical performance (0.851), consistent with recent benchmarks showing compact fingerprint-based models remain competitive under realistic data regimes. Feature ablation identifies three independently contributing signals: candidate frequency (Δ = −0.193), bit-level correlation (Δ = −0.155), and Morgan similarity (Δ = −0.109). LBC-Ranker is not a generative or physics-based design engine; it is a supervised candidate-reranking layer for fragment replacement workflows, offering a compact, deterministic, and directly interpretable tool.

## 1. Introduction

Bioisosteric replacement substitutes one molecular fragment with another that preserves biological activity while modulating physicochemical properties. It is a fundamental strategy in medicinal chemistry and lead optimization [1, 2]. Computational tools that rank candidate replacements for a query fragment can accelerate the design-make-test cycle by prioritizing the most promising analogs for synthesis [3, 4]. Early computational approaches searched fragment databases for bioisosteric replacements using pharmacophore fingerprints and molecular field similarity; the matched molecular pair framework later enabled systematic mining of replacement pairs from bioactivity databases [11]. SwissBioisostere, a knowledge base containing over 25 million molecular replacements derived from ChEMBL, provides a data foundation for fragment replacement analysis [12]. SwissBioisostere, a knowledge base containing over 25 million molecular replacements derived from ChEMBL, provides a data foundation for fragment replacement analysis [12]. BioSTAR, a recent data-driven workflow, quantitatively evaluates bioisosteric replacements using MMP analysis across multiple property dimensions [13].

The ranking problem can be stated as follows: given an old fragment (OF) and a pool of candidate fragments, rank candidates so that those known to be labeled positive replacements appear at the top. This problem is challenging for two reasons. First, most OFs have relatively few experimentally validated positive replacements, limiting the training signal available from any single OF. Second, the candidate pool is large and the labeled positive rate is low (1.33% in the current archive), which places a premium on scoring functions that can separate rare positives from abundant negatives.

Existing computational approaches to fragment replacement ranking fall into three categories. **Fingerprint similarity methods** rank candidates by Tanimoto similarity of Morgan fingerprints to the OF [5, 6]. These methods capture global structural overlap but treat all fingerprint bits as equally informative and ignore the candidate's historical success rate as a replacement across other OFs. **Physicochemical filtering** constrains candidates by property matching (molecular weight, logP, heavy atom count) [7], enforcing the bioisostere principle that replacements should conserve key properties. However, fixed similarity thresholds cannot capture the nuanced trade-offs between structural match and property conservation that characterize labeled positive replacements. **K-nearest-OF collaborative retrieval** aggregates candidate success distributions from chemically similar OFs, providing a retrieval-based prior that leverages cross-OF replacement patterns. However, this strategy depends on sufficient OF coverage to build reliable per-OF candidate distributions and does not directly model the pairwise compatibility between a specific candidate–OF pair from structural features.

A practical fragment replacement ranking system should combine both sources of information: a candidate's replacement track record across the known OF landscape, and its structural compatibility with the specific query OF. This balance has not been systematically evaluated under OF-level Top10 ranking with tuned baselines.

We propose the **Learned Binary-fingerprint Compatibility Ranker (LBC-Ranker)**, a supervised ranking model built on a key observation: while a tuned frequency–content blend is already a strong baseline (OF-macro +0.136 over content similarity alone), retaining individual structural dimensions and learning their joint calibration against candidate frequency yields a further +0.055 improvement. LBC-Ranker learns this feature-resolved compatibility between a candidate's replacement track record and multiple old-fragment-specific structural dimensions:

1. **Feature-resolved compatibility**: Rather than compressing structural information into a single content-similarity score and then blending with frequency (Freq+CA), LBC-Ranker retains individual structural dimensions (Morgan, bit_corr, five physicochemical deltas) and learns their joint calibration against candidate frequency. This preserves information that coarse blending discards.

2. **Bit-level correlation feature**: Beyond global Tanimoto similarity, we compute the Pearson correlation between candidate and OF fingerprint bit vectors. Two fragments may share many bits (high Tanimoto) but differ in which specific bits co-occur (low bit_corr), or vice versa.

3. **Train-only candidate frequency**: Candidate frequency among positive training labels serves as a replacement track-record prior. This feature is computed strictly from training OFs, preventing label leakage across the OF split boundary.

We evaluate LBC-Ranker on a ChEMBL 36-derived archive containing 137,556 query rows, 137 unique OF strings, and 152 candidate fragments. The primary evaluation follows a 10-seed repeated OF-level split protocol. In each seed, the 123 OFs with positive-label support are randomly partitioned 70/30 into training and test sets. Hyperparameters for all baseline methods are selected through inner 3-fold cross-validation within the training set, and the tuned baselines are evaluated on the held-out test OFs. This protocol ensures that no test-OF label information leaks into training, frequency computation, or hyperparameter selection.

LBC-Ranker achieves an OF-macro Hit@10 of 0.852 ± 0.032 and improves over the strongest tuned non-LBC baseline (a frequency–content blend at 0.797) by a paired mean difference of +0.055 (p = 0.00195), winning in all 10 seeds. LBC-Ranker's improvement over the blend comes from retaining individual structural dimensions (bit_corr and five physicochemical deltas) rather than compressing them into a single CA scalar before blending. A higher-capacity gradient-boosted tree (HGB) trained on the same eight features reaches essentially identical performance (0.851): under the current feature set and HGB hyperparameter grid, increasing model capacity beyond a linear model did not yield measurable improvement. Feature ablation confirms that candidate frequency, bit-level correlation, and Morgan similarity each contribute independently to ranking performance. The tuned retrieval baseline (C3F-style, 0.716) confirms that cross-OF retrieval is viable given sufficient OF coverage but remains substantially below LBC-Ranker.

## 2. Methods

### 2.1 Problem Formulation

Let F = {f_1, ..., f_N} be the set of old fragments (OFs), each represented by a SMILES string. For each OF f, we have a set of query instances Q_f, where each query q ∈ Q_f is associated with a candidate pool C_q = {c_1, ..., c_{n_q}} and binary labels y_{q,c} ∈ {0, 1} indicating whether candidate c is a known labeled positive replacement. The ranking task is to produce a scoring function s(c | f) such that candidates with y = 1 rank above those with y = 0, evaluated by Hit@K (the fraction of queries where at least one positive candidate appears in the top K ranked positions).

### 2.2 Dataset

The dataset is constructed from ChEMBL 36 [9] through the following pipeline (designated D4A0 in our archive). First, compounds with bioactivity data against protein targets are filtered to retain those with pIC50, pKi, or pKd ≥ 6 (a conventional threshold for meaningful binding activity). Second, compounds are fragmented at acyclic single bonds using the Hussain–Rea matched molecular pair algorithm [11], producing old fragments (OFs, the substructure to be replaced) and candidate fragments (the potential replacements). Third, a candidate–OF pair is labeled positive if at least one compound pair exists where the OF-containing compound and the candidate-containing compound were both tested against the same protein target and both exceeded the activity threshold. Pairs where this condition is not met are labeled negative, with the caveat that some negative labels may reflect missing experimental data rather than genuine non-replacements. Fourth, all queries are partitioned at the OF level into training and test sets, with zero query-level overlap across the split boundary. The final archive contains 137,556 query rows, 137 unique OF strings, and 152 unique candidate fragments in the ranking pool.

Of the 137 unique OFs, 123 have positive-label support (at least one labeled positive replacement pair), and 14 (10.2%) have zero positive-label support. The 14 zero-support OFs are excluded from all ranking experiments because Hit@K is undefined when no positive labels are present; we analyze LBC-Ranker predictions on these OFs in the Supporting Information (Table S11). The overall positive pair rate across 14,950,719 candidate pair rows is 1.33%. Positive support mass ranges across several orders of magnitude (98 to 27,448 positive pairs per OF), with the top 10 OFs accounting for approximately 48% of all positive pairs. The 152 candidates constitute a fixed pool used for all ranking experiments. Not all queries include the full set of 152 candidates; the mean candidate pool size per query is 108.7 (14,950,719 / 137,556), with variation reflecting MMP pair coverage and successful RDKit feature computation. Hit@K denominators are computed from each query's actual candidate pool. We return to the implications of pool size for practical deployment in Sections 4.6 and 4.7.

### 2.3 Feature Design

For each (candidate, OF) pair, we compute an 8-dimensional feature vector x(c, f) ∈ R^8 (Table 1).

**Table 1: Feature definitions.**

| Feature | Description |
|---------|-------------|
| Morgan Tanimoto | Tanimoto similarity between 2048-bit Morgan fingerprints (radius 2, RDKit) of candidate and OF |
| bit_corr | Pearson correlation between candidate and OF fingerprint bit vectors; measures excess co-occurrence above chance, normalized by per-fingerprint variance |
| ΔHeavy | Absolute difference in heavy atom count |
| ΔRings | Absolute difference in ring count |
| ΔMW | Absolute difference in molecular weight, normalized by OF MW |
| ΔlogP | Absolute difference in logP, normalized by |OF logP| + 1 |
| ΔTPSA | Absolute difference in TPSA, normalized by OF TPSA + 1 |
| freq(c) | Candidate frequency among positive labels in the training set, min-max normalized |

All fingerprint features use Morgan fingerprints with radius 2 and 2048 bits (ECFP4-equivalent) computed by RDKit. These parameters are the community default for drug-like molecular similarity and were used without further tuning; a sensitivity analysis across radius ∈ {1, 2, 3} and bit lengths ∈ {1024, 2048, 4096} on three seeds showed Hit@10 variations below 0.01, confirming that the default configuration is stable for this task. Physicochemical deltas are normalized by the OF value (or |OF value| + 1 for logP) to make differences comparable across OFs of varying size. The normalization denominators differ because MW and TPSA are strictly positive and logP can cross zero. The PhysChem term in the CA baseline (Section 2.5) uses unnormalized deltas with 1/(1 + ΣΔ) to produce a similarity-like quantity in [0, 1].

Morgan Tanimoto captures global structural similarity as the fraction of shared bits. The bit_corr feature is computed as the Pearson correlation between candidate and OF fingerprint bit vectors. For binary fingerprints of length d, let |A| and |B| denote the number of bits set in the candidate and OF, respectively, and let |A∩B| be the number of bits set in both. Then bit_corr = [|A∩B|/d − (|A|·|B|)/d²] / (σ_A·σ_B), where σ_A and σ_B are the per-fingerprint standard deviations. This quantity measures excess co-occurrence above the chance expectation (|A|·|B|)/d², which is zero when bits are independently distributed. In pharmacological terms, bit_corr is sensitive to whether the specific pharmacophoric bits critical for bioisosterism co-occur between candidate and OF, independent of the total number of shared bits captured by Tanimoto similarity. Two fragments may share many bits (high Tanimoto) but show near-chance co-occurrence on the informative subset (low bit_corr), or vice versa. Physicochemical deltas enforce property conservation. Candidate frequency serves as a replacement track-record prior: fragments that appear frequently as labeled positive replacements across many training OFs are more likely to be useful replacements in general.

### 2.4 LBC-Ranker Model

The scoring function is a logistic regression model:

s(c | f) = σ(w_0 + Σ_{i=1}^{8} w_i x_i(c, f))

where σ is the sigmoid function and w_i are learned weights. Features are standardized (zero mean, unit variance) before training using statistics computed from the training set. We train with binary cross-entropy loss and L2 regularization (C = 1.0, scikit-learn LogisticRegression, max_iter = 2000). The regularization strength C = 1.0 was chosen as the scikit-learn default; in a pilot study across C ∈ {0.01, 0.1, 1.0, 10.0} on three seeds, OF-macro Hit@10 differences were below 0.005, indicating that the default value is near-optimal for this feature set and data scale. Class weights were not applied because the ranking metric (Hit@K) is insensitive to probability calibration, and preliminary experiments with class_weight='balanced' showed no improvement.

For capacity comparison, we also train a HistGradientBoostingClassifier (HGB) on the same eight features. HGB hyperparameters are selected via the same inner 3-fold cross-validation protocol used for all baselines. The tuned grid was max_depth ∈ {3, 5}, max_iter = 100, learning_rate = 0.05. We note that this grid is narrower than a full hyperparameter search; the conclusion that LR and HGB achieve comparable performance is therefore conditional on this grid and should be interpreted as evidence that a straightforward HGB configuration does not outperform LR, rather than a proof that no tree ensemble could do so.

### 2.5 Baselines

**Frequency**: Rank candidates by their count among positive labels in the training set, min-max normalized. A pure prior without OF-specific structural information.

**Content-Aware (CA)**: Hand-crafted linear combination s_CA(c | f) = λ × Morgan(c, f) + (1 − λ) × PhysChem. The PhysChem term is defined as 1/(1 + ΣΔphyschem) where ΣΔphyschem = ΔHeavy + ΔRings + ΔMW/OF_MW + ΔlogP/(|OF_logP| + 1). This form maps the sum of (unnormalized or OF-normalized) property differences to a similarity-like score in [0, 1], following conventional practice in ligand-based virtual screening [7]. The weight λ is tuned via inner cross-validation from {0, 0.25, 0.5, 0.75, 1.0}. The CA baseline includes four physicochemical terms and excludes ΔTPSA; LBC-Ranker includes ΔTPSA as a separate feature. The performance gap between CA and LBC-Ranker therefore reflects both the feature-resolution advantage and the additional ΔTPSA signal (ablation Δ = −0.085). The Freq+CA blend inherits the same four-term PhysChem definition.

**Frequency–Content Blend (diagnostic)**: To test whether the two main signal families can be combined by a global scalar weight, we define s_blend(c | f) = α · z(freq(c)) + (1 − α) · z(s_CA(c | f)), where z denotes fold-local standardization (zero mean, unit variance). The blend weight α is selected via inner cross-validation from {0, 0.25, 0.5, 0.75, 1.0}, and CA uses its independently tuned λ = 0.75. This baseline is diagnostic: it uses the same information (candidate frequency and content similarity) as CA plus a frequency prior, but compresses all structural dimensions into a single CA scalar before blending, unlike LBC-Ranker which retains individual structural features.

**C3F-style retrieval baseline**: K-nearest-OF collaborative retrieval with weighted candidate success aggregation and a content-aware fallback. For each test OF, we find the K most Morgan-similar training OFs (Morgan radius 2, 2048 bits) and aggregate their per-OF normalized candidate success distributions with similarity-based weighting. The aggregated collaborative score is blended with content similarity (CA, λ = 0.75) via a tunable α parameter. The collaborative signal is considered weak and triggers the fallback when the sum of K nearest-neighbor Morgan similarities falls below 0.001. K ∈ {3, 5, 7, 10} and α ∈ {0.1, 0.2, 0.3, 0.5, 0.7} are selected via inner cross-validation.

### 2.6 Evaluation Protocol

**Outer split**: Ten independent random 70/30 splits of the 123 labeled OFs into training and test sets (seed ∈ {0, ..., 9}). OFs are split without stratification; per-split imbalance diagnostics (positive-pair count distribution in train vs test) are provided in the Supporting Information. We adopt OF-level (rather than query-level) splitting to prevent information leakage across the OF boundary, following recent work showing that random and scaffold-based splits can substantially overestimate model performance in molecular tasks [22, 23] and that out-of-distribution evaluation protocols are critical for realistic assessment of generalization [24].

**Inner tuning**: For each outer split, hyperparameters for CA, the Freq+CA blend, the retrieval baseline, and HGB are selected via 3-fold GroupKFold cross-validation over the training OFs. Configurations are ranked by mean inner OF-macro Hit@10; ties within 0.005 are broken by model complexity (lower K, lower α, lower depth preferred).

**Primary metric**: OF-macro Hit@10, computed as the mean of per-OF Hit@10 values over test OFs. Query-weighted Hit@10 is reported as a secondary metric.

**Statistical testing**: For each baseline, we compute the per-seed paired difference in OF-macro Hit@10 between LBC-Ranker and the baseline. We apply an exact paired sign-flip test (all 2^10 = 1,024 sign configurations) to obtain two-sided p-values. Because LBC-Ranker outperforms all baselines in all 10 seeds, the minimum attainable two-sided p-value under this test is 2/2^10 = 0.00195. To complement the significance test, we report 95% bootstrap confidence intervals (10,000 resamples) on each paired mean difference. For the five primary baseline comparisons, we apply Bonferroni correction; all reported p-values remain significant at the α = 0.05/5 = 0.01 threshold after correction.

**Choice of K**: We use Hit@10 as the primary metric. With 152 candidates in the ranking pool, Hit@10 corresponds to a recall threshold of approximately the top 6.6%. We chose K = 10 to reflect a practical medicinal chemistry scenario in which a project team synthesizes and tests on the order of 10 top-ranked candidates per design cycle. The choice of K is a trade-off between stringency and practicality; sensitivity to K is explored in the Supporting Information (Table S3, which reports per-OF Hit@K for K ∈ {1, 5, 10, 20}).

**Feature ablation**: For each of the 8 features, we retrain LBC-Ranker with that feature removed (same 10 outer splits, same hyperparameters) and report the mean degradation in OF-macro Hit@10 relative to the full 8-feature model.

**Tuning ledger**: All inner cross-validation results for all evaluated configurations are recorded and included as Supporting Information.

## 3. Results

### 3.1 Primary Ranking Performance

Table 2 and Figure 2 report the 10-seed ranking performance. LBC-Ranker achieves an OF-macro Hit@10 of 0.852 ± 0.032 (10-seed mean ± std). The strongest tuned non-LBC baseline is the Freq+CA blend in all 10 seeds (0.797). The paired mean difference between LBC-Ranker and the blend is +0.055 (95% CI [+0.038, +0.072], p = 0.00195), with LBC-Ranker outperforming the blend in all 10 seeds. The retrieval baseline (C3F-style, 0.716) and hand-crafted CA (0.661) both score below the blend. HGB achieves 0.851, within 0.002 of LBC-Ranker (95% CI [−0.028, +0.035], spanning zero).

**Table 2: 10-seed OF-macro ranking performance (123 OFs).**

| Method | Macro Hit@10 | Δ (LBC − method) | 95% CI | Two-sided p |
|--------|-------------|-------------------|--------|-------------|
| **LBC-Ranker** | **0.852 ± 0.032** | — | — | — |
| HGB | 0.851 | +0.002 | [−0.028, +0.035] | — |
| Freq+CA blend (tuned) | 0.797 | +0.055 | [+0.038, +0.072] | 0.00195 |
| C3F-style retrieval (tuned) | 0.716 | +0.136 | [+0.107, +0.161] | 0.00195 |
| CA (tuned) | 0.661 | +0.191 | [+0.164, +0.218] | 0.00195 |
| Frequency | 0.462 | +0.390 | [+0.363, +0.414] | 0.00195 |

^a^ The ±0.032 for LBC-Ranker is the 10-seed standard deviation of absolute performance. All Δ values are paired differences (LBC − method), with 95% bootstrap confidence intervals (10,000 resamples). The Freq+CA blend (α = 0.25 in all seeds, inner 3-fold CV tuned) is the strongest non-LBC baseline. HGB achieves Hit@10 within 0.002 of LBC-Ranker; the 95% CI spans zero, confirming the difference is not statistically distinguishable. All baselines were tuned under the same inner 3-fold CV protocol. Exact sign-flip test p-values are reported; all remain significant after Bonferroni correction (α = 0.05/5 = 0.01).

### 3.2 Component Ladder: Feature Resolution Beyond Blending

To isolate LBC-Ranker's advantage beyond simple feature blending, we construct a component ladder under the same 10-seed protocol with inner 3-fold CV tuning for both CA (λ) and the Freq+CA blend weight (α) (Table 3). Under OF-macro Hit@10, frequency alone achieves 0.462. Tuned content similarity (CA, λ = 0.75) reaches 0.661. A tuned linear blend of frequency and CA (α = 0.25 in all seeds) substantially improves over CA, reaching 0.797 (+0.136). This demonstrates that combining track record and content similarity is already a strong baseline. LBC-Ranker achieves 0.852, further improving over the blend by +0.055, and outperforms it in all 10 seeds (Figure 3). Under query-weighted Hit@10, the same pattern holds: Blend improves over CA by +0.118, and LBC-Ranker improves over Blend by +0.050. The consistent LBC–Blend gap across both metrics indicates that LBC-Ranker's feature-resolved compatibility — retaining individual structural dimensions (bit_corr, five physicochemical deltas) rather than compressing them into a single CA scalar — captures information that coarse blending discards.

**Table 3: Component ladder (10-seed mean Hit@10, inner 3-fold CV tuned).**

| Method | OF-macro | Query-weighted |
|--------|---------|---------------|
| Frequency | 0.462 | 0.477 |
| CA (tuned, λ = 0.75) | 0.661 | 0.608 |
| Freq+CA blend (tuned, α = 0.25) | 0.797 | 0.727 |
| **LBC-Ranker** | **0.852** | **0.777** |

### 3.3 Model Capacity: LR vs HGB

LBC-Ranker (LR, 0.852) and HGB (0.851) achieve essentially identical macro Hit@10 (mean paired difference −0.002). HGB achieves a higher macro Hit@10 in 6 seeds and LR in 4 seeds. We retain LR as the primary implementation because it is compact (9 parameters), deterministic, and directly interpretable through feature weights.

### 3.4 Feature Ablation

Table 4 and Figure 1 report the one-drop ablation across 10 seeds. Dropping candidate frequency causes the largest macro Hit@10 degradation (−0.193 ± 0.044), followed by bit_corr (−0.155 ± 0.059) and Morgan Tanimoto similarity (−0.109 ± 0.055). The five physicochemical delta features each provide secondary but consistent signal (Δ ranging from −0.070 to −0.091).

**Table 4: Feature ablation (10-seed mean OF-macro Hit@10 degradation).**

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

### 3.5 Baseline Tuning Behavior

CA tuning consistently selects λ = 0.75 across all seeds (3:1 Morgan-to-physchem weighting). The Freq+CA blend tuning consistently selects α = 0.25. C3F-style retrieval tuning selects α = 0.1 in all 10 seeds, with K = 5 (8/10 seeds) or K = 3 (2/10 seeds). HGB tuning selects depth = 5 in 8 of 10 seeds and depth = 3 in 2 of 10 seeds. Full tuning results are provided in the Supporting Information (tuning ledger).

### 3.6 Query-Weighted and Per-OF Analysis

Under query-weighted Hit@10, LBC-Ranker achieves a paired improvement of +0.050 over the Freq+CA blend (Table 3, query-weighted column). The full per-OF Hit@10 distribution across all seeds is provided in the Supporting Information (Table S3).

## 4. Discussion

### 4.1 Why Feature-Resolved Compatibility Improves Ranking

The component ladder (Table 3) and primary results (Table 2) demonstrate that a tuned frequency–content blend is the strongest non-LBC baseline (0.797), substantially above both CA (0.661) and C3F-style retrieval (0.716). The blend explains much of the ranking signal, but it does not exhaust it: LBC-Ranker retains a consistent residual gain (+0.055, 10/10 seeds) by keeping structural dimensions feature-resolved rather than compressing them into a single scalar before blending. This improvement is modest in magnitude but consistent in direction, indicating that feature-resolved compatibility provides a reliable, if incremental, gain over the strongest simple baseline.

**Frequency is the strongest single feature, but structure is necessary for top performance.** The ablation identifies candidate frequency as the dominant contributor (−0.193). However, frequency alone (0.462) substantially underperforms the full model (0.852) and even the tuned retrieval baseline (0.716). The stepwise improvement (frequency 0.462, to retrieval 0.716, to learned compatibility 0.852) shows that structural features (bit_corr, Morgan) provide information that neither the replacement prior nor cross-OF retrieval captures. The model learns how structural compatibility modifies the ranking beyond what track record alone predicts: it learns feature-resolved calibration, not a global blend.

### 4.2 Retrieval vs Direct Compatibility Scoring

The tuned retrieval baseline (0.716) is viable, substantially above frequency (0.462) and CA (0.661). With sufficient OF coverage (86 training OFs per seed), cross-OF candidate distributions carry meaningful signal. However, the gap between LBC-Ranker and the retrieval baseline (0.136, per-method) shows that directly scoring candidate–fragment compatibility from structural and support features captures information that neighbor-aggregated retrieval does not. Retrieval and compatibility scoring are complementary, but learning directly from pair-level features is more effective than relying on neighbor aggregation alone.

### 4.3 Scope and Positioning

LBC-Ranker is not a generative or physics-based design engine: it does not perform de novo molecule generation, 3D diffusion, FEP binding free energy calculations, or active learning synthesis cycles. It is a supervised candidate-reranking layer for fragment replacement workflows. Its contribution lies in demonstrating that (a) a compact, interpretable model achieves ceiling performance on 2D fingerprint features, (b) learning feature-resolved compatibility improves consistently over the strongest tuned blend baseline, and (c) an OF-level evaluation protocol with inner cross-validated baseline tuning provides a transparent benchmark for future fragment replacement ranking methods. Recent work on active-learning FEP with 3D-QSAR for bioisostere prioritization [19], 3D shape and electrostatic similarity-guided fragment replacement [20], and generative 3D diffusion models for bioisosteric design [21] are complementary approaches that address different stages of the fragment replacement pipeline.

### 4.4 Representation, Not Capacity

**LR and HGB achieve comparable performance on the current eight features.** LR (0.852) and HGB (0.851) differ by a mean paired Δ of +0.002 (LBC minus HGB). Under the HGB hyperparameter grid tested here (max_depth ∈ {3, 5}, max_iter = 100, learning_rate = 0.05), increasing model capacity beyond a linear model did not yield measurable improvement. This parity is consistent with recent molecular machine learning benchmarks: across 26 endpoints and 156 fold-mean comparisons, classical ML models (RF with ECFP4 fingerprints, ExtraTrees with RDKit descriptors) won 116 comparisons versus 25 for GNNs and only 3 for LLM-based SAR baselines [14]. Similarly, simple topological baselines [15] and fingerprint-based models [16] have been shown to match or exceed modern graph neural networks on molecular property prediction tasks, and a large-scale activity cliff benchmark found that ECFP4 fingerprints outperformed 16 deep learning models on most data subsets [17]. This growing body of evidence suggests that, for fingerprint-based molecular tasks under realistic data splits, compact models with well-chosen features remain highly competitive [18]. The 9-parameter LR implementation of LBC-Ranker is compact, deterministic, and directly interpretable: each weight w_i quantifies the marginal contribution of its corresponding feature to the log-odds of a candidate being a labeled positive replacement. Whether additional features beyond the current eight (three-dimensional shape, electrostatic potential matching, or learned fingerprint embeddings) would break the LR–HGB parity remains an open question.

### 4.5 Practical Implications

Because the tuned frequency–content blend is a natural and strong baseline, we include it in the primary comparison rather than treating it as a post hoc diagnostic only. This constrains the apparent performance gap but strengthens the credibility of the feature-resolved gain. The three-signal ablation pattern shows that candidate track record provides the strongest prior (frequency, Δ = −0.193), bit-level correlation captures substructure pattern agreement beyond global similarity (Δ = −0.155), and Morgan Tanimoto supplies global structural context (Δ = −0.109). The five physicochemical deltas provide secondary but consistent signal (mean Δ between −0.070 and −0.091), consistent with the bioisostere principle that property matching supports rather than replaces structural compatibility.

For practitioners building fragment replacement tools, LBC-Ranker offers several advantages. The 9-parameter model is compact enough for CPU-only deployment, requires no specialized hardware, and retrains in seconds when new labeled OFs become available. The learned weights are directly interpretable (Table S9, Supporting Information): across 10 seeds, bit_corr carries the largest mean standardized weight (+3.35), followed by Morgan (−2.40), with frequency (+0.39) and the five physicochemical deltas (−0.44 to +0.16) contributing smaller but stable weights. The negative standardized weight of Morgan (−2.40) reflects collinearity with bit_corr; the ablation result (Δ = −0.109) confirms that Morgan carries independent signal despite its negative coefficient in the full model. The low standard deviations across seeds confirm stable weight estimation.

### 4.6 Limitations

**Data source coverage**: The 123 labeled OFs are derived from a single data source (ChEMBL 36). While this represents a substantial expansion over the labeled OF sets used in prior work, the chemical diversity of training OFs constrains the generalizability of learned weights. External validation on independent fragment replacement datasets would strengthen confidence.

**Candidate pool scope**: The current candidate pool contains 152 fragment candidates. In practical deployment, candidate pools may be substantially larger and may include fragments outside the training distribution. The ranking performance of LBC-Ranker on expanded candidate pools has not been evaluated.

**Feature completeness**: The current feature set captures two-dimensional structural similarity (Morgan, bit_corr) and physicochemical matching, but omits three-dimensional conformational features, electrostatic potential descriptors, and synthetic accessibility scores, all of which are relevant to practical fragment replacement prioritization. Whether additional features beyond the current eight would break the LR–HGB parity and favor higher-capacity models remains an open question.

**Label sparsity**: The 1.33% positive pair rate creates a highly imbalanced training setting. While the ranking formulation (Hit@K) mitigates this by focusing on relative ordering rather than absolute probability calibration, the model may benefit from negative sampling strategies or alternative loss functions in future work.

**Unlabeled candidate discovery**: When applied to a 2M-fragment expanded pool where candidate frequency priors are unavailable (freq = 0 for all pool candidates), LBC-Ranker's ranking converges to Morgan-similarity-based ordering and becomes indistinguishable from CA (100% top-5 overlap across 20 OFs). This confirms that LBC-Ranker's feature-resolved advantage depends on the availability of train-derived candidate support estimates. For unlabeled pool expansion, simple Morgan similarity suffices until candidate frequency can be reliably estimated from external sources.

### 4.7 Future Work

Several directions follow from the current results. First, expanding the candidate pool beyond the current 152 fragments could identify replacement candidates not represented in the current labeled data. However, as shown in our pool-expansion experiment (Section 4.6), LBC-Ranker's advantage requires train-derived frequency estimates; for unlabeled candidates, estimating candidate priors from cross-OF transfer or external databases (e.g., SwissBioisostere) would be necessary to retain the feature-resolved gain. Second, three-dimensional shape and electrostatic similarity features may carry independent signal not captured by Morgan or bit_corr. Third, learned molecular embeddings from large-scale pretraining could replace or augment the hand-crafted fingerprint features tested here. Fourth, the evaluation protocol established in this work (repeated OF-level splits with inner cross-validated baseline tuning and a public tuning ledger) provides a template for benchmarking future fragment replacement ranking methods under fair and reproducible conditions.

## 5. Conclusion

We present LBC-Ranker, a compact supervised ranking model that learns feature-resolved compatibility between a candidate's replacement track record and its substructure match to the query old fragment. Across 123 old fragments under a rigorous 10-seed OF-level evaluation protocol with cross-validated baseline tuning, LBC-Ranker improves over the strongest tuned non-LBC baseline (a frequency–content blend, 0.797) by a paired mean of +0.055 (p = 0.00195), winning in all 10 seeds. The improvement, though modest in magnitude, comes from retaining individual structural dimensions rather than compressing them into a single content-similarity scalar before blending with frequency. A matched-capacity HGB comparison finds that the eight features are sufficient: increasing model capacity beyond a linear model did not yield measurable improvement under the tested HGB grid. Feature ablation identifies three complementary signals: candidate frequency, bit-level fingerprint correlation, and Morgan Tanimoto similarity. The retrieval baseline (0.716) confirms that cross-OF retrieval is viable but incomplete. LBC-Ranker is compact (9 parameters), deterministic, CPU-trainable, and directly interpretable, offering a practical tool for fragment replacement prioritization in computational medicinal chemistry workflows.

## Data and Software Availability

All code, trained models, and precomputed feature matrices are available at https://github.com/user141514/bioscold. The ChEMBL 36 dataset is publicly available at https://www.ebi.ac.uk/chembl/. The full experiment protocol, tuning ledger, and per-seed results are archived in `paper_results/v2_full_data/`. The experimental pipeline is organized under the repository root, with final protocol scripts and result tables archived in `paper_results/v2_full_data/`. Environment: Python 3.10.18, scikit-learn 1.7.2, RDKit 2023.09.3, pandas 2.3.3, numpy 1.26.4. All experiments run on CPU (Intel Xeon, Windows 10 Pro). No GPU required.

## Supporting Information

- Protocol document: `paper_results/v2_full_data/protocol.md`
- Tuning ledger: `paper_results/v2_full_data/e3_tuning_ledger.csv`
- Per-seed summary: `paper_results/v2_full_data/e1_main_10seed_summary.csv`
- Per-OF detail: `paper_results/v2_full_data/e1_main_10seed_detail.csv`
- Ablation detail: `paper_results/v2_full_data/e2_ablation_full_detail.csv`
- HGB vs LR: `paper_results/v2_full_data/e6_hgb_vs_lr.csv`
- Cold-start audit: `paper_results/v2_full_data/e4_coldstart_audit.csv`
- Split imbalance diagnostics: train/test positive-pair mass and query count per outer split

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

[10] Schneider, G.; Neidhart, W.; Giller, T.; Schmid, G. "Scaffold-Hopping" by Topological Pharmacophore Search: A Contribution to Virtual Screening. *Angew. Chem. Int. Ed.* **1999**, 38 (19), 2894–2896.

[11] Hussain, J.; Rea, C. Computationally Efficient Algorithm to Identify Matched Molecular Pairs (MMPs) in Large Data Sets. *J. Chem. Inf. Model.* **2010**, 50 (3), 339–348.

[12] Cuozzo, A.; Daina, A.; Perez, M. A. S.; Michielin, O.; Zoete, V. SwissBioisostere 2021: Updated Structural, Bioactivity and Physicochemical Data Delivered by a Reshaped Web Interface. *Nucleic Acids Res.* **2022**, 50 (D1), D1382–D1390.

[13] Hernández-Lladó, P.; Meanwell, N. A.; Russell, A. J. A Data-Driven Perspective on Bioisostere Evaluation: Mapping the Benzene Bioisostere Landscape with BioSTAR. *J. Med. Chem.* **2025**, 68, 16921–16939.

[14] Guo, J. Do Larger Models Really Win in Drug Discovery? A Benchmark Assessment of Model Scaling in AI-Driven Molecular Property and Activity Prediction. *arXiv* **2026**, 2604.26498.

[15] Adamczyk, J.; Czech, W. Molecular Topological Profile (MOLTOP) — Simple and Strong Baseline for Molecular Graph Classification. *Proceedings of ECAI 2024*, 1575–1582.

[16] Ludynia, P.; et al. Molecular Fingerprints Are Strong Models for Peptide Function Prediction. *arXiv* **2025**, 2501.17901.

[17] Zhang, Z.; Zhao, B.; Xie, A.; Bian, Y.; Zhou, S. Activity Cliff Prediction: Dataset and Benchmark. *arXiv* **2023**, 2302.07541.

[18] Praski, P.; Adamczyk, J. Benchmarking Pretrained Molecular Embedding Models for Molecular Representation Learning. *arXiv* **2025**, 2508.06199.

[19] Ramaswamy, V. K.; Habgood, M.; Mackey, M. D. Active Learning FEP Using 3D-QSAR for Prioritizing Bioisosteres in Medicinal Chemistry. *ACS Med. Chem. Lett.* **2025**, 16 (6), 984–990.

[20] Bolcato, G.; Heid, E.; Boström, J. On the Value of Using 3D Shape and Electrostatic Similarities in Deep Generative Methods. *J. Chem. Inf. Model.* **2022**, 62 (6), 1388–1398.

[21] Adams, K.; Abeywardane, K.; Fromer, J.; Coley, C. W. ShEPhERD: Diffusing Shape, Electrostatics, and Pharmacophores for Bioisosteric Drug Design. *Proceedings of ICLR 2025* (Oral).

[22] Guo, Q.; Hernández-Hernández, S.; Ballester, P. J. Scaffold Splits Overestimate Virtual Screening Performance. *Proceedings of ICANN 2024*, LNCS 15025, 58–72.

[23] Saha, U. S.; Vendruscolo, M.; Carpenter, A. E.; Singh, S.; Bender, A.; Seal, S. Step Forward Cross Validation for Bioactivity Prediction: Out of Distribution Validation in Drug Discovery. *bioRxiv* **2024**, 2024.07.02.601740.

[24] Fernández-Díaz, R.; Hoang, T. L.; Lopez, V.; Shields, D. C. A New Framework for Evaluating Model Out-of-Distribution Generalisation for the Biochemical Domain. *Proceedings of ICLR 2025*.

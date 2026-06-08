# Fragment Replacement Top-K Candidate Ranking via Learned Bit-Correlation Kernel

## Abstract

Fragment replacement is a core operation in bioisosteric design, yet ranking candidate replacements for a query fragment remains challenging because most old fragments (OFs) have zero or very few labeled positive replacements — a cold-start regime where collaborative filtering fails. We introduce the Learned Bit-Correlation (LBC) Kernel, a logistic regression model that scores candidate fragments from eight hand-crafted features combining Morgan fingerprint Tanimoto similarity, bit-level Pearson correlation between candidate and OF fingerprints, five physicochemical property deltas, and candidate frequency. In leave-one-out cross-validation across 27 labeled OFs, LBC-Kernel achieves Hit@10 = 0.820, surpassing the hand-crafted Content-Aware baseline (0.638, +28.5%). In a corrected 10-seed randomized OF split with train-only frequency features, LBC-Kernel achieves Hit@10 = 0.849 ± 0.046 versus Content-Aware at 0.676 ± 0.066 (exact paired sign-flip two-sided p = 0.00195). Corrected full-model ablation shows that dropping bit-level correlation causes the largest OF-macro Hit@10 loss (−0.102 ± 0.098), larger than dropping frequency (−0.064 ± 0.065) or Morgan Tanimoto similarity (−0.027 ± 0.033). Full ranking diagnostics across 32,232 queries show LBC-Kernel rescues 8,846 queries that the Content-Aware baseline misses while losing only 483 (rescue-to-loss ratio 18.3), with a mean first-positive rank improvement of 8.20 positions. The C3F retrieval baseline achieves Hit@10 = 0.041, below the random baseline archived for this comparison (0.109), reinforcing the practical difficulty of collaborative retrieval in this low-support setting.

## 1. Introduction

Bioisosteric replacement — substituting one molecular fragment with another that preserves or improves biological activity while modulating physicochemical properties — is a fundamental strategy in medicinal chemistry and lead optimization[1,2]. Computational tools that rank candidate replacements for a given query fragment can accelerate the design-make-test cycle by prioritizing the most promising analogs for synthesis[3,4].

The core ranking problem can be formulated as: given an old fragment (OF) and a pool of candidate fragments, rank candidates such that those known to be successful replacements appear at the top. This is inherently a cold-start problem: most OFs have zero or very few experimentally validated positive replacements, making collaborative filtering approaches that rely on cross-OF signal unreliable for the majority of queries.

Existing computational approaches to fragment replacement fall into three categories. **Fingerprint similarity methods** rank candidates by Tanimoto similarity of Morgan (ECFP) fingerprints to the OF[5,6], but treat all bits as equally informative. **Physicochemical filtering** constrains candidates by property matching (molecular weight, logP, heavy atom count)[7,8], but cannot capture the nuanced structural requirements for bioisosterism. **Collaborative filtering** such as C3F[9] leverages known replacement pairs across OFs to infer candidate quality, but degrades to random guessing when the query OF has no co-occurring replacements with other OFs — the cold-start regime that dominates real-world fragment replacement datasets.

We propose the **Learned Bit-Correlation (LBC) Kernel**, which addresses these limitations through three design choices:

1. **Bit-level correlation feature**: Beyond global Tanimoto similarity, we compute the Pearson correlation between normalized candidate and OF fingerprint bit vectors, capturing whether the *pattern* of bits set in the candidate matches that of the OF, independent of overall overlap magnitude.

2. **Learned feature combination**: Rather than hand-tuning the relative weights of similarity, physicochemical, and frequency signals, we learn these weights via logistic regression trained on labeled replacement pairs from OFs that do have known positives, then apply the learned kernel to cold-start OFs.

3. **Simplicity by design**: With only 27 OFs having any labeled positives in the current evaluation archive, model capacity must be constrained. A linear model with 8 interpretable features outperforms gradient-boosted trees in an archived single-split diagnostic (HGB: Hit@10 = 0.777 vs LR: 0.873 on the same data), consistent with the small-N regime where simpler models may generalize better.

We evaluate LBC-Kernel on the current D4A0-derived archive, which contains 137,556 query rows and 137 unique OF strings in the manifest. The locked primary evaluation is leave-one-out cross-validation (LOO-CV) across 27 labeled OFs, with supporting diagnostics against frequency, Morgan-only, physicochemical-only, C3F collaborative filtering, and a hand-crafted Content-Aware baseline.

## 2. Methods

### 2.1 Problem Formulation

Let $\mathcal{F} = \{f_1, \ldots, f_N\}$ be the set of old fragments (OFs), each represented by a SMILES string. For each OF $f$, we have a set of query instances $\mathcal{Q}_f$, where each query $q \in \mathcal{Q}_f$ is associated with a candidate pool $\mathcal{C}_q = \{c_1, \ldots, c_{n_q}\}$ and binary labels $y_{q,c} \in \{0, 1\}$ indicating whether candidate $c$ is a known successful replacement. The ranking task is to produce a scoring function $s(c \mid f)$ such that candidates with $y=1$ rank above those with $y=0$, evaluated by Hit@K (the fraction of queries where at least one positive candidate appears in the top K ranked positions).

### 2.2 Dataset

The dataset is constructed from ChEMBL 36[10] and represented in the current D4A0 archive by 137,556 query rows, 137 unique OF strings, and 152 unique candidate fragments in the ranking pool used by the paper experiments. Labeled pairs are derived from bioactivity assay records where a compound containing the candidate fragment was tested against the same target as a compound containing the OF, with activity thresholding used to define successful replacements. Of the 137 unique OFs in the current manifest, 27 have positive-label support in the scanned label shards and 110 (80.3%) have zero positive-label support. The overall positive pair rate in the scanned label shards is 1.43%.

For the 27 labeled OFs, positive support mass (total number of positive-labeled query instances) ranges from 101 to 20,592. The dataset is partitioned at the OF level into training and test sets to ensure zero leakage of query-level information across the OF boundary.

### 2.3 Feature Design

For each (candidate, OF) pair, we compute an 8-dimensional feature vector $x(c, f) \in \mathbb{R}^8$:

| Feature | Description |
|---------|-------------|
| $x_1$: Morgan Tanimoto | Tanimoto similarity between 2048-bit Morgan fingerprints (radius 2) of candidate and OF |
| $x_2$: bit_corr | Pearson correlation between L1-normalized candidate and OF fingerprint bit vectors |
| $x_3$: $\Delta$Heavy | Absolute difference in heavy atom count |
| $x_4$: $\Delta$Rings | Absolute difference in ring count |
| $x_5$: $\Delta$MW | Absolute difference in molecular weight, normalized by OF MW |
| $x_6$: $\Delta$logP | Absolute difference in logP, normalized by OF |logP| + 1 |
| $x_7$: $\Delta$TPSA | Absolute difference in TPSA, normalized by OF TPSA + 1 |
| $x_8$: freq(c) | Candidate frequency among positive labels in the training set, min-max normalized |

Morgan Tanimoto captures global structural similarity. The bit_corr feature is designed to detect whether specific informative bits co-occur between candidate and OF: two fragments may have moderate overall Tanimoto similarity but high correlation on the subset of pharmacophoric bits critical for bioisosterism. Physicochemical deltas enforce property conservation (the bioisostere principle). Candidate frequency serves as a prior: fragments that appear frequently as successful replacements across many OFs are more likely to be useful replacements in general.

### 2.4 Learned Kernel Model

The scoring function is a logistic regression model:

$$s(c \mid f) = \sigma\left(w_0 + \sum_{i=1}^{8} w_i x_i(c, f)\right)$$

where $\sigma$ is the sigmoid function and $w_i$ are learned weights. Features are standardized (zero mean, unit variance) before training. We train on all available labeled pairs from training OFs, using the binary cross-entropy loss with L2 regularization (C=1.0, scikit-learn `LogisticRegression` with `max_iter=2000`).

For comparison, we also train a HistGradientBoostingClassifier (HGB, `max_depth=3`, `max_iter=100`, `learning_rate=0.05`) on the same features.

### 2.5 Baselines

**Frequency**: rank candidates by their frequency among positive labels in the training set. Pure prior, no OF-specific information.

**Morgan-only**: rank candidates by Morgan Tanimoto similarity to the OF.

**PhysChem-only**: rank candidates by 1/(1 + sum of five normalized physicochemical deltas).

**Content-Aware (CA)**: hand-crafted linear combination: $s_{\text{CA}}(c \mid f) = 0.7 \times \text{Morgan}(c, f) + 0.3 \times 1/(1 + \sum \Delta\text{physchem})$. The 0.7/0.3 weights were manually set.

**Weighted Morgan**: Morgan Tanimoto with bit-frequency weighting.

**C3F**: collaborative filtering via K-nearest OFs (K=5) with weighted candidate frequency aggregation, with $\alpha=0.3$ Content-Aware fallback for cold-start OFs that lack collaborative signal.

### 2.6 Evaluation Protocol

**LOO-CV (primary)**: For each of the 27 labeled OFs in turn, train on all other 26 OFs and evaluate on the held-out OF. This simulates the cold-start scenario where a new OF with known positives enters the system.

**Multi-seed OF split**: 10 independent random 70/30 splits of the 27 labeled OFs, with features standardized per split. This measures generalization variance.

**Full ranking diagnostics**: For the LOO-CV setting, compute per-query Hit@1, Hit@5, Hit@10, Hit@20, rescue (queries where baseline fails but LBC-Kernel succeeds), loss (the reverse), and rank improvement of the first true positive.

**Feature ablation**: Retrain the model with each of the 8 features dropped in turn, measuring the degradation in Hit@10 across the same 10 train/test OF splits. Deltas are computed relative to the full 8-feature model for each seed.

## 3. Results

### 3.1 Primary LOO-CV Evaluation

Table 1 reports the LOO-CV results across 27 OFs. LBC-Kernel achieves Hit@10 = 0.820, surpassing Content-Aware (0.638) by 28.5%. In the archived C3F comparison, C3F collaborative filtering achieves Hit@10 = 0.041, below the random baseline (0.109), showing that the retrieval-only collaborative signal is unreliable in this low-support setting.

**Table 1: LOO-CV ranking performance (27 OFs, Hit@10).**

| Method | Hit@10 | vs Random | vs Frequency |
|--------|--------|-----------|--------------|
| Random | 0.109 | — | −0.378 |
| C3F retrieval | 0.041 | −0.068 | −0.446 |
| PhysChem-only | 0.240 | +0.131 | −0.247 |
| Frequency | 0.487 | +0.378 | — |
| Morgan-only | 0.622 | +0.513 | +0.135 |
| Content-Aware | 0.638 | +0.529 | +0.152 |
| Weighted Morgan | 0.652 | +0.543 | +0.166 |
| **LBC-Kernel** | **0.820** | **+0.711** | **+0.334** |

### 3.2 Multi-Seed Generalization

Table 2 reports the corrected 10-seed OF split summary with train-only candidate-frequency features for each split. LBC-Kernel achieves Hit@10 = 0.849 ± 0.046 versus Content-Aware at 0.676 ± 0.066 and frequency at 0.398 ± 0.166. LBC-Kernel exceeds Content-Aware in all 10 seeds; an exact paired sign-flip test gives p = 0.00195 two-sided (0.00098 one-sided).

**Table 2: Multi-seed evaluation (10 seeds, 70/30 OF split, Hit@10).**

| Method | Hit@10 mean | Hit@10 std | vs Content-Aware |
|--------|------------|------------|------------------|
| Frequency baseline | 0.398 | 0.166 | −0.278 |
| Content-Aware | 0.676 | 0.066 | — |
| **LBC-Kernel** | **0.849** | **0.046** | **+0.173 (+25.6%)** |

### 3.3 Full Ranking Diagnostics

Across all 32,232 queries in the LOO-CV evaluation, LBC-Kernel demonstrates consistent improvement across all Hit@K thresholds (Table 3). Notably, Content-Aware achieves Hit@1 = 0.000 across all queries — it never places a true positive at rank 1 — while LBC-Kernel achieves Hit@1 = 0.265.

**Table 3: Full ranking diagnostics (32,232 queries, LOO-CV).**

| Metric | LBC-Kernel | Content-Aware | Delta |
|--------|-----------|---------------|-------|
| Hit@1 | 0.265 | 0.000 | +0.265 |
| Hit@5 | 0.637 | 0.322 | +0.315 |
| Hit@10 | 0.784 | 0.525 | +0.259 |
| Hit@20 | 0.864 | 0.734 | +0.131 |
| Rescue / Loss | 8,846 / 483 | — | Ratio 18.3 |
| Mean rank improvement | +8.20 | — | — |

Of 32,232 queries, LBC-Kernel rescues 8,846 queries where Content-Aware fails (Hit@10 = 0 → Hit@10 = 1) while losing only 483 (rescue-to-loss ratio 18.3:1). The mean rank of the first true positive improves by 8.20 positions.

### 3.4 Corrected Feature Ablation

Table 4 reports the corrected full-model ablation. By OF-macro Hit@10, dropping bit-level correlation causes the largest loss (−0.102), followed by candidate frequency (−0.064) and Morgan Tanimoto similarity (−0.027). This supports the claim that bit-pattern alignment contributes information beyond global fingerprint overlap in the OF-macro evaluation. Under query-weighted Hit@10, candidate frequency has the largest one-drop loss; we therefore keep the feature-importance claim explicitly scoped to OF-macro performance.

**Table 4: Corrected full-model feature ablation (10-seed mean, OF-macro Hit@10 delta).**

| Feature Dropped | Delta Hit@10 | Std |
|----------------|-------------|-----|
| **bit_corr** | **−0.102** | **0.098** |
| frequency | −0.064 | 0.065 |
| Morgan | −0.027 | 0.033 |
| dRings | −0.009 | 0.035 |
| dLogP | −0.007 | 0.012 |
| dTPSA | −0.002 | 0.037 |
| dHeavy | −0.001 | 0.037 |
| dMW | +0.003 | 0.012 |

### 3.5 Model Complexity

Table 5 compares logistic regression against gradient-boosted trees on the same 8-feature input. LR (0.873) outperforms HGB (0.777), and LR without the frequency feature (0.732) still outperforms the hand-crafted Content-Aware baseline (0.636), confirming that the learned feature weights — not merely the inclusion of frequency — drive the improvement. The LR-over-HGB result is consistent with the small-N regime: with only 27 training OFs, the higher capacity of gradient boosting leads to overfitting rather than better generalization.

**Table 5: Model comparison (single 70/30 split, Hit@10).**

| Model | Features | Hit@10 |
|-------|----------|--------|
| LR | 8 features | 0.873 |
| HGB | 8 features | 0.777 |
| CA (hand-crafted) | Morgan + physchem | 0.636 |
| LR | 7 features (no freq) | 0.732 |
| HGB | 7 features (no freq) | 0.702 |

### 3.6 Cold-Start Analysis

In the current archive, the manifest contains 137 unique OF strings; 27 have positive-label support and 110 (80.3%) have zero positive-label support in the scanned label shards. The LOO support diagnostic provides a second, narrower cold-start view: 10/27 LOO rows have zero train-neighbor support and 21/27 have fewer than two labeled neighbors under the archived robustness definition. This label skew helps explain why a simple LR model that combines OF-specific features with a candidate-frequency prior can outperform pure similarity methods, and why retrieval-only C3F underperforms random in the archived comparison.

## 4. Discussion

### 4.1 Why bit_corr matters in OF-macro ranking

The corrected ablation identifies bit-level correlation as the largest OF-macro contributor among the eight features. Morgan Tanimoto measures *how much* two fragments overlap in fingerprint space; bit_corr measures *whether the pattern of bits* aligns between candidate and OF. Two fragments can have moderate Tanimoto similarity (e.g., 0.5) if they share roughly half their bits, but if the shared bits are distributed across pharmacophorically irrelevant positions while the discriminating bits are anti-correlated, the candidate may be a poor replacement. The ablation result suggests that this pattern-level signal adds value beyond global similarity and candidate frequency when each OF contributes equally to the evaluation.

### 4.2 Simplicity in the Small-N Regime

With only 27 labeled OFs, the effective training sample for distinguishing useful from useless replacements is constrained less by the number of candidate-OF rows than by the number of independent chemical contexts (OFs) from which to learn generalizable patterns. A linear model with 8 features and L2 regularization provides sufficient capacity to learn the relative importance of each signal without overfitting to idiosyncratic OF-candidate relationships. The underperformance of HGB in the archived model-comparison split (Hit@10 = 0.777 vs LR = 0.873) supports this interpretation: with only 27 independent training contexts, the higher-variance tree-based model may fit noise rather than signal.

This has practical implications for fragment replacement tool development: complex architectures (graph neural networks, deep sets, attention-based pooling) are unlikely to outperform simple linear models until substantially larger labeled datasets — with hundreds to thousands of independently labeled OFs — become available.

### 4.3 Limitations

**Label sparsity**: Most candidate-OF pairs in the archive are negative. This reflects the reality of fragment replacement (most candidates are not useful replacements), but it also means the model is trained in a highly imbalanced setting. While the ranking formulation (Hit@K) mitigates this by focusing on relative ordering, the absolute probability calibration may be unreliable.

**OF coverage**: With only 27 labeled OFs, the learned weights may not generalize to OF chemotypes absent from the training set. The corrected multi-seed diagnostic is encouraging, but the 27 OFs span limited chemical diversity (predominantly aromatic fragments with common linkers).

**Feature completeness**: The current 8 features capture structural similarity (Morgan, bit_corr), physicochemical matching, and frequency. Missing are 3D conformational features, electrostatic potential matching, and synthetic accessibility scores — all relevant to practical fragment replacement prioritization.

## 5. Conclusion

We present the Learned Bit-Correlation (LBC) Kernel, a logistic regression model for ranking fragment replacement candidates that combines Morgan Tanimoto similarity, bit-level fingerprint correlation, physicochemical property deltas, and candidate frequency. On a locked leave-one-out evaluation across 27 labeled old fragments from ChEMBL-derived data, LBC-Kernel achieves Hit@10 = 0.820, outperforming the Content-Aware baseline by 28.5%. Corrected 10-seed splits show consistent LBC-over-Content-Aware gains, and corrected full-model ablation identifies bit_corr as the largest OF-macro contributor. Query-level diagnostics show a rescue/loss ratio of 18.3 against Content-Aware, and a single-split capacity diagnostic favors LR over HGB in this small-OF setting. The model trains on CPU, requires no specialized hardware, and is implemented compactly in Python.

## Data and Software Availability

All code, trained models, and precomputed feature matrices are available at [repository URL]. The ChEMBL 36 dataset is publicly available at https://www.ebi.ac.uk/chembl/. The active claim-evidence lock is stored at `goal/PAPER_CLAIM_EVIDENCE_LOCK/lbc_kernel_20260606/`. The experimental pipeline uses scripts in both `paper_results/` and the project root: `paper_experiments.py` (LOO-CV, rank diagnostics, and model comparison), `lbc_lock_repair_experiments.py` (corrected multi-seed and full-model ablation), `robustness_check.py` (support diagnostics), `lk_fast.py` (historical multi-seed learned kernel), `c3f_content_aware.py` (C3F and Content-Aware baselines), and `learned_kernel.py` (initial kernel experiments). Environment: Python 3.10, scikit-learn 1.7.2, RDKit, pandas, numpy. No GPU required.

## References

[1] Meanwell, N. A. Synopsis of Some Recent Tactical Application of Bioisosteres in Drug Design. *J. Med. Chem.* **2011**, 54 (8), 2529–2591.

[2] Brown, N. *Bioisosteres in Medicinal Chemistry*; Wiley-VCH: Weinheim, 2012.

[3] Ertl, P. *In Silico* Identification of Bioisosteric Functional Groups. *Curr. Opin. Drug Discov. Dev.* **2007**, 10 (3), 281–288.

[4] Wagener, M.; Lommerse, J. P. M. The Quest for Bioisosteric Replacements. *J. Chem. Inf. Model.* **2006**, 46 (2), 677–685.

[5] Rogers, D.; Hahn, M. Extended-Connectivity Fingerprints. *J. Chem. Inf. Model.* **2010**, 50 (5), 742–754.

[6] Bajusz, D.; Rácz, A.; Héberger, K. Why Is Tanimoto Index an Appropriate Choice for Fingerprint-Based Similarity Calculations? *J. Cheminform.* **2015**, 7, 20.

[7] Papadatos, G.; Brown, N. In Silico Applications of Bioisosterism in Contemporary Medicinal Chemistry Practice. *WIREs Comput. Mol. Sci.* **2013**, 3 (5), 449–458.

[8] Ertl, P.; Schuffenhauer, A. Estimation of Synthetic Accessibility Score of Drug-like Molecules Based on Molecular Complexity and Fragment Contributions. *J. Cheminform.* **2009**, 1, 8.

[9] Humbeck, L.; Weigang, S.; Schäfer, T.; Mutzel, P.; Koch, O. C3F: A Collaborative Filtering Algorithm for the Prediction of Bioisosteric Replacement. *J. Chem. Inf. Model.* **2017**, 57 (7), 1652–1663.

[10] Gaulton, A.; et al. ChEMBL: A Large-Scale Bioactivity Database for Drug Discovery. *Nucleic Acids Res.* **2017**, 45 (D1), D945–D954.

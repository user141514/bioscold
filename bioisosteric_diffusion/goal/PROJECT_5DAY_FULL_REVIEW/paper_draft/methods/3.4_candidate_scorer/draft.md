## 3.4 Candidate-Level Scorer

### 3.4.1 Architecture and Feature Overview

The base ranking methods in Section 3.3 provide broad but shallow assessments of replacement candidates: they capture frequency signals (Attachment-Frequency), learned pairwise preferences (Dual Encoder), graph-structural features (HGB), ensemble rank aggregation (Borda Fusion), and cross-ranker score blending (Score Blend). Yet none operate at the granularity needed to distinguish chemically subtle differences between plausible and implausible replacements---a candidate ranked 11th by every base ranker may be chemically superior to one ranked 3rd, if the base rankers are merely exploiting shallow frequency or similarity artifacts. We address this gap with a **candidate-level scorer** that combines 77 chemically motivated features with a gradient boosting model to produce a fine-grained replacement score for every candidate in a query's candidate pool.

Given query $q_i$ and candidate fragment $c$, the final scorer computes a feature vector

$$
\boldsymbol{\phi}_{ic} = \Phi(q_i,c) \in \mathbb{R}^{77},
\label{eq:feature_vector}
$$

and passes it through a histogram gradient-boosted model $F_{\Theta}$ that maps the feature vector to a candidate-level probability:

$$
\widehat{p}_{ic}
=
F_{\Theta}(\boldsymbol{\phi}_{ic}),
\qquad
0 \le \widehat{p}_{ic} \le 1.
\label{eq:scorer_output}
$$

Candidates are ranked within each query by decreasing $\widehat{p}_{ic}$. The training target is $y_{ic} = \mathbb{I}[c \in \mathcal{P}_i]$, and the model is fit by weighted binary log loss over query-candidate rows,

$$
\mathcal{L}_{\mathrm{HistGB}}(\Theta)
=
-
\sum_{i=1}^{N}\sum_{c\in\mathcal{C}_i}
 w_{ic}
\left[
 y_{ic}\log \widehat{p}_{ic}
 +(1-y_{ic})\log(1-\widehat{p}_{ic})
\right],
\label{eq:histgb_loss}
$$

where $w_{ic}$ is the sample weight induced by class balancing or negative subsampling.

We choose HistGB over alternative model classes for three reasons. First, the feature space is heterogeneous---continuous scores, discrete counts, binary flags, and one-hot categorical encodings---and tree-based models naturally accommodate mixed-type inputs without specialized normalization or scaling per feature type. Second, preliminary experiments with logistic regression (which cannot model feature interactions) and fully-connected networks (which required extensive hyperparameter tuning and overfit the moderate-sized training set without regularization) both underperformed HistGB, consistent with the observation that fragment replacement involves non-linear combinations of features---for instance, a high prior score coupled with a large molecular weight delta signals a fundamentally different replacement context than either feature in isolation. Third, HistGB provides built-in feature importance estimates, enabling us to assess the contribution of each feature family without committing to a manual feature selection procedure.

### 3.4.2 Feature Families

The 77 features organize into 9 chemically motivated families, summarized in Table 3. Every feature is computed from the molecular graph of the query, the candidate, or their conjunction, using RDKit [Landrum, 2024] for molecular descriptors and scikit-learn [Pedregosa et al., 2011] for encoding. Complete per-feature definitions, computation formulas, and RDKit parameters are provided in Supplementary Table S1.

\begin{table}[t]
\centering
\caption{Feature families of the candidate-level scorer. Each family captures a distinct aspect of fragment replacement chemistry. Counts reflect dimensionality before one-hot expansion of categorical variables; the final 77-dimensional vector includes the one-hot columns.}
\label{tab:feature_families}
\small
\begin{tabular}{lp{3.8cm}cc}
\toprule
Family & Description & Count & Chemical Rationale \\
\midrule
F1: Model Scores & Blended and z-normalized scores from base rankers & 5 & Calibrated input signal from complementary ranking methods \\
F2: Model Raw Scores & Raw, unnormalized outputs from each base method & 4 & Preserves magnitude information lost in normalization \\
F3: Model Ranks & Per-query rank positions from each base ranker & 6 & Relative ordering context within the candidate pool \\
F4: Top-K Flags & Binary indicators of top-10 membership per base ranker & 4 & Consensus identification across independent methods \\
F5: Prior Scores & Logit-transformed scores from hierarchical prior distributions & 5 & Empirical replacement plausibility from training data \\
F6: Prior Statistics & Smoothed rates, support counts, and positive counts across four provenance levels & 12 & Statistical reliability of each prior estimate \\
F7: PMI Scores \& Ranks & Pointwise mutual information and ranks under different conditioning & 6 & Information gain beyond baseline conditioning \\
F8: Molecular Descriptors & Physicochemical properties of the candidate and deltas from the original fragment & 10 & Steric and electronic compatibility constraints \\
F9: Similarity \& Frequency & Morgan fingerprint similarity, global and attachment-conditional frequency & 3 & Fragment-level structural familiarity \\
Categorical (one-hot) & Fragment identity, attachment signature, frequency bin & 3 $\rightarrow$ $\sim$19 & Fragment-specific and attachment-specific effects \\
\bottomrule
\end{tabular}
\end{table}

**Model-derived features (F1--F4).** The first four families capture the outputs and consensus patterns of the base ranking methods described in Section 3.3. Model scores (F1) and raw scores (F2) provide the primary signal from each ranker type: a candidate that receives high scores from multiple independent rankers is more likely to be a genuine replacement. Model ranks (F3) convert these scores into relative ordering within each query's candidate pool---information that is complementary to the raw scores, since a high score in a dense cluster of high-scoring candidates is less informative than an equivalent score in a sparse pool where the correct replacement is unambiguously top-ranked. Top-K flags (F4) provide discrete consensus signals: a candidate appearing in all four base rankers' top-10 lists is a strong consensus candidate, while one appearing in none is unlikely regardless of any individual ranker's score.

**Prior and statistical features (F5--F7).** These families encode empirical replacement frequencies computed from the training data across four provenance levels (defined in Section 2.2): P0 (global fragment frequency), P1 (attachment-conditioned frequency $\widehat{p}_{\mathrm{att}}(c\mid\sigma)$), P3 (cluster-conditioned frequency within the same MMP transformation class), and P4 (transferred probability from chemically similar training fragments). Rather than supplying raw probabilities, we provide logit-transformed scores (F5) to linearize the probability scale for the tree model, which otherwise must learn to invert the sigmoid relationship internally. The smoothed rates, support counts, and positive counts (F6) quantify the statistical reliability of each estimate: a high-probability prior based on 200 observations is qualitatively different from the same probability based on 2 observations, and the model can use these support features to appropriately discount sparsely supported priors. The pointwise mutual information features (F7) capture the information gain at each conditioning level relative to the baseline---for example, $\mathrm{PMI}(P3, P1) = \log P(P3) - \log P(P1)$ measures how much additional information the cluster-level context provides beyond the attachment-level context alone. A fragment with a large positive cluster PMI is one whose replacement plausibility depends strongly on its MMP transformation class, a pattern the scorer can learn to weigh differently than fragments with flat PMI profiles.

**Molecular descriptors (F8) and similarity features (F9).** These features ensure that the scorer considers physicochemical compatibility, not merely statistical precedent. The molecular descriptors (F8) include both the candidate's absolute properties (heavy atom count, molecular weight, logP, TPSA) and the deltas between candidate and original fragment. The delta features directly encode the medicinal chemistry intuition that a replacement must be sterically and electronically compatible with the binding site---a candidate that differs from the original by more than 100 Da in molecular weight or 4 logP units is unlikely to preserve activity regardless of its empirical frequency. The similarity and frequency features (F9) contribute Morgan fingerprint Tanimoto similarity (2048-bit radius-2 fingerprints), the candidate's global replacement frequency in the training set, and its attachment-conditioned frequency. The Tanimoto similarity provides the model with a direct structural comparison, while the frequency features anchor the prediction in the empirical distribution of fragment usage.

**Categorical features.** Three categorical variables---old fragment identity (SMILES), attachment signature, and replacement frequency bin (low, medium, high)---are transformed via one-hot encoding with special handling for unseen categories (Section 3.4.4). The resulting dimensionality depends on the number of distinct categories observed in the validation set (typically 8--9 categories for old fragment identity, 5--6 for attachment signature, and 3--4 for frequency bin). These features enable the model to learn fragment-specific effects: replacing a chlorine atom follows different chemical rules than replacing a phenyl ring, and the categorical encodings capture these differences without requiring explicit hand-written rules.

### 3.4.3 Model Architecture and Training Protocol

The scorer is implemented as a histogram gradient boosting classifier (`HistGradientBoostingClassifier`; scikit-learn [Pedregosa et al., 2011]) configured with a maximum of 200 boosting iterations, a maximum tree depth of 6, a learning rate of 0.1, `class_weight='balanced'` to compensate for the severe class imbalance inherent in the candidate ranking task (each query contributes exactly one positive among tens of negatives), and `random_state=42` for reproducibility. The model minimizes log loss and produces a calibrated probability as its output score.

Training follows a 5-fold GroupKFold cross-validation strategy, where each fold holds out all candidates from a disjoint set of queries (grouped by query identifier to prevent within-query leakage between training and validation). Within each fold, 80\% of queries form the training set and 20\% form the validation set. To mitigate the class imbalance, we apply negative subsampling at a 5:1 ratio (five randomly sampled negatives per positive) during training, following preliminary experiments that showed this ratio balanced training efficiency against representativeness. The five folds produce five separate models; for validation evaluation, we use each fold's model to score its held-out queries, yielding out-of-fold predictions for the complete validation set.

For secondary blind evaluation, we train a single model on the complete validation set (all queries, no held-out fold) and apply it once to the secondary blind set---a *one-shot* protocol that simulates the deployment scenario in which the model is trained on available data and applied to new queries without any iterative refinement, model selection, or feature tuning on the target distribution. No secondary blind set statistics inform any aspect of model development. Feature standardization (zero mean, unit variance) is computed from the validation set statistics and applied to the secondary blind set using the same parameters, ensuring strict protocol separation between the validation and blind splits.

All 77 features are presented to the model without manual pre-selection or dimensionality reduction. Preliminary experiments with feature pre-filtering (univariate correlation screening, recursive feature elimination) did not improve performance and introduced risks of indirect leakage. HistGB's built-in feature importance mechanism effectively down-weights irrelevant dimensions, making pre-filtering unnecessary while preserving the complete chemically motivated feature set.

### 3.4.4 Feature Alignment via cat_templates

An implementation detail critical to the scorer's secondary blind set generalization is the handling of categorical features through template alignment. When one-hot encoding categorical variables, the validation and secondary blind sets may contain disjoint sets of category values: secondary blind queries involve 19 old fragment identities that were never observed in the validation set, meaning standard one-hot encoding would produce matrices with different column counts and orderings in each split.

We address this with a template-based encoding strategy (cat_templates). First, the one-hot encoder is fitted on the validation set, recording the set of observed categories for each variable and the resulting column ordering. For secondary blind set encoding, any category not present in the validation set is mapped to a reserved OTHER category, and the secondary blind one-hot matrix is forced to exactly match the validation set's column count and ordering. A runtime assertion ($\texttt{val\_X.shape[1] == blind\_X.shape[1]}$) guarantees schema consistency.

The impact of this alignment is substantial: without cat_templates, a prototype without template alignment achieved a secondary blind Top-10 accuracy of 0.7120---because the model's learned feature weights were applied to semantically incorrect columns when the one-hot dimensions shifted between splits. With proper template alignment, the same architecture achieved 0.8851, a relative improvement of 24.3\% attributable entirely to correct feature indexing. This result underscores the sensitivity of feature-based models to categorical encoding consistency, particularly in molecular machine learning where fragment vocabularies shift substantially between training and deployment distributions.

The ablation study in Section 3.5 validates this feature composition through systematic leave-one-family-out analysis.

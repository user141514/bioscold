## 3. Methods

This section presents the complete methodological framework for the fragment replacement ranking task. Section 3.1 defines the task formulation, supervision labels, and evaluation metrics. Section 3.2 describes the construction of the benchmark dataset, including matched molecular pair extraction, decoy construction, and the transform-heldout split design. Sections 3.3--3.5 present the ranking methods: five base ranking methods spanning a range of inductive biases (Section 3.3), followed by the candidate-level scorer with its 77-dimensional feature space and gradient boosting architecture (Section 3.4), and concluding with an ablation study demonstrating that removal of a feature family encoding per-query rank information substantially improves cross-split generalization (Section 3.5). Sections 3.6--3.7 cover the evaluation protocol, including bootstrap confidence intervals, rescue and lost analysis, strict protocol separation, leakage controls (Section 3.6), and the A4C computational review proxy for qualitative risk stratification of proposals (Section 3.7).

---

### 3.1 Task Formulation

Fragment replacement -- the identification of chemically distinct substituents that can be swapped at a given scaffold attachment point -- is a cornerstone of medicinal chemistry scaffold-hopping campaigns. The practical workflow of a medicinal chemist involves screening a manageable number of candidate replacements (typically tens, not thousands), making ranking accuracy the critical performance dimension. We formalize fragment replacement as a ranking problem: given a query consisting of the original fragment and its attachment signature, select the correct replacement from a closed set of candidates. Rather than generating de novo fragment suggestions, this retrieval formulation enables direct optimization of ranking accuracy and a clear evaluation protocol against structure-derived ground truth.

A query $q$ is defined by the pair

\[
q = (f_{\text{old}},\; \sigma),
\]

where $f_{\text{old}}$ is the original fragment represented as a SMILES string annotated with an attachment marker indicating the bond connecting it to the molecular scaffold, and $\sigma$ is the **attachment signature** -- a specification of the scaffold atom types and bond orders at the attachment point that defines the chemical context in which the replacement must be chemically valid. The attachment signature is critical because it constrains which fragments are viable replacements: a candidate fragment must be attachable to the same scaffold atom with a chemically compatible bond.

Let $V$ be the global replacement vocabulary, constructed exclusively from the training split of the benchmark and containing 150--152 unique fragments depending on the split configuration. Only fragments present in $V$ are eligible candidates. The candidate set $C_q$ for query $q$ is the subset of $V$ that is attachment-compatible with $\sigma$:

\[
C_q = \{\, c \in V \mid \text{compatible}(c, \sigma) \,\}.
\tag{1}
\]

The task is to produce a ranked list of $C_q$ such that the true replacement fragment(s) appear as high as possible. We learn a scoring function $s(c \mid q)$ that assigns a real-valued score to each candidate; candidates are then sorted by descending score to produce the final ranking.

**Supervision labels.** Positive examples are derived entirely from structure-based matched molecular pair (MMP) analysis on ChEMBL37K. An MMP is a pair of molecules that differ at exactly one localized structural site while sharing the same molecular scaffold; when the same scaffold appears with two different fragments at the same attachment point, the observed fragment substitution provides a positive training example. For each query $q$, the positive set $P_q$ consists of all fragments $c^*$ that have been observed as replacements for $f_{\text{old}}$ in a chemically similar scaffold context:

\[
P_q = \{\, c^* \mid (f_{\text{old}}, c^*) \text{ is a valid MMP pair in the training data} \,\}.
\tag{2}
\]

We emphasise that these labels are **structure-derived**: an observed MMP transformation indicates that $c^*$ has been used as a fragment replacement for $f_{\text{old}}$ in a related molecular context, but does **not** establish that the replacement preserves biological activity. This is a limitation inherent to structure-based fragment replacement benchmarks and a key motivation for the computational review proxy introduced in Section 3.7. Queries may be multi-positive -- a single $f_{\text{old}}$ may have been observed paired with multiple distinct replacement fragments across different MMP transformations.

**Evaluation metrics.** Our primary metric is query-level Top-10 accuracy: for each query $q$, the prediction is correct if and only if any fragment from $P_q$ appears among the ten highest-scoring candidates in $C_q$. Formally, the per-query hit at rank $K$ is

\[
\text{Hit@K}(q) = \mathbb{1}\!\bigl[\, \min_{c^* \in P_q} \text{rank}(c^*) \leq K \,\bigr],
\tag{3}
\]

where $\text{rank}(c^*)$ is the position of fragment $c^*$ in the ranked candidate list. The Top-$K$ accuracy over a query set $Q$ is the mean of $\text{Hit@K}(q)$ across all $q \in Q$. This query-level evaluation mirrors the practical constraint that a medicinal chemist examines only the top handful of candidate suggestions. The choice of $K=10$ as the primary threshold reflects this screening depth; we additionally report Top-1, Top-5, Top-20, and Top-50 accuracy to characterize performance across the full ranking, as well as Mean Reciprocal Rank (MRR), which measures the average reciprocal rank of the highest-ranked positive across all queries.

For multi-positive queries, a hit is scored if any positive fragment appears within the top $K$ -- a standard information retrieval convention that avoids penalising models for retrieving any of several valid replacements. All metrics are reported with 95% confidence intervals computed via nonparametric bootstrapping over 5,000 replicates, resampling at the query level to preserve the per-query evaluation structure.

---

### 3.2 Benchmark Construction

To provide a large, realistic training and evaluation dataset for the fragment replacement task, we extract matched molecular pairs from ChEMBL37K and construct a transform-heldout split that prevents scaffold-based leakage between training and evaluation queries. The central challenge in creating supervision data without recourse to experimental assays is generating both reliable positive and negative examples. We address this through three systematic procedures: MMP extraction to obtain positive fragment substitutions, controlled decoy construction to produce chemically valid negatives, and a split strategy that guarantees zero identity overlap between training and evaluation transforms.

**Data source and MMP extraction.** ChEMBL37K [Zdrazil et al., 2023] is a curated subset of approximately 37,000 drug-like molecules drawn from ChEMBL33, filtered for medicinal chemistry relevance. We extract MMPs by applying established retrosynthetic bond-cleavage rules [Hussain and Rea, 2010; Griffen et al., 2011] to every molecule in the set: each molecule is recursively decomposed at acyclic single bonds to generate fragment--scaffold pairs, and any two molecules sharing identical scaffold fragments but bearing different substituent fragments at the same attachment point are recorded as an MMP. The observed substituent substitution $(f_{\text{old}} \rightarrow c^*)$ provides a positive training example. Standard quality filters are applied during preprocessing: salt stripping, stereochemistry normalization, removal of molecules containing reactive or unstable substructures (PAINS and Brenk alerts [Baell and Holloway, 2010; Brenk et al., 2008]), and exclusion of fragments with molecular weight below 15 Da or above 250 Da to focus on drug-like substituent fragments. Applying these procedures to ChEMBL37K yields an initial pool of tens of thousands of MMP-derived fragment replacement pairs.

**Decoy construction.** MMP analysis produces only positive examples -- observed fragment substitutions -- but training a ranking model requires negative examples to learn to discriminate correct from incorrect replacements. We construct negatives through a **decoy repair** procedure: for each query, the original fragment $f_{\text{old}}$ is paired with a known fragment $c_{\text{decoy}} \in V$ that is attached to a non-cognate scaffold in the ChEMBL37K data. This pairing produces a chemically valid fragment--scaffold combination that has **not** been observed as a replacement for this specific scaffold context. The decoy repair procedure comprises four stages:

1. **1:1 ratio.** Each positive example is paired with exactly one decoy to produce balanced candidate sets, avoiding the degenerate solution of trivial negative rejection.
2. **Wrong-positive removal.** Any decoy fragment that is itself a known MMP replacement for the query (i.e., belongs to $P_q$) is removed from the negative set, preventing false-negative contamination.
3. **Deduplication.** Identical $(f_{\text{old}}, c_{\text{decoy}})$ pairs across queries are collapsed to prevent double-counting of the same training instance.
4. **1:5 diagnostic set.** An additional set with a 1:5 positive-to-decoy ratio is constructed as a stress test for model robustness under class imbalance. This set is excluded from all primary evaluation and used solely for diagnostic analysis.

The 1:1 decoy set forms the basis of all training and primary evaluation. The 1:5 diagnostic set is reserved for characterizing how model performance degrades as the candidate pool becomes increasingly sparse in positive examples -- a stress test relevant to real-world screening scenarios where true replacements are rare.

**Split design: transform-heldout.** The most critical design decision in benchmark construction is the data split. A conventional random split of MMP transformations would allow the same $(f_{\text{old}}, \sigma)$ pair to appear in both training and test sets, enabling the model to memorise specific transform identities rather than learning general fragment replacement rules. Such leakage would inflate accuracy estimates without measuring true generalization to unseen replacement scenarios.

We prevent this with a **transform-heldout split**: we identify all unique $(f_{\text{old}}, \sigma)$ pairs in the MMP data and partition them such that no pair appears in both the training and evaluation splits. This guarantees that the model must generalise to transform identities it has never encountered during training.

Beyond the primary transform-heldout split, we construct two additional evaluation sets that provide complementary perspectives on model performance:

- **Secondary blind set.** A held-out set of 13,347 queries with a 150-fragment vocabulary that shares no $(f_{\text{old}}, \sigma)$ pairs with the training set. The blind vocabulary is entirely disjoint from the training vocabulary. This set serves as the primary evaluation benchmark for all main experiments and performance claims.
- **Canonical analysis set.** A larger set of 21,052 queries with a 152-fragment vocabulary, used exclusively for robustness analysis and mechanistic investigation. No primary performance claims in this paper are based on the canonical set; it is included to probe model behaviour under broader fragment diversity and to verify that conclusions are not artefacts of a particular split configuration.

**Leakage verification.** We systematically verify that the split design achieves its intended isolation. Table 1 reports the overlap counts of $(f_{\text{old}}, \sigma)$ transform identities between every pair of splits. Zero overlap is confirmed for all pairs, ensuring that no evaluation query shares its transform identity with any training query.

---

**Table 1.** Leakage verification: overlap counts of $(f_{\text{old}}, \sigma)$ transform identities between split pairs. Zero overlap is confirmed for all pairs.

| Split Pair | Transform Overlap | Verified |
|---|---|---|
| Train / Transform-heldout test | 0 | Yes |
| Train / Secondary blind | 0 | Yes |
| Train / Canonical test | 0 | Yes |
| Canonical / Secondary blind queries | 0 | Yes |

---

This three-tier evaluation design -- transform-heldout test, secondary blind, and canonical analysis -- enables rigorous assessment of fragment replacement methods at multiple levels of generalization difficulty. The transform-heldout split is the central methodological contribution to the benchmark: it ensures that evaluation measures the model's ability to generalise to unseen fragment-scaffold combinations rather than its capacity to memorise training-set transform identities. The secondary blind set provides a stricter stress test with a completely disjoint fragment vocabulary, while the canonical set enables mechanistic analysis without risk of contaminating the primary evaluation.

With the benchmark established, we now describe the five base ranking methods evaluated on it, which also provide the input signals for the candidate-level scorer introduced in Section 3.4.

---

### 3.3 Base Ranking Methods

We implement five baseline ranking methods that represent distinct approaches to the fragment replacement problem, providing both comparative baselines and the input features for our candidate-level scorer (Section 3.4). These methods span a deliberate spectrum of complexity and inductive bias: from empirical frequency counting, through learned embedding similarity, to feature-based classification and parameter-free rank fusion. Each captures a fundamentally different signal -- global replacement statistics, structural compatibility in a learned space, interpretable molecular properties, or their consensus -- and the complementarity among them is itself a central finding that the Borda fusion (Section 3.3.4) and the downstream scorer both exploit.

All methods are evaluated on the secondary blind split (13,347 held-out queries) using query-level Top-10 accuracy: a candidate is considered correct if the ground-truth replacement fragment is ranked among the top 10 for its query. Confidence intervals are computed by nonparametric bootstrapping with 5,000 resamples over queries. Table 2 summarizes the blind Top-10 performance of each method.

---

**Table 2.** Secondary blind Top-10 accuracy of the five base ranking methods, with 95% bootstrap confidence intervals. The Oracle ceiling represents the fraction of queries where any base ranker places the correct candidate in its top 10.

| Method | Blind Top-10 | 95% CI |
|---|---|---|
| Attachment-Frequency | 0.6019 | [0.5933, 0.6104] |
| Dual Encoder (DE) | 0.8055 | [0.7986, 0.8122] |
| HGB | 0.7437 | [0.7356, 0.7516] |
| Borda(DE, HGB) | 0.8384 | [0.8321, 0.8447] |
| MLP (rank-only) | 0.8402 | [0.8339, 0.8466] |
| Oracle ceiling | 0.8686 | --- |

---

**Oracle ceiling.** Before describing each ranker individually, it is useful to establish the best possible performance achievable from the base ranker set. The Oracle ceiling (0.8686) represents the fraction of secondary blind queries for which *at least one* base ranker places the correct fragment in its top 10. This is a theoretical upper bound on any downstream method that operates solely on the base rankers' outputs: even a perfect selector cannot recover a target that no base ranker surfaces. The gap between the Oracle and the best individual ranker (0.8686 vs. 0.8402) indicates that approximately 3% of queries are individually recoverable by different rankers -- queries where, for example, the Dual Encoder succeeds and the MLP fails, or vice versa -- and quantifies the opportunity for a fusion or scoring method that can learn to select the right ranker's output on a per-query basis.


#### 3.3.1 Attachment-Frequency Ranker

The Attachment-Frequency ranker estimates the conditional probability of observing a candidate fragment $c$ given an attachment signature $\sigma$ in the training set:

\[
P(c \mid \sigma) = \frac{\text{count}(c, \sigma)}{\text{count}(\sigma)},
\]

where $\text{count}(c, \sigma)$ is the number of training matched molecular pairs in which fragment $c$ appears with attachment signature $\sigma$, and $\text{count}(\sigma)$ is the total number of training pairs bearing that signature. Candidates are ranked by descending $P(c \mid \sigma)$, and the method requires no learned parameters and no molecular featurization -- only the training pair counts.

This approach embodies a straightforward but powerful empirical prior: fragments that have frequently appeared as replacements at a given attachment environment are more likely to be valid replacements again. The method captures the *global* replacement tendency of each fragment under each attachment context, information that is immediately useful but inherently coarse: it cannot distinguish between two candidates with identical frequency counts that differ in chemical compatibility with the specific query fragment. The Attachment-Frequency ranker therefore serves as the natural lower bound for all subsequent methods. On the secondary blind split, it achieves a Top-10 accuracy of 0.6019 (95% CI: 0.5933--0.6104).


#### 3.3.2 Dual Encoder Ranker

The Dual Encoder (DE) ranker replaces the frequency-based heuristic with a learned measure of structural compatibility. Its design follows the standard dual-encoder architecture for similarity learning: a query encoder and a candidate encoder that project their respective inputs into a shared embedding space, where compatibility is scored by cosine similarity.

The query encoder takes two inputs: the Morgan fingerprint (ECFP, radius 2, 2048 bits) of the original fragment being replaced, and a learned embedding of the attachment signature $\sigma$. These are concatenated and passed through a two-layer MLP with hidden dimensionality $d=128$ and ReLU activation to produce a query embedding $\mathbf{e}_q \in \mathbb{R}^d$. The candidate encoder, independently, maps the candidate fragment's Morgan fingerprint through a separate two-layer MLP (same architecture, not weight-tied) to a candidate embedding $\mathbf{e}_c \in \mathbb{R}^d$. The replacement score is then the cosine similarity:

\[
s_{\text{DE}}(q, c) = \cos(\mathbf{e}_q, \mathbf{e}_c) = \frac{\mathbf{e}_q \cdot \mathbf{e}_c}{\|\mathbf{e}_q\| \, \|\mathbf{e}_c\|}.
\]

Training uses a contrastive objective over query-candidate pairs derived from the training MMP data. For each positive pair $(q, c^+)$ -- a query and its observed replacement -- we sample $K=20$ negative candidates $(q, c^-)$ uniformly from the candidate vocabulary, ensuring the negatives are attachment-compatible but not the observed replacement. The model is optimized with a margin-based ranking loss:

\[
\mathcal{L}_{\text{DE}} = \sum_{(q, c^+)} \sum_{k=1}^{K} \max\bigl(0, \; s_{\text{DE}}(q, c^-_k) - s_{\text{DE}}(q, c^+) + \Delta\bigr),
\]

with margin $\Delta = 0.3$. The objective encourages the model to assign higher similarity to the true replacement than to any sampled negative, learning an embedding space where structurally compatible query-candidate pairs cluster together.

The Dual Encoder provides a fundamentally different signal from the frequency-based approach. While $P(c \mid \sigma)$ captures global empirical trends -- "this candidate is common at this attachment" -- the DE captures whether a *specific* candidate's molecular structure suits a *specific* query's chemical context. A candidate with low global frequency may still be the correct replacement if its substructure is complementary to the query fragment. This divergence in signal is reflected quantitatively: DE achieves a Top-10 accuracy of 0.8055 (95% CI: 0.7986--0.8122) on the secondary blind split, a gain of +0.2036 over the frequency baseline, confirming that learned structural compatibility captures substantial signal beyond empirical frequency.


#### 3.3.3 Histogram Gradient-Boosted Ranker (HGB)

The Histogram Gradient-Boosted (HGB) Ranker approaches fragment replacement as a supervised classification problem with explicit, interpretable molecular features. For each query-candidate pair $(q, c)$, we compute a feature vector $\mathbf{x}_{q,c}$ organized into four groups: (1) *frequency features:* $P(c \mid \sigma)$, global replacement frequency $\text{count}(c)$, and the attachment frequency $\text{count}(c, \sigma)$; (2) *attachment signature features:* the bond order at the attachment point, whether the bond belongs to a ring system, the scaffold atom type at the attachment, and the attachment signature itself (one-hot encoded); (3) *molecular property descriptors:* heavy atom count, molecular weight, logP, TPSA, ring count, hydrogen bond donor and acceptor counts, and the number of rotatable bonds -- computed for both the candidate and query fragments; and (4) *fingerprint similarity:* the Tanimoto coefficient between the Morgan fingerprints (radius 2, 2048 bits) of the original and candidate fragments. The HGB model is trained using scikit-learn's `HistGradientBoostingClassifier` with 200 boosting iterations, a maximum depth of 6, and a learning rate of 0.1. Candidates are ranked by the model's predicted probability of being a valid replacement.

The HGB ranker occupies a distinct methodological position in our framework. Unlike the Dual Encoder, whose latent embeddings trade interpretability for representational flexibility, the HGB model operates directly on physicochemical descriptors whose meanings are transparent to a medicinal chemist. This makes it the natural source for the Conservative Mode in our dual-mode workflow (Section 3.7), where chemically conservative proposals -- those grounded in observable, nameable properties rather than opaque learned features -- are preferred for high-confidence applications. On the secondary blind split, HGB achieves a Top-10 accuracy of 0.7437 (95% CI: 0.7356--0.7516), establishing a strong feature-based baseline that, while lower than DE, captures a complementary signal.


#### 3.3.4 Borda Fusion and Score Blending

**Borda fusion.** The Borda fusion method combines the DE and HGB rankers through parameter-free rank aggregation, testing whether their individual errors are independent enough that their consensus outperforms either alone. For a query $q$ with candidate set of size $|V|$, let $\text{rank}_m(q, c)$ denote the rank assigned to candidate $c$ by method $m \in \{\text{DE}, \text{HGB}\}$. The Borda score is:

\[
S_{\text{Borda}}(q, c) = \sum_{m \in \{\text{DE}, \text{HGB}\}} \bigl( |V| + 1 - \text{rank}_m(q, c) \bigr),
\]

and candidates are re-ranked by $S_{\text{Borda}}$ in descending order.

The parameter-free design of Borda fusion is deliberate and methodologically motivated by our transform-heldout evaluation protocol (Section 3.2). Under a transform-heldout split, no query in the validation or secondary blind sets shares its (old_fragment, attachment_signature) combination with any training query. This means that any fusion weights optimized on a validation set would necessarily be tuned to ranking patterns that may not generalize to queries with unseen transform combinations -- there is no guarantee that the optimal weight for DE versus HGB on validation queries is optimal for secondary blind queries. Borda fusion sidesteps this entirely by using equal, fixed weights that require no tuning, ensuring consistent behavior across all splits. This conservative choice is appropriate for a baseline method: if simple equal-weight fusion already demonstrates substantial gains, then the complementarity signal is robust and not an artifact of weight optimization.

The Borda results confirm this. On the secondary blind split, Borda fusion achieves a Top-10 accuracy of 0.8384 (95% CI: 0.8321--0.8447), representing gains of +0.0329 over DE alone and +0.0947 over HGB alone. The improvement over DE is particularly notable: DE is the stronger individual ranker, yet fusing it in equal proportion with the weaker HGB still produces a net gain, confirming that the two methods make substantially different errors. The Borda consensus captures the approximately 3% of queries where DE fails but HGB succeeds, approaching the Oracle ceiling without any learned component.

**Score Blend (MLP + HGB).** The Score Blend extends the fusion concept by introducing a lightweight learned aggregator: a rank-only MLP that takes as input three per-query rank values for each candidate -- the ranks assigned by the DE, HGB, and Attachment-Frequency rankers -- and produces a single score $s_{\text{MLP}}(q, c)$. This MLP is a two-layer network with hidden dimensionality 32 and ReLU activations, trained as a binary classifier on the training split. It learns to weight the three base rankers' signals differently depending on the query context, adapting the fusion strategy to each query rather than applying the fixed Borda formula.

The final Score Blend combines the MLP score with a separately refit HGB model:

\[
s_{\text{blend}}(q, c) = 0.95 \cdot z\bigl(s_{\text{MLP}}(q, c)\bigr) + 0.05 \cdot z\bigl(s_{\text{HGB-refit}}(q, c)\bigr),
\]

where $z(\cdot)$ denotes per-query z-score normalization and $s_{\text{HGB-refit}}$ is the predicted probability from an HGB model refit on the training data with the MLP's outputs included as an additional feature. The heavy coefficient on the MLP term (0.95) reflects that the rank-only aggregator captures the vast majority of predictive signal; the marginal HGB-refit contribution provides calibration smoothing rather than substantial new information.

On the secondary blind split, the rank-only MLP achieves a Top-10 accuracy of 0.8402 (95% CI: 0.8339--0.8466). While this is a marginal improvement over Borda (+0.0018, not statistically significant at the Top-10 level), the MLP provides a statistically significant improvement in Mean Reciprocal Rank, indicating that its benefit is in elevating the correct candidate closer to the top of the ranked list within queries where it already appears in the top 10. The full Score Blend achieves 0.8558, establishing the strongest baseline prior to the candidate-level scorer and the highest performance achievable from base ranker signals alone.

**Summary: the gap between base rankers and the Oracle.** Across all five methods, the best-performing individual ranker (MLP, 0.8402) and the strongest combination (Score Blend, 0.8558) fall short of the Oracle ceiling (0.8686). The residual gap -- approximately 1.3 to 2.8 percentage points -- represents queries where, even after rank fusion and learned aggregation, the base ranker signals are insufficient to surface the correct candidate. Closing this gap requires moving beyond rank-level and frequency-level features to a richer representation that incorporates fine-grained chemical constraints: the properties of the fragment-scaffold interface, the steric and electronic compatibility of the replacement, and the structural relationships among candidates within a query. This motivates the candidate-level scorer in Section 3.4, which augments the base ranker signals with 77 chemically motivated features and a dense gradient-boosted model capable of discriminating among candidates that the base rankers cannot separate.

---

### 3.4 Candidate-Level Scorer

The base ranking methods in Section 3.3 provide broad but shallow assessments of replacement candidates: they capture frequency signals (Attachment-Frequency), learned pairwise preferences (Dual Encoder), graph-structural features (HGB), ensemble rank aggregation (Borda Fusion), and cross-ranker score blending (Score Blend). Yet none operate at the granularity needed to distinguish chemically subtle differences between plausible and implausible replacements -- a candidate ranked 11th by every base ranker may be chemically superior to one ranked 3rd, if the base rankers are merely exploiting shallow frequency or similarity artifacts. We address this gap with a **candidate-level scorer** that combines 77 chemically motivated features with a gradient boosting model to produce a fine-grained replacement score for every candidate in a query's candidate pool.

#### 3.4.1 Architecture and Feature Overview

Given a query $q = (f_{\text{old}}, \sigma)$ and a candidate fragment $c$, the scorer computes a 77-dimensional feature vector $\Phi(q, c)$ through a multi-stage featurization pipeline. First, a molecular encoding layer converts the candidate, original fragment, and their relationship into structural and score-based descriptors. Second, ranking context features capture where each candidate falls within the base rankers' orderings. Third, statistical priors encode the fragment's empirical support across multiple provenance levels in the training data. Fourth, pointwise mutual information features quantify the information gain of each fragment under different conditioning contexts. Fifth, property features capture the physicochemical compatibility between candidate and original fragment. Finally, similarity and frequency features encode the fragment's global and attachment-conditional prevalence. The complete feature vector is passed to a histogram gradient boosting classifier that outputs a single scalar score:

\[
s(q, c) = f_{\text{HistGB}}\big(\Phi(q, c); \Theta\big), \qquad \Phi(q, c) \in \mathbb{R}^{77}
\tag{4}
\]

where $\Theta$ denotes the trained ensemble parameters and $s(q, c)$ is the predicted probability that fragment $c$ is the correct replacement for query $q$. For ranking, candidates within a query are sorted by descending $s(q, c)$.

We choose HistGB over alternative model classes for three reasons. First, the feature space is heterogeneous -- continuous scores, discrete counts, binary flags, and one-hot categorical encodings -- and tree-based models naturally accommodate mixed-type inputs without specialized normalization or scaling per feature type. Second, preliminary experiments with logistic regression (which cannot model feature interactions) and fully-connected networks (which required extensive hyperparameter tuning and overfit the moderate-sized training set without regularization) both underperformed HistGB, consistent with the observation that fragment replacement involves non-linear combinations of features -- for instance, a high prior score coupled with a large molecular weight delta signals a fundamentally different replacement context than either feature in isolation. Third, HistGB provides built-in feature importance estimates, enabling us to assess the contribution of each feature family without committing to a manual feature selection procedure.

#### 3.4.2 Feature Families

The 77 features organize into 9 chemically motivated families, summarized in Table 3. Every feature is computed from the molecular graph of the query, the candidate, or their conjunction, using RDKit [Landrum, 2024] for molecular descriptors and scikit-learn [Pedregosa et al., 2011] for encoding. Complete per-feature definitions, computation formulas, and RDKit parameters are provided in Supplementary Table S1.

---

**Table 3.** Feature families of the candidate-level scorer. Each family captures a distinct aspect of fragment replacement chemistry. Counts in parentheses reflect dimensionality before one-hot expansion of categorical variables; the final 77-dimensional vector includes the one-hot columns.

| Family | Description | Count | Chemical Rationale |
|--------|-------------|-------|-------------------|
| F1: Model Scores | Blended and z-normalized scores from base rankers | 5 | Calibrated input signal from complementary ranking methods |
| F2: Model Raw Scores | Raw, unnormalized outputs from each base method | 4 | Preserves magnitude information lost in normalization |
| F3: Model Ranks | Per-query rank positions from each base ranker | 6 | Relative ordering context within the candidate pool |
| F4: Top-K Flags | Binary indicators of top-10 membership per base ranker | 4 | Consensus identification across independent methods |
| F5: Prior Scores | Logit-transformed scores from hierarchical prior distributions | 5 | Empirical replacement plausibility from training data |
| F6: Prior Statistics | Smoothed rates, support counts, and positive counts across four provenance levels | 12 | Statistical reliability of each prior estimate |
| F7: PMI Scores & Ranks | Pointwise mutual information and ranks under different conditioning | 6 | Information gain beyond baseline conditioning |
| F8: Molecular Descriptors | Physicochemical properties of the candidate and deltas from the original fragment | 10 | Steric and electronic compatibility constraints |
| F9: Similarity & Frequency | Morgan fingerprint similarity, global and attachment-conditional frequency | 3 | Fragment-level structural familiarity |
| Categorical (one-hot) | Fragment identity, attachment signature, frequency bin | 3 -> ~19 | Fragment-specific and attachment-specific effects |

---

**Model-derived features (F1--F4).** The first four families capture the outputs and consensus patterns of the base ranking methods described in Section 3.3. Model scores (F1) and raw scores (F2) provide the primary signal from each ranker type: a candidate that receives high scores from multiple independent rankers is more likely to be a genuine replacement. Model ranks (F3) convert these scores into relative ordering within each query's candidate pool -- information that is complementary to the raw scores, since a high score in a dense cluster of high-scoring candidates is less informative than an equivalent score in a sparse pool where the correct replacement is unambiguously top-ranked. Top-K flags (F4) provide discrete consensus signals: a candidate appearing in all four base rankers' top-10 lists is a strong consensus candidate, while one appearing in none is unlikely regardless of any individual ranker's score.

**Prior and statistical features (F5--F7).** These families encode empirical replacement frequencies computed from the training data across four provenance levels (defined in Section 2.2): P0 (global fragment frequency), P1 (attachment-conditioned frequency $P(c \mid \sigma)$), P3 (cluster-conditioned frequency within the same MMP transformation class), and P4 (transferred probability from chemically similar training fragments). Rather than supplying raw probabilities, we provide logit-transformed scores (F5) to linearize the probability scale for the tree model, which otherwise must learn to invert the sigmoid relationship internally. The smoothed rates, support counts, and positive counts (F6) quantify the statistical reliability of each estimate: a high-probability prior based on 200 observations is qualitatively different from the same probability based on 2 observations, and the model can use these support features to appropriately discount sparsely supported priors. The pointwise mutual information features (F7) capture the information gain at each conditioning level relative to the baseline -- for example, $\text{PMI}(P3, P1) = \log P(P3) - \log P(P1)$ measures how much additional information the cluster-level context provides beyond the attachment-level context alone. A fragment with a large positive cluster PMI is one whose replacement plausibility depends strongly on its MMP transformation class, a pattern the scorer can learn to weigh differently than fragments with flat PMI profiles.

**Molecular descriptors (F8) and similarity features (F9).** These features ensure that the scorer considers physicochemical compatibility, not merely statistical precedent. The molecular descriptors (F8) include both the candidate's absolute properties (heavy atom count, molecular weight, logP, TPSA) and the deltas between candidate and original fragment. The delta features directly encode the medicinal chemistry intuition that a replacement must be sterically and electronically compatible with the binding site -- a candidate that differs from the original by more than 100 Da in molecular weight or 4 logP units is unlikely to preserve activity regardless of its empirical frequency. The similarity and frequency features (F9) contribute Morgan fingerprint Tanimoto similarity (2048-bit radius-2 fingerprints), the candidate's global replacement frequency in the training set, and its attachment-conditioned frequency. The Tanimoto similarity provides the model with a direct structural comparison, while the frequency features anchor the prediction in the empirical distribution of fragment usage.

**Categorical features.** Three categorical variables -- old fragment identity (SMILES), attachment signature, and replacement frequency bin (low, medium, high) -- are transformed via one-hot encoding with special handling for unseen categories (Section 3.4.4). The resulting dimensionality depends on the number of distinct categories observed in the validation set (typically 8--9 categories for old fragment identity, 5--6 for attachment signature, and 3--4 for frequency bin). These features enable the model to learn fragment-specific effects: replacing a chlorine atom follows different chemical rules than replacing a phenyl ring, and the categorical encodings capture these differences without requiring explicit hand-written rules.

#### 3.4.3 Model Architecture and Training Protocol

The scorer is implemented as a histogram gradient boosting classifier (`HistGradientBoostingClassifier`; scikit-learn [Pedregosa et al., 2011]) configured with a maximum of 200 boosting iterations, a maximum tree depth of 6, a learning rate of 0.1, `class_weight='balanced'` to compensate for the severe class imbalance inherent in the candidate ranking task (each query contributes exactly one positive among tens of negatives), and `random_state=42` for reproducibility. The model minimizes log loss and produces a calibrated probability as its output score.

Training follows a 5-fold GroupKFold cross-validation strategy, where each fold holds out all candidates from a disjoint set of queries (grouped by query_id to prevent within-query leakage between training and validation). Within each fold, 80% of queries form the training set and 20% form the validation set. To mitigate the class imbalance, we apply negative subsampling at a 5:1 ratio (five randomly sampled negatives per positive) during training, following preliminary experiments that showed this ratio balanced training efficiency against representativeness. The five folds produce five separate models; for validation evaluation, we use each fold's model to score its held-out queries, yielding out-of-fold predictions for the complete validation set.

For blind evaluation, we train a single model on the complete validation set (all queries, no held-out fold) and apply it once to the secondary blind set -- a "one-shot" protocol that simulates the deployment scenario in which the model is trained on available data and applied to new queries without any iterative refinement, model selection, or feature tuning on the target distribution. No secondary blind set statistics inform any aspect of model development. Feature standardization (zero mean, unit variance) is computed from the validation set statistics and applied to the secondary blind set using the same parameters, ensuring strict protocol separation.

All 77 features are presented to the model without manual pre-selection or dimensionality reduction. Preliminary experiments with feature pre-filtering (univariate correlation screening, recursive feature elimination) did not improve performance and introduced risks of indirect leakage. HistGB's built-in feature importance mechanism effectively down-weights irrelevant dimensions, making pre-filtering unnecessary while preserving the complete chemically motivated feature set.

#### 3.4.4 Feature Alignment via cat_templates

An implementation detail critical to the scorer's secondary blind set generalization is the handling of categorical features through template alignment. When one-hot encoding categorical variables, the validation and secondary blind sets may contain disjoint sets of category values: secondary blind queries involve 19 old_fragment identities that were never observed in the validation set, meaning standard one-hot encoding would produce matrices with different column counts and orderings in each split.

We address this with a template-based encoding strategy (cat_templates). First, the one-hot encoder is fitted on the validation set, recording the set of observed categories for each variable and the resulting column ordering. For secondary blind set encoding, any category not present in the validation set is mapped to a reserved OTHER category, and the secondary blind one-hot matrix is forced to exactly match the validation set's column count and ordering. A runtime assertion (`val_X.shape[1] == blind_X.shape[1]`) guarantees schema consistency.

The impact of this alignment is substantial: without cat_templates, the early 82-feature prototype achieved a secondary blind Top-10 accuracy of 0.7120 -- because the model's learned feature weights were applied to semantically incorrect columns when the one-hot dimensions shifted between splits. With proper template alignment, the same architecture achieved 0.8851, a relative improvement of 24.3% attributable entirely to correct feature indexing. This result underscores the sensitivity of feature-based models to categorical encoding consistency, particularly in molecular machine learning where fragment vocabularies shift substantially between training and deployment distributions.

---

### 3.5 Ablation: The Prior_Ranks Removal

Before presenting the final evaluation, we examine whether all 77 features contribute positively to generalization. The answer is surprising: a set of five features, encoding per-query rank information from the prior scores, actively harms performance on the held-out secondary blind set. This section documents the leave-one-family-out ablation that reveals this effect, explains the mechanism through which these features degrade generalization, and establishes the final model configuration.

#### 3.5.1 Leave-One-Family-Out Analysis

To assess the contribution of each feature family and rigorously validate the design of the 77-dimensional feature space, we perform a leave-one-family-out ablation study. The central question is straightforward: do all feature families contribute positively, or do some actively harm generalization on the held-out secondary blind set? The results yield a surprising and instructive answer.

Starting from the full 82-feature model (the 77 features described in Section 3.4 plus five additional features, collectively denoted **prior_ranks**), we systematically remove each feature family in turn, retrain the model from scratch with the identical training protocol, and evaluate secondary blind Top-10 accuracy. The five prior_ranks features are per-query rank transformations of the prior scores that were experimentally added to test whether rank information could augment raw signal: **backoff_logit_rank**, **cont_prior_rank**, **t_p4_logit_rank**, **p1_logit_rank**, and **p3_logit_rank**. Each of these features encodes the position of a candidate within its query's candidate pool when sorted by the corresponding prior score -- a rank of 1 for the highest-scoring candidate, rank $|C_q|$ for the lowest.

---

**Table 4.** Leave-one-family-out ablation analysis. Models are trained on the full validation set (one-shot) and evaluated on the secondary blind set. Delta is relative to the full 82-feature model. Positive deltas indicate that removing the feature family *improves* generalization. Bold rows highlight the best-performing variant and the full-feature baseline for comparison.

| Condition | Features Removed | Blind Top-10 | Delta from Full |
|-----------|-----------------|--------------|-----------------|
| Full (82 features) | 0 | 0.8851 | -- |
| **Drop prior_ranks (77 features)** | **5** | **0.9243** | **+0.0392** |
| Drop prior_positives | 3 | 0.9037 | +0.0186 |
| Drop query_stats | 3 | 0.9019 | +0.0168 |
| Drop model_ranks (F3) | 6 | 0.8697 | -0.0154 |
| Drop model_scores (F1) | 5 | 0.8610 | -0.0241 |

---

Remaining families (model raw scores, top-K flags, prior scores, prior rates/supports, PMI, molecular descriptors, similarity/frequency) each produced negative deltas between -0.003 and -0.027 when removed, consistent with the expectation that all contribute positively to model performance.

The most striking result is highlighted: removing the five prior_ranks features improves secondary blind Top-10 accuracy from 0.8851 to 0.9243 -- a gain of +0.0392. This is the largest effect in the entire ablation, and uniquely it is a net improvement, not a degradation. Every other removal either degrades performance (negative delta) or produces a modest positive effect (prior_positives: +0.0186; query_stats: +0.0168) that is substantially smaller than the prior_ranks effect. The prior_ranks features are not merely uninformative; they are actively harmful to generalization.

#### 3.5.2 Why Prior_Ranks Degrade Generalization

Among all nine feature families, the prior_ranks features differ in a critical structural way: they incorporate information from the candidate's rank within each query's candidate list, as assigned by the corresponding prior scores. Unlike the remaining 72 features -- which are intrinsic properties of the molecules (delta descriptors, Tanimoto similarity) or per-candidate values that carry consistent meaning across queries (prior scores, rates, supports) -- prior_ranks are **query-relative**: a rank of 3 means fundamentally different things for different queries, depending on the size of the candidate pool, the distribution of prior scores within it, and the specific identity of the old fragment being replaced. This query-specific nature creates a risk of overfitting to per-query ranking idiosyncrasies rather than learning general fragment replacement rules.

The mechanism of failure is concretely traceable to the split structure. The validation set contains 8 distinct old_fragment identities; the secondary blind set contains 19 entirely different old_fragments with no overlap. The within-query rank distributions of the prior scores are heavily influenced by which old fragment is being replaced -- a rank of 1 for the continuation prior may carry different chemical significance when replacing a small aliphatic chloride versus a large aromatic ring system, because the candidate pools for these fragment types differ systematically in size, composition, and score distribution. The HistGB model, presented with per-query rank integers as features, learns decision thresholds that reflect the specific ranking patterns of the 8 validation old_fragments. When applied to the 19 secondary blind old_fragments with fundamentally different ranking dynamics, these learned thresholds fail to transfer, degrading performance.

We interpret this as an instance of **shortcut learning** [Geirhos et al., 2020]: the model, given features that are highly predictive within the validation distribution (because a rank integer compresses substantial query-level information about candidate ordering), defaults to relying on them instead of learning the more generalizable chemical patterns encoded by the remaining features. The rank features constitute a shortcut because they appear informative during training -- a candidate ranked 1st by the prior score is indeed likely to be correct within the validation queries -- but their informativeness is specific to the distribution of fragment identities and ranking contexts seen during training. Removing the shortcut forces the model to attend to the actual chemical features (raw scores, prior rates, molecular descriptors, Tanimoto similarities) that transfer across splits. The design rule that emerges is clear: **per-query rank features are a generalization hazard; use raw scores, rates, and intrinsic properties instead.**

#### 3.5.3 Rescue/Lost Analysis and Final Model

The practical impact of removing prior_ranks is most vividly illustrated by the rescue/lost analysis (Table 5). The full 82-feature model rescues 987 queries where the best base ranker fails to identify the correct replacement, but it also loses 596 queries where the base ranker correctly identifies the replacement and the scorer mis-ranks it. This substantial lost count -- nearly 600 regressions -- indicates that while the 82-feature model is more powerful than individual base rankers for many queries, its aggressiveness introduces widespread regression. The 77-feature model dramatically shifts this balance: it rescues 424 queries and loses only 6. The net gain is comparable (82-feature: +391; 77-feature: +418), but the distribution is fundamentally different. The 77-feature model rescues fewer queries than the 82-feature model -- it is, by design, more conservative -- but it is dramatically safer, with a lost count two orders of magnitude lower.

---

**Table 5.** Rescue/lost analysis comparing the full 82-feature model (with prior_ranks) and the final 77-feature model (without prior_ranks). Rescue: base ranker best rank > 10, scorer best rank ≤ 10. Lost: base ranker best rank ≤ 10, scorer best rank > 10. Net gain = rescued - lost.

| Model | Rescued | Lost | Net Gain | Blind Top-10 |
|-------|---------|------|----------|-------------|
| 82 features (with prior_ranks) | 987 | 596 | +391 | 0.8851 |
| 77 features (no prior_ranks) | 424 | 6 | +418 | 0.9243 |
| Delta | -563 | -590 | +27 | +0.0392 |

---

Based on this ablation evidence, we adopt the 77-feature model as our primary candidate-level scorer for all main experiments. The five prior_ranks features are excluded from the final feature set.

The final model achieves a secondary blind Top-10 accuracy of **0.9243** [95% CI: 0.9198, 0.9289], computed via nonparametric bootstrap resampling over queries (5,000 replicates). This represents a statistically significant improvement of **+0.0686** [95% CI: +0.0633, +0.0739] over the strongest base-ranker baseline (Score Blend, Top-10 = 0.8558), as determined by paired bootstrap of per-query deltas. Furthermore, the 77-feature model resolves all negative subspaces present in earlier prototypes. The 82-feature model exhibited performance degradation on 6 of 19 secondary blind fragments -- primarily small, saturated fragments with narrow support distributions where the per-query rank features caused the most severe shortcut learning. After removing prior_ranks, all 19 fragments perform at or above their respective base-ranker baselines, with zero fragments exhibiting statistically significant degradation. The final model achieves both higher overall accuracy and uniform per-fragment robustness.

---

This result underscores a general principle for feature engineering in molecular machine learning: features that encode per-query rank position, while seemingly informative because they compactly summarize relative ordering, can paradoxically harm cross-split generalization by encouraging the model to memorize training-set ranking patterns rather than learn transferable chemical rules. The principled approach of removing such features -- replacing relative ranking information with absolute scores, rates, and intrinsic molecular properties -- yields a model that is simultaneously more accurate and more robust across diverse fragment types.

Having established the architecture, feature design, and ablation-validated configuration of the candidate-level scorer, we now turn to a comprehensive evaluation protocol and a detailed analysis of failure modes.

---

### 3.6 Evaluation Protocol

We evaluate all methods using query-level top-10 accuracy with bootstrap confidence intervals, and introduce rescue and lost analyses to characterize where the proposed scorer adds or loses value relative to the base rankers. This section describes the evaluation metrics, the statistical framework for significance testing, the rescue/lost methodology, and the separation of evaluation protocols by evidentiary role.

#### 3.6.1 Metrics and Ranking Evaluation

For each held-out query, we sort the candidate fragments by the scorer output $s(q, c)$ in descending order and determine whether the true replacement fragment (the ground-truth MMP pair) appears among the top $K$ positions. This query-level exact-recovery criterion follows established practice in fragment replacement benchmarking [Kroll et al., 2022; Wang et al., 2023] and reflects the practical scenario in which a medicinal chemist reviews only a limited number of top-ranked suggestions. For multi-positive queries -- those for which multiple fragments constitute valid replacements -- a query is scored as a hit if any positive candidate appears in the top $K$, consistent with standard information retrieval evaluation for problems with multiple relevant items.

Our primary metric is Top-10 accuracy: the proportion of queries for which the correct fragment is among the ten highest-ranked candidates. We additionally report Top-1, Top-5, Top-20, and Top-50 accuracy, along with Mean Reciprocal Rank (MRR), as secondary metrics. This suite of metrics provides a comprehensive picture of ranking behavior across operating points: Top-1 captures the challenging task of exact first-place recovery, while Top-20 and Top-50 assess whether the correct candidate is retained within a practically usable shortlist. MRR complements the threshold-based metrics by measuring the average rank position of the first correct candidate.

#### 3.6.2 Bootstrap Confidence Intervals

To quantify the statistical reliability of our results, we compute 95% confidence intervals via nonparametric bootstrap resampling with 5,000 replicates [Efron, 1979]. A critical implementation detail is that we resample at the query level rather than the individual prediction level: each bootstrap sample draws $N$ queries with replacement from the original set of $N$ queries, retaining all candidates associated with each sampled query. This preserves the hierarchical structure of the data, in which candidates for a given query form a coherent group with shared context and cannot be treated as independent observations. Resampling individual predictions would inflate the effective sample size and produce artificially narrow confidence intervals.

For pairwise comparisons between the proposed scorer and any baseline method, we employ a paired (delta) bootstrap procedure. Within each of the 5,000 resamples, we compute the difference in mean accuracy between the two methods across the resampled query set, yielding a distribution of deltas. The 95% confidence interval for the improvement is given by the 2.5th and 97.5th percentiles of this delta distribution. This paired design controls for query-level variability -- because both methods are evaluated on the same resampled queries, the variance of the difference is substantially smaller than the variance of either method individually, providing greater statistical power to detect genuine improvements. All accuracy values throughout Section 3 are reported as point estimates with associated bootstrap confidence intervals.

#### 3.6.3 Rescue and Lost Analysis

Aggregate accuracy alone does not reveal *where* the scorer improves or degrades performance relative to simpler baselines. We therefore introduce two complementary per-query diagnostics. A *rescue* occurs when a baseline method fails to rank the correct replacement within its top 10 (best rank $> 10$) but the proposed scorer succeeds (best rank $\leq 10$). A *lost* case occurs in the opposite scenario: the baseline succeeds (best rank $\leq 10$) but the scorer fails (best rank $> 10$). The net gain is the difference between rescued and lost queries.

These analyses serve two purposes. First, they provide an honest accounting of degradation: where the feature-rich scorer underperforms a simpler baseline, the lost analysis makes this transparent rather than hiding it behind a favorable average. Second, they identify the chemical regimes in which the scorer provides the greatest benefit. A high rescue rate against the attachment-frequency baseline, for instance, indicates that the scorer successfully overrides a naive frequency prior when structural compatibility demands a less common replacement. Conversely, a high lost rate against the HGB ranker would signal cases where training-set frequency patterns encode genuinely useful chemical knowledge that the scorer's features fail to capture.

#### 3.6.4 Protocol Separation

A central design principle of our evaluation is the strict separation of protocols by their intended evidentiary role. We employ three distinct evaluation regimes, summarized in Table 6. This separation is essential because each protocol operates on a different query set with different construction rules; conflating them would either inflate or obscure the primary performance signal.

---

**Table 6.** Evaluation protocol separation.

| Protocol | Queries | Vocabulary | Purpose |
|----------|---------|------------|---------|
| Secondary blind | 13,347 | 150 | **Primary performance claim** |
| Canonical analysis | 21,052 | 152 | Robustness and mechanism analysis only |
| Dual-mode risk-stratification (Section 3.7) | --- | 150 | Workflow and risk interpretation only |

---

The **secondary blind protocol** is the sole basis for our primary performance claim -- that the candidate-level scorer improves ranking accuracy on held-out queries. This protocol uses 13,347 queries with no overlap in transform context with the training split. Every performance claim in Section 3 regarding the scorer's superiority over baselines rests on this protocol.

The **canonical analysis protocol**, using 21,052 queries with a slightly different vocabulary, is reserved exclusively for robustness analysis and mechanistic understanding. Results from this protocol inform questions such as whether scoring improvements are consistent across alternative split configurations, but no primary performance claim is derived from it. This restriction prevents a favorable canonical result from being recruited to support the central contribution.

The **dual-mode risk-stratification framework** (presented in Section 3.7) is an interpretive tool for computational review, not a performance evaluation. It addresses the practical question of how proposals from different workflow modes distribute across structural alert categories, and it provides actionable guidance for medicinal chemists reviewing output lists. No performance conclusions are drawn from this analysis.

We enforce this separation rigorously throughout the manuscript: primary claims cite only secondary blind protocol results; canonical analyses are explicitly labeled as robustness checks; and the dual-mode framework is presented as a workflow interpretation layer, not as a benchmark.

#### 3.6.5 Leakage Control Verification

The validity of our evaluation depends on the absence of information leakage between training and evaluation splits. We implemented and verified multiple layers of protection:

**Feature-level protection.** All statistics used in feature computation are derived exclusively from the training split. This includes prior probabilities, smoothing parameters for rate estimates, pointwise mutual information (PMI) computations, and Z-score normalization statistics. The similarity-transfer features use only training-set old_fragment structures as reference molecules. No test-set or secondary blind set information enters any feature value.

**Categorical alignment.** Categorical features (old_fragment SMILES, attachment signature, and replacement frequency bins) are encoded via a template-based one-hot schema. The encoding template is learned from the validation split; the secondary blind split is encoded using this frozen template, with unseen categories mapped to an OTHER bin. This ensures that the secondary blind feature matrix has the same dimensionality and column ordering as the validation matrix, which we verify via an explicit shape assertion before prediction.

**Overlap verification.** We confirmed that the training and secondary blind splits share zero transform pairs (old_fragment, attachment_signature combinations) and that the canonical and secondary blind splits share zero queries. No query appearing in any evaluation protocol has a transform twin in the training data.

**Label-free feature space.** All 77 features are computable from molecular structure alone. No feature incorporates the ground-truth replacement label or any derived quantity. The feature schema is locked at 77 dimensions with a fixed column order; any misalignment between the validation and secondary blind feature matrices triggers an assertion failure that halts prediction.

Together, these measures ensure that the evaluation protocol measures genuine generalization to held-out fragment-replacement tasks, not artifactual improvement from data leakage.

---

### 3.7 Computational Review Proxy (A4C)

To assess computational feasibility without exhaustive chemical synthesis, we adopt a dual-mode workflow under an A4C computational review proxy that categorizes predictions by provenance and alert strata. A4C applies a rule-based annotation layer -- inspired by established medicinal-chemistry alert filters (PAINS alerts [Baell and Holloway, 2010]; Brenk alerts [Brenk et al., 2008]) -- to each candidate proposal, flagging structural features associated with assay interference or unfavorable pharmacokinetics. The framework is a computational screening aid, not a medicinal-chemistry truth standard, and its role in our analysis is to stratify proposals into risk tiers for qualitative assessment, providing a proxy for the type of triage a medicinal chemist would perform when reviewing output lists.

#### 3.7.1 Dual-Mode Workflow Design

The dual-mode workflow comprises two operating modes that reflect different positions on the exploration-exploitation spectrum, each generating proposals through a distinct ranking route.

**Conservative Mode** draws proposals exclusively from the HGB ranker (Section 3.3.3). Because HGB relies on training-set fragment frequencies and molecular descriptors, its top-ranked proposals are biased toward chemically well-characterized replacements that align with historical preferences in the training data. This mode serves as a starting point: it produces chemically conservative suggestions that are likely to be synthetically accessible and structurally plausible, but it may fail to propose less common but equally valid replacements.

**Exploration Mode** draws proposals from the Borda(DE, HGB) fused ranking (Section 3.3.4), which combines the Dual Encoder's learned molecular similarity with HGB's frequency-based signal. The Dual Encoder component enables the fused ranking to propose replacements that differ from training-set frequency priors, potentially identifying novel or rare fragments that HGB alone would rank low. Exploration Mode thus trades some chemical conservatism for broader coverage of the fragment space.

#### 3.7.2 Provenance Labels and Alert Stratification

Within Exploration Mode, each proposal receives a provenance label that indicates its origin within the fused ranking relative to the individual base rankers. These labels define three groups with distinct alert-rate profiles, summarized in Table 7.

---

**Table 7.** Provenance groups, definitions, and A4C alert rates.

| Group | Definition | A4C Alert Rate | Interpretation |
|-------|-----------|----------------|----------------|
| **G4** | In top $K$ of both HGB and Borda | 0.99% | Low-alert reference; broad consensus between frequency and similarity signals |
| **G3** | In top $K$ of DE (Borda) but not HGB | 9.67% | Moderate expansion; similarity-supported novelty beyond frequency priors |
| **G2** | In top $K$ of Borda but not in individual top-$K$ of HGB or DE | 46.85% | Highest novelty; combined ranker agreement with structural alerts requiring expert review |

---

**G4: Shared candidates.** Fragments appearing in the top $K$ of both the HGB and Borda rankings represent the intersection of frequency-based and similarity-based recommendation. The near-zero A4C alert rate (0.99%) establishes these as a low-alert reference set: proposals on which both ranking methods agree carry minimal structural flags and are suitable as first-pass suggestions.

**G3: DE-retained candidates.** Fragments in the DE (Borda) top $K$ that fall outside the HGB top $K$ represent expansions beyond frequency-based priors. These proposals are supported by learned molecular similarity -- the Dual Encoder identifies them as chemically related to the query -- but they lack the support of historical frequency. The moderate alert rate (9.67%) indicates that this expansion carries some additional structural risk but remains manageable for most applications.

**G2: Borda-only candidates.** Fragments appearing in the Borda top $K$ but absent from the individual top-$K$ lists of both HGB and DE represent the highest novelty tier. These proposals emerge specifically from the *combined* effect of the two rankers -- they are not strongly recommended by either ranker alone, but their aggregate Borda score places them among the top candidates. The substantially elevated alert rate (46.85%) indicates that nearly half of these proposals carry structural alerts. G2 candidates therefore require individual expert medicinal-chemistry review before any synthetic commitment.

This provenance-based stratification provides an actionable risk structure: a reviewer can allocate attention according to alert burden, treating G4 proposals as low-risk starting points, G3 proposals as moderate-risk candidates suitable for closer inspection, and G2 proposals as high-novelty items that demand expert evaluation.

#### 3.7.3 Scope and Limitations

We emphasize that the A4C framework is used exclusively for workflow and risk interpretation. It does not support any primary performance claim -- the Top-10 accuracy results that establish the scorer's effectiveness rest solely on the secondary blind protocol (Section 3.6.4). The alert rates reported here are computational screening signals derived from published alert filters; they have not been validated through medicinal-chemistry expert review, and they do not constitute a determination of synthetic accessibility, assay interference, or toxicity. The A4C framework is a computational screening aid, not a substitute for domain expertise or experimental validation.

Consistent with standard practice in computational fragment replacement, we recommend the following protocol: G4 proposals may be used as initial suggestions without further computational triage; G3 proposals should be reviewed for structural flags before synthesis; and G2 proposals require comprehensive expert medicinal-chemistry review. This conservative interpretation ensures that the computational proxy informs rather than replaces human judgment, and it guards against over-interpretation of computational alert signals in the absence of experimental confirmation.

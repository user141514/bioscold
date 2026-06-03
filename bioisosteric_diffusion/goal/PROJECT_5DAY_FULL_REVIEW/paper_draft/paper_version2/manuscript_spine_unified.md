# Audited Ranking for Closed-Vocabulary Fragment Replacement

## Abstract

Fragment replacement is a common operation in lead optimization, but ranking models trained on matched molecular pairs can appear effective when evaluation rewards repeated fragment–attachment transform identities. We frame closed-vocabulary scaffold-conditioned fragment replacement ranking as a signal-transfer problem under transform-heldout distribution shift. The benchmark removes exact transform-level memorization and evaluates structure-derived replacement recovery on 13,347 secondary blind queries.

We evaluate an audited candidate-level scoring framework based on a histogram gradient-boosted scorer that integrates base-ranker outputs, train-derived priors, molecular descriptors, and frozen categorical features. The prospective 82-feature scorer achieved Top-10 = 0.8851. A post-audit locked 77-feature scorer achieved Top-10 = 0.9243 (95% CI [0.9199, 0.9287]) and improved Top-10 recovery over the Score Blend baseline (Top-10 = 0.8558) by $\Delta$Top-10 = 0.0686 (95% CI [0.0638, 0.0733]).

Post-audit analysis identified query-relative prior ranks as non-transferable shortcuts under fragment distribution shift. Pruning these features produced a gain of +0.0393 over the 82-feature scorer (95% CI [0.0354, 0.0432]) and reduced Score Blend hit loss from 596 to 101 queries, a 5.9× reduction. Because this pruning was post-selection, the 77-feature result motivates prospective replication rather than serving as a fully prospective feature-selection result. The benchmark measures structure-derived recovery, not activity preservation, and A4C provenance strata are reported only as coverage-limited computational triage.

---

## 1. Introduction

Fragment replacement is a recurring operation in lead optimization, but rigorous evaluation of computational replacement ranking remains difficult. We study closed-vocabulary scaffold-conditioned fragment replacement ranking: given an old fragment and its attachment signature, a model ranks attachment-compatible candidate replacements drawn from a training-derived vocabulary. The task is deliberately bounded to structure-derived replacement recovery from matched molecular pairs (MMPs), not activity preservation.

The main evaluation risk is that a model can look useful for the wrong reason. Random splits can place identical fragment–attachment transform identities in training and evaluation, rewarding transform-level memorization rather than transfer to unseen transform contexts. Even after exact transform recurrence is removed, empirical features can still behave as shortcuts when the fragment distribution shifts between development and blind evaluation. Query-relative prior ranks are especially suspect because their meaning depends on the candidate set and the fragment composition of the query distribution.

We address this evaluation problem directly: how can structure-derived fragment replacement ranking be evaluated without rewarding transform memorization or unstable shortcut features? First, we introduce a transform-heldout, closed-vocabulary benchmark that removes exact fragment–attachment transform memorization. Second, we evaluate an audited candidate-level scoring framework that integrates base-ranker outputs, train-derived priors, descriptors, similarity features, and frozen categorical encodings under secondary blind conditions. Third, post-audit analysis identifies query-relative prior ranks as non-transferable shortcuts under fragment distribution shift. A4C provenance stratification is kept as a supporting workflow layer for coverage-limited computational triage, not as medicinal chemistry validation.

The central spine of the study is therefore a signal-transfer question: when transform-level memorization is removed, which replacement-ranking signals still transfer, and which only looked useful because the split allowed shortcuts? Candidate-level scoring tests whether stable query-candidate evidence can transfer under the transform-heldout benchmark. The prior_ranks audit tests whether query-relative empirical ranks remain stable when the old-fragment distribution changes.

On 13,347 secondary blind queries, the prospective 82-feature candidate-level scorer achieved Top-10 = 0.8851. The post-audit locked 77-feature scorer achieved Top-10 = 0.9243 (95% CI [0.9199, 0.9287]) and improved over Score Blend (Top-10 = 0.8558) by $\Delta$Top-10 = 0.0686 (95% CI [0.0638, 0.0733]). Pruning prior_ranks improved Top-10 by +0.0393 over the 82-feature scorer (95% CI [0.0354, 0.0432]) and reduced Score Blend hit loss from 596 to 101 queries, a 5.9× reduction. Because this deletion was identified after blind diagnostics, the 77-feature scorer is a post-selection result that motivates prospective replication rather than a fully prospective feature-selection result.

The manuscript is organized by evidence role. The benchmark and prospective 82-feature scorer form the primary evaluation thread. The 77-feature scorer and prior_ranks interpretation form the post-audit mechanism thread. A4C provenance strata are reported as coverage-limited computational triage, not as activity validation or safety scoring.

---

## 2. Related Work

**Matched molecular pairs and structure-derived replacements.** Bioisosteric replacement is a longstanding design concept in medicinal chemistry [4]. Matched molecular pair analysis extracts local substitution patterns from large compound collections by identifying molecules that differ at a single structural site while sharing an identical scaffold [2,3]. Databases such as ChEMBL [1] make structure-derived replacement catalogues feasible at scale. These data record observed structural substitutions and serve as supervision for replacement proposal, but they do not encode activity continuity---a limitation inherited by any model trained on them.

**Rule-based and database-driven tools.** SwissBioisostere organizes observed replacements into a searchable database for ligand design [5], while CReM uses fragment environments from known compounds to perform context-aware replacement generation [6]. NeBULA extends this paradigm by mining over 700 medicinal chemistry references to compile 311,543 qualitative bioisosteric replacement reaction rules encoded as SMARTS and SMIRKS patterns, covering five replacement categories from scaffold hopping to general isosteric substitution [7]. These frameworks exploit empirical replacement context at scale, but they define different task formulations---typically open-ended generation or database lookup---rather than the closed-vocabulary query-level ranking benchmark studied here.

**Learning-based methods.** Deep learning methods for bioisosteric replacement include descriptor-based neural networks [8] and deep generative models that autonomously select fragments for removal and insertion while controlling multiple properties such as logP, drug-likeness, and synthetic accessibility simultaneously [9]. GraphBioisostere applies a Siamese graph neural network with attention-based global pooling (MMGX encoder) to ChEMBL-derived molecular pairs, training on 779,081 compound pairs across 1,897 targets, and demonstrates transfer learning for target-specific potency change prediction across five pharmaceutically relevant targets [10]. These methods show that structural representations capture replacement-relevant information beyond frequency statistics, but they are typically evaluated on random splits that may overestimate generalization. Our work complements the generative formulation with a ranking perspective that explicitly separates candidate scoring from candidate generation.

**Rank aggregation and fusion.** Borda count is a classical parameter-free aggregation rule for combining ranked lists [11]. Reciprocal rank fusion [12] provides a weighted alternative. Our use of fusion is narrower: we test whether structural and empirical rankers remain complementary under leakage-controlled evaluation and use Borda's parameter-free property as a practical necessity---the transform-heldout split discourages development-set-based tuning of fusion weights.

**Computational alert and risk filters.** PAINS [13] and Brenk alerts [14] are widely used medicinal-chemistry screening heuristics. Recent systematic activity-profile studies demonstrate that bioisosteric replacements, even structurally conservative ones, can produce unexpected off-target effects [15]. We adopt a similar philosophy for the A4C proxy: a rule-based annotation layer that organizes provenance strata and flags candidate-level structural alerts without converting exploratory outputs into approval decisions.

---

## 3. Methods

We present a leakage-controlled ranking workflow for closed-vocabulary fragment replacement. Section 3.1 formalizes the task, labels, and ranking metrics. Section 3.2 describes the transform-heldout benchmark and the secondary blind set. Sections 3.3 and 3.4 define the base rankers and the candidate-level scorer. Section 3.5 documents the post-audit removal of five prior_ranks features. Sections 3.6 and 3.7 describe evaluation, bootstrap uncertainty, rescue/lost analysis, leakage control, and the A4C computational review proxy.

The evidentiary roles are separated throughout the manuscript. Base-ranker architectures, scorer feature families F1-F9, and training hyperparameters were configured on a development/calibration set drawn from the training partition. The initial 82-feature scorer is the prospective candidate-level scorer evaluated on the secondary blind set. The 77-feature scorer is a locked post-audit configuration obtained after blind diagnostics motivated removal of five sparse prior-rank features. It is therefore suitable for a reproducible post-selection claim and for motivating prospective replication, but not for claiming fully prospective feature selection.

Baseline naming is also fixed. The HGB row in the main performance table is the standalone base ranker with Top-10 = 0.7437. The HGB-refit candidate score used in diagnostic rescue/lost analysis has Top-10 = 0.8623 and is not the same evidence row. These values are kept separate to avoid merging base-ranker and candidate-matrix diagnostics under a single HGB label.

---

### 3.1 Task Formulation

We formalize scaffold-conditioned fragment replacement as a closed-vocabulary, multi-positive learning-to-rank problem. The candidate vocabulary $\mathcal{V}_{\mathrm{train}}$ is training-derived and closed: it is constructed exclusively from the training split. Transform identities $(f^{\mathrm{old}}, \sigma)$ are held out across splits (Section 3.2). Let $\mathcal{Q} = \{q_i\}_{i=1}^{N}$ denote a query set. Each query pairs an old fragment with its attachment context,

$$
q_i = \bigl(f_i^{\mathrm{old}}, \sigma_i\bigr),
$$

where $f_i^{\mathrm{old}}$ is the fragment to be replaced and $\sigma_i$ is the attachment signature---the atom and bond context through which the fragment is connected to the scaffold. For query $q_i$, only candidates that are attachment-compatible with $\sigma_i$ are eligible:

$$
\mathcal{C}_i
=
\bigl\{c\in\mathcal{V}_{\mathrm{train}}:\chi(c,\sigma_i)=1\bigr\},
$$

where $\chi(c,\sigma_i)\in\{0,1\}$ is a deterministic attachment-compatibility predicate encoding valence and bond-order constraints. A scoring model with parameters $\theta$ assigns each candidate a real-valued score

$$
s_{\theta} : (q_i,c) \mapsto s_{\theta}(i,c)\in\mathbb{R},\qquad c\in\mathcal{C}_i.
$$

Candidates are ranked in decreasing order of $s_{\theta}(i,c)$. To make the ranking function well-defined under score ties, let $\tau_i(c)$ be a fixed deterministic tie-breaking key. The induced rank is

$$
\rho_{\theta}(i,c)
=
1+
\sum_{c'\in\mathcal{C}_i}
\mathbb{I}\!\bigl[
 s_{\theta}(i,c')>s_{\theta}(i,c)
 \;\lor\;
 \bigl(s_{\theta}(i,c')=s_{\theta}(i,c)\land \tau_i(c')<\tau_i(c)\bigr)
\bigr].
$$

**Supervision labels.** Let $\mathcal{M}$ denote the set of structure-derived replacement triples from MMP extraction after the train/evaluation split has been fixed (Section 3.2). For query $q_i$, the positive set is

$$
\mathcal{P}_i
=
\bigl\{c\in\mathcal{C}_i:
(f_i^{\mathrm{old}},\sigma_i,c)\in\mathcal{M}\bigr\},
$$

with candidate-level binary label

$$
y_{ic} = \mathbb{I}\!\bigl[c\in\mathcal{P}_i\bigr], \qquad c\in\mathcal{C}_i.
$$

Queries can be multi-positive. Labels are structure-derived: they indicate observed structural substitution, not activity preservation. This limitation motivates the computational review proxy (Section 3.7).

**Evaluation metrics.** The best positive rank for query $q_i$ under model $\theta$ is

$$
r_i^+(\theta)=\min_{c\in\mathcal{P}_i}\rho_\theta(i,c).
$$

The per-query hit at cutoff $K$ is

$$
h_i^{(K)}(\theta)=\mathbb{I}\!\bigl[r_i^+(\theta)\le K\bigr],
$$

with a hit scored if any positive fragment appears in the top $K$. Query-level Top-$K$ accuracy and mean reciprocal rank are

$$
\operatorname{Top@K}(\theta;\mathcal{Q})
=
\frac{1}{N}\sum_{i=1}^{N}h_i^{(K)}(\theta),
\qquad
\operatorname{MRR}(\theta;\mathcal{Q})
=
\frac{1}{N}\sum_{i=1}^{N}\frac{1}{r_i^+(\theta)}.
$$

Top-10 is the primary metric; Top-1, Top-5, Top-20, Top-50, and MRR are secondary diagnostics. Confidence intervals use nonparametric bootstrap at the query level (Section 3.6.2).

---

### 3.2 Benchmark Construction

**Data source and MMP extraction.** We use ChEMBL37K [1], a curated subset of approximately 37,000 drug-like molecules from ChEMBL33. MMPs are extracted by applying retrosynthetic bond-cleavage rules [2,3]: each molecule is recursively decomposed at acyclic single bonds to generate fragment--scaffold pairs; any two molecules sharing identical scaffold fragments but bearing different substituent fragments at the same attachment point are recorded as an MMP. The observed substitution $(f^{\mathrm{old}} \rightarrow c^*)$ provides a positive training example. Quality filters include salt stripping, stereochemistry normalization, removal of molecules with PAINS or Brenk alerts [13,14], and exclusion of fragments with molecular weight below 15 Da or above 250 Da. Because PAINS/Brenk filtering is applied during data construction, the A4C alert analysis in Section 3.7 is conditional on a pre-filtered fragment vocabulary and should not be interpreted as an alert rate for an unfiltered replacement catalogue.

**Decoy construction.** Negatives are constructed through a decoy repair procedure: for each query $q_i$, $f_i^{\mathrm{old}}$ is paired with a fragment $c_{\mathrm{decoy}} \in \mathcal{V}_{\mathrm{train}}$ attached to a non-cognate scaffold in the ChEMBL37K data. Four stages: (1) 1:1 positive-to-decoy ratio for balanced candidate sets; (2) wrong-positive removal: any decoy belonging to $\mathcal{P}_i$ is excluded; (3) deduplication of identical $(f_i^{\mathrm{old}}, c_{\mathrm{decoy}})$ pairs; (4) a 1:5 diagnostic set for class-imbalance stress testing, excluded from primary evaluation.

**Split design: transform-heldout.** A conventional random split of MMP transformations would allow the same $(f^{\mathrm{old}}, \sigma)$ pair---the *transform identity*---to appear in both training and test sets, enabling memorization rather than generalization. We prevent this with a **transform-heldout split**: all unique $(f^{\mathrm{old}}, \sigma)$ pairs are partitioned such that no pair appears in both training and evaluation splits.

We construct three evaluation tiers:

- **Development/calibration set.** A held-out subset of the training partition, used during method development for architecture selection, feature engineering, and hyperparameter configuration. The base ranker architectures, scorer feature families F1--F9, and all training hyperparameters were configured on this set. The subsequent removal of the experimentally added prior_ranks family is described separately in Section 3.5 as a post-audit feature-pruning step.
- **Secondary blind set.** 13,347 queries. Transform-heldout from training: zero $(f^{\mathrm{old}}, \sigma)$ overlap. The candidate vocabulary $\mathcal{V}_{\mathrm{train}}$ is training-derived and closed; transform identities are held out, but the blind queries and training queries draw candidates from the same vocabulary. This set serves as the primary evaluation benchmark.
- **Canonical analysis set.** 21,052 queries with a slightly larger vocabulary, used exclusively for robustness analysis and mechanistic investigation. No primary performance claims rest on this set.

The candidate matrices use a fixed closed vocabulary of 150 replacement candidates per query. Table M1 summarizes the train, development, and secondary blind matrices used for the JCIM-facing evidence patch.

---

**Table M1.** Candidate-matrix statistics for the closed-vocabulary ranking benchmark.

| Split | Candidate rows | Queries | Positive rows | Unique old fragments | Transform identities | Single-positive queries | Multi-positive queries |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 16,497,750 | 109,985 | 181,483 | 109 | 351 | 73,870 | 36,115 |
| Development/calibration | 2,006,250 | 13,375 | 21,408 | 8 | 27 | 8,686 | 4,689 |
| Secondary blind | 2,002,050 | 13,347 | 20,082 | 19 | 63 | 9,377 | 3,970 |

Each query has 150 candidate rows in these matrices. The median positive-set size is 1 in all three splits, with maxima of 23, 17, and 10 for train, development, and secondary blind, respectively.

---

**Leakage verification.** Table M2 reports overlap counts of $(f^{\mathrm{old}}, \sigma)$ transform identities between every pair of splits. Zero overlap is confirmed for all pairs.

---

**Table M2.** Leakage verification: overlap counts of $(f^{\mathrm{old}}, \sigma)$ transform identities.

| Split Pair | Transform Overlap | Verified |
|---|---|---|
| Train / Development set | 0 | Yes |
| Train / Secondary blind | 0 | Yes |
| Train / Canonical test | 0 | Yes |
| Canonical / Secondary blind queries | 0 | Yes |

---

This benchmark design combines a transform-heldout split with three evaluation tiers: development/calibration, secondary blind, and canonical analysis. The transform-heldout split is the central leakage-control mechanism: it ensures that evaluation measures generalization to unseen fragment-scaffold combinations rather than memorization of training-set transform identities.

---

### 3.3 Base Ranking Methods

We implement five baseline ranking methods spanning a range of inductive biases. These provide comparative baselines and the input features for the candidate-level scorer (Section 3.4). Each captures a distinct signal: global replacement statistics, learned structural compatibility, interpretable molecular properties, and their consensus.

All methods are evaluated under the protocol described in Section 3.6. Results appear in Section 4.

**Best-of-DE+HGB diagnostic.** As a reference bound, we compute the fraction of secondary blind queries for which at least one of DE or HGB places the correct fragment in its top 10. This is a diagnostic bound for per-query selection between the two structurally complementary rankers, not a ceiling for models using additional candidate-level features. Results are reported in Section 4.1.

#### 3.3.1 Attachment-Frequency Ranker

The Attachment-Frequency ranker estimates the empirical training-set prior. Let $n_{\mathrm{train}}(c,\sigma)$ be the count of training replacement pairs where $c$ appears under attachment $\sigma$:

$$
s_{\mathrm{att}}(c,\sigma)
=
\widehat{p}_{\mathrm{att}}(c\mid\sigma)
=
\frac{n_{\mathrm{train}}(c,\sigma)}
       {\sum_{c'\in\mathcal{V}_{\mathrm{train}}} n_{\mathrm{train}}(c',\sigma)}.
$$

The method has no learned parameters and serves as a simple empirical frequency baseline.

#### 3.3.2 Dual Encoder (DE) Ranker

The DE ranker learns a measure of structural compatibility through a dual-encoder architecture. The query encoder takes the Morgan fingerprint (ECFP, radius 2, 2048 bits) of $f_i^{\mathrm{old}}$ and a learned embedding of $\sigma_i$, concatenates them, and passes the result through a two-layer MLP (hidden dimensionality $d=128$, ReLU). The candidate encoder independently maps the candidate's Morgan fingerprint through a separate two-layer MLP (same architecture, not weight-tied):

$$
\mathbf{h}^{q}_{i}
=
g_{\phi}\!\left(x(f_i^{\mathrm{old}}),a(\sigma_i)\right),
\qquad
\mathbf{h}^{c}_{c}
=
h_{\psi}\!\left(x(c)\right).
$$

The DE score is cosine similarity:

$$
s_{\mathrm{DE}}(i,c)
=
\frac{\langle \mathbf{h}^{q}_{i},\mathbf{h}^{c}_{c}\rangle}
     {\lVert\mathbf{h}^{q}_{i}\rVert_2\,\lVert\mathbf{h}^{c}_{c}\rVert_2}.
$$

Training uses a margin-based ranking loss with $K=20$ negatives per positive and margin $\delta=0.3$:

$$
\mathcal{L}_{\mathrm{DE}}
=
\sum_{i=1}^{N}
\sum_{c^{+}\in\mathcal{P}_i}
\sum_{c^{-}\in\mathcal{N}_i}
\left[\delta - s_{\mathrm{DE}}(i,c^{+}) + s_{\mathrm{DE}}(i,c^{-})\right]_{+}.
$$

#### 3.3.3 Histogram Gradient-Boosted Ranker (HGB)

The HGB ranker uses explicit molecular features organized into four groups: (1) frequency features ($\widehat{p}_{\mathrm{att}}(c\mid\sigma)$, global fragment count, attachment-conditional count); (2) attachment signature features (bond order, ring membership, scaffold atom type, one-hot attachment signature); (3) molecular property descriptors (heavy atom count, molecular weight, logP, TPSA, ring count, HBD/HBA counts, rotatable bonds---for both candidate and query fragments); (4) fingerprint similarity (Tanimoto coefficient between Morgan fingerprints, radius 2, 2048 bits). The model is `HistGradientBoostingClassifier` (scikit-learn) with 200 iterations, max depth 6, learning rate 0.1. Candidates are ranked by predicted probability of being a valid replacement.

The HGB ranker's interpretability makes it the source for the Conservative Mode in the dual-mode workflow (Section 3.7).

#### 3.3.4 Borda Fusion and Score Blend

**Borda fusion.** For query $q_i$, let $\rho_m(i,c)$ be the rank assigned by method $m\in\{\mathrm{DE},\mathrm{HGB}\}$:

$$
S_{\mathrm{Borda}}(i,c)
= \sum_{m\in\{\mathrm{DE},\mathrm{HGB}\}}
\left(|\mathcal{C}_i| + 1 - \rho_m(i,c)\right).
$$

The parameter-free design is deliberate: the transform-heldout split discourages development-set-based tuning of fusion weights, since weights optimized on development queries need not generalize to unseen transform combinations.

**Score Blend (MLP + HGB).** A rank-only MLP (two layers, hidden dimensionality 32, ReLU) takes three per-candidate rank values (DE, HGB, Attachment-Frequency) and outputs $s_{\mathrm{MLP}}(q_i, c)$. The blended score with within-query $z$-score normalization is

$$
s_{\mathrm{blend}}(i,c)
= \lambda\, z_i\{s_{\mathrm{MLP}}\}(c)
+ (1-\lambda)\, z_i\{s_{\mathrm{HGB\text{-}refit}}\}(c),
\qquad \lambda=0.95,
$$

where $s_{\mathrm{HGB\text{-}refit}}$ is an HGB model refit with the MLP output as an additional feature. Weights were fixed on the development set.

---

### 3.4 Candidate-Level Scorer

#### 3.4.1 Architecture

The base rankers (Section 3.3) provide broad but shallow assessments. We introduce a **candidate-level scorer** that combines 77 features with histogram gradient boosting to produce per-candidate scores at finer granularity.

Given query $q_i$ and candidate $c$, the scorer computes $\boldsymbol{\phi}_{ic} = \Phi(q_i,c) \in \mathbb{R}^{77}$ and maps it to a probability:

$$
\widehat{p}_{ic} = F_{\Theta}(\boldsymbol{\phi}_{ic}), \qquad 0 \le \widehat{p}_{ic} \le 1.
$$

The training target is $y_{ic}$, fit by weighted binary log loss:

$$
\mathcal{L}(\Theta) = -\sum_{i=1}^{N}\sum_{c\in\mathcal{C}_i} w_{ic}
\left[ y_{ic}\log \widehat{p}_{ic} +(1-y_{ic})\log(1-\widehat{p}_{ic}) \right].
$$

We choose HistGB for three reasons: (1) mixed-type inputs (continuous, discrete, binary, one-hot) are handled natively; (2) non-linear feature interactions are captured---a high prior score with a large molecular weight difference signals a different replacement context than either alone; (3) built-in feature importance estimates support ablation without manual selection. We do not claim architectural novelty in the classifier itself; the methodological contribution is the leakage-controlled ranking setup, the audited candidate-level feature design, and the explicit separation between prospective evaluation and post-audit feature pruning.

#### 3.4.2 Feature Families with Audit Trail

The 77 features organize into 9 families. Table M3 provides each family's description, source, and audit status: whether it is train-only, label-free, retained in the final model, and whether it poses leakage risk.

---

**Table M3.** Feature families of the candidate-level scorer with audit trail.

| Family | Count | Source | Train-Only | Label-Free | Retained | Leakage Control |
|--------|-------|--------|------------|------------|----------|-----------------|
| F1: Model Scores | 5 | Base ranker outputs ($z$-normalized) | Yes | Yes | Yes | Train-set statistics only |
| F2: Model Raw Scores | 4 | Base ranker raw outputs | Yes | Yes | Yes | No evaluation-set dependence |
| F3: Model Ranks | 6 | Per-query ranks from base rankers | Yes | Yes | Yes | Query-relative; see Section 3.5 |
| F4: Top-K Flags | 4 | Binary: candidate in base ranker top-10 | Yes | Yes | Yes | Thresholds fixed at development time |
| F5: Prior Scores | 5 | Logit-transformed empirical priors (P0/P1/P3/P4) | Yes | Yes | Yes | Priors from train counts only |
| F6: Prior Statistics | 12 | Smoothed rates, support/positive counts | Yes | Yes | Yes | Counts from train; smoothing fixed on development set |
| F7: PMI / Conditional-Prior Contrast Scores | 6 | Pointwise mutual information and contrast scores across prior levels | Yes | Yes | Yes | PMI from train co-occurrence |
| F8: Molecular Descriptors | 10 | RDKit [18]: candidate properties + deltas from query | No fitting | Yes | Yes | Structure-derived; no label dependence |
| F9: Similarity & Frequency | 3 | Tanimoto similarity, global/attachment frequency | Yes | Yes | Yes | Frequency from train counts |
| Categorical (one-hot) | 22 | Fragment identity, attachment signature, frequency bin (one-hot expanded) | Yes | Yes | Yes | Frozen schema; unseen categories mapped to OTHER (Section 3.4.4) |
| Prior_Ranks (removed) | 5 | Per-query rank of prior scores | Yes | Yes | No | Removed post-audit; shortcut hazard |
| **Total Retained** | **77** | | | | | |

---

**Model-derived features (F1--F4).** Model scores (F1) and raw scores (F2) carry the primary signal from each base ranker. Model ranks (F3) encode relative ordering within each query's candidate pool. Top-K flags (F4) provide discrete consensus signals. Retained base-ranker ranks (F3) differ from the removed prior_ranks: base-ranker ranks summarize validated model outputs from trained rankers, whereas prior_ranks are sparse prior-score rank transforms that were removed as non-transferable query-local frequency shortcuts (Section 3.5.2).

**Prior and statistical features (F5--F7).** These encode empirical replacement frequencies across four provenance levels: P0 (global fragment frequency), P1 (attachment-conditioned), P3 (cluster-conditioned within the same MMP transformation class), P4 (transferred probability from chemically similar training fragments). Logit-transformed scores (F5) linearize the probability scale. Smoothed rates, support counts, and positive counts (F6) quantify statistical reliability. PMI / conditional-prior contrast scores (F7) capture information gain at each conditioning level relative to the baseline---for example, $\mathrm{PMI}(P3, P1)$ measures how much additional information the cluster-level context provides beyond the attachment-level context alone. These are computed from training-set co-occurrence statistics and are distinct from the removed per-query prior_ranks features (Section 3.5).

**Molecular descriptors (F8) and similarity (F9).** Absolute candidate properties and deltas from the query fragment (heavy atom count, molecular weight, logP, TPSA) encode steric and electronic compatibility. Morgan fingerprint Tanimoto similarity (radius 2, 2048 bits) and frequency features anchor the prediction in structural and empirical context.

**Categorical features.** Old fragment identity (SMILES), attachment signature, and replacement frequency bin are one-hot encoded via frozen categorical schema alignment (Section 3.4.4). The encoding template is learned from the development set; categories unseen in development (e.g., novel fragment identities in the secondary blind set) are mapped to a reserved OTHER category, and output dimensionality is forced to match the development schema. This captures fragment-specific and attachment-specific effects without hand-written rules, while ensuring consistent feature dimensionality across splits.

#### 3.4.3 Training Protocol

The scorer uses `HistGradientBoostingClassifier` (scikit-learn [17]) with max 200 iterations, max depth 6, learning rate 0.1, `class_weight='balanced'`, `random_state=42`.

Training uses 5-fold GroupKFold cross-validation grouped by query identifier, preventing within-query leakage. Negative subsampling at 5:1 mitigates class imbalance. All hyperparameters were fixed on the development set. The later post-audit change (Section 3.5) affected only the feature set---removing five prior_ranks features---and did not modify the HistGB hyperparameters.

For secondary blind evaluation, a single model is trained on the full development set and applied to the secondary blind set. No blind labels or candidate-level outcomes were used to compute feature values or train model parameters. Feature standardization parameters are computed from development set statistics. The final 77-feature configuration was selected through post-audit analysis prompted by the initial blind evaluation, as described in Section 3.5; this timeline is documented transparently therein.

The main reproducibility entry points are the locked scorer run, the query-aligned rescue/lost audit, and the Best-of-DE+HGB diagnostic recomputation. These scripts are listed in the Data and Software Availability statement so that score columns, query identifiers, and diagnostic bounds can be audited from the released project tree.

#### 3.4.4 Frozen Categorical Schema Alignment

Categorical features pose a schema-mismatch risk because the development and secondary blind sets can contain disjoint fragment identities. Without a frozen schema, one-hot encoded columns may differ in both dimensionality and semantic order across splits, causing learned feature weights to be applied to incorrect columns at inference time. We therefore use a frozen categorical schema: the one-hot encoder is fitted on the development set, recording observed categories and column ordering. For secondary blind encoding, unseen categories are mapped to a reserved OTHER category, and the output matrix is forced to match the development set's column count and ordering. A shape assertion guarantees consistency before prediction. The empirical impact of this alignment step is reported in Section 4.

---

### 3.5 Feature Ablation: The Prior_Ranks Removal

We used leave-one-family-out ablation to identify the post-audit feature-pruned configuration. This section documents the ablation design, its methodological outcome, and the timeline of the prior_ranks removal.

The prior_ranks removal was identified during post-audit analysis of the initial 82-feature scorer after blind-set diagnostics revealed fragment-specific degradation. We therefore report the 77-feature scorer as a post-audit locked model in this study, but not as a fully prospective pre-registered feature-selection result.

#### 3.5.1 Ablation Design

During initial development, five additional features encoding per-query rank positions of prior scores were added to the base 77-feature set (F1--F9 plus categorical features), yielding an augmented 82-feature configuration. These five features, collectively denoted **prior_ranks**, are backoff_logit_rank, cont_prior_rank, t_p4_logit_rank, p1_logit_rank, and p3_logit_rank. For prior score $u_j(i,c)$, the rank feature is:

$$
r_j^{\mathrm{prior}}(i,c)
=
1+
\sum_{c'\in\mathcal{C}_i}
\mathbb{I}\!\left[
 u_j(i,c')>u_j(i,c)
 \;\lor\;
 \bigl(u_j(i,c')=u_j(i,c)\land \tau_i(c')<\tau_i(c)\bigr)
\right].
$$

Let $\mathcal{A}_{82}$ be the index set of the augmented vector and $\mathcal{R}_{\mathrm{prior}} \subset \mathcal{A}_{82}$ the five prior_ranks indices. The 77-feature map is the coordinate projection:

$$
\boldsymbol{\phi}^{(77)}_{ic} = \Pi_{\mathcal{A}_{82}\setminus\mathcal{R}_{\mathrm{prior}}}\!\left(\boldsymbol{\phi}^{(82)}_{ic}\right).
$$

Each family was removed in turn, the model retrained from scratch with the identical protocol, and evaluated. The central question: do all families contribute positively?

#### 3.5.2 Why Prior_Ranks Degrade Generalization

The prior_ranks features differ structurally from all others: they encode per-query rank position, which is **query-relative**. A rank of 3 means different things for different queries depending on candidate pool size, score distribution, and fragment identity. The development set contains 8 distinct old_fragment identities; the secondary blind set contains 19. The HistGB model, presented with rank integers, learns thresholds reflecting the ranking patterns of the 8 development fragments. When applied to the 19 blind fragments with different ranking dynamics, these thresholds fail to transfer.

We interpret this as **shortcut learning** [16]: rank features are predictive within the development distribution, so the model defaults to them instead of learning transferable chemical patterns. Their informativeness is specific to the fragment identities and ranking contexts seen during development. Removing the shortcut forces the model to attend to raw scores, rates, and intrinsic molecular properties.

**Design rule.** Sparse prior-score rank transforms can become generalization hazards in cross-split molecular ranking; raw scores, rates, and intrinsic molecular properties are safer default representations under fragment distribution shift.

#### 3.5.3 Final Feature Set

The five prior_ranks features are excluded from the final model. The locked configuration uses the 77-feature set described in Section 3.4.2. Ablation results and final model performance are reported in Section 4.2. Rescue and lost analysis methodology is defined in Section 3.6.3; results appear in Section 4.3.

---

### 3.6 Evaluation Protocol

#### 3.6.1 Metrics and Ranking Evaluation

For each held-out query, candidates are sorted by descending $s_\theta(i,c)$. Multi-positive queries ($|\mathcal{P}_i|>1$) score a hit if any positive candidate appears in the top $K$. Primary metric: Top-10 accuracy $\operatorname{Top@10}(\theta;\mathcal{Q})$ as defined in Section 3.1. Secondary: Top-1, Top-5, Top-20, Top-50, MRR.

#### 3.6.2 Bootstrap Confidence Intervals

95% confidence intervals use nonparametric bootstrap with $B=5{,}000$ replicates, resampling at the query level (all candidates of a sampled query retained together). For the $b$-th replicate, $\mathcal{Q}^*_b$ of size $N$ is drawn with replacement from $\mathcal{Q}$:

$$
\hat{\theta}^*_b = \frac{1}{N}\sum_{i\in\mathcal{Q}^*_b} h_i^{(K)}(\theta),
\qquad
\operatorname{CI}_{95\%}(\theta)
= \left[\, \hat{\theta}^*_{(0.025)},\; \hat{\theta}^*_{(0.975)} \,\right]\!.
$$

For pairwise comparisons, a paired difference bootstrap is used:

$$
\hat{\Delta}^*_b = \hat{\theta}^*_b(m) - \hat{\theta}^*_b(b),
\qquad
\operatorname{CI}_{95\%}(\Delta)
= \left[\, \hat{\Delta}^*_{(0.025)},\; \hat{\Delta}^*_{(0.975)} \,\right]\!.
$$

This paired design controls for query-level variability, providing greater statistical power than independent intervals.

#### 3.6.3 Rescue and Lost Analysis

Aggregate accuracy alone does not reveal where a method improves or degrades relative to baselines. Let $b$ be a baseline and $m$ the model under evaluation:

$$
\operatorname{Rescue}_i(m,b)
=
\mathbb{I}\!\left[r_i^+(b)>10\right]\;
\mathbb{I}\!\left[r_i^+(m)\le 10\right],
$$

$$
\operatorname{Lost}_i(m,b)
=
\mathbb{I}\!\left[r_i^+(b)\le 10\right]\;
\mathbb{I}\!\left[r_i^+(m)>10\right].
$$

Aggregated counts $R(m,b)=\sum_i\operatorname{Rescue}_i$, $L(m,b)=\sum_i\operatorname{Lost}_i$, and net gain $G=R-L$ decompose accuracy changes into recoveries and regressions. Results are reported in Section 4.3.

#### 3.6.4 Protocol Separation

We enforce strict separation of evaluation protocols by evidentiary role (Table M4).

---

**Table M4.** Evaluation protocol separation.

| Protocol | Queries | Purpose |
|----------|---------|---------|
| Development/calibration | Held-out train partition | Architecture, feature, and hyperparameter selection |
| Secondary blind | 13,347 | **Primary performance claim** |
| Canonical analysis | 21,052 | Robustness and mechanism analysis only |
| Dual-mode risk-stratification (Section 3.7) | --- | Workflow and risk interpretation only |

---

Primary claims cite only secondary blind results. Canonical analyses are explicitly labeled as robustness checks. The dual-mode framework is a workflow interpretation layer.

#### 3.6.5 Leakage Control

**Feature-level.** All statistics (prior probabilities, smoothing parameters, PMI, $z$-score statistics) derive exclusively from the training split. Transfer features use only training-set reference molecules.

**Categorical alignment.** The frozen categorical schema (Section 3.4.4) ensures consistent dimensionality across splits; unseen categories map to OTHER.

**Overlap verification.** Zero $(f^{\mathrm{old}}, \sigma)$ overlap between training and any evaluation split (Table M2). Zero query overlap between canonical and secondary blind sets.

**Label-free feature space.** All 77 features are computable without evaluation labels. They are derived from molecular structure, base-ranker outputs, or training-set statistics. No feature incorporates ground-truth replacement labels. The schema is locked at 77 dimensions with fixed column order.

---

### 3.7 Computational Review Proxy (A4C)

Bioisosteric replacement, even when chemically conservative, can produce unexpected off-target effects---a risk documented in systematic studies of bioisosteric substitution patterns [15]. We adopt a dual-mode workflow under an A4C computational review proxy that categorizes proposals by provenance within the ranking pipeline and by structural alerts.

The A4C annotation layer applies PAINS alerts [13] and Brenk alerts [14] to flag structural features associated with assay interference, reactivity, or unfavorable pharmacokinetics. Property shifts (molecular weight, logP, TPSA, HBD/HBA counts) relative to the query fragment are also recorded. The framework is a computational screening aid, not a medicinal-chemistry truth standard.

#### 3.7.1 Dual-Mode Workflow

**Conservative Mode** draws proposals from the HGB ranker (Section 3.3.3). Its top-ranked candidates are biased toward chemically well-characterized replacements aligned with training-set frequency patterns.

**Exploration Mode** draws proposals from Borda(DE, HGB) (Section 3.3.4). The DE component enables proposals that differ from frequency priors, potentially identifying novel fragments HGB alone would miss.

#### 3.7.2 Provenance Labels and Alert Stratification

Within Exploration Mode, proposals receive provenance labels (Table M5). Let $\mathcal{K}_{m}$ denote the top-$K$ candidates from ranker $m \in \{\mathrm{HGB}, \mathrm{DE}\}$, and $\mathcal{K}_{\mathrm{Borda}}$ the Borda top-$K$ set.

---

**Table M5.** Provenance groups and definitions. Alert rates are reported in Section 4.5.

| Group | Definition | Interpretation |
|-------|-----------|----------------|
| $\mathrm{G4}$ | $\mathcal{K}_{\mathrm{HGB}} \cap \mathcal{K}_{\mathrm{Borda}}$ | Shared: consensus between frequency and similarity signals |
| $\mathrm{G3}$ | $\mathcal{K}_{\mathrm{Borda}} \setminus \mathcal{K}_{\mathrm{HGB}}$ (DE-elevated) | Expansion beyond frequency priors; similarity-supported |
| $\mathrm{G2}$ | $\mathcal{K}_{\mathrm{Borda}} \setminus (\mathcal{K}_{\mathrm{HGB}} \cup \mathcal{K}_{\mathrm{DE}})$ | Borda-emergent: combined ranker agreement, highest novelty |

---

This stratification provides a coverage-limited computational triage structure: G4 marks shared provenance, G3 marks DE-elevated exploratory proposals, and G2 marks Borda-emergent proposals with the highest observed alert burden among fully covered groups.

#### 3.7.3 Scope and Limitations

The A4C framework is used exclusively for workflow and risk interpretation. It does not support primary performance claims. Alert rates are computational screening signals derived from published rule-based filters; they have not been validated through medicinal-chemistry expert review and do not constitute determinations of synthetic accessibility, assay interference, or toxicity. Recent activity-profile studies indicate that off-target effects can arise from replacements that appear structurally conservative, underscoring that no rule-based proxy replaces experimental profiling. We therefore treat G2/G3/G4 as illustrative provenance strata for downstream inspection, not as validated decision rules.


## 4. Results

We evaluated all methods on the secondary blind protocol (13,347 queries; Section 3.2) using query-level Top-10 accuracy as the primary metric. Query-level 95% bootstrap confidence intervals were computed from 5,000 resamples unless otherwise stated. Sections 4.1-4.3 report the central ranking result, the post-audit feature-pruning analysis, and the rescue/lost profile. Section 4.4 documents a categorical schema alignment audit. Section 4.5 reports A4C provenance-group alert stratification.

---

### 4.1 Candidate-Level Scoring Improves Top-10 Recovery Under a Locked Evidence Hierarchy

Candidate-level scoring improved Top-10 recovery under the transform-heldout blind protocol, but the prospective and post-audit results have different evidentiary status. The initial 82-feature scorer achieved Top-10 = 0.8851 on 13,347 secondary blind queries and is the prospective candidate-level result. The locked 77-feature scorer, obtained after post-audit removal of prior_ranks features, achieved Top-10 = 0.9243. Against the strongest pre-D4S baseline, Score Blend, the 77-feature scorer improved Top-10 by +0.0686 using unrounded query-level hit indicators, with a 95% paired bootstrap CI of [+0.0638, +0.0733].

Table 1 places this result in context. Here, HGB denotes the standalone base ranker; the HGB-refit candidate-score diagnostic used in rescue/lost analysis is reported separately and should not be merged with this HGB row.

---

**Table 1.** Secondary blind Top-10 accuracy, with 95% bootstrap confidence intervals. The 82-feature scorer is the prospective candidate-level scorer. The 77-feature scorer is the locked post-audit feature-pruned scorer after prior_ranks removal. Best-of-DE+HGB is a diagnostic bound for per-query selection between DE and HGB, not a ceiling for models using additional candidate-level features.

| Method | Blind Top-10 | 95% CI |
|---|---:|---|
| Attachment-Frequency | 0.6019 | [0.5933, 0.6104] |
| HGB base ranker | 0.7437 | [0.7356, 0.7516] |
| Dual Encoder (DE) | 0.8055 | [0.7986, 0.8122] |
| Borda(DE, HGB) | 0.8384 | [0.8321, 0.8447] |
| MLP (rank-only) | 0.8402 | [0.8339, 0.8466] |
| Score Blend (MLP + HGB) | 0.8558 | [0.8495, 0.8621] |
| Initial 82-feature scorer | 0.8851 | [0.8796, 0.8903] |
| **Post-audit 77-feature scorer** | **0.9243** | **[0.9199, 0.9287]** |
| Best-of-DE+HGB diagnostic | 0.8686 | --- |

---

The Top-10 result should be interpreted as a triage-recall improvement. The 77-feature scorer improved Top-10 over Score Blend by +0.0686, but it was lower on Top-1 (-0.0341, 95% CI [-0.0428, -0.0255]) and mean reciprocal rank (-0.0073, 95% CI [-0.0133, -0.0013]). It therefore did not dominate early-rank metrics. We report the central advantage as improved recovery of at least one known replacement within the top 10, which matches the intended use case of candidate triage.

The initial 82-feature scorer also improved over Score Blend prospectively by +0.0293 in Top-10. The 77-feature scorer improved over the 82-feature scorer by +0.0393 (95% CI [+0.0354, +0.0432]) and also improved Top-1 and mean reciprocal rank relative to the 82-feature configuration. Because the prior_ranks removal was identified after blind diagnostics, this comparison supports the locked post-audit configuration on the same held-out set and motivates independent prospective replication.

Best-of-DE+HGB is included only as a diagnostic bound for per-query selection between DE and HGB. It is not a ceiling for candidate-level models using additional information. The 77-feature scorer can exceed this diagnostic because it uses molecular descriptors, train-derived prior statistics, PMI contrast scores, similarity features, and frozen-schema categorical encodings beyond the DE/HGB hit indicators.

Block-level diagnostics were favourable at coarse levels but not uniformly non-negative. Against Score Blend, the old-fragment macro Top-10 difference was +0.0434 with block bootstrap CI [0.0058, 0.0931], and the attachment-signature macro difference was +0.0679 with CI [0.0428, 0.0986]. However, two of 19 old-fragment blocks were negative, and the old-fragment/attachment identity macro difference was +0.0241 with CI [-0.0251, 0.0729]. We therefore avoid the older all-fragment non-degradation claim and treat the block results as diagnostic support for mostly favourable Top-10 recovery.

### 4.2 Removing Non-Transferable Prior-Rank Shortcuts Improves Blind Transfer

The strongest feature-pruning result was the removal of a feature family rather than the addition of a new module. Starting from the initial 82-feature candidate-level scorer, removing the five prior_ranks features improved blind Top-10 from 0.8851 to 0.9243. The paired query-level gain was +0.0393 with a 95% bootstrap CI of [+0.0354, +0.0432]. This result remains post-audit because the deletion was motivated by blind diagnostics, but the paired uncertainty estimate shows that the observed gain is not a sampling artefact of the query set.

---

**Table 2.** Leave-one-family-out ablation. Delta is relative to the full 82-feature configuration. Positive values indicate that removing the feature family improved Top-10 on the secondary blind set. The paired CI is shown for the locked prior_ranks deletion; additional exploratory ablations are reported in the Supporting Information.

| Condition | Features Removed | Blind Top-10 | Delta from 82-feature scorer | 95% CI for Delta |
|-----------|-----------------:|--------------:|-----------------------------:|------------------|
| Full 82-feature scorer | 0 | 0.8851 | -- | -- |
| **Drop prior_ranks (77-feature scorer)** | **5** | **0.9243** | **+0.0393** | **[+0.0354, +0.0432]** |
| Drop model_ranks (F3) | 6 | 0.8697 | -0.0154 | not reported |
| Drop model_scores (F1) | 5 | 0.8610 | -0.0241 | not reported |

---

The result is consistent with shortcut learning under fragment distribution shift. The removed prior_ranks features encode within-query rank positions of sparse empirical prior scores. Such ranks can be predictive in the development distribution, but their semantics change when the old-fragment composition shifts from the 8-fragment development setting to the 19-fragment secondary blind setting. Removing them forces the scorer to rely more heavily on features with more stable cross-split meaning, including raw model scores, train-derived rates and supports, molecular descriptors, fingerprint similarities, and frozen-schema categorical features.

The ablation should not be read as a broad causal claim that every retained family is individually necessary. The supported claim is narrower: prior_ranks had the largest locked post-audit deletion effect, whereas other feature-family deletions require separate paired uncertainty or prospective validation before being interpreted mechanistically.

### 4.3 Feature Pruning Recovers Baseline Misses While Reducing Regression

The 77-feature scorer changed the error profile as well as the aggregate Top-10 rate. Relative to Score Blend, the 77-feature scorer rescued 1,016 queries that the baseline missed and lost 101 queries that the baseline recovered. This produced a net per-query Top-10 gain of +915 hits, equivalent to +0.0686, with a 95% paired bootstrap CI of [+0.0638, +0.0733]. The rescue-to-loss ratio was 10.06, with a bootstrap CI of [8.25, 12.52]. Relative to the initial 82-feature scorer, the 77-feature scorer rescued 626 queries and lost 102, giving a net gain of +524 queries.

---

**Table 3.** Rescue/lost analysis for the post-audit 77-feature scorer. These are per-query Top-10 metrics computed from candidate-matrix score columns. The HGB-refit candidate-score row is diagnostic and is distinct from the HGB base-ranker row in Table 1. N = 13,347 blind queries.

| Reference (candidate-matrix score column) | Ref Top-10 | Rescue | Lost | Net | Arithmetic |
|-----------|-----------:|--------:|------:|-----:|-----------|
| Score Blend | 0.8558 | 1,016 | 101 | +915 | Yes |
| Initial 82-feature scorer | 0.8851 | 626 | 102 | +524 | Yes |
| Borda (candidate-matrix) | 0.8456 | 1,152 | 101 | +1,051 | Yes |
| HGB-refit candidate score | 0.8623 | 977 | 150 | +827 | Yes |
| DE | 0.8055 | 1,754 | 168 | +1,586 | Yes |

---

This rescue/lost shift explains why the post-audit 77-feature scorer is preferable for Top-10 triage. The initial 82-feature model rescued many baseline misses but also introduced 596 lost queries against Score Blend in the 82-feature audit. After prior_ranks removal, lost queries against Score Blend fell to 101, a 5.9-fold reduction, while the net Top-10 gain remained strongly positive. The result supports feature pruning as a reliability intervention for Top-10 recovery, not as evidence that all ranking metrics improved.

### 4.4 Categorical Schema Alignment Prevents Invalid Blind Prediction

During development, we identified a silent prediction bug caused by categorical feature misalignment. Naive one-hot encoding of fragment identity, attachment signature, and frequency-bin categories on the blind set produced a column space incompatible with the development set. Learned weights were therefore applied to semantically incorrect columns. This produced a blind Top-10 of 0.7120 on the 82-feature prototype.

The frozen categorical schema described in Section 3.4.4 corrected the same prototype to Top-10 = 0.8851. We report this as an implementation audit finding rather than a model contribution. The result documents a practical hazard in molecular machine-learning pipelines, where fragment vocabularies can shift between development and deployment even when the candidate vocabulary is closed.

### 4.5 Dual-Mode Provenance Strata Provide Coverage-Limited Triage Signals

Table 4 reports A4C alert rates for the Borda/HGB provenance groups used in the dual-mode workflow. This analysis is tied to workflow strata defined by base-ranker top-K sets, not to the 77-feature scorer's primary Top-10 evaluation.

---

**Table 4.** A4C alert rates by provenance group. G2 and G3 have complete A4C coverage. G4 is not assigned a group-wide alert estimate because only 5.63% of G4 candidates are covered by the A4C annotation layer.

| Group | Definition | A4C Coverage | Alert Rate |
|-------|-----------|--------------|------------|
| G4 | Shared ($\mathcal{K}_{\mathrm{HGB}} \cap \mathcal{K}_{\mathrm{Borda}}$) | 5.63% | 17.60% among covered; 94.37% unknown |
| G3 | DE-elevated ($\mathcal{K}_{\mathrm{Borda}} \setminus \mathcal{K}_{\mathrm{HGB}}$) | 100% | 9.67% |
| G2 | Borda-emergent ($\mathcal{K}_{\mathrm{Borda}} \setminus (\mathcal{K}_{\mathrm{HGB}} \cup \mathcal{K}_{\mathrm{DE}})$) | 100% | 46.85% |

---

G2 and G3 have complete A4C coverage and separate high-alert exploratory proposals from more moderate DE-supported expansion. G2 carried a hard-alert rate of 46.85%, whereas G3 carried a hard-alert rate of 9.67%. G4 had sparse A4C coverage: only 5.63% of candidates were covered, 17.60% of covered candidates had alerts, and 94.37% had unknown A4C status. We therefore interpret provenance labels as computational triage signals rather than calibrated safety scores. These alert rates are computational screening signals on a PAINS/Brenk pre-filtered vocabulary and are not experimentally validated determinations.

## 5. Discussion

### 5.1 Principal Findings

This study makes three linked contributions to leakage-controlled evaluation of closed-vocabulary fragment replacement ranking. First, the transform-heldout benchmark removes exact fragment–attachment transform identities from the training/evaluation overlap, shifting the task from transform-level memorization to signal transfer under distribution shift. Under this benchmark, the prospective 82-feature candidate-level scorer achieved Top-10 = 0.8851 on 13,347 secondary blind queries. The post-audit locked 77-feature scorer achieved Top-10 = 0.9243 and improved Top-10 recovery over Score Blend by +0.0686 (95% CI [+0.0638, +0.0733]).

Second, the audit shows why benchmark design matters beyond lowering performance numbers. Query-relative prior ranks appeared useful before post-audit inspection, but they became non-transferable shortcuts when the old-fragment distribution shifted. Removing five prior_ranks features improved Top-10 by +0.0393 over the 82-feature scorer (95% CI [+0.0354, +0.0432]) and reduced Score Blend hit loss from 596 to 101 queries, a 5.9× reduction. Because this deletion was identified after blind diagnostics, the 77-feature scorer remains a post-selection result rather than a fully prospective feature-selection result.

Third, A4C provenance stratification is a supporting workflow layer rather than a primary performance claim. G2 and G3 have 100% A4C coverage and alert rates of 46.85% and 9.67%, respectively. G4 has 5.63% A4C coverage, an alert rate of 17.60% among covered candidates, and 94.37% unknown status. These values support coverage-limited computational triage, not medicinal chemistry validation or replacement safety scoring.

The query-level routing experiments are best read as a secondary design observation. Their failure to improve deployment performance supports the emphasis on candidate-level scoring, but it is not a separate central contribution. The core message is that a transform-heldout benchmark changes which ranking signals appear trustworthy: stable candidate-level evidence can transfer, whereas query-relative prior ranks can become shortcuts.

### 5.2 Why Candidate-Level Scoring Succeeds Where Query-Level Routing Fails

A recurring development result was the failure of query-level routing strategies to improve deployment performance. One conditional-prior gate attempted to select among base rankers per query, and a later query router trained a classifier to predict which ranker would perform best for each query. Although the router was above chance as a classifier, neither approach improved over scoring every candidate independently and ranking within each query.

This pattern suggests that the useful signal in this benchmark resides mainly in query-candidate interactions. A query-level gate compresses all candidate evidence into one routing decision, discarding information that distinguishes candidates within the same query. The candidate-level scorer instead evaluates each query-candidate pair with structural, empirical, descriptor, similarity, and categorical features. This gives the model access to evidence at the level where the replacement decision is made.

### 5.3 Prior_Ranks as Non-Transferable Shortcuts

The prior_ranks finding clarifies the signal provenance problem created by transform-heldout evaluation. We separate four signal roles: transform-memory signal is removed by the split; transferable candidate-level signal comes from model scores, train-derived rates, descriptors, similarity, and frozen schema; query-relative shortcut signal appears in prior_ranks; and workflow triage signal appears in A4C outside the ranking evidence. Once exact transform memory is unavailable, model behaviour depends on which remaining signals transfer across fragment distribution shift. Query-relative prior ranks are less stable because they encode where a sparse empirical prior falls within a particular query's candidate set.

This instability is visible in the development-to-blind shift. The development/calibration setting contained 8 old fragments, whereas the secondary blind setting contained 19 old fragments. A query-relative prior-rank value can therefore change its meaning when the old-fragment distribution and candidate context change. In that setting, prior_ranks can help the model exploit distribution-specific regularities rather than transferable replacement evidence. The post-audit pruning result is thus not merely a feature-deletion anecdote. It shows that leakage-controlled benchmarks can reveal which signals were trustworthy and which only appeared useful under a permissive evaluation regime.

The finding should remain bounded. It does not imply that all rank features are harmful, since retained base-ranker rank features summarise model outputs evaluated under the same transform-heldout protocol. It also does not prove a single causal mechanism. Alternative explanations include multicollinearity among empirical-prior features, unstable rank transforms under sparse support, and threshold effects in histogram gradient-boosted trees. The supported interpretation is narrower: under this fragment distribution shift, query-relative prior ranks behaved as non-transferable shortcuts, while other candidate-level signals retained transferable value.

A4C is separate from this ranking-signal analysis. Its provenance strata help triage candidate outputs by coverage and computational alert burden, but they do not explain the prior_ranks effect and do not validate replacement safety. Keeping A4C separate preserves the central inference: the primary audit concerns signal transfer in ranking, not downstream medicinal-chemistry acceptability.

### 5.4 Dual-Mode Workflow and Computational Review

The dual-mode workflow addresses a reporting problem rather than validating medicinal chemistry outcomes. Conservative proposals aligned with training-set frequency patterns may be easier to trust, whereas exploratory proposals can expand coverage at the cost of higher structural-alert burden. Provenance labels and A4C strata provide a vocabulary for communicating this trade-off.

Three limitations apply. First, the A4C proxy relies on rule-based filters and simple property-shift thresholds, not target-specific activity or off-target data. Second, structurally plausible substitutions can still change activity profiles in unexpected ways. Third, alert rates are specific to the ChEMBL37K-derived and pre-filtered candidate vocabulary used here. The A4C analysis should therefore guide review priority rather than decision making.

### 5.5 Relationship to Generative and Database-Driven Approaches

The closed-vocabulary ranking formulation complements database-driven and generative approaches. Database resources can provide broad catalogues of observed transformations, while generative models can propose open-vocabulary modifications. Our setting asks a narrower question: can a model rank training-vocabulary replacements for a fixed attachment context under transform-heldout evaluation?

This narrower setting has an advantage. Because candidate generation is fixed, the benchmark can focus on leakage control, candidate scoring, and claim auditing. The limitation is equally clear: the method cannot propose fragments outside the training-derived vocabulary. Extending the framework to open-vocabulary candidates would require a separate evaluation design, especially to avoid confusing generative novelty with ranking performance.

### 5.6 Limitations and Future Work

Several limitations define the scope of this work. First, the benchmark uses one ChEMBL37K-derived dataset. Independent MMP datasets are needed to test generality. Second, labels are structure-derived and do not establish activity preservation. Third, the transform-heldout split controls transform-identity leakage but does not remove all possible chemical relatedness between development and evaluation examples. Fourth, the 77-feature configuration is post-audit locked rather than fully prospective. Fifth, the A4C proxy has not been validated against expert review or experimental profiling.

Future work should address these limitations directly. A pre-registered 77-feature configuration on a fresh blind split would test whether the post-audit pruning generalises. Activity-based endpoints would connect replacement ranking more directly to drug-discovery utility, although assay heterogeneity would need careful control. Expert medicinal-chemistry review of A4C-stratified candidates would also clarify whether computational alert strata are useful as triage signals.

## 6. Conclusion

This paper addresses a core evaluation problem in structure-derived fragment replacement ranking: random or permissive splits can reward transform-level memorization instead of transferable ranking signals. We built a transform-heldout, closed-vocabulary benchmark and evaluated candidate-level scoring on 13,347 secondary blind queries. The prospective 82-feature scorer achieved Top-10 = 0.8851, and the post-audit locked 77-feature scorer achieved Top-10 = 0.9243 (95% CI [0.9199, 0.9287]), improving over Score Blend by $\Delta$Top-10 = 0.0686 (95% CI [0.0638, 0.0733]).

The audit revealed that query-relative prior ranks can become non-transferable shortcuts under fragment distribution shift. Pruning prior_ranks improved Top-10 by +0.0393 over the 82-feature scorer (95% CI [0.0354, 0.0432]) and reduced Score Blend hit loss from 596 to 101 queries, a 5.9× reduction. Because this decision was made after blind diagnostics, the 77-feature scorer is a post-selection result and should motivate pre-registered prospective replication rather than be read as fully prospective feature selection.

The broader contribution is to frame fragment replacement ranking as a signal-transfer problem under transform-heldout distribution shift. The labels are structure-derived and do not establish activity preservation, and A4C provenance strata provide coverage-limited computational triage rather than validation or safety scoring. Future work should test the locked feature set on independent datasets, add activity-aware endpoints where assay heterogeneity can be controlled, and calibrate workflow triage against expert review. The benchmark-audit framing is the main takeaway: remove transform memorization first, then ask which ranking signals still transfer.

---

## Data and Software Availability

The analysis uses ChEMBL37K, a curated subset of ChEMBL33. A public source-code and evidence archive is available at https://github.com/user141514/paper1/tree/codex/jcim-algorithm-archive, with the curated experiment-code folder at `bioisosteric_diffusion/paper1_experiment_code_archive`. The code-archive commit is `b750de266d6d47e63de0eeaa945cc999e6f3e08c`, and the manuscript/supplement synchronization commit is `72b7807948c9a1432b99bf1529d98b3d994a3d1f`; the submission-candidate package is frozen under the repository tag `jcim-submission-candidate-20260601`. The repository includes the analysis scripts, split and feature-schema documentation, query-level audit tables, bootstrap summaries, and small evidence files required to audit the reported numbers. Large processed candidate matrices and any redistribution-restricted ChEMBL-derived artifacts will be deposited or documented separately before publication, subject to the redistribution terms of the underlying ChEMBL-derived data. An archival DOI will be provided upon acceptance.

---

## References

1. Zdrazil, B.; Felix, E.; Hunter, F.; et al. The ChEMBL Database in 2023: A Resource for Drug Discovery. *Nucleic Acids Res.* **2024**, *52* (D1), D1180-D1192. https://doi.org/10.1093/nar/gkad1064.
2. Hussain, J.; Rea, C. Computationally Efficient Algorithm to Identify Matched Molecular Pairs (MMPs) in Large Data Sets. *J. Chem. Inf. Model.* **2010**, *50* (3), 339-348. https://doi.org/10.1021/ci900450m.
3. Griffen, E.; Leach, A. G.; Robb, G. R.; Warner, D. J. Matched Molecular Pairs as a Medicinal Chemistry Tool. *J. Med. Chem.* **2011**, *54* (22), 7739-7750. https://doi.org/10.1021/jm200452d.
4. Patani, G. A.; LaVoie, E. J. Bioisosterism: A Rational Approach in Drug Design. *Chem. Rev.* **1996**, *96* (8), 3147-3176. https://doi.org/10.1021/cr950066q.
5. Wirth, M.; Zoete, V.; Michielin, O.; Sauer, W. H. B. SwissBioisostere: A Database of Molecular Replacements for Ligand Design. *Nucleic Acids Res.* **2013**, *41* (D1), D1137-D1143. https://doi.org/10.1093/nar/gks1059.
6. Polishchuk, P. CReM: Chemically Reasonable Mutations Framework for Structure Generation. *J. Cheminform.* **2020**, *12*, 28. https://doi.org/10.1186/s13321-020-00431-w.
7. Huang, S.; Wang, S.; Dong, J.; Xu, M.; Yuan, S. NeBULA: A Web-Based Novel Drug Design Platform for Up-to-Date Bioisosteric Replacement. *Med. Drug Discov.* **2025**, *28*, 100231. https://doi.org/10.1016/j.medidd.2025.100231.
8. Ertl, P. Identification of Bioisosteric Substituents by a Deep Neural Network. *J. Chem. Inf. Model.* **2020**, *60* (7), 3369-3375. https://doi.org/10.1021/acs.jcim.0c00290.
9. Kim, H.; Moon, S.; Zhung, W.; Lim, J.; Kim, W. Y. DeepBioisostere: Discovering Bioisosteres with Deep Learning for a Fine Control of Multiple Molecular Properties. *arXiv* **2025**, 2403.02706v2. Preprint; not peer-reviewed.
10. Masunaga, S.; Furui, K.; Kengkanna, A.; Ohue, M. GraphBioisostere: General Bioisostere Prediction Model with Deep Graph Neural Network. *J. Supercomput.* **2026**, *82*, 132. https://doi.org/10.1007/s11227-026-08232-y.
11. Dwork, C.; Kumar, R.; Naor, M.; Sivakumar, D. Rank Aggregation Methods for the Web. In *Proceedings of WWW10*; 2001; pp 613-622.
12. Cormack, G. V.; Clarke, C. L. A.; Buettcher, S. Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. In *Proceedings of SIGIR*; 2009; pp 758-759.
13. Baell, J. B.; Holloway, G. A. New Substructure Filters for Removal of Pan Assay Interference Compounds (PAINS). *J. Med. Chem.* **2010**, *53* (7), 2719-2740. https://doi.org/10.1021/jm901137j.
14. Brenk, R.; Schipani, A.; James, D.; et al. Lessons Learnt from Assembling Screening Libraries for Drug Discovery for Neglected Diseases. *ChemMedChem* **2008**, *3* (3), 435-444. https://doi.org/10.1002/cmdc.200700139.
15. Helmke, P. S.; Kandler, J.; Ilie, S.; Gaskin, L.; Ecker, G. F. Data-Driven Assessment of Bioisosteric Replacements and Their Influence on Off-Target Activity Profiles. *RSC Med. Chem.* **2025**, *16*, 6048-6058. https://doi.org/10.1039/d5md00686d.
16. Geirhos, R.; Jacobsen, J.-H.; Michaelis, C.; et al. Shortcut Learning in Deep Neural Networks. *Nat. Mach. Intell.* **2020**, *2* (11), 665-673. https://doi.org/10.1038/s42256-020-00257-z.
17. Pedregosa, F.; Varoquaux, G.; Gramfort, A.; et al. Scikit-Learn: Machine Learning in Python. *J. Mach. Learn. Res.* **2011**, *12*, 2825-2830.
18. Landrum, G.; et al. RDKit: Open-Source Cheminformatics Software. https://www.rdkit.org/ (accessed 2026-05-31).

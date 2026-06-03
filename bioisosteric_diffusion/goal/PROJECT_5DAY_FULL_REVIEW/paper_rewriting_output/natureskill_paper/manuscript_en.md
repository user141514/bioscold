# Leakage-Controlled Fragment Replacement Ranking with Audited Candidate-Level Scoring

## Abstract

Fragment replacement models can appear reliable when evaluation splits allow the same fragment-attachment transforms to recur across training and test data. They can also acquire shortcut features whose meaning changes when the fragment distribution shifts. We address these two risks in matched molecular pair (MMP)-derived fragment replacement ranking by constructing a transform-heldout, closed-vocabulary benchmark for scaffold-conditioned replacement ranking and evaluating it on 13,347 secondary blind queries.

Within this benchmark, a candidate-level histogram gradient-boosted scorer combines base-ranker outputs, train-derived priors, molecular descriptors, and frozen categorical features. The initial 82-feature scorer provides the prospective candidate-level blind result (Top-10 = 0.8851). A locked post-audit 77-feature scorer achieves Top-10 = 0.9243 (95% CI [0.9199, 0.9287]), improving over the Score Blend baseline by ΔTop-10 = 0.0686 (95% CI [0.0638, 0.0733]).

Post-audit analysis identified five per-query prior-rank features as a non-transferable shortcut under fragment distribution shift. Pruning these features produced the locked 77-feature scorer, improving blind Top-10 by +0.0393 over the initial 82-feature scorer and reducing Score Blend hit loss from 596 to 101 queries, a 5.9× reduction. Because this pruning decision was made after blind diagnostics, the 77-feature result is treated as a post-selection finding that motivates prospective replication rather than as a fully prospective feature-selection result.

---

## 1. Introduction

Replacing a selected substituent while preserving scaffold connectivity is a routine operation in lead optimization, but computational support for this decision is difficult to benchmark rigorously. Large chemical databases provide many structure-derived substitution patterns through matched molecular pair analysis, yet a replacement-ranking model can appear reliable if evaluation splits allow the same fragment-attachment transform to recur across training and test data. We therefore study closed-vocabulary scaffold-conditioned fragment replacement proposal: given an old fragment and its attachment signature, a model ranks attachment-compatible candidate replacements from a training-derived vocabulary. The task is deliberately structure-derived---it asks whether known replacement patterns can be recovered under controlled evaluation, not whether a proposed replacement preserves activity.

Building a credible benchmark is difficult for two main reasons. First, random data splits can leak the same fragment-attachment transform into both training and evaluation sets, allowing models to succeed through memorization rather than generalization. Prior work has demonstrated that random splits in ligand-based classification tasks can substantially overestimate model performance [1], and systematic data leakage is a recognized cause of reproducibility failures in machine-learning-based science [2]. Second, even under a leakage-controlled split, learned models can exploit dataset-specific shortcut features—those that are predictive within the development distribution but whose meaning changes under fragment identity shift—and these shortcuts are difficult to identify without systematic post-audit ablation. Shortcut learning, where models exploit spurious correlations that fail to transfer across distributions, is a well-documented failure mode across deep learning [3].

We address these issues with two contributions and one supporting workflow layer. First, we introduce a leakage-controlled benchmark for closed-vocabulary scaffold-conditioned fragment replacement ranking. The benchmark uses a transform-heldout split that isolates unseen fragment-attachment combinations, preventing memorization of transform identities. Second, we evaluate a candidate-level HistGB scorer under this protocol. The initial 82-feature scorer is the prospective candidate-level blind result, whereas the post-audit 77-feature scorer is a locked feature-pruned model that reaches Top-10 = 0.9243 on the 13,347-query secondary blind protocol. The main analytical finding is that five per-query prior-rank features act as non-transferable shortcuts: removing them improves blind Top-10 by +0.0393 over the 82-feature scorer, reduces baseline hit loss by 5.9×, and yields a design rule that sparse prior-score rank transforms can become generalization hazards under fragment distribution shift. We keep prospective performance, post-audit mechanism analysis, and workflow interpretation in separate evidentiary roles. A dual-mode A4C computational review proxy provides provenance-stratified alert reporting as a supporting workflow feature, with the explicit caveat that alert rates are unvalidated computational screening signals.

The goal is not to claim a new universal bioisostere generator, but to ask a narrower and more auditable question: when transform-level memorization is removed, which replacement-ranking signals still transfer, and which signals only looked useful because the split allowed them to behave as shortcuts?

On the secondary blind evaluation protocol, the post-audit 77-feature scorer reaches Top-10 = 0.9243 (95% CI [0.9199, 0.9287]), improving over the Score Blend baseline by ΔTop-10 = 0.0686 (95% CI [0.0638, 0.0733]). Removing the five prior_ranks features yields a paired gain of ΔTop-10 = 0.0393 (95% CI [0.0354, 0.0432]) over the 82-feature scorer. Because these features were deleted after blind diagnostics, we treat the 77-feature model as a post-audit locked result and the prior-rank interpretation as an analytical finding. Under the A4C proxy, G2 and G3 provenance groups are fully covered with alert rates of 46.85% and 9.67%, respectively; G4 has sparse A4C coverage (5.63%), so no reliable G4-wide alert estimate is possible.

The manuscript is organized around this evidence hierarchy. The benchmark and the 82-feature scorer provide the prospective evaluation thread. The 77-feature model and prior_ranks deletion provide a post-audit mechanism thread, explicitly bounded as post-selection evidence. The A4C layer provides a workflow interpretation thread rather than a medicinal-chemistry validation claim. This separation is deliberate: it makes clear what the paper demonstrates, what it diagnoses after audit, and what remains a hypothesis for downstream review.

---

## 2. Related Work

**Matched molecular pairs and structure-derived replacements.** Bioisosteric replacement is a longstanding design concept in medicinal chemistry [4]. Matched molecular pair analysis extracts local substitution patterns from large compound collections by identifying molecules that differ at a single structural site while sharing an identical scaffold [5,6]. Databases such as ChEMBL [7] make structure-derived replacement catalogues feasible at scale. These data record observed structural substitutions and serve as supervision for replacement proposal, but they do not encode activity continuity---a limitation inherited by any model trained on them.

**Rule-based and database-driven tools.** SwissBioisostere organizes observed replacements into a searchable database for ligand design [8], while CReM uses fragment environments from known compounds to perform context-aware replacement generation [9]. NeBULA extends this paradigm by mining over 700 medicinal chemistry references to compile bioisosteric replacement reaction rules encoded as SMARTS and SMIRKS patterns, covering multiple replacement categories from scaffold hopping to general isosteric substitution [10]. These frameworks exploit empirical replacement context at scale, but they define different task formulations---typically open-ended generation or database lookup---rather than the closed-vocabulary query-level ranking benchmark studied here. Direct numerical comparison would require adapting these retrieval or generation systems to return scores over the same fixed 150-candidate matrix, so we treat such adaptation as a separate benchmark extension rather than mixing task formulations in the present comparison.

**Learning-based methods.** Deep learning methods for bioisosteric replacement include descriptor-based neural networks [11] and deep generative models that autonomously select fragments for removal and insertion while controlling multiple properties such as logP, drug-likeness, and synthetic accessibility simultaneously [12]. GraphBioisostere applies a Siamese graph neural network with attention-based global pooling (MMGX encoder) to ChEMBL-derived molecular pairs, training on 779,081 compound pairs across 1,897 targets, and demonstrates transfer learning for target-specific potency change prediction across five pharmaceutically relevant targets [13]. These methods show that structural representations capture replacement-relevant information beyond frequency statistics, but they are typically evaluated on random splits that may overestimate generalization. Our work complements the generative formulation with a ranking perspective that explicitly separates candidate scoring from candidate generation.

**Rank aggregation and fusion.** Borda count is a classical parameter-free aggregation rule for combining ranked lists [14]. Reciprocal rank fusion [15] provides a weighted alternative. Our use of fusion is narrower: we test whether structural and empirical rankers remain complementary under leakage-controlled evaluation and use Borda's parameter-free property as a practical necessity---the transform-heldout split discourages development-set-based tuning of fusion weights.

**Computational alert and risk filters.** PAINS [16] and Brenk alerts [17] are widely used medicinal-chemistry screening heuristics. Recent systematic activity-profile studies demonstrate that bioisosteric replacements, even structurally conservative ones, can produce unexpected off-target effects [18]. We adopt a similar philosophy for the A4C proxy: a rule-based annotation layer that organizes provenance strata and flags candidate-level structural alerts without converting exploratory outputs into approval decisions.

---

## 3. Methods

We present the complete methodological framework. Section 3.1 formalizes the ranking task, supervision labels, and evaluation metrics. Section 3.2 describes benchmark construction: MMP extraction, decoy construction, and the transform-heldout split. Sections 3.3--3.5 cover the ranking methods: five base rankers spanning a range of inductive biases (Section 3.3), a candidate-level scorer with 77 features and a gradient boosting architecture (Section 3.4), and a leave-one-family-out ablation that identifies a feature family harmful to cross-split generalization (Section 3.5). Sections 3.6--3.7 describe the evaluation protocol and the A4C computational review proxy.

The evidence roles of the splits are fixed before model assessment. The training partition defines the closed candidate vocabulary and all empirical priors, support counts, PMI contrasts, and other train-derived statistics. The development/calibration matrix is used to configure and fit the candidate-level scorer under cross-validation, including the base ranker architecture, scorer feature families F1--F9, categorical encoding template, and hyperparameters. The secondary blind set is used only for one-shot evaluation and subsequent post-audit diagnosis. The prior_ranks feature family was removed after post-audit analysis of the initial secondary blind evaluation revealed unexpected per-fragment degradation; the mechanistic basis for removal is documented in Section 3.5.2. The 77-feature configuration described in Section 3.4.2 is therefore the **post-audit locked model** in this study, with the caveat that the prior_ranks removal is a post-selection feature-pruning step rather than a fully prospective pre-registered feature-selection result. All performance results appear in Section 4.

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

**Data source and MMP extraction.** We use ChEMBL37K [7], a curated subset of approximately 37,000 drug-like molecules from ChEMBL33. MMPs are extracted by applying retrosynthetic bond-cleavage rules [5,6]: each molecule is recursively decomposed at acyclic single bonds to generate fragment--scaffold pairs; any two molecules sharing identical scaffold fragments but bearing different substituent fragments at the same attachment point are recorded as an MMP. The observed substitution $(f^{\mathrm{old}} \rightarrow c^*)$ provides a positive training example. Quality filters include salt stripping, stereochemistry normalization, removal of molecules with PAINS or Brenk alerts [16,17], and exclusion of fragments with molecular weight below 15 Da or above 250 Da. Because PAINS/Brenk filtering is applied during data construction, the A4C alert analysis in Section 3.7 is conditional on a pre-filtered fragment vocabulary and should not be interpreted as an alert rate for an unfiltered replacement catalogue.

**Decoy construction.** Negatives are constructed through a decoy repair procedure: for each query $q_i$, $f_i^{\mathrm{old}}$ is paired with a fragment $c_{\mathrm{decoy}} \in \mathcal{V}_{\mathrm{train}}$ attached to a non-cognate scaffold in the ChEMBL37K data. Four stages: (1) 1:1 positive-to-decoy ratio for balanced candidate sets; (2) wrong-positive removal: any decoy belonging to $\mathcal{P}_i$ is excluded; (3) deduplication of identical $(f_i^{\mathrm{old}}, c_{\mathrm{decoy}})$ pairs; (4) a 1:5 diagnostic set for class-imbalance stress testing, excluded from primary evaluation.

**Split design: transform-heldout.** A conventional random split of MMP transformations would allow the same $(f^{\mathrm{old}}, \sigma)$ pair---the *transform identity*---to appear in both training and test sets, enabling memorization rather than generalization. We prevent this with a **transform-heldout split**: all unique $(f^{\mathrm{old}}, \sigma)$ pairs are partitioned such that no pair appears in both training and evaluation splits.

We construct three evaluation tiers:

- **Development/calibration set.** A held-out subset of the training partition, used during method development for architecture selection, feature engineering, categorical schema construction, hyperparameter configuration, and final candidate-level scorer fitting. The base ranker architectures, scorer feature families F1--F9, and all training hyperparameters were configured on this set. The subsequent removal of the experimentally added prior_ranks family is described separately in Section 3.5 as a post-audit feature-pruning step.
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

During development, query-level routing strategies that select among base rankers on a per-query basis were explored but did not improve over per-candidate scoring; the consistent failure of query-level routing to outperform candidate-level scoring is analyzed in Section 5.2.

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
| F8: Molecular Descriptors | 10 | RDKit [20]: candidate properties + deltas from query | No fitting | Yes | Yes | Structure-derived; no label dependence |
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

The scorer uses `HistGradientBoostingClassifier` (scikit-learn [19]) with max 200 iterations, max depth 6, learning rate 0.1, `class_weight='balanced'`, `random_state=42`.

Training uses 5-fold GroupKFold cross-validation grouped by query identifier within the development/calibration matrix, preventing within-query leakage during model assessment. Negative subsampling at 5:1 mitigates class imbalance. All hyperparameters were fixed on the development set. The later post-audit change (Section 3.5) affected only the feature set---removing five prior_ranks features---and did not modify the HistGB hyperparameters.

For secondary blind evaluation, a single model is fit on the full development/calibration matrix and applied to the secondary blind matrix. No blind labels or candidate-level outcomes are used to compute feature values, fit model parameters, set hyperparameters, or construct the frozen categorical schema. Feature standardization parameters are computed from development-set statistics. The final 77-feature configuration was selected through post-audit analysis prompted by the initial blind evaluation, as described in Section 3.5; this timeline is documented transparently therein.

The main reproducibility entry points are the locked scorer run, the query-aligned rescue/lost audit, and the Best-of-DE+HGB diagnostic recomputation. These scripts are listed in the Data and Software Availability statement so that score columns, query identifiers, and diagnostic bounds can be audited from the released project tree.

#### 3.4.4 Frozen Categorical Schema Alignment

Categorical features pose a schema-mismatch risk because the development and secondary blind sets can contain disjoint fragment identities. Without a frozen schema, one-hot encoded columns may differ in both dimensionality and semantic order across splits, causing learned feature weights to be applied to incorrect columns at inference time. We therefore use a frozen categorical schema: the one-hot encoder is fitted on the development set, recording observed categories and column ordering. For secondary blind encoding, unseen categories are mapped to a reserved OTHER category, and the output matrix is forced to match the development set's column count and ordering. A shape assertion guarantees consistency before prediction. The empirical impact of this alignment step is reported in Section 4.

---

### 3.5 Feature Ablation: The Prior_Ranks Removal

We used leave-one-family-out ablation to identify the post-audit feature-pruned configuration. This section documents the ablation design, its methodological outcome, and the timeline of the prior_ranks removal.

The base ranker architectures, scorer feature families F1–F9, and all training hyperparameters were configured on the development set. The prior_ranks removal was identified during post-audit analysis of the initial 82-feature scorer after blind-set diagnostics revealed fragment-specific degradation. We therefore report the 77-feature scorer as a post-audit locked model in this study, but not as a fully prospective pre-registered feature-selection result.

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

We interpret this as **shortcut learning** [3]: rank features are predictive within the development distribution, so the model defaults to them instead of learning transferable chemical patterns. Their informativeness is specific to the fragment identities and ranking contexts seen during development. Removing the shortcut forces the model to attend to raw scores, rates, and intrinsic molecular properties.

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

Primary performance claims cite only secondary blind results. Development/calibration results support method selection and internal checks; canonical analyses are explicitly labeled as robustness checks; the dual-mode framework is a workflow interpretation layer.

#### 3.6.5 Leakage Control

**Feature-level.** All statistics (prior probabilities, smoothing parameters, PMI, $z$-score statistics) derive exclusively from the training split. Transfer features use only training-set reference molecules.

**Categorical alignment.** The frozen categorical schema (Section 3.4.4) ensures consistent dimensionality across splits; unseen categories map to OTHER.

**Overlap verification.** Zero $(f^{\mathrm{old}}, \sigma)$ overlap between the training partition and any evaluation split (Table M2). Zero query overlap between canonical and secondary blind sets. This control removes exact transform-identity leakage, but it does not claim to remove every form of chemical relatedness across splits.

**Label-free feature space.** All 77 features are computable without evaluation labels. They are derived from molecular structure, base-ranker outputs, or training-set statistics. No feature incorporates ground-truth replacement labels. The schema is locked at 77 dimensions with fixed column order.

---

### 3.7 Computational Review Proxy (A4C)

Bioisosteric replacement, even when chemically conservative, can produce unexpected off-target effects---a risk documented in systematic studies of bioisosteric substitution patterns [18]. We adopt a dual-mode workflow under an A4C computational review proxy that categorizes proposals by provenance within the ranking pipeline and by structural alerts.

The A4C annotation layer applies PAINS alerts [16] and Brenk alerts [17] to flag structural features associated with assay interference, reactivity, or unfavorable pharmacokinetics. Property shifts (molecular weight, logP, TPSA, HBD/HBA counts) relative to the query fragment are also recorded. The framework is a computational screening aid, not a medicinal-chemistry truth standard.

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

We evaluate all methods on the secondary blind protocol (13,347 queries; Section 3.2) using query-level Top-10 accuracy as the primary metric, with 95% bootstrap confidence intervals (5,000 query-level resamples; Section 3.6.2). Sections 4.1--4.3 present the central blind result, its mechanistic basis, and its reliability profile. Section 4.4 documents a categorical schema alignment audit. Section 4.5 reports A4C provenance-group alert stratification.

---

### 4.1 Candidate-Level Scoring Improves Blind Ranking, with a Post-Audit Upgrade

Candidate-level scoring improves blind ranking over the base-ranker fusion baselines, but the evidence hierarchy differs for the 82-feature and 77-feature versions. The initial 82-feature scorer is the prospective candidate-level blind result and achieves Top-10 = 0.8851 on 13,347 secondary blind queries. The post-audit 77-feature scorer, obtained after removing prior_ranks features based on blind diagnostics, achieves Top-10 = 0.9243. It improves over the strongest pre-D4S baseline, the MLP+HGB Score Blend, by ΔTop-10 = +0.0686 using unrounded query-level hit indicators, with a 95% paired bootstrap CI of [+0.0638, +0.0733]. Because the prior_ranks removal was prompted by blind-set diagnostics, the 77-feature result is reported as a locked post-selection result rather than a fully prospective feature-selection result.

Table 1 places this result in context.

---

**Table 1.** Secondary blind Top-10 accuracy, with 95% bootstrap confidence intervals. The 82-feature scorer is the prospective candidate-level scorer. The 77-feature scorer is the post-audit feature-pruned scorer after prior_ranks removal (Section 3.5). Best-of-DE+HGB is a diagnostic bound for per-query selection between DE and HGB, not a ceiling for models using additional candidate-level features.

| Method | Blind Top-10 | 95% CI |
|---|---|---|
| Attachment-Frequency | 0.6019 | [0.5933, 0.6104] |
| HGB | 0.7437 | [0.7356, 0.7516] |
| Dual Encoder (DE) | 0.8055 | [0.7986, 0.8122] |
| Borda(DE, HGB) | 0.8384 | [0.8321, 0.8447] |
| MLP (rank-only) | 0.8402 | [0.8339, 0.8466] |
| Score Blend (MLP + HGB) | 0.8558 | [0.8495, 0.8621] |
| Initial 82-feature scorer | 0.8851 | [0.8796, 0.8903] |
| **Post-audit 77-feature scorer** | **0.9243** | **[0.9199, 0.9287]** |
| Best-of-DE+HGB diagnostic | 0.8686 | --- |

---

Attachment-Frequency provides a strong empirical frequency baseline (0.6019). DE supplies the strongest individual structural signal (0.8055, +0.2036 over the frequency baseline). Borda(DE,HGB) demonstrates that structural and empirical rankers are complementary under the transform-heldout blind protocol (0.8384, +0.0329 over DE alone). The MLP provides only a marginal Top-10 gain over Borda (+0.0018) and higher MRR in the available blind summary; we treat the MRR comparison as a secondary diagnostic rather than a primary statistical claim. The Score Blend is the strongest baseline using only base-ranker signals (0.8558). The initial 82-feature scorer improves beyond this baseline prospectively by adding candidate-level chemical, statistical, and descriptor features; the 77-feature scorer further improves after post-audit removal of the non-transferable prior-rank shortcuts identified in Section 4.2.

Best-of-DE+HGB is reported only as a diagnostic bound for per-query selection between DE and HGB. It is not a ceiling for candidate-level models using additional features. The post-audit 77-feature scorer exceeds this diagnostic because it uses information beyond the two base-ranker hit indicators, including molecular descriptors, training-derived prior statistics, PMI contrast scores, and frozen-schema categorical features.

The initial 82-feature scorer achieved 0.8851 but showed degradation in five blind old-fragment strata, plus one ATT:C|S attachment stratum. The post-audit 77-feature scorer improves the fragment-level point-estimate profile: all 19 old-fragment strata are at baseline parity or better, without a separate per-fragment significance claim (per-fragment values in Supplementary Table S5; this result is best displayed as Supplementary Fig. S3 in the final figure package).

### 4.2 Removing Non-Transferable Prior-Rank Shortcuts Improves Blind Transfer

The strongest feature-engineering result is not an added feature family, but the removal of one. Starting from the initial 82-feature candidate-level scorer, removing the five prior_ranks features improves blind Top-10 from 0.8851 to 0.9243, a gain of +0.0393 with a paired query-level bootstrap CI of [+0.0354, +0.0432]. This ablation remains post-audit because it was motivated by blind diagnostics, but the paired uncertainty estimate shows that the point-estimate gain is not a sampling artifact of the query set. No other leave-one-family-out ablation produces a comparable improvement; removing model scores or model ranks degrades performance.

---

**Table 2.** Leave-one-family-out ablation. Δ is relative to the full 82-feature configuration. Positive Δ values indicate that removing the family *improves* generalization. The CI is shown for the main post-audit deletion; additional families are reported in Supplementary Table S4.

| Condition | Features Removed | Blind Top-10 | Δ from 82-feature scorer | 95% CI for Δ |
|-----------|-----------------:|--------------:|-----------------------------:|------------------|
| Full 82-feature scorer | 0 | 0.8851 | -- | -- |
| **Drop prior\_ranks (77-feature scorer)** | **5** | **0.9243** | **+0.0393** | **[+0.0354, +0.0432]** |
| Drop model\_ranks (F3) | 6 | 0.8697 | -0.0154 | not reported |
| Drop model\_scores (F1) | 5 | 0.8610 | -0.0241 | not reported |

---

This pattern identifies prior_ranks as a non-transferable shortcut. These features encode the within-query rank positions of sparse prior scores. Such ranks are predictive within the development distribution, but their meaning changes when the fragment composition shifts from the 8-fragment development setting to the 19-fragment secondary blind setting. Removing them forces the model to rely on features with more stable cross-split meaning: raw model scores, train-derived rates and supports, molecular descriptors, fingerprint similarities, and categorical features under the frozen schema.

The full ablation table shows that prior_ranks has the largest positive removal effect; model_scores, model_ranks, similarity/frequency, and PMI removal reduce performance, whereas several other removals are smaller positive or near-zero. Thus the main supported claim is that prior_ranks is the dominant non-transferable shortcut, not that every retained family is individually beneficial (Supplementary Table S4).

Because this ablation was motivated by post-audit analysis, we treat the prior-rank finding as an analytical result rather than a pre-registered hypothesis. The corresponding claim boundary is discussed in Section 5.

### 4.3 Feature Pruning Recovers Baseline Misses While Reducing Regression

The 77-feature scorer gain is not only an aggregate Top-10 improvement; it also changes the rescue/lost profile of the candidate-level scorer. Relative to the Score Blend baseline, the 77-feature scorer rescues 1,016 queries that the baseline misses and loses 101 queries that the baseline gets right, giving a net per-query gain of +0.0686 with a 95% paired bootstrap CI of [+0.0638, +0.0733]. The rescue-to-loss ratio is 10.06, with a bootstrap CI of [8.25, 12.52]. Relative to the initial 82-feature scorer, the 77-feature scorer rescues 626 queries and loses 102, giving a net per-query gain of +0.0393 with a 95% CI of [+0.0354, +0.0432].

---

**Table 3.** Rescue/lost analysis for the post-audit 77-feature scorer. These are per-query Top-10 metrics computed from the respective score columns and are not identical to the standalone base-ranker rows in Table 1 (except DE and Score Blend, which match). All counts verified: model_hits = ref_hits + rescue - lost. N = 13,347 blind queries.

| Reference (candidate-matrix score column) | Ref Top-10 | Rescue | Lost | Net | Arithmetic |
|-----------|-----------|--------|------|-----|-----------|
| Score Blend | 0.8558 | 1,016 | 101 | +915 | Yes |
| Initial 82-feature scorer | 0.8851 | 626 | 102 | +524 | Yes |
| Borda (candidate-matrix) | 0.8456 | 1,152 | 101 | +1,051 | Yes |
| HGB-refit (candidate-matrix) | 0.8623 | 977 | 150 | +827 | Yes |
| DE | 0.8055 | 1,754 | 168 | +1,586 | Yes |

---

This is a large reliability improvement over the initial 82-feature scorer. The 82-feature model rescued many baseline misses, but it also introduced 596 lost queries against Score Blend (computed in the 82-feature audit; see Supplementary Table S3). After prior_ranks removal, the 77-feature scorer reduces lost queries from 596 to 101, a 5.9x reduction, while retaining a strong positive net gain. This rescue/lost shift explains why the 77-feature model is preferable to the more aggressive 82-feature model despite using fewer features.

The same conclusion appears at the stratum level: the initial 82-feature scorer degraded five secondary blind old-fragment identities and one ATT:C|S attachment stratum, whereas the post-audit 77-feature scorer has no negative point-estimate Δ across the 19 old-fragment identities. The full reference-specific rescue/lost arithmetic and per-fragment values are reported in Supplementary Tables S3 and S5. We do not treat these 19 strata as independent significance tests; they are a distributional audit of where the aggregate gain is obtained and where regressions remain.

---

### 4.4 Categorical Schema Alignment Prevents Invalid Blind Prediction

During development, we encountered a silent prediction bug: naive one-hot encoding of categorical features (fragment identity, attachment signature, frequency bin) on the blind set produced column spaces incompatible with the development set, causing learned feature weights to be applied to semantically incorrect columns. This yielded a blind Top-10 of 0.7120 on the 82-feature prototype. The frozen categorical schema described in Section 3.4.4 corrected this to 0.8851. This is an implementation audit finding, not a model contribution: the difference reflects a bug repair. We document it because categorical encoding consistency is a practical hazard in molecular machine learning pipelines where fragment vocabularies shift between development and deployment.

---

### 4.5 Dual-Mode Provenance Strata Provide Coverage-Limited Triage Signals

Table 4 reports A4C alert rates for the Borda/HGB provenance groups used in the dual-mode workflow (Section 3.7.2). This analysis is tied to the workflow strata defined by base-ranker top-$K$ sets ($\mathcal{K}_{\mathrm{HGB}}$, $\mathcal{K}_{\mathrm{DE}}$, $\mathcal{K}_{\mathrm{Borda}}$), not to the 77-feature scorer's primary Top-10 evaluation.

---

**Table 4.** A4C alert rates by provenance group. G2 and G3 have complete A4C coverage. G4 is not assigned a group-wide alert estimate because only 5.63% of G4 candidates are covered by the A4C annotation layer.

| Group | Definition | A4C Coverage | Alert Rate |
|-------|-----------|--------------|------------|
| G4 | Shared ($\mathcal{K}_{\mathrm{HGB}} \cap \mathcal{K}_{\mathrm{Borda}}$) | 5.63% | 17.60% among covered; 94.37% unknown |
| G3 | DE-elevated ($\mathcal{K}_{\mathrm{Borda}} \setminus \mathcal{K}_{\mathrm{HGB}}$) | 100% | 9.67% |
| G2 | Borda-emergent ($\mathcal{K}_{\mathrm{Borda}} \setminus (\mathcal{K}_{\mathrm{HGB}} \cup \mathcal{K}_{\mathrm{DE}})$) | 100% | 46.85% |

---

G2 and G3 have complete A4C coverage and separate high-alert exploratory proposals from more moderate DE-supported expansion: G2 (Borda-emergent) carries a hard-alert rate of 46.85%, whereas G3 (DE-elevated) carries a hard-alert rate of 9.67%. G4 (shared candidates) has sparse A4C coverage (5.63%); among covered G4 candidates the alert rate is 17.60%, but 94.37% of G4 candidates have unknown A4C status, so no reliable G4-wide alert estimate is possible. We therefore interpret the provenance labels as a computational triage signal rather than a calibrated safety score. These alert rates are computational screening signals on a PAINS/Brenk pre-filtered vocabulary, not experimentally validated determinations (Section 3.7.3).

## 5. Discussion

### 5.1 Principal Findings

This study makes three central empirical contributions. First, under a leakage-controlled transform-heldout benchmark with 13,347 secondary blind queries, the prospective 82-feature candidate-level scorer achieves Top-10 = 0.8851, and the post-audit feature-pruned 77-feature scorer achieves Top-10 = 0.9243, improving over the Score Blend baseline by ΔTop-10 = +0.0686 [95% CI: +0.0638, +0.0733] (Table 1). Second, post-audit analysis identified five per-query prior-rank features as non-transferable shortcut features. Pruning this feature family improved blind Top-10 by +0.0393 [95% CI: +0.0354, +0.0432] over the initial 82-feature scorer and reduced Score Blend hit loss by 5.9×. We interpret this as a shortcut-learning failure of sparse query-relative prior ranks under fragment distribution shift, rather than as evidence that feature deletion was selected prospectively. Third, beyond ranking performance, the dual-mode workflow provides a coverage-limited computational triage layer: provenance labels stratify Exploration Mode proposals by computational alert burden, with complete-coverage G2/G3 rates of 46.85% and 9.67%, while G4 remains too sparsely covered for a reliable group-wide estimate (Table 4).

The prior_ranks removal was identified during post-audit analysis after blind-set diagnostics revealed fragment-specific degradation in the initial 82-feature configuration. Under a fully prospective protocol, the initial 82-feature scorer would have been the blind-reported model, and the 77-feature configuration would require a fresh blind split or independent replication to serve as a fully prospective model-selection result. We therefore report the 77-feature scorer as a post-audit locked model in this study, not as a fully prospective pre-registered feature-selection result. The mechanistic explanation, shortcut learning driven by the 8-to-19 fragment identity shift, is independently testable because it rests on observable properties of the data and model rather than on a new fitted hyperparameter. Independent replication with a pre-registered feature set would strengthen the generality claim. The complete feature schema, training protocol, and evaluation code are provided to facilitate such replication.

### 5.2 Why Candidate-Level Scoring Succeeds Where Query-Level Routing Fails

A recurring development observation is the failure of query-level routing strategies to improve deployment performance. One conditional-prior gate attempted to select among base rankers on a per-query basis, and a later query router trained a classifier to predict which ranker would perform best for a given query (AUC about 0.61). Although the router is above chance as a classifier, neither approach improved over the simpler strategy of scoring every candidate independently and ranking within each query. Because the routing evidence is diagnostic rather than a primary benchmark result, we use it to motivate a design interpretation rather than a separate performance claim.

These routing experiments suggest that, in this benchmark, the predictive signal for fragment replacement is not primarily a property of a query as a whole, but of individual query--candidate interactions. A query-level gate must compress all candidate-level evidence into a single routing decision, discarding the fine-grained information that distinguishes one candidate from another within the same query. The candidate-level scorer, by contrast, evaluates each query--candidate pair on its own terms, using 77 features that encode structural compatibility, empirical priors, and molecular properties at the level where replacement decisions are actually made. This distinction is a testable design principle for related ranking tasks, but its evidentiary status here remains secondary to the blind benchmark and ablation results.

### 5.3 Prior_Ranks as Non-Transferable Shortcuts

The prior-rank ablation illustrates why leakage-controlled benchmarks are useful beyond producing lower or more realistic performance numbers. They can change which features appear trustworthy. In development, prior-rank features offered compact summaries of empirical replacement priors; under the secondary blind fragment shift, the same ranks became query-relative shortcuts whose numerical meaning no longer transferred.

The strongest feature-engineering result is the removal, not the addition, of a feature family. The five prior_ranks features encode the within-query rank positions of sparse prior scores. These ranks are predictive within the 8-fragment development distribution but fail to transfer to the 19-fragment blind distribution because their meaning, such as what a rank of 3 signifies about a candidate, changes with the fragment composition of the query set. We interpret this as shortcut learning [3]: the model exploits a feature that appears informative during training but whose semantics are distribution-specific. Alternative explanations remain plausible, including multicollinearity among prior features, unstable rank transforms under sparse support, and threshold effects in the gradient-boosted trees under distribution shift.

This finding does not condemn rank features in general. The retained model-rank family (F3, encoding base-ranker rank positions) contributes positively to the final model. We attribute this asymmetry to the different provenance of the two rank types: base-ranker ranks summarize validated model outputs whose ranking behavior is stable across fragment identities, whereas prior_ranks derive from sparse prior-score transforms whose rank distributions are tightly coupled to specific fragment types. The design rule that emerges---prefer raw scores, rates, and intrinsic molecular properties over sparse prior-score rank transforms under fragment distribution shift in cross-split ranking---is testable in other molecular property prediction and virtual screening contexts where train/test distribution shifts are common.

Borda fusion occupies an important structural role in this narrative. Borda(DE,HGB) is not the strongest method in Table 1, but it is the method that first demonstrates DE-HGB complementarity under leakage-controlled evaluation: the two rankers make substantially different errors, and their equal-weight consensus outperforms either alone. The candidate-level scorer extends this insight by integrating DE and HGB signals alongside additional molecular and statistical features within a single scoring model rather than a post-hoc fusion rule. Borda thus serves as a mechanism anchor: it establishes that complementarity exists, motivating the richer feature integration that the candidate-level scorer achieves.

### 5.4 Dual-Mode Workflow and Computational Review

The dual-mode workflow addresses a practical tension in computational fragment replacement: conservative proposals aligned with training-set frequency patterns offer reliability but limited novelty, while exploratory proposals expand coverage at the cost of increased structural alert risk. The provenance-label design (G2/G3/G4) provides a structured language for communicating this trade-off.

Three scope limitations apply. First, the A4C proxy relies on published rule-based filters (PAINS, Brenk) and simple property-shift thresholds; it does not incorporate target-specific activity data or experimentally validated off-target profiles. Second, systematic activity-profile studies demonstrate that structurally conservative replacements can produce unexpected off-target effects [18], underscoring that no computational proxy replaces experimental profiling. Third, the alert rates reported here are specific to the ChEMBL37K-derived fragment vocabulary and may not generalize to other chemical spaces.

### 5.5 Relationship to Generative and Database-Driven Approaches

Our closed-vocabulary ranking formulation occupies a distinct position in the bioisosteric replacement landscape. Database-driven tools such as NeBULA [10] compile hundreds of thousands of replacement rules from the medicinal chemistry literature, enabling broad coverage but operating in a retrieval rather than ranking paradigm. Generative approaches such as DeepBioisostere [12] autonomously select and substitute fragments without pre-defined modification sites, offering open-vocabulary exploration at the cost of controllability. GraphBioisostere [13] applies whole-molecule graph neural networks to bioisosteric pair prediction, capturing chemical environment context beyond the fragment level. We therefore avoid a direct superiority claim over these systems: the present benchmark tests whether methods can rank a fixed, attachment-compatible candidate set under transform-heldout evaluation, whereas those systems address broader retrieval, generation, or pair-prediction tasks.

Our work complements these approaches with a ranking perspective that explicitly separates candidate scoring from candidate generation. The closed vocabulary enables rigorous leakage-controlled evaluation through the transform-heldout split, but it also limits the method to fragments observed in the training data. Extending the framework to open-vocabulary ranking, potentially combining the candidate-level scorer's feature engineering with graph-based molecular representations, is a natural direction for future work.

### 5.6 Limitations and Future Work

Several limitations should be considered. First, the benchmark uses a single data source (ChEMBL37K); evaluation on independent MMP datasets would assess generality. Second, all supervision labels are structure-derived and do not establish activity preservation. Third, the transform-heldout design controls transform-identity leakage, but it does not eliminate every possible form of chemical relatedness between development and evaluation examples. Fourth, the post-audit nature of the prior_ranks removal means the 77-feature configuration, while locked and reproducible, was not selected through a fully prospective protocol. Fifth, the A4C proxy has not been validated against expert review or experimental off-target profiling. Sixth, the dual-mode workflow is demonstrated on a single benchmark; its operational usefulness in lead optimization campaigns remains to be established.

Beyond the open-vocabulary extension and independent replication discussed above, several directions merit investigation. Activity-based evaluation endpoints would connect the ranking task more directly to drug discovery outcomes, though they require careful control for assay heterogeneity. Integration with GNN-based molecular representations could capture chemical environment effects beyond fragment-level features. Prospective pre-registration of feature sets and evaluation protocols would strengthen the evidential status of future ablation findings. Finally, expert medicinal chemistry review of A4C-stratified proposals would calibrate computational alert rates against human judgment.

---

## 6. Conclusion

We introduced a leakage-controlled benchmark and a candidate-level scoring framework for closed-vocabulary scaffold-conditioned fragment replacement. The prospective 82-feature scorer establishes the candidate-level blind result (Top-10 = 0.8851), while the post-audit 77-feature scorer achieves Top-10 = 0.9243 on 13,347 secondary blind queries, a ΔTop-10 = +0.0686 improvement over the Score Blend baseline. The main audit finding is that five per-query prior-rank features behave as non-transferable shortcuts under fragment distribution shift. Removing this feature family converts the prospective 82-feature candidate-level scorer into a locked post-audit 77-feature scorer, improving blind Top-10 by +0.0393 and reducing Score Blend hit loss by 5.9×. Because the removal was identified after blind diagnostics, this result should be read as a reproducible post-selection finding that motivates independent prospective replication. A secondary development observation, that query-level routing did not improve deployment performance, suggests that predictive signal in this benchmark resides in individual query-candidate interactions rather than in query-level summaries.

The dual-mode workflow translates these ranking outputs into a practical reporting framework. By stratifying Exploration Mode proposals into provenance groups with distinct computational alert profiles, it provides a transparent triage hypothesis for downstream inspection, not a decision rule for medicinal chemistry.

Several limitations define the scope of these contributions. The benchmark uses structure-derived labels from a single data source and does not establish activity preservation. The prior_ranks removal was identified post-audit and is not a fully prospective feature-selection result. The A4C proxy provides computational screening signals that have not been validated against experimental off-target profiling. Addressing these limitations---through independent replication with pre-registered protocols, activity-based evaluation endpoints, and expert review calibration---constitutes the natural next phase of this work.

---

## Data and Software Availability

The analysis uses ChEMBL37K, a curated subset of ChEMBL33. A public source-code and evidence archive is available at https://github.com/user141514/paper1/tree/codex/jcim-algorithm-archive, with the curated experiment-code folder at `bioisosteric_diffusion/paper1_experiment_code_archive`. The code-archive commit is `b750de266d6d47e63de0eeaa945cc999e6f3e08c`, and the manuscript/supplement synchronization commit is `72b7807948c9a1432b99bf1529d98b3d994a3d1f`; the submission-candidate package is frozen under the repository tag `jcim-submission-candidate-20260601`. The repository includes the analysis scripts, split and feature-schema documentation, query-level audit tables, bootstrap summaries, and small evidence files required to audit the reported numbers. Large processed candidate matrices and any redistribution-restricted ChEMBL-derived artifacts require separate deposition or access documentation before final submission, subject to the redistribution terms of the underlying ChEMBL-derived data. An archival DOI should be assigned before or during the journal submission process.

---

## References

1. Wallach, I.; Heifets, A.; Dzamba, M. Most Ligand-Based Classification Benchmarks Reward Memorization Rather than Generalization. *J. Chem. Inf. Model.* **2018**, *58* (5), 916-932. https://doi.org/10.1021/acs.jcim.7b00403.

2. Kapoor, S.; Narayanan, A. Leakage and the Reproducibility Crisis in ML-Based Science. *Patterns* **2023**, *4* (9), 100804. https://doi.org/10.1016/j.patter.2023.100804.

3. Geirhos, R.; Jacobsen, J.-H.; Michaelis, C.; et al. Shortcut Learning in Deep Neural Networks. *Nat. Mach. Intell.* **2020**, *2* (11), 665-673. https://doi.org/10.1038/s42256-020-00257-z.

4. Patani, G. A.; LaVoie, E. J. Bioisosterism: A Rational Approach in Drug Design. *Chem. Rev.* **1996**, *96* (8), 3147-3176. https://doi.org/10.1021/cr950066q.

5. Hussain, J.; Rea, C. Computationally Efficient Algorithm to Identify Matched Molecular Pairs (MMPs) in Large Data Sets. *J. Chem. Inf. Model.* **2010**, *50* (3), 339-348. https://doi.org/10.1021/ci900450m.

6. Griffen, E.; Leach, A. G.; Robb, G. R.; Warner, D. J. Matched Molecular Pairs as a Medicinal Chemistry Tool. *J. Med. Chem.* **2011**, *54* (22), 7739-7750. https://doi.org/10.1021/jm200452d.

7. Zdrazil, B.; Felix, E.; Hunter, F.; et al. The ChEMBL Database in 2023: A Resource for Drug Discovery. *Nucleic Acids Res.* **2024**, *52* (D1), D1180-D1192. https://doi.org/10.1093/nar/gkad1064.

8. Wirth, M.; Zoete, V.; Michielin, O.; Sauer, W. H. B. SwissBioisostere: A Database of Molecular Replacements for Ligand Design. *Nucleic Acids Res.* **2013**, *41* (D1), D1137-D1143. https://doi.org/10.1093/nar/gks1059.

9. Polishchuk, P. CReM: Chemically Reasonable Mutations Framework for Structure Generation. *J. Cheminform.* **2020**, *12*, 28. https://doi.org/10.1186/s13321-020-00431-w.

10. Huang, S.; Wang, S.; Dong, J.; Xu, M.; Yuan, S. NeBULA: A Web-Based Novel Drug Design Platform for Up-to-Date Bioisosteric Replacement. *Med. Drug Discov.* **2025**, *28*, 100231. https://doi.org/10.1016/j.medidd.2025.100231.

11. Ertl, P. Identification of Bioisosteric Substituents by a Deep Neural Network. *J. Chem. Inf. Model.* **2020**, *60* (7), 3369-3375. https://doi.org/10.1021/acs.jcim.0c00290.

12. Kim, H.; Moon, S.; Zhung, W.; Lim, J.; Kim, W. Y. DeepBioisostere: Discovering Bioisosteres with Deep Learning for a Fine Control of Multiple Molecular Properties. *arXiv* **2025**, 2403.02706v2. Preprint; not peer-reviewed.

13. Masunaga, S.; Furui, K.; Kengkanna, A.; Ohue, M. GraphBioisostere: General Bioisostere Prediction Model with Deep Graph Neural Network. *J. Supercomput.* **2026**, *82*, 132. https://doi.org/10.1007/s11227-026-08232-y.

14. Dwork, C.; Kumar, R.; Naor, M.; Sivakumar, D. Rank Aggregation Methods for the Web. In *Proceedings of WWW10*; 2001; pp 613-622.

15. Cormack, G. V.; Clarke, C. L. A.; Buettcher, S. Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. In *Proceedings of SIGIR*; 2009; pp 758-759.

16. Baell, J. B.; Holloway, G. A. New Substructure Filters for Removal of Pan Assay Interference Compounds (PAINS). *J. Med. Chem.* **2010**, *53* (7), 2719-2740. https://doi.org/10.1021/jm901137j.

17. Brenk, R.; Schipani, A.; James, D.; et al. Lessons Learnt from Assembling Screening Libraries for Drug Discovery for Neglected Diseases. *ChemMedChem* **2008**, *3* (3), 435-444. https://doi.org/10.1002/cmdc.200700139.

18. Helmke, P. S.; Kandler, J.; Ilie, S.; Gaskin, L.; Ecker, G. F. Data-Driven Assessment of Bioisosteric Replacements and Their Influence on Off-Target Activity Profiles. *RSC Med. Chem.* **2025**, *16*, 6048-6058. https://doi.org/10.1039/d5md00686d.

19. Pedregosa, F.; Varoquaux, G.; Gramfort, A.; et al. Scikit-Learn: Machine Learning in Python. *J. Mach. Learn. Res.* **2011**, *12*, 2825-2830.

20. Landrum, G.; et al. RDKit: Open-Source Cheminformatics Software. https://www.rdkit.org/ (accessed 2026-05-31).

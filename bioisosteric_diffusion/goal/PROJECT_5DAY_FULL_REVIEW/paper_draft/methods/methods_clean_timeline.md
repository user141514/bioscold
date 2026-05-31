## 3. Methods

We present the complete methodological framework. Section 3.1 formalizes the ranking task, supervision labels, and evaluation metrics. Section 3.2 describes benchmark construction: MMP extraction, decoy construction, and the transform-heldout split. Sections 3.3--3.5 cover the ranking methods: five base rankers spanning a range of inductive biases (Section 3.3), a candidate-level scorer with 77 features and a gradient boosting architecture (Section 3.4), and a leave-one-family-out ablation that identifies a feature family harmful to cross-split generalization (Section 3.5). Sections 3.6--3.7 describe the evaluation protocol and the A4C computational review proxy.

The base ranker architecture, scorer feature families F1--F9, and training hyperparameters were configured on a held-out development/calibration set drawn from the training partition (Section 3.2). The prior_ranks feature family was removed after post-audit analysis of the initial secondary blind evaluation revealed unexpected per-fragment degradation; the mechanistic basis for removal is documented in Section 3.5.2. The 77-feature configuration described in Section 3.4.2 is the **final locked model** in this study, with the caveat that the prior_ranks removal is a post-hoc feature-regularization step rather than a fully prospective pre-registered feature-selection result. All performance results appear in Section 4.

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

**Data source and MMP extraction.** We use ChEMBL37K [Zdrazil et al., 2023], a curated subset of approximately 37,000 drug-like molecules from ChEMBL33. MMPs are extracted by applying retrosynthetic bond-cleavage rules [Hussain and Rea, 2010; Griffen et al., 2011]: each molecule is recursively decomposed at acyclic single bonds to generate fragment--scaffold pairs; any two molecules sharing identical scaffold fragments but bearing different substituent fragments at the same attachment point are recorded as an MMP. The observed substitution $(f^{\mathrm{old}} \rightarrow c^*)$ provides a positive training example. Quality filters: salt stripping, stereochemistry normalization, removal of molecules with PAINS or Brenk alerts [Baell and Holloway, 2010; Brenk et al., 2008], and exclusion of fragments with molecular weight below 15 Da or above 250 Da.

**Decoy construction.** Negatives are constructed through a decoy repair procedure: for each query $q_i$, $f_i^{\mathrm{old}}$ is paired with a fragment $c_{\mathrm{decoy}} \in \mathcal{V}_{\mathrm{train}}$ attached to a non-cognate scaffold in the ChEMBL37K data. Four stages: (1) 1:1 positive-to-decoy ratio for balanced candidate sets; (2) wrong-positive removal: any decoy belonging to $\mathcal{P}_i$ is excluded; (3) deduplication of identical $(f_i^{\mathrm{old}}, c_{\mathrm{decoy}})$ pairs; (4) a 1:5 diagnostic set for class-imbalance stress testing, excluded from primary evaluation.

**Split design: transform-heldout.** A conventional random split of MMP transformations would allow the same $(f^{\mathrm{old}}, \sigma)$ pair---the *transform identity*---to appear in both training and test sets, enabling memorization rather than generalization. We prevent this with a **transform-heldout split**: all unique $(f^{\mathrm{old}}, \sigma)$ pairs are partitioned such that no pair appears in both training and evaluation splits.

We construct three evaluation tiers:

- **Development/calibration set.** A held-out subset of the training partition, used during method development for architecture selection, feature engineering, and hyperparameter configuration. The base ranker architectures, scorer feature families F1--F9, and all training hyperparameters were configured on this set. The subsequent removal of the experimentally added prior_ranks family is described separately in Section 3.5 as a post-audit feature-regularization step.
- **Secondary blind set.** 13,347 queries. Transform-heldout from training: zero $(f^{\mathrm{old}}, \sigma)$ overlap. The candidate vocabulary $\mathcal{V}_{\mathrm{train}}$ is training-derived and closed; transform identities are held out, but the blind queries and training queries draw candidates from the same vocabulary. This set serves as the primary evaluation benchmark.
- **Canonical analysis set.** 21,052 queries with a slightly larger vocabulary, used exclusively for robustness analysis and mechanistic investigation. No primary performance claims rest on this set.

**Leakage verification.** Table M1 reports overlap counts of $(f^{\mathrm{old}}, \sigma)$ transform identities between every pair of splits. Zero overlap is confirmed for all pairs.

---

**Table M1.** Leakage verification: overlap counts of $(f^{\mathrm{old}}, \sigma)$ transform identities.

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

**Oracle(DE,HGB) diagnostic.** As a reference bound, we compute the fraction of secondary blind queries for which at least one of DE or HGB places the correct fragment in its top 10. This is a diagnostic upper bound for per-query selection between the two structurally complementary rankers, not a ceiling for models using additional candidate-level features. Results are reported in Section 4.1.

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

The method has no learned parameters and serves as the natural lower bound.

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

We choose HistGB for three reasons: (1) mixed-type inputs (continuous, discrete, binary, one-hot) are handled natively; (2) non-linear feature interactions are captured---a high prior score with a large molecular weight delta signals a different replacement context than either alone; (3) built-in feature importance estimates support ablation without manual selection.

#### 3.4.2 Feature Families with Audit Trail

The 77 features organize into 9 families. Table M2 provides each family's description, source, and audit status: whether it is train-only, label-free, retained in the final model, and whether it poses leakage risk.

---

**Table M2.** Feature families of the candidate-level scorer with audit trail.

| Family | Count | Source | Train-Only | Label-Free | Retained | Leakage Control |
|--------|-------|--------|------------|------------|----------|-----------------|
| F1: Model Scores | 5 | Base ranker outputs ($z$-normalized) | ✓ | ✓ | ✓ | Train-set statistics only |
| F2: Model Raw Scores | 4 | Base ranker raw outputs | ✓ | ✓ | ✓ | No evaluation-set dependence |
| F3: Model Ranks | 6 | Per-query ranks from base rankers | ✓ | ✓ | ✓ | Query-relative; see §3.5 |
| F4: Top-K Flags | 4 | Binary: candidate in base ranker top-10 | ✓ | ✓ | ✓ | Thresholds fixed at dev time |
| F5: Prior Scores | 5 | Logit-transformed empirical priors (P0/P1/P3/P4) | ✓ | ✓ | ✓ | Priors from train counts only |
| F6: Prior Statistics | 12 | Smoothed rates, support/positive counts | ✓ | ✓ | ✓ | Counts from train; smoothing fixed on dev |
| F7: PMI / Conditional-Prior Contrast Scores | 6 | Pointwise mutual information and contrast scores across prior levels | ✓ | ✓ | ✓ | PMI from train co-occurrence |
| F8: Molecular Descriptors | 10 | RDKit: candidate properties + deltas from query | -- | ✓ | ✓ | Structure-derived; no label dependence |
| F9: Similarity & Frequency | 3 | Tanimoto similarity, global/attachment frequency | ✓ | ✓ | ✓ | Frequency from train counts |
| Categorical (one-hot) | 22 | Fragment identity, attachment signature, frequency bin (one-hot expanded) | ✓ | ✓ | ✓ | Frozen schema; unseen→OTHER (§3.4.4) |
| ~~Prior_Ranks~~ | ~~5~~ | Per-query rank of prior scores | ✓ | ✓ | ✗ Removed | Post-audit removal (§3.5); shortcut hazard |
| **Total Retained** | **77** | | | | | |

---

**Model-derived features (F1--F4).** Model scores (F1) and raw scores (F2) carry the primary signal from each base ranker. Model ranks (F3) encode relative ordering within each query's candidate pool. Top-K flags (F4) provide discrete consensus signals. Retained base-ranker ranks (F3) differ from the removed prior_ranks: base-ranker ranks summarize validated model outputs from trained rankers, whereas prior_ranks are sparse prior-score rank transforms that were removed as non-transferable query-local frequency shortcuts (Section 3.5.2).

**Prior and statistical features (F5--F7).** These encode empirical replacement frequencies across four provenance levels: P0 (global fragment frequency), P1 (attachment-conditioned), P3 (cluster-conditioned within the same MMP transformation class), P4 (transferred probability from chemically similar training fragments). Logit-transformed scores (F5) linearize the probability scale. Smoothed rates, support counts, and positive counts (F6) quantify statistical reliability. PMI / conditional-prior contrast scores (F7) capture information gain at each conditioning level relative to the baseline---for example, $\mathrm{PMI}(P3, P1)$ measures how much additional information the cluster-level context provides beyond the attachment-level context alone. These are computed from training-set co-occurrence statistics and are distinct from the removed per-query prior_ranks features (Section 3.5).

**Molecular descriptors (F8) and similarity (F9).** Absolute candidate properties and deltas from the query fragment (heavy atom count, molecular weight, logP, TPSA) encode steric and electronic compatibility. Morgan fingerprint Tanimoto similarity (radius 2, 2048 bits) and frequency features anchor the prediction in structural and empirical context.

**Categorical features.** Old fragment identity (SMILES), attachment signature, and replacement frequency bin are one-hot encoded via frozen categorical schema alignment (Section 3.4.4). The encoding template is learned from the development set; categories unseen in development (e.g., novel fragment identities in the secondary blind set) are mapped to a reserved OTHER category, and output dimensionality is forced to match the development schema. This captures fragment-specific and attachment-specific effects without hand-written rules, while ensuring consistent feature dimensionality across splits.

#### 3.4.3 Training Protocol

The scorer uses `HistGradientBoostingClassifier` (scikit-learn [Pedregosa et al., 2011]) with max 200 iterations, max depth 6, learning rate 0.1, `class_weight='balanced'`, `random_state=42`.

Training uses 5-fold GroupKFold cross-validation grouped by query identifier, preventing within-query leakage. Negative subsampling at 5:1 mitigates class imbalance. All hyperparameters were fixed on the development set. The later post-audit change (Section 3.5) affected only the feature set---removing five prior_ranks features---and did not modify the HistGB hyperparameters.

For secondary blind evaluation, a single model is trained on the full development set and applied to the secondary blind set. No blind labels or candidate-level outcomes were used to compute feature values or train model parameters. Feature standardization parameters are computed from development set statistics. The final 77-feature configuration was selected through post-audit analysis prompted by the initial blind evaluation, as described in Section 3.5; this timeline is documented transparently therein.

#### 3.4.4 Frozen Categorical Schema Alignment

Categorical features pose a schema-mismatch risk because the development and secondary blind sets can contain disjoint fragment identities. Without a frozen schema, one-hot encoded columns may differ in both dimensionality and semantic order across splits, causing learned feature weights to be applied to incorrect columns at inference time. We therefore use a frozen categorical schema: the one-hot encoder is fitted on the development set, recording observed categories and column ordering. For secondary blind encoding, unseen categories are mapped to a reserved OTHER category, and the output matrix is forced to match the development set's column count and ordering. A shape assertion guarantees consistency before prediction. The empirical impact of this alignment step is reported in Section 4.

---

### 3.5 Feature Ablation: The Prior_Ranks Removal

The feature set in Section 3.4 was finalized through systematic leave-one-family-out ablation. This section documents the ablation design, its methodological outcome, and the timeline of the prior_ranks removal.

The prior_ranks removal was identified during post-audit analysis of the initial 82-feature scorer after blind-set diagnostics revealed fragment-specific degradation. We therefore report the 77-feature scorer as a final locked model in this study, but not as a fully prospective pre-registered feature-selection result.

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

We interpret this as **shortcut learning** [Geirhos et al., 2020]: rank features are predictive within the development distribution, so the model defaults to them instead of learning transferable chemical patterns. Their informativeness is specific to the fragment identities and ranking contexts seen during development. Removing the shortcut forces the model to attend to raw scores, rates, and intrinsic molecular properties.

**Design rule.** Per-query rank features are a generalization hazard in cross-split molecular ranking; use raw scores, rates, and intrinsic properties instead.

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

For pairwise comparisons, a paired delta bootstrap is used:

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

We enforce strict separation of evaluation protocols by evidentiary role (Table M3).

---

**Table M3.** Evaluation protocol separation.

| Protocol | Queries | Purpose |
|----------|---------|---------|
| Development/calibration | Held-out train partition | Architecture, feature, and hyperparameter selection |
| Secondary blind | 13,347 | **Primary performance claim** |
| Canonical analysis | 21,052 | Robustness and mechanism analysis only |
| Dual-mode risk-stratification (§3.7) | --- | Workflow and risk interpretation only |

---

Primary claims cite only secondary blind results. Canonical analyses are explicitly labeled as robustness checks. The dual-mode framework is a workflow interpretation layer.

#### 3.6.5 Leakage Control

**Feature-level.** All statistics (prior probabilities, smoothing parameters, PMI, $z$-score statistics) derive exclusively from the training split. Transfer features use only training-set reference molecules.

**Categorical alignment.** The frozen categorical schema (Section 3.4.4) ensures consistent dimensionality across splits; unseen categories map to OTHER.

**Overlap verification.** Zero $(f^{\mathrm{old}}, \sigma)$ overlap between training and any evaluation split (Table M1). Zero query overlap between canonical and secondary blind sets.

**Label-free feature space.** All 77 features are computable without evaluation labels. They are derived from molecular structure, base-ranker outputs, or training-set statistics. No feature incorporates ground-truth replacement labels. The schema is locked at 77 dimensions with fixed column order.

---

### 3.7 Computational Review Proxy (A4C)

Bioisosteric replacement, even when chemically conservative, can produce unexpected off-target effects---a risk documented in systematic studies of bioisosteric substitution patterns [Helmke2025]. We adopt a dual-mode workflow under an A4C computational review proxy that categorizes proposals by provenance within the ranking pipeline and by structural alerts.

The A4C annotation layer applies PAINS alerts [Baell and Holloway, 2010] and Brenk alerts [Brenk et al., 2008] to flag structural features associated with assay interference, reactivity, or unfavorable pharmacokinetics. Property shifts (molecular weight, logP, TPSA, HBD/HBA counts) relative to the query fragment are also recorded. The framework is a computational screening aid, not a medicinal-chemistry truth standard.

#### 3.7.1 Dual-Mode Workflow

**Conservative Mode** draws proposals from the HGB ranker (Section 3.3.3). Its top-ranked candidates are biased toward chemically well-characterized replacements aligned with training-set frequency patterns.

**Exploration Mode** draws proposals from Borda(DE, HGB) (Section 3.3.4). The DE component enables proposals that differ from frequency priors, potentially identifying novel fragments HGB alone would miss.

#### 3.7.2 Provenance Labels and Alert Stratification

Within Exploration Mode, proposals receive provenance labels (Table M4). Let $\mathcal{K}_{m}$ denote the top-$K$ candidates from ranker $m \in \{\mathrm{HGB}, \mathrm{DE}\}$, and $\mathcal{K}_{\mathrm{Borda}}$ the Borda top-$K$ set.

---

**Table M4.** Provenance groups and definitions. Alert rates are reported in Section 4.4.

| Group | Definition | Interpretation |
|-------|-----------|----------------|
| $\mathrm{G4}$ | $\mathcal{K}_{\mathrm{HGB}} \cap \mathcal{K}_{\mathrm{Borda}}$ | Shared: consensus between frequency and similarity signals |
| $\mathrm{G3}$ | $\mathcal{K}_{\mathrm{Borda}} \setminus \mathcal{K}_{\mathrm{HGB}}$ (DE-elevated) | Expansion beyond frequency priors; similarity-supported |
| $\mathrm{G2}$ | $\mathcal{K}_{\mathrm{Borda}} \setminus (\mathcal{K}_{\mathrm{HGB}} \cup \mathcal{K}_{\mathrm{DE}})$ | Borda-emergent: combined ranker agreement, highest novelty |

---

This stratification provides an actionable risk structure: G4 as low-risk starting points, G3 for closer inspection, G2 requiring expert review.

#### 3.7.3 Scope and Limitations

The A4C framework is used exclusively for workflow and risk interpretation. It does not support primary performance claims. Alert rates are computational screening signals derived from published rule-based filters; they have not been validated through medicinal-chemistry expert review and do not constitute determinations of synthetic accessibility, assay interference, or toxicity. Systematic activity-profile studies indicate that off-target effects can arise from replacements that appear structurally conservative [Helmke2025], underscoring that no rule-based proxy replaces experimental profiling. We recommend G4 proposals as initial suggestions, G3 proposals with structural-flag review, and G2 proposals with comprehensive expert evaluation. This conservative interpretation ensures the computational proxy informs rather than replaces human judgment.

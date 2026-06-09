# Leakage-Controlled Benchmarking of Closed-Vocabulary Fragment Replacement Ranking

## Abstract

Matched molecular pair (MMP)-derived fragment replacement ranking provides a scalable structure-derived benchmark for observed substitution patterns, but permissive splits can repeat fragment-attachment contexts across training and evaluation. We study closed-vocabulary scaffold-conditioned replacement ranking under query-side transform-heldout evaluation, where the same old-fragment/attachment-context key is not shared across train and blind partitions. This is a small closed-vocabulary recovery benchmark: candidates are drawn from the training-derived replacement vocabulary rather than from an open or chemically enumerated replacement universe.

The primary prospective evidence comes from an internal prospectively constructed replication split by the same team, which we refer to as Fresh Blind2. On this split, the 82-feature HistGB candidate-level scorer achieved query-level Top-10 = 0.8480 with wider group-level Top-10 uncertainty, improving over Score Blend (Top-10 = 0.8077) and Borda(DE,HGB) (Top-10 = 0.8176). The no-prior-rank 77-feature configuration remained above Score Blend but did not reproduce its earlier Top-10 advantage over the 82-feature scorer (77 - 82 Top-10 = -0.0070, 95% CI [-0.0094, -0.0044]), indicating that prior-rank deletion is better interpreted as a post-audit instability diagnostic than as a replicated model-selection improvement.

The methodological lesson is conservative: candidate-level scoring is observed as a query-level recovery gain on Fresh Blind2, whereas prior-rank deletion does not replicate as a Top-10 improvement. Although Top-10 group-level generalizability is not established by the current evidence, HistGB82 MRR improvements over Score Blend remain positive under grouped uncertainty analyses, providing more stable support for candidate-level scoring. The benchmark evaluates structure-derived replacement recovery only; it does not support activity-outcome inference, biological-outcome claims, external temporal transfer, or absolute score-to-outcome claims.

---

## 1. Introduction

Replacing a selected substituent while preserving scaffold connectivity is a routine decision in lead optimization, yet computational support for this step remains difficult to benchmark rigorously. We study closed-vocabulary scaffold-conditioned fragment replacement ranking: given an old fragment and its attachment signature, a model ranks candidate replacements from a closed training-derived vocabulary. The task is deliberately structure-derived---it asks whether observed replacement patterns can be recovered under controlled evaluation, not whether a proposed replacement preserves activity.

Structure-derived replacement pairs from ChEMBL [1], extracted through matched molecular pair (MMP) analysis [2,3], provide scalable supervision for this setting. Building a credible benchmark remains difficult for three reasons. First, random data splits can leak the same fragment-attachment transform into both training and evaluation sets, allowing models to succeed through memorization rather than generalization. Second, simple attachment-frequency baselines are already strong, and learned rankers can overfit dataset-specific regularities unless tested under explicit leakage control. Third, structural and empirical ranking signals are complementary in principle, but combining them effectively without overfitting to development-set transform patterns is non-trivial.

We separate three evidentiary roles. First, the benchmark contribution is a query-side transform-heldout evaluation design that removes exact old-fragment/attachment-context overlap between train and blind partitions. Second, the prospective scoring contribution is a fresh Blind2 evaluation in which the 82-feature HistGB candidate-level scorer is the primary Top-10 HistGB result. Third, the prior-rank analysis is a diagnostic: the older no-prior-rank gain observed after secondary-blind diagnostics does not replicate as a prospective Top-10 improvement on Blind2.

On Fresh Blind2, the 82-feature HistGB scorer achieves Top-10 = 0.8480, exceeding Score Blend (0.8077) and Borda(DE,HGB) (0.8176). The no-prior-rank 77-feature configuration remains baseline-beating (Top-10 = 0.8411) and slightly improves MRR, but its Top-10 is lower than the 82-feature scorer (Delta = -0.0070, 95% CI [-0.0094, -0.0044]). These are query-level comparisons; group-level Top-10 generalizability is bounded in Section 4.3. We therefore treat prior_ranks as an unstable query-relative feature family rather than as a cleanly transferable deletion target.

The resulting contribution hierarchy is benchmark first, prospective candidate-level scoring second, and diagnostic signal-transfer analysis third. Secondary workflow, activity-comparable, calibration, and architecture diagnostics are used only to define boundaries and future directions.

---

## 2. Related Work

**Matched molecular pairs and structure-derived replacements.** Bioisosteric replacement is a longstanding design concept in medicinal chemistry [4]. Matched molecular pair analysis extracts local substitution patterns from large compound collections by identifying molecules that differ at a single structural site while sharing an identical scaffold [2,3]. Databases such as ChEMBL [1] make structure-derived replacement catalogues feasible at scale. These data record observed structural substitutions and serve as supervision for replacement proposal, but they do not encode activity continuity---a limitation inherited by any model trained on them.

**Rule-based and database-driven tools.** SwissBioisostere organizes observed replacements into a searchable database for ligand design [5], while CReM uses fragment environments from known compounds to perform context-aware replacement generation [6]. NeBULA extends this paradigm by mining over 700 medicinal chemistry references to compile 311,543 qualitative bioisosteric replacement reaction rules encoded as SMARTS and SMIRKS patterns, covering five replacement categories from scaffold hopping to general isosteric substitution [7]. These frameworks exploit empirical replacement context at scale, but they define different task formulations---typically open-ended generation or database lookup---rather than the closed-vocabulary query-level ranking benchmark studied here.

**Learning-based methods.** Deep learning methods for bioisosteric replacement include descriptor-based neural networks [8] and deep generative models that autonomously select fragments for removal and insertion while controlling multiple properties such as logP, drug-likeness, and synthetic accessibility simultaneously [9]. GraphBioisostere applies a Siamese graph neural network with attention-based global pooling (MMGX encoder) to ChEMBL-derived molecular pairs, training on 779,081 compound pairs across 1,897 targets, and demonstrates transfer learning for target-specific potency change prediction across five pharmaceutically relevant targets [10]. These methods show that structural representations capture replacement-relevant information beyond frequency statistics, but they are typically evaluated on random splits that may overestimate generalization. Our work complements the generative formulation with a ranking perspective that explicitly separates candidate scoring from candidate generation.

**Rank aggregation and fusion.** Borda count is a classical parameter-free aggregation rule for combining ranked lists [11]. Reciprocal rank fusion [12] provides a weighted alternative. Our use of fusion is narrower: we test whether structural and empirical rankers remain complementary under leakage-controlled evaluation and use Borda's parameter-free property as a practical necessity---the transform-heldout split discourages development-set-based tuning of fusion weights.

**Learning-to-rank context.** Pairwise and listwise learning-to-rank methods are natural alternatives to pointwise candidate scoring, but we treat them as future pre-specified architecture work rather than current main evidence.

**Computational alert and risk filters.** Because structure-derived replacement recovery does not imply activity continuity, PAINS/Brenk filters [13,14] and activity-profile studies [15] motivate cautious downstream annotation; in this work, such annotations are reported only as supplementary diagnostics.

---

## 3. Methods

We present the complete methodological framework. Section 3.1 formalizes the ranking task, supervision labels, and evaluation metrics. Section 3.2 describes benchmark construction: MMP extraction, candidate-matrix construction, the query-side transform-heldout split, and the Fresh Blind2 replication policy. Sections 3.3--3.5 cover the ranking methods: five base rankers spanning a range of inductive biases (Section 3.3), a candidate-level HistGB scorer (Section 3.4), and the prior-rank ablation that is now interpreted as an instability diagnostic rather than as a replicated improvement (Section 3.5). Sections 3.6--3.7 describe the evaluation protocol and supplementary workflow diagnostics.

The base ranker architecture, scorer feature families F1--F9, and training hyperparameters were configured on development/calibration data. The original secondary-blind 77-feature result was identified after blind diagnostics and is therefore post-audit. Fresh Blind2 is the primary prospective replication: Train2 supplies vocabulary, priors, and base-ranker artifacts; Dev2/calibration fits the HistGB candidate-level scorers; Blind2 is evaluated once. In that protocol, the 82-feature HistGB scorer is the primary prospective Top-10 HistGB configuration, while the 77-feature no-prior-rank configuration is a pre-specified replication test and diagnostic comparator. All performance results appear in Section 4.

---

### 3.1 Task Formulation

We formalize scaffold-conditioned fragment replacement as a closed-vocabulary, multi-positive learning-to-rank problem. The candidate vocabulary $\mathcal{V}_{\mathrm{train}}$ is training-derived and closed: it is constructed exclusively from the training split. Transform identities $(f^{\mathrm{old}}, \sigma)$ are held out across splits (Section 3.2). Let $\mathcal{Q} = \{q_i\}_{i=1}^{N}$ denote a query set. Each query pairs an old fragment with its attachment context,

$$
q_i = \bigl(f_i^{\mathrm{old}}, \sigma_i\bigr),
$$

where $f_i^{\mathrm{old}}$ is the fragment to be replaced and $\sigma_i$ is the attachment signature---the atom and bond context through which the fragment is connected to the scaffold. In each closed-vocabulary protocol, each query is ranked against the full training-derived candidate vocabulary for that protocol:

$$
\mathcal{C}_i = \mathcal{V}_{\mathrm{train}}^{(p)},
$$

where $p$ indexes the candidate-matrix protocol. For Fresh Blind2, this is the Train2-derived 161-fragment vocabulary; for the original secondary-blind diagnostic matrix, this is the earlier 150-fragment vocabulary. The attachment signature $\sigma_i$ is supplied as query context and as a candidate-level feature rather than used as the primary hard filter. Attachment-compatible candidate subsets are evaluated separately as sensitivity analyses to distinguish ranking quality from candidate-coverage effects. A scoring model with parameters $\theta$ assigns each candidate a real-valued score

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

Queries can be multi-positive. Labels are structure-derived: they indicate observed structural substitution, not activity-outcome continuity. This limitation motivates reporting A4C and activity-comparable analyses only as supplementary workflow or boundary diagnostics (Sections 3.6.4 and 3.7).

Metric-bearing evaluations require at least one in-matrix positive per query; zero-positive matrices are reported as candidate-coverage diagnostics. The primary Fresh Blind2 full-161 evaluation has no zero-positive queries (Table 1).

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

Top-10 is the primary metric; Top-1, Top-5, Top-20, Top-50, and MRR are secondary diagnostics. The random expected Top-$K$ baseline accounts for multiple positives per query:

$$
\mathbb{E}[\mathrm{Random\ Top}\text{-}K]
=
\frac{1}{N}
\sum_i
\left[
1 -
\frac{\binom{|\mathcal{C}_i|-|\mathcal{P}_i|}{K}}
{\binom{|\mathcal{C}_i|}{K}}
\right].
$$

For Fresh Blind2 with $K=10$, this gives random expected Top-10 = 0.0987. Confidence intervals use nonparametric bootstrap at the query level (Section 3.6.2).

---

### 3.2 Benchmark Construction

**Data source and MMP extraction.** We use ChEMBL37K [1], a curated subset of approximately 37,000 drug-like molecules from ChEMBL33. MMPs are extracted by applying retrosynthetic bond-cleavage rules [2,3]: each molecule is recursively decomposed at acyclic single bonds to generate fragment--scaffold pairs; any two molecules sharing identical scaffold fragments but bearing different substituent fragments at the same attachment point are recorded as an MMP. The observed substitution $(f^{\mathrm{old}} \rightarrow c^*)$ provides a positive training example. Quality filters include salt stripping, stereochemistry normalization, removal of molecules with PAINS or Brenk alerts [13,14], and exclusion of fragments with molecular weight below 15 Da or above 250 Da. Additional curation and vocabulary-construction details are reported in Supplementary Methods SM1. Because PAINS/Brenk filtering is applied during data construction, supplementary A4C annotations are conditional post-ranking workflow annotations on a pre-filtered fragment vocabulary; they should not be interpreted as alert rates for an unfiltered replacement catalogue or as a re-estimate of construction-filter failures.

**Candidate construction.** The original secondary-blind benchmark uses a fixed 150-row closed-vocabulary matrix per query and is retained as diagnostic history. Fresh Blind2 uses a full Train2-vocabulary policy as the primary prospective matrix: each Blind2 query is scored against all 161 Train2-derived replacement fragments. The value 161 is the complete Train2-derived replacement vocabulary under the fixed curation pipeline before Blind2 scoring, not a vocabulary size selected by optimizing Blind2 positive coverage. The observed zero zero-positive Blind2 queries is reported as a coverage outcome of this fixed full-vocabulary policy. This full-vocabulary policy is intentionally query-conditioned rather than hard attachment-filtered; the attachment signature is represented in the feature space, and a Train2-observed attachment-subset analysis is reported only as a sensitivity check. Decoy repair and 1:5 negative subsampling are training-stage devices for constructing balanced or class-imbalance-controlled fitting rows; they are not the definition of the Fresh Blind2 full-161 evaluation matrix.

**Split design: query-side transform-heldout.** A conventional random split of MMP transformations would allow the same $(f^{\mathrm{old}}, \sigma)$ pair---the *query-side transform key*---to appear in both training and test sets, enabling memorization rather than generalization. In this manuscript, "transform key" means the query-side pair $(f^{\mathrm{old}}, \sigma)$, not the full replacement triple $(f^{\mathrm{old}}, \sigma, c)$. The split prevents repeated old-fragment/attachment-context queries across training and blind partitions, while replacement candidates are drawn from a shared closed training-derived vocabulary.

The original Train/Development/Secondary blind split and Fresh Blind2 split are derived from the same curated ChEMBL37K/MMP extraction pipeline but are different partitions, not nested subsets of one another. Train2 has fewer queries than the original Train split because transform keys were repartitioned to create a wider Dev2/calibration layer and a fresh blind set after the original secondary-blind audit. Dev2 intentionally contains substantially more old-fragment diversity than the original development set so that candidate-level fitting is less dependent on a narrow calibration fragment set. Consequently, absolute scores on the original secondary blind and Fresh Blind2 are not interpreted as difficulty-matched cross-benchmark numbers.

We construct four evaluation tiers:

- **Development/calibration set.** A held-out subset of the training partition, used during method development for architecture selection, feature engineering, and hyperparameter configuration. The base ranker architectures, scorer feature families F1--F9, and all training hyperparameters were configured on this set. The subsequent removal of the experimentally added prior_ranks family is described separately in Section 3.5 as a post-audit diagnostic ablation, not as primary model selection.
- **Original secondary blind set.** 13,347 queries. Transform-heldout from training: zero $(f^{\mathrm{old}}, \sigma)$ overlap. The candidate vocabulary $\mathcal{V}_{\mathrm{train}}$ is training-derived and closed; transform identities are held out, but the blind queries and training queries draw candidates from the same vocabulary. In this revised evidence hierarchy, this set is demoted to diagnostic and robustness support because the 77-feature deletion was selected after secondary-blind diagnostics.
- **Fresh Blind2 set.** 17,058 queries. This is the primary prospective evaluation for the revised manuscript. Train2 provides the closed 161-fragment replacement vocabulary, priors, and base-ranker artifacts; Dev2/calibration fits candidate-level HistGB scorers; Blind2 is evaluated once after the 82-feature and 77-feature configurations are fixed.
- **Canonical analysis set.** 21,052 queries with a slightly larger vocabulary, used exclusively for robustness analysis and mechanistic investigation. No primary performance claims rest on this set.

The Fresh Blind2 benchmark uses the full Train2 vocabulary of 161 candidates per query as the primary prospective matrix (Table 1). The original secondary-blind benchmark uses a fixed 150-row closed-vocabulary matrix (Table 2) and is retained as diagnostic history. A top-150 Blind2 policy is retained only as a comparability sensitivity because truncation produces zero-positive query matrices.

---

**Table 1.** Fresh Blind2 full-161 candidate matrix used for the primary prospective evaluation.

| Split | Queries | Candidates/query | Candidate rows | Positive-set size sum | Query-side transform keys | Old fragments | Attachment signatures | Zero in-matrix-positive queries | Vocabulary size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train2 | 93,083 | 161 | 14,986,363 | 151,216 | 316 | 131 | 5 | 0 | 161 |
| Dev2/calibration | 27,415 | 161 | 4,413,815 | 45,064 | 72 | 57 | 5 | 65 | 161 |
| Blind2 | 17,058 | 161 | 2,746,338 | 28,330 | 54 | 48 | 5 | 0 | 161 |

For Blind2, all recorded positives are covered by the full-161 candidate matrix, and the random expected Top-10 under the multi-positive formula is 0.0987. The Dev2 row is reported to document calibration-set coverage; the primary prospective claim uses Blind2.

---

**Table 2.** Original secondary-blind diagnostic candidate-matrix statistics (retained as diagnostic history).

| Split | Candidate rows | Queries | Positive rows | Unique old fragments | Transform identities | Single-positive queries | Multi-positive queries |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 16,497,750 | 109,985 | 181,483 | 109 | 351 | 73,870 | 36,115 |
| Development/calibration | 2,006,250 | 13,375 | 21,408 | 8 | 27 | 8,686 | 4,689 |
| Secondary blind | 2,002,050 | 13,347 | 20,082 | 19 | 63 | 9,377 | 3,970 |

Each query has 150 candidate rows in these matrices. The median positive-set size is 1 in all three splits, with maxima of 23, 17, and 10 for train, development, and secondary blind, respectively.

---

**Leakage verification.** Table 3 reports Fresh Blind2 overlap counts across split pairs for four relatedness dimensions: query-side transform-key overlap, old-fragment overlap, old-to-replacement pair overlap $(f^{\mathrm{old}}, c)$ excluding attachment signature, and attachment-signature overlap. Query-side transform-key overlap is zero for all Fresh Blind2 split pairs. The split removes exact query-side old-fragment/attachment-context overlap. It does not remove all old-fragment, replacement-pair, or attachment-signature relatedness. The original secondary-blind split is retained as diagnostic history; its non-query residual-overlap columns were not recomputed under the V7 unique-key convention and are not used for symmetric residual-relatedness comparison.

---

**Table 3.** Fresh Blind2 leakage and overlap verification across split pairs. Counts are unique-key overlaps under the current V7 convention.

| Split Pair | Query-side $(f^{\mathrm{old}}, \sigma)$ overlap | Old-fragment overlap | Old-to-replacement pair overlap $(f^{\mathrm{old}}, c)$, excluding attachment signature | Attachment-signature overlap |
|---|---:|---:|---:|---:|
| Train2 / Dev2 | 0 | 53 | 393 | 5 |
| Train2 / Blind2 | 0 | 46 | 430 | 5 |
| Dev2 / Blind2 | 0 | 17 | 188 | 5 |

All Fresh Blind2 split pairs have zero query-side transform-key overlap, confirming the transform-heldout property under the primary evaluation protocol.

---

For Fresh Blind2, Train2/Blind2 query-side transform-key overlap is 0, while old-fragment overlap and old-to-replacement pair overlap remain nonzero by design. The transform-heldout split is therefore a control against exact query-context memorization, not a claim of removing all residual chemical relatedness.

---

### 3.3 Base Ranking Methods

We implement five baseline ranking methods spanning a range of inductive biases. These provide comparative baselines and the input features for the candidate-level scorer (Section 3.4). Each captures a distinct signal: global replacement statistics, learned structural compatibility, interpretable molecular properties, and their consensus.

All methods are evaluated under the protocol described in Section 3.6. Results appear in Section 4.

**Best-of-DE+HGB diagnostic.** As a reference bound, we compute the fraction of secondary blind queries for which at least one of DE or HGB places the correct fragment in its top 10. This is a diagnostic bound for per-query selection between the two structurally complementary rankers, not a ceiling for models using additional candidate-level features. Complementarity diagnostics are reported in Section 4.5 and the Supplementary Information.

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

The HGB ranker uses explicit molecular features organized into four groups: (1) frequency features ($\widehat{p}_{\mathrm{att}}(c\mid\sigma)$, global fragment count, attachment-conditional count); (2) attachment signature features (bond order, ring membership, scaffold atom type, one-hot attachment signature); (3) molecular property descriptors (heavy atom count, molecular weight, logP, TPSA, ring count, HBD/HBA counts, rotatable bonds---for both candidate and query fragments); (4) fingerprint similarity (Tanimoto coefficient between Morgan fingerprints, radius 2, 2048 bits). The model is `HistGradientBoostingClassifier` (scikit-learn) with 200 iterations, max depth 6, learning rate 0.1. Candidates are ranked by the model output score for observed structure-derived positives; calibration diagnostics show that these outputs should be used as ranking scores rather than absolute decision scores.

The HGB ranker's interpretability also makes it useful for supplementary workflow diagnostics, but those diagnostics do not determine the main performance hierarchy.

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

#### 3.3.5 Base-Ranker and Feature-Generation Provenance

Base-ranker and candidate-level fitting provenance is reported in Supplementary Table S13. Dev2 base-ranker score columns are fitted-model scores rather than out-of-fold predictions unless explicitly stated; they are used for candidate-level fitting and diagnostics, not as independent Dev2 performance estimates. The base-ranker score columns are generated by applying the fixed fitted base-ranker pipeline to Dev2 candidates, then using those columns as inputs to the candidate-level fitting layer. This can make Dev2 score behavior optimistic as a development diagnostic, but it does not use Blind2 labels or outcomes. Primary claims do not use Dev2 metrics; Dev2 fitted score columns serve only as candidate-level fitting inputs, and the fixed fitted pipeline is applied once to Blind2.

### 3.4 Candidate-Level Scorer

#### 3.4.1 Architecture

The base rankers (Section 3.3) provide broad but shallow assessments. We introduce a **candidate-level scorer** that combines feature families with histogram gradient boosting to produce per-candidate ranking scores at finer granularity.

Given query $q_i$ and candidate $c$, the scorer computes $\boldsymbol{\phi}_{ic} = \Phi(q_i,c)$ and maps it to a real-valued ranking score:

$$
s_{ic} = F_{\Theta}(\boldsymbol{\phi}_{ic}).
$$

The training target is $y_{ic}$, fit by weighted binary log loss. Although the loss is classifier-style, the output is used only as a ranking score:

$$
\mathcal{L}(\Theta) = -\sum_{i=1}^{N}\sum_{c\in\mathcal{C}_i} w_{ic}
\left[ y_{ic}\log \sigma(s_{ic}) +(1-y_{ic})\log(1-\sigma(s_{ic})) \right],
$$

We choose HistGB for three reasons: (1) mixed-type inputs (continuous, discrete, binary, one-hot) are handled natively; (2) non-linear feature interactions are captured---a high prior score with a large molecular weight difference signals a different replacement context than either alone; (3) built-in feature importance estimates support ablation without manual selection. We do not claim architectural novelty in the classifier itself; the methodological contribution is the leakage-controlled ranking setup, the audited candidate-level feature design, and the explicit separation between prospective evaluation and post-audit feature pruning.

#### 3.4.2 Feature Families with Audit Trail

The 82-feature representation contains the full audited feature set, including five query-relative prior_ranks features. The 77-feature representation removes prior_ranks and is used as a no-prior-rank diagnostic comparator rather than as the primary prospective Top-10 configuration. Table 4 compares feature-family usage in HistGB82 and HistGB77.

---

**Table 4.** Feature families used by HistGB82 and HistGB77.

| Family | Count | Source | Used in HistGB82 | Used in HistGB77 | Evidence role | Leakage control |
|--------|------:|--------|:---:|:---:|----------|-----------------|
| F1: Model scores | 5 | Base ranker outputs ($z$-normalized) | Yes | Yes | candidate-level scoring signal | Train-set statistics only |
| F2: Model raw scores | 4 | Base ranker raw outputs | Yes | Yes | candidate-level scoring signal | No evaluation-set dependence |
| F3: Model ranks | 6 | Per-query ranks from base rankers | Yes | Yes | model-output ordering signal | Query-relative model output; distinct from prior_ranks |
| F4: Top-K flags | 4 | Binary: candidate in base ranker top-10 | Yes | Yes | consensus signal | Thresholds fixed at development time |
| F5: Prior scores | 5 | Logit-transformed empirical priors (P0/P1/P3/P4) | Yes | Yes | train-derived prior signal | Priors from train counts only |
| F6: Prior statistics | 12 | Smoothed rates, support/positive counts | Yes | Yes | prior reliability signal | Counts from train; smoothing fixed on development set |
| F7: PMI / conditional-prior contrast scores | 6 | Pointwise mutual information and contrast scores across prior levels | Yes | Yes | train-derived contrast signal | PMI from train co-occurrence |
| F8: Molecular descriptors | 10 | RDKit [18]: candidate properties + deltas from query | Yes | Yes | structure-derived compatibility signal | Structure-derived; no label dependence |
| F9: Similarity and frequency | 3 | Tanimoto similarity, global/attachment frequency | Yes | Yes | structural and empirical context | Frequency from train counts |
| Categorical one-hot features | 22 | Fragment identity, attachment signature, frequency bin | Yes | Yes | schema-controlled context signal | Frozen schema; unseen categories mapped to OTHER (Section 3.4.4) |
| Prior_Ranks | 5 | Per-query rank of prior scores | Yes | No | diagnostic contrast / instability test | Train-derived but query-relative; audited as unstable |
| **Total** | **82 / 77** | | **82 features** | **77 features** | | |

---

**Model-derived features (F1--F4).** Model scores (F1) and raw scores (F2) carry the primary signal from each base ranker. Model ranks (F3) encode relative ordering within each query's candidate pool. Top-K flags (F4) provide discrete consensus signals. Retained base-ranker ranks (F3) differ from the removed prior_ranks: base-ranker ranks summarize validated model outputs from trained rankers, whereas prior_ranks are sparse prior-score rank transforms that were audited as potentially unstable query-local frequency shortcuts (Section 3.5.2).

**Prior and statistical features (F5--F7).** These encode empirical replacement frequencies across four provenance levels: P0 (global fragment frequency), P1 (attachment-conditioned), P3 (cluster-conditioned within the same MMP transformation class), P4 (transferred prior score from chemically similar training fragments). Logit-transformed scores (F5) linearize empirical frequency scales. Smoothed rates, support counts, and positive counts (F6) quantify statistical reliability. PMI / conditional-prior contrast scores (F7) capture information gain at each conditioning level relative to the baseline---for example, $\mathrm{PMI}(P3, P1)$ measures how much additional information the cluster-level context provides beyond the attachment-level context alone. These are computed from training-set co-occurrence statistics and are distinct from the removed per-query prior_ranks features (Section 3.5).

**Molecular descriptors (F8) and similarity (F9).** Absolute candidate properties and deltas from the query fragment (heavy atom count, molecular weight, logP, TPSA) encode steric and electronic compatibility. Morgan fingerprint Tanimoto similarity (radius 2, 2048 bits) and frequency features anchor the prediction in structural and empirical context.

**Categorical features.** Old fragment identity (SMILES), attachment signature, and replacement frequency bin are one-hot encoded via frozen categorical schema alignment (Section 3.4.4). The encoding template is learned from the development set; categories unseen in development/calibration data (e.g., novel fragment identities in Blind2 or in the secondary-blind diagnostic set) are mapped to a reserved OTHER category, and output dimensionality is forced to match the development schema. This captures fragment-specific and attachment-specific effects without hand-written rules, while ensuring consistent feature dimensionality across splits.

#### 3.4.3 Training Protocol

The scorer uses `HistGradientBoostingClassifier` (scikit-learn [17]) with max 200 iterations, max depth 6, learning rate 0.1, `class_weight='balanced'`, `random_state=42`.

For Fresh Blind2, the reported 82-feature and 77-feature scores are produced by a single HistGB model per feature set fitted on the Dev2/calibration candidate matrix under fixed hyperparameters, then applied once to Blind2. This design treats Dev2/calibration as the fixed candidate-level fitting layer after Train2-derived vocabulary, priors, and base-ranker signals have been generated. Development-time grouped resampling is used only for diagnostics reported outside the primary Blind2 scoring table; the primary Blind2 score columns are not a cross-validation ensemble. Negative subsampling at 5:1 keeps all Dev2 positive rows and samples negatives with a fixed random seed to mitigate class imbalance. The 65 Dev2 zero-positive queries reported in Table 1 are excluded from scorer fitting before row-level negative subsampling; they are retained only as candidate-coverage diagnostics and are not used for metric-bearing evaluation. The 82-feature and 77-feature Blind2 scorers use the same HistGB hyperparameters.

For the original secondary blind evaluation, the post-audit 77-feature configuration was fixed after blind diagnostics and is therefore reported as diagnostic. For Fresh Blind2, the 82-feature and 77-feature configurations are fitted on Dev2/calibration and applied once to Blind2. No Blind2 labels or candidate-level outcomes were used to compute feature values, fit encoders, train model parameters, or choose hyperparameters. Feature standardization and categorical templates are fitted on development/calibration data.

The main reproducibility entry points are the Fresh Blind2 split, full-161 candidate matrix, 82/77 HistGB scorer runs, grouped-uncertainty audit, and stronger-baseline diagnostics. These scripts are listed in the Data and Software Availability statement so that score columns, query identifiers, and diagnostic bounds can be audited from the released project tree.

#### 3.4.4 Frozen Categorical Schema Alignment

Categorical features pose a schema-mismatch risk because development and blind sets can contain disjoint fragment identities. Without a frozen schema, one-hot encoded columns may differ in both dimensionality and semantic order across splits, causing learned feature weights to be applied to incorrect columns at inference time. We therefore use a frozen categorical schema: the one-hot encoder is fitted on the development set, recording observed categories and column ordering. For blind encoding, unseen categories are mapped to a reserved OTHER category, and the output matrix is forced to match the development schema. A shape assertion guarantees consistency before prediction. Implementation details for this audit are reported in Supplementary Methods SM6 and Supplementary Figure S4 rather than treated as a scientific result.

---

### 3.5 Prior_Ranks as a Post-Audit Instability Diagnostic

We used leave-one-family-out ablation to audit the prior_ranks feature family. This section documents the ablation design, the post-audit timeline, and the Fresh Blind2 non-replication that changes the interpretation from a positive feature-deletion claim to an instability diagnostic.

The prior_ranks removal was identified during post-audit analysis of the initial secondary-blind 82-feature scorer after blind-set diagnostics revealed fragment-specific degradation. In the revised evidence hierarchy, the old 77-feature secondary-blind result is post-audit only. Fresh Blind2 tests the no-prior-rank configuration prospectively and shows that it does not replicate a Top-10 advantage over the 82-feature scorer.

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

#### 3.5.2 Why Prior_Ranks Were Suspected to be Unstable

The prior_ranks features differ structurally from all others: they encode per-query rank position, which is **query-relative**. A rank of 3 means different things for different queries depending on candidate pool size, score distribution, and fragment identity. In the original secondary-blind audit, this made prior_ranks a plausible shortcut feature family because thresholds learned from development ranking patterns need not transfer to held-out fragment contexts.

Fresh Blind2 changes the strength of the conclusion. The old audit is consistent with shortcut learning [16], but the no-prior-rank configuration did not reproduce a Top-10 advantage over the 82-feature scorer in the prospective Blind2 evaluation. We therefore treat prior_ranks as an instability candidate rather than as a feature family proven to degrade generalization in all transform-heldout settings.

**Audit lesson.** Sparse query-relative prior-rank transforms should be treated as instability candidates and prospectively tested before being used as deletion targets.

#### 3.5.3 Final Feature Set

Removal of the five prior_ranks features defines the no-prior-rank 77-feature configuration used for the Blind2 replication test. In the revised evidence hierarchy, this configuration is a diagnostic comparator rather than the main Top-10 HistGB result. Fresh Blind2 performance is reported in Section 4.2, and prior-rank non-replication is reported in Section 4.4.

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

Aggregated counts $R(m,b)=\sum_i\operatorname{Rescue}_i$, $L(m,b)=\sum_i\operatorname{Lost}_i$, and net gain $G=R-L$ decompose accuracy changes into recoveries and regressions. Original secondary-blind rescue/lost diagnostics are reported in Supplementary Table S3.

#### 3.6.4 Protocol Separation

We enforce strict separation of evaluation protocols by evidentiary role (Table 5).

---

**Table 5.** Evaluation protocol separation.

| Protocol | Queries | Role in revised manuscript |
|----------|---------|---------|
| Development/calibration | varies | Architecture, feature, hyperparameter, and scorer fitting |
| Original secondary blind | 13,347 | Diagnostic history, post-audit evidence, and stress-test context |
| Fresh Blind2 | 17,058 | **Primary prospective evaluation** |
| Canonical analysis | 21,052 | Robustness and mechanism support only |
| A4C / activity diagnostics | varies | Boundary and workflow diagnostics only |

---

Primary prospective performance claims cite Fresh Blind2. Original secondary-blind results are retained as diagnostic history and post-audit evidence. Canonical, A4C, and activity-comparable analyses are explicitly labeled as robustness, workflow, or boundary diagnostics.

#### 3.6.5 Leakage Control

**Feature-level.** All statistics (prior scores, smoothing parameters, PMI, $z$-score statistics) derive exclusively from the training split. Transfer features use only training-set reference molecules.

**Categorical alignment.** The frozen categorical schema (Section 3.4.4) ensures consistent dimensionality across splits; unseen categories map to OTHER.

**Overlap verification.** Query-side $(f^{\mathrm{old}}, \sigma)$ overlap is zero for the transform-heldout evaluation pairs used for claims (Table 3). Residual old-fragment and old-to-replacement pair relatedness are tracked as diagnostic leakage-scope quantities rather than treated as eliminated.

**Evaluation-label-free feature space.** All 77 no-prior-rank features, as well as the five prior_ranks features used in HistGB82, are computable without Fresh Blind2 evaluation labels. They are derived from molecular structure, base-ranker outputs, or training-set statistics. No feature incorporates Blind2 ground-truth replacement labels. The 77-feature schema is fixed with a consistent column order, and the 82-feature schema adds the five prior_ranks columns for the prospective HistGB82 comparison.

---

### 3.7 Supplementary Workflow Diagnostics

A4C annotations are reported only in Supplementary Methods SM2 and Supplementary Table S9 as workflow-level diagnostics. They are not used for fitting, feature selection, ranking claims, activity-outcome claims, safety-related claims, or medicinal-chemistry claims. They remain useful for documenting coverage-limited downstream review categories, but they do not affect the Fresh Blind2 performance hierarchy.


## 4. Results

Fresh Blind2 is the primary prospective evaluation in the revised evidence hierarchy. The original secondary blind results remain useful for robustness and diagnostic history, but the old no-prior-rank gain was identified after blind diagnostics and is therefore reported in Supplementary Table S1 rather than used as the main performance result. We first describe the benchmark and candidate matrix, then report Fresh Blind2 performance, uncertainty, prior-rank non-replication, DE/HGB complementarity as a secondary-blind diagnostic, candidate-matrix sensitivity, and boundary diagnostics.

---

### 4.1 Benchmark and Candidate Matrix Construction

The benchmark evaluates closed-vocabulary recovery of observed structure-derived positives under query-side transform-heldout evaluation. In this manuscript, the held-out transform key is the query-side pair `(old fragment, attachment signature)`, not the full replacement triple. This design removes exact reuse of old-fragment/attachment contexts across train and blind partitions while preserving a closed train-derived candidate vocabulary.

The original secondary-blind benchmark uses a fixed 150-row candidate matrix. Fresh Blind2 uses a primary full-vocabulary policy with 161 Train2-derived candidates per query, which avoids zero-positive Blind2 queries and preserves all recorded Blind2 positives (Table 1). A top-150 Blind2 construction is retained only as a comparability sensitivity because truncation can create zero-positive query matrices. Under the full-161 Blind2 policy, random expected Top-10 is 0.0987.

The large change in Attachment-Frequency performance is a benchmark-composition diagnostic rather than a model-transfer result. Attachment-Frequency achieved Top-10 = 0.6019 on the original secondary-blind fixed-150 matrix but 0.1764 on Fresh Blind2. We do not interpret this drop as a direct biological or algorithmic effect. The two evaluations differ in split construction, old-fragment composition, and candidate policy: Fresh Blind2 ranks each query against the full Train2-derived vocabulary, whereas the original secondary-blind matrix is retained only as diagnostic history. Because a pure frequency ranker is especially sensitive to train-derived support and candidate-universe construction, old secondary-blind and Fresh Blind2 absolute scores are not treated as difficulty-matched. The Fresh Blind2 claims are therefore within-split comparisons against Score Blend and Borda, not direct numerical comparisons to the original secondary-blind scores.

### 4.2 Fresh Blind2 Primary Prospective Performance

Fresh Blind2 supports candidate-level scoring as the primary prospective result. On 17,058 Blind2 queries, the 82-feature HistGB candidate-level scorer achieved the strongest Top-10 among the original HistGB configurations and improved over both Score Blend and Borda(DE,HGB) at the query level (Table 6). HistGB82 also improved MRR over Score Blend (0.5383 versus 0.4242), with grouped MRR deltas reported in Table 7. The no-prior-rank 77-feature configuration remained above Score Blend and had slightly higher MRR than the 82-feature scorer, but it did not reproduce a Top-10 advantage over the 82-feature scorer.

**Table 6. Fresh Blind2 primary prospective ranking performance.** Metrics are computed on the full 161-candidate Train2 vocabulary. HistGB82 is the primary prospective Top-10 HistGB configuration. HistGB77 is a pre-specified no-prior-rank replication test on Blind2 and a diagnostic relative to the original secondary-blind audit. Original secondary-blind performance numbers are reported in Supplementary Table S1 as diagnostic history; they are not directly comparable to Fresh Blind2 numbers.

| Method | Top-1 | Top-5 | Top-10 | Top-20 | Top-50 | MRR | Evidence role |
|---|---:|---:|---:|---:|---:|---:|---|
| Attachment-Frequency | 0.1764 | 0.4367 | 0.5603 | 0.6493 | 0.8680 | 0.2988 | baseline |
| HGB | 0.2336 | 0.6181 | 0.7688 | 0.8806 | 0.9807 | 0.3976 | base ranker |
| Dual Encoder (DE) | 0.2497 | 0.6541 | 0.7907 | 0.8825 | 0.9941 | 0.4280 | base ranker |
| Borda(DE,HGB) | 0.2511 | 0.6503 | 0.8176 | 0.9104 | 0.9909 | 0.4284 | fusion baseline |
| MLP (rank-only) | 0.2527 | 0.6488 | 0.8058 | 0.9231 | 0.9856 | 0.4252 | fusion baseline |
| Score Blend | 0.2467 | 0.6568 | 0.8077 | 0.9191 | 0.9856 | 0.4242 | strongest pre-candidate baseline |
| HistGB82 | 0.3758 | 0.7452 | 0.8480 | 0.9425 | 0.9986 | 0.5383 | prospective candidate-level scorer |
| HistGB77 | 0.3808 | 0.7503 | 0.8411 | 0.9447 | 0.9987 | 0.5409 | no-prior-rank diagnostic |

### 4.3 Query-Level and Grouped Uncertainty

Queries are correlated within old-fragment and transform-key groups, so query-level bootstrap intervals are best interpreted as fixed-query uncertainty and can be anti-conservative for generalization across fragment groups. The scale of this dependence is substantial: 17,058 Blind2 queries collapse to 54 query-side transform keys and 48 old fragments, corresponding to average resampling groups of about 316 and 355 queries, respectively. We therefore treat grouped intervals as the conservative uncertainty summary for group-level generalization. HistGB82 showed an observed query-level Top-10 gain over Score Blend of +0.0403 with a query-level 95% CI of [0.0355, 0.0446], but transform-key and old-fragment Top-10 intervals cross zero. HistGB77 also improved over Score Blend at query level, while the 77-minus-82 Top-10 comparison was negative at query level. MRR sensitivity is more stable across grouped intervals (Table 7). Thus the supported Top-10 statement is an observed query-level improvement, not an established fragment- or transform-level generalization claim.

**Table 7. Fresh Blind2 paired deltas and grouped uncertainty.** Query-level intervals resample queries. Grouped intervals resample query-side transform keys or old fragments; full attachment-signature and cluster analyses are reported in Supplementary Table S7.

| Comparison | Metric | Delta | Query CI | Transform-key CI | Old-fragment CI | Interpretation |
|---|---|---:|---:|---:|---:|---|
| HistGB82 - Score Blend | Top-10 | +0.0403 | [+0.0355, +0.0446] | [-0.0068, +0.0734] | [-0.0101, +0.0717] | observed query-level gain; group-level Top-10 not established |
| HistGB82 - Score Blend | MRR | +0.1141 | [+0.1099, +0.1184] | [+0.0852, +0.1505] | [+0.0841, +0.1491] | positive under grouped uncertainty |
| HistGB77 - HistGB82 | Top-10 | -0.0070 | [-0.0094, -0.0044] | [-0.0207, +0.0048] | [-0.0216, +0.0050] | no prospective 77-over-82 Top-10 replication |
| HistGB77 - HistGB82 | MRR | +0.0026 | [+0.0007, +0.0045] | [-0.0086, +0.0203] | [-0.0075, +0.0219] | small MRR tradeoff; group sensitivity heterogeneous |
| HistGB77 - Score Blend | Top-10 | +0.0334 | [+0.0284, +0.0381] | [-0.0119, +0.0649] | [-0.0131, +0.0633] | baseline-beating query-level diagnostic; group-level Top-10 not established |

### 4.4 Prior-Rank Deletion Does Not Prospectively Replicate as a Top-10 Improvement

The prior-rank story changes under Fresh Blind2. In the original secondary-blind audit, deleting prior_ranks appeared to increase Top-10 after blind-set diagnostics. Fresh Blind2 tested that no-prior-rank configuration prospectively as the HistGB77 diagnostic comparator. It remained above Score Blend but ranked below HistGB82 on Top-10 (0.8411 versus 0.8480; Delta = -0.0070, 95% CI [-0.0094, -0.0044]) while showing a small MRR increase (+0.0026, 95% CI [0.0008, 0.0044]).

We therefore interpret prior_ranks as an unstable query-relative feature family rather than as a replicated feature-deletion improvement. The bounded conclusion is that sparse prior-rank transforms can behave differently across blind partitions and should be audited before being treated as transferable ranking signals. The old secondary-blind 82/77 comparison is retained in Supplementary Table S1 as diagnostic history, not as the primary model-selection result.

### 4.5 Secondary Diagnostics and Boundaries

Additional diagnostics are reported in the Supplementary Information. They support DE/HGB complementarity (Supplementary Table S12), candidate-matrix traceability (Supplementary Methods SM5 and Supplementary Table S8), and future learning-to-rank exploration, but they do not alter the main Fresh Blind2 evidence hierarchy. Candidate-matrix analyses are limited to the reported closed-vocabulary policies and do not establish robustness to all possible chemically enumerated, fixed-300, external, or hard-negative candidate universes.

Boundary diagnostics further restrict interpretation. Activity-comparable reranking analyses show that structure-derived positives can be linked to same-assay activity-comparable pairs, but degraded pairs can also receive high ranks; therefore no activity-outcome, biological-outcome, or medicinal-chemistry prioritization claim is supported (Supplementary Methods SM3 and Supplementary Table S10). Calibration diagnostics show that raw HistGB outputs should be interpreted as ranking scores rather than absolute probabilities (Supplementary Methods SM4 and Supplementary Table S11).

## 5. Discussion

### 5.1 What Replicates

The main prospective finding is that candidate-level HistGB scoring gives an observed query-level improvement under Fresh Blind2. HistGB82 achieved higher query-level Top-10 than Score Blend and Borda(DE,HGB), supporting candidate-level integration of base-ranker outputs, train-derived priors, molecular descriptors, and frozen categorical schema features for the fixed Blind2 query set. However, grouped Top-10 intervals cross zero, so generalizability across fragment or transform groups cannot be established from the current Top-10 evidence. MRR provides the more stable positive support: the HistGB82 improvement over Score Blend remains positive under transform-key and old-fragment grouped intervals. The primary Top-10 claim therefore remains query-level, with MRR serving as the stronger grouped secondary metric.

### 5.2 What Does Not Replicate

Fresh Blind2 changes the interpretation of the earlier no-prior-rank result. Although prior_ranks deletion appeared beneficial in the original secondary-blind post-audit analysis, the pre-specified no-prior-rank HistGB77 configuration did not replicate a Top-10 advantage over HistGB82 on Blind2. The value of the prior-rank analysis is diagnostic: it shows that post-audit feature deletions must be prospectively replicated before being promoted to modeling rules.

### 5.3 What the Benchmark Controls

The query-side transform-heldout split removes exact reuse of the old-fragment/attachment-context key across train and blind partitions. It does not eliminate all chemical relatedness: old-fragment, old-to-replacement pair, attachment-signature, and candidate-vocabulary relatedness remain measurable boundary conditions. Candidate-matrix audits support the reported full-161 and fixed-150 closed-vocabulary policies, but they do not establish robustness to every possible candidate universe.

### 5.4 What Remains Future Work

Future work should test pre-specified categorical-aware learning-to-rank architectures, chemically larger candidate universes, external or temporal validation, activity-labeled evaluation, and expert review. These directions are deliberately separated from the frozen claims of this manuscript. The present paper remains a leakage-controlled benchmark-audit study rather than an algorithm paper, drug-discovery validation study, or calibrated-probability model.

## 6. Conclusion

We introduced a leakage-controlled closed-vocabulary benchmark for structure-derived fragment replacement ranking and prospectively evaluated candidate-level scoring on Fresh Blind2. The 82-feature HistGB scorer showed an observed query-level Top-10 improvement over Score Blend and Borda(DE,HGB), while grouped Top-10 intervals crossed zero, reflecting correlation among fragment and transform groups.

The no-prior-rank 77-feature configuration remained baseline-beating and slightly improved MRR, but it did not reproduce its earlier Top-10 advantage over the 82-feature scorer. We therefore treat prior-rank deletion as a diagnostic of unstable feature transfer rather than as a replicated model-selection improvement.

The benchmark remains structure-derived: it does not support activity-outcome inference, biological-outcome claims, external temporal transfer, medicinal-chemistry utility claims, or absolute score-to-outcome claims. The central benchmark-audit lesson is to remove exact query-side transform memorization first, then ask which ranking signals still transfer.

## Data and Software Availability

The analysis uses ChEMBL37K, a curated subset of ChEMBL33. The V5/Fresh Blind2 evidence archive is available in the public repository at https://github.com/user141514/paper1/tree/codex/jcim-algorithm-archive, with the curated experiment-code folder at `bioisosteric_diffusion/paper1_experiment_code_archive`. The repository path uses legacy project naming; the benchmark claims remain structure-derived and do not imply activity-outcome or biological-outcome claims.

The primary prospective evidence uses the Fresh Blind2 evaluation. The evidence-lock manifest is `goal/PROJECT_5DAY_FULL_REVIEW/jcim_submission_candidate/V5_FRESH_BLIND2_EVIDENCE_LOCK_MANIFEST.md`; it records SHA256 hashes for the Claim Map V5, Fresh Blind2 split artifacts, the full-161 candidate matrix definition, HistGB82 and HistGB77 scorer outputs, grouped-uncertainty audit tables, candidate-matrix sensitivity diagnostics, learning-to-rank diagnostic outputs, activity/calibration boundary diagnostics, and the V7 manuscript/SI package. Analysis scripts, feature-schema documentation, query-level audit tables, and bootstrap summaries required to audit the reported numbers are included in the evidence archive or referenced by the manifest.

Large processed candidate matrices and redistribution-restricted ChEMBL-derived artifacts are documented in the evidence manifest and will be distributed only subject to the redistribution terms of the underlying ChEMBL-derived data.

---

## References

1. Zdrazil, B.; Felix, E.; Hunter, F.; et al. The ChEMBL Database in 2023: A Resource for Drug Discovery. *Nucleic Acids Res.* **2024**, *52* (D1), D1180-D1192. https://doi.org/10.1093/nar/gkad1004.
2. Hussain, J.; Rea, C. Computationally Efficient Algorithm to Identify Matched Molecular Pairs (MMPs) in Large Data Sets. *J. Chem. Inf. Model.* **2010**, *50* (3), 339-348. https://doi.org/10.1021/ci900450m.
3. Griffen, E.; Leach, A. G.; Robb, G. R.; Warner, D. J. Matched Molecular Pairs as a Medicinal Chemistry Tool. *J. Med. Chem.* **2011**, *54* (22), 7739-7750. https://doi.org/10.1021/jm200452d.
4. Patani, G. A.; LaVoie, E. J. Bioisosterism: A Rational Approach in Drug Design. *Chem. Rev.* **1996**, *96* (8), 3147-3176. https://doi.org/10.1021/cr950066q.
5. Wirth, M.; Zoete, V.; Michielin, O.; Sauer, W. H. B. SwissBioisostere: A Database of Molecular Replacements for Ligand Design. *Nucleic Acids Res.* **2013**, *41* (D1), D1137-D1143. https://doi.org/10.1093/nar/gks1059.
6. Polishchuk, P. CReM: Chemically Reasonable Mutations Framework for Structure Generation. *J. Cheminform.* **2020**, *12*, 28. https://doi.org/10.1186/s13321-020-00431-w.
7. Huang, S.; Wang, S.; Dong, J.; Xu, M.; Yuan, S. NeBULA: A Web-Based Novel Drug Design Platform for Up-to-Date Bioisosteric Replacement. *Med. Drug Discov.* **2025**, *28*, 100231. https://doi.org/10.1016/j.medidd.2025.100231.
8. Ertl, P. Identification of Bioisosteric Substituents by a Deep Neural Network. *J. Chem. Inf. Model.* **2020**, *60* (7), 3369-3375. https://doi.org/10.1021/acs.jcim.0c00290.
9. Kim, H.; Moon, S.; Zhung, W.; Kim, S.; Lim, J.; Kim, W. Y. DeepBioisostere: Discovering Bioisosteres with Deep Learning for a Fine Control of Multiple Molecular Properties. *arXiv* **2024**, arXiv:2403.02706v2 (version 2 revised 2025). Preprint; not peer-reviewed.
10. Masunaga, S.; Furui, K.; Kengkanna, A.; Ohue, M. GraphBioisostere: General Bioisostere Prediction Model with Deep Graph Neural Network. *J. Supercomput.* **2026**, *82*, 132. https://doi.org/10.1007/s11227-026-08232-y.
11. Dwork, C.; Kumar, R.; Naor, M.; Sivakumar, D. Rank Aggregation Methods for the Web. In *Proceedings of WWW10*; 2001; pp 613-622.
12. Cormack, G. V.; Clarke, C. L. A.; Buettcher, S. Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. In *Proceedings of SIGIR*; 2009; pp 758-759.
13. Baell, J. B.; Holloway, G. A. New Substructure Filters for Removal of Pan Assay Interference Compounds (PAINS). *J. Med. Chem.* **2010**, *53* (7), 2719-2740. https://doi.org/10.1021/jm901137j.
14. Brenk, R.; Schipani, A.; James, D.; et al. Lessons Learnt from Assembling Screening Libraries for Drug Discovery for Neglected Diseases. *ChemMedChem* **2008**, *3* (3), 435-444. https://doi.org/10.1002/cmdc.200700139.
15. Helmke, P. S.; Kandler, J.; Ilie, S.; Gaskin, L.; Ecker, G. F. Data-Driven Assessment of Bioisosteric Replacements and Their Influence on Off-Target Activity Profiles. *RSC Med. Chem.* **2025**, *16*, 6048-6058. https://doi.org/10.1039/d5md00686d.
16. Geirhos, R.; Jacobsen, J.-H.; Michaelis, C.; et al. Shortcut Learning in Deep Neural Networks. *Nat. Mach. Intell.* **2020**, *2* (11), 665-673. https://doi.org/10.1038/s42256-020-00257-z.
17. Pedregosa, F.; Varoquaux, G.; Gramfort, A.; et al. Scikit-Learn: Machine Learning in Python. *J. Mach. Learn. Res.* **2011**, *12*, 2825-2830.
18. Landrum, G.; et al. RDKit: Open-Source Cheminformatics Software. https://www.rdkit.org/ (accessed 2026-05-31).

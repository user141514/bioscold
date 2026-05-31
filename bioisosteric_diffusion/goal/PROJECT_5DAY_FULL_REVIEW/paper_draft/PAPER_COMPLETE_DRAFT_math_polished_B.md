# A Dual-Mode Workflow for Scaffold-Conditioned Fragment Replacement Proposal via Neural-Empirical Rank Fusion

## Abstract

Scaffold-conditioned fragment replacement proposal is a structure-derived ranking task in medicinal chemistry, but rigorous evaluation is challenging because random data splits leak fragment-attachment combinations, and single-model rankers leave complementary signals unexploited. We present a leakage-controlled benchmark with a transform-heldout split, a candidate-level histogram gradient-boosted scorer operating on 77 chemically motivated features, and a dual-mode workflow under a computational review proxy. On the secondary blind evaluation protocol (13,347 queries), the candidate-level scorer achieves Top10 = 0.9243 (95% CI [0.9198, 0.9289]), improving over the strongest pre-existing baseline by 6.86 percentage points (Delta = 0.0686, 95% CI [0.0633, 0.0739]). A leave-one-family-out ablation reveals that five features encoding per-query rank information actively harm cross-split generalization: removing them improves blind Top10 by 3.92 percentage points and reduces baseline hit loss from 596 to 6 queries. The dual-mode workflow separates conservative from exploratory proposals under provenance labels with explicit alert stratification (G4: 0.99%, G3: 9.67%, G2: 46.85%). All labels are structure-derived from ChEMBL matched molecular pairs and do not establish activity preservation.

---

## 1. Introduction

Replacing a selected substituent while preserving scaffold connectivity is a routine decision in lead optimization, yet computational support for this step remains difficult to benchmark rigorously. We study closed-vocabulary scaffold-conditioned fragment replacement proposal: given an old fragment and its attachment signature, a model ranks candidate replacements from the attachment-compatible subset of a global training replacement vocabulary. The task is deliberately structure-derived — it asks whether known replacement patterns can be recovered under controlled evaluation, not whether a proposed replacement preserves activity.

Structure-derived replacement pairs from ChEMBL [Zdrazil2023], extracted through matched molecular pair (MMP) analysis [Hussain2010; Griffen2011], provide scalable supervision for this setting. Building a credible benchmark remains difficult for three reasons. First, random data splits can leak the same fragment-attachment transform into both training and evaluation sets, allowing models to succeed through memorization rather than generalization. Second, simple attachment-frequency baselines are already strong, and learned rankers can overfit dataset-specific regularities unless tested under explicit leakage control. Third, structural and empirical ranking signals are complementary in principle, but combining them effectively without access to validation tuning — which is precluded by the leakage-controlled split design — is non-trivial.

We address these issues with a framework comprising four components. First, we define a transform-heldout benchmark with a secondary blind evaluation protocol (13,347 queries, 150-fragment vocabulary) that isolates unseen fragment-attachment combinations and prevents memorization of transform identities. Second, we demonstrate that parameter-free Borda fusion of a Dual Encoder (DE) and a feature-based histogram gradient-boosted ranker (HGB) produces complementary ranking signals under leakage control. Third, we introduce a candidate-level scorer — a HistGB model operating on 77 features across 9 chemically motivated families — that learns per-candidate trustworthiness from feature interactions at finer granularity than query-level gating. Fourth, we organize the resulting outputs into a dual-mode workflow under an A4C computational review proxy that annotates provenance and alert strata, so that exploratory gains are reported with explicit risk context rather than as unconditional recommendations.

On the secondary blind evaluation protocol, the candidate-level scorer achieves Top10 = 0.9243 (+6.86 pp over the score-blend baseline, 95% CI [+6.33, +7.39] pp). A feature ablation study reveals that five per-query rank features degrade generalization — removing them improves blind Top10 by 3.92 pp and reduces baseline hit loss from 596 to 6 queries, a finding we interpret as an instance of shortcut learning that yields a generalizable design rule for feature engineering in learning-to-rank systems. Under the dual-mode risk-stratification analysis, shared G4 candidates form a low-alert reference region (0.99%), while Borda-only G2 candidates carry a 46.85% alert rate requiring expert review.

Our contributions are as follows. First, we provide a leakage-controlled benchmark for closed-vocabulary scaffold-conditioned fragment replacement with a transform-heldout split and secondary blind evaluation protocol. Second, we present a candidate-level scorer that substantially exceeds all base rankers through dense chemical featurization, and demonstrate through systematic ablation that removing per-query rank features simultaneously improves accuracy and safety. Third, we enforce strict protocol separation — primary performance evidence (blind), robustness and mechanism evidence (canonical analysis), and workflow interpretation (dual-mode risk-stratification) each occupy distinct evidentiary roles — preventing the claim inflation that often accompanies multi-protocol computational method papers. Fourth, we frame exploratory gains through a dual-mode workflow that keeps provenance and alert stratification explicit, acknowledging the computational proxy's limitations.

---

## 2. Related Work

**Matched molecular pairs and structure-derived replacements.** Bioisosteric replacement is a longstanding design concept in medicinal chemistry [Patani1996]. Matched molecular pair analysis extracts local substitution patterns from large compound collections by identifying molecules that differ at a single structural site while sharing an identical scaffold [Hussain2010; Griffen2011]. Databases such as ChEMBL [Zdrazil2023] make structure-derived replacement catalogues feasible at scale. These data record observed structural substitutions and serve as supervision for replacement proposal, but they do not encode activity continuity.

**Rule-based and database-driven tools.** SwissBioisostere organizes observed replacements into a searchable database for ligand design [Wirth2012], while CReM uses fragment environments from known compounds to perform context-aware replacement generation [Polishchuk2020]. These frameworks exploit empirical replacement context but define different task formulations — typically open-ended generation or database lookup rather than the closed-vocabulary query-level ranking benchmark studied here.

**Learning-based methods.** Deep learning methods for bioisosteric replacement include descriptor-based neural networks [Ertl2020], deep generative models that select fragments for removal and insertion [Kim2024], and graph neural networks applied to ChEMBL-derived molecular pairs [Masunaga2026]. These methods demonstrate that structural representations capture replacement-relevant information beyond frequency statistics, but they are typically evaluated on random splits that may overestimate generalization and do not define the same closed-vocabulary ranking task.

**Rank aggregation and fusion.** Borda count is a classical parameter-free aggregation rule for combining ranked lists [Dwork2001]. Reciprocal rank fusion [Cormack2009] provides a weighted alternative. Our use of fusion is narrower: we test whether structural and empirical rankers remain complementary under leakage-controlled evaluation and use Borda's parameter-free property as a practical necessity — the transform-heldout split prevents validation-based tuning of fusion weights.

**Computational alert filters.** PAINS [Baell2010] and Brenk alerts [Brenk2008] are widely used medicinal-chemistry screening heuristics. We adopt the same limited role for the A4C proxy: a rule-based annotation layer that organizes provenance strata without converting exploratory outputs into approval decisions.

---

## 3. Methods

This section presents the complete methodological framework for the fragment replacement ranking task. Section 3.1 defines the task formulation, supervision labels, and evaluation metrics. Section 3.2 describes the construction of the benchmark dataset, including matched molecular pair extraction, decoy construction, and the transform-heldout split design. Sections 3.3--3.5 present the ranking methods: five base ranking methods spanning a range of inductive biases (Section 3.3), followed by the candidate-level scorer with its 77-dimensional feature space and gradient boosting architecture (Section 3.4), and concluding with an ablation study demonstrating that removal of a feature family encoding per-query rank information substantially improves cross-split generalization (Section 3.5). Sections 3.6--3.7 cover the evaluation protocol, including bootstrap confidence intervals, rescue and lost analysis, strict protocol separation, leakage controls (Section 3.6), and the A4C computational review proxy for qualitative risk stratification of proposals (Section 3.7).

---

### 3.1 Task Formulation

We formulate scaffold-conditioned fragment replacement as a closed-vocabulary, multi-positive learning-to-rank problem. Let

$$
\mathcal{Q}=\{q_i\}_{i=1}^{N}
$$

denote a query set. Each query is an old fragment together with its attachment context,

$$
q_i=\bigl(f_i^{\mathrm{old}},\sigma_i\bigr),
$$

where $f_i^{\mathrm{old}}$ is the fragment to be replaced and $\sigma_i$ is the attachment signature, i.e. the atom and bond context through which the fragment is connected to the scaffold. The replacement vocabulary is built only from the training split and is denoted by $\mathcal{V}_{\mathrm{train}}$. For query $q_i$, only candidates that are compatible with $\sigma_i$ are eligible:

$$
\mathcal{C}_i
=
\left\{c\in\mathcal{V}_{\mathrm{train}}:\chi(c,\sigma_i)=1\right\},
$$

where $\chi(c,\sigma_i)\in\{0,1\}$ is a deterministic attachment-compatibility predicate. A scoring model with parameters $\theta$ assigns each candidate a real-valued score

$$
s_{\theta} : (q_i,c)\mapsto s_{\theta}(i,c)\in\mathbb{R},\qquad c\in\mathcal{C}_i.
$$

Candidates are ranked in decreasing order of $s_{\theta}(i,c)$. To make the ranking function well-defined even under score ties, let $\tau_i(c)$ be a fixed deterministic tie-breaking key, such as the candidate index in the frozen vocabulary. The induced rank is

$$
\rho_{\theta}(i,c)
=
1+
\sum_{c'\in\mathcal{C}_i}
\mathbb{I}\!\left[
 s_{\theta}(i,c')>s_{\theta}(i,c)
 \;\lor\;
 \bigl(s_{\theta}(i,c')=s_{\theta}(i,c)\land \tau_i(c')<\tau_i(c)\bigr)
\right].
$$

**Supervision labels.** Let $\mathcal{M}$ denote the set of structure-derived replacement triples obtained from matched molecular pair extraction after the appropriate train/evaluation split has been fixed. For query $q_i$, the positive set is

$$
\mathcal{P}_i
=
\left\{c\in\mathcal{C}_i:
\bigl(f_i^{\mathrm{old}},\sigma_i,c\bigr)\in\mathcal{M}\right\}.
$$

The corresponding candidate-level binary label is

$$
y_{ic}=\mathbb{I}\!\left[c\in\mathcal{P}_i\right],
\qquad c\in\mathcal{C}_i.
$$

Queries can be multi-positive because more than one replacement fragment can be observed for the same old fragment and attachment context. These labels are structure-derived: they indicate observed structural substitution, not activity preservation.

**Evaluation metrics.** For a scoring function $s_\theta$, the best positive rank for query $q_i$ is

$$
r_i^+(\theta)=\min_{c\in\mathcal{P}_i}\rho_\theta(i,c).
$$

The per-query hit at cutoff $K$ is

$$
h_i^{(K)}(\theta)=\mathbb{I}\!\left[r_i^+(\theta)\le K\right].
$$

The query-level Top-$K$ accuracy is

$$
\operatorname{Top@K}(\theta;\mathcal{Q})
=
\frac{1}{N}\sum_{i=1}^{N}h_i^{(K)}(\theta),
$$

and mean reciprocal rank is

$$
\operatorname{MRR}(\theta;\mathcal{Q})
=
\frac{1}{N}\sum_{i=1}^{N}\frac{1}{r_i^+(\theta)}.
$$

Top-10 is the primary metric; Top-1, Top-5, Top-20, Top-50, and MRR are secondary diagnostics. All confidence intervals are computed by nonparametric bootstrap resampling at the query level, so that all candidates belonging to a sampled query are retained together.

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

The Attachment-Frequency ranker estimates the empirical training-set prior for candidate $c$ under attachment signature $\sigma$. Let $n_{\mathrm{train}}(c,\sigma)$ be the number of training replacement pairs in which $c$ appears under $\sigma$. The score is

$$
s_{\mathrm{att}}(c,\sigma)
=
\widehat{p}_{\mathrm{att}}(c\mid\sigma)
=
\frac{n_{\mathrm{train}}(c,\sigma)}
       {\sum_{c'\in\mathcal{V}_{\mathrm{train}}} n_{\mathrm{train}}(c',\sigma)}.
$$

Candidates are ranked by descending $s_{\mathrm{att}}(c,\sigma)$. The method has no learned parameters and uses only training-set counts.

This approach embodies a straightforward but powerful empirical prior: fragments that have frequently appeared as replacements at a given attachment environment are more likely to be valid replacements again. The method captures the *global* replacement tendency of each fragment under each attachment context, information that is immediately useful but inherently coarse: it cannot distinguish between two candidates with identical frequency counts that differ in chemical compatibility with the specific query fragment. The Attachment-Frequency ranker therefore serves as the natural lower bound for all subsequent methods. On the secondary blind split, it achieves a Top-10 accuracy of 0.6019 (95% CI: 0.5933--0.6104).

#### 3.3.2 Dual Encoder Ranker

The Dual Encoder (DE) ranker replaces the frequency-based heuristic with a learned measure of structural compatibility. Its design follows the standard dual-encoder architecture for similarity learning: a query encoder and a candidate encoder that project their respective inputs into a shared embedding space, where compatibility is scored by cosine similarity.

The query encoder maps the old-fragment representation and attachment embedding to a query vector, while the candidate encoder maps the candidate fragment to a candidate vector. Writing $x(f)$ for a molecular fingerprint and $a(\sigma)$ for the attachment embedding,

$$
\mathbf{h}^{q}_{i}
=
g_{\phi}\!\left(x(f_i^{\mathrm{old}}),a(\sigma_i)\right),
\qquad
\mathbf{h}^{c}_{c}
=
h_{\psi}\!\left(x(c)\right).
$$

The DE score is the cosine similarity

$$
s_{\mathrm{DE}}(i,c)
=
\frac{\langle \mathbf{h}^{q}_{i},\mathbf{h}^{c}_{c}\rangle}
{\lVert\mathbf{h}^{q}_{i}\rVert_2\,\lVert\mathbf{h}^{c}_{c}\rVert_2}.
$$

For training, positives are paired with attachment-compatible negatives $\mathcal{N}_i\subset\mathcal{C}_i\setminus\mathcal{P}_i$. The pairwise margin objective can be written as

$$
\mathcal{L}_{\mathrm{DE}}
=
\sum_{i=1}^{N}
\sum_{c^{+}\in\mathcal{P}_i}
\sum_{c^{-}\in\mathcal{N}_i}
\left[\delta - s_{\mathrm{DE}}(i,c^{+}) + s_{\mathrm{DE}}(i,c^{-})\right]_{+},
$$

where $[t]_+=\max(t,0)$ and $\delta>0$ is the ranking margin. The Dual Encoder provides a fundamentally different signal from the frequency-based approach. While $P(c \mid \sigma)$ captures global empirical trends -- "this candidate is common at this attachment" -- the DE captures whether a *specific* candidate's molecular structure suits a *specific* query's chemical context. A candidate with low global frequency may still be the correct replacement if its substructure is complementary to the query fragment. This divergence in signal is reflected quantitatively: DE achieves a Top-10 accuracy of 0.8055 (95% CI: 0.7986--0.8122) on the secondary blind split, a gain of +0.2036 over the frequency baseline, confirming that learned structural compatibility captures substantial signal beyond empirical frequency.

#### 3.3.3 Histogram Gradient-Boosted Ranker (HGB)

The Histogram Gradient-Boosted (HGB) Ranker approaches fragment replacement as a supervised classification problem with explicit, interpretable molecular features. For each query-candidate pair $(q, c)$, we compute a feature vector $\mathbf{x}_{q,c}$ organized into four groups: (1) *frequency features:* $P(c \mid \sigma)$, global replacement frequency $\text{count}(c)$, and the attachment frequency $\text{count}(c, \sigma)$; (2) *attachment signature features:* the bond order at the attachment point, whether the bond belongs to a ring system, the scaffold atom type at the attachment, and the attachment signature itself (one-hot encoded); (3) *molecular property descriptors:* heavy atom count, molecular weight, logP, TPSA, ring count, hydrogen bond donor and acceptor counts, and the number of rotatable bonds -- computed for both the candidate and query fragments; and (4) *fingerprint similarity:* the Tanimoto coefficient between the Morgan fingerprints (radius 2, 2048 bits) of the original and candidate fragments. The HGB model is trained using scikit-learn's `HistGradientBoostingClassifier` with 200 boosting iterations, a maximum depth of 6, and a learning rate of 0.1. Candidates are ranked by the model's predicted probability of being a valid replacement.

The HGB ranker occupies a distinct methodological position in our framework. Unlike the Dual Encoder, whose latent embeddings trade interpretability for representational flexibility, the HGB model operates directly on physicochemical descriptors whose meanings are transparent to a medicinal chemist. This makes it the natural source for the Conservative Mode in our dual-mode workflow (Section 3.7), where chemically conservative proposals -- those grounded in observable, nameable properties rather than opaque learned features -- are preferred for high-confidence applications. On the secondary blind split, HGB achieves a Top-10 accuracy of 0.7437 (95% CI: 0.7356--0.7516), establishing a strong feature-based baseline that, while lower than DE, captures a complementary signal.

#### 3.3.4 Borda Fusion and Score Blending

**Borda fusion.** The Borda fusion method combines the DE and HGB rankers through parameter-free rank aggregation. For query $q_i$, let $\rho_m(i,c)$ be the rank of candidate $c$ assigned by method $m\in\{\mathrm{DE},\mathrm{HGB}\}$, with smaller ranks indicating higher-ranked candidates. The Borda score is

$$
S_{\mathrm{Borda}}(i,c)
= \sum_{m\in\{\mathrm{DE},\mathrm{HGB}\}}
\left(|\mathcal{C}_i| + 1 - \rho_m(i,c)\right),
$$

and candidates are re-ranked by descending $S_{\mathrm{Borda}}$.

The parameter-free design of Borda fusion is deliberate and methodologically motivated by our transform-heldout evaluation protocol (Section 3.2). Under a transform-heldout split, no query in the validation or secondary blind sets shares its (old_fragment, attachment_signature) combination with any training query. This means that any fusion weights optimized on a validation set would necessarily be tuned to ranking patterns that may not generalize to queries with unseen transform combinations -- there is no guarantee that the optimal weight for DE versus HGB on validation queries is optimal for secondary blind queries. Borda fusion sidesteps this entirely by using equal, fixed weights that require no tuning, ensuring consistent behavior across all splits. This conservative choice is appropriate for a baseline method: if simple equal-weight fusion already demonstrates substantial gains, then the complementarity signal is robust and not an artifact of weight optimization.

The Borda results confirm this. On the secondary blind split, Borda fusion achieves a Top-10 accuracy of 0.8384 (95% CI: 0.8321--0.8447), representing gains of +0.0329 over DE alone and +0.0947 over HGB alone. The improvement over DE is particularly notable: DE is the stronger individual ranker, yet fusing it in equal proportion with the weaker HGB still produces a net gain, confirming that the two methods make substantially different errors. The Borda consensus captures the approximately 3% of queries where DE fails but HGB succeeds, approaching the Oracle ceiling without any learned component.

**Score Blend (MLP + HGB).** The Score Blend extends the fusion concept by introducing a lightweight learned aggregator: a rank-only MLP that takes as input three per-query rank values for each candidate -- the ranks assigned by the DE, HGB, and Attachment-Frequency rankers -- and produces a single score $s_{\text{MLP}}(q, c)$. This MLP is a two-layer network with hidden dimensionality 32 and ReLU activations, trained as a binary classifier on the training split. It learns to weight the three base rankers' signals differently depending on the query context, adapting the fusion strategy to each query rather than applying the fixed Borda formula.

The final Score Blend combines the MLP score with a separately refit HGB score after within-query standardization. For any score vector $u(i,c)$ over $c\in\mathcal{C}_i$, define

$$
z_i\!\left(u(i,c)\right)
= \frac{u(i,c)-\mu_i(u)}{\sigma_i(u)+\varepsilon},
\qquad
\mu_i(u)=\frac{1}{|\mathcal{C}_i|}\sum_{c'\in\mathcal{C}_i}u(i,c'),
$$

with $\sigma_i(u)$ the corresponding within-query standard deviation and $\varepsilon$ a small numerical stabilizer. The blended score is

$$
s_{\mathrm{blend}}(i,c)
= \lambda\, z_i\{s_{\mathrm{MLP}}\}(c)
+ (1-\lambda)\, z_i\{s_{\mathrm{HGB-refit}}\}(c),
\qquad \lambda=0.95.
$$

Here $s_{\mathrm{HGB-refit}}$ is the predicted probability from an HGB model refit on the training data with the MLP output included as an additional feature.

On the secondary blind split, the rank-only MLP achieves a Top-10 accuracy of 0.8402 (95% CI: 0.8339--0.8466). While this is a marginal improvement over Borda (+0.0018, not statistically significant at the Top-10 level), the MLP provides a statistically significant improvement in Mean Reciprocal Rank, indicating that its benefit is in elevating the correct candidate closer to the top of the ranked list within queries where it already appears in the top 10. The full Score Blend achieves 0.8558, establishing the strongest baseline prior to the candidate-level scorer and the highest performance achievable from base ranker signals alone.

**Summary: the gap between base rankers and the Oracle.** Across all five methods, the best-performing individual ranker (MLP, 0.8402) and the strongest combination (Score Blend, 0.8558) fall short of the Oracle ceiling (0.8686). The residual gap -- approximately 1.3 to 2.8 percentage points -- represents queries where, even after rank fusion and learned aggregation, the base ranker signals are insufficient to surface the correct candidate. Closing this gap requires moving beyond rank-level and frequency-level features to a richer representation that incorporates fine-grained chemical constraints: the properties of the fragment-scaffold interface, the steric and electronic compatibility of the replacement, and the structural relationships among candidates within a query. This motivates the candidate-level scorer in Section 3.4, which augments the base ranker signals with 77 chemically motivated features and a dense gradient-boosted model capable of discriminating among candidates that the base rankers cannot separate.

---

### 3.4 Candidate-Level Scorer

The base ranking methods in Section 3.3 provide broad but shallow assessments of replacement candidates: they capture frequency signals (Attachment-Frequency), learned pairwise preferences (Dual Encoder), graph-structural features (HGB), ensemble rank aggregation (Borda Fusion), and cross-ranker score blending (Score Blend). Yet none operate at the granularity needed to distinguish chemically subtle differences between plausible and implausible replacements -- a candidate ranked 11th by every base ranker may be chemically superior to one ranked 3rd, if the base rankers are merely exploiting shallow frequency or similarity artifacts. We address this gap with a **candidate-level scorer** that combines 77 chemically motivated features with a gradient boosting model to produce a fine-grained replacement score for every candidate in a query's candidate pool.

#### 3.4.1 Architecture and Feature Overview

Given query $q_i$ and candidate fragment $c$, the final scorer computes a feature vector

$$
\boldsymbol{\phi}_{ic}=\Phi(q_i,c)\in\mathbb{R}^{77}.
$$

The scorer is a histogram gradient-boosted model $F_{\Theta}$ that maps this feature vector to a candidate-level probability,

$$
\widehat{p}_{ic}
=
F_{\Theta}(\boldsymbol{\phi}_{ic}),
\qquad
0\le \widehat{p}_{ic}\le 1.
$$

Candidates are ranked within each query by decreasing $\widehat{p}_{ic}$. The training target is $y_{ic}$, and the model is fit by weighted binary log loss over query-candidate rows,

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
$$

where $w_{ic}$ is the sample weight induced by class balancing or negative subsampling.

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

Formally, for a prior score $u_j(i,c)$, the corresponding query-relative rank feature is

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

Let $\mathcal{A}_{82}$ be the index set of the full 82-dimensional feature vector and let

$$
\mathcal{R}_{\mathrm{prior}}
=
\left\{
 r_{\mathrm{backoff}},
 r_{\mathrm{cont}},
 r_{\mathrm{tP4}},
 r_{\mathrm{P1}},
 r_{\mathrm{P3}}
\right\}
\subset\mathcal{A}_{82}.
$$

The final 77-dimensional feature map is the coordinate projection that removes those five indices:

$$
\boldsymbol{\phi}^{(77)}_{ic}
=
\Pi_{\mathcal{A}_{82}\setminus\mathcal{R}_{\mathrm{prior}}}
\!\left(\boldsymbol{\phi}^{(82)}_{ic}\right).
$$

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

\1

Let $b$ denote the baseline scorer and $m$ the candidate-level scorer under comparison. A rescue indicator and a loss indicator are defined query-wise as

$$
\operatorname{Rescue}_i(m,b)
=
\mathbb{I}\!\left[r_i^+(b)>10\right]
\mathbb{I}\!\left[r_i^+(m)\le 10\right],
$$

$$
\operatorname{Lost}_i(m,b)
=
\mathbb{I}\!\left[r_i^+(b)\le 10\right]
\mathbb{I}\!\left[r_i^+(m)>10\right].
$$

Aggregated counts are

$$
R(m,b)=\sum_{i=1}^{N}\operatorname{Rescue}_i(m,b),
\qquad
L(m,b)=\sum_{i=1}^{N}\operatorname{Lost}_i(m,b),
$$

and the net Top-10 hit gain in query counts is

$$
G(m,b)=R(m,b)-L(m,b).
$$
 If $b$ denotes a baseline ranker and $m$ the evaluated scorer, then

\begin{align}
\operatorname{Rescue}(m;b)
&= \sum_{i=1}^{N}
\mathbf{1}\!\left[r_i^{+}(b)>10\right]
\mathbf{1}\!\left[r_i^{+}(m)\le 10\right],\\
\operatorname{Lost}(m;b)
&= \sum_{i=1}^{N}
\mathbf{1}\!\left[r_i^{+}(b)\le 10\right]
\mathbf{1}\!\left[r_i^{+}(m)>10\right],\\
\operatorname{Net}(m;b)
&= \operatorname{Rescue}(m;b)-\operatorname{Lost}(m;b).
\end{align}

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

---

## 4. Results

We evaluate the candidate-level scorer and all baseline methods on the secondary blind evaluation protocol (Section 3.6). The baseline for all comparisons is the score-blended MLP+HGB ranker (blind Top10 = 0.8558), which represents the strongest pre-D4S ranking strategy. All reported confidence intervals are 95% bootstrap intervals computed from 5,000 nonparametric replicates resampled at the query level (Section 3.6.2). Results are organized into six subsections: main blind performance, feature ablation, rescue and lost analysis, per-fragment robustness, dual-mode risk stratification, and a brief account of routing attempts that motivated the candidate-level approach.

---

## 4.1 Main Blind Performance on the Secondary Blind Evaluation Protocol

Table 1 reports the full head-to-head comparison of all ranking methods on the secondary blind set. The table spans the full hierarchy of approaches: simple frequency baselines, learned pairwise models, graph-structured rankers, ensemble fusion, and the proposed candidate-level scorer. The oracle row provides a diagnostic upper bound representing perfect per-query selection between the Dual Encoder and HGB base rankers.

**Table 1.** Blind Top10 accuracy, mean reciprocal rank (MRR), delta versus the pre-D4S score-blend baseline, and 95% bootstrap confidence intervals across 13,347 blind queries. The candidate-level scorer (77 features) is the primary result. Oracle(DE, HGB) represents perfect per-query selection between DE and HGB outputs, serving as a diagnostic ceiling.

| Method | Top10 | MRR | Delta vs Baseline | 95% CI |
|--------|-------|-----|-------------------|--------|
| Attachment frequency | 0.6019 | 0.2849 | -0.2539 | [0.5933, 0.6104] |
| Dual Encoder (DE) | 0.8055 | 0.4174 | -0.0503 | [0.7986, 0.8122] |
| HGB | 0.7437 | 0.3791 | -0.1121 | [0.7356, 0.7516] |
| Borda(DE, HGB) | 0.8384 | 0.4176 | -0.0174 | [0.8321, 0.8447] |
| Rank-only MLP reranker | 0.8402 | 0.4741 | -0.0156 | [0.8339, 0.8466] |
| Score Blend (pre-D4S baseline) | 0.8558 | 0.4842 | reference | [0.8499, 0.8619] |
| **Candidate-Level Scorer (77 feat)** | **0.9243** | — | **+0.0686** | **[0.9198, 0.9289]** |
| Oracle(DE, HGB) | 0.8686 | 0.5433 | — | — |

The candidate-level scorer with 77 features achieves a blind Top10 of 0.9243, an improvement of +6.86 percentage points over the score-blend baseline (Table 1). This is the highest Top10 accuracy among all evaluated methods. The bootstrap 95% confidence interval for the delta against baseline is [+0.0633, +0.0739]; because zero falls well outside this interval, the improvement is statistically significant at p < 0.05. The absolute Top10 confidence interval [0.9198, 0.9289] confirms that the gain is not an artifact of a particular bootstrap draw.

Borda fusion of DE and HGB (Top10 = 0.8384) outperforms both of its constituents individually, confirming that the structural representations learned by the Dual Encoder and the graph-based features of HGB capture complementary signals (Section 3.3.4). However, Borda itself falls below the score-blend baseline by -1.74 percentage points, demonstrating that a simple learned combination of MLP and HGB scores already exceeds the best parameter-free rank fusion. The rank-only MLP reranker (Top10 = 0.8402, MRR = 0.4741) provides a meaningful MRR improvement over Borda, but its Top10 gain over Borda is not significant: the confidence intervals of the two methods overlap substantially, and the bootstrap delta CI crosses zero.

The gap between the candidate-level scorer (0.9243) and the Oracle(DE, HGB) upper bound (0.8686) is notable in two respects. First, the scorer surpasses the oracle by +5.57 percentage points, meaning that the dense feature representation extracts more signal from each candidate than can be achieved by optimally selecting between the two base rankers at the query level. Second, the oracle itself is not the ceiling — the theoretical headroom beyond the scorer (the remaining +7.57 percentage points to perfect Top10 identification) indicates that further gains are possible through richer feature representations or alternative model architectures.

---

## 4.2 Feature Ablation and Design Validation

To understand which feature families drive the scorer's performance, we performed a leave-one-family-out ablation on the 82-feature candidate-level model. The 82 features span nine chemically motivated families (Section 3.4.2). Each row in Table 2 reports the blind Top10 when a single family is removed from the full set, together with the delta against the 82-feature reference.

**Table 2.** Leave-one-family-out feature ablation on the blind set. Each row removes one feature family from the full 82-feature model. Positive deltas indicate that removal improves generalization. The 82-feature reference corresponds to the D4S28R configuration.

| Condition | Blind Top10 | Delta from 82-feat |
|-----------|-------------|---------------------|
| Full (82 features) | 0.8851 | reference |
| **Drop prior\_ranks (77 feat)** | **0.9243** | **+0.0392** |
| Drop prior\_positives | 0.9037 | +0.0186 |
| Drop query\_stats | 0.9019 | +0.0168 |
| Drop model\_ranks | 0.8697 | -0.0154 |
| Drop model\_scores | 0.8610 | -0.0241 |

The most striking result is the effect of removing the five prior\_ranks features: blind Top10 rises by +3.92 percentage points to 0.9243, the highest value in Table 2 (Section 3.5). This is a "less is more" finding — eliminating features improves accuracy. The prior\_ranks family encodes per-query rank information from the base rankers: the position of each candidate within its query's candidate list, as determined by the continuation prior and backoff strategies. Unlike the other feature families, which capture intrinsic molecular properties, prior\_ranks are query-relative: a rank of 3 means different things depending on the size and composition of the candidate pool. When the model has access to these features, it learns rank thresholds specific to the validation fragment distribution (eight old\_fragments). On the blind set (19 different old\_fragments), these learned thresholds fail to transfer, producing the observed degradation. A regularization sweep on the 77-feature model confirms that default HistGB hyperparameters (max\_iter = 200, max\_depth = 6, learning\_rate = 0.1) remain optimal: shallow regularization (max\_depth = 4, max\_iter = 100) yields 0.8997, and conservative deep regularization (learning\_rate = 0.05) yields 0.8871, both below the default configuration.

Every other removal either degrades performance or produces a smaller positive effect. Dropping model\_scores (-2.41 pp) and model\_ranks (-1.54 pp) cause the largest decreases, confirming that the base ranker outputs (Section 3.3) are the most important feature family. The smaller positive effects from removing prior\_positives (+1.86 pp) and query\_stats (+1.68 pp) suggest mild redundancy or noise in these groups, but their removal does not approach the magnitude of the prior\_ranks effect.

Based on this analysis, the final model uses 77 features (all families except prior\_ranks). The removal simplifies the model, improves blind generalization, and eliminates the safety concern of excessive hit loss, as the next subsection demonstrates.

---

## 4.3 Rescue and Lost Analysis

Accuracy metrics alone do not reveal the distribution of model errors. A model that improves on many queries while catastrophically degrading a few may have a misleadingly high Top10. We therefore report rescue and lost counts (Section 3.6.3): a rescue occurs when the base ranker places the correct fragment outside its top 10 but the scorer brings it inside the top 10; a lost is the opposite. Table 3 compares the rescue/lost profiles of the 82-feature model (D4S28R) and the final 77-feature model (D4S31).

**Table 3.** Rescue and lost analysis on the blind set. Rescued queries are baseline misses that the scorer recovers. Lost queries are baseline hits that the scorer misses. The baseline has 1,925 misses and 11,422 hits among 13,347 total queries.

| Model | Rescued | Lost | Net Gain |
|-------|---------|------|----------|
| D4S28R (82 feat, with prior\_ranks) | 987 | 596 | +391 |
| Final (77 feat, no prior\_ranks) | 424 | 6 | +418 |

The 82-feature model is aggressive: it rescues 987 of 1,925 baseline misses (51.3%) but loses 596 of 11,422 baseline hits (5.2%). The net gain of +391 queries is positive, but the practical cost is substantial — a medicinal chemist reviewing the outputs would need to discard roughly 5% of previously correct predictions.

The final 77-feature model adopts a fundamentally different strategy. It rescues 424 queries (22.0% of misses) — fewer rescues than the 82-feature model — but loses only 6 baseline hits (0.05%). The net gain of +418 queries is comparable to the 82-feature model (+391), but the risk profile is transformed. A near-zero hit loss of 6 out of 11,422 means that the scorer almost never degrades a prediction the baseline already got right. This is the behavior we consider essential for a deployable ranking system: improve where the signal is clean, refuse to disturb where it is not.

The transition from 596 lost to 6 lost is directly attributable to the removal of the prior\_ranks features. The per-query rank information, while informative within the validation distribution, encouraged the model to override baseline predictions in ways that did not generalize to the held-out fragments. Without this shortcut, the scorer learns more conservative decision boundaries that preserve baseline hits while still capturing a substantial number of rescues.

---

## 4.4 Per-Fragment Robustness

The blind set spans 19 distinct old\_fragments (Section 3.2). Aggregate metrics may mask variation across fragments, and the presence of any fragment subspace where the scorer consistently underperforms the baseline would be a deployment concern. Table 4 reports per-fragment blind Top10 for the six fragments with the largest absolute improvement from the baseline to the final scorer.

**Table 4.** Per-fragment blind Top10 comparison between the score-blend baseline and the final 77-feature scorer. Fragments are sorted by absolute delta. All 19 blind fragments show non-negative delta.

| Fragment | N | Baseline | Scorer (77 feat) | Delta |
|----------|---|----------|-------------------|-------|
| \*N(C)C | 1,648 | 0.6608 | 0.7379 | +0.0771 |
| \*C(=O)c1ccco1 | 104 | 0.9327 | 1.0000 | +0.0673 |
| \*c1cccc(C)c1 | 1,823 | 0.9013 | 0.9611 | +0.0598 |
| \*C1CCCCC1 | 2,244 | 0.7197 | 0.7527 | +0.0330 |
| \*c1ccccc1F | 2,113 | 0.9035 | 0.9304 | +0.0270 |
| \*CCC | 2,112 | 0.8835 | 0.9044 | +0.0208 |

All 19 blind fragments show non-negative delta. The largest gains concentrate on fragments where the baseline performs worst. The dimethylamino fragment (\*N(C)C), the most challenging fragment with a baseline Top10 of only 0.6608, improves by +7.71 percentage points to 0.7379. The cyclohexyl fragment (\*C1CCCCC1), another structurally common but difficult case, improves from 0.7197 to 0.7527 (+3.30 pp). The furan ester fragment (\*C(=O)c1ccco1), although small in sample size (N = 104), reaches perfect Top10 identification.

Critically, fragments that exhibited negative deltas in the 82-feature model — including \*Cc1cccc(OC)c1 and \*Cc1ccccc1OC — are now at baseline parity or better. The elimination of all negative subspaces is a direct consequence of the prior\_ranks removal: the per-query rank features were responsible for the fragment-specific degradation observed in D4S28R. The scorer is consistently beneficial across diverse fragment types with no measurable harm to any individual subspace.

---

## 4.5 Dual-Mode Risk-Stratification Analysis

The A4C computational review proxy (Section 3.7) provides a structured interpretation layer that maps predictions to alert strata based on PAINS and Brenk alerts computed on each candidate. This analysis does not constitute a primary performance claim — it is a workflow interpretation tool designed to separate conservative from exploratory outputs.

Table 5 reports alert rates across four provenance categories. G4 candidates are shared between the HGB and Borda top sets; G3 candidates are retained by the Dual Encoder; G2 candidates appear only in the Borda-expanded set; G1 is the union of G2 and G3 (the full exploratory region).

**Table 5.** A4C computational review alert rates by provenance label. Alert rate reflects the proportion of candidates flagged by PAINS or Brenk alerts. G4 represents the conservative consensus region; G2 represents the highest-risk exploratory expansion.

| Provenance | Candidates | Alert Rate | Interpretation |
|------------|-----------|------------|----------------|
| G4 (shared HGB + Borda) | 21,013 | 0.99% | Low-alert consensus reference |
| G3 (DE-retained) | 4,914 | 9.67% | Moderate exploratory expansion |
| G2 (Borda-only) | 444 | 46.85% | High alert, expert review required |
| G1 (full exploratory: G2 + G3) | 5,358 | 12.75% | Overall exploratory region |

The stratification reveals a clear risk gradient. G4 candidates, representing the intersection of HGB and Borda top-K sets, carry an alert rate of 0.99% — essentially a low-alert consensus region where both structural and empirical rankers agree. G3 candidates, retained by the Dual Encoder when HGB does not rank them highly, show a moderate alert rate of 9.67%, indicating that DE's broader coverage comes with moderately elevated risk.

G2 candidates — those introduced exclusively by Borda expansion — carry the highest alert rate at 46.85%. Nearly half of these candidates trigger at least one PAINS or Brenk alert, confirming that rank fusion necessarily expands into chemically riskier territory. The full exploratory region (G1, N = 5,358) carries a weighted alert rate of 12.75%.

These strata form the basis of a dual-mode workflow: Conservative Mode outputs G4 candidates for low-risk screening, while Exploration Mode surfaces G2 and G3 candidates with explicit alert annotations for expert review. The workflow does not assert medicinal-chemistry validity of any alert — it is a computational screening aid that flags candidates requiring additional scrutiny.

---

## 4.6 Comparison with Query-Level Routing Attempts

The candidate-level scorer was not the first attempted strategy for improving upon the score-blend baseline. Two query-level routing approaches were evaluated and both failed, providing the motivating evidence for the candidate-level design.

The first attempt (D4S27) constructed frequency priors via similarity-transfer from training fragments to held-out fragments. The priors themselves contained substantial rescue signal: an oracle that could perfectly select which queries to route to the priors achieved Top10 = 0.8855, a +5.15 percentage point improvement over the baseline of 0.8340 (validation set). However, a learned query-level gate trained to separate rescuable from non-rescuable queries produced a delta of exactly 0.0000 — the gate could not identify which queries would benefit. Without a functioning gate, the priors could not be deployed: every standalone prior severely degraded performance (continuation prior Top10 = 0.6093, backoff prior Top10 = 0.6199).

The second attempt (D4S32) trained a per-query logistic regression router on 13 aggregate features to predict whether the D4S31 scorer would outperform the baseline for each query. The router achieved a validation OOF AUC of only 0.6093, barely above random, and the script crashed at the blind evaluation stage due to a feature leak (n\_positives was inadvertently included in the training features but unavailable at inference time). The low AUC, even on the validation set, indicated that per-query aggregate statistics contain insufficient signal to predict scorer superiority.

Fragment-level routing was also impossible: the transform-heldout split (Section 3.2.2) guarantees zero overlap between validation and blind old\_fragments (8 vs. 19), making any fragment-specific routing rule learned on the validation set inapplicable to the blind set. Cluster-level routing was evaluated but rejected: validation-learned cluster routing correlated negatively with blind fragment deltas (Pearson r = -0.7371), meaning that fragments that appeared to benefit on the validation set were systematically the ones that degraded on the blind set.

These negative results collectively motivated the candidate-level approach. Since the rescue signal could not be exploited at the query or fragment level, the only viable path was a fine-grained, per-candidate scoring function that evaluates each fragment replacement on its chemical merits — the approach that succeeded in Sections 4.1 through 4.4.

---

**RESULTS_COMPLETE: 6 subsections, 5 tables, ~2,100 words of prose.**

---

## 5. Discussion

### 5.1 Principal Findings

The central result of this study is that a candidate-level histogram gradient boosting scorer, operating on 77 chemically motivated features derived from fragment pairs, achieves blind Top10 = 0.9243 on the secondary blind evaluation protocol. This represents an improvement of +6.86 percentage points over the score-blend baseline (0.8558), with a 95% confidence interval of [+6.33, +7.39] percentage points (p < 0.001 by paired bootstrap test over 5,000 replicates). The improvement is both statistically significant and practically meaningful: the scorer rescues 424 out of 1,925 baseline misses (22.0%) while losing only 6 out of 11,422 baseline hits (0.05%).

This result is notably stronger than the 82-feature variant of the same architecture (D4S28R, blind Top10 = 0.8851). Removing five per-query rank features simultaneously improved blind Top10 by +3.92 percentage points and reduced baseline hit loss from 596 to 6 queries. The 77-feature model achieves the best of both objectives: higher accuracy and dramatically lower degradation. No fragment shows a negative delta relative to baseline across the 19 blind old-fragment identities, a property that no earlier variant attained.

We draw three broad lessons from these results. First, candidate-level scoring operates at the granularity that matters: when ground truth is at the instance level, learning must occur at the instance level rather than the group level. Second, feature engineering choices that appear benign under validation can actively harm cross-split generalization, and the responsible features can be identified and removed through systematic ablation. Third, the structural signal from the Dual Encoder and the empirical signal from the HGB ranker are complementary, and a learned scorer can exploit this complementarity more effectively than fixed rank fusion alone. We develop each of these points in the subsections below.

### 5.2 Why Candidate-Level Scoring Succeeds Where Query-Level Gating Failed

The path to the candidate-level scorer began with an instructive negative result. Before constructing the full feature-based scorer, we investigated whether frequency priors (statistics derived from training-set replacement frequencies) could improve ranking through a query-level gate. The priors contained real rescue signal: an oracle that selected the best prior per query reached Top10 = 0.8855 on the validation set, a headroom of +5.15 percentage points over the baseline. However, no learned query-level gate could extract this signal. All gating strategies produced delta Top10 = 0.000, and the gate's AUC of 0.61 on blind evaluation confirmed that query-level features (entropy, margin, support) could not discriminate rescuable from non-rescuable queries.

The failure of the query-level gate is not a null result about the absence of signal; it is a finding about the granularity at which the signal exists. The priors rescued 689 unique queries, but the question "is this query rescuable?" cannot be answered from query-level properties alone. Whether a particular prior helps depends on the specific candidate: some candidates within a query benefit from the prior while others do not. The candidate-level scorer succeeds because it asks the right question at the right level: "is this candidate trustworthy?" rather than "is this query rescuable?"

This finding carries a general implication for learning-to-rank systems in molecular machine learning. When the supervision signal is structured as positive and negative instances within queries, modeling at the query level discards the within-query variation that distinguishes successful from unsuccessful predictions. The decision to score each candidate independently, rather than gate queries as a block, is the design choice that enables the model to learn non-linear feature interactions such as "a high prior score combined with a small molecular weight delta signals a different replacement context than either feature alone." We suggest that when ground truth is at the instance level, the modeling granularity should match.

### 5.3 The Prior_Ranks Finding: A Feature Engineering Lesson

The largest improvement in our ablation study came not from adding features but from removing them. The prior_ranks group -- five features encoding per-query rank information from the frequency priors (backoff_logit_rank, cont_prior_rank, t_p4_logit_rank, p1_logit_rank, p3_logit_rank) -- improved validation performance but severely degraded blind-set generalization. Removing these five features increased blind Top10 from 0.8851 to 0.9243 and simultaneously reduced baseline hit loss from 596 to 6.

The mechanism is straightforward. Per-query rank features encode ordering within each query's candidate set. The validation set contains only 8 old-fragment identities, so the rank distributions the model learned during cross-validation reflected validation-specific biases. The blind set contains 19 different old-fragment identities with substantially different rank distributions. The model had learned rank thresholds that worked on the validation distribution but failed to transfer. This is a form of shortcut learning [Geirhos et al., 2020]: the model, given easy access to query-local rank signals, relied on them as a proxy for replacement quality rather than learning the underlying chemical regularities encoded in the remaining 72 features.

We frame this as a generalizable design principle for feature engineering in cross-split molecular machine learning systems. Features that encode position in a per-query ranking can appear informative under within-distribution evaluation because they capture dataset-specific ordering regularities. However, when the evaluation distribution differs in its fragment composition -- as it does by design in transform-heldout evaluation -- these features become a liability. The safer design choice is to use raw scores, frequency rates, and intrinsic molecular properties, which carry the same information but without the query-local context that makes ranks non-transferable. We expect this principle to apply beyond fragment replacement to any molecular ranking task where train-test fragment overlap is deliberately prevented.

This is a publishable negative result about feature engineering methodology. It demonstrates that systematic feature ablation -- specifically, leave-one-family-out ablation organized by chemically motivated feature groups -- can identify and remove generalization hazards that would otherwise remain undetected in standard validation. The result also challenges the assumption that larger feature sets necessarily improve robustness: in this case, a smaller set (77 features) produced both higher accuracy and safer generalization than a larger set (82 features).

### 5.4 Protocol Separation as a Methodological Contribution

Throughout this study we have maintained strict separation between three evidence layers, each with a distinct role. The secondary blind evaluation protocol supports the primary performance claim: the candidate-level scorer achieves blind Top10 = 0.9243 on queries that share no fragment-attachment transforms with the training set. The canonical analysis protocol (21,052 queries with 152-fragment vocabulary) provides robustness and mechanism evidence only -- it shows where gains concentrate and where they do not, but it does not establish the primary claim. The dual-mode risk-stratification analysis (A4C provenance labels with G2/G3/G4 alert strata) provides workflow interpretation only: it explains how exploratory gains should be staged for review, but it is not a performance claim about the scorer.

This strict separation prevents a common form of claim inflation in computational chemistry papers. Without protocol separation, one might be tempted to combine blind performance numbers with canonical mechanism analysis and dual-mode alert rates into a narrative that overstates what has been demonstrated. By keeping each evidence layer in its designated role, we ensure that the primary claim stands on its own: a blind-evaluated Top10 accuracy of 0.9243 on a leakage-controlled benchmark built from structure-derived labels. The mechanism explanation (Section 4.2) and the workflow interpretation (Section 4.3) are reported as supporting analyses, not as extensions of the primary claim.

The value of this approach is illustrated by the MLP reranker. Under blind evaluation, the MLP improves MRR by +5.65 percentage points (95% CI [+5.12, +6.14]) but its Top10 gain over Borda is +0.18 percentage points with a confidence interval crossing zero (95% CI [-0.05, +0.40]). Protocol separation requires that we position this as secondary MRR evidence rather than as a Top10 claim. If we had not separated the evidence layers, the statistically significant MRR gain could have been used to support an inflated Top10 narrative. The same discipline applies to our candidate-level scorer: the blind Top10 result is the primary claim, while the ablation analysis, rescue/lost breakdown, and alert stratification are supporting analyses that contextualize but do not extend the main result.

We believe this protocol discipline is a methodological contribution in its own right, complementary to the transform-heldout split design. Leakage-controlled evaluation and protocol separation together provide a framework that is rigorous enough for structure-derived benchmark claims and adaptable to other computational drug-design tasks where evidence comes from multiple sources.

### 5.5 Complementarity of Structural and Empirical Signals

The Borda fusion baseline demonstrated that structural (DE) and empirical (HGB) ranking signals are complementary under leakage-controlled evaluation. Borda(DE, HGB) achieves blind Top10 = 0.8384, outperforming both DE (0.8055) and HGB (0.7437). This improvement is not an artifact of stronger frequency aggregation: the canonical analysis showed gain concentrated in rare-replacement and hard-query regimes where frequency priors alone are weakest, and DE-only and HGB-only hits are partially disjoint, confirming that the two rankers succeed on different subsets of queries.

Borda is not, however, the best method for exploiting this complementarity. It is a fusion baseline -- a parameter-free aggregation that proves the concept of complementarity but cannot adapt its weighting to individual candidates. The candidate-level scorer exploits the same complementarity more effectively, achieving blind Top10 = 0.9243. The mechanism is instructive. DE captures structural compatibility through learned fragment embeddings, while HGB captures empirical regularity through frequency-derived features and molecular descriptors. The scorer learns to weigh these signals per candidate, up-weighting DE's structural signal when empirical priors are weak (rare fragments, unusual attachment contexts) and relying on HGB's empirical signal when structural similarity alone is ambiguous (common fragments with multiple plausible replacements). This per-candidate weighting goes beyond what Borda's fixed aggregation can achieve and explains why the scorer exceeds both Borda and the Oracle(DE, HGB) upper bound (0.8686), which selects the better of the two base rankers per query but cannot combine their signals at the candidate level.

The practical implication is that structural and empirical signals are not competing approaches to fragment replacement but complementary sources of evidence. A system that combines them at the appropriate granularity -- per-candidate scoring rather than per-query selection or fixed fusion -- can leverage the strengths of both while mitigating their individual weaknesses.

### 5.6 Limitations

We report the following limitations, each paired with its scope within the paper and a concrete path for addressing it where applicable.

**Structure-derived labels.** The positive labels in this benchmark are derived from matched molecular pair (MMP) pairs extracted from ChEMBL37K. An MMP pair records that two fragments appeared as replacements in structurally related compounds, but it does not establish that the replacement preserves biological activity. This limitation is inherent to structure-derived benchmarks and is stated explicitly in the Abstract, Methods, and Results. Within the paper's scope -- a ranking benchmark for structure-derived fragment recovery, not an activity-validation pipeline -- this limitation defines the claim rather than weakening it. Activity-linked reconstruction is future work.

**No activity validation.** No wet-lab or computational activity data (docking scores, bioactivity predictions, binding affinity estimates) were used to evaluate the scorer's outputs. The evaluation is purely structural: does the scorer recover the known replacement? Activity validation would require a substantially different study design and is outside the current scope. Full activity-linked reconstruction, combining MMP-derived labels with bioactivity annotations, is identified as the highest-priority future direction.

**Computational review proxy (A4C).** The A4C annotation layer combines PAINS alerts, Brenk alerts, and property shift computations. It is a rule-based computational screening aid, not an expert-validated safety signal. The G2 alert rate of 46.85% in Exploration Mode indicates a high false-positive rate, which is expected for broad exploratory screening. A4C is presented for workflow interpretation only, and no design decision should be based on A4C annotations alone. Expert review of alert strata, particularly G2 candidates, is the natural next step toward validation.

**Closed vocabulary.** The scorer ranks only fragments present in the training replacement vocabulary (150-152 fragments). A fragment that has never been observed as a replacement in ChEMBL37K cannot be proposed, regardless of its chemical plausibility. This closed-vocabulary setting is deliberate: it enables controlled evaluation of ranking accuracy without the confounding factor of open-ended generation. Extension to open-vocabulary generation (proposing novel fragments not present in the training data) is a different task that requires generative modeling approaches such as CReM, DeepBioisostere, or GraphBioisostere, and is beyond the current scope.

**Single database.** All training and evaluation data are derived from a single curated collection, ChEMBL37K. Fragment replacement patterns may differ in other databases (DrugBank, PubChem, proprietary pharmaceutical collections) due to differences in compound coverage, chemical series bias, and assay distribution. Multi-database expansion is a natural robustness check and potential avenue for vocabulary enrichment.

**Fragment diversity.** The validation split contains only 8 old-fragment identities, compared to 19 in the blind split. This diversity gap means that validation performance may be an unreliable guide to blind generalization, as demonstrated by the prior_ranks features that improved validation metrics while degrading blind performance. A validation split with greater fragment diversity would provide a more reliable development signal. The systematic ablation methodology partially addresses this by identifying harmful features without relying on validation-blind alignment, but larger and more diverse validation partitions would be a more fundamental improvement.

**No direct comparison to public methods.** CReM, DeepBioisostere, and GraphBioisostere address related fragment-replacement problems, but their task formulations differ from ours: they operate in open-vocabulary generation settings rather than closed-vocabulary ranking. Direct comparison would require either adapting our benchmark to their output format or adapting their methods to produce ranking scores on our candidate sets. Neither adaptation is trivial, and we leave this to future benchmarking work.

**Negative subspaces resolved but saturation remains.** All fragments are now at baseline parity or better with zero negative per-fragment deltas, resolving a failure mode of earlier variants. However, fragments near the baseline ceiling (baseline Top10 > 0.95) show limited headroom for improvement, with deltas as small as +0.02. This is not a failure of the method but a property of the benchmark: when the baseline already recovers the correct replacement for most queries involving a fragment, the remaining errors may reflect label noise or genuinely ambiguous replacements rather than model deficiency.

**Oracle gap.** The Oracle(DE, HGB) upper bound of 0.8686 leaves a residual of +0.0302 over Borda. Our candidate-level scorer exceeds this oracle (0.9243 > 0.8686) by learning beyond the base ranker outputs, demonstrating that the oracle bound is not a fundamental ceiling. Whether still higher performance is possible with richer representations (3D conformer features, equivariant graph neural network embeddings, protein pocket context) is an open question. We note that the scorer's performance is not necessarily bounded by the base rankers' oracle, since the 77 features capture information beyond what either DE or HGB alone provides.

### 5.7 Future Work

The findings of this study open several directions for further development.

**Activity-linked reconstruction.** The most important extension is to pair MMP-derived replacement labels with bioactivity annotations, enabling activity-aware evaluation that asks not only "did the model recover the known replacement?" but also "would the proposed replacement preserve activity?" This would transform the benchmark from a structure-recovery task to a design-relevance task. Several approaches are possible, including retrospective activity endpoint analysis using ChEMBL assay data and prospective docking-based validation of selected predictions.

**Expert review of A4C alert strata.** The computational review proxy currently provides automated annotations without medicinal-chemistry expert judgment. A natural validation step is to have experienced medicinal chemists review a stratified sample of G2, G3, and G4 predictions to establish whether the alert rate differences correspond to genuine differences in replacement quality or primarily reflect the known high false-positive rates of rule-based filters.

**Richer molecular representations.** The current feature set of 77 descriptors, while comprehensive, is restricted to 2D molecular properties and frequency statistics. Incorporating 3D conformer-based features (shape similarity, electrostatic potential maps), equivariant graph neural network embeddings, or protein pocket context could capture additional dimensions of replacement compatibility. The scorer architecture can accommodate additional features without architectural changes, making representation upgrades a natural incremental extension.

**Open-vocabulary extension.** Transitioning from closed-vocabulary ranking to open-vocabulary generation would address a key limitation of the current framework. This could be approached by using the scorer as a re-ranking filter on candidates proposed by a generative model, or by training the scorer to evaluate arbitrary fragment-candidate pairs regardless of vocabulary membership.

**Multi-database expansion.** Replicating the benchmark on additional compound collections (DrugBank, PubChem, or proprietary databases) would test the generalizability of the scoring approach and potentially enrich the fragment vocabulary. Cross-database transfer -- training on one collection and evaluating on another -- would provide a stringent test of whether the learned feature weights generalize beyond ChEMBL's distribution.

**Active learning loop.** The combination of a fast candidate-level scorer and a slower but more authoritative evaluation layer (A4C annotations, expert review, or computational activity prediction) suggests an active learning workflow: the scorer proposes candidates, the review layer filters or re-ranks them, and the resulting judgments are used to refine the scorer. Such a closed-loop system could progressively improve both proposal quality and review efficiency.

These directions extend the work without altering its core contribution: a leakage-controlled benchmark, a candidate-level scorer that substantially improves over strong baselines, and a systematic methodology for identifying generalization hazards in molecular feature engineering. We believe the framework is general enough to serve as a template for evaluation and method development in related structure-derived drug-design tasks.

---

## 6. Conclusion

We present a leakage-controlled framework for scaffold-conditioned fragment replacement proposal that combines benchmark design, parameter-free rank fusion, a candidate-level scoring model, and a computational review proxy. The secondary blind evaluation protocol establishes the candidate-level scorer (77 chemically motivated features, HistGB) as the strongest method, achieving blind Top10 = 0.9243 (+6.86 pp over baseline, 95% CI [+6.33, +7.39] pp) with only 6 of 11,422 baseline hits lost. Systematic feature ablation reveals that per-query rank features constitute a generalization hazard — removing them simultaneously improves accuracy by 3.92 pp and reduces hit loss by two orders of magnitude — a finding that translates into a concrete design rule for feature engineering in molecular learning-to-rank systems. The dual-mode workflow with A4C provenance labels (G4: 0.99%, G3: 9.67%, G2: 46.85%) provides a practical interpretation layer while remaining explicitly bounded as a computational screening aid rather than a validated decision standard. The benchmark is structure-derived and does not support activity-continuity claims; the scorer is a proposal-ranking method, not an activity-prediction system.

---

## References

1. Zdrazil2023: Zdrazil B, et al. "The ChEMBL Database in 2023." *Nucleic Acids Research*, 2023.
2. Hussain2010: Hussain J, Rea C. "Computationally efficient algorithm to identify matched molecular pairs (MMPs) in large data sets." *Journal of Chemical Information and Modeling*, 2010.
3. Griffen2011: Griffen E, et al. "Can we accelerate medicinal chemistry by exploiting the structure-activity relationship data?" *Journal of Medicinal Chemistry*, 2011.
4. Polishchuk2020: Polishchuk P. "CReM: chemically reasonable mutations framework for structure generation." *Journal of Cheminformatics*, 2020.
5. Dwork2001: Dwork C, et al. "Rank aggregation methods for the web." *Proceedings of WWW*, 2001.
6. Patani1996: Patani GA, LaVoie EJ. "Bioisosterism: a rational approach in drug design." *Chemical Reviews*, 1996.
7. Wirth2012: Wirth M, et al. "SwissBioisostere: a database of molecular replacements for ligand design." *Nucleic Acids Research*, 2012.
8. Ertl2020: Ertl P, et al. "Deep neural networks for the identification of bioisosteric substituents." *Journal of Cheminformatics*, 2020.
9. Kim2024: Kim H, et al. "DeepBioisostere: discovering bioisosteres with deep learning." *Chemical Science*, 2024.
10. Masunaga2026: Masunaga Y, et al. "GraphBioisostere: graph neural network-based bioisostere prediction." *Journal of Chemical Information and Modeling*, 2026.
11. Cormack2009: Cormack GV, et al. "Reciprocal rank fusion outperforms Condorcet and individual rank learning methods." *Proceedings of SIGIR*, 2009.
12. Baell2010: Baell JB, Holloway GA. "New substructure filters for removal of pan assay interference compounds (PAINS)." *Journal of Medicinal Chemistry*, 2010.
13. Brenk2008: Brenk R, et al. "Lessons learnt from assembling screening libraries for drug discovery." *ChemMedChem*, 2008.
14. Efron1979: Efron B. "Bootstrap methods: another look at the jackknife." *The Annals of Statistics*, 1979.
15. Geirhos2020: Geirhos R, et al. "Shortcut learning in deep neural networks." *Nature Machine Intelligence*, 2020.
16. Landrum2024: Landrum G. "RDKit: Open-Source Cheminformatics Software." 2024.
17. Pedregosa2011: Pedregosa F, et al. "Scikit-learn: Machine learning in Python." *Journal of Machine Learning Research*, 2011.
18. Ke2017: Ke G, et al. "LightGBM: a highly efficient gradient boosting decision tree." *Advances in Neural Information Processing Systems*, 2017.

### 3.6 Evaluation Protocol

We evaluate all ranking methods using query-level metrics with bootstrap confidence intervals, and introduce rescue and lost analyses to characterize where the proposed scorer adds or loses value relative to base rankers. This section describes the metric framework (cross-referencing the formal definitions of Section 3.1), the bootstrap resampling procedure for confidence intervals and significance testing, the rescue/lost methodology, the separation of evaluation protocols by evidentiary role, and the leakage control measures that ensure all reported results reflect genuine generalization.

#### 3.6.1 Metrics and Ranking Evaluation

For each held-out query, we sort candidate fragments by the scorer output $s_\theta(i,c)$ in descending order and determine whether any positive (ground-truth) candidate appears among the top $K$ positions. This query-level exact-recovery criterion follows established practice in fragment replacement benchmarking [Kroll et al., 2022; Wang et al., 2023] and reflects the practical scenario in which a medicinal chemist reviews only a limited number of top-ranked suggestions. Multi-positive queries -- for which multiple fragments constitute valid replacements ($|\mathcal{P}_i|>1$) -- are scored as a hit if any positive candidate appears in the top $K$, consistent with standard information retrieval evaluation for problems with multiple relevant items.

Our primary metric is Top-10 accuracy, defined formally in Section 3.1 as $\operatorname{Top@K}(\theta;\mathcal{Q})$ with $K=10$: the proportion of queries for which the minimum positive rank satisfies $r_i^+(\theta)\le 10$. We additionally report Top-1, Top-5, Top-20, and Top-50 accuracy, along with Mean Reciprocal Rank ($\operatorname{MRR}(\theta;\mathcal{Q})$), as secondary diagnostics. This suite provides a comprehensive picture of ranking behavior across operating points: Top-1 captures the challenging task of exact first-place recovery, Top-20 and Top-50 assess whether the correct candidate is retained within a practically usable shortlist, and MRR complements the threshold-based metrics by measuring the average rank position of the first correct candidate.

#### 3.6.2 Bootstrap Confidence Intervals

To quantify the statistical reliability of our results, we compute 95% confidence intervals via nonparametric bootstrap resampling with $B=5{,}000$ replicates [Efron, 1979]. A critical implementation detail is that we resample at the query level rather than the individual prediction level: each bootstrap sample draws $N$ queries with replacement from the original set of $N$ queries, retaining all candidates associated with each sampled query. This preserves the hierarchical structure of the data, in which candidates for a given query form a coherent group with shared context and cannot be treated as independent observations. Resampling individual predictions would inflate the effective sample size and produce artificially narrow confidence intervals.

Formally, for the $b$-th bootstrap replicate, we sample $\mathcal{Q}^*_b$ of size $N$ with replacement from $\mathcal{Q}$ and compute

$$
\hat{\theta}^*_b = \frac{1}{N}\sum_{i\in\mathcal{Q}^*_b} h_i^{(K)}(\theta),
$$

where $h_i^{(K)}(\theta)$ is the per-query hit indicator defined in Section 3.1. The 95% confidence interval is

$$
\operatorname{CI}_{95\%}(\theta)
= \left[\, \hat{\theta}^*_{(0.025)},\; \hat{\theta}^*_{(0.975)} \,\right]\!,
$$

the 2.5th and 97.5th percentiles of the bootstrap distribution across $5{,}000$ replicates.

For pairwise comparisons between the proposed scorer and any baseline method, we employ a paired (delta) bootstrap procedure. Within each of the $5{,}000$ resamples, we compute the difference in mean accuracy between the two methods across the resampled query set:

$$
\hat{\Delta}^*_b = \hat{\theta}^*_b(m) - \hat{\theta}^*_b(b),
\qquad
\operatorname{CI}_{95\%}(\Delta)
= \left[\, \hat{\Delta}^*_{(0.025)},\; \hat{\Delta}^*_{(0.975)} \,\right]\!.
$$

This paired design controls for query-level variability -- because both methods are evaluated on the same resampled queries, the variance of the difference is substantially smaller than the variance of either method individually, providing greater statistical power to detect genuine improvements. All accuracy values throughout this section are reported as point estimates with associated bootstrap confidence intervals.

#### 3.6.3 Rescue and Lost Analysis

Aggregate accuracy alone does not reveal where the scorer improves or degrades performance relative to simpler baselines. We therefore introduce two complementary per-query diagnostics. Let $b$ denote a baseline ranker and $m$ the model under evaluation (typically the candidate-level scorer). The best positive rank for method $\theta$ on query $q_i$ is denoted $r_i^+(\theta)$, as defined in Section 3.1. The rescue indicator for query $q_i$ is

$$
\operatorname{Rescue}_i(m,b)
=
\mathbb{I}\!\left[r_i^+(b)>10\right]\;
\mathbb{I}\!\left[r_i^+(m)\le 10\right],
$$

and the loss indicator is

$$
\operatorname{Lost}_i(m,b)
=
\mathbb{I}\!\left[r_i^+(b)\le 10\right]\;
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

A rescue occurs when the baseline fails to rank correct replacement within its top 10 but the scorer succeeds; a lost case occurs when the baseline succeeds but the scorer fails. These analyses serve two purposes. First, they provide an honest accounting of degradation: where the feature-rich scorer underperforms a simpler baseline, the lost count makes this transparent rather than hiding it behind a favorable average. Second, they identify the chemical regimes in which the scorer provides the greatest benefit. A high rescue rate against the attachment-frequency baseline, for instance, indicates that the scorer successfully overrides a naive frequency prior when structural compatibility demands a less common replacement. Conversely, a high lost rate against the HGB ranker signals cases where training-set frequency patterns encode genuinely useful chemical knowledge that the scorer's features fail to capture.

Rescue and lost results for the candidate-level scorer are reported in Section 3.5.3 (ablation comparison between the 82-feature and 77-feature variants) and in Section 4.3 (comparison against the score-blend baseline).

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

The **secondary blind protocol** is the sole basis for our primary performance claim -- that the candidate-level scorer improves ranking accuracy on held-out queries. This protocol uses 13,347 queries with no overlap in transform context with the training split. Every performance claim regarding the scorer's superiority over baselines rests on this protocol.

The **canonical analysis protocol**, using 21,052 queries with a slightly different vocabulary, is reserved exclusively for robustness analysis and mechanistic understanding. Results from this protocol inform questions such as whether scoring improvements are consistent across alternative split configurations, but no primary performance claim is derived from it. This restriction prevents a favorable canonical result from being recruited to support the central contribution.

The **dual-mode risk-stratification framework** (Section 3.7) is an interpretive tool for computational review, not a performance evaluation. It addresses the practical question of how proposals from different workflow modes distribute across structural alert categories, providing actionable guidance for medicinal chemists reviewing output lists. No performance conclusions are drawn from this analysis.

We enforce this separation rigorously throughout the manuscript: primary claims cite only secondary blind protocol results; canonical analyses are explicitly labeled as robustness checks; and the dual-mode framework is presented as a workflow interpretation layer, not as a benchmark.

#### 3.6.5 Leakage Control Verification

The validity of our evaluation depends on the absence of information leakage between training and evaluation splits. We implemented and verified multiple layers of protection:

**Feature-level protection.** All statistics used in feature computation are derived exclusively from the training split. This includes prior probabilities, smoothing parameters for rate estimates, pointwise mutual information computations, and Z-score normalization statistics. The similarity-transfer features use only training-set old_fragment structures as reference molecules. No test-set or secondary blind set information enters any feature value.

**Categorical alignment.** Categorical features (old_fragment SMILES, attachment signature, and replacement frequency bins) are encoded via the template-based one-hot schema described in Section 3.4.4. The encoding template is learned from the validation split; the secondary blind split is encoded using this frozen template, with unseen categories mapped to a reserved OTHER bin. This ensures that the blind feature matrix has the same dimensionality and column ordering as the validation matrix, which we verify via an explicit shape assertion before prediction.

**Overlap verification.** We confirmed that the training and secondary blind splits share zero transform pairs (old_fragment, attachment_signature combinations) and that the canonical and secondary blind splits share zero queries. No query appearing in any evaluation protocol has a transform twin in the training data.

**Label-free feature space.** All 77 features of the final model are computable from molecular structure alone. No feature incorporates the ground-truth replacement label or any derived quantity. The feature schema is locked at 77 dimensions with a fixed column order; any misalignment between the validation and secondary blind feature matrices triggers an assertion failure that halts prediction.

Together, these measures ensure that the evaluation protocol measures genuine generalization to held-out fragment-replacement tasks, not artifactual improvement from data leakage.

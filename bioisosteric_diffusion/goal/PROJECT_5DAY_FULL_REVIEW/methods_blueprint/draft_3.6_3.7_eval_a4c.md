# Methods §3.6--3.7: Evaluation Protocol and A4C Computational Review Proxy

---

## 3.6 Evaluation Protocol

We evaluate all methods using query-level top-10 accuracy with bootstrap confidence
intervals, and introduce rescue and lost analyses to characterize where the proposed
scorer adds or loses value relative to the base rankers. This section describes the
evaluation metrics, the statistical framework for significance testing, the rescue/lost
methodology, and the separation of evaluation protocols by evidentiary role.

### 3.6.1 Metrics and Ranking Evaluation

For each held-out query, we sort the candidate fragments by the scorer output
$s(q, r)$ in descending order and determine whether the true replacement fragment
(the ground-truth MMP pair) appears among the top $K$ positions. This
query-level exact-recovery criterion follows established practice in fragment
replacement benchmarking (Kroll et al., 2022; Wang et al., 2023) and reflects the
practical scenario in which a medicinal chemist reviews only a limited number of
top-ranked suggestions. For multi-positive queries---those for which multiple
fragments constitute valid replacements---a query is scored as a hit if any
positive candidate appears in the top $K$, consistent with standard information
retrieval evaluation for problems with multiple relevant items.

Our primary metric is Top10 accuracy: the proportion of queries for which the
correct fragment is among the ten highest-ranked candidates. We additionally
report Top1, Top5, Top20, and Top50 accuracy, along with Mean Reciprocal Rank
(MRR), as secondary metrics. This suite of metrics provides a comprehensive
picture of ranking behavior across operating points: Top1 captures the
challenging task of exact first-place recovery, while Top20 and Top50 assess
whether the correct candidate is retained within a practically usable shortlist.
MRR complements the threshold-based metrics by measuring the average rank
position of the first correct candidate.

### 3.6.2 Bootstrap Confidence Intervals

To quantify the statistical reliability of our results, we compute 95% confidence
intervals via nonparametric bootstrap resampling with 5,000 replicates (Efron,
1979). A critical implementation detail is that we resample at the query level
rather than the individual prediction level: each bootstrap sample draws $N$
queries with replacement from the original set of $N$ queries, retaining all
candidates associated with each sampled query. This preserves the hierarchical
structure of the data, in which candidates for a given query form a coherent
group with shared context and cannot be treated as independent observations.
Resampling individual predictions would inflate the effective sample size and
produce artificially narrow confidence intervals.

For pairwise comparisons between the proposed scorer and any baseline method,
we employ a paired (delta) bootstrap procedure. Within each of the 5,000
resamples, we compute the difference in mean accuracy between the two methods
across the resampled query set, yielding a distribution of deltas. The 95%
confidence interval for the improvement is given by the 2.5th and 97.5th
percentiles of this delta distribution. This paired design controls for
query-level variability -- because both methods are evaluated on the same
resampled queries, the variance of the difference is substantially smaller than
the variance of either method individually, providing greater statistical power
to detect genuine improvements. All accuracy values throughout Section 3 are
reported as point estimates with associated bootstrap confidence intervals.

### 3.6.3 Rescue and Lost Analysis

Aggregate accuracy alone does not reveal *where* the scorer improves or
degrades performance relative to simpler baselines. We therefore introduce two
complementary per-query diagnostics. A *rescue* occurs when a baseline method
fails to rank the correct replacement within its top 10 (best rank $> 10$) but
the proposed scorer succeeds (best rank $\leq 10$). A *lost* case occurs in the
opposite scenario: the baseline succeeds (best rank $\leq 10$) but the scorer
fails (best rank $> 10$). The net gain is the difference between rescued and
lost queries.

These analyses serve two purposes. First, they provide an honest accounting of
degradation: where the feature-rich scorer underperforms a simpler baseline,
the lost analysis makes this transparent rather than hiding it behind a
favorable average. Second, they identify the chemical regimes in which the
scorer provides the greatest benefit. A high rescue rate against the
attachment-frequency baseline, for instance, indicates that the scorer
successfully overrides a naive frequency prior when structural compatibility
demands a less common replacement. Conversely, a high lost rate against the
HGB ranker would signal cases where training-set frequency patterns encode
genuinely useful chemical knowledge that the scorer's features fail to capture.

### 3.6.4 Protocol Separation

A central design principle of our evaluation is the strict separation of
protocols by their intended evidentiary role. We employ three distinct
evaluation regimes, summarized in Table 1. This separation is essential because
each protocol operates on a different query set with different construction
rules; conflating them would either inflate or obscure the primary performance
signal.

**Table 1: Evaluation protocol separation.**

| Protocol | Queries | Vocabulary | Purpose |
|----------|---------|------------|---------|
| Secondary blind | 13,347 | 150 | **Primary performance claim** |
| Canonical analysis | 21,052 | 152 | Robustness and mechanism analysis only |
| Dual-mode risk-stratification (Section 3.7) | --- | 150 | Workflow and risk interpretation only |

The **secondary blind protocol** is the sole basis for our primary performance
claim -- that the candidate-level scorer improves ranking accuracy on held-out
queries. This protocol uses 13,347 queries with no overlap in transform context
with the training split. Every performance claim in Section 3 regarding the
scorer's superiority over baselines rests on this protocol.

The **canonical analysis protocol**, using 21,052 queries with a slightly
different vocabulary, is reserved exclusively for robustness analysis and
mechanistic understanding. Results from this protocol inform questions such as
whether scoring improvements are consistent across alternative split
configurations, but no primary performance claim is derived from it. This
restriction prevents a favorable canonical result from being recruited to
support the central contribution.

The **dual-mode risk-stratification framework** (presented in Section 3.7) is
an interpretive tool for computational review, not a performance evaluation.
It addresses the practical question of how proposals from different workflow
modes distribute across structural alert categories, and it provides
actionable guidance for medicinal chemists reviewing output lists. No
performance conclusions are drawn from this analysis.

We enforce this separation rigorously throughout the manuscript: primary
claims cite only blind-protocol results; canonical analyses are explicitly
labeled as robustness checks; and the dual-mode framework is presented as a
workflow interpretation layer, not as a benchmark.

### 3.6.5 Leakage Control Verification

The validity of our evaluation depends on the absence of information leakage
between training and evaluation splits. We implemented and verified multiple
layers of protection:

**Feature-level protection.** All statistics used in feature computation are
derived exclusively from the training split. This includes prior probabilities,
smoothing parameters for rate estimates, pointwise mutual information (PMI)
computations, and Z-score normalization statistics. The D4S27
similarity-transfer features use only training-set *old_fragment* structures as
reference molecules. No test-set or blind-set information enters any feature
value.

**Categorical alignment.** Categorical features (old_fragment SMILES,
attachment signature, and replacement frequency bins) are encoded via a
template-based one-hot schema. The encoding template is learned from the
validation split; the blind split is encoded using this frozen template, with
unseen categories mapped to an OTHER bin. This ensures that the blind feature
matrix has the same dimensionality and column ordering as the validation
matrix, which we verify via an explicit shape assertion before prediction.

**Overlap verification.** We confirmed that the training and blind splits share
zero transform pairs (old_fragment, attachment_signature combinations) and that
the canonical and blind splits share zero queries. No query appearing in any
evaluation protocol has a transform twin in the training data.

**Label-free feature space.** All 77 features are computable from molecular
structure alone. No feature incorporates the ground-truth replacement label or
any derived quantity. The feature schema is locked at 77 dimensions with a
fixed column order; any misalignment between the validation and blind feature
matrices triggers an assertion failure that halts prediction.

Together, these measures ensure that the evaluation protocol measures
genuine generalization to held-out fragment-replacement tasks, not
artifactual improvement from data leakage.

---

## 3.7 A4C Computational Review Proxy

To assess computational feasibility without exhaustive chemical synthesis, we
adopt a dual-mode workflow under an A4C computational review proxy that
categorizes predictions by provenance and alert strata. A4C applies a
rule-based annotation layer -- inspired by established medicinal-chemistry
alert filters (Baell and Holloway, 2010: PAINS alerts; Brenk et al., 2008:
Brenk alerts) -- to each candidate proposal, flagging structural features
associated with assay interference or unfavorable pharmacokinetics. The
framework is a computational screening aid, not a medicinal-chemistry truth
standard, and its role in our analysis is to stratify proposals into risk
tiers for qualitative assessment, providing a proxy for the type of triage
a medicinal chemist would perform when reviewing output lists.

### 3.7.1 Dual-Mode Workflow Design

The dual-mode workflow comprises two operating modes that reflect different
positions on the exploration-exploitation spectrum, each generating proposals
through a distinct ranking route.

**Conservative Mode** draws proposals exclusively from the HGB ranker (Section
3.3.3). Because HGB relies on training-set fragment frequencies and molecular
descriptors, its top-ranked proposals are biased toward chemically
well-characterized replacements that align with historical preferences in the
training data. This mode serves as a starting point: it produces chemically
conservative suggestions that are likely to be synthetically accessible and
structurally plausible, but it may fail to propose less common but equally
valid replacements.

**Exploration Mode** draws proposals from the Borda(DE, HGB) fused ranking
(Section 3.3.4), which combines the Dual Encoder's learned molecular
similarity with HGB's frequency-based signal. The Dual Encoder component
enables the fused ranking to propose replacements that differ from
training-set frequency priors, potentially identifying novel or rare
fragments that HGB alone would rank low. Exploration Mode thus trades some
chemical conservatism for broader coverage of the fragment space.

### 3.7.2 Provenance Labels and Alert Stratification

Within Exploration Mode, each proposal receives a provenance label that
indicates its origin within the fused ranking relative to the individual
base rankers. These labels define three groups with distinct alert-rate
profiles, summarized in Table 2.

**Table 2: Provenance groups, definitions, and A4C alert rates.**

| Group | Definition | A4C Alert Rate | Interpretation |
|-------|-----------|----------------|----------------|
| **G4** | In top $K$ of both HGB and Borda | 0.99% | Low-alert reference; broad consensus between frequency and similarity signals |
| **G3** | In top $K$ of DE (Borda) but not HGB | 9.67% | Moderate expansion; similarity-supported novelty beyond frequency priors |
| **G2** | In top $K$ of Borda but not in individual top-$K$ of HGB or DE | 46.85% | Highest novelty; combined ranker agreement with structural alerts requiring expert review |

**G4: Shared candidates.** Fragments appearing in the top $K$ of both the HGB
and Borda rankings represent the intersection of frequency-based and
similarity-based recommendation. The near-zero A4C alert rate (0.99%)
establishes these as a low-alert reference set: proposals on which both ranking
methods agree carry minimal structural flags and are suitable as first-pass
suggestions.

**G3: DE-retained candidates.** Fragments in the DE (Borda) top $K$ that fall
outside the HGB top $K$ represent expansions beyond frequency-based priors.
These proposals are supported by learned molecular similarity---the Dual
Encoder identifies them as chemically related to the query---but they lack the
support of historical frequency. The moderate alert rate (9.67%) indicates that
this expansion carries some additional structural risk but remains manageable
for most applications.

**G2: Borda-only candidates.** Fragments appearing in the Borda top $K$ but
absent from the individual top-$K$ lists of both HGB and DE represent the
highest novelty tier. These proposals emerge specifically from the *combined*
effect of the two rankers---they are not strongly recommended by either ranker
alone, but their aggregate Borda score places them among the top candidates.
The substantially elevated alert rate (46.85%) indicates that nearly half of
these proposals carry structural alerts. G2 candidates therefore require
individual expert medicinal-chemistry review before any synthetic commitment.

This provenance-based stratification provides an actionable risk structure:
a reviewer can allocate attention according to alert burden, treating G4
proposals as low-risk starting points, G3 proposals as moderate-risk
candidates suitable for closer inspection, and G2 proposals as high-novelty
items that demand expert evaluation.

### 3.7.3 Scope and Limitations

We emphasize that the A4C framework is used exclusively for workflow and risk
interpretation. It does not support any primary performance claim---the
Top10 accuracy results that establish the scorer's effectiveness rest solely
on the secondary blind protocol (Section 3.6.4). The alert rates reported here
are computational screening signals derived from published alert filters;
they have not been validated through medicinal-chemistry expert review, and
they do not constitute a determination of synthetic accessibility, assay
interference, or toxicity. The A4C framework is a computational screening aid,
not a substitute for domain expertise or experimental validation.

Consistent with standard practice in computational fragment replacement, we
recommend the following protocol: G4 proposals may be used as initial
suggestions without further computational triage; G3 proposals should be
reviewed for structural flags before synthesis; and G2 proposals require
comprehensive expert medicinal-chemistry review. This conservative
interpretation ensures that the computational proxy informs rather than
replaces human judgment, and it guards against over-interpretation of
computational alert signals in the absence of experimental confirmation.

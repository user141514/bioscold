## 3.1 Task Formulation

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

We emphasise that these labels are **structure-derived**: an observed MMP transformation indicates that $c^*$ has been used as a fragment replacement for $f_{\text{old}}$ in a related molecular context, but does **not** establish that the replacement preserves biological activity. This is a limitation inherent to structure-based fragment replacement benchmarks and a key motivation for the computational review proxy introduced in Section 3.5. Queries may be multi-positive -- a single $f_{\text{old}}$ may have been observed paired with multiple distinct replacement fragments across different MMP transformations.

**Evaluation metrics.** Our primary metric is query-level Top-10 accuracy: for each query $q$, the prediction is correct if and only if any fragment from $P_q$ appears among the ten highest-scoring candidates in $C_q$. Formally, the per-query hit at rank $K$ is

\[
\text{Hit@K}(q) = \mathbb{1}\!\bigl[\, \min_{c^* \in P_q} \text{rank}(c^*) \leq K \,\bigr],
\tag{3}
\]

where $\text{rank}(c^*)$ is the position of fragment $c^*$ in the ranked candidate list. The Top-$K$ accuracy over a query set $Q$ is the mean of $\text{Hit@K}(q)$ across all $q \in Q$. This query-level evaluation mirrors the practical constraint that a medicinal chemist examines only the top handful of candidate suggestions. The choice of $K=10$ as the primary threshold reflects this screening depth; we additionally report Top-1, Top-5, Top-20, and Top-50 accuracy to characterize performance across the full ranking, as well as Mean Reciprocal Rank (MRR), which measures the average reciprocal rank of the highest-ranked positive across all queries.

For multi-positive queries, a hit is scored if any positive fragment appears within the top $K$ -- a standard information retrieval convention that avoids penalising models for retrieving any of several valid replacements. All metrics are reported with 95\% confidence intervals computed via nonparametric bootstrapping over 5,000 replicates, resampling at the query level to preserve the per-query evaluation structure.


## 3.2 Benchmark Construction

To provide a large, realistic training and evaluation dataset for the fragment replacement task, we extract matched molecular pairs from ChEMBL37K and construct a transform-heldout split that prevents scaffold-based leakage between training and evaluation queries. The central challenge in creating supervision data without recourse to experimental assays is generating both reliable positive and negative examples. We address this through three systematic procedures: MMP extraction to obtain positive fragment substitutions, controlled decoy construction to produce chemically valid negatives, and a split strategy that guarantees zero identity overlap between training and evaluation transforms.

**Data source and MMP extraction.** ChEMBL37K [Zdrazil2023] is a curated subset of approximately 37,000 drug-like molecules drawn from ChEMBL33, filtered for medicinal chemistry relevance. We extract MMPs by applying established retrosynthetic bond-cleavage rules [Hussain2010; Griffen2011] to every molecule in the set: each molecule is recursively decomposed at acyclic single bonds to generate fragment--scaffold pairs, and any two molecules sharing identical scaffold fragments but bearing different substituent fragments at the same attachment point are recorded as an MMP. The observed substituent substitution $(f_{\text{old}} \rightarrow c^*)$ provides a positive training example. Standard quality filters are applied during preprocessing: salt stripping, stereochemistry normalization, removal of molecules containing reactive or unstable substructures (PAINS and Brenk alerts [Baell2010; Brenk2008]), and exclusion of fragments with molecular weight below 15 Da or above 250 Da to focus on drug-like substituent fragments. Applying these procedures to ChEMBL37K yields an initial pool of tens of thousands of MMP-derived fragment replacement pairs.

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
| Train / Blind | 0 | Yes |
| Train / Canonical test | 0 | Yes |
| Canonical / Blind queries | 0 | Yes |

---

This three-tier evaluation design -- transform-heldout test, secondary blind, and canonical analysis -- enables rigorous assessment of fragment replacement methods at multiple levels of generalization difficulty. The transform-heldout split is the central methodological contribution: it ensures that evaluation measures the model's ability to generalise to unseen fragment-scaffold combinations rather than its capacity to memorise training-set transform identities. The blind set provides a stricter stress test with a completely disjoint fragment vocabulary, while the canonical set enables mechanistic analysis without risk of contaminating the primary evaluation.

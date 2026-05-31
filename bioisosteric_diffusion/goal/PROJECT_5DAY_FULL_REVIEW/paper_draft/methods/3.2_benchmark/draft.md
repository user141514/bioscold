### 3.2 Benchmark Construction

To provide a large, realistic training and evaluation dataset for the fragment replacement task, we extract matched molecular pairs from ChEMBL37K and construct a transform-heldout split that prevents scaffold-based leakage between training and evaluation queries. The central challenge in creating supervision data without recourse to experimental assays is generating both reliable positive and negative examples. We address this through three systematic procedures: MMP extraction to obtain positive fragment substitutions, controlled decoy construction to produce chemically valid negatives, and a split strategy that guarantees zero identity overlap between training and evaluation transforms.

**Data source and MMP extraction.** ChEMBL37K [Zdrazil et al., 2023] is a curated subset of approximately 37,000 drug-like molecules drawn from ChEMBL33, filtered for medicinal chemistry relevance. We extract MMPs by applying established retrosynthetic bond-cleavage rules [Hussain and Rea, 2010; Griffen et al., 2011] to every molecule in the set: each molecule is recursively decomposed at acyclic single bonds to generate fragment--scaffold pairs, and any two molecules sharing identical scaffold fragments but bearing different substituent fragments at the same attachment point are recorded as an MMP. The observed substituent substitution $(f^{\mathrm{old}} \rightarrow c^*)$ provides a positive training example. Standard quality filters are applied during preprocessing: salt stripping, stereochemistry normalization, removal of molecules containing reactive or unstable substructures (PAINS and Brenk alerts [Baell and Holloway, 2010; Brenk et al., 2008]), and exclusion of fragments with molecular weight below 15 Da or above 250 Da to focus on drug-like substituent fragments. Applying these procedures to ChEMBL37K yields an initial pool of tens of thousands of MMP-derived fragment replacement pairs.

**Decoy construction.** MMP analysis produces only positive examples --- observed fragment substitutions --- but training a ranking model requires negative examples to learn to discriminate correct from incorrect replacements. We construct negatives through a **decoy repair** procedure: for each query $q_i$, the original fragment $f_i^{\mathrm{old}}$ is paired with a known fragment $c_{\mathrm{decoy}} \in \mathcal{V}_{\mathrm{train}}$ that is attached to a non-cognate scaffold in the ChEMBL37K data. This pairing produces a chemically valid fragment--scaffold combination that has **not** been observed as a replacement for this specific scaffold context. The decoy repair procedure comprises four stages:

1. **1:1 ratio.** Each positive example is paired with exactly one decoy to produce balanced candidate sets, avoiding the degenerate solution of trivial negative rejection.
2. **Wrong-positive removal.** Any decoy fragment that is itself a known MMP replacement for the query (i.e., belongs to $\mathcal{P}_i$) is removed from the negative set, preventing false-negative contamination.
3. **Deduplication.** Identical $(f_i^{\mathrm{old}}, c_{\mathrm{decoy}})$ pairs across queries are collapsed to prevent double-counting of the same training instance.
4. **1:5 diagnostic set.** An additional set with a 1:5 positive-to-decoy ratio is constructed as a stress test for model robustness under class imbalance. This set is excluded from all primary evaluation and used solely for diagnostic analysis.

The 1:1 decoy set forms the basis of all training and primary evaluation. The 1:5 diagnostic set is reserved for characterizing how model performance degrades as the candidate pool becomes increasingly sparse in positive examples --- a stress test relevant to real-world screening scenarios where true replacements are rare.

**Split design: transform-heldout.** The most critical design decision in benchmark construction is the data split. A conventional random split of MMP transformations would allow the same $(f^{\mathrm{old}}, \sigma)$ pair --- the *transform identity* --- to appear in both training and test sets, enabling the model to memorise specific transform identities rather than learning general fragment replacement rules. Such leakage would inflate accuracy estimates without measuring true generalization to unseen replacement scenarios.

We prevent this with a **transform-heldout split**: we identify all unique $(f^{\mathrm{old}}, \sigma)$ pairs in the MMP data and partition them such that no pair appears in both the training and evaluation splits. This guarantees that the model must generalise to transform identities it has never encountered during training.

Beyond the primary transform-heldout split, we construct two additional evaluation sets that provide complementary perspectives on model performance:

- **Secondary blind set.** A held-out set of 13,347 queries with a 150-fragment vocabulary that shares no $(f^{\mathrm{old}}, \sigma)$ pairs with the training set. The blind vocabulary is entirely disjoint from the training vocabulary. This set serves as the primary evaluation benchmark for all main experiments and performance claims.
- **Canonical analysis set.** A larger set of 21,052 queries with a 152-fragment vocabulary, used exclusively for robustness analysis and mechanistic investigation. No primary performance claims in this paper are based on the canonical set; it is included to probe model behaviour under broader fragment diversity and to verify that conclusions are not artefacts of a particular split configuration.

**Leakage verification.** We systematically verify that the split design achieves its intended isolation. Table 1 reports the overlap counts of $(f^{\mathrm{old}}, \sigma)$ transform identities between every pair of splits. Zero overlap is confirmed for all pairs, ensuring that no evaluation query shares its transform identity with any training query.

---

**Table 1.** Leakage verification: overlap counts of $(f^{\mathrm{old}}, \sigma)$ transform identities between split pairs. Zero overlap is confirmed for all pairs.

| Split Pair | Transform Overlap | Verified |
|---|---|---|
| Train / Transform-heldout test | 0 | Yes |
| Train / Secondary blind | 0 | Yes |
| Train / Canonical test | 0 | Yes |
| Canonical / Secondary blind | 0 | Yes |

---

This three-tier evaluation design --- transform-heldout test, secondary blind, and canonical analysis --- enables rigorous assessment of fragment replacement methods at multiple levels of generalization difficulty. The transform-heldout split is the central methodological contribution to the benchmark: it ensures that evaluation measures the model's ability to generalise to unseen fragment-scaffold combinations rather than its capacity to memorise training-set transform identities. The secondary blind set provides a stricter stress test with a completely disjoint fragment vocabulary, while the canonical set enables mechanistic analysis without risk of contaminating the primary evaluation.

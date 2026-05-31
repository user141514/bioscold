# 3.2 Benchmark Construction -- Spec

## Coverage
Describes construction of the evaluation benchmark: ChEMBL37K data source, MMP extraction pipeline, decoy repair procedure for negatives, transform-heldout split design, secondary blind and canonical analysis sets, leakage verification.

## Key Equations / Notation

MMP extraction yields replacement triples:
$$
\mathcal{M} = \{(f_{\text{old}}, \sigma, c^*) : \text{observed MMP substitution}\}
$$

Decoy construction (procedural -- 4 stages):
1. 1:1 ratio of positive:decoy per query
2. Wrong-positive removal: reject $c_{\text{decoy}} \in \mathcal{P}_i$
3. Deduplication: collapse identical $(f_{\text{old}}, c_{\text{decoy}})$ pairs
4. 1:5 diagnostic set (positive:decoy) for stress testing

Transform-heldout split constraint:
$$
\forall (q_i, q_j) \text{ in train x eval}: (f_i^{\text{old}}, \sigma_i) \neq (f_j^{\text{old}}, \sigma_j)
$$

Three evaluation tiers:
- **Transform-heldout test**: primary held-out split
- **Secondary blind**: 13,347 queries, 150-fragment vocabulary, no overlap with train
- **Canonical analysis**: 21,052 queries, 152-fragment vocabulary, robustness only

## Figures / Tables Needed
- **Table 1**: Leakage verification matrix (Train/Test/Blind/Canonical zero overlaps)
- **Figure**: MMP extraction pipeline diagram (optional but helpful -- molecules $\rightarrow$ bond cleavage $\rightarrow$ fragment-scaffold pairs $\rightarrow$ MMPs $\rightarrow$ quality filters)

## Dependencies
- Uses query notation $\mathcal{Q}, q_i, f_i^{\text{old}}, \sigma_i$ from 3.1
- Must define $\mathcal{M}$ (set of MMP triples) used by 3.1's $\mathcal{P}_i$ definition
- Split definitions are used by 3.3 (base ranker evaluation), 3.4 (training protocol), 3.5 (ablation)

## What Needs Updating from Existing Draft

### Issues in Full Draft (lines 135-177)
1. **MMP extraction detail**: Both drafts describe retrosynthetic bond cleavage rules at acyclic single bonds. This is good. Add a brief note on the specific quality filter thresholds (MW 15-250 Da, PAINS/Brenk removal).
2. **Decoy construction**: The 4-stage procedure is well-described. The unified draft has slightly cleaner language. Prefer the full draft's clearer mathematical framing.
3. **Split naming**: The full draft uses "transform-heldout test" as one of three tiers. The outline uses "transform-heldout" as the general strategy with "secondary blind" and "canonical analysis" as specific sets. Clarify: the *strategy* is transform-heldout; the *specific evaluation sets* are secondary blind and canonical analysis.
4. **Vocabulary sizes**: 150 vs 152 fragments. State explicitly: "150 fragments in the secondary blind vocabulary (disjoint from training), 152 fragments in the canonical vocabulary (training vocabulary size TBD)."
5. **Leakage verification table**: Present as a formatted table with all 4 split pairs and zero verified.
6. **1:5 diagnostic set**: Mention its purpose but note it is excluded from all primary evaluation.
7. **Query count consistency**: 13,347 secondary blind queries (both drafts agree). 21,052 canonical queries (both drafts agree). If training queries are known include them.

### Final Output Requirements
- ~800-1000 words
- References: Zdrazil2023 (ChEMBL), Hussain2010, Griffen2011 (MMP rules), Baell2010, Brenk2008 (filters)

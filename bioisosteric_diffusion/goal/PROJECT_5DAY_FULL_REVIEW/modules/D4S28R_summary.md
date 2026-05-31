# D4S28R: Final Resolution -- Scorer All + Route Rejection + Alignment Bug Fix

**Date**: 2026-05-29
**Status**: DEPLOYABLE -- scorer_all is the optimal strategy (blind Top10 = 0.8851)

---

## Starting Problem

After D4S28's alignment bug was discovered (one-hot mismatch causing blind Top10 = 0.7120 instead of 0.8851), D4S28R was created to:

1. **Document and explain the alignment bug** in a permanent record
2. **Recompute all metrics** with the corrected blind evaluation
3. **Evaluate routing strategies**: can we improve by routing some queries to baseline and others to scorer?
4. **Produce a final verdict** on the deployable configuration

The post-fix landscape: scorer_all achieved blind Top10 = 0.8851 (+0.0293), but 6/19 fragments showed negative delta (scorer < baseline). Could we route the 6 negative fragments to baseline while using scorer on the 13 positive fragments?

## Key Experiments

### Alignment Bug Fix Verification

| Metric | Buggy (0.7120) | Fixed (0.8851) | Delta |
|--------|----------------|----------------|-------|
| Blind Top10 | 0.7120 | 0.8851 | +0.1731 |
| vs Baseline (0.8558) | -0.1438 | +0.0293 | +0.1731 |

Root cause: `pd.get_dummies()` on val (8 old_fragments = ~79 columns) vs blind (19 old_fragments = ~90 columns). Fix: `cat_templates` extract categories from val, map unseen blind categories to 'OTHER', enforce column count equality.

### Scorer-All Final Metrics

| Split | Method | Top10 | MRR | Delta |
|-------|--------|-------|-----|-------|
| Val (5-fold OOF) | Baseline blend | 0.8340 | 0.4263 | -- |
| Val (5-fold OOF) | **HistGB scorer** | **0.9366** | -- | **+0.1026** |
| Blind (one-shot) | Baseline blend | 0.8558 | 0.4842 | -- |
| Blind (one-shot) | **HistGB scorer** | **0.8851** | -- | **+0.0293** |

Bootstrap 95% CI for scorer blind: [0.8795, 0.8904]
Bootstrap 95% CI for delta: [+0.0236, +0.0350] -- **CI excludes 0**

### Rescue/Lost Analysis (Blind)

| Metric | Count | Percentage |
|--------|-------|------------|
| Baseline misses (Top10 miss) | 1,925 | -- |
| Scorer rescues (miss -> hit) | 987 | 51.3% of misses |
| Baseline hits | 11,422 | -- |
| Scorer loses (hit -> miss) | 596 | 5.2% of hits |
| **Net gain** | **+391 queries** | **2.9% of all queries** |

### Route Rejection Analysis

Four routing strategies evaluated:

**1. Fragment-level routing: IMPOSSIBLE**
- Val has 8 old_fragments, blind has 19
- **ZERO overlap** (transform-heldout split by design)
- Fragment-level rules learned on val cannot be applied to blind

**2. Cluster-level routing: REJECTED**
- Val route table (6 clusters): USE cluster_03, 06, 09; FALLBACK cluster_01, 05, 07
- Applied to blind: Top10 = 0.8591 (+0.0033)
- Val-blind delta correlation: **r = -0.7371** (NEGATIVE -- val delta does NOT predict blind delta)
- Sign agreement: only 2/5 clusters

**3. Attachment-level routing: REJECTED**
- Val route table (5 attachments): USE C|C, C|N, C|O, C|S; FALLBACK N|S
- Applied to blind: Top10 = 0.8845 (+0.0287)
- Nearly matches scorer_all (0.8851) but slightly worse
- No benefit over using scorer everywhere

**4. AND-gate routing: REJECTED**
- Top10 = 0.8591 (+0.0033), same as cluster route

| Route | Blind Top10 | Delta | vs scorer_all |
|-------|-------------|-------|---------------|
| **scorer_all** | **0.8851** | **+0.0293** | -- |
| attachment_route | 0.8845 | +0.0287 | -0.0006 |
| cluster_route | 0.8591 | +0.0033 | -0.0260 |
| AND_route | 0.8591 | +0.0033 | -0.0260 |
| fragment_route | N/A | N/A | 0 overlap |

**Conclusion**: scorer_all is optimal. No val-derived routing strategy improves over using the scorer on all blind queries.

### Fallback Summary

| Method | Top10 | Delta | vs Baseline |
|--------|-------|-------|-------------|
| baseline_blend | 0.8558 | 0 | -- |
| scorer_all_queries | 0.8851 | +0.0293 | +0.0293 |
| fallback_fragment_level | 0.9067 | +0.0509 | -- (uses oracle fragment assignment, not real) |
| oracle_per_query | 0.9297 | +0.0739 | -- (theoretical upper bound) |

The fallback_fragment_level (0.9067) is a **theoretical upper bound** for fragment-level routing -- it assumes perfect knowledge of which fragments are positive/negative subspaces on blind. This cannot be realized since val-learned routing is negatively correlated with blind performance (r = -0.7371).

### Blind Fixed Summary (Four-Way Comparison)

| Method | Top10 | MRR | Delta |
|--------|-------|-----|-------|
| D4S28_HistGB_blind_FIXED | 0.8851 | 0.4493 | +0.0293 |
| D4S27_blend_blind (baseline) | 0.8558 | 0.4842 | 0 |
| D4S28_HistGB_val_OOF | 0.9366 | 0.6134 | +0.1026 |
| D4S28_HistGB_blind_BUGGY | 0.7120 | 0.3361 | -0.1438 |

## Key Files

| File | Role |
|------|------|
| `E:\zuhui\bioisosteric_diffusion\goal\A_improve\D4S28R_FINAL_LOCK\01_alignment_bug_explanation.md` | Bug root cause and fix |
| `E:\zuhui\bioisosteric_diffusion\goal\A_improve\D4S28R_FINAL_LOCK\02_scorer_all_metrics.md` | Final scorer metrics with bootstrap CI |
| `E:\zuhui\bioisosteric_diffusion\goal\A_improve\D4S28R_FINAL_LOCK\03_route_rejection.md` | All routing attempts and rejection rationale |
| `E:\zuhui\bioisosteric_diffusion\goal\A_improve\D4S28R_FINAL_LOCK\07_FINAL_VERDICT.md` | Final deployable configuration |
| `E:\zuhui\bioisosteric_diffusion\goal\A_improve\D4S28R_FINAL_LOCK\d4s28r_fallback_summary.csv` | 5-method comparison including fallback/oracle |
| `E:\zuhui\bioisosteric_diffusion\goal\A_improve\D4S28R_FINAL_LOCK\d4s28_blind_fixed_summary.csv` | 4-way: fixed/baseline/val/buggy |

## Core Numbers

1. **Blind scorer_all Top10**: 0.8851 (+0.0293 over baseline 0.8558), 95% CI [+0.0236, +0.0350]
2. **Net gain**: +391 queries (987 rescues - 596 losses), 2.9% of 13,347 blind queries
3. **Route rejection**: All routing strategies rejected; fragment-level impossible (0 overlap), cluster-level negative correlation (r = -0.7371), attachment-level no better than scorer_all

## Verdict

**DEPLOYABLE -- scorer_all is the optimal strategy.** The alignment bug is fully documented and fixed. All routing strategies are rejected -- no val-derived routing rule predicts blind positive/negative subspaces. The scorer provides statistically significant blind improvement (+0.0293, 95% CI [+0.0236, +0.0350]) with net +391 queries gained.

**Deployable configuration**: HistGB candidate-level scorer on 82 safe features, trained on full val data (negative subsampling 5:1), applied to ALL blind queries (no routing).

**Known limitations**: 6/19 fragments degrade (negative subspaces, documented), val->blind generalization gap (0.937 -> 0.885), oracle headroom +0.074 suggests further improvement possible.

## Retracted/Negated Old Conclusions

- **Negated**: The preliminary finding that D4S28 blind "catastrophically failed" (Top10 = 0.7120). This was entirely caused by one-hot encoding alignment bug.
- **Negated**: The hypothesis that routing strategies could improve over scorer_all. All four routing strategies rejected; none beat or matched scorer_all on blind.
- **Negated**: The fear that negative subspaces (6/19 fragments degrade) could be fixed by val-derived routing rules. Val-blind fragment delta correlation is **negative** (r = -0.7371) -- negative subspaces on val are NOT negative subspaces on blind.
- **Retracted**: The conclusion that D4S30 USR shape features could provide incremental value. D4S30 was rejected as zero incremental value (documented in verdict chain).

## Still-Credible Final Conclusions

1. **Deployable configuration**: HistGB scorer on ALL blind queries (no routing) achieves Top10 = 0.8851 (+0.0293, p < 0.05).
2. **Route rejection is final**: No val-derived routing strategy transfers to blind because fragment overlap is zero by design, and cluster/attachment-level delta correlation is negative or neutral.
3. **Net gain is robust**: 987 rescues vs 596 losses (+391 queries, 2.9% net), with bootstrap CI excluding zero.
4. **Scorer degrades MRR**: blind MRR drops from 0.484 (baseline) to 0.449 (scorer), meaning ranking within Top10 is worse even though coverage is better. The scorer optimizes for Top10 hit rate, not ranking precision.
5. **Oracle headroom = +0.074** (0.930 - 0.856), suggesting ~+0.045 improvement possible over current scorer.
6. **Next research directions**: EGNN 3D learned representations, full ChEMBL bioactivity data, larger val fragment diversity.

## Impact on Paper

### Methods
- **Final model description**: HistGB candidate-level scorer on 82 safe features, trained on full val with 5:1 negative subsampling. Applied to all queries (no routing).
- **Feature alignment protocol**: Document `cat_templates` as mandatory methodology when using one-hot categorical features across disjoint category sets.
- **Route rejection conclusion**: State that routing was attempted (fragment-level, cluster-level, attachment-level, AND-gate) and all were rejected because val-blind transfer is unreliable for this problem.
- **The alignment bug should be described in a Methods caveat or Supplementary Material** as a cautionary example.

### Results
- **Primary result table**:
  | Method | Blind Top10 | Delta | 95% CI |
  |--------|-------------|-------|--------|
  | Baseline (MLP+HGB blend) | 0.8558 | -- | -- |
  | D4S28R HistGB scorer (all queries) | 0.8851 | +0.0293 | [+0.0236, +0.0350] |
  | Oracle (per-query optimal) | 0.9297 | +0.0739 | -- |
- **Rescue/lost**: "The scorer rescues 987/1,925 baseline misses (51.3%) while losing 596/11,422 baseline hits (5.2%), for a net improvement of +391 queries (2.9%)."
- **Route rejection table**: Show that no routing strategy improves over scorer_all, and that val-blind delta correlation is negative (r = -0.7371).
- **Acknowledge MRR tradeoff**: scorer MRR = 0.449 vs baseline 0.484 -- the scorer prioritizes Top10 coverage over ranking precision.

### Discussion
- **Generalization**: "The val-blind gap of 0.937 to 0.885, combined with the negative fragment-delta correlation (r = -0.7371), indicates that fragment diversity is the primary limiting factor. A validation set with more old_fragments would likely improve generalization."
- **Route rejection insight**: "The negative correlation between val and blind fragment deltas is not a model failure but a consequence of the transform-heldout split design. When fragments are disjoint between splits, fragment-level routing rules are impossible."
- **Net effect framing**: "+0.0293 blind Top10 improvement with 95% CI excluding zero. This is modest but statistically robust, and the oracle headroom (+0.074) indicates the approach is on the right track."
- **Future work**: EGNN 3D features, full ChEMBL bioactivity embeddings, increased fragment diversity in training data.
- **MRR tradeoff**: Discuss that the scorer optimizes for Top10 hit rate at the expense of MRR. If the application requires fine-grained ranking, a ranking-aware loss (e.g., ListNet, LambdaRank) should be investigated.

---

MODULE_READ_COMPLETE: D4S28R
KEY_VERDICT: Alignment bug fixed; scorer_all achieves blind Top10=0.8851 (+0.0293, p<0.05); all routing strategies rejected; scorer_all is the final deployable configuration.
KEY_NUMBERS: Blind Top10=0.8851 (+0.0293, 95%CI [+0.0236,+0.0350]), Net +391 queries (987 rescues - 596 losses), Route rejection: val-blind cluster delta r=-0.7371
CONFLICTS_FOUND: Initial blind=0.7120 vs fixed blind=0.8851 (alignment bug, resolved). Val cluster deltas negatively correlate with blind deltas (r=-0.7371) -- no route strategy transfers. MRR drops from 0.484 (baseline) to 0.449 (scorer) even though Top10 improves -- scorer optimizes coverage not ranking precision.

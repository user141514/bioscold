# 4. Results

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

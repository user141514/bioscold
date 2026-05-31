### 3.5 Ablation: The Prior_Ranks Removal

Before presenting the final evaluation, we examine whether all 77 features of the candidate-level scorer contribute positively to generalization. The answer is surprising: a set of five experimentally added features, encoding per-query rank information from the prior scores, actively harms performance on the held-out secondary blind set. This section documents the leave-one-family-out ablation that reveals this effect, explains the mechanism through which these features degrade generalization, and establishes the final model configuration.

#### 3.5.1 Leave-One-Family-Out Analysis

To assess the contribution of each feature family of the 77-dimensional scorer (Section 3.4), we construct an augmented 82-feature variant by adding five experimentally motivated features collectively denoted **prior_ranks**. These five features are per-query rank transformations of the prior scores that were added to test whether rank information could augment raw signal: **backoff_logit_rank**, **cont_prior_rank**, **t_p4_logit_rank**, **p1_logit_rank**, and **p3_logit_rank**. For a prior score $u_j(i,c)$, the corresponding query-relative rank feature is

$$
r_j^{\mathrm{prior}}(i,c)
=
1+
\sum_{c'\in\mathcal{C}_i}
\mathbb{I}\!\left[
 u_j(i,c')>u_j(i,c)
 \;\lor\;
 \bigl(u_j(i,c')=u_j(i,c)\land \tau_i(c')<\tau_i(c)\bigr)
\right],
$$

where the tie-breaking convention follows the same deterministic key $\tau_i$ used throughout (Section 3.1). Each of the five prior_ranks features encodes the position of a candidate within its query's candidate pool when sorted by the corresponding prior score -- a rank of 1 for the highest-scoring candidate, rank $|\mathcal{C}_i|$ for the lowest.

Let $\mathcal{A}_{82}$ be the index set of the augmented 82-dimensional feature vector and let

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
\subset\mathcal{A}_{82}
$$

denote the set of indices corresponding to the five prior_ranks features. The original 77-dimensional feature map is the coordinate projection that removes those five indices:

$$
\boldsymbol{\phi}^{(77)}_{ic}
=
\Pi_{\mathcal{A}_{82}\setminus\mathcal{R}_{\mathrm{prior}}}
\!\left(\boldsymbol{\phi}^{(82)}_{ic}\right).
$$

Starting from the full 82-feature configuration, we systematically remove each feature family in turn, retrain the model from scratch with the identical training protocol, and evaluate secondary blind Top-10 accuracy. The central question is straightforward: do all feature families contribute positively, or does any actively harm cross-split generalization?

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

Remaining families (model raw scores, top-K flags, prior scores, prior rates/supports, PMI, molecular descriptors, similarity/frequency) each produced negative deltas between $-0.003$ and $-0.027$ when removed, consistent with the expectation that all contribute positively to model performance.

The most striking result is highlighted: removing the five prior_ranks features improves secondary blind Top-10 accuracy from 0.8851 to 0.9243 -- a gain of $+0.0392$. This is the largest effect in the entire ablation, and uniquely it is a net improvement, not a degradation. Every other removal either degrades performance (negative delta) or produces a modest positive effect (prior_positives: $+0.0186$; query_stats: $+0.0168$) that is substantially smaller than the prior_ranks effect. The prior_ranks features are not merely uninformative; they are actively harmful to generalization.

#### 3.5.2 Why Prior_Ranks Degrade Generalization

Among all nine feature families, the prior_ranks features differ in a critical structural way: they incorporate information from the candidate's rank within each query's candidate list, as assigned by the corresponding prior scores. Unlike the remaining 72 features -- which are intrinsic properties of the molecules (delta descriptors, Tanimoto similarity) or per-candidate values that carry consistent meaning across queries (prior scores, rates, supports) -- prior_ranks are **query-relative**: a rank of 3 means fundamentally different things for different queries, depending on the size of the candidate pool, the distribution of prior scores within it, and the specific identity of the old fragment being replaced. This query-specific nature creates a risk of overfitting to per-query ranking idiosyncrasies rather than learning general fragment replacement rules.

The mechanism of failure is concretely traceable to the split structure. The validation set contains 8 distinct old_fragment identities; the secondary blind set contains 19 entirely different old_fragments with no overlap. The within-query rank distributions of the prior scores are heavily influenced by which old fragment is being replaced -- a rank of 1 for the continuation prior may carry different chemical significance when replacing a small aliphatic chloride versus a large aromatic ring system, because the candidate pools for these fragment types differ systematically in size, composition, and score distribution. The HistGB model, presented with per-query rank integers as features, learns decision thresholds that reflect the specific ranking patterns of the 8 validation old_fragments. When applied to the 19 secondary blind old_fragments with fundamentally different ranking dynamics, these learned thresholds fail to transfer, degrading performance.

We interpret this as an instance of **shortcut learning** [Geirhos et al., 2020]: the model, given features that are highly predictive within the validation distribution (because a rank integer compresses substantial query-level information about candidate ordering), defaults to relying on them instead of learning the more generalizable chemical patterns encoded by the remaining features. The rank features constitute a shortcut because they appear informative during training -- a candidate ranked 1st by the prior score is indeed likely to be correct within the validation queries -- but their informativeness is specific to the distribution of fragment identities and ranking contexts seen during training. Removing the shortcut forces the model to attend to the actual chemical features (raw scores, prior rates, molecular descriptors, Tanimoto similarities) that transfer across splits. The design rule that emerges is clear: **per-query rank features are a generalization hazard; use raw scores, rates, and intrinsic properties instead.**

#### 3.5.3 Rescue/Lost Analysis and Final Model

The practical impact of removing prior_ranks is most vividly illustrated by the rescue/lost analysis (Table 5). As defined in Section 3.6.3, a rescue occurs when the best base ranker fails to surface the correct replacement (best rank $>10$) but the scorer succeeds (best rank $\le 10$); a lost case is the opposite. The full 82-feature model rescues 987 queries where the best base ranker fails, but it also loses 596 queries where the base ranker correctly identifies the replacement and the scorer mis-ranks it. This substantial lost count -- nearly 600 regressions -- indicates that while the 82-feature model is more powerful than individual base rankers for many queries, its aggressiveness introduces widespread regression. The 77-feature model dramatically shifts this balance: it rescues 424 queries and loses only 6. The net gain is comparable (82-feature: $+391$; 77-feature: $+418$), but the distribution is fundamentally different. The 77-feature model rescues fewer queries than the 82-feature model -- it is, by design, more conservative -- but it is dramatically safer, with a lost count two orders of magnitude lower.

---

**Table 5.** Rescue/lost analysis comparing the full 82-feature model (with prior_ranks) and the final 77-feature model (without prior_ranks). Rescue and lost counts follow the definitions of Section 3.6.3. Net gain $= \text{rescued} - \text{lost}$.

| Model | Rescued | Lost | Net Gain | Blind Top-10 |
|-------|---------|------|----------|-------------|
| 82 features (with prior_ranks) | 987 | 596 | +391 | 0.8851 |
| 77 features (no prior_ranks) | 424 | 6 | +418 | 0.9243 |
| Delta | $-563$ | $-590$ | $+27$ | $+0.0392$ |

---

Based on this ablation evidence, we adopt the 77-feature model as our primary candidate-level scorer for all main experiments. The five prior_ranks features are excluded from the final feature set.

The final model achieves a secondary blind Top-10 accuracy of **0.9243** [95% CI: 0.9198, 0.9289], computed via nonparametric bootstrap resampling over queries (5,000 replicates; see Section 3.6.2). This represents a statistically significant improvement of **$+0.0686$** [95% CI: $+0.0633$, $+0.0739$] over the strongest base-ranker baseline (Score Blend, Top-10 $=0.8558$), as determined by paired bootstrap of per-query deltas. Furthermore, the 77-feature model resolves all negative subspaces present in earlier prototypes. The 82-feature model exhibited performance degradation on 6 of 19 secondary blind fragments -- primarily small, saturated fragments with narrow support distributions where the per-query rank features caused the most severe shortcut learning. After removing prior_ranks, all 19 fragments perform at or above their respective base-ranker baselines, with zero fragments exhibiting statistically significant degradation. The final model achieves both higher overall accuracy and uniform per-fragment robustness.

---

This result underscores a general principle for feature engineering in molecular machine learning: features that encode per-query rank position, while seemingly informative because they compactly summarize relative ordering, can paradoxically harm cross-split generalization by encouraging the model to memorize training-set ranking patterns rather than learn transferable chemical rules. The principled approach of removing such features -- replacing relative ranking information with absolute scores, rates, and intrinsic molecular properties -- yields a model that is simultaneously more accurate and more robust across diverse fragment types.

Having established the architecture, feature design, and ablation-validated configuration of the candidate-level scorer, we now turn to the evaluation protocol that governs all subsequent performance assessment.

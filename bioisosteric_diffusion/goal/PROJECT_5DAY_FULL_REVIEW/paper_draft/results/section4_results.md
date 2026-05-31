## 4. Results

We evaluate all methods on the secondary blind protocol (13,347 queries; Section 3.2) using query-level Top-10 accuracy as the primary metric, with 95% bootstrap confidence intervals (5,000 query-level resamples; Section 3.6.2). Sections 4.1--4.3 present the central blind result, its mechanistic basis, and its reliability profile. Section 4.4 documents a categorical schema alignment audit. Section 4.5 reports A4C provenance-group alert stratification.

---

### 4.1 The Feature-Regularized Candidate Scorer Sets the Strongest Blind Result

The central blind-set result is that the final feature-regularized candidate-level scorer, D4S31, achieves Top-10 = 0.9243 on 13,347 secondary blind queries. This improves over the strongest pre-D4S baseline, the MLP+HGB Score Blend, by Δ = +0.0686 using unrounded query-level hit indicators, with a 95% paired bootstrap CI of [+0.0633, +0.0739]. D4S31 is also substantially above the fixed Borda(DE,HGB) fusion baseline and the initial 82-feature candidate-level scorer.

Table 1 places this result in context.

---

**Table 1.** Secondary blind Top-10 accuracy, with 95% bootstrap confidence intervals. D4S28R: initial 82-feature scorer. D4S31: final 77-feature scorer after prior_ranks removal (Section 3.5). Oracle(DE,HGB) is a diagnostic bound for per-query selection between DE and HGB, not a ceiling for models using additional candidate-level features.

| Method | Blind Top-10 | 95% CI |
|---|---|---|
| Attachment-Frequency | 0.6019 | [0.5933, 0.6104] |
| HGB | 0.7437 | [0.7356, 0.7516] |
| Dual Encoder (DE) | 0.8055 | [0.7986, 0.8122] |
| Borda(DE, HGB) | 0.8384 | [0.8321, 0.8447] |
| MLP (rank-only) | 0.8402 | [0.8339, 0.8466] |
| Score Blend (MLP + HGB) | 0.8558 | [0.8495, 0.8621] |
| D4S28R (82-feature scorer) | 0.8851 | [0.8796, 0.8906] |
| **D4S31 (77-feature scorer)** | **0.9243** | **[0.9198, 0.9289]** |
| Oracle(DE,HGB) diagnostic | 0.8686 | --- |

---

Attachment-Frequency provides a strong empirical lower bound (0.6019). DE supplies the strongest individual structural signal (0.8055, +0.2036 over the frequency baseline). Borda(DE,HGB) demonstrates that structural and empirical rankers are complementary under the transform-heldout blind protocol (0.8384, +0.0329 over DE alone). The MLP provides a marginal Top-10 gain over Borda (+0.0018, not significant) but a statistically significant MRR improvement (paired bootstrap $p < 0.01$; full MRR values in Supplementary Table S2). The Score Blend is the strongest baseline using only base-ranker signals (0.8558). D4S31 improves beyond this baseline by adding candidate-level chemical, statistical, and descriptor features while removing the non-transferable prior-rank shortcuts identified in Section 4.2.

Oracle(DE,HGB) is reported only as a diagnostic bound for per-query selection between DE and HGB. It is not a ceiling for candidate-level models using additional features. D4S31 exceeds this diagnostic because it uses information beyond the two base-ranker hit indicators, including molecular descriptors, training-derived prior statistics, PMI contrast scores, and frozen-schema categorical features.

The initial 82-feature scorer (D4S28R) achieved 0.8851 but exhibited degradation on 6 of 19 blind fragments. D4S31 achieves uniform per-fragment robustness: all 19 fragments perform at or above base-ranker baselines, with zero fragments exhibiting statistically significant degradation (per-fragment values in Supplementary Table S5).

### 4.2 Removing Non-Transferable Prior-Rank Shortcuts Improves Blind Transfer

The strongest feature-engineering result is not an added feature family, but the removal of one. Starting from the initial 82-feature candidate-level scorer, removing the five prior_ranks features improves blind Top-10 from 0.8851 to 0.9243, a gain of +0.0392. No other leave-one-family-out ablation produces a comparable improvement; removing model scores or model ranks degrades performance.

---

**Table 2.** Leave-one-family-out ablation. Delta is relative to the full 82-feature configuration (D4S28R). Positive deltas indicate that removing the family *improves* generalization. Two additional families (prior_positives, query_stats) were also evaluated; results are reported in Supplementary Table S4.

| Condition | Features Removed | Blind Top-10 | $\Delta$ from Full |
|-----------|-----------------|--------------|---------------------|
| Full (82 features; D4S28R) | 0 | 0.8851 | -- |
| **Drop prior\_ranks (77 features; D4S31)** | **5** | **0.9243** | **+0.0392** |
| Drop model\_ranks (F3) | 6 | 0.8697 | $-$0.0154 |
| Drop model\_scores (F1) | 5 | 0.8610 | $-$0.0241 |

---

This pattern identifies prior_ranks as a non-transferable shortcut. These features encode the within-query rank positions of sparse prior scores. Such ranks are predictive within the development distribution, but their meaning changes when the fragment composition shifts from the 8-fragment development setting to the 19-fragment secondary blind setting. Removing them forces the model to rely on features with more stable cross-split meaning: raw model scores, train-derived rates and supports, molecular descriptors, fingerprint similarities, and categorical features under the frozen schema.

The remaining documented families (model raw scores, top-K flags, prior scores, prior rates/supports, PMI contrast scores, molecular descriptors, similarity/frequency, categorical features) each produced negative deltas (−0.003 to −0.027) when removed, consistent with each contributing positively (full per-family values in Supplementary Table S4).

Because this ablation was motivated by post-audit analysis, we treat the prior-rank finding as an analytical result rather than a pre-registered hypothesis. The corresponding claim boundary is discussed in Section 5.

### 4.3 Feature Regularization Recovers Baseline Misses While Reducing Regression

The D4S31 gain is not only an aggregate Top-10 improvement; it also changes the rescue/lost profile of the candidate-level scorer. Relative to the Score Blend baseline, D4S31 rescues 1,016 queries that the baseline misses and loses 101 queries that the baseline gets right. Thus the scorer produces an approximately 10:1 rescue-to-loss ratio against the strongest pre-D4S baseline.

---

**Table 3.** Rescue/lost analysis for D4S31 against reference score columns from the candidate-level dataset. These are per-query Top-10 metrics computed from the respective score columns and are not identical to the standalone base-ranker rows in Table 1 (except DE and Score Blend, which match). All counts verified: model_hits = ref_hits + rescue − lost. N = 13,347 blind queries.

| Reference (candidate-matrix score column) | Ref Top-10 | Rescue | Lost | Net | Arithmetic |
|-----------|-----------|--------|------|-----|-----------|
| Score Blend | 0.8558 | 1,016 | 101 | +915 | ✓ |
| Borda (candidate-matrix) | 0.8456 | 1,152 | 101 | +1,051 | ✓ |
| HGB-refit (candidate-matrix) | 0.8623 | 977 | 150 | +827 | ✓ |
| DE | 0.8055 | 1,754 | 168 | +1,586 | ✓ |

---

This is a large reliability improvement over the initial 82-feature scorer. The D4S28R model rescued many baseline misses, but it also introduced 596 lost queries against Score Blend (computed in the D4S28R audit; see Supplementary Table S3). After prior_ranks removal, D4S31 reduces lost queries from 596 to 101, a 5.9× reduction, while retaining a strong positive net gain. This rescue/loss shift explains why the 77-feature model is preferable to the more aggressive 82-feature model despite using fewer features.

The same conclusion appears at the fragment level: D4S28R degraded 6 of 19 secondary blind old-fragment identities, whereas D4S31 shows no statistically significant degradation across the 19 fragments. The full reference-specific rescue/lost arithmetic and per-fragment values are reported in Supplementary Tables S3 and S5.

---

### 4.4 Categorical Schema Alignment Prevents Invalid Blind Prediction

During development, we encountered a silent prediction bug: naive one-hot encoding of categorical features (fragment identity, attachment signature, frequency bin) on the blind set produced column spaces incompatible with the development set, causing learned feature weights to be applied to semantically incorrect columns. This yielded a blind Top-10 of 0.7120 on the 82-feature prototype. The frozen categorical schema described in Section 3.4.4 corrected this to 0.8851. This is an implementation audit finding, not a model contribution: the difference reflects a bug repair. We document it because categorical encoding consistency is a practical hazard in molecular machine learning pipelines where fragment vocabularies shift between development and deployment.

---

### 4.5 Dual-Mode Provenance Strata Show Risk Gradients

Table 4 reports A4C alert rates for the Borda/HGB provenance groups used in the dual-mode workflow (Section 3.7.2). This analysis is tied to the workflow strata defined by base-ranker top-$K$ sets ($\mathcal{K}_{\mathrm{HGB}}$, $\mathcal{K}_{\mathrm{DE}}$, $\mathcal{K}_{\mathrm{Borda}}$), not to the D4S31 scorer's primary Top-10 evaluation.

---

**Table 4.** A4C alert rates by provenance group.

| Group | Definition | Alert Rate |
|-------|-----------|------------|
| G4 | Shared ($\mathcal{K}_{\mathrm{HGB}} \cap \mathcal{K}_{\mathrm{Borda}}$) | 0.99% |
| G3 | DE-elevated ($\mathcal{K}_{\mathrm{Borda}} \setminus \mathcal{K}_{\mathrm{HGB}}$) | 9.67% |
| G2 | Borda-emergent ($\mathcal{K}_{\mathrm{Borda}} \setminus (\mathcal{K}_{\mathrm{HGB}} \cup \mathcal{K}_{\mathrm{DE}})$) | 46.85% |

---

G4 (shared candidates) exhibits a near-zero alert rate of 0.99%, establishing a low-alert reference region. G3 (DE-elevated) carries a moderate alert rate of 9.67%: similarity-supported expansion beyond frequency priors introduces manageable additional risk. G2 (Borda-emergent) carries a substantially elevated alert rate of 46.85%: nearly half of proposals that emerge only through combined ranker agreement carry at least one structural alert, requiring individual expert review. The alert-rate gradient (0.99% $\rightarrow$ 9.67% $\rightarrow$ 46.85%) supports the practical usefulness of the provenance-label design as a computational triage signal. These alert rates are computational screening signals, not experimentally validated determinations (Section 3.7.3).

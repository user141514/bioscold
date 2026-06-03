## 5. Discussion

### 5.1 Principal Finding

The main prospective finding is that candidate-level HistGB scoring improves structure-derived replacement recovery under Fresh Blind2. HistGB82 achieved higher Top-10 than Score Blend and Borda(DE,HGB), supporting candidate-level integration of base-ranker outputs, train-derived priors, molecular descriptors, and frozen categorical schema features. This is the primary prospective Top-10 HistGB result of the revised manuscript.

### 5.2 Prior-Rank Instability Rather than a Replicated Deletion Gain

Fresh Blind2 changes the interpretation of the earlier no-prior-rank result. Although prior_ranks deletion appeared beneficial in the original secondary-blind post-audit analysis, the pre-specified no-prior-rank HistGB77 configuration did not replicate a Top-10 advantage over HistGB82 on Blind2. The safer conclusion is not that prior_ranks should universally be removed, but that query-relative prior-rank transforms have unstable transfer behavior across blind partitions.

### 5.3 Correlated Queries and Candidate-Matrix Boundaries

Query-level CIs are positive for candidate-level scoring comparisons, but grouped Top-10 CIs are wider because the blind set contains correlated queries sharing old fragments, transform keys, and attachment signatures. This limits fragment-level generalization claims and motivates reporting group-level sensitivity alongside query-level intervals. Candidate-matrix audits similarly support the evaluated full-161 and fixed-150 policies, but they do not establish robustness to every possible candidate universe.

### 5.4 Learning-to-Rank as the Natural Next Algorithmic Direction

Categorical-aware LambdaRank was evaluated as a post-hoc architecture diagnostic after the HistGB evidence hierarchy was established. It robustly improves over Score Blend, but its comparison against HistGB is heterogeneous at group level. It is therefore a future pre-specified architecture path, not the current main method.

### 5.5 Biological and Calibration Limits

The benchmark measures recovery of observed structure-derived positives, not activity-outcome continuity. The activity-comparable diagnostic does not show that top-ranked replacements preserve activity, and no biological, experimental, or external-corpus claim is made. Calibration diagnostics further show that HistGB outputs are ranking scores rather than absolute probabilities; they should not be interpreted as absolute replacement probabilities.

### 5.6 Scope

The revised paper is therefore best read as a leakage-controlled benchmark and audit study. It supports candidate-level scoring under query-side transform-heldout evaluation, identifies prior-rank features as unstable across blind partitions, and motivates categorical-aware learning-to-rank as a future algorithmic upgrade. It does not claim open-vocabulary generation, general bioisostere validity, activity-outcome inference, or experimentally validated medicinal-chemistry utility.

## 6. Conclusion

We introduced a leakage-controlled closed-vocabulary benchmark for structure-derived fragment replacement ranking and prospectively evaluated candidate-level scoring on Fresh Blind2. The 82-feature HistGB scorer improved Top-10 over Score Blend and Borda(DE,HGB), supporting candidate-level integration of base-ranker, prior, descriptor, and categorical features.

The no-prior-rank 77-feature configuration remained baseline-beating and slightly improved MRR, but it did not reproduce its earlier Top-10 advantage over the 82-feature scorer. We therefore treat prior-rank deletion as a diagnostic of unstable feature transfer rather than as a replicated model-selection improvement.

The benchmark remains structure-derived: it does not support activity-outcome inference, external transfer, biological validation, experimental validation, or absolute score-to-outcome claims. The central benchmark-audit lesson is to remove exact query-side transform memorization first, then ask which ranking signals still transfer.

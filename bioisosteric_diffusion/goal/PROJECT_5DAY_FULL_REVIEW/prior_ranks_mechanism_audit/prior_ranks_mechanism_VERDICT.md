# Prior-Ranks Targeted Mechanism Audit Verdict

## Verdict

**SUPPORTED_BOUNDED_SHORTCUT_INTERPRETATION.**

Post-audit diagnostics support the interpretation that query-relative prior-rank features behaved as non-transferable shortcut signals under the secondary blind fragment shift. The evidence is bounded: it identifies prior ranks as the dominant harmful prior-family transform in this benchmark, but it does not prove that prior ranks are universally harmful, that all rank features are bad, or that a single causal mechanism has been proven.

## Evidence Summary

### 1. Targeted prior-family ablation

The prior-rank deletion is the largest positive targeted ablation relative to the initial 82-feature scorer:

| Diagnostic model | Top-10 | Delta vs 82-feature scorer | 95% CI |
|---|---:|---:|---:|
| Full 82-feature scorer | 0.8851 | 0.0000 | not applicable |
| Drop prior ranks / 77-feature scorer | 0.9243 | +0.0393 | [0.0354, 0.0432] |
| Drop all non-rank prior families | 0.9024 | +0.0174 | [0.0143, 0.0204] |
| Drop F6 prior statistics | 0.9001 | +0.0151 | [0.0124, 0.0178] |
| Drop F5 prior scores | 0.8924 | +0.0073 | [0.0049, 0.0097] |
| Drop F7 PMI / conditional-prior contrast | 0.8814 | -0.0037 | [-0.0064, -0.0010] |

Interpretation: the harmful effect is **not equally distributed across all prior information**. Removing prior ranks produces the largest gain. Some non-rank prior removals also improve Top-10, so the manuscript should not claim that raw prior information is uniformly beneficial. The safer claim is that sparse prior-rank transforms are the dominant unstable prior-family component, while retained raw scores/rates/PMI/model outputs are more stable relative to prior ranks in this fixed benchmark.

### 2. Support/sparsity audit

Queries were stratified into tertiles by query median `query_prior_max_support`, excluding the global `p0_support` constant. The 77-feature minus 82-feature gain is largest in the low-support regime:

| Support bin | n queries | Top-10 82 | Top-10 77 | Delta 77-82 | 95% CI | Rescue / lost |
|---|---:|---:|---:|---:|---:|---:|
| Low support | 4,449 | 0.9162 | 0.9649 | +0.0488 | [0.0423, 0.0553] | 227 / 10 |
| Medium support | 4,449 | 0.8519 | 0.8863 | +0.0344 | [0.0276, 0.0414] | 206 / 53 |
| High support | 4,449 | 0.8872 | 0.9218 | +0.0346 | [0.0281, 0.0411] | 193 / 39 |

Interpretation: the support audit is consistent with the sparse-support shortcut hypothesis. It is not a formal causal proof because support bins are post-audit diagnostics and not a randomized intervention.

### 3. Old-fragment block audit

Across 19 old-fragment strata:

- Initial 82-feature scorer has five negative old-fragment point-estimate deltas relative to Score Blend.
- Post-audit 77-feature scorer has two negative old-fragment point-estimate deltas relative to Score Blend.
- Relative to the 82-feature scorer, the 77-feature scorer improves 17 of 19 old-fragment strata, is unchanged or positive in the severe degradation strata, and has only two small negative point-estimate deltas (`*CCC`: -0.0057; `*c1ccccc1F`: -0.0033).

The largest repaired 82-feature regressions are:

| Old fragment | 82 vs Score Blend | 77 vs Score Blend | 77 vs 82 |
|---|---:|---:|---:|
| `*Cc1cccc(OC)c1` | -1.0000 | -0.0593 | +0.9407 |
| `*Cc1ccccc1OC` | -0.8056 | 0.0000 | +0.8056 |
| `*Cc1ccccc1Cl` | -0.2930 | 0.0000 | +0.2930 |
| `*Cc1cccnc1` | -0.1266 | -0.0063 | +0.1203 |

Interpretation: prior-rank pruning reduces fragment-specific degradation rather than merely shifting aggregate Top-10.

## Answers to Required Questions

**Is the harmful effect specific to prior_ranks or to all prior information?**

It is most strongly associated with prior_ranks. Dropping prior_ranks gives the largest gain (+0.0393), whereas dropping all non-rank prior families gives a smaller gain (+0.0174), dropping prior scores gives a small gain (+0.0073), and dropping PMI/conditional-prior contrast harms performance (-0.0037). The evidence supports a prior-rank-specific dominant shortcut claim, not a claim that all prior information is harmful.

**Do raw prior scores/rates appear more stable than query-relative ranks?**

Relative to prior_ranks, yes, with caveats. Raw and statistical prior families are not uniformly indispensable, because removing some of them also improves Top-10. However, their removals do not explain the main +0.0393 gain, and one prior contrast family is beneficial. The stable manuscript wording should say that retained raw prior scores/rates, descriptors, and model outputs are more stable than sparse query-relative prior-rank transforms in this benchmark, not that every raw prior feature is always beneficial.

**Is the shortcut interpretation supported, bounded, or weak?**

Supported but bounded. The targeted ablation, support-bin concentration, and fragment-block repair are mutually consistent with non-transferable shortcut learning. The evidence is post-audit and observational; it does not prove a unique causal mechanism.

**What exact claim is safe for the manuscript?**

Safe wording:

> Post-audit analysis indicates that query-relative prior-rank features behaved as non-transferable shortcuts under the secondary blind fragment shift. The largest targeted ablation gain came from removing prior ranks, the gain was strongest in low train-derived support regimes, and fragment-level regressions were reduced after pruning. Raw prior scores, prior statistics, PMI contrasts, molecular descriptors, and model outputs provide more stable candidate-level signals than sparse prior-rank transforms in this benchmark, although some non-rank prior families also show post-audit instability and the mechanism should not be read as a universal claim about rank features.

## Claim Boundaries

Do not claim:

- prior_ranks are universally harmful.
- all rank features are bad.
- all non-rank prior features are always beneficial.
- the 77-feature scorer is fully prospective.
- this proves a single causal mechanism.
- this mechanism transfers automatically to other datasets or open-vocabulary generation.

## Source Files

- `prior_ranks_targeted_ablation.csv`
- `prior_ranks_support_bin_audit.csv`
- `prior_ranks_old_fragment_delta.csv`
- `figure_s_prior_ranks_mechanism.svg`
- `figure_s_prior_ranks_mechanism.pdf`
- `figure_s_prior_ranks_mechanism.png`

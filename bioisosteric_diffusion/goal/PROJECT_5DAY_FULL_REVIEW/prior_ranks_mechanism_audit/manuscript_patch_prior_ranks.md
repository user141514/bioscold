# Manuscript Patch: Prior-Ranks Mechanism Claim

## Patch Purpose

This patch strengthens the prior_ranks mechanism language without changing locked numerical results or converting the 77-feature scorer into a fully prospective claim.

## Suggested Placement

- Results 4.2: include one compact paragraph covering targeted prior-family ablation and the support-bin core result.
- Results 4.3: keep old-fragment repair to one sentence; do not paste the full block audit into the main text.
- Discussion 5.3: use the bounded mechanism interpretation.
- Supplementary material: place the detailed old-fragment block audit and full support-bin table.

Do not paste all proposed paragraphs into the main manuscript. The main text should carry the evidence spine; detailed block-level arithmetic belongs in Supplementary material.

## Proposed Results Paragraph

**Role: targeted mechanism evidence.**

To test whether the prior-rank deletion reflected a rank-specific shortcut rather than a generic removal of prior information, we ran targeted leave-family-out diagnostics under the same fixed HistGB protocol. Removing prior_ranks produced the largest Top-10 increase relative to the initial 82-feature scorer (Delta = +0.0393, 95% CI [0.0354, 0.0432]). Dropping all non-rank prior families while retaining prior ranks produced a smaller increase (Delta = +0.0174, 95% CI [0.0143, 0.0204]); dropping prior statistics gave Delta = +0.0151, dropping prior scores gave Delta = +0.0073, and dropping PMI/conditional-prior contrasts reduced Top-10 (Delta = -0.0037, 95% CI [-0.0064, -0.0010]). Thus, the post-audit gain is concentrated most strongly in the query-relative prior-rank transforms, although some non-rank prior families also show smaller instability.

## Proposed Support/Sparsity Paragraph

**Role: support for sparse-shortcut interpretation; use only if Results 4.2 has room, otherwise keep in Supplementary.**

A support-bin audit further supports the shortcut interpretation. Query strata were formed by tertiles of train-derived `query_prior_max_support`, excluding the global `p0_support` constant. The 77-feature minus 82-feature gain was largest in the low-support bin (Delta = +0.0488, 95% CI [0.0423, 0.0553]; rescue/lost = 227/10), compared with the medium-support bin (Delta = +0.0344, 95% CI [0.0276, 0.0414]; rescue/lost = 206/53) and high-support bin (Delta = +0.0346, 95% CI [0.0281, 0.0411]; rescue/lost = 193/39). This concentration in low-support regimes is consistent with sparse prior-rank transforms acting as non-transferable shortcuts under fragment distribution shift.

## Proposed Old-Fragment Block Paragraph

**Role: fragment-level mechanism audit; Supplementary-first, main text one sentence only.**

The old-fragment block audit shows that prior-rank pruning reduces fragment-specific degradation rather than merely improving the aggregate mean. The initial 82-feature scorer had five negative old-fragment point-estimate deltas relative to Score Blend. The post-audit 77-feature scorer reduced this to two negative old-fragment point-estimate deltas. Relative to the 82-feature scorer, the 77-feature scorer improved 17 of 19 old-fragment strata, including the largest initial regressions: `*Cc1cccc(OC)c1` improved from -1.0000 vs Score Blend to -0.0593, `*Cc1ccccc1OC` improved from -0.8056 to 0.0000, and `*Cc1ccccc1Cl` improved from -0.2930 to 0.0000.

## Proposed Discussion Replacement

**Role: bounded interpretation and reviewer defense.**

The prior_ranks result is best interpreted as a bounded shortcut-learning diagnosis. The five removed features encode within-query rank positions of sparse prior scores. Under secondary blind fragment shift, these ranks can change semantic meaning because a rank value depends on the composition of the candidate list and on sparse train-derived support. The targeted audit supports this interpretation: prior_ranks deletion is the dominant positive prior-family ablation, the gain is largest in low-support query regimes, and fragment-level regressions are reduced after pruning. At the same time, this evidence is post-audit and observational. It does not show that prior ranks are universally harmful, that all rank features are bad, or that a single causal mechanism has been proven. The retained model-rank family differs from prior_ranks because it summarizes base-ranker outputs rather than sparse empirical prior-score transforms. The safer design rule is therefore benchmark-specific: under fragment distribution shift, prefer raw scores, support-aware statistics, molecular descriptors, and model outputs over sparse query-relative prior-rank transforms unless the latter are prospectively validated.

## Suggested Supplementary Figure Caption

**Supplementary Figure Sx. Prior-ranks mechanism audit.** Panel A shows targeted prior-family ablations relative to the initial 82-feature scorer under the fixed HistGB protocol. Removing prior ranks gives the largest Top-10 increase, whereas dropping PMI/conditional-prior contrast reduces performance. Panel B stratifies queries by train-derived prior support and shows that the 77-feature minus 82-feature gain is largest in the low-support bin. This is a post-audit targeted mechanism diagnostic, not prospective feature selection. These diagnostics support a bounded interpretation of prior_ranks as non-transferable shortcuts; they do not establish a universal claim about rank features.

## Self-Review

- **Clarity:** The patch separates targeted ablation, support-bin evidence, and block-level repair.
- **Claim support:** Every numerical claim maps to `prior_ranks_mechanism_audit` outputs.
- **Terminology:** Uses post-audit, diagnostic, and bounded mechanism consistently.
- **Unsupported claims avoided:** No universal prior-ranks claim; no fully prospective 77-feature claim.
- **Residual risk:** Some non-rank prior removals also improve Top-10, so the final manuscript should not say all raw prior information is stable or beneficial.

## Claim-Evidence Map

| Claim | Evidence | Status |
|---|---|---|
| Prior_ranks deletion is the dominant positive prior-family ablation. | `prior_ranks_targeted_ablation.csv` | Supported as post-audit diagnostic |
| Low train-derived support queries benefit most from prior-rank removal. | `prior_ranks_support_bin_audit.csv` | Supported as support-bin diagnostic |
| Prior-rank pruning reduces fragment-specific degradation. | `prior_ranks_old_fragment_delta.csv` | Supported as block-level point-estimate diagnostic |
| The mechanism is a non-transferable shortcut. | Targeted ablation + support bin + old-fragment audit | Supported but bounded; not causal proof |

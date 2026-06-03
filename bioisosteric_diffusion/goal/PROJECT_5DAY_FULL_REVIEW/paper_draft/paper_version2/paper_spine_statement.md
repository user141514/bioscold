# Paper Spine Statement

## Central Pain Point

The JCIM-facing problem is not whether a larger model can rank fragments on a convenient split. The problem is how structure-derived fragment replacement ranking can be evaluated without rewarding transform-level memorization or unstable shortcut features. Random splits can repeat fragment–attachment transform identities across training and evaluation, and empirical ranks can behave differently when the fragment distribution shifts.

## Central Claim

Fragment replacement ranking should be evaluated as a signal-transfer problem under transform-heldout distribution shift. Once exact transform memorization is removed, candidate-level scoring remains effective, while post-audit analysis exposes query-relative prior ranks as non-transferable shortcuts.

## Contribution Hierarchy

1. Benchmark: a transform-heldout, closed-vocabulary fragment replacement ranking benchmark removes exact fragment–attachment transform memorization.
2. Scoring: an audited candidate-level scoring framework integrates base-ranker outputs, train-derived priors, descriptors, similarity features, and frozen categorical encodings, and remains effective under secondary blind evaluation.
3. Mechanism/audit finding: post-audit analysis identifies query-relative prior ranks as non-transferable shortcuts under fragment distribution shift; pruning them produces a post-audit locked 77-feature scorer and reduces baseline hit loss.

## Prospective Evidence

The prospective candidate-level blind result is the initial 82-feature scorer. It achieved Top-10 = 0.8851 on 13,347 secondary blind queries.

## Post-Audit Evidence

The 77-feature scorer is post-audit locked and post-selection. It achieved Top-10 = 0.9243 (95% CI [0.9199, 0.9287]) and improved over Score Blend (Top-10 = 0.8558) by $\Delta$Top-10 = 0.0686 (95% CI [0.0638, 0.0733]). Pruning prior_ranks improved Top-10 by +0.0393 over the 82-feature scorer (95% CI [0.0354, 0.0432]) and reduced Score Blend hit loss from 596 to 101 queries, a 5.9× reduction. This motivates prospective replication rather than serving as fully prospective feature selection.

## Workflow-Only Layer

A4C provenance stratification is coverage-limited computational triage. G2 and G3 have 100% A4C coverage with alert rates of 46.85% and 9.67%, respectively. G4 has 5.63% coverage, 17.60% alert rate among covered candidates, and 94.37% unknown status. These strata are not medicinal chemistry validation and are not safety scoring.

## Explicitly Not Claimed

- The paper is limited to closed-vocabulary ranking rather than open-ended generation across all medicinal-chemistry contexts.
- The paper treats MMP labels as structure-derived observations rather than evidence of biological activity continuity.
- The 77-feature scorer is treated as post-audit and post-selection.
- prior_ranks pruning is described as a blind-diagnostic audit finding.
- A4C is limited to workflow triage and is not used as safety evidence.
- The paper does not claim open-vocabulary generation.
- The paper does not claim a generic HistGB architecture advance.

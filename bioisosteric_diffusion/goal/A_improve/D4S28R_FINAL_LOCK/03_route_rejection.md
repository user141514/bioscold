# D4S28R Route Rejection

**Date**: 2026-05-29
**Verdict**: All val-derived routing strategies rejected — scorer_all is optimal.

## Why Routing Was Attempted

D4S28R scorer has 6 negative subspaces (fragments where scorer < baseline) and 18 positive subspaces. The hypothesis: route queries to scorer only in positive subspaces, fallback to baseline elsewhere.

## Why Routing Failed

### Fragment-Level Routing: IMPOSSIBLE

Val has 8 old_fragments. Blind has 19 old_fragments. **ZERO overlap** by design (transform-heldout split). Fragment-level routing rules learned on val cannot be applied to blind.

### Cluster-Level Routing: REJECTED

Val route table (6 clusters): USE cluster_03, 06, 09; FALLBACK cluster_01, 05, 07.
Applied to blind: Top10=0.8591 (+0.0033).

Val->blind delta correlation: **r = -0.7371** (NEGATIVE).
Sign agreement: 2/5 clusters. Val delta does NOT predict blind delta.

### Attachment-Level Routing: REJECTED

Val route table (5 attachments): USE C|C, C|N, C|O, C|S; FALLBACK N|S.
Applied to blind: Top10=0.8845 (+0.0287).

Almost matches scorer_all (0.8851) but slightly worse. No benefit over using scorer everywhere.

Val->blind delta correlation: sign agreement 3/5.

### AND-Gate Routing: REJECTED

Top10=0.8591 (+0.0033). Same as cluster route.

## Comparison

| Route | Blind Top10 | Delta | vs scorer_all |
|-------|-------------|-------|---------------|
| scorer_all | **0.8851** | +0.0293 | — |
| attachment_route | 0.8845 | +0.0287 | -0.0006 |
| cluster_route | 0.8591 | +0.0033 | -0.0260 |
| AND_route | 0.8591 | +0.0033 | -0.0260 |
| fragment_route | N/A | N/A | 0 overlap |

## Conclusion

**scorer_all is the optimal deployable strategy.** No val-derived routing improves over using the scorer on all blind queries. The scorer's positive effect on 13/19 fragments outweighs negative effect on 6/19. Attempting to route only reduces performance.

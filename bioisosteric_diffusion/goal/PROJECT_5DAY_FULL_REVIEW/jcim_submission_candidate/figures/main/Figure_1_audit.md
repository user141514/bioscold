# Figure 1 audit

1. Does the figure define the closed-vocabulary ranking task? Yes. Panel A shows query, candidate vocabulary, closed-vocabulary ranking, and Top-10 recovery objective.
2. Does the figure distinguish random split leakage from transform-heldout control? Yes. Panel B contrasts shared transform identities under random splitting with zero transform-identity overlap under transform-heldout splitting.
3. Does the figure show evaluation tiers and their roles? Yes. Panel C separates train, development/calibration, secondary blind, and canonical analysis.
4. Is secondary blind identified as the primary claim set? Yes.
5. Is canonical analysis bounded as robustness only? Yes.
6. Are modeling details from Figure 2 absent? Yes. No base rankers, Score Blend, 82-feature scorer, 77-feature scorer, prior-rank pruning, or A4C modules are included.
7. Are performance values absent? Yes. The figure contains no measured Top-10 values, confidence intervals, rescue/lost counts, or A4C rates.
8. Does the SVG keep editable text? Yes. The script sets svg.fonttype='none'.

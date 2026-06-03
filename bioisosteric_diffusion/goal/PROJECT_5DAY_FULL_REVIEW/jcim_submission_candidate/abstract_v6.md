# Abstract

Matched molecular pair (MMP)-derived fragment replacement ranking provides a scalable structure-derived benchmark for observed substitution patterns, but permissive splits can allow repeated fragment-attachment contexts across training and evaluation. We therefore study closed-vocabulary scaffold-conditioned replacement ranking under query-side transform-heldout evaluation, where the same old-fragment/attachment-context key is not shared across train and blind partitions.

The primary prospective evidence comes from a fresh Blind2 evaluation. On this split, the 82-feature HistGB candidate-level scorer achieved Top-10 = 0.8480, improving over Score Blend (Top-10 = 0.8077) and Borda(DE,HGB) (Top-10 = 0.8176). The no-prior-rank 77-feature configuration remained above Score Blend but did not reproduce its earlier Top-10 advantage over the 82-feature scorer (77 - 82 Top-10 = -0.0070, 95% CI [-0.0094, -0.0044]), indicating that prior-rank deletion is better interpreted as a post-audit instability diagnostic than as a replicated model-selection improvement.

Grouped uncertainty and boundary diagnostics motivate cautious interpretation: query-level gains are stronger than group-level Top-10 robustness, and activity, calibration, external-transfer, or absolute likelihood claims remain unsupported.

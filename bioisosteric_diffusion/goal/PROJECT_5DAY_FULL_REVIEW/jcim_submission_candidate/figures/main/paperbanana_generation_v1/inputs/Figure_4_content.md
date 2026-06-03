Create a locked-value JCIM mechanism figure from scratch.

Scope:
Explain why post-audit prior-rank pruning improves transfer. Use exactly two panels. Do not include Figure 3's full performance hierarchy, A4C, MRR, Best-of-DE+HGB, per-fragment deltas, schema alignment, canonical analysis, or activity-validation language.

Panel A title:
A. Ablation of feature families

Panel A plot:
Effect-size / lollipop plot. X-axis: Delta Top-10 vs initial 82-feature scorer.
X-axis range approximately -0.03 to +0.045.
Show a vertical zero line.

Locked values:
- Drop prior ranks: +0.0393, CI [0.0354, 0.0432]
- Drop model ranks: -0.0154, no CI
- Drop model scores: -0.0241, no CI

Rules:
- Use prose labels, no underscores.
- Show CI only for Drop prior ranks.
- Do not claim statistical significance for values without CI.

Panel B title:
B. Reliability audit

Panel B structure:
Show a compact reliability summary, not a dashboard.

Lost cases vs Score Blend:
- Initial 82-feature scorer lost vs Score Blend = 596
- Post-audit 77-feature scorer lost vs Score Blend = 101
- 5.9x fewer lost queries

Compact rescue/lost table:
Reference | Rescue | Lost | Net
Score Blend | 1016 | 101 | +915
82-feature scorer | 626 | 102 | +524

Style:
Use muted teal for rescue/positive/pruned effect, muted red for lost/regression, gray for neutral text. White background, no icons, no gradients, no KPI blocks, no large floating annotations.

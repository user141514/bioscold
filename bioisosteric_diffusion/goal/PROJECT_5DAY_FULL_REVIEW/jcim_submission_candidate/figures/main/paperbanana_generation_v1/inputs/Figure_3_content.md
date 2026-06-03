Create a locked-value JCIM performance overview plot from scratch.

Scope:
This is a quantitative performance figure. It must be a single horizontal forest/dot plot with 95% confidence intervals. Do not use a bar chart. Do not include rescue/lost, MRR, A4C, prior-rank ablation, canonical analysis, or extra metrics.

Title:
Secondary blind Top-10 performance

Y-axis method order, top to bottom:
Attachment-Frequency
HGB
Dual Encoder
Borda(DE,HGB)
MLP
Score Blend
82-feature scorer (prospective)
77-feature scorer (post-audit)
Best-of-DE+HGB (diagnostic)

Locked values:
- Attachment-Frequency: Top-10 0.6019, CI [0.5933, 0.6104]
- HGB: 0.7437, CI [0.7356, 0.7516]
- Dual Encoder: 0.8055, CI [0.7986, 0.8122]
- Borda(DE,HGB): 0.8384, CI [0.8321, 0.8447]
- MLP: 0.8402, CI [0.8339, 0.8466]
- Score Blend: 0.8558, CI [0.8495, 0.8621]
- 82-feature scorer: 0.8851, CI [0.8796, 0.8903]
- 77-feature scorer: 0.9243, CI [0.9199, 0.9287]
- Best-of-DE+HGB: 0.8686, no CI

Visual rules:
- X-axis label: Secondary blind Top-10
- X-axis range approximately 0.58 to 0.94.
- Use muted gray for base rankers, muted blue-gray for fusion baselines, muted orange for 82-feature scorer, muted teal for 77-feature scorer, hollow light-gray marker for Best-of-DE+HGB.
- Best-of-DE+HGB must be labeled diagnostic only. Do not use the words Oracle, ceiling, or upper bound.
- No right-side evidence table, no delta column, no floating annotations.

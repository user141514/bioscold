# Figure 3 refinement audit v3

## Scope

This audit covers:

- `Figure_3_secondary_blind_performance_final_v3.pdf`
- `Figure_3_secondary_blind_performance_final_v3.svg`
- `Figure_3_secondary_blind_performance_final_v3.png`
- `Figure_3_secondary_blind_performance_data.csv`

The v3 revision keeps the single-panel horizontal forest/dot plot with confidence intervals and applies only layout micro-polish.

## Visual changes

| Check | Result | Notes |
|---|---:|---|
| In-plot numeric labels removed or weakened | PASS | The inline `0.8851` and `0.9243` labels were removed from the plot area. Exact values remain in the source CSV and manuscript tables. |
| No right-side explanatory column | PASS | The figure has no evidence-role text column. Role information is limited to small parenthetical y-label tags. |
| No delta column | PASS | No delta values are plotted or shown in the figure body. |
| No floating annotation | PASS | The plot contains no callout boxes, bracket arrows, or free-floating explanatory text. |
| More compact than previous version | PASS | Figure height was reduced from 4.55 in to 3.92 in, and row spacing was compressed while retaining readable method labels. |
| Title de-emphasized | PASS | The title was reduced and remains left-aligned with no subtitle. |
| Evidence tags weakened | PASS | The tags are parenthetical, smaller than the main method labels, and rendered in lighter tones. |
| Separators weakened | PASS | Group separators use consistent faint gray lines with reduced line weight. |
| Diagnostic row remains secondary | PASS | The Best-of-DE+HGB point is a hollow light-gray marker without confidence interval. |

## Evidence hierarchy

| Method row | Status in figure |
|---|---|
| 82-feature scorer | Labeled as `(prospective)` in the y-axis label. |
| 77-feature scorer | Labeled as `(post-audit)` in the y-axis label. |
| Best-of-DE+HGB | Labeled as `(diagnostic)` and drawn as a hollow light-gray point with no interval. |

The v3 figure distinguishes these roles without adding a right-side role column or extra prose inside the plot.

## Locked-value audit

| Method | Top-10 | CI low | CI high | Status |
|---|---:|---:|---:|---|
| Attachment-Frequency | 0.6019 | 0.5933 | 0.6104 | PASS |
| HGB | 0.7437 | 0.7356 | 0.7516 | PASS |
| Dual Encoder (DE) | 0.8055 | 0.7986 | 0.8122 | PASS |
| Borda(DE,HGB) | 0.8384 | 0.8321 | 0.8447 | PASS |
| MLP (rank-only) | 0.8402 | 0.8339 | 0.8466 | PASS |
| Score Blend (MLP + HGB) | 0.8558 | 0.8495 | 0.8621 | PASS |
| Initial 82-feature scorer | 0.8851 | 0.8796 | 0.8903 | PASS |
| Post-audit 77-feature scorer | 0.9243 | 0.9199 | 0.9287 | PASS |
| Best-of-DE+HGB diagnostic | 0.8686 | not reported | not reported | PASS |

## Forbidden-content audit

| Check | Result |
|---|---:|
| No task-forbidden diagnostic naming in figure exports or plotting script | PASS |
| No rescue/lost content | PASS |
| No MRR content | PASS |
| No A4C content | PASS |
| No prior-rank ablation content | PASS |
| No extra non-locked metric | PASS |

## Final assessment

Figure 3 v3 is a compact JCIM-style benchmark performance plot rather than a slide-style performance graphic. It keeps the evidence hierarchy visible through weak inline y-label tags, keeps the plot area dominant, and leaves detailed roles and caveats to the caption and manuscript text.

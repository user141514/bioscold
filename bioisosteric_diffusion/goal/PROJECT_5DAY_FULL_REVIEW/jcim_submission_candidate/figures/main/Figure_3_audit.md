# Figure 3 Audit

## Figure Contract

- Figure: `Figure_3_secondary_blind_performance_final`
- Endpoint: secondary blind Top-10 only
- Plot type: single-panel horizontal forest/dot plot with 95% CI error bars
- Data source: locked values from the strict Figure 3 specification
- Excluded content: no rescue/lost counts, no MRR, no A4C, no ablation panels, and no non-locked metrics

## Mandatory Checks

| Check | Result | Evidence |
|---|---:|---|
| Is the figure a single-panel horizontal forest/dot plot? | PASS | The figure contains one plotting axis with method rows, point estimates, and horizontal CI error bars. |
| Are there no right-side evidence-role or delta columns? | PASS | The final plot has no separate right-side text/table columns. Evidence roles are only represented by compact y-label second lines for the last three rows and by the caption. |
| Are all plotted values exactly equal to locked values? | PASS | The source CSV contains the nine locked Top-10 values without recalculation. |
| Are all CI values exactly equal to locked values? | PASS | The eight reported CIs match the locked values; the diagnostic row has blank CI fields because no CI was reported. |
| Is Best-of-DE+HGB hollow/light-gray and diagnostic-only? | PASS | The final row is labeled `Best-of-DE+HGB diagnostic`, has evidence role `diagnostic_only`, uses a hollow light-gray marker, and has no CI bar. |
| Are forbidden diagnostic terms absent? | PASS | The forbidden diagnostic terms specified in the task are absent from the final figure, caption, data CSV, and plotting script. |
| Is the 77-feature scorer not presented as fully prospective? | PASS | The y-label marks it as `post-audit`; the CSV evidence role is `post_audit_locked_result`; the caption states it is locked post-audit. |
| Is there no rescue/lost, MRR, A4C, or ablation content? | PASS | None of these endpoints or analyses appear in the figure or source CSV. |
| Does the plot area occupy at least 65% of figure width? | PASS | The plot axis spans approximately 67% of the canvas width. |
| Are there no floating report-style annotations? | PASS | Only two small inline value labels are shown for candidate-level scorers; there are no floating callouts, arrows, brackets, or explanatory text blocks. |
| Are there no large blank areas? | PASS | The canvas is a compact single-panel figure; the plot area occupies the majority of the width. |
| Is the figure readable at single-column or 1.5-column width? | PASS | The minimum explicit SVG font size is 7 pt, method labels do not overlap, and CI bars are visible. |

## Locked Values

| Method | Display label | Top-10 | CI low | CI high | Group | Evidence role | Delta vs Score Blend |
|---|---|---:|---:|---:|---|---|---:|
| Attachment-Frequency | Attachment-Frequency | 0.6019 | 0.5933 | 0.6104 | base_ranker | baseline |  |
| HGB | HGB | 0.7437 | 0.7356 | 0.7516 | base_ranker | baseline |  |
| Dual Encoder (DE) | Dual Encoder | 0.8055 | 0.7986 | 0.8122 | base_ranker | baseline |  |
| Borda(DE,HGB) | Borda(DE,HGB) | 0.8384 | 0.8321 | 0.8447 | fusion_baseline | baseline |  |
| MLP (rank-only) | MLP | 0.8402 | 0.8339 | 0.8466 | fusion_baseline | baseline |  |
| Score Blend (MLP + HGB) | Score Blend | 0.8558 | 0.8495 | 0.8621 | fusion_baseline | strongest_pre_D4S_baseline | 0.0000 |
| Initial 82-feature scorer | 82-feature scorer / prospective | 0.8851 | 0.8796 | 0.8903 | candidate_level | prospective_candidate_level_result | +0.0293 |
| Post-audit 77-feature scorer | 77-feature scorer / post-audit | 0.9243 | 0.9199 | 0.9287 | candidate_level | post_audit_locked_result | +0.0686 |
| Best-of-DE+HGB diagnostic | Best-of-DE+HGB / diagnostic | 0.8686 | not reported | not reported | diagnostic | diagnostic_only | +0.0128 |

## Output Files

| File | Status |
|---|---:|
| Figure_3_secondary_blind_performance_final.pdf | PASS |
| Figure_3_secondary_blind_performance_final.svg | PASS |
| Figure_3_secondary_blind_performance_final.png | PASS |
| Figure_3_secondary_blind_performance_data.csv | PASS |
| Figure_3_caption_final.md | PASS |

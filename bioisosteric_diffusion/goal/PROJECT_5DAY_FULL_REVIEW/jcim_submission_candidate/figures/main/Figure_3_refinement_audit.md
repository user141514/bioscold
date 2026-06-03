# Figure 3 Refinement Audit

## Figure Contract

- Core message: secondary blind Top-10 performance increases from simple/base rankers through fusion baselines to candidate-level scorers, with the 77-feature scorer clearly marked as a locked post-audit result.
- Figure type: horizontal forest/dot plot with compact right-side evidence table columns.
- Data source: locked values supplied in the FIGURE_3_SECONDARY_BLIND_PERFORMANCE_STRICT specification.
- Excluded content: no rescue/lost counts, no MRR, no A4C, no prior_ranks ablation, and no canonical-analysis values.

## Final Audit Checklist

| Check | Result | Evidence |
|---|---:|---|
| Are all plotted values exactly equal to locked values? | PASS | The source CSV contains the nine locked Top-10 values and no recalculated protocol values. |
| Are all CIs exactly equal to locked values? | PASS | The eight reported CIs match the locked values; the diagnostic row has blank CI fields because no CI was reported. |
| Is the plot a horizontal forest/dot plot, not a bar chart? | PASS | Methods are rows; points show Top-10; horizontal error bars show 95% CI. |
| Are role labels placed in a right-side aligned evidence column rather than floating in the plot area? | PASS | Evidence roles are rendered in the separate right-side `Evidence role` column. |
| Is Best-of-DE+HGB labeled diagnostic only? | PASS | The method label is exactly `Best-of-DE+HGB diagnostic`; the evidence-role column says `diagnostic only`; it is shown as a hollow light-gray marker without CI. |
| Forbidden diagnostic terminology absent? | PASS | The forbidden diagnostic terms are absent from the figure, caption, data CSV, and plotting script. |
| Is the 77-feature scorer clearly post-audit locked, not fully prospective? | PASS | The evidence role is `post_audit_locked_result`, and the right-side role column says `post-audit locked result`. |
| Are rescue/lost, MRR, A4C, and prior_ranks ablation absent? | PASS | None of those endpoints or analyses are plotted or included in the CSV. The caption mentions prior_ranks only to state the pruning caveat for the locked 77-feature scorer. |
| Is the figure readable at single-column width? | PASS | Text size is at least 7 pt; method labels do not overlap; role and delta columns are separated from the data area. |
| Does it look like a benchmark performance figure rather than a PPT report slide? | PASS | The design uses a compact forest plot with aligned evidence columns, minimal annotation, subtle group separators, and no floating callouts. |
| Did the reference-style search influence only layout discipline, not copied visual assets? | PASS | Reference papers informed table-like organization, forest-plot discipline, and evidence-boundary separation only. No visual assets, colors, icons, or exact layouts were copied. |

## Locked Values

| Method | Top-10 | CI low | CI high | Group | Evidence role | Delta vs Score Blend |
|---|---:|---:|---:|---|---|---:|
| Attachment-Frequency | 0.6019 | 0.5933 | 0.6104 | base_ranker | baseline |  |
| HGB | 0.7437 | 0.7356 | 0.7516 | base_ranker | baseline |  |
| Dual Encoder (DE) | 0.8055 | 0.7986 | 0.8122 | base_ranker | baseline |  |
| Borda(DE,HGB) | 0.8384 | 0.8321 | 0.8447 | fusion_baseline | baseline |  |
| MLP (rank-only) | 0.8402 | 0.8339 | 0.8466 | fusion_baseline | baseline |  |
| Score Blend (MLP + HGB) | 0.8558 | 0.8495 | 0.8621 | fusion_baseline | strongest_pre_D4S_baseline | 0.0000 |
| Initial 82-feature scorer | 0.8851 | 0.8796 | 0.8903 | candidate_level | prospective_candidate_level_result | +0.0293 |
| Post-audit 77-feature scorer | 0.9243 | 0.9199 | 0.9287 | candidate_level | post_audit_locked_result | +0.0686 |
| Best-of-DE+HGB diagnostic | 0.8686 | not reported | not reported | diagnostic | diagnostic_only | +0.0128 |

## Output Files

| File | Status |
|---|---:|
| Figure_3_secondary_blind_performance_refined.pdf | PASS |
| Figure_3_secondary_blind_performance_refined.svg | PASS |
| Figure_3_secondary_blind_performance_refined.png | PASS |
| Figure_3_secondary_blind_performance_data.csv | PASS |
| Figure_3_caption_refined.md | PASS |
| Figure_3_reference_style_notes.md | PASS |

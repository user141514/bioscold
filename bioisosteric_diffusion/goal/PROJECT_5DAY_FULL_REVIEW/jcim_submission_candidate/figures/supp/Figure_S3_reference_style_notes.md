# Supplementary Figure S3 Reference Style Notes

## Scope

The requested literature scan was used only to extract layout discipline for a JCIM-style supplementary audit figure. No figure assets, icons, colors, molecular drawings, or specific graphical compositions were copied.

## Cheminformatics / Molecular Benchmark References

| Reference | Link | Layout principle used for Figure S3 |
|---|---|---|
| Exposing the Limitations of Molecular Machine Learning with Activity Cliffs | https://pubs.acs.org/doi/10.1021/acs.jcim.2c01073 | Local failure should be separated from aggregate performance and framed as subgroup evidence rather than a global claim. |
| Most Ligand-Based Classification Benchmarks Reward Memorization Rather than Generalization | https://arxiv.org/abs/1706.06619 | Benchmark-bias or leakage-risk evidence should be stated with disciplined labels and not overpromoted beyond the measured split. |
| GuacaMol: Benchmarking Models for De Novo Molecular Design | https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839 | Benchmark figures benefit from restrained performance comparison, direct panel titles, and minimal legends when labels already define the conditions. |
| DataSAIL: Data splitting to avoid information leakage | https://www.nature.com/articles/s41467-025-58606-8 | Leakage/audit visuals should keep separation logic explicit and avoid decorative complexity. |

## CV/DL Layout References

| Reference | Link | Layout principle used for Figure S3 |
|---|---|---|
| ResNet / Deep Residual Learning for Image Recognition | https://arxiv.org/abs/1512.03385 | Use clean module/axis alignment and avoid unnecessary arrows or extra visual encodings. |
| U-Net | https://arxiv.org/abs/1505.04597 | Keep paired structures visually balanced and make cross-panel comparisons easy to scan. |
| Attention Is All You Need | https://arxiv.org/abs/1706.03762 | Use repeated visual structure and concise labels for comparable modules/panels. |
| Mask R-CNN | https://arxiv.org/abs/1703.06870 | Express a workflow/result comparison with few visual primitives and clear panel hierarchy. |

## Applied Decisions

- Use two panels instead of one shared x-axis because the 82-feature regressions and 77-feature near-zero deltas live on different numeric scales.
- Show only five negative 82-feature strata in Panel A because this panel's role is to document local regressions.
- Show all 19 77-feature strata in Panel B because this panel's role is to document the absence of negative point-estimate deltas after pruning.
- Use fragment IDs plus n values on the y-axis; full old-fragment identities remain in Supplementary Table S5 and the source-data CSV.
- Use point-estimate language only; do not imply per-fragment statistical significance.

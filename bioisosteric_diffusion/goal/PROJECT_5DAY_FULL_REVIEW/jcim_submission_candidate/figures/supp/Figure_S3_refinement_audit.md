# Supplementary Figure S3 Refinement Audit

## Figure Contract

- Core message: the initial 82-feature scorer has five negative old-fragment point-estimate strata; the post-audit 77-feature scorer has no negative old-fragment point-estimate strata across 19 strata.
- Figure type: two-panel supplementary audit result figure.
- Backend: Python/matplotlib only.
- Source data: D4S28 fixed blind strata and D4S31 blind strata CSV files.

## Final Audit Checklist

| Check | Answer | Evidence |
|---|---:|---|
| Does Panel A clearly state that it only shows negative strata? | YES | Panel A title reads `Initial 82-feature scorer: negative strata only` and the panel annotation states `5/19 old-fragment strata negative`. |
| Does Panel B show all 19 old-fragment strata? | YES | Panel B contains F01-F19, each with an n value. |
| Are the two x-axis ranges explained in the caption? | YES | The caption states that panels use different x-axis ranges to show large 82-feature regressions and near-zero/small-positive 77-feature deltas. |
| Does the figure avoid implying statistical significance? | YES | The figure and caption use `point-estimate delta` language and do not use significance language. |
| Are long SMILES absent from the plot area? | YES | The y-axis uses compact F01-F19 labels with n values; full old-fragment identities are assigned to Supplementary Table S5 and the source-data CSV. |
| Are only key numbers labeled? | YES | Panel A labels the five negative values; Panel B labels only the four largest positive values. |
| Is heavy colored background avoided? | YES | Panel A uses only very pale x<0 shading and a clear zero line; Panel B uses no colored background. |
| Is the figure readable at single-column scale? | YES | Minimum explicit SVG font size is 7 pt; axis labels and key annotations remain readable in the PNG/PDF preview. |
| Does it fit a JCIM supplementary audit style? | YES | The design uses white background, muted red/teal, light grid lines, no legend, no decorative iconography, and minimal annotations. |
| Were reference figures used only for layout principles, not copied assets? | YES | The reference review informed alignment, grouping, and annotation discipline only; no visual assets, icons, color schemes, or figure elements were copied. |

## Numeric Checks

| Quantity | Value | Status |
|---|---:|---:|
| Number of old-fragment strata | 19 | PASS |
| 82-feature negative strata | 5 | PASS |
| 77-feature negative strata | 0 | PASS |
| Labeled 82-feature negative deltas | -1.000, -0.806, -0.293, -0.127, -0.005 | PASS |
| Labeled 77-feature positive deltas | +0.033, +0.060, +0.067, +0.077 | PASS |

## Exported Files

| File | Status |
|---|---:|
| Figure_S3_fragment_deltas_final.pdf | PASS |
| Figure_S3_fragment_deltas_final.svg | PASS |
| Figure_S3_fragment_deltas_final.png | PASS |
| Figure_S3_fragment_deltas_final_source_data.csv | PASS |
| Figure_S3_caption_final.md | PASS |
| Figure_S3_reference_style_notes.md | PASS |

# Supplementary Figure S4 Refinement Audit

## Figure Contract

- Core message: naive independent one-hot encoding can shift categorical column semantics across splits; frozen schema alignment repairs the encoding path by preserving feature meaning.
- Figure type: schematic-led supplementary audit figure with a visually secondary quantitative inset.
- Backend: Python/matplotlib only.
- Export bundle: PDF, SVG, PNG, caption, audit note, and source-data CSV.

## Layout References Reviewed

The final polish used classic computer-vision/deep-learning figures only as layout references, not as visual assets. The borrowed principles were disciplined module alignment, mirror-like pathway structure, short arrows, repeated module sizing, and clear separation of primary workflow from secondary checks.

## Final Self-Check

| Check | Answer | Evidence |
|---|---:|---|
| Are the first arrows from the start boxes fully horizontal? | YES | Both start-box arrows share a single y-coordinate from the start box center to the schema block center. |
| Are the bug path and repair path mirror-aligned? | YES | Start boxes, schema blocks, intermediate boxes, and final outcome boxes use matched x coordinates and widths across rows. |
| Is the Column 4 inset separated from the main workflow? | YES | `Audit-only Top-10 check` is placed in the lower-right fourth column, outside the main workflow arrows and boxes. |
| Is the top-row OTHER column pale enough to avoid implying a true frozen OTHER mapping? | YES | The top-row `OTHER` column uses pale gray text, pale cell borders, and empty cells. It is a ghost alignment column for layout only, not a true frozen schema field. |
| Is Delta Top-10 visually subordinate to the main outcome boxes? | YES | Delta Top-10 is small, unbolded, and placed inside the secondary inset below the main workflow. |
| Does the figure communicate bug repair rather than model contribution? | YES | The inset note states `implementation repair effect; not a model contribution`, and the main flow labels the change as schema alignment. |
| Is all text readable after single-column scaling? | YES | The minimum explicit font size in the SVG is 7 pt. |
| Do arrows avoid crossing text or tables? | YES | Arrows terminate in whitespace between modules and do not cross text or table cells. |
| Are there no added decorative elements? | YES | The figure uses white background, muted red, muted teal, gray, thin borders, and no icons, gradients, shadows, or decorative backgrounds. |

## Required Wording

| Required phrase | Status |
|---|---:|
| bug path | PASS |
| repair path | PASS |
| Naive independent encoding | PASS |
| Frozen schema alignment | PASS |
| column-semantic mismatch | PASS |
| misaligned blind prediction | PASS |
| fixed schema | PASS |
| schema-aligned blind prediction | PASS |
| column 1 changes meaning | PASS |
| unseen old=D mapped to OTHER | PASS |
| Audit-only Top-10 check | PASS |
| implementation repair effect; not a model contribution | PASS |

## Numeric Lock

| Quantity | Value | Status |
|---|---:|---:|
| Naive schema Top-10 | 0.7120 | PASS |
| Frozen schema Top-10 | 0.8851 | PASS |
| Delta Top-10 | +0.1731 | PASS |

## Exported Files

| File | Status |
|---|---:|
| Figure_S4_categorical_schema_alignment_audit_final.pdf | PASS |
| Figure_S4_categorical_schema_alignment_audit_final.svg | PASS |
| Figure_S4_categorical_schema_alignment_audit_final.png | PASS |
| Figure_S4_caption_final.md | PASS |
| Figure_S4_categorical_schema_alignment_audit_final_source_data.csv | PASS |

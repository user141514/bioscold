# Figure Style Audit

This v4 pass uses the local PaperBanana skill as a style-planning layer: its output constraints were translated into deterministic SVG edits rather than image-generation edits, because Figures 3/4 contain locked numeric evidence.

1. **Fonts consistent?** Yes. All harmonized SVG text styles are mapped to `Arial, Helvetica, DejaVu Sans, sans-serif`.
2. **Colors mapped consistently?** Yes. Navy `#0B3F99` is used for task/base evidence, teal `#087A74` for heldout/main/post-audit/triage routes, red `#B31F24` for leakage/removed/alert signals, and gray tones for neutral text, borders, and grids.
3. **Panel labels consistent?** Mostly yes. Existing panel structures were preserved; labels were harmonized through shared font and palette without redesigning panel logic.
4. **Line widths and arrows consistent?** Yes. Main colored boxes are kept near 1.0-1.2 pt, panel/table/grid strokes are lightened, and arrow strokes are normalized to a restrained 1.0-1.2 pt range where represented in SVG styles.
5. **Figure 1 content corrections applied?** Yes. Panel B now shows the same leaked transform identity (`T2`) on both train and evaluation sides, and Panel C uses `Train transforms` / `Blind transforms`.
6. **Do Figures 3/4 preserve all locked values?** Yes. The harmonization pass changes only SVG styling/text presentation; plotted values, ordering, CIs, and panel logic are not altered.
7. **Does Figure 5 preserve A4C claim boundary?** Yes. The note `A4C = computational triage only; not experimental validation.` is retained, and G2/G3/G4 remain provenance/triage strata rather than experimental toxicity or activity claims.
8. **What changed relative to v3?** Figure 2 removes the isolated Score Blend/A4C footnote from the canvas and crops excess bottom whitespace; Figure 5 further lowers formula/table visual density; Figure 1/3/4 keep their content and locked evidence.

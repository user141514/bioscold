Create a JCIM-style three-panel schematic for dual-mode provenance triage.

Scientific scope:
The figure explains exploratory replacement proposal routes and provenance strata. It must not imply experimental validation, toxicity verdicts, or expert validation. A4C is computational triage only.

Panel A: Dual-mode proposal routes.
- Conservative Mode: HGB route -> frequency-aligned proposals.
- Exploration Mode: Borda(DE,HGB) route -> exploratory proposals -> provenance strata.
- Show the two routes as parallel lanes.
- Keep labels short and aligned.

Panel B: Provenance strata.
- G4 shared: K_HGB intersection K_Borda, consensus.
- G3 DE-elevated: K_Borda minus K_HGB, similarity-supported expansion.
- G2 Borda-emergent: K_Borda minus (K_HGB union K_DE), highest novelty.
- Use restrained math typography, but do not let equations dominate the panel.

Panel C: Coverage-limited A4C triage.
- Compact table with Group, Coverage, Alert signal.
- G2: coverage 100%, alert signal 46.85%.
- G3: coverage 100%, alert signal 9.67%.
- G4: coverage 5.63%; alert signal 17.60% among covered and 94.37% unknown.
- G4 coverage-limited / unknown status must remain visually visible.
- Bottom note must be retained: A4C = computational triage only; not experimental validation.

Style:
Use the global JCIM style. Three panels should share the same sans-serif font, light-gray panel borders, muted colors, and thin table/grid lines. G2 red should indicate alert/novelty, not an experimental toxicity verdict.

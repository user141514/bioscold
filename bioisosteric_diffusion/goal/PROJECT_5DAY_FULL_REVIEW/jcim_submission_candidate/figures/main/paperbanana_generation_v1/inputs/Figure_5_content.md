Create a JCIM three-panel schematic from scratch.

Scope:
Explain dual-mode provenance triage for exploratory replacement proposals. The figure must not imply experimental validation, toxicity verdicts, expert validation, activity preservation, or wet-lab support. A4C is computational triage only.

Panel A:
Dual-mode proposal routes
- Conservative Mode: HGB route -> frequency-aligned proposals.
- Exploration Mode: Borda(DE,HGB) route -> exploratory proposals -> provenance strata.
- Two parallel lanes, clean and aligned.

Panel B:
Provenance strata
- G4 shared: K_HGB intersection K_Borda; consensus.
- G3 DE-elevated: K_Borda minus K_HGB; similarity-supported expansion.
- G2 Borda-emergent: K_Borda minus (K_HGB union K_DE); highest novelty.
- Use restrained math typography. Equations should not dominate.

Panel C:
Coverage-limited A4C triage
Table with columns: Group, Coverage, Alert signal.
Locked values:
- G2: coverage 100%, alert signal 46.85%.
- G3: coverage 100%, alert signal 9.67%.
- G4: coverage 5.63%; alert signal 17.60% among covered and 94.37% unknown.
- G4 coverage-limited / unknown status must remain visible.
- Include the note: A4C = computational triage only; not experimental validation.

Style:
Use sans-serif font, light-gray panel and table lines, muted navy/teal/red, white background. G2 red indicates alert/novelty, not experimental toxicity.

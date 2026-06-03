Create a JCIM method architecture figure from scratch.

Scope:
Explain the candidate-level scoring pipeline and evidence hierarchy. The figure should look like a clean ML architecture schematic, not a chemical workflow or presentation slide. Do not show performance numbers, rescue/lost counts, A4C rates, or long explanatory paragraphs.

Main pipeline:
Query-candidate pair
(old fragment, attachment, candidate)
flows to parallel base-ranker evidence:
- Attachment-Frequency
- HGB
- Dual Encoder
- Borda(DE,HGB)
- MLP / Score Blend

These signals merge into the central visual module:
Candidate-level feature matrix
- model outputs
- train-derived priors
- molecular descriptors
- frozen categorical schema

Then:
HistGB scorer -> Final ranking.

Audit ribbon:
Place a compact post-audit ribbon below the feature matrix/scorer, visually attached but secondary:
Initial 82-feature scorer
F1-F9 + categorical + prior ranks
[prospective]
-> secondary blind diagnostics
-> remove prior ranks
[non-transferable shortcut]
-> Post-audit 77-feature scorer
F1-F9 + categorical
[post-audit locked]

Important claim boundaries:
- Score Blend appears only inside base-ranker evidence as the strongest pre-D4S baseline.
- A4C should not appear as a standalone module in this figure.
- The 77-feature scorer is post-audit locked, not fully prospective.

Style:
Use a restrained JCIM / ML architecture style: white background, sans-serif font, thin arrows, clean grid alignment, candidate-level feature matrix as the visual center, navy for base evidence, teal for candidate-level/post-audit, red only for removed prior ranks.

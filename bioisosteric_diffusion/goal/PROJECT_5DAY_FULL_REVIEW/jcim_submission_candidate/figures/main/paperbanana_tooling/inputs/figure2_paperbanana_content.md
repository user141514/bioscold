Create a method architecture schematic for a JCIM manuscript.

Scientific scope:
The figure explains a candidate-level scoring pipeline and evidence hierarchy. It should read like an ML architecture diagram, not like a chemical workflow or slide deck. Do not include performance numbers, rescue/lost counts, A4C rates, or long explanation blocks.

Main pipeline:
Query-candidate pair
(old fragment, attachment, candidate)
flows to parallel base-ranker evidence:
- Attachment-Frequency
- HGB
- Dual Encoder
- Borda(DE,HGB)
- MLP / Score Blend

These signals merge into the visual center:
Candidate-level feature matrix
- model outputs
- train-derived priors
- molecular descriptors
- frozen categorical schema

Then:
HistGB scorer -> Final ranking.

Audit ribbon:
Place a compact ribbon below the feature matrix/scorer, attached to the main pipeline but visually secondary.
Initial 82-feature scorer
F1-F9 + categorical + prior ranks
[prospective]
-> secondary blind diagnostics
-> remove prior ranks
[non-transferable shortcut]
-> Post-audit 77-feature scorer
F1-F9 + categorical
[post-audit locked]

Score Blend and A4C:
Score Blend may appear only inside base-ranker evidence. A4C should not enter the main pipeline; if mentioned, keep it in the caption rather than as a standalone card.

Style:
Use the global JCIM style. Candidate-level feature matrix must be the visual center. Base rankers should be parallel evidence inputs. All modules should align to a clean grid. Each box should contain at most three lines of text.

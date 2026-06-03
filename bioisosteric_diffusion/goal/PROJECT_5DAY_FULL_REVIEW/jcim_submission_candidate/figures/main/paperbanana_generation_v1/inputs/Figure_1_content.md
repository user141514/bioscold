Create a JCIM introduction figure from scratch.

Scope:
Only explain the closed-vocabulary fragment replacement ranking task and leakage-controlled evaluation. Do not show any model architecture, base rankers, Borda, MLP, Score Blend, HistGB, feature pruning, A4C, performance numbers, or rescue/lost counts.

Required structure:

A. One query, many candidates
- Show a single query defined by an old fragment and attachment signature.
- Show a closed train-derived candidate vocabulary with approximately 150 replacement candidates.
- Show several candidates, with two true replacements marked by small stars.
- Show a ranked list and a Top-10 bracket.
- Use short labels: closed vocabulary, multi-positive ranking, Top-10 hit.
- Include the idea: a Top-10 hit is counted if any true replacement appears in the top 10.

B. Random split leakage
- Show train examples and evaluation examples.
- The same transform identity must appear on both sides.
- Use Train: T1, T2, T3 and Evaluation: T2, T4, T5.
- Highlight T2 in both columns.
- Conclude with the short label: shared transform identity -> leakage risk.

C. Transform-heldout protocol
- Show train transforms and blind transforms as disjoint sets.
- Label zero old-fragment plus attachment-signature overlap.
- Include a small protocol rail: Train -> Development/calibration -> Secondary blind.
- Add a small note: no blind tuning.

Style:
Use a restrained JCIM / cheminformatics manuscript style: white background, sans-serif font, light gray panel borders, navy for task, muted red for leakage, muted teal for heldout protocol, no gradients, no shadows, no dashboard/KPI styling, no decorative icons.

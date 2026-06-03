Create an introduction-embedded benchmark overview figure for a JCIM manuscript.

Scientific scope:
The figure explains only the task definition and leakage-controlled benchmark design for closed-vocabulary fragment replacement ranking. It must not show model details, base rankers, Borda, MLP, Score Blend, HistGB, 82-feature or 77-feature scorers, prior-rank pruning, A4C, performance numbers, or rescue/lost analysis.

Required structure:
Panel A: One query, many candidates.
- A query is defined by an old fragment and an attachment signature.
- Candidates come from a closed train-derived vocabulary, approximately 150 candidates per query.
- Multiple true replacements may exist for one query.
- A ranked list is produced.
- Top-10 hit if any true replacement appears in the top 10.
- Include compact labels: closed vocabulary, multi-positive ranking, Top-10.

Panel B: Random split leakage.
- Show train examples and evaluation examples.
- The same transform identity must appear on both sides.
- Use Train: T1, T2, T3 and Evaluation: T2, T4, T5.
- Highlight T2 on both sides.
- Use the short conclusion: shared transform identity across train/evaluation -> leakage risk.

Panel C: Transform-heldout protocol.
- Show train transforms and blind transforms as disjoint sets.
- Label zero (old fragment, attachment signature) overlap.
- Include a compact protocol rail: Train -> Development/calibration -> Secondary blind.
- Train builds vocabulary; development/calibration tunes protocol only; secondary blind is primary evaluation.
- Include small note: no blind tuning.

Style:
Use the global JCIM style. Panel A should be visually largest. Panels B and C should be smaller and balanced. Avoid poster-like heavy panel borders.

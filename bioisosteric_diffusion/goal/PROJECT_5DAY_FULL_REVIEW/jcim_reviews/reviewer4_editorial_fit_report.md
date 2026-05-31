# Reviewer 4 - JCIM Editorial Fit Review

## Verdict

**Major Revision Required.** The work fits JCIM, but the draft still looks like an internal project report rather than a journal manuscript.

## Critical Findings

1. **No Data and Software Availability section.**
   - JCIM expects clear data/code availability.
   - Current script paths are scattered in Methods but there is no repository, license, dependency, or reproducibility statement.

2. **No figures.**
   - A JCIM paper needs at least:
     - benchmark/split schematic;
     - ranking pipeline schematic;
     - Top-K performance curve;
     - chemical examples;
     - feature-family ablation or importance plot.

3. **No chemical examples.**
   - The reader never sees a concrete old_fragment to replacement example.
   - Chemical credibility is weak without representative successes/failures.

4. **Title foregrounds internal process rather than contribution.**
   - "Post-Audit Feature-Pruned" sounds like a lab-note caveat.
   - Better title direction: "Leakage-Controlled Closed-Vocabulary Fragment Replacement Ranking with Candidate-Level Scoring."

## Major Findings

1. Internal IDs D4S31 and D4S28R should not be model names.
   - Use descriptive names such as `Feature-Pruned Candidate Scorer` and `Initial 82-Feature Scorer`.

2. Contribution hierarchy is inflated.
   - Real contributions are benchmark and scorer.
   - Protocol separation and A4C should be enabling safeguards, not co-equal contributions.

3. Abstract is over-caveated.
   - It is honest but reads defensively.
   - Restructure into: problem, method, result, caveat, scope.

4. Related work is descriptive, not argumentative.
   - Need explicitly state what NeBULA, DeepBioisostere, GraphBioisostere, SwissBioisostere, and CReM do not test relative to this benchmark.

5. Prior_ranks story is repeated too many times.
   - Methods should define and design.
   - Results should report numbers.
   - Discussion should interpret.

## Minor Findings

- Avoid boldface emphasis in Methods.
- Attachment-Frequency is not a lower bound; call it a frequency baseline.
- Table 1 diagnostic row needs CI or explanation.
- Standardize "secondary blind" vs "blind."
- Keep repository paths out of main Methods if they belong in Data/Code Availability.

## Recommended Edits

1. Add Figure 1: transform-heldout split and scorer pipeline.
2. Add Figure 2: Top-K curves.
3. Add Figure 3: representative chemical examples.
4. Add Figure 4: feature ablation or importance.
5. Add Data and Software Availability.
6. Rename internal model IDs in prose.
7. Rewrite title and abstract for JCIM.


# Reviewer 1 - JCIM Methods Review

## Verdict

**Major Revision.** The benchmark and audit trail are promising, but the current manuscript still reads as if a post-audit model is the main blind result. JCIM reviewers will require clearer protocol hierarchy and more reproducible feature definitions.

## Critical Findings

1. **"Blind" and "post-audit" remain in tension.**
   - D4S31 Top-10 = 0.9243 was obtained after blind-set diagnostics motivated prior_ranks removal.
   - Under a strict prospective protocol, D4S28R Top-10 = 0.8851 is the last fully prospective blind model.
   - Required fix: either make D4S28R the primary prospective blind result and D4S31 the post-audit analytical upgrade, or stop using "blind" as the primary evidentiary label for D4S31.

2. **P3/P4 prior definitions are not reproducible from the text.**
   - P3 is described as cluster-conditioned, but clustering algorithm, similarity metric, thresholds, and assignment rules are missing.
   - P4 is described as probability transfer from chemically similar training fragments, but the similarity function, neighbor set, smoothing, and transfer equation are missing.
   - Required fix: add Supplementary Methods formulas and a table for P0/P1/P3/P4.

3. **The central ablation delta lacks uncertainty.**
   - The +0.0392 prior_ranks removal effect has no paired bootstrap CI.
   - If it remains a central methods claim, compute CI; otherwise downgrade Table 2 and Section 4.2 to descriptive post-audit analysis.

## Major Findings

1. Missing benchmark statistics:
   - train/development sizes;
   - candidate vocabulary size;
   - candidates per query;
   - positives per query;
   - single-positive vs multi-positive distribution;
   - old-fragment and attachment-signature counts per split.

2. The shortcut interpretation may conflate non-transferable rank semantics with reduced model capacity.
   - A random-drop-5-feature negative control would strengthen the causal claim.
   - If not run, explicitly acknowledge capacity reduction as an alternative explanation.

3. Supplementary evidence traceability is incomplete.
   - Missing or incomplete MRR table and per-fragment joined table should be resolved before submission.

4. G4 A4C lower-bound reporting is weak.
   - With 94.37% unknown status, the 0.99% lower bound is not a reliable group-wide estimate.
   - Reword as "no reliable G4-wide alert estimate is possible."

## Minor Findings

- "Natural lower bound" for Attachment-Frequency is misleading; use "strong empirical frequency baseline."
- Clarify whether GroupKFold is only a development diagnostic or contributes to the final model.
- The 1:5 diagnostic decoy set is mentioned but not reported.
- Add script output artifact names, not only script paths.

## Recommended Edits

1. Add a dataset summary table in Methods.
2. Add a P0/P1/P3/P4 prior-definition appendix.
3. Add paired bootstrap CI for D4S28R to D4S31.
4. Move D4S31 from "fully prospective blind" framing to "post-audit locked upgrade."
5. Clarify A4C G4 missingness more aggressively.


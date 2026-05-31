# Reviewer 3 - JCIM ML and Statistics Review

## Verdict

**Major Revision.** The statistical infrastructure is good for the main Score Blend comparison, but underdeveloped for the ablation and reliability claims that now carry the manuscript.

## Critical Findings

1. **No CI for the prior_ranks ablation.**
   - The +0.0392 D4S28R-to-D4S31 gain is central but lacks uncertainty.
   - Required fix: paired query bootstrap CI for the ablation delta.

2. **No uncertainty for rescue/lost counts.**
   - 1,016 rescue / 101 lost and 596-to-101 lost reduction are descriptive counts.
   - Required fix: bootstrap CI for net gain and rescue/lost ratio, or keep them explicitly descriptive.

3. **D4S31 post-audit selection implies test-set reuse.**
   - Blind labels were used to identify fragment-specific degradation, then features were changed.
   - D4S31 Top-10 = 0.9243 may be optimistically biased.
   - Required fix: fresh blind split, nested CV, optimism correction, or downgrade D4S31 from primary performance claim.

4. **No multiple-comparison treatment.**
   - Table 1 compares many methods.
   - Ablation compares many feature families.
   - Per-fragment analysis compares 19 strata.
   - Required fix: Holm/FDR discussion or explicit exploratory-only framing.

## Major Findings

1. Secondary metrics are underreported.
   - Top-1, Top-5, Top-20, Top-50, and MRR are defined but not presented for the main result.

2. Best-of-DE+HGB diagnostic invites confusion.
   - It is non-deployable and should be separated from deployable methods or heavily labeled.

3. "All 19 strata at parity or better" is a point-estimate claim.
   - Per-stratum CI and multiplicity should be reported if this remains a reliability claim.

4. The 8-to-19 old-fragment shift is central but underdeveloped.
   - With only 8 development old fragments, model-selection stability is questionable.

5. Frozen schema bug timeline needs clarification.
   - If bug repair was discovered via blind labels, this is another blind-set reuse.

## Minor Findings

- Make paired bootstrap mechanics explicit: same resampled query set for both methods.
- Explain `class_weight='balanced'` plus 5:1 negative subsampling.
- Justify Score Blend lambda = 0.95.
- Expand A4C acronym.
- Be consistent about preprint labeling in references.

## Recommended Edits

1. Add CIs for ablation deltas.
2. Add CIs for rescue/lost net gain and ratio.
3. Add a full secondary-metric table.
4. Separate deployable methods from diagnostic upper bounds.
5. Add an explicit post-selection optimism paragraph.


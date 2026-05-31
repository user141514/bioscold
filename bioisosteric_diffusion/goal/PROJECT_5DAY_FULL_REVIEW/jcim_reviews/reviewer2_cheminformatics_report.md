# Reviewer 2 - JCIM Cheminformatics Review

## Verdict

**Major Revision Required.** The technical ranking pipeline is useful, but the manuscript must be reframed as structure-derived MMP replacement recovery unless activity evidence is added.

## Critical Findings

1. **The central task is not activity-validated bioisosteric replacement.**
   - The manuscript admits labels are structure-derived and do not establish activity preservation.
   - Bioisosteric replacement normally implies functional or property equivalence, while this benchmark recovers observed MMP substitutions.
   - Required fix: frame as closed-vocabulary MMP-derived fragment replacement ranking, not bioisostere prediction.

2. **PAINS/Brenk filtering creates circularity with A4C.**
   - Methods state that molecules with PAINS/Brenk alerts are removed during data construction.
   - A4C then applies PAINS/Brenk alerts to interpret output strata.
   - This means A4C alert rates are conditional on a pre-cleaned vocabulary and likely underestimate alert burden in an unfiltered catalogue.
   - Required fix: explicitly state this conditionality and remove any absolute alert-burden interpretation.

## Major Findings

1. Decoy construction is chemically underjustified.
   - Non-cognate scaffold decoys may be valid but unobserved replacements.
   - Property matching between positives and decoys is not shown.
   - The 1:5 diagnostic set is not reported.

2. Fragmentation rules are underspecified.
   - Need exact MMP algorithm variant, number of cuts, recursion depth, scaffold identity definition, stereochemistry handling, and fragment statistics.

3. Attachment compatibility predicate is a black box.
   - Need formal definition of attachment signature, valence/bond-order checks, aromaticity/ring handling, and candidate-pass statistics.

4. Closed-vocabulary limitation should be more prominent.
   - The model only scores fragments seen in the training replacement vocabulary.
   - It cannot propose truly novel replacements.

5. Transform-heldout controls identity leakage but not chemical-similarity leakage.
   - Consider reporting train-blind nearest-neighbor similarity over old fragments or transform environments.

## Minor Findings

- G4 A4C coverage of 5.63% is too sparse for a reliable group-wide alert estimate.
- The abstract's "596 to 101" phrasing can be misread; clarify that it is D4S28R-to-D4S31 lost-hit reduction.
- The paper needs basic MMP database statistics: total MMPs, unique fragments, attachment signatures, fragment sizes.

## Recommended Edits

1. Rename contribution around "MMP-derived closed-vocabulary fragment replacement ranking."
2. Add an MMP extraction and attachment-compatibility appendix.
3. Add a pre-filter caveat to A4C.
4. Add property-distribution comparison for positives vs decoys.
5. Move any bioisostere language behind strong structure-derived caveats.


# V6 Methods Patch: Table Renumbering and Alignment Edits

## Scope
This patch documents the V6 changes applied to the methods sections of main_manuscript.md to produce manuscript_v6_aligned.md. No new methods, experiments, or algorithmic content were added.

## 1. Table Renumbering

### Table M1 (was M1b): Fresh Blind2 Primary Candidate Matrix
- Caption updated from "Fresh Blind2 full-161 candidate matrix used for the primary prospective evaluation" to include "Table M1." prefix.
- Position: Moved before the original secondary-blind diagnostic table so Fresh Blind2 appears first in the document.
- Content unchanged: Train2 (93,083 queries), Dev2 (27,415), Blind2 (17,058), all with 161 candidates/query.

### Table M2 (was M1): Original Secondary-Blind Diagnostic Candidate Matrix
- Caption updated from "Candidate-matrix statistics for the closed-vocabulary ranking benchmark" to "Original secondary-blind diagnostic candidate-matrix statistics (retained as diagnostic history)."
- Position: Moved after Fresh Blind2 table.
- Content unchanged.

### Table M3 (was M2): Leakage and Overlap Verification (Expanded)
- Caption updated from "Leakage verification: overlap counts of (fold, sigma) query-side transform identities" to "Leakage and overlap verification across split pairs."
- Expanded columns: Added old-fragment overlap, full old-to-replacement overlap, and attachment-signature overlap.
- Fresh Blind2 overlap values added:
  - Train2/Dev2: query-side 0, old-fragment 53, full old-to-replacement 393, attachment-signature 5
  - Train2/Blind2: query-side 0, old-fragment 46, full old-to-replacement 430, attachment-signature 5
  - Dev2/Blind2: query-side 0, old-fragment 17, full old-to-replacement 188, attachment-signature 5
- Original split pairs: old-fragment/full-overlap/attachment columns marked as not computed using current conventions.

### Table M4 (was M3): Feature Families
- Caption updated from "Table M3." to "Table M4."
- Content unchanged.

### Table M5 (was M4): Protocol Separation
- Caption updated from "Table M4." to "Table M5."
- Content unchanged.

## 2. Cross-Reference Updates
All text references to table numbers updated:
- "Table M1b" changed to "Table M1" (3 occurrences)
- "Table M3 provides each family" changed to "Table M4 provides each family"
- "Table M2" in leakage control section changed to "Table M3"

## 3. Section 3.7 (A4C Supplementary Workflow Diagnostics)
- Replaced with task-required exact wording: "A4C annotations are reported only in Supplementary Information as workflow-level diagnostics. They are not used for fitting, feature selection, ranking claims, activity validation, safety scoring, or medicinal-chemistry validation."

## 4. Related Work A4C Paragraph (Section 2)
- Replaced provenance-strata language with: "workflow-level structural alert annotations, which are reported only in the Supplementary Information as diagnostic aids rather than as validation endpoints."

## 5. Leakage Language
- Updated leakage verification intro text to specify four overlap dimensions.
- Added: "The split removes exact query-side old-fragment/attachment-context overlap. It does not remove all old-fragment, replacement-triple, or attachment-signature relatedness."

## 6. Data Availability (Section: Data and Software Availability)
- Added paragraph listing Fresh Blind2 evidence artifacts explicitly.
- Added references to: Fresh Blind2 split artifacts, full-161 candidate matrix, HistGB82/77 scorer outputs, grouped-uncertainty audit tables, candidate-matrix sensitivity diagnostics, LambdaRank diagnostic outputs, and Claim Map V5.

## 7. No-Add Constraints Enforced
- No new experiments added.
- No new algorithm track opened.
- 77-feature not re-promoted.
- LambdaRankCat not made main method.
- A4C not expanded.
- No new evidence invented.

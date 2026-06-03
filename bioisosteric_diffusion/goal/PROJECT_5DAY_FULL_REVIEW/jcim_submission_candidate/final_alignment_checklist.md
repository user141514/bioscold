# V6 Final Alignment Checklist

| # | Item | Answer | Notes |
|---|------|--------|-------|
| 1 | Is Fresh Blind2 clearly the primary prospective evaluation? | **Yes** | Abstract, Introduction, Methods (Sections 3.2, 3.6.4), Results (4.1-4.2), Discussion (5.1), Conclusion all identify Fresh Blind2 as primary. Table 5 labels it "Primary prospective evaluation." |
| 2 | Is HistGB82 clearly the main prospective Top-10 HistGB result? | **Yes** | Abstract states "the 82-feature HistGB candidate-level scorer achieved Top-10 = 0.8480". Table 6 labels HistGB82 as the prospective candidate-level scorer. Discussion 5.1 calls it "the primary prospective Top-10 HistGB result." |
| 3 | Is HistGB77 clearly diagnostic, not main method? | **Yes** | Table 6 labels HistGB77 as "no-prior-rank diagnostic." Section 4.4 concludes prior-rank deletion does not prospectively replicate. Conclusion treats it as "diagnostic of unstable feature transfer." |
| 4 | Is old secondary blind demoted to diagnostic history? | **Yes** | Table 2 caption: "retained as diagnostic history." Table 5 labels it "Diagnostic history, post-audit evidence, and stress-test context." Old 77-feature result (0.9243) relegated to Supplementary Table S1. |
| 5 | Is prior_ranks deletion described as non-replicated for Top-10? | **Yes** | Section 4.4: "Prior-Rank Deletion Does Not Prospectively Replicate as a Top-10 Improvement." Discussion 5.2: "did not replicate a Top-10 advantage." |
| 6 | Is C_i defined consistently with the full-vocabulary policy? | **Yes** | Section 3.1: C_i = V_train^(p), protocol-indexed. Fresh Blind2 uses Train2-derived 161-fragment vocabulary. Attachment signature is query context / feature input, not primary hard filter. |
| 7 | Is Fresh Blind2 candidate matrix table placed before old secondary-blind table? | **Yes** | Table 1 (Fresh Blind2) appears before Table 2 (original secondary-blind) in the document. |
| 8 | Does leakage table quantify residual relatedness beyond query-side key overlap? | **Yes** | Table 3 includes columns for old-fragment overlap, full old-to-replacement overlap, and attachment-signature overlap. Fresh Blind2 values are populated. Text states: "does not remove all old-fragment, replacement-triple, or attachment-signature relatedness." |
| 9 | Is A4C supplementary-only? | **Yes** | Section 3.7: "A4C annotations are reported only in Supplementary Information." Explicitly states they are not used for fitting, feature selection, ranking claims, activity validation, safety scoring, or medicinal-chemistry validation. |
| 10 | Is LambdaRankCat future direction only? | **Yes** | Section 4.7: "Learning-to-Rank Diagnostics Suggest a Future Architecture Path." Discussion 5.4: "a future pre-specified architecture path, not the current main method." |
| 11 | Are activity/calibration diagnostics explicitly non-validating? | **Yes** | Section 4.8 states that the diagnostic does not support activity-outcome, biological-validation, or medicinal-chemistry prioritization claims. Calibration outputs are described as ranking scores rather than absolute probabilities. |
| 12 | Does Data Availability point to V5/Fresh Blind2 evidence? | **No - submission blocker** | Data Availability points to the intended public repository tag `v5-fresh-blind2-evidence-lock-20260603`, but remote tag visibility could not be verified. HTTPS push/ls-remote fails because `github.com:443` is unreachable, SSH transport is reachable but no GitHub SSH key is configured, and the public tag URL currently returns 404. The public tag must be pushed and verified before submission. |
| 13 | Are all internal project labels removed from main text? | **Yes** | Forbidden-term scan confirms no internal project labels or run labels in main text. |
| 14 | Does the conclusion end with benchmark-audit framing? | **Yes** | Final sentence: "The central benchmark-audit lesson is to remove exact query-side transform memorization first, then ask which ranking signals still transfer." |

## Summary
Manuscript-facing alignment checks pass, but the package is not submission-ready until the public repository tag is pushed and verified. See `final_blockers.md`.

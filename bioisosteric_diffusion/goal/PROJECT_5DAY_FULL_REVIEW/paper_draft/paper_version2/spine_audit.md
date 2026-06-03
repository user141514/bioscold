# Spine Audit

## Scope Note

This file audits the manuscript spine and framing. It is not a substitute for data, reference, table, or figure-caption verification. The claim-level evidence inventory for numerical and diagnostic statements is provided in `claim_to_evidence_map.md`.

## Checklist

1. Yes. The central question is explicit by the end of the Introduction's third paragraph: how can structure-derived fragment replacement ranking be evaluated without rewarding transform memorization or unstable shortcut features?
2. Yes. The Abstract states that the 82-feature scorer is prospective and that the 77-feature scorer is post-audit locked.
3. Yes. The Abstract states that the benchmark measures structure-derived recovery, not activity preservation.
4. Yes. The Introduction identifies leakage-controlled evaluation as the JCIM pain point, specifically transform-level memorization and shortcut features under fragment distribution shift.
5. Yes. The Introduction orders the core contributions as benchmark, candidate-level scoring, and prior-rank shortcut diagnosis.
6. Yes. A4C is presented as coverage-limited computational triage and not as validation or safety scoring.
7. Yes. Query-level routing is mentioned in Discussion 5.1 only as a secondary design observation.
8. Yes. Locked values are preserved: 13,347; 0.8851; 0.9243; [0.9199, 0.9287]; 0.8558; 0.0686; [0.0638, 0.0733]; +0.0393; [0.0354, 0.0432]; 596 to 101; 5.9×; G2 100% and 46.85%; G3 100% and 9.67%; G4 5.63%, 17.60%, and 94.37%.
9. Yes. The 77-feature scorer is never described as fully prospective; the text says it is post-audit locked and post-selection.
10. Yes. prior_ranks pruning is never described as pre-registered; the text says it was identified after blind diagnostics.
11. Yes. No unsupported performance claim was introduced. Secondary Top-1, MRR, and block-level diagnostic claims are listed in `claim_to_evidence_map.md` with evidence sources.
12. Yes. The Conclusion ends with the benchmark-audit framing: remove transform memorization first, then ask which ranking signals still transfer.
13. Yes. The Abstract, Introduction, and Conclusion state that labels or recovery are structure-derived and do not establish activity preservation.
14. Yes. The rewrite avoids the vague slogan that the main contribution is a deletion. prior_ranks pruning is framed as a post-audit mechanism finding.
15. Yes. The rewrite preserves caution while making the core contribution clearer: prospective 82-feature evidence, post-audit 77-feature evidence, and workflow-only A4C are separated.

## Verdict

Pass. No checklist item failed.

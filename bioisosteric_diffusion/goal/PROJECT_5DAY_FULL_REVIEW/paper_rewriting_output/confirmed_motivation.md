# Confirmed Motivation

**Status:** CONFIRMED by user
**Date:** 2026-06-02

## Chosen Motivation

**Option A — Benchmark-driven methods paper with prior_ranks shortcut audit as primary secondary thread.**

Fragment replacement ranking needs leakage-controlled evaluation because random splits enable shortcut memorization of transform identities. We introduce a transform-heldout benchmark (13,347 blind queries) and a candidate-level HistGB scorer (77 features, Top-10 = 0.9243). The central methodological finding is that the most impactful feature-engineering decision is a deletion: removing five per-query prior-rank features eliminates a non-transferable shortcut (+0.0393 over 82-feature scorer). The dual-mode A4C workflow provides provenance-stratified computational triage as a supporting workflow feature.

## Narrative Hierarchy

1. **Primary spine:** Transform-heldout benchmark + candidate-level scorer (Arc A)
2. **Strongest secondary thread:** Prior_ranks shortcut deletion as mechanism finding (Arc B)
3. **Supporting:** Dual-mode A4C workflow (Arc C) — keep current scope, no expansion
4. **Elevated from Discussion:** Candidate-level > query-level routing insight (Arc D) — consider more prominent Results subsection placement

## Rejected Options and Why

- **Option B alone:** Shortcut learning audit is too narrow to carry JCIM paper alone; better as mechanism thread within benchmark arc
- **Option C alone:** A4C has sparse G4 coverage and unvalidated alerts — can't support full paper
- **Option D alone:** Architecture principle is interesting but insufficiently developed for primary contribution

## Scope Limits and Forbidden Overclaims

- 77-feature result remains post-audit locked finding, NOT prospective pre-registered result
- A4C alerts are computational triage signals only, NOT medicinal-chemistry validation
- Single data source (ChEMBL37K) limits generalizability — must stay acknowledged
- Structure-derived labels; no activity preservation claims
- No "first," "novel," or "state-of-the-art" without quantitative justification
- Preserve all quantitative results and CIs as locked numbers
- Maintain protocol separation (development / secondary blind / canonical / A4C workflow)

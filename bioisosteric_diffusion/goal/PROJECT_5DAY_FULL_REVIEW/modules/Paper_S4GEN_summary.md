# Module 8 — Paper-S4GEN Summary

## Starting Problem

The S4GEN manuscript needed to be converted from an internally-coded, project-internal draft into a polished general computational-journal paper suitable for submission venues like BIB, JCIM, J. Cheminformatics, or Bioinformatics. The starting point was the S3TEXT journal-facing manuscript, which was already clean, had real citations, and used standard 7-section structure. The task was light polish: strengthen the algorithm narrative, add missing provenance label definitions, restore a complementary finding from S4GEN analysis, and ensure the manuscript reads as a general computational methods paper (not Nature-style, not project-internal).

---

## Key Findings

### Manuscript Production

- **Final verdict**: A — PAPER_S4GEN_READY_FOR_FIGURE_AND_FORMAT_FINALIZATION.
- **Version**: S4GEN v2, based on S3TEXT (not the discarded v1 Nature-style rewrite).
- **Seven standard sections preserved**: Abstract, Introduction, Related Work, Methods, Results, Discussion, Conclusion.
- **Provisional title**: "A Dual-Mode Workflow for Scaffold-Conditioned Fragment Replacement Proposal via Neural-Empirical Rank Fusion."
- **Abstract**: ~200 words, within 180-230 target.

### Nature-Style v1 Rejected

- The v1 Nature-style rewrite was **DISCARDED** because it changed the title, compressed the abstract, restructured sections, and violated task.md policy.
- v2 restores the S3TEXT-base: standard structure, full abstract, general journal format.

### Claim Safety

- **Forbidden claim scan**: CLEAN. No activity-preserving prediction, wet-lab validation, review-safe production system, open-vocabulary SOTA, or universal improvement claims.
- **Allowed claim verification**: ALL PRESENT AND CORRECTLY SCOPED.
- **Internal code scan**: CLEAN. Zero internal project codes (D4/D4S/Paper-S*/D4P1/etc.) found in manuscript-facing text.
- **Terminology compliance**: PASS. All journal-facing terms used per task.md policy.
- **Protocol role separation**: PASS. Blind protocol = primary performance claim. Canonical analysis = robustness/mechanism only. Dual-mode risk-stratification = workflow/risk interpretation only.

### Numerical and Citation Integrity

- **49 numerical values preserved** (verified by CSV cross-check against S3TEXT).
- **14 citation keys preserved** (none added, removed, or modified): Zdrazil2023, Hussain2010, Griffen2011, Polishchuk2020, Dwork2001, Patani1996, Wirth2012, Ertl2020, Kim2024, Masunaga2026, Cormack2009, Baell2010, Brenk2008, Efron1979.

### Changes Applied (Allowed per task.md)

- **Methods Section 3.3**: Added pipeline stage framing + design rationale for each component (why parameter-free matters under transform-heldout).
- **Methods Section 3.6**: Added G2/G3/G4 provenance label definitions at first use (previously deferred to Results).
- **Discussion Section 5**: Added feature-engineering complementary finding (5 prior_rank features, 596 to 6 hit-loss reduction, blind Top10 0.885 to 0.924 improvement).
- **Section 2.5**: Typo fix (capitalization).
- **Introduction**: Added one sentence on why parameter-free matters under transform-heldout constraint.
- **Minor sentence flow improvements** throughout.

### Reviewer Risk Assessment

- **Overall risk**: LOW.
- **Main reviewer risks**: (a) perceived simplicity of Borda, (b) lack of activity validation.
- **Both risks are explicitly addressed** in the manuscript text.
- Reviewer-facing risk memo documents 10 Q&A covering all anticipated reviewer objections.

### Remaining Pre-Submission Tasks

1. Create Figure 1 (framework overview: pipeline topology + dual-mode workflow).
2. Format Table 1 for target journal style.
3. Prepare .bib reference file from 14 citation keys.
4. Draft data/code availability statements (if required by target journal).
5. Add author list and affiliations.

### Target Journal Realism

BIB / JCIM / Journal of Cheminformatics / Bioinformatics are realistic targets for a computational benchmark and method paper with this scope.

---

## Key Files

| File | Path |
|------|------|
| Polished Manuscript | `plan_results/routeA_paper_s4gen_general_journal_polish/paper_s4gen_polished_manuscript.md` |
| Verdict | `plan_results/routeA_paper_s4gen_general_journal_polish/PAPER_S4GEN_GENERAL_JOURNAL_POLISH_VERDICT.md` |
| Decision Log | `plan_results/routeA_paper_s4gen_general_journal_polish/MAIN_DECISION_LOG.md` |
| Claim Safety Report | `plan_results/routeA_paper_s4gen_general_journal_polish/paper_s4gen_claim_safety_report.md` |
| Reviewer Risk Memo | `plan_results/routeA_paper_s4gen_general_journal_polish/paper_s4gen_reviewer_risk_memo.md` |

---

## Core Numbers

N/A (paper module — no new experimental numbers; all 49 numerical values from S3TEXT preserved).

The key results reported in the manuscript (unchanged):
- Blind Borda(DE,HGB) Top10 = **0.8384**.
- Blind Borda vs HGB delta = **+0.0947**, CI [0.0890, 0.1008].
- Blind DE Top10 = **0.8055** (strongest single model).
- Blind MLP MRR gain = **+0.0565** over Borda, CI [0.0512, 0.0614].
- Blind MLP Top10 gain = **+0.0018** over Borda, CI [-0.0005, 0.0040] (not significant).
- Canonical Borda Top10 = **0.7642**, gain over HGB = **+0.0425**.
- G2 alert rate = **46.85%**, G3 = **9.67%**, G4 = **0.99%**.

---

## Verdict

**S4GEN v2 is a clean, general-journal-facing computational methods paper with explicit claim boundaries, preserved numerical truth, and improved algorithm construction narrative. It is ready for figure creation and format finalization. The reviewer risk is low, all forbidden claims are absent, and all protocol roles are correctly separated.**

---

## Retracted/Negated Old Conclusions

- **Nature-style v1 rewrite was discarded** because it changed title, compressed abstract, and restructured sections inappropriately. v2 restores standard general-journal format.
- **The "paper-main protocol" terminology was removed** and replaced with "secondary blind evaluation protocol" to avoid confusing reviewers.
- **Borda was downgraded** from the "primary fusion method" (S3TEXT position) to "strong fixed rank-fusion Top10 proposal baseline" in the S4GEN text, with explicit caveat that it is not the only method.

---

## Still-Credible Final Conclusions

1. **S4GEN v2 is submission-ready** pending figure creation and format finalization.
2. **All 49 numbers and 14 citations are correct** and unchanged from S3TEXT.
3. **The paper is claim-bounded** — no forbidden claims, all limitations explicit.
4. **Protocol separation is clean** — blind = primary, canonical = mechanism, dual-mode = workflow.
5. **Borda(DE,HGB) remains the core methodological contribution** — a parameter-free fusion demonstrating DE-HGB complementarity.
6. **Reviewer risk is low** — Borda simplicity and activity-validation gap are both explicitly addressed.
7. **Figure 1 is the highest-priority pre-submission task** for improving first-impression clarity.

---

## Impact on Paper

- The S4GEN manuscript is the **primary paper line** for this project.
- It establishes the **Borda+DE+HGB benchmark claim** with full protocol separation.
- It positions the work as a **computational methods/benchmark paper**, not an activity-validation study.
- The S4GEN v2 **supersedes** all earlier manuscript versions (S3TEXT, v1 Nature-style).
- The feature-engineering finding (5 prior_rank features removed) was added to Discussion as a **complementary insight**, not a primary claim.
- The G2/G3/G4 definitions were moved to Methods Section 3.6 for **first-use clarity**.
- **Figure, formatting, and author metadata** remain the only gaps before submission.

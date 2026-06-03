# Research Dossier — JCIM Submission

**Target:** JCIM (Journal of Chemical Information and Modeling), ACS Publications
**Date:** 2026-06-02

---

## 1. Venue Requirements

**Article types.** JCIM accepts Articles (full research papers, ~8-12 journal pages), Letters (short, urgent), Reviews, Perspectives, and Application Notes. This manuscript is an Article.

**Length.** No strict word limit, but typical Articles are 6,000-10,000 words with 6-10 figures/tables. Current draft is within range.

**Abstract.** Unstructured, typically 150-250 words. Must be self-contained: state problem, method, key results with quantitative support, and significance.

**Structure.** No mandatory IMRaD enforcement, but standard structure (Introduction, Methods/Computational Details, Results and Discussion, Conclusions) is expected and followed by current draft.

**Data and software availability.** JCIM requires a Data and Software Availability statement. Code must be archived in a public repository with DOI. Data derived from third-party sources (ChEMBL) must be cited with version/accession. Current draft includes this section.

**AI-use policy.** ACS requires disclosure of AI tool use in manuscript preparation. If PaperSpine is used for polishing, disclosure is needed.

**Cover letter.** Required. Should summarize significance, confirm originality, suggest reviewers (optional), and include the ACS Paragon Plus submission statement. Existing cover letter at `jcim_submission_candidate/cover_letter.md`.

**Templates.** ACS LaTeX or Word template. Current draft uses Markdown with LaTeX math; conversion to ACS template needed for final submission.

**Figure/table rules.** Figures must be high-resolution (≥300 dpi), tables editable (not images). Chemical structures need consistent rendering. All figures require descriptive captions.

## 2. Review Criteria

JCIM reviewers evaluate:
- **Methodological rigor:** Is the computational method clearly described and reproducible? Are parameters justified? Is the evaluation protocol appropriate?
- **Chemical validity:** Are the chemical interpretations sound? Are conclusions appropriately scoped to the computational nature of the evidence?
- **Novelty vs. incrementalism:** Does the work advance beyond existing tools (SwissBioisostere, CReM, DeepBioisostere)? Current draft addresses this in Related Work.
- **Quantitative evidence:** Are performance claims supported by appropriate statistics (CIs, paired tests)? Current draft uses bootstrap — well-suited.
- **Limitations disclosure:** Are limitations honestly acknowledged? Current draft has strong limitations section (Section 5.6).
- **Reproducibility:** Is code/data sufficient for independent verification? Code archive exists on GitHub.

## 3. Accepted Paper Patterns

**Pattern 1 — Computational benchmark paper.** JCIM frequently publishes benchmark-focused papers that introduce a new evaluation protocol alongside a method. The accepted pattern is: (i) motivate why existing benchmarks fail (leakage), (ii) design a principled split, (iii) demonstrate that the new split reveals different method rankings than random splits, (iv) propose a method that performs well under the rigorous protocol. Current draft follows this pattern closely.

**Pattern 2 — Feature engineering + mechanistic analysis.** JCIM reviewers value not just "method X beats baseline Y" but mechanistic understanding of WHY. The prior_ranks ablation (Section 3.5, 4.2) is exactly this pattern: identify a specific feature family, demonstrate its harm, and explain the mechanism (shortcut learning under distribution shift). This is a JCIM-level contribution pattern.

## 4. Constraints for This Paper

1. **Post-audit finding must NOT be presented as prospective.** The 77-feature result was identified from blind diagnostics. The manuscript already handles this well with explicit caveats. No changes needed to the framing.

2. **A4C must remain scoped as computational triage.** The alert rates are from rule-based filters on a pre-filtered vocabulary. Must not claim medicinal-chemistry validation. Current framing is appropriate.

3. **Stats communication must be precise.** Bootstrap CIs should be consistently reported with replication count (B=5000). Current draft does this.

4. **No overclaiming on generalizability.** Single data source (ChEMBL37K), structure-derived labels. Already addressed in limitations.

5. **JCIM word limit and structure.** 6,000-10,000 words. Current draft is comprehensive but dense — may benefit from tightening without losing precision.

6. **Abbreviation discipline.** Every abbreviation defined at first use. Current draft is consistent.

7. **Conservative comparative language.** "Improves over" rather than "outperforms," backed by CIs. Current draft largely uses this register.

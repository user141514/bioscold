# Target Journal Decision - 2026-05-31

## Manuscript Assessed

- `goal/PROJECT_5DAY_FULL_REVIEW/paper_draft/combined_draft_clean.md`

## Current Manuscript Identity

The manuscript is strongest as a **cheminformatics / molecular design ranking-method paper**:

- closed-vocabulary scaffold-conditioned fragment replacement;
- transform-heldout leakage control;
- candidate-level feature-pruned ranking;
- post-audit shortcut analysis;
- computational triage layer with explicit caveats.

It is not yet a strong **computational biology endpoint paper**:

- no target-specific activity-preservation claim;
- no wet-lab validation;
- no expert medicinal-chemistry validation;
- no independent prospective blind split after D4S31 post-audit feature pruning;
- A4C remains a computational proxy.

## Journal Fit Matrix

| Candidate journal | Fit | Upside | Main reviewer risk | Decision |
|---|---:|---|---|---|
| **Journal of Chemical Information and Modeling (JCIM)** | High | Strong home for chemical informatics, molecular modeling, AI/ML for chemical and biological data, and computer-aided molecular design | Will demand precise methods, reproducibility, baseline comparison, and no overclaiming of medicinal chemistry validation | **Primary target** |
| **Digital Discovery** | High | Good fit for AI/high-throughput computational methods for molecular and materials design | Wants broader discovery framing and may ask for stronger generalizable workflow evidence | Secondary high-upside target |
| **Journal of Cheminformatics** | High | Natural fit for open cheminformatics methods, reproducibility, software/data | May be less prestigious than the current ambition but technically well aligned | Safe target |
| **Nature Communications** | Medium-low | Broad multidisciplinary visibility | Likely asks for broader biological/chemical impact, independent validation, and stronger external evidence | Not primary unless upgraded |
| **Briefings in Bioinformatics** | Medium-low | Scope includes chemoinformatics and computational biology with molecular foundation | Current manuscript is not review-like and lacks direct biological endpoint emphasis | Not primary |
| **Bioinformatics** | Low-medium | Strong computational biology venue | Scope is more genome/bioinformatics centered; drug-design cheminformatics paper may be desk-risky without stronger biological endpoint | Not primary |

## Decision

**Primary target: Journal of Chemical Information and Modeling (JCIM).**

Rationale:

1. The paper's strongest contribution is methodological chemical informatics, not broad biological discovery.
2. JCIM explicitly fits new methodology in chemical informatics, AI/ML for chemical and biological data, and computer-aided molecular design.
3. The current evidence package is enough to support a careful computational method paper if claims stay bounded.
4. The post-audit D4S31 caveat is survivable in JCIM if framed transparently; it is more dangerous for Nature Communications.

## Target-Specific Reframing for JCIM

### Main Claim

Use:

> We introduce a leakage-controlled closed-vocabulary benchmark and a post-audit feature-pruned candidate scorer for scaffold-conditioned fragment replacement ranking.

Avoid:

> We discover activity-preserving bioisosteres.

### Contribution Hierarchy

1. Transform-heldout benchmark for replacement ranking.
2. Candidate-level scoring improves Top-10 over Score Blend.
3. Post-audit feature pruning exposes prior-rank shortcut learning.
4. Borda remains a mechanism anchor for DE/HGB complementarity.
5. A4C is a computational reporting layer, not validation.

### Reviewer Team for Next Round

For JCIM, use four targeted reviewer roles:

1. **JCIM Methods Reviewer**
   - Attacks feature definitions, train-only fitting, leakage control, and reproducibility.
2. **JCIM Cheminformatics Reviewer**
   - Attacks MMP construction, fragment vocabulary, attachment signatures, decoy construction, and chemical plausibility.
3. **JCIM ML/Statistics Reviewer**
   - Attacks post-audit D4S31 selection, ablation uncertainty, bootstrap design, multiple comparisons, and baseline fairness.
4. **JCIM Medicinal Chemistry Boundary Reviewer**
   - Attacks A4C, activity-preservation language, bioisostere terminology, and practical workflow claims.

## Required Next Edits Before JCIM Submission

1. Add a compact "Data and Code Availability / Reproducibility" section.
2. Add a Supplementary Methods table for P0/P1/P3/P4 priors, smoothing, and similarity-transfer definitions.
3. Add a short paragraph clarifying that the post-audit D4S31 model should be prospectively replicated.
4. Keep A4C out of the abstract's central contribution framing.
5. Add a reviewer-facing statement that the classifier architecture is not the novelty.
6. Avoid direct superiority claims over NeBULA, DeepBioisostere, GraphBioisostere, or open-vocabulary methods.

## Source Notes Used for Journal Fit

- ACS JCIM describes its scope around new methodologies in chemical informatics and molecular modeling, AI/ML models applied to chemical and biological data, computer-aided molecular design, and computational methods/software.
- RSC Digital Discovery covers AI and high-throughput computational methodologies for molecular, materials and formulation design.
- Journal of Cheminformatics publishes original peer-reviewed research in cheminformatics and molecular modelling.
- Nature Communications is broad and high visibility but would likely require stronger general biological or chemical impact.
- Bioinformatics and Briefings in Bioinformatics are closer to computational biology; the current paper would need a stronger biological endpoint framing to be a natural fit.


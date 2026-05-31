# 3.7 Computational Review Proxy (A4C) -- Spec

## Coverage
Describes the dual-mode workflow for computational review: Conservative Mode (HGB proposals) and Exploration Mode (Borda(DE, HGB) proposals) with G2/G3/G4 provenance labels and A4C alert stratification. Covers workflow design, provenance groups and their alert rates, and scope/limitations of the computational proxy.

## Key Equations / Notation

No formal equations -- this is a workflow description with conceptual groupings.

Provenance groups (defined operationally):

| Group | Definition | Alert Rate |
|-------|-----------|------------|
| G4 | In top-$K$ of BOTH HGB and Borda | 0.99% |
| G3 | In top-$K$ of Borda (via DE) but NOT HGB | 9.67% |
| G2 | In top-$K$ of Borda but NOT in individual top-$K$ of HGB or DE | 46.85% |

Conservative Mode: proposals from HGB ranker (Section 3.3.3)
Exploration Mode: proposals from Borda(DE, HGB) fusion (Section 3.3.4)

## Figures / Tables Needed
- **Table 7**: Provenance groups, definitions, A4C alert rates, and interpretation
- **Figure**: Dual-mode workflow diagram showing Conservative Mode (HGB) and Exploration Mode (Borda) pathways, with G2/G3/G4 provenance labeling and alert rate feedback (recommended -- helps readability)

## Dependencies
- Requires HGB ranker (3.3.3) for Conservative Mode
- Requires Borda fusion (3.3.4) for Exploration Mode
- Requires DE ranker (3.3.2) for G3 definition
- Alert filters (PAINS, Brenk) introduced in Introduction / Related Work
- Protocol separation constraints from 3.6.4 apply (this is not a performance claim)
- Results table goes in Section 4.5 (dual-mode risk stratification)

## What Needs Updating from Existing Draft

### Issues (full draft lines 607-648, unified draft lines 385-425)
1. **G2 definition clarity**: The full draft says G2 = "In top K of Borda but not in individual top-K of HGB or DE." Ensure this is unambiguous: G2 means the candidate is in Borda top-K but NOT in HGB top-K AND NOT in DE top-K. The unified draft has the same definition. This is correct.
2. **G3 definition precision**: "In top K of DE (Borda)" is potentially confusing. It means: candidates that are in the Borda top-K *because* DE elevated them -- i.e., they are in the Borda top-K but NOT in the HGB top-K. The DE component of Borda is responsible. Clarify: "Candidates in the Borda top-K that are outside the HGB top-K. Since DE is the non-HGB component of Borda, these represent similarity-supported novelty."
3. **Alert rate provenance**: The alert rates (0.99%, 9.67%, 46.85%) are results. In the Methods section, present the *framework* and define what G2/G3/G4 mean. The actual alert numbers can appear as a preview in Table 7, with full discussion in Section 4.5.
4. **Conservative/Exploration framing**: Both drafts frame this well as exploration-exploitation tradeoff. Keep the language.
5. **Scope limitations (3.7.3)**: Essential for honest presentation. The unified draft ends with a recommended protocol (G4: no further triage, G3: review, G2: expert review). The full draft has similar. Keep this -- it is the actionable output of the analysis.
6. **No equations**: This section is prose + tables. Ensure the writing is clear and the G2/G3/G4 logic is unambiguous.
7. **Positioning**: Section 3.7 is the last Methods subsection before Results. It should transition to: "With these definitions in place, we now report results..."

### Final Output Requirements
- ~600-700 words
- One table (provenance groups)
- Clear operational definitions of G2/G3/G4
- Explicit scope limitations

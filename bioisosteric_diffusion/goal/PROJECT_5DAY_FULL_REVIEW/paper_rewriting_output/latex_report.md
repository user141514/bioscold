# LaTeX Report

**Manuscript:** Leakage-Controlled Fragment Replacement Ranking with Audited Candidate-Level Scoring
**Target:** JCIM (Journal of Chemical Information and Modeling)
**Date:** 2026-06-02

---

## Source

- **Main .tex:** `final_paper/main.tex`
- **Class:** `article` (standard, adaptable to ACS `achemso`)
- **Engine:** pdflatex / xelatex
- **Encoding:** UTF-8

## Compilation Status

- **TeX engine available:** No (neither pdflatex nor xelatex found)
- **Compilation attempted:** Skipped — no TeX distribution
- **PDF produced:** No — requires TeX Live or MiKTeX installation

## LaTeX Structure

| Element | Status | Notes |
|---|---|---|
| Document class | article 12pt | Adaptable to achemso |
| Math packages | amsmath, amssymb | All display/inline math preserved |
| Tables | booktabs | 9 tables (M1-M5, 1-4) |
| Figures | — | No figures in current draft; placeholders in figures/ |
| Bibliography | thebibliography | 20 references in ACS style |
| Hyperlinks | hyperref | DOI links clickable |

## Known Issues

| Issue | Severity | Status |
|---|---|---|
| No ACS achemso template | Low | article class used; adapt for final submission |
| No figures included | Low | Current draft is table-only; add figures/ before final compile |
| Preprint ref [9] not peer-reviewed | Info | Noted in reference entry |
| Software ref [18] URL only | Info | Standard for RDKit |

## Compile Command (when TeX available)

```bash
cd final_paper
pdflatex main.tex
bibtex main   # if using .bib
pdflatex main.tex
pdflatex main.tex
```

## ACS Template Adaptation Notes

To adapt to JCIM achemso template:
1. Change `\documentclass{article}` to `\documentclass{achemso}`
2. Add `\author`, `\affiliation`, `\email` fields
3. Tables remain compatible (booktabs)
4. Math remains compatible (amsmath)
5. Bibliography format may need conversion from `thebibliography` to `.bib`

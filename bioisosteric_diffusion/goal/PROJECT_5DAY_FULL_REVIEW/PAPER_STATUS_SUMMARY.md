# Paper Status Summary

**2026-05-31 audit update:** `combined_draft_clean.md` is the current manuscript. D4S31 Top10 was rerun as 0.924328 with CI [0.9198303, 0.9288979]. Delta over ScoreBlend was rerun as +0.068555 with paired CI [0.0636847, 0.0734997]. ScoreBlend rescue/lost must use the repaired query_id-merge values 1,016/101, not the stale positional 424/6 values. A4C G4 must be read as a total lower-bound alert rate (0.99%) with limited coverage (5.63%) and high unknown status (94.37%), not as a complete low-risk alert rate.

**Date:** 2026-05-30
**Project:** Route-A — Bioisosteric Fragment Replacement
**Paper:** "A Dual-Mode Workflow for Scaffold-Conditioned Fragment Replacement Proposal via Neural-Empirical Rank Fusion"

---

## Manuscript Completion

| Section | File | Words | Status |
|---------|------|-------|--------|
| Abstract | `frontmatter/abstract.md` | ~200 | Frozen |
| §1 Introduction | `frontmatter/section1_introduction.md` | ~550 | Frozen |
| §2 Related Work | `frontmatter/section2_related_work.md` | ~450 | Frozen |
| §3 Methods | `methods/methods_clean_timeline.md` | ~4800 | Frozen |
| §4 Results | `results/section4_results.md` | ~1400 | Frozen |
| §5 Discussion | `discussion/section5_discussion.md` | ~1200 | Frozen |
| §6 Conclusion | `section6_conclusion.md` | ~200 | Frozen |
| References | `references.bib` | 4 new refs | Needs legacy refs |
| Combined | `combined_draft_clean.md` | ~8000 | All synced |

**Total:** ~8,800 words (complete manuscript)

---

## Key Numbers Locked

| Number | Value | Evidence |
|--------|-------|----------|
| D4S31 Top-10 | 0.9243 [0.9198, 0.9289] | `d4s31_final_metrics.csv` |
| Delta over Score Blend | +0.068555 [0.0636847, 0.0734997] | same CSV |
| Score Blend | 0.8558 | `paper_p0_main_table1_newblind_candidate.csv` |
| Ablation gain | +0.0392 | `d4s31_ablation.csv` |
| Rescue (D4S31 vs ScoreBlend) | 1,016 | `d4s31_rescue_lost_merged.csv` (repair script) |
| Lost (D4S31 vs ScoreBlend) | 101 | same CSV |
| Lost (D4S28R vs ScoreBlend) | 596 | `02_final_metrics.md` |
| Loss reduction | 5.9× | 596→101 |
| A4C G2/G3/G4 alert rates | G2 46.85%, G3 9.67%; G4 total lower-bound 0.99% with coverage 5.63% and unknown 94.37% | `d4a3t_risk_decomposition.csv`, `final_metric_table.csv` |
| Oracle(DE,HGB) | 0.8686 | `paper_p0_main_table1_newblind_candidate.csv` |
| N (blind queries) | 13,347 | Verified across all files |
| Features | 77 retained, 5 removed | `d4s31_lockdown.py` |
| Fragment degradation | D4S28R: 6/19, D4S31: 0/19 | `d4s31_blind_strata.csv` |

---

## Critical Bugs Fixed

| Bug | Original | Fixed | Method |
|-----|----------|-------|--------|
| Rescue/lost arithmetic | 424/6 (impossible) | 1,016/101 (verified) | `d4s31_rescue_lost_repair.py` — query_id merge |
| Positional assignment | `q_bl["Top10"].values` | `merge(on="query_id")` | Script fix (line 148) |
| Δ precision | 0.0685 (display subtraction) | 0.0686 (full-precision 0.068555) | Verified against CSV |
| Oracle value | 0.8998 (repair script artifact) | 0.8686 (authoritative) | Locked D4S-B0 protocol |
| "Two orders of magnitude" | 596→6 narrative | 596→101, 5.9× | Verified counts |

---

## Timeline Claims Fixed

| Stale Claim | Fix |
|-------------|-----|
| "Feature schema finalized before any blind evaluation" | F1-F9 pre-blind; prior_ranks post-audit |
| "Secondary blind accessed exactly once" | Removed; §3.5 documents actual timeline |
| "Validation set" | → "Development/calibration set" |
| "Validation-based tuning" | → "Development-set-based tuning" |
| "Blind vocabulary entirely disjoint" | → "Transform identities held out; same vocabulary" |
| "All features computable from structure alone" | → "Structure, base-ranker outputs, or train-set statistics" |
| "Oracle ceiling" | → "Oracle(DE,HGB) diagnostic bound" |

---

## References Integrated

| Ref | Citation Key | Status |
|-----|-------------|--------|
| NeBULA 2025 | Huang2025 | Formally cited (§2, §5.5) |
| GraphBioisostere 2026 | Masunaga2026 | Formally cited (§2, §5.5) |
| DeepBioisostere 2025 v2 | Kim2025 | Formally cited (§2, §5.5); arXiv preprint noted |
| Helmke et al. 2025 | Helmke2025 | Formally cited (§2, §3.7, §5.4); no specific numbers |

---

## Forbidden Claims (Not in Manuscript)

- "Activity-preserving bioisostere"
- "Fully prospective feature selection"
- "Two orders of magnitude" regression reduction
- Oracle as "ceiling" or "upper bound"
- Helmke with specific CHRM2/10×/88 targets numbers

---

## Evidence Base

- **Tier 1 (Authoritative):** `plan_results/` — 100+ routeA directories with VERDICT.md sign-off
- **Tier 2 (Verified):** `goal/A_improve/` — D4S27-D4S32 artifacts, fresh script runs
- **Claim lock:** `goal/PAPER_CLAIM_EVIDENCE_LOCK/` — 24 claims verified, 5 blocked

---

## Remaining Tasks

| Priority | Task |
|----------|------|
| HIGH | Complete legacy references in BibTeX (Zdrazil2023, Hussain2010, Griffen2011, etc.) |
| HIGH | Fill Supplementary S2 (MRR values — 5/9 missing) |
| MEDIUM | Populate Supplementary S3 with D4S28R rescue/lost (currently from audit doc, not CSV) |
| MEDIUM | Lock D4S31 evidence into `plan_results/` (currently only in `goal/A_improve/`) |
| LOW | Figure generation (Figure 1 workflow, Figure 2 main performance) |
| LOW | Final formatting for target journal |

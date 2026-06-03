# Nature-Skill Revision Change Log

## Files

- `manuscript_en.md`: English natureskill revision.
- `manuscript_zh.md`: Chinese aligned revision.
- `strict_audit.md`: post-revision reviewer-style audit.
- `figure_s3_spec.md`: specification for the per-fragment blind-delta supplementary figure.

## Main Story Changes

1. Rebuilt the abstract into a clearer three-step story: leakage/shortcut risk, benchmark and scorer evidence, prior_ranks deletion with post-audit boundary.
2. Reframed the Introduction around evidence hierarchy rather than a flat list of contributions.
3. Clarified split roles: training partition for vocabulary/statistics, development/calibration for scorer fitting and schema, secondary blind for one-shot evaluation.
4. Added an explicit guard against direct superiority comparisons to NeBULA, CReM, DeepBioisostere, and GraphBioisostere.
5. Demoted query-level routing from a complementary finding to a diagnostic development observation.
6. Strengthened per-fragment robustness wording by stating that strata are point-estimate audits, not per-fragment significance tests.
7. Revised data availability language from future promise toward a submission-readiness requirement.
8. Added a sharper narrative spine: the benchmark is framed as a way to reveal which ranking signals survive once transform memorization is unavailable.

## Claims Preserved

All locked numerical claims were preserved:

- 82-feature scorer Top-10 = 0.8851.
- 77-feature scorer Top-10 = 0.9243.
- 77-feature vs Score Blend ΔTop-10 = +0.0686.
- 77-feature vs 82-feature ΔTop-10 = +0.0393.
- Score Blend lost hits reduced from 596 to 101.
- A4C G2/G3/G4 rates and coverage retained.

## Remaining Open Items

- Generate Supplementary Fig. S3.
- Ensure Supplementary Tables S3-S5 are included.
- Verify and migrate bibliography during final LaTeX conversion.
- Finalize data/software deposition wording.

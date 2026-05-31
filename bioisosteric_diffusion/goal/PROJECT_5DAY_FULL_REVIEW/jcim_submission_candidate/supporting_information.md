# Supporting Information

## Leakage-Controlled Closed-Vocabulary Fragment Replacement Ranking with Candidate-Level Scoring

This Supporting Information file contains the verified metric, rescue/lost, ablation, and per-fragment tables cited by the main manuscript. It is intended for ACS/JCIM submission as Supporting Information for Publication.

## Table Directory

- Supplementary Table S2. Secondary blind metrics recomputed in the JCIM evidence patch.
- Supplementary Table S3. Rescue/lost analysis by query_id merge.
- Supplementary Table S4. Leave-one-family-out ablation.
- Supplementary Table S5. Per-fragment Top-10 on the secondary blind set.

## Notes on Evidence Boundaries

Supplementary Table S4 reports a paired confidence interval only for the main post-audit prior_ranks deletion; the other leave-one-family-out rows are point estimates. Supplementary Table S5 reports per-fragment point estimates and should not be read as a set of per-fragment significance claims.

---

# Supplementary Tables: Verified Evidence Chain

All rescue/lost values are computed by query identifier merge rather than positional assignment. The current evidence source for the JCIM-facing revision is `goal/PROJECT_5DAY_FULL_REVIEW/jcim_major_revision/`.

---

## Supplementary Table S2. Secondary Blind Metrics Recomputed in the JCIM Evidence Patch

**Computation:** `scripts/jcim_major_revision_patch.py`; output file `jcim_full_secondary_metrics_with_ci.csv`.

| Method | N queries | Top-10 | 95% CI for Top-10 | MRR | 95% CI for MRR |
|--------|----------:|-------:|------------------:|----:|---------------:|
| Attachment-Frequency | 13,347 | 0.6019 | [0.5936, 0.6102] | 0.2858 | [0.2799, 0.2914] |
| DE | 13,347 | 0.8055 | [0.7986, 0.8122] | 0.4174 | [0.4111, 0.4237] |
| HGB-refit candidate score | 13,347 | 0.8624 | [0.8564, 0.8681] | 0.4662 | [0.4602, 0.4721] |
| Borda candidate-matrix score | 13,347 | 0.8456 | [0.8391, 0.8516] | 0.4799 | [0.4737, 0.4862] |
| Score Blend | 13,347 | 0.8558 | [0.8497, 0.8616] | 0.4842 | [0.4779, 0.4904] |
| Initial 82-feature scorer | 13,347 | 0.8851 | [0.8796, 0.8903] | 0.4529 | [0.4472, 0.4587] |
| Post-audit 77-feature scorer | 13,347 | 0.9243 | [0.9199, 0.9287] | 0.4769 | [0.4712, 0.4829] |

The HGB-refit and Borda candidate-matrix rows are computed from candidate-matrix score columns and are used for aligned rescue/lost diagnostics. The main paper retains the historical pre-D4S method hierarchy where Score Blend is the strongest pre-D4S baseline and the 77-feature scorer is a post-audit locked result.

---

## Supplementary Table S3. Rescue/Lost Analysis by query_id Merge

**Script:** `scripts/jcim_major_revision_patch.py`; output files `jcim_rescue_lost_bootstrap.csv` and `jcim_query_level_82_77_scoreblend_audit.csv`.

### S3a. Post-audit 77-feature scorer vs Score Blend and the initial 82-feature scorer

N = 13,347 common secondary blind queries. The post-audit 77-feature scorer has 12,337 Top-10 hits (Top-10 = 0.9243).

| Reference | Ref Top-10 | Rescue | Lost | Net | Net per query | 95% CI for net per query | Rescue/Lost ratio | 95% CI for ratio |
|-----------|-----------:|-------:|-----:|----:|--------------:|-------------------------:|------------------:|-----------------:|
| Score Blend | 0.8558 | 1,016 | 101 | +915 | +0.0686 | [+0.0638, +0.0733] | 10.06 | [8.25, 12.52] |
| Initial 82-feature scorer | 0.8851 | 626 | 102 | +524 | +0.0393 | [+0.0354, +0.0432] | 6.14 | [5.03, 7.65] |

### S3b. Initial 82-feature scorer vs Score Blend

This comparison is retained to explain why the 77-feature post-audit model is preferable to the initial 82-feature scorer.

| Reference | Ref hits | Ref Top-10 | Rescue | Lost | Net | 82-feature hits | 82-feature Top-10 |
|-----------|---------:|-----------:|-------:|-----:|----:|----------------:|------------------:|
| Score Blend | 11,422 | 0.8558 | 987 | 596 | +391 | 11,813 | 0.8851 |

Arithmetic check: 11,422 + 987 - 596 = 11,813, and 11,813/13,347 = 0.8851 after rounding.

### S3c. Key reliability shift

| Metric | Initial 82-feature scorer | Post-audit 77-feature scorer | Change |
|--------|--------------------------:|------------------------------:|-------:|
| Lost queries vs Score Blend | 596 | 101 | 5.9-fold reduction |
| Net queries vs Score Blend | +391 | +915 | 2.3-fold increase |
| Blind Top-10 | 0.8851 | 0.9243 | +0.0393 |

The post-audit prior_ranks removal improves aggregate accuracy while substantially reducing regressions against the Score Blend baseline. Because the removal was prompted by blind diagnostics, this is reported as a locked post-selection result rather than a fully prospective feature-selection claim.

---

## Supplementary Table S4. Leave-One-Family-Out Ablation

**Source:** `goal/A_improve/D4S31_FINAL_LOCK/01_ablation_findings.md`; cross-checked against `scripts/jcim_major_revision_patch.py` for the paired confidence interval of the main deletion.

| Feature family removed | Features removed | Description | Blind Top-10 | Δ from full 82-feature scorer | 95% CI for Δ |
|------------------------|-----------------:|-------------|-------------:|------------------------------:|-------------:|
| None: full 82-feature scorer | 0 | All retained feature families plus prior_ranks | 0.8851 | -- | -- |
| prior_ranks | 5 | Per-query ranks of sparse prior scores | 0.9243 | +0.0393 | [+0.0354, +0.0432] |
| prior_positives | 3 | Per-query counts of positive prior observations | 0.9037 | +0.0187 | not reported |
| query_stats | 3 | Per-query summary statistics of candidate score distribution | 0.9019 | +0.0168 | not reported |
| mol_props | 10 | Molecular property descriptors and deltas | 0.8930 | +0.0079 | not reported |
| prior_scores | 5 | Sparse prior score transforms | 0.8924 | +0.0073 | not reported |
| model_ranks | 6 | Base-ranker rank positions | 0.8697 | -0.0154 | not reported |
| model_scores | 5 | Base-ranker score outputs | 0.8610 | -0.0241 | not reported |
| sim_freq | 3 | Similarity and frequency descriptors | 0.8649 | -0.0202 | not reported |

Positive Δ means that removing the family improves blind Top-10 relative to the full 82-feature scorer. Only the prior_ranks deletion has a paired bootstrap confidence interval in the current evidence lock.

---

## Supplementary Table S5. Per-Fragment Top-10 on the Secondary Blind Set

**Source:** `goal/A_improve/D4S31_FINAL_LOCK/04_blind_strata.md` and the fresh 2026-05-30 D4S31 lockdown run.

| Fragment | n | Score Blend | Post-audit 77-feature scorer | Δ |
|----------|--:|------------:|-----------------------------:|--:|
| *C(=O)NC | 128 | 1.0000 | 1.0000 | 0.0000 |
| *C(=O)c1ccc(OC)cc1 | 120 | 1.0000 | 1.0000 | 0.0000 |
| *C(=O)c1ccco1 | 104 | 0.9327 | 1.0000 | +0.0673 |
| *C1CCCCC1 | 2,244 | 0.7197 | 0.7527 | +0.0330 |
| *CCC | 2,112 | 0.8835 | 0.9044 | +0.0208 |
| *CCCc1ccccc1 | 216 | 1.0000 | 1.0000 | 0.0000 |
| *Cc1cccc(OC)c1 | 118 | 1.0000 | 1.0000 | 0.0000 |
| *Cc1ccccc1Cl | 215 | 1.0000 | 1.0000 | 0.0000 |
| *Cc1ccccc1OC | 108 | 1.0000 | 1.0000 | 0.0000 |
| *Cc1ccccn1 | 233 | 0.5708 | 0.5708 | 0.0000 |
| *Cc1cccnc1 | 158 | 1.0000 | 1.0000 | 0.0000 |
| *N(C)C | 1,648 | 0.6608 | 0.7379 | +0.0771 |
| *N1CCCCC1 | 1,208 | 1.0000 | 1.0000 | 0.0000 |
| *Nc1ccc(C)cc1 | 234 | 1.0000 | 1.0000 | 0.0000 |
| *OC(F)(F)F | 253 | 1.0000 | 1.0000 | 0.0000 |
| *c1ccc(C(C)C)cc1 | 101 | 1.0000 | 1.0000 | 0.0000 |
| *c1cccc(C)c1 | 1,823 | 0.9013 | 0.9611 | +0.0598 |
| *c1ccccc1C(F)(F)F | 211 | 1.0000 | 1.0000 | 0.0000 |
| *c1ccccc1F | 2,113 | 0.9035 | 0.9304 | +0.0270 |

Summary: all 19 old-fragment strata have non-negative point-estimate Δ relative to Score Blend. These are stratum-level point estimates, not separate per-fragment significance claims.

---

## Evidence Provenance

| Item | Script or evidence file | Run date | Status |
|------|-------------------------|----------|--------|
| S2 metrics and MRR | `scripts/jcim_major_revision_patch.py` | 2026-05-31 | Verified from `jcim_full_secondary_metrics_with_ci.csv` |
| S3 rescue/lost | `scripts/jcim_major_revision_patch.py` | 2026-05-31 | Query-id merge; bootstrap by query rows |
| S4 ablation | `D4S31_FINAL_LOCK/01_ablation_findings.md`; `scripts/jcim_major_revision_patch.py` | 2026-05-31 | Main prior_ranks Δ has paired CI |
| S5 per-fragment strata | `D4S31_FINAL_LOCK/04_blind_strata.md` | 2026-05-30 | Verified point-estimate table |
| Main Table 1 | `jcim_full_secondary_metrics_with_ci.csv`; historical baseline lock for selected pre-D4S rows | 2026-05-31 | Values cross-checked in final audit |
| Main Table 2 | `jcim_paired_bootstrap_key_deltas.csv`; `D4S31_FINAL_LOCK/01_ablation_findings.md` | 2026-05-31 | Main deletion CI verified |
| Main Table 3 | `jcim_rescue_lost_bootstrap.csv` | 2026-05-31 | Arithmetic verified |
| Main Table 4 | `d4s30_audit.py` evidence lock | 2026-05-31 | Coverage caveat retained |

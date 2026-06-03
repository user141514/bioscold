# Supporting Information

## Leakage-Controlled Closed-Vocabulary Fragment Replacement Ranking with Candidate-Level Scoring

This Supporting Information file contains the verified metric, rescue/lost, ablation, and per-fragment tables cited by the main manuscript. It is intended for ACS/JCIM submission as Supporting Information for Publication.

## Table Directory

- Supplementary Table S2. Full secondary blind metrics recomputed in the JCIM evidence patch.
- Supplementary Table S3. Rescue/lost analysis by query_id merge.
- Supplementary Table S4. Leave-one-family-out ablation.
- Supplementary Table S5. Per-fragment Top-10 on the secondary blind set.
- Supplementary Table S6. Random/permissive split stress test for locked standalone base rankers.

## Figure Directory

- Supplementary Figure S3. Fragment-level Top-10 deltas before and after post-audit prior-rank pruning.
- Supplementary Figure S4. Categorical schema alignment audit.
- Supplementary Figure S5. Prior-ranks mechanism audit.

## Notes on Evidence Boundaries

Supplementary Table S4 reports a paired confidence interval only for the main post-audit prior_ranks deletion; the other leave-one-family-out rows are point estimates. Supplementary Table S5 reports per-fragment point estimates and should not be read as a set of per-fragment significance claims. Supplementary Table S6 is a split-protocol stress test for locked base rankers only; Score Blend and candidate-level scorers were not rebuilt under that random/permissive split.

---

# Supplementary Tables: Verified Evidence Chain

All rescue/lost values are computed by query identifier merge rather than positional assignment. The current evidence source for the JCIM-facing revision is `goal/PROJECT_5DAY_FULL_REVIEW/jcim_major_revision/`.

---

## Supplementary Table S2. Full Secondary Blind Metrics Recomputed in the JCIM Evidence Patch

**Computation:** `scripts/jcim_major_revision_patch.py`; output file `jcim_full_secondary_metrics_with_ci.csv`.

| Method | Top-1 | Top-5 | Top-10 | Top-20 | Top-50 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Attachment-Frequency | 0.1489 | 0.4168 | 0.6019 | 0.7318 | 0.9355 | 0.2858 |
| DE | 0.2616 | 0.6332 | 0.8055 | 0.9204 | 0.9998 | 0.4174 |
| HGB | 0.2759 | 0.6981 | 0.8624 | 0.9536 | 0.9968 | 0.4662 |
| Borda(DE,HGB) | 0.3096 | 0.7207 | 0.8456 | 0.9814 | 0.9989 | 0.4799 |
| Score Blend | 0.3021 | 0.7228 | 0.8558 | 0.9783 | 0.9989 | 0.4842 |
| Initial 82-feature scorer | 0.2511 | 0.7247 | 0.8851 | 0.9719 | 0.9987 | 0.4529 |
| Post-audit 77-feature scorer | 0.2680 | 0.7651 | 0.9243 | 0.9808 | 0.9987 | 0.4769 |

Metric-completeness flag: the post-audit 77-feature scorer improves Top-5, Top-10, and Top-20 recovery relative to Score Blend, with the largest claim at Top-10. It is lower than Score Blend on Top-1 and MRR, and is effectively tied/slightly lower at Top-50. The main manuscript should therefore frame the gain as Top-10 recovery and not as broader ranking dominance.

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

| Feature family removed | Features removed | Description | Blind Top-10 | Delta from full 82-feature scorer | 95% CI for Delta |
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

Positive Delta means that removing the family improves blind Top-10 relative to the full 82-feature scorer. Only the prior_ranks deletion has a paired bootstrap confidence interval in the current evidence lock.

---

## Supplementary Table S5. Per-Fragment Top-10 on the Secondary Blind Set

**Source:** `goal/PROJECT_5DAY_FULL_REVIEW/prior_ranks_mechanism_audit/prior_ranks_old_fragment_delta.csv`; query-id joined post-review block audit.

| Fragment | n | Score Blend | Initial 82-feature scorer | Post-audit 77-feature scorer | Delta 77 vs Score Blend | Delta 77 vs 82 |
|----------|--:|------------:|--------------------------:|-----------------------------:|------------------------:|---------------:|
| *CCC | 2,112 | 0.8835 | 0.8987 | 0.8930 | +0.0095 | -0.0057 |
| *c1ccccc1F | 2,113 | 0.9035 | 0.9612 | 0.9579 | +0.0544 | -0.0033 |
| *N1CCCCC1 | 1,208 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| *OC(F)(F)F | 253 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| *Nc1ccc(C)cc1 | 234 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| *c1ccccc1C(F)(F)F | 211 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| *C(=O)NC | 128 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| *C(=O)c1ccc(OC)cc1 | 120 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| *C(=O)c1ccco1 | 104 | 0.9327 | 1.0000 | 1.0000 | +0.0673 | 0.0000 |
| *c1ccc(C(C)C)cc1 | 101 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| *c1cccc(C)c1 | 1,823 | 0.9013 | 0.9731 | 0.9742 | +0.0730 | +0.0011 |
| *CCCc1ccccc1 | 216 | 1.0000 | 0.9954 | 1.0000 | 0.0000 | +0.0046 |
| *C1CCCCC1 | 2,244 | 0.7197 | 0.7518 | 0.7745 | +0.0548 | +0.0227 |
| *Cc1ccccn1 | 233 | 0.5708 | 0.8841 | 0.9356 | +0.3648 | +0.0515 |
| *N(C)C | 1,648 | 0.6608 | 0.8083 | 0.9278 | +0.2670 | +0.1195 |
| *Cc1cccnc1 | 158 | 1.0000 | 0.8734 | 0.9937 | -0.0063 | +0.1203 |
| *Cc1ccccc1Cl | 215 | 1.0000 | 0.7070 | 1.0000 | 0.0000 | +0.2930 |
| *Cc1ccccc1OC | 108 | 1.0000 | 0.1944 | 1.0000 | 0.0000 | +0.8056 |
| *Cc1cccc(OC)c1 | 118 | 1.0000 | 0.0000 | 0.9407 | -0.0593 | +0.9407 |

Summary: the initial 82-feature scorer has five negative old-fragment point estimates relative to Score Blend. The post-audit 77-feature scorer reduces this to two negative old-fragment point estimates relative to Score Blend, and 17 of 19 old-fragment strata are non-negative relative to the initial 82-feature scorer. These are stratum-level point estimates, not separate per-fragment significance claims.

---

## Supplementary Table S6. Random/Permissive Split Stress Test for Base Rankers

**Source:** `goal/PAPER_CLAIM_EVIDENCE_LOCK/random_transform_overlap_split_inflation_20260602/random_overlap_base_ranker_inflation_deltas.csv`.

The random/permissive query split preserved old-test quarantine and train-blind query non-overlap, but allowed train-blind transform-key overlap for 840/840 blind transform identities. This table is a split-protocol diagnostic for locked standalone base rankers only; Score Blend and candidate-level scorers were not rebuilt under this split.

| Method | Random/permissive Top-10 | Transform-heldout Top-10 | Delta random minus heldout | 95% CI for Delta | Train-blind transform overlap |
|--------|-------------------------:|--------------------------:|---------------------------:|-----------------:|------------------------------:|
| Attachment-Frequency | 0.5727 | 0.6019 | -0.0292 | [-0.0404, -0.0173] | 840/840 |
| DE | 0.8784 | 0.8055 | +0.0729 | [0.0643, 0.0814] | 840/840 |
| HGB | 0.8191 | 0.7437 | +0.0755 | [0.0655, 0.0853] | 840/840 |
| Borda(DE,HGB) | 0.8699 | 0.8384 | +0.0316 | [0.0230, 0.0398] | 840/840 |

This diagnostic shows that permissive transform overlap changes which base-ranker signals appear transferable. It does not support a claim that random splits inflate every baseline, because Attachment-Frequency decreases under this split, and it does not provide locked random-split results for Score Blend, the 82-feature scorer, or the post-audit 77-feature scorer.

---

## Evidence Provenance

| Item | Script or evidence file | Run date | Status |
|------|-------------------------|----------|--------|
| S2 metrics and MRR | `scripts/jcim_major_revision_patch.py` | 2026-05-31 | Verified from `jcim_full_secondary_metrics_with_ci.csv` |
| S3 rescue/lost | `scripts/jcim_major_revision_patch.py` | 2026-05-31 | Query-id merge; bootstrap by query rows |
| S4 ablation | `D4S31_FINAL_LOCK/01_ablation_findings.md`; `scripts/jcim_major_revision_patch.py` | 2026-05-31 | Main prior_ranks Delta has paired CI |
| S5 per-fragment strata | `prior_ranks_mechanism_audit/prior_ranks_old_fragment_delta.csv`; `post_review_20260602_group_top10_detail.csv` | 2026-06-02 | Query-id joined point-estimate table; 2/19 negative vs Score Blend, 17/19 non-negative vs 82-feature |
| S6 random/permissive split stress test | `random_overlap_base_ranker_inflation_deltas.csv`; `random_overlap_transform_audit.csv` | 2026-06-02 | Locked base-ranker diagnostic only; candidate scorers not rebuilt |
| Supplementary Figure S5 | `prior_ranks_mechanism_audit/figure_s_prior_ranks_mechanism.*` | 2026-06-02 | Post-audit targeted mechanism diagnostic |
| Main Table 1 | `jcim_full_secondary_metrics_with_ci.csv`; historical baseline lock for selected pre-D4S rows | 2026-05-31 | Values cross-checked in final audit |
| Main Table 2 | `jcim_paired_bootstrap_key_deltas.csv`; `D4S31_FINAL_LOCK/01_ablation_findings.md` | 2026-05-31 | Main deletion CI verified |
| Main Table 3 | `jcim_rescue_lost_bootstrap.csv` | 2026-05-31 | Arithmetic verified |
| Main Table 4 | `d4s30_audit.py` evidence lock | 2026-05-31 | Coverage caveat retained |

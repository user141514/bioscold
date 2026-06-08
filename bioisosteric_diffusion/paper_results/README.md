# LBC-Ranker: Fragment Replacement Ranking via Learned Binary-fingerprint Compatibility

Date: 2026-06-08
Current evidence lock: **manuscript_v2 / v2_full_data**

## Current Manuscript (V2)

**Manuscript**: `manuscript_v2/main_manuscript.md`
**Supporting Information**: `manuscript_v2/supporting_information.md`
**Figures**: `manuscript_v2/fig*.pdf`
**Reviewer Report**: `manuscript_v2/reviewer_report.md`
**Cover Letter**: `manuscript_v2/cover_letter.md`

### Key Results (123 OFs, 10-seed, inner 3-fold CV baseline tuning)

| Method | Macro Hit@10 |
|--------|-------------|
| **LBC-Ranker** | **0.852 ± 0.032** |
| HGB | 0.851 |
| C3F-style retrieval (tuned) | 0.716 |
| CA (tuned) | 0.661 |
| Frequency | 0.462 |

- Δ vs best non-LBC baseline: +0.130 ± 0.044 (p = 0.00195, 10/10 wins)
- Ablation: freq (−0.193) > bit_corr (−0.155) > morgan (−0.109)
- LR ≈ HGB (Δ = −0.002)

### Experiment Data

All outputs archived in `v2_full_data/`:

| File | Content |
|------|---------|
| e1_main_10seed_summary.csv | Per-seed macro results |
| e1_main_10seed_detail.csv | Per-OF per-seed results |
| e1_main_paired_tests.csv | Exact paired sign-flip tests |
| e2_ablation_full_summary.csv | Feature ablation summary |
| e2_ablation_full_detail.csv | Per-seed ablation detail |
| e3_tuning_ledger.csv | All inner CV config scores |
| e4_coldstart_audit.csv | OF-level label statistics |
| e4_split_imbalance_audit.csv | Per-seed train/test imbalance |
| e5_rank_diag.csv | Per-OF method comparison |
| e6_hgb_vs_lr.csv | LR vs HGB per seed |
| e7_learned_weights.csv | Feature weights across seeds |
| e8_zero_of_predictions.csv | Predictions on zero-positive OFs |
| protocol.md | Locked experiment protocol |
| run_v2_experiments.py | Experiment script |

### Technique Transfer

`../technique_transfer/TECHNIQUE_TRANSFER_REPORT.md` — Three techniques tested, all negative. LBC-Ranker at 2D FP information ceiling.

## Archived: 27-OF Exploratory Pilot

`archive_27OF_exploratory/` — Initial exploration on 40-shard label subset.
**Not used for manuscript claims.** Kept for internal record.

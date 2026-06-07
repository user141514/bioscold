# LBC-Kernel Reproducibility Archive

Date: 2026-06-05
Lock update: 2026-06-06

Paper: Fragment Replacement Top-10 Candidate Ranking via Learned Bit-Correlation Kernel

Claim-evidence lock:
`goal/PAPER_CLAIM_EVIDENCE_LOCK/lbc_kernel_20260606/`

## Environment

- Python 3.10.18 (conda env: `accfg`, path: `E:/Anaconda3/envs/accfg/python`)
- Key packages: sklearn 1.7.2, pandas 2.3.3, numpy 1.26.4, RDKit
- OS: Windows 10 Pro
- No GPU required

## Data Sources

| Source | Path | Description |
|---|---|---|
| D4A0 manifest | `plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/d4a0_query_split_manifest.jsonl` | 137,556 query rows; 137 unique OF strings in the current archive |
| D4A0 train shards | `.../matrices/train/train_features_shard_*.jsonl` | Labeled candidate-pair shards used by scripts |
| D4A0 test shards | `.../matrices/test/test_features_shard_*.jsonl` | Labeled candidate-pair shards used by scripts |
| C3F legacy result | `paper_results/c3f_content_aware_results.csv` mirrored from `plan_results/routeA_newpaper_phase0_protocol_lock/c3f_content_aware.csv` | Supports C3F = 0.041 and random = 0.109 |
| ChEMBL 36 | `E:/zuhui/chembl_data/chembl_36.sdf` | Source molecule collection for candidate pool construction |
| ChEMBL pool | `chembl_candidate_pool/batch_*.parquet` | Filtered candidate pool with Morgan fingerprints |

## Script Inventory

Scripts copied into `paper_results/`:

| File | Experiment | Lock status |
|---|---|---|
| `lk_fast.py` | Historical multi-seed learned kernel | Superseded by corrected archive for locked multi-seed claims |
| `c3f_content_aware.py` | Content-Aware/C3F script copy | Result CSV mirrored as `paper_results/c3f_content_aware_results.csv` |
| `learned_kernel.py` | Initial learned-kernel discovery | Diagnostic only |

Scripts in project root:

| File | Experiment | Lock status |
|---|---|---|
| `paper_experiments.py` | E1-E6: LOO, cold-start, rank diagnostics, model comparison, historical ablation | LOO/rank/model comparison locked; historical E2 superseded |
| `lbc_lock_repair_experiments.py` | Corrected E1 multi-seed and strict full-model E2 ablation | Multi-seed, paired sign-flip, and corrected ablation locked |
| `lbc_dataset_support_audit.py` | Dataset support audit | 137 OFs, 27 positive-support OFs, 110 zero-positive-support OFs locked |
| `robustness_check.py` | R1-R5 support diagnostics | `r4_coldstart_def.csv` locked with scope |
| `build_candidate_pool.py` | ChEMBL candidate pool construction | Supporting pipeline |
| `build_extended_matrix.py` | Extended candidate matrix construction | Supporting pipeline |
| `build_max_v2.py` | Hybrid builder | Supporting pipeline |
| `protocol_loader.py` | Unified data loading | Supporting pipeline |

## Reproduction Order

```text
1. python build_candidate_pool.py
2. python build_extended_matrix.py
3. python build_max_v2.py
4. python paper_experiments.py
5. python robustness_check.py
6. python lbc_lock_repair_experiments.py --seeds 10
7. python lbc_dataset_support_audit.py
```

## Locked Key Results

### Primary LOO-OF (27 labeled OFs)

Macro values from `paper_results/e3_loocv.csv`:

| Method | Hit@10 |
|---|---:|
| Content-Aware | 0.638 |
| **LBC-Kernel** | **0.820** |

Query-level diagnostics from `paper_results/e5_rank_diag.csv`:

| Method | Hit@1 | Hit@5 | Hit@10 | Hit@20 |
|---|---:|---:|---:|---:|
| Content-Aware | 0.000 | 0.322 | 0.525 | 0.734 |
| **LBC-Kernel** | **0.265** | **0.637** | **0.784** | **0.864** |

Paired diagnostics:

- Rescue/loss: 8,846/483 (ratio 18.3)
- Mean rank improvement: +8.20 positions
- Boundary repair: 27.4% query rows

### Model Comparison

From `paper_results/e6_model_compare.csv`:

| Model | Features | Hit@10 |
|---|---|---:|
| LR | 8 features | 0.872 |
| HGB | 8 features | 0.777 |
| LR | 7 features (no freq) | 0.732 |
| HGB | 7 features (no freq) | 0.702 |
| CA (hand-crafted) | Morgan + physchem | 0.636 |

## Corrected Multi-Seed and Ablation

### Multi-Seed (10 seeds, 70/30 split)

Corrected leakage-free train-only frequency values from `paper_results/e1_multiseed_summary.csv`:

| Method | Hit@10 |
|---|---:|
| Frequency | 0.398 +/- 0.166 |
| Content-Aware | 0.676 +/- 0.066 |
| LBC-Kernel | 0.849 +/- 0.046 |

Exact paired sign-flip test from `paper_results/e1_multiseed_paired_tests.csv`:

| Comparison | Mean delta | One-sided p | Two-sided p |
|---|---:|---:|---:|
| LBC vs Content-Aware | +0.173 | 0.00098 | 0.00195 |
| LBC vs Frequency | +0.451 | 0.00098 | 0.00195 |

### Corrected Feature Ablation

Corrected OF-macro deltas from `paper_results/e2_ablation_corrected_summary.csv`:

| Drop | Delta Hit@10 |
|---|---:|
| bit_corr | -0.102 +/- 0.098 |
| frequency | -0.064 +/- 0.065 |
| Morgan | -0.027 +/- 0.033 |
| dRings | -0.009 +/- 0.035 |
| dLogP | -0.007 +/- 0.012 |
| dTPSA | -0.002 +/- 0.037 |
| dHeavy | -0.001 +/- 0.037 |
| dMW | +0.003 +/- 0.012 |

Boundary: `bit_corr` has the largest OF-macro one-drop loss; `freq` has the largest query-weighted one-drop loss.

## Other Analysis Documents

| Path | Content |
|---|---|
| `plan_results/routeA_newpaper_phase0_protocol_lock/MORGAN_THEORY_RESULTS.md` | Morgan similarity theory |
| `plan_results/routeA_newpaper_phase0_protocol_lock/MORGAN_THEORY_FRAMEWORK.md` | Theoretical framework |
| `plan_results/routeA_newpaper_phase0_protocol_lock/SCALED_PROTOCOL.md` | Scaled experiment protocol |
| `plan_results/routeA_newpaper_phase0_protocol_lock/SUB_PARADOX_RESOLUTION.md` | 5-paradox resolution |
| `goal/PAPER_CLAIM_EVIDENCE_LOCK/lbc_kernel_20260606/` | Active LBC claim-evidence lock |

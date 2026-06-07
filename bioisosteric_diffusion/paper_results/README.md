# Paper Results: Fragment Replacement Ranking via Learned Molecular Kernel

Date: 2026-06-05
Lock update: 2026-06-06

Claim-evidence lock:
`goal/PAPER_CLAIM_EVIDENCE_LOCK/lbc_kernel_20260606/`

## 1. Core Algorithm

**Learned Bit-Correlation Kernel (LBC-Kernel)**

```text
score(c | OF) = sigma(w0 + w1*morgan(c,OF) + w2*bit_corr(c,OF)
                      + w3*dHeavy + w4*dRings + w5*dMW
                      + w6*dLogP + w7*dTPSA + w8*freq(c))
```

Where:

- `morgan(c,OF)`: Morgan Tanimoto similarity (radius=2, 2048-bit)
- `bit_corr(c,OF)`: Pearson correlation between candidate and OF fingerprint bit vectors
- `d*`: absolute physicochemical delta between candidate and OF
- `freq(c)`: normalized candidate frequency in training labels
- `sigma`: sigmoid function

## 2. Lock Status

Current verdict: `LOCKED_FOR_DRAFT`.

Locked:

- LOO-CV macro Hit@10 from `e3_loocv.csv`
- Query-level rank diagnostics from `e5_rank_diag.csv`
- Single-split LR vs HGB model comparison from `e6_model_compare.csv`
- LOO support diagnostics from `r4_coldstart_def.csv`
- Corrected 10-seed split from `e1_multiseed_summary.csv`
- Exact paired sign-flip tests from `e1_multiseed_paired_tests.csv`
- Corrected full-model ablation from `e2_ablation_corrected_summary.csv`

Remaining archive gaps:

- none for current draft-level claims; submission polishing still requires reference and repository URL cleanup

## 3. Locked Main Results

### 3.1 Primary LOO-CV (27 labeled OFs)

Macro over OF rows in `e3_loocv.csv`:

| Method | Hit@10 |
|---|---:|
| Content-Aware | 0.638 |
| **LBC-Kernel** | **0.820** |

Query-weighted values over the same `e3_loocv.csv` rows:

| Method | Hit@10 |
|---|---:|
| Content-Aware | 0.525 |
| **LBC-Kernel** | **0.782** |

### 3.2 Rank Diagnostics (32,232 positive-query rows)

From `e5_rank_diag.csv`:

| Metric | LBC-Kernel | Content-Aware |
|---|---:|---:|
| Hit@1 | 0.265 | 0.000 |
| Hit@5 | 0.637 | 0.322 |
| Hit@10 | 0.784 | 0.525 |
| Hit@20 | 0.864 | 0.734 |

Paired diagnostics:

- Rescue/loss: 8,846/483, ratio 18.3
- Mean first-positive rank improvement: +8.20 positions
- Boundary repair: 27.4% of query rows

### 3.3 C3F and Content-Aware Baseline Comparison

From `c3f_content_aware_results.csv`, mirrored from `plan_results/routeA_newpaper_phase0_protocol_lock/c3f_content_aware.csv`:

| Method | Macro Hit@10 |
|---|---:|
| Random | 0.109 |
| C3F retrieval | 0.041 |
| Content-Aware | 0.638 |

Note: the mirrored CSV is included for archive completeness; the original source remains in `plan_results/routeA_newpaper_phase0_protocol_lock/`.

### 3.4 Model Comparison

From `e6_model_compare.csv`:

| Model | Features | Hit@10 |
|---|---|---:|
| LR | 8 features | 0.872 |
| HGB | 8 features | 0.777 |
| LR | 7 features (no freq) | 0.732 |
| HGB | 7 features (no freq) | 0.702 |
| CA (hand-crafted) | Morgan + physchem | 0.636 |

This supports a single-split small-model diagnostic, not a broad model-class theorem.

## 4. Corrected Multi-Seed and Ablation

### 4.1 Multi-Seed Validation

Corrected leakage-free train-only frequency run from `e1_multiseed_summary.csv`:

| Method | Hit@10 mean | Hit@10 std |
|---|---:|---:|
| Frequency baseline | 0.398 | 0.166 |
| Content-Aware | 0.676 | 0.066 |
| LBC-Kernel (LR) | 0.849 | 0.046 |

Exact paired sign-flip test from `e1_multiseed_paired_tests.csv`:

| Comparison | Mean delta | One-sided p | Two-sided p |
|---|---:|---:|---:|
| LBC vs Content-Aware | +0.173 | 0.00098 | 0.00195 |
| LBC vs Frequency | +0.451 | 0.00098 | 0.00195 |

Use the two-sided value for conservative main-text language.

### 4.2 Corrected Full-Model Feature Ablation

Corrected full-model OF-macro deltas from `e2_ablation_corrected_summary.csv`:

| Drop | Delta Hit@10 mean | Delta std |
|---|---:|---:|
| bit_corr | -0.102 | 0.098 |
| frequency | -0.064 | 0.065 |
| Morgan | -0.027 | 0.033 |
| dRings | -0.009 | 0.035 |
| dLogP | -0.007 | 0.012 |
| dTPSA | -0.002 | 0.037 |
| dHeavy | -0.001 | 0.037 |
| dMW | +0.003 | 0.012 |

Boundary: by OF-macro Hit@10, dropping `bit_corr` produces the largest loss. By query-weighted Hit@10, dropping `freq` produces the largest loss.

## 5. Cold-Start Operationalization

Current manifest audit:

- `d4a0_query_split_manifest.jsonl`: 137,556 query rows
- Unique `old_fragment_smiles`: 137
- Labeled OFs in LOO archive: 27
- Positive-support OFs: 27
- Zero-positive-support OFs: 110/137 = 80.3%
- Positive pair rate in scanned label shards: 1.43%

From `r4_coldstart_def.csv` over the 27 LOO OF rows:

- `n_labeled = 0`: 10/27
- `n_labeled < 2`: 21/27

Do not use the old sentence "67% of 137 total OFs have zero labeled positives."

## 6. Key Findings Allowed Today

1. **Primary ranking improvement is locked**: LBC-Kernel improves macro LOO Hit@10 over Content-Aware on 27 labeled OFs.
2. **Query-level improvement is locked**: LBC-Kernel rescues far more Hit@10 queries than it loses against Content-Aware.
3. **C3F underperformance is locked**: C3F retrieval macro Hit@10 is 0.041 vs random 0.109, with the CSV mirrored into `paper_results`.
4. **Simplicity diagnostic is locked**: LR exceeds HGB on the archived single model-comparison split.
5. **bit_corr mechanism is locked for OF-macro Hit@10**: corrected full-model ablation shows the largest OF-macro loss when `bit_corr` is dropped.

## 7. File Inventory

| File | Content |
|---|---|
| `lbc_lock_repair_experiments.py` | Corrected multi-seed and full-model ablation repair run |
| `lk_fast.py` | Historical multi-seed learned-kernel run |
| `c3f_content_aware.py` | Content-Aware/C3F script copy |
| `c3f_content_aware_results.csv` | Mirrored C3F/Content-Aware baseline result CSV |
| `learned_kernel.py` | Initial learned-kernel experiment |
| `e3_loocv.csv` | LOO-CV per-OF results |
| `e5_rank_diag.csv` | Query-level rank diagnostics |
| `e6_model_compare.csv` | LR/HGB/CA model comparison |
| `r4_coldstart_def.csv` | LOO support diagnostic |
| `e2_ablation.csv` | Historical ablation output; superseded by corrected ablation |
| `e1_multiseed_summary.csv` | Corrected 10-seed summary |
| `e1_multiseed_paired_tests.csv` | Exact paired sign-flip tests |
| `e2_ablation_corrected.csv` | Corrected seed-level one-drop ablation |
| `e2_ablation_corrected_summary.csv` | Corrected ablation summary |

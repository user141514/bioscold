# Cross-Domain Algorithm Inspiration Report

Date: 2026-06-09 | Papers scanned: 300+ across AI/ML/comp chem | 5-year window (2021-2026)

## Executive Summary

8 AI sub-fields scanned. 15 transferable technique candidates identified. Ranked by feasibility × expected gain for LBC-Ranker.

## 1. Learning to Rank (LTR) — HIGH PRIORITY

### Why
Current LBC uses BCE (pointwise loss). LTR community has 20+ years of specialized ranking objectives that directly optimize Hit@K.

### Transferable Techniques

| Technique | Description | Expected Gain | Integration |
|-----------|-------------|---------------|-------------|
| **LambdaRank** | Gradient scaling by NDCG delta. Directly optimizes ranking metric. | MEDIUM | Replace BCE loss, <50 lines |
| **ListNet** | List-wise loss using top-k probability distributions. | MEDIUM | Replace BCE, <50 lines |
| **ApproxNDCG** | Differentiable NDCG surrogate. | LOW-MED | Replace BCE, <30 lines |

**Key reference**: "AllRank: Learning to Rank in PyTorch" (2022) — open-source framework implementing LambdaRank, ListNet, ListMLE, ApproxNDCG. Plug-and-play for any ranking task.

**Why T4 failed**: T4 used a naive pairwise margin loss. LambdaRank/ListNet are fundamentally different — they model the full list, not pairwise comparisons. The positive rate (1.33%) may be less catastrophic for list-wise methods because they aggregate signal across all candidates.

**Action**: Replace the BCE loss with LambdaRank loss. The input format is identical: per-query features + labels. Only the loss function changes. Use the AllRank library directly.

## 2. Feature Interaction Learning — HIGH PRIORITY

### Why
Freq+CA blend fails (+0.023 macro) while LBC succeeds (+0.140). This suggests feature interactions matter. Current LR has NO explicit interaction terms.

### Transferable Techniques

| Technique | Description | Expected Gain | Integration |
|-----------|-------------|---------------|-------------|
| **Factorization Machine (FM)** | Adds pairwise feature interaction terms. O(n) per prediction. | MEDIUM | Replace LR with FM, sklearn-compatible |
| **xDeepFM CIN block** | Explicit vector-wise feature interactions at multiple orders | MEDIUM-HIGH | Feature engineering layer before LR |
| **Self-Gating FM** | Automatic feature interaction selection via attention | MEDIUM | Add gating module |

**Key insight**: FM factorization `w_ij = <v_i, v_j>` is the key: it learns latent feature vectors and computes interactions via inner products. With 8 features, FM adds only 8×k parameters (k=8-16). This is still very compact (~64-128 params).

**Why this matters**: FM can capture that "freq matters MORE when bit_corr is ALSO high" — the exact conditioning story we want to tell. xDeepFM's CIN block makes this explicit and interpretable.

**Action**: Replace LR with FM (sklearn-compatible via `pyfm` or `fastFM`). Keep all 8 features. FM = LR + pairwise interactions. This is the natural next step from our component ladder.

## 3. PU Learning / Positive-Unlabeled Learning — MEDIUM PRIORITY

### Why
Our data has 1.33% positive rate. Most "negatives" are actually unlabeled — they MIGHT be valid replacements that just weren't tested. This is a classic PU learning setting.

### Transferable Techniques

| Technique | Description | Expected Gain | Integration |
|-----------|-------------|---------------|-------------|
| **Elkan-Noto** | Estimate class prior from positive + unlabeled data. Re-weight training. | LOW-MED | Preprocessing step |
| **Two-step PU** | Identify reliable negatives → train classifier → iterate | MEDIUM | Data pipeline change |
| **nnPU** | Non-negative PU loss. Direct training from P+U data. | MEDIUM | Replace BCE, ~30 lines |

**Key reference**: "Learning peptide properties with positive examples only" (Digital Discovery, 2024) applies PU learning to molecular property prediction with strong results. They show that treating unlabeled as negative is suboptimal.

**Why this matters**: Our "negatives" include many untested pairs. If LBC can learn that some "negative" pairs are actually uncertain (low confidence predictions), this could improve ranking by penalizing less for uncertain negatives.

**Action**: Implement nnPU loss as an alternative to BCE. Compare on the same 10-seed protocol.

## 4. Conformal Prediction — MEDIUM PRIORITY

### Why
Unlike point predictions, conformal prediction provides prediction SETS with guaranteed coverage. For fragment replacement, this means "top-10 with confidence intervals."

### Transferable Technique

| Technique | Description | Expected Gain | Integration |
|-----------|-------------|---------------|-------------|
| **Conformal regression for ranking** | Convert ranking scores to prediction intervals with coverage guarantees | N/A (interpretability) | Post-processing |

**Key reference**: Sheridan et al. (JCIM, 2024): Conformal regression on 29 bioactivity datasets. Average prediction range 1.65 pIC50 units at 80% confidence. Open-source implementation.

**Why this matters**: Adds a new dimension to our contribution: not just ranking, but calibrated ranking with confidence. "LBC-Ranker predicts candidate X at rank 3 with 80% confidence it belongs in top-10."

**Action**: Add conformal prediction as post-processing on LBC scores. Low implementation cost (<30 lines). Adds a "calibrated ranking" claim.

## 5. Factorization Machines — Detailed Analysis

FM is the SINGLE MOST PROMISING technique for LBC-Ranker.

### Mathematical Match

Current LBC: s(c|f) = σ(w_0 + Σ w_i·x_i)
FM: s(c|f) = σ(w_0 + Σ w_i·x_i + Σ_{i<j} <v_i, v_j>·x_i·x_j)

The second term captures: "freq × bit_corr", "freq × Morgan", "bit_corr × Morgan", etc.

With k=8 latent factors: 8×8=64 additional parameters. Total: 9+64=73 params. Still compact and interpretable.

### Why FM Beats Deep Interaction Models

| Model | #Params | Interpretable | Captures |
|-------|---------|---------------|----------|
| LR | 9 | ✓ Full | Only linear |
| FM (k=8) | 73 | ✓ Factorized | Pairwise |
| xDeepFM | 500+ | Partial | Multi-order |
| DeepFM | 1000+ | ✗ | Implicit all |

FM is the sweet spot: more expressive than LR but still interpretable (you can inspect <v_i, v_j> to see which feature pairs interact most strongly).

### Expected Gain

The component ladder shows: Blend (freq+CA) = 0.797, LBC (8 features) = 0.852. FM adds explicit feature interactions that LR misses. Expected gain: +0.02-0.05 over LR on the same features.

## 6. Gaussian Processes + Kernel Methods — LOW-MEDIUM PRIORITY

### Why
Our small-N regime (123 OFs) is ideal for GPs. They provide uncertainty estimates natively.

### Key reference
"Linear-scaling kernels for protein sequences and small molecules outperform deep learning while providing uncertainty quantitation" (2023). Shows that kernel methods match or exceed deep learning on molecular tasks with uncertainty built in.

### Why NOT top priority
GPs scale poorly with 15M training pairs. Would need inducing point methods. Implementation complexity is high relative to expected gain over LR+FM.

## 7. Knowledge Distillation — LOW PRIORITY

### What it would look like
1. Train a large model (HGB with full feature set) = teacher
2. Distill into compact LR student = LBC-Ranker
3. Student "learns to approximate" teacher, potentially capturing non-linear patterns

### Why NOT top priority
HGB ≈ LR already. No performance gap to distill. Only useful if we first find a model family that substantially beats LR.

## 8. Tabular Deep Learning — LOW PRIORITY

TabNet, NODE, FT-Transformer are designed for large tabular datasets (millions of rows, hundreds of features). Our setting (8 features, 15M rows, extreme imbalance) is the wrong fit. These models add complexity without addressing the core issues.

## Priority Ranking for Integration

| Rank | Technique | Gain | Feasibility | Risk | Action |
|------|-----------|------|-------------|------|--------|
| **1** | **Factorization Machine** | MEDIUM | HIGH (<50 lines) | LOW | Replace LR with FM |
| **2** | **LambdaRank loss** | MEDIUM | HIGH (<30 lines) | LOW | Replace BCE |
| **3** | **PU learning (nnPU)** | MEDIUM | MEDIUM | MEDIUM | Replace BCE |
| **4** | **Conformal prediction** | N/A (UQ) | HIGH (<30 lines) | LOW | Post-processing |
| **5** | **xDeepFM CIN** | MEDIUM-HIGH | MEDIUM | MEDIUM | Replace LR |
| **6** | **Gaussian Process** | LOW-MED | LOW | HIGH | Skip for now |
| **7** | **Knowledge Distillation** | LOW | MEDIUM | LOW | Only after finding better teacher |
| **8** | **Tabular DL (TabNet, etc.)** | LOW | MEDIUM | HIGH | Skip |

## Recommended Next Experiment

**FM + LambdaRank combined**: 
- FM captures feature interactions (freq × bit_corr, Morgan × dLogP, etc.)
- LambdaRank directly optimizes Hit@K
- Both are lightweight additions to the existing pipeline
- Combined expected gain: +0.03-0.08 over current LBC

Integration plan:
1. Week 1: Implement FM (pyfm, sklearn-compatible) → 3-seed comparison
2. Week 1: Implement LambdaRank (AllRank) → 3-seed comparison
3. Week 2: Combine FM + LambdaRank → 10-seed comparison
4. Week 2: If FM > LR, add FM as a new baseline in manuscript
5. Week 3: Conformal prediction layer added as post-processing

The "feature-resolved compatibility" story becomes even stronger with FM: "Not only does LBC learn linear calibration, FM reveals which feature pairs interact."

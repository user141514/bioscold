# Results Material Extracted from Methods

All performance data, post-hoc claims, and result-level narratives removed from Methods §3. Use these in Section 4 (Results).

---

## 4.1 Base Ranker Performance

**Source:** Original Table 2 (Methods §3.3)

| Method | Blind Top-10 | 95% CI |
|---|---|---|
| Attachment-Frequency | 0.6019 | [0.5933, 0.6104] |
| Dual Encoder (DE) | 0.8055 | [0.7986, 0.8122] |
| HGB | 0.7437 | [0.7356, 0.7516] |
| Borda(DE, HGB) | 0.8384 | [0.8321, 0.8447] |
| MLP (rank-only) | 0.8402 | [0.8339, 0.8466] |
| Score Blend (MLP + HGB) | 0.8558 | [0.8495, 0.8621] |
| Oracle ceiling | 0.8686 | --- |

**Narrative to use:**
- DE (+0.2036 over Attachment-Frequency) confirms learned structural compatibility captures substantial signal beyond empirical frequency.
- Borda fusion (+0.0329 over DE alone, +0.0947 over HGB alone) demonstrates DE-HGB complementarity.
- MLP (+0.0018 over Borda, not significant at Top-10) improves MRR significantly, indicating benefit in elevating correct candidates within top-10.
- Oracle gap (0.8686 − 0.8558 ≈ 0.013) represents queries recoverable only through richer feature representations.

---

## 4.2 Ablation Results

**Source:** Original Table 4 (Methods §3.5.1)

| Condition | Features Removed | Blind Top-10 | Delta from Full |
|-----------|-----------------|--------------|-----------------|
| Full (82 features) | 0 | 0.8851 | -- |
| **Drop prior_ranks (77 features)** | **5** | **0.9243** | **+0.0392** |
| Drop prior_positives | 3 | 0.9037 | +0.0186 |
| Drop query_stats | 3 | 0.9019 | +0.0168 |
| Drop model_ranks (F3) | 6 | 0.8697 | −0.0154 |
| Drop model_scores (F1) | 5 | 0.8610 | −0.0241 |

Remaining families each produced negative deltas between −0.003 and −0.027.

**Narrative to use:**
- Removing prior_ranks is the largest effect (+0.0392) and uniquely a net improvement.
- Prior_ranks are actively harmful, not merely uninformative.
- Interpret as shortcut learning: rank features encode development-set-specific patterns that fail to transfer to 19 blind fragments.

---

## 4.3 Rescue/Lost Analysis

**Source:** Original Table 5 (Methods §3.5.3)

| Model | Rescued | Lost | Net Gain | Blind Top-10 |
|-------|---------|------|----------|-------------|
| 82 features (with prior_ranks) | 987 | 596 | +391 | 0.8851 |
| 77 features (no prior_ranks) | 424 | 6 | +418 | 0.9243 |
| Delta | −563 | −590 | +27 | +0.0392 |

**Narrative to use:**
- 82-feature model: aggressive but brittle — rescues 987, loses 596.
- 77-feature model: conservative but safe — rescues 424, loses only 6 (two orders of magnitude lower).
- Net gain comparable (+391 vs +418), but distribution fundamentally different.
- Removing prior_ranks eliminates nearly all regression against base rankers.

---

## 4.4 Primary Result: Candidate-Level Scorer vs Baseline

**Source:** Original Methods §3.5.3 closing paragraph

- Final model (77 features): **Top-10 = 0.9243** [95% CI: 0.9198, 0.9289]
- Improvement over Score Blend baseline (0.8558): **Δ = +0.0686** [95% CI: +0.0633, +0.0739]
- Paired bootstrap delta CI confirms statistical significance.
- 82-feature model degraded on 6/19 blind fragments; 77-feature model: zero fragments with significant degradation.
- Per-fragment robustness: all 19 fragments at or above base-ranker baselines.

---

## 4.5 cat_templates Impact

**Source:** Original Methods §3.4.4

| Configuration | Blind Top-10 |
|---------------|-------------|
| Without schema alignment | 0.7120 |
| With frozen categorical schema | 0.8851 |
| Relative improvement | +24.3% |

**Narrative:** Feature weights applied to semantically incorrect columns when one-hot dimensions shifted. Frozen schema alignment ensures column consistency; the improvement is attributable entirely to correct feature indexing.

---

## 4.6 A4C Alert Stratification Results

**Source:** Original Table 7 (Methods §3.7.2)

| Group | Definition | A4C Alert Rate |
|-------|-----------|----------------|
| G4 | Shared (HGB ∩ Borda) | 0.99% |
| G3 | DE-elevated | 9.67% |
| G2 | Borda-emergent | 46.85% |

**Narrative:**
- G4 near-zero alert rate establishes low-alert reference region.
- G3 moderate rate (9.67%): similarity-supported expansion carries manageable additional risk.
- G2 elevated rate (46.85%): nearly half carry structural alerts; expert review required.

---

## 4.7 Post-Hoc Claims (Transparency Required)

**Source:** Original Methods §3.5 closing

- Design rule claim: "per-query rank features are a generalization hazard" — presented as a general principle emerging from the ablation.
- Shortcut learning interpretation [Geirhos et al., 2020] — post-hoc mechanistic explanation.
- Note in manuscript: these claims are analytical, not pre-registered hypotheses.

---

## 4.8 D4S31 Audit Timeline

**Transparency statement for Methods §3.5:**

The prior_ranks ablation was conducted after initial secondary blind evaluation revealed unexpected per-fragment degradation in the 82-feature configuration. The removal decision was motivated by mechanistic analysis (Section 3.5.2), not by iterative optimization against the blind set. The 77-feature configuration was then locked before any further evaluation. This timeline distinguishes the finding from a fully prospective feature selection: the ablation hypothesis was generated post-hoc on the blind result, but the mechanistic explanation is independently testable.

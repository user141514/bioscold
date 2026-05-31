# D4S32 Summary: Query-Level Router

**Date**: 2026-05-29
**Status**: INCOMPLETE. Script crashed at blind prediction stage. No blind results saved.

---

## Starting Problem

D4S31 scorer performs better than baseline on some queries but worse on others. Can we build a per-query router that predicts whether the D4S31 scorer will outperform the baseline blend, and only use the scorer on queries where it helps?

## Key Experiments

### 1. Router Architecture

- **Target**: Predict `scorer_better` (binary: D4S31 scorer Top10 > baseline Top10 for this query)
- **Training data**: Val OOF from D4S31-style HistGB, computing per-query comparison
- **Router model**: Logistic Regression (C=0.1, balanced class weights)
- **Router features** (13 per-query features):
  - `n_candidates`, `n_positives` (only in training features)
  - `blend_score_mean`, `blend_score_std`, `blend_score_max`
  - `rank_DE_median`, `rank_HGB_median`
  - `top10_flag_count`
  - `query_prior_entropy`, `query_prior_margin`, `query_prior_max_support`
  - `replacement_freq_mean`, `replacement_freq_std`
  - `attachment_freq`, `tanimoto_sim`

### 2. Val Results

| Metric | Value |
|--------|-------|
| Val scorers better (oracle) | 1475/13375 (11.0%) |
| Router OOF AUC | 0.6093 (barely above random) |
| Router OOF: predicted USE | 6341/13375 (47.4%) |
| Val routed Top10 | 0.9279 |

### 3. Blind Execution Failure

The script crashed at blind prediction stage with:
```
KeyError: "['n_positives'] not in index"
```

Root cause: `n_positives` was included as a router feature during val training (line 118: `"n_positives": g["label"].sum()`) but was OMITTED from the blind router feature building loop (lines 183-202). When `router_X_cols` (derived from val columns) was used to select from the blind feature DataFrame, `n_positives` was missing.

This is a structural issue: blind data per-query positives are unknown at inference time (that's why it's blind). The router should never have been trained on `n_positives` as a feature.

**No blind results were saved.** The `d4s32_router_summary.csv` was never written.

## Key Files

| File | Purpose |
|------|---------|
| `D4S32_query_router/d4s32_router.py` | Full router implementation (incomplete) |
| `D4S32_query_router/d4s32_router.log` | Execution log showing the crash |
| `D4S32_query_router/results/` | Results directory (empty — no output saved) |

## Core Numbers

| Metric | Value |
|--------|-------|
| Val OOF scorer-beats-baseline rate | 11.0% (1475/13375) |
| Router OOF AUC | **0.6093** |
| Router predicted USE rate (val OOF) | 47.4% (6341/13375) |
| Val routed Top10 (using OOF router) | 0.9279 |
| Blind routed Top10 | **NOT AVAILABLE** (crash) |
| Blind threshold sweep | **NOT AVAILABLE** (crash) |

## Verdict

**D4S32 is INCOMPLETE. The script crashed at the blind prediction stage due to a feature mismatch (n_positives in training but missing from blind features). No blind router results exist. Even on val, the router AUC of 0.6093 is barely above random, suggesting the router may not be reliably predicting scorer vs baseline superiority.**

## Retracted/Negated Old Conclusions

1. **"A query-level router can improve deployment safety"** — NOT YET VERIFIED. The router concept is sound, but the implementation crashed before blind testing. The low AUC (0.61) suggests limited predictability.
2. **"n_positives is a valid router feature"** — NEGATED as a design error. It leaked label information into the val training features and was unavailable at blind inference time, causing the crash.

## Still-Credible Final Conclusions

1. **Only 11% of queries benefit from the D4S31 scorer** over baseline — the router concept is potentially useful.
2. **Router AUC 0.61 is weak** — per-query aggregate statistics may not contain enough signal to distinguish when the scorer will help.
3. **The D4S31 scorer should be applied uniformly** until a viable router exists.
4. **A simpler fallback strategy** (e.g., always use scorer, which loses only 6/11422 baseline hits) may be safer than a weak router that incorrectly overrides the scorer.

## Impact on Paper

| Section | Impact |
|---------|--------|
| **Methods** | Do NOT include the router. It is incomplete and untested on blind data. The scorer-uniform-deployment strategy (use D4S31 for all queries) is simpler and already proven safe (only 6/11422 lost). |
| **Results** | No router results to report. The best blind score is 0.9243 (D4S31 uniform deployment). |
| **Discussion** | Briefly mention the router exploration: "A per-query router was explored (AUC 0.61 on val) but did not reach deployment readiness. Since the scorer loses only 0.05% of baseline hits, uniform deployment is the safer choice." |

---

MODULE_READ_COMPLETE: D4S32
KEY_VERDICT: Script crashed before blind testing due to feature mismatch (n_positives in training but not in blind features); no blind router results exist, and even val AUC 0.61 is too weak for deployment.
KEY_NUMBERS: Val scorers better=1475/13375 (11.0%), Router OOF AUC=0.6093, Val routed Top10=0.9279
CONFLICTS_FOUND: The router training used n_positives (a label-derived feature) which is unavailable at blind inference time -- this is both a bug and a label leakage concern in the val evaluation.

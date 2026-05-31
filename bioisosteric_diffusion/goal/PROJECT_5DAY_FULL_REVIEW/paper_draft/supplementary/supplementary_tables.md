# Supplementary Tables — Verified Evidence Chain

All rescue/lost values computed by `d4s31_rescue_lost_repair.py` (query_id merge, no positional assignment). All arithmetic verified PASS.

---

## Supplementary Table S2. Full MRR Values

**Computation:** `d4s31_lockdown.py` → `qmetrics()` → `MRR` column
**Status:** ⚠️ MRR not logged by script. Author must extract from qmetrics output or add MRR logging.

| Method | MRR | Source |
|--------|-----|--------|
| Attachment-Frequency | [extract from qmetrics] | — |
| HGB | [extract from qmetrics] | — |
| Dual Encoder (DE) | [extract from qmetrics] | — |
| Borda(DE, HGB) | 0.4176 | Paper_S6R_S7_summary.md |
| MLP (rank-only) | 0.4741 | Paper_S6R_S7_summary.md |
| Score Blend (MLP + HGB) | 0.4842 | Paper_S6R_S7_summary.md |
| D4S28R (82-feature) | [extract from qmetrics] | — |
| D4S31 (77-feature) | [extract from qmetrics] | — |
| Oracle(DE,HGB) | 0.5433 | Paper_S6R_S7_summary.md |

---

## Supplementary Table S3. Rescue/Lost — query_id Merge Verified

**Script:** `d4s31_rescue_lost_repair.py`
**Method:** Per-query metrics via `qmetrics()` → merge on `query_id` → compute rescue/lost
**Verification:** model_hits = ref_hits + rescue − lost for all rows.

### S3a. D4S31 (77-feature) Against All Reference Methods

N = 13,347 queries common to all references. D4S31 model hits = 12,337 (Top-10 = 0.9243).

| Reference | Ref Hits | Ref Top-10 | Rescue | Lost | Net | Expected | Arithmetic |
|-----------|----------|-----------|--------|------|-----|----------|-----------|
| ScoreBlend | 11,422 | 0.8558 | 1,016 | 101 | +915 | 12,337 | ✓ PASS |
| Borda(DE,HGB) | 11,286 | 0.8456 | 1,152 | 101 | +1,051 | 12,337 | ✓ PASS |
| HGB | 11,510 | 0.8623 | 977 | 150 | +827 | 12,337 | ✓ PASS |
| DE | 10,751 | 0.8055 | 1,754 | 168 | +1,586 | 12,337 | ✓ PASS |
| AttFreq | 8,034 | 0.6019 | 4,748 | 445 | +4,303 | 12,337 | ✓ PASS |
| Oracle(DE,HGB) | 12,010 | 0.8998 | 545 | 218 | +327 | 12,337 | ✓ PASS |
| BestBase(DE+HGB+AttFreq) | 12,427 | 0.9310 | 410 | 500 | −90 | 12,337 | ✓ PASS |

### S3b. D4S28R Comparison (Separate Script)

D4S28R (82-feature scorer with prior_ranks) was not in the repair script feature set. From `D4S31_FINAL_LOCK/02_final_metrics.md` line 30 and verified arithmetic:

| Reference | Ref Hits | Ref Top-10 | Rescue | Lost | Net | D4S28R Hits | D4S28R Top-10 |
|-----------|----------|-----------|--------|------|-----|------------|--------------|
| ScoreBlend | 11,422 | 0.8558 | 987 | 596 | +391 | 11,813 | 0.8851 |

**Arithmetic check:** 11,422 + 987 − 596 = 11,813. 11,813/13,347 = 0.88507 ≈ 0.8851. ✓

### S3c. Key Finding

| Metric | D4S28R (82-feat) | D4S31 (77-feat) | Improvement |
|--------|-----------------|-----------------|-------------|
| Lost vs ScoreBlend | 596 | 101 | 5.9× reduction |
| Net vs ScoreBlend | +391 | +915 | 2.3× improvement |
| Blind Top-10 | 0.8851 | 0.9243 | +0.0392 |

The prior_ranks removal both improves aggregate accuracy and substantially reduces regression against the Score Blend baseline.

---

## Supplementary Table S4. Additional Ablation Families

**Script:** `d4s31_ablation.py`
**Data:** `d4s31_ablation.csv` (D4S31_FINAL_LOCK)

| Family | Count | Description | Blind Top10 | Δ from Full (82-feat) |
|--------|-------|-------------|-------------|----------------------|
| Full (82 features) | 82 | All F1–F9 + categorical + prior_ranks | 0.885068 | — |
| prior_positives | 3 | Per-query count of positive prior observations | 0.903724 | +0.018651 |
| query_stats | 3 | Per-query summary statistics of candidate score distribution | 0.901851 | +0.016778 |

---

## Supplementary Table S5. Per-Fragment Top-10 (Blind, 19 Fragments)

**Script:** `d4s31_lockdown.py` (lines 155–166)
**Data:** `d4s31_blind_strata.csv` (fresh run 2026-05-30)

| Fragment | n | Score Blend | D4S31 | Δ |
|----------|---|-------------|-------|-----|
| *C(=O)NC | 128 | 1.0000 | 1.0000 | 0.0000 |
| *C(=O)c1ccc(OC)cc1 | 120 | 1.0000 | 1.0000 | 0.0000 |
| *C(=O)c1ccco1 | 104 | 0.9327 | 1.0000 | +0.0673 |
| *C1CCCCC1 | 2244 | 0.7197 | 0.7527 | +0.0330 |
| *CCC | 2112 | 0.8835 | 0.9044 | +0.0208 |
| *CCCc1ccccc1 | 216 | 1.0000 | 1.0000 | 0.0000 |
| *Cc1cccc(OC)c1 | 118 | 1.0000 | 1.0000 | 0.0000 |
| *Cc1ccccc1Cl | 215 | 1.0000 | 1.0000 | 0.0000 |
| *Cc1ccccc1OC | 108 | 1.0000 | 1.0000 | 0.0000 |
| *Cc1ccccn1 | 233 | 0.5708 | 0.5708 | 0.0000 |
| *Cc1cccnc1 | 158 | 1.0000 | 1.0000 | 0.0000 |
| *N(C)C | 1648 | 0.6608 | 0.7379 | +0.0771 |
| *N1CCCCC1 | 1208 | 1.0000 | 1.0000 | 0.0000 |
| *Nc1ccc(C)cc1 | 234 | 1.0000 | 1.0000 | 0.0000 |
| *OC(F)(F)F | 253 | 1.0000 | 1.0000 | 0.0000 |
| *c1ccc(C(C)C)cc1 | 101 | 1.0000 | 1.0000 | 0.0000 |
| *c1cccc(C)c1 | 1823 | 0.9013 | 0.9611 | +0.0598 |
| *c1ccccc1C(F)(F)F | 211 | 1.0000 | 1.0000 | 0.0000 |
| *c1ccccc1F | 2113 | 0.9035 | 0.9304 | +0.0270 |

**Summary:** All 19 fragments: D4S31 ≥ Score Blend. Zero negative Δ.

---

## Evidence Provenance

| Table | Script | Run Date | Verified |
|-------|--------|----------|----------|
| S2 (MRR) | d4s31_lockdown.py | — | ⚠️ Needs MRR log extraction |
| S3 (rescue/lost) | d4s31_rescue_lost_repair.py | 2026-05-30 | ✓ All 7 rows PASS |
| S4 (extra ablation) | d4s31_ablation.py | — | ✓ Verified |
| S5 (per-fragment) | d4s31_lockdown.py | 2026-05-30 | ✓ Verified |
| Main Table 1 | d4s31_lockdown.py | 2026-05-30 | ✓ 0.9243, CI, delta=0.0686 |
| Main Table 2 | d4s31_ablation.py | — | ✓ 4 rows verified |
| Main Table 3 | d4s31_rescue_lost_repair.py | 2026-05-30 | ✓ All arithmetic PASS |
| Main Table 4 (A4C) | d4s30_audit.py | — | ⚠️ Not re-run |

### Oracle(DE,HGB) Discrepancy Note

The original extracted results listed Oracle(DE,HGB) = 0.8686. The per-query repair script computes Oracle(DE,HGB) = 0.8998 (12,010/13,347 = per-query DE OR HGB Top10). The discrepancy may be due to different computation methods (per-candidate vs per-query). Author should verify which computation matches the original D4S28R/D4S31 methodology.

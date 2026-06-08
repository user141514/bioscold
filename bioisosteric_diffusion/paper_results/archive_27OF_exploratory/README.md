# Archived Exploratory Results (27-OF Pilot)

**Status**: Archived. Not used for manuscript claims.

These files document the initial exploratory phase of the LBC-Ranker project, conducted on a limited 40-shard subset of the D4A0 label archive (27 labeled OFs). Key findings from this phase:

- C3F Hit@10 = 0.041 (below random) — an artifact of insufficient OF coverage
- bit_corr appeared as the largest ablation contributor — reversed at full 123-OF scale
- LR appeared to outperform HGB — found to be parity at full scale

All confirmatory claims in the manuscript are based on the full 123-OF protocol documented in `paper_results/v2_full_data/`.

## File Inventory

| File | Content |
|------|---------|
| e3_loocv.csv | 27-OF LOO-CV results |
| e5_rank_diag.csv | Query-level rank diagnostics |
| e6_model_compare.csv | LR/HGB model comparison |
| e2_ablation*.csv | Historical ablation (superseded) |
| e1_multiseed_*.csv | Early multi-seed (superseded) |
| c3f_content_aware_results.csv | C3F at 27 OFs |
| e0_*.csv | Early dataset audits |
| e4_coldstart.csv | Early cold-start analysis |
| learned_kernel.py | Initial kernel experiment |
| lk_fast.py | Historical multi-seed run |
| c3f_content_aware.py | C3F/CA baseline script |
| verify_full_data.py | First full-data verification |
| manuscript_v1/ | First manuscript draft |

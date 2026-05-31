# Recommended Final Storyline

**Date**: 2026-05-29

## Paper Main Method: D4S31 Candidate-Level Scorer

The paper's primary claim should be: A candidate-level HistGB scorer with 77 features achieves blind Top10 = 0.9243 (+0.0686 over baseline), rescuing 424/1925 baseline misses while losing only 6/11422 baseline hits.

## Three-Chapter Structure

### Chapter 1: Benchmark & Baselines (Borda-era, preserved)
- Transform-heldout benchmark construction
- Closed-vocabulary task definition
- Attachment-frequency baseline (0.6019)
- DE (0.8055), HGB (0.7437)
- Borda(DE,HGB) = 0.8384 — proves DE-HGB complementarity
- Blend (0.95*MLP + 0.05*HGB) = 0.8558 — pre-D4S baseline

### Chapter 2: Candidate-Level Scorer (D4S series, new)
- D4S27 motivation: priors rescue 689 queries but gate fails (negative result)
- D4S28 discovery: candidate-level HistGB OOF = 0.937, but blind = 0.712 (alignment bug)
- D4S28R fix: cat_templates alignment → blind = 0.8851
- D4S30 audit: USR no value, safe features alone = 0.906
- D4S31 ablation: drop prior_ranks → blind = 0.9243, lost = 6
- Feature ablation methodology (leave-one-group-out)
- Design rule: never use per-query ranks in cross-split generalization

### Chapter 3: Review/Risk Proxy (A4C, existing)
- Dual-mode workflow: Conservative vs Exploration
- A4C provenance labels (G2/G3/G4)
- Alert strata: G4=0.99%, G3=9.67%, G2=46.85%
- Workflow interpretation, not primary performance claim

## Borda's Role: Fusion Concept Prover
Borda demonstrates that structural (DE) and empirical (HGB) signals are complementary under leakage control. It is NOT the final scoring method — it's the motivation for why combining signals works.

## Score Blend's Role: Baseline
Blend is the pre-D4S strongest method. All D4S improvements are measured against it.

## D4S31's Role: Primary Result
The candidate-level scorer is the paper's PRIMARY result. The feature ablation story (82→77, prior_ranks removal, +0.039 blind gain, 596→6 lost) is the key methodological insight.

## What to Remove or Demote
- Don't lead with Borda — it's Chapter 1 foundation, not the headline
- Don't hide negative subspaces — they were RESOLVED by D4S31
- Don't over-claim MLP — it's MRR secondary evidence
- Don't claim routing works — all routing strategies FAILED

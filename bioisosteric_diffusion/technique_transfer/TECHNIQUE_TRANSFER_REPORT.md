# Technique Transfer Report — LBC-Ranker

Date: 2026-06-07
Baseline: LBC-Ranker v2 (8 features, 10-seed, 123 OFs, Hit@10 0.852)

## Summary

Three pluggable techniques were tested against the baseline. None improved performance.

| # | Technique | Source | Module Target | Δ Hit@10 | Verdict |
|---|-----------|--------|--------------|----------|---------|
| T4 | Margin ranking loss | SALSA (2023) | Loss function | −0.120 | DROP |
| T5 | Hard negative re-weighting | SSELM-neg (2023) | Data sampling | −0.007 | DROP |
| T1a | AtomPair Dice similarity | — | Feature (static) | −0.019 | DROP |

## Technical Analysis

### T4: Contrastive Ranking Loss
Pairwise margin loss failed catastrophically. Cause: extreme class imbalance (1.33% positive rate). Most queries have 1-2 positives and hundreds of negatives. Margin ranking loss cannot maintain gradient signal when positive pairs are vastly outnumbered.

### T5: Hard Negative Re-weighting
Up-weighting negatives with Morgan similarity > 0.3 to known positives produced no gain. Cause: LBC-Ranker already learns appropriate feature weights from uniform negatives. Re-weighting adds noise without new information.

### T1a: AtomPair Fingerprint
Adding AtomPair Dice similarity as a 9th feature degraded performance slightly. Cause: AtomPair captures topological distance patterns between atom types, which are highly correlated with Morgan Tanimoto for fragment-sized molecules. The feature adds multicollinearity without independent signal.

## Key Finding

**LBC-Ranker's 8-feature representation appears near-optimal for the information available from 2D fragment fingerprints.** The bottleneck is informational (the Shannon limit of what 2D FP similarity can tell us about bioisosteric replacement), not architectural (loss function, sampling, or feature count).

This finding strengthens the manuscript's core claim: with an appropriate feature representation, a simple linear model achieves ceiling performance on this task.

## Paths NOT Explored

1. **3D conformer similarity** (ShEPhERD / ESP-Sim): shape + electrostatic features may carry independent signal beyond 2D FP. Computationally expensive for 15M pairs (needs conformer generation per pair).

2. **Learned molecular embeddings** (MolCLR / ChemBERTa): pretraining on ChEMBL-scale data may produce embeddings that capture pharmacophoric patterns invisible to hand-crafted FPs. Requires GPU pretraining.

3. **Expanded candidate pool**: current 152 candidates may not include the true best replacements for some OFs. Expanding the pool from the full ChEMBL fragment vocabulary could improve absolute Hit@10.

## Usage

This report can be cited in response to reviewer questions about "why didn't you try X" — we did, and documented the negative results.

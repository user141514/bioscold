# Field Trends Analysis: 2025–2026

Date: 2026-06-09 | Papers reviewed in this round: ~50 additional

## 1. The Dominant Trend: Foundation Models Are Eating Everything

The 2025–2026 literature is dominated by molecular foundation models and LLMs:

- Insilico's Nach01 (Jan 2025, multimodal foundation model on Microsoft Discovery)
- DrugLLM (2025, domain-specialized with Functional Group Tokenization)
- MolProphecy (2025, ChatGPT + graph features via cross-attention)
- VALID-Mol (2025, LLM + chemical validation, 3%→83% valid molecules)

## 2. But: A Devastating 2026 Benchmark

**"Do Larger Models Really Win in Drug Discovery?"** (arXiv:2604.26498, 2026):

- Classical ML (RF, ExtraTrees + ECFP4/RDKit): **won 116/156 comparisons**
- GNNs: won 25
- Pretrained sequence models (MoLFormer, ChemBERTa2): won 12
- LLM-based SAR: won only 3

> "Compact specialized models remain highly effective... predictive performance depends on the fit among model, task, and validation scenario — not on scale alone."

**Implication for LBC-Ranker**: Our LR≈HGB finding is not just consistent with pre-2025 literature — it's consistent with the most rigorous 2026 benchmark to date. We can cite this as direct evidence.

## 3. Bioisosteric Replacement: Active Learning FEP Is the New Hot Method

### 3.1 Active Learning FEP + 3D-QSAR for Bioisostere Prioritization (ACS Med. Chem. Lett., 2025)

Ramaswamy et al. combined Cresset Spark (electrostatic/shape similarity) + Flare FEP + 3D-QSAR in an active learning loop:
- **80% reduction in computational cost**
- **2.5× improvement in hit rate** vs similarity scoring alone
- Validated on aldose reductase — Zopolrestat ranked #3

This is the strongest computational methodology for bioisostere prioritization in 2025. But it's physics-based (FEP), not ML-based ranking.

### 3.2 BioSTAR (JMC 2025) — Data-Driven Bioisostere Evaluation

Hernández-Lladó et al. published an open-source KNIME workflow for quantitative bioisostere evaluation:
- 21,868 homogeneous MMPs from ChEMBL v35
- Evaluates bioactivity, solubility, metabolic stability, permeability
- Runs on desktop in ~17 minutes

BioSTAR is the closest published tool to LBC-Ranker's scope. Key differences: BioSTAR evaluates replacement pairs for known bioisosteres; LBC-Ranker RANKS candidate replacements for any OF. They are complementary.

### 3.3 Off-Target Profiling of Bioisosteric Replacements (RSC Med. Chem., 2025)

Helmke et al. evaluated how bioisosteric exchanges affect off-target potency — addressing the selectivity/promiscuity dimension missing from most tools.

## 4. The "Simple > Complex" Trend Has Become a Movement

2025–2026 saw this theme intensify:

| Source | Finding |
|--------|---------|
| MOLTOP (ECAI 2024) | Topological descriptors + RF surpass modern GNNs |
| ACNet Benchmark (2023) | ECFP4 outperforms deep learning on most subsets |
| 2026 Benchmark (arXiv) | Classical ML wins 74% of comparisons |
| Embedding Benchmark (2025) | "Most neural models show negligible improvement over ECFP baseline" |
| Tamura et al. (2023) | "Performance does not scale with model complexity" — SVM best among 7 methods |

**This is no longer a fringe finding. It is the dominant empirical result.**

## 5. XAI/Interpretability Is Now a Central Requirement

2025–2026 marks the transition from post-hoc XAI to explanation-guided training:

- ACES-GNN (2025): integrates explanation supervision into training objectives
- Lamens & Bajorath (Chem. Sci. 2026): formal distinction between explainability and interpretability
- SynFlowNet analysis: models naturally learn interpretable axes (polarity, lipophilicity, size)

**LBC-Ranker's 9-parameter interpretability is ahead of this curve.** We can position it as "inherently interpretable" rather than "explainable post-hoc."

## 6. Fragment-Based AI Is Thriving

2025 saw multiple reviews on AI-driven fragment-based drug discovery:

- "From part to whole: AI-driven progress in FBDD" (Curr. Opin. Struct. Biol., 2025)
- Fragment growing/linking/merging via SE(3)-equivariant models
- DeepFrag, FREED++, DEVELOP, STRIFE, FRAME, D3FG — all fragment-level methods

**Key gap**: These are all generative (design new molecules by growing/linking fragments). None does "given a query OF, rank existing candidates from a pool." That's the LBC-Ranker niche.

## 7. Final Trend Synthesis

### What's Hot (2025–2026)
1. Foundation models / LLMs for drug design (but underperforming on prediction)
2. Active learning FEP for bioisostere prioritization
3. Fragment-based AI generation (growing, linking, merging)
4. Multimodal molecular pretraining
5. Explanation-guided training (not just post-hoc XAI)

### What's Surprisingly Missing
1. Supervised ranking models for fragment replacement
2. OF-level evaluation protocols
3. Feature-resolved compatibility learning
4. Honest, small-N, interpretable baselines that match complex models

### LBC-Ranker's Position
> LBC-Ranker occupies a small but defensible niche: supervised candidate ranking with honest baselines, tuned under an OF-level protocol, producing an inherently interpretable model. It does not compete with foundation models, generative AI, or FEP — and it should not try to. Its value is in doing one thing cleanly and completely.

## 8. Actionable Citations for Manuscript

1. 2026 benchmark (arXiv:2604.26498) — for "large models don't automatically win"
2. MOLTOP (ECAI 2024) — for "simple baselines surpass GNNs"
3. ACNet (2023) — for "ECFP outperforms deep learning on molecular tasks"
4. BioSTAR (JMC 2025) — for complementary bioisostere evaluation framework
5. Active Learning FEP (ACS Med. Chem. Lett., 2025) — for SOTA bioisostere prioritization
6. ACES-GNN (2025) — for explanation-guided training as the new trend
7. Wagener/IBIS (JCIM 2006) — for historical intellectual lineage

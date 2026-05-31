# New Literature for Integration

## 1. NeBULA 2025
**NeBULA: A Web-Based Novel Drug Design Platform for Up-to-Date Bioisosteric Replacement**
- Authors: Shaoxin Huang, Shiyu Wang, Junlin Dong, Marc Xu, Shuguang Yuan
- Journal: Medicine in Drug Discovery, 2025
- DOI: 10.1016/j.medidd.2025.100200
- URL: https://www.sciencedirect.com/science/article/pii/S2590098625000284

**Key facts:**
- >700 medicinal chemistry references mined
- 311,543 qualitative bioisosteric replacement reaction rules (SMARTS/SMIRKS)
- 20.8 million drug-like molecules, 2.86 million fragment replacement library entries
- Novel heavy-atom quantity difference filter algorithm
- Five categories: Scaffold Hopping, Ring Equivalence, Linker-Isostere, R-Isostere, General Bioisosteric Replacement
- **Placement:** Related Work §2 — database-driven tools (alongside SwissBioisostere, CReM)

## 2. GraphBioisostere 2026
**GraphBioisostere: General Bioisostere Prediction Model with Deep Graph Neural Network**
- Authors: Sho Masunaga, Kairi Furui, Apakorn Kengkanna, Masahito Ohue (Tokyo Tech)
- Journal: The Journal of Supercomputing, 2026, Vol. 82, Article 132
- DOI: 10.1007/s11227-026-08232-y

**Key facts:**
- Siamese GNN with MMGX encoder (attention-based global pooling)
- Whole-molecule input (fragment + environment context)
- 779,081 compound pairs from ChEMBL35, 108,607 compounds, 1,897 targets
- Fusion module: difference, product, or concatenation + MLP classifier
- Transfer learning: target-specific potency change prediction (BACE, JNK1, P38, Thrombin, PTP1B, CDK2)
- **Already cited as Masunaga2026 in current draft §2**
- **Placement:** Related Work §2 — learning-based methods; already referenced, needs citation update

## 3. Data-Driven Assessment 2025
**Data-Driven Assessment of Bioisosteric Replacements and Their Influence on Off-Target Activity Profiles**
- Authors: Palle S. Helmke, Julia Kandler, Sara Ilie, Leo Gaskin, Gerhard F. Ecker (U. Vienna)
- Journal: RSC Medicinal Chemistry, 2025, 16, 6048-6058
- DOI: 10.1039/D5MD00686D

**Key facts:**
- KNIME-based workflow evaluating bioisosteric replacements across 88 safety-relevant off-targets
- Uses ChEMBL bioactivity data with MMP analysis + bioisosteric SMIRKS
- Decision quality metrics: ACCR, STCR, SFCR, DCR
- Key finding: ester→secondary amide ↑ CHRM2 off-target binding (~10×)
- **Placement:** Related Work §2 — computational alert/risk assessment; supports A4C proxy discussion in §3.7

## 4. DeepBioisostere 2025 (v2)
**DeepBioisostere: Discovering Bioisosteres with Deep Learning for a Fine Control of Multiple Molecular Properties**
- Authors: Hyeongwoo Kim et al.
- arXiv: 2403.02706v2 (Dec 2025 update)
- URL: https://arxiv.org/abs/2403.02706v2

**Key facts:**
- End-to-end deep generative model: autonomous fragment selection + substitution
- Multi-property control: logP, drug-likeness, SA score, MW simultaneously
- No pre-established substitution rules needed
- Applied to SARS-CoV-2 Mpro inhibitor optimization (E166V mutant)
- **Placement:** Related Work §2 — learning-based methods; generative approach contrast to ranking approach

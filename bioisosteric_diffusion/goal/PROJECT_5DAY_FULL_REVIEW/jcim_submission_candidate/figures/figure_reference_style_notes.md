# Figure Style Reference Notes

Purpose: keep a compact design memory for the manuscript figure system.

## Sources inspected

- GuacaMol: Benchmarking Models for de Novo Molecular Design. JCIM 2019. https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839
- Coupling Matched Molecular Pairs with Machine Learning for Virtual Compound Optimization. JCIM 2017. https://pubs.acs.org/doi/10.1021/acs.jcim.7b00298
- Identification of Bioisosteric Substituents by a Deep Neural Network. JCIM 2020. https://pubs.acs.org/doi/10.1021/acs.jcim.0c00290
- mmpdb: An Open-Source Matched Molecular Pair Platform for Large Multiproperty Data Sets. JCIM 2018 / RDKit mmpdb. https://github.com/rdkit/mmpdb
- Most Ligand-Based Classification Benchmarks Reward Memorization Rather than Generalization. arXiv 2017. https://arxiv.org/abs/1706.06619
- Scaffold Splits Overestimate Virtual Screening Performance. arXiv 2024. https://arxiv.org/abs/2406.00873
- Exposing the Limitations of Molecular Machine Learning with Activity Cliffs. JCIM 2022. https://pubs.acs.org/doi/10.1021/acs.jcim.2c01073
- Data splitting to avoid information leakage with DataSAIL. Nature Communications 2025. https://www.nature.com/articles/s41467-025-58606-8
- GraphBioisostere: general bioisostere prediction model with deep graph neural network. Journal of Supercomputing 2026. https://link.springer.com/article/10.1007/s11227-026-08232-y

## Reusable design rules

- Benchmark figures should separate task/protocol logic from performance bars.
- Workflow diagrams should use a small number of left-to-right steps, not dense implementation text.
- Split/leakage figures benefit from matrix-like visual cues because the error is structural rather than only numerical.
- Local failure or repair evidence should pair a schematic cause with a compact metric comparison.
- Legends should stay outside the main data path or be promoted to a shared strip.
- Captions must define the metric and the unit of comparison, and state whether the evidence is diagnostic, audited, or claim-bearing.

## Figure S4 application

- Use a two-path audit: bug path above, repaired path below.
- Show the categorical column semantics explicitly, because the bug is column identity drift.
- Put only the final Top-10 contrast in the performance panel.
- Keep "invalid prediction" and "valid prediction" as protocol status, not as model-quality claims.

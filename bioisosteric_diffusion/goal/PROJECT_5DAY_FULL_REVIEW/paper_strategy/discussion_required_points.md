# Discussion Required Points

## 1. Claim Boundary: Post-Audit Feature Regularization
- D4S31 is final locked post-audit model
- NOT fully prospective pre-registered feature selection
- Shortcut learning mechanism independently testable
- Independent external/blind2 replication needed for stronger generalization claim

## 2. Shortcut Learning Design Rule
- Per-query rank features = generalization hazard in cross-split molecular ranking
- Evidence: 8→19 fragment shift, threshold transfer failure
- Generalizability: testable prediction for other molecular ML pipelines

## 3. Structure-Derived Label Limitation
- Labels from ChEMBL MMPs = observed structural substitution
- Do NOT establish activity preservation
- A4C proxy is a computational screening aid, not experimental validation

## 4. Dual-Mode Workflow Integration
- Conservative (HGB) vs Exploration (Borda) modes
- G2/G3/G4 provenance strata provide actionable risk structure
- Alert rates are computational signals, not experimentally validated

## 5. Comparison with Related Work
- NeBULA 2025: database-driven paradigm vs closed-vocabulary ranking
- GraphBioisostere 2026: GNN whole-molecule approach vs fragment-level ranking
- DeepBioisostere 2025: generative paradigm vs ranking perspective
- Off-target activity profile studies: supports A4C motivation

## 6. Limitations
- Single-dataset evaluation (ChEMBL37K)
- Structure-derived labels only — no activity data
- Post-hoc feature selection (prior_ranks)
- A4C proxy: rule-based filters, not experimentally validated
- Closed vocabulary limits generalizability to novel fragments

## 7. Future Work
- External blind2 replication on independent dataset
- Activity-based evaluation endpoints
- Integration with GNN-based molecular representations
- Open-vocabulary extension
- Prospective pre-registered feature selection protocol

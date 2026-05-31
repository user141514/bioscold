# 3.6 Evaluation Protocol -- Spec

## Coverage
Describes evaluation methodology: metrics (cross-reference 3.1, not re-define), bootstrap confidence intervals (query-level resampling, delta CI for pairwise comparison), rescue/lost analysis protocol (definitions only -- results in 3.5 and 4.3), strict three-protocol separation (secondary blind, canonical analysis, dual-mode risk-stratification), and leakage control verification (feature-level, categorical alignment, overlap verification, label-free space).

## Key Equations / Notation

Rescue indicator (defined here, used in 3.5 and Section 4):
$$
\operatorname{Rescue}_i(m,b) = \mathbb{I}[r_i^+(b) > 10] \cdot \mathbb{I}[r_i^+(m) \le 10]
$$

Lost indicator:
$$
\operatorname{Lost}_i(m,b) = \mathbb{I}[r_i^+(b) \le 10] \cdot \mathbb{I}[r_i^+(m) > 10]
$$

Aggregate counts:
$$
R(m,b) = \sum_i \operatorname{Rescue}_i(m,b), \quad
L(m,b) = \sum_i \operatorname{Lost}_i(m,b), \quad
G(m,b) = R(m,b) - L(m,b)
$$

Bootstrap CI (query-level resampling, $B = 5000$ replicates):
$$
\hat{\theta}^*_b = \frac{1}{N}\sum_i h_i^{(K)}(\theta; \mathcal{Q}^*_b), \quad
\text{CI}_{95\%} = [\hat{\theta}^*_{(0.025)}, \hat{\theta}^*_{(0.975)}]
$$

Delta bootstrap for paired comparison:
$$
\hat{\Delta}^*_b = \hat{\theta}^*_b(m) - \hat{\theta}^*_b(b), \quad
\text{CI}_{95\%}(\Delta) = [\hat{\Delta}^*_{(0.025)}, \hat{\Delta}^*_{(0.975)}]
$$

## Figures / Tables Needed
- **Table 6**: Protocol separation (secondary blind: 13,347 queries, 150 vocab, primary claim; canonical analysis: 21,052 queries, 152 vocab, robustness; dual-mode: 150 vocab, workflow interpretation)
- May include a bootstrap schematic figure (optional)

## Dependencies
- Metrics (Top@K, MRR) defined in 3.1 -- do NOT re-define, cross-reference
- Rescue/lost definitions used by 3.5 ablation results and Section 4.3
- Protocol separation boundaries are referenced in Section 4 for result presentation
- Leakage controls are implemented in 3.4.4 (cat_templates) -- cross-reference

## What Needs Updating from Existing Draft

### Issues (full draft lines 502-603, unified draft lines 323-381)
1. **Metric re-definition**: Both drafts re-define Top@K and MRR in 3.6.1 even though they were defined in 3.1. **Fix**: 3.6.1 should briefly restate (1-2 sentences) and cross-reference 3.1 for formal definitions. Focus instead on *how* the metrics are computed in practice and the choice of evaluation thresholds.
2. **Rescue/lost duplication**: This is the key issue. Currently rescue/lost formulas appear in BOTH 3.5 and 3.6.3. **Fix**: Define rescue/lost methodology ONLY in 3.6.3 with the full equations. In 3.5, remove the equations and add a sentence: "As defined in Section 3.6.3." The unified draft handles this better (3.5 already cross-references 3.6.3 for the table), but the full draft has equations in both.
3. **Protocol separation table**: Table 6 is well-defined. Ensure the dual-mode row notes it references 3.7.
4. **Leakage control (3.6.5)**: The unified draft has cleaner 4-bullet structure (feature-level, categorical alignment, overlap, label-free). The full draft expands each. Keep the unified draft's structure with the full draft's detail per bullet.
5. **Bootstrap detail**: The query-level resampling detail (not per-prediction) is important and well-explained. Keep it.
6. **Delta bootstrap**: The paired/delta bootstrap procedure is well-explained. Ensure the variance-reduction rationale is clear.

### Final Output Requirements
- ~800-900 words
- One table (protocol separation)
- Cross-reference 3.1 for metrics, 3.5 for ablation results
- No results -- pure methodology

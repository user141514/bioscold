# 3.3 Base Ranking Methods -- Spec

## Coverage
Describes five baseline ranking methods spanning distinct inductive biases: (1) Attachment-Frequency ranker, (2) Dual Encoder (DE) ranker, (3) Histogram Gradient-Boosted (HGB) ranker, (4) Borda fusion, (5) Score Blend (MLP + HGB). Includes Oracle ceiling discussion.

## Key Equations / Notation

**Attachment-Frequency:**
$$
s_{\text{att}}(c, \sigma) = \hat{p}_{\text{att}}(c \mid \sigma) = \frac{n_{\text{train}}(c, \sigma)}{\sum_{c'\in\mathcal{V}_{\text{train}}} n_{\text{train}}(c', \sigma)}
$$

**Dual Encoder:**
Query encoder + candidate encoder:
$$
\mathbf{h}_i^q = g_\phi(x(f_i^{\text{old}}), a(\sigma_i)), \quad
\mathbf{h}_c^c = h_\psi(x(c))
$$

Score (cosine similarity):
$$
s_{\text{DE}}(i, c) = \frac{\langle \mathbf{h}_i^q, \mathbf{h}_c^c \rangle}{\|\mathbf{h}_i^q\|_2 \|\mathbf{h}_c^c\|_2}
$$

Contrastive loss:
$$
\mathcal{L}_{\text{DE}} = \sum_i \sum_{c^+ \in \mathcal{P}_i} \sum_{c^- \in \mathcal{N}_i} [\delta - s_{\text{DE}}(i, c^+) + s_{\text{DE}}(i, c^-)]_+
$$

**HGB Ranker:** Feature vector $\mathbf{x}_{q,c}$ with 4 groups (frequency, attachment signature, molecular descriptors, fingerprint similarity). Uses HistGradientBoostingClassifier.

**Borda Fusion:**
$$
S_{\text{Borda}}(i, c) = \sum_{m \in \{\text{DE}, \text{HGB}\}} (|\mathcal{C}_i| + 1 - \rho_m(i, c))
$$

**Score Blend:**
$$
s_{\text{blend}}(i, c) = \lambda \cdot z_i\{s_{\text{MLP}}\}(c) + (1-\lambda) \cdot z_i\{s_{\text{HGB-refit}}\}(c), \quad \lambda = 0.95
$$

where $z_i(\cdot)$ is within-query z-score normalization.

**Oracle ceiling:** Fraction of queries where any base ranker has best_rank $\le 10$.

## Figures / Tables Needed
- **Table 2**: Blind Top-10 accuracy of all five methods + Oracle ceiling (0.6019, 0.8055, 0.7437, 0.8384, 0.8402, 0.8558, Oracle=0.8686)

## Dependencies
- Requires 3.1 notation ($\mathcal{C}_i$, $\mathcal{P}_i$, $\rho_\theta$, Top@K)
- Requires 3.2 split definitions (secondary blind set)
- Provides input features for 3.4 Candidate-Level Scorer
- HGB ranker is source of Conservative Mode in 3.7
- DE and Borda are sources of Exploration Mode in 3.7

## What Needs Updating from Existing Draft

### Issues (full draft lines 180-304, unified draft lines 95-187)
1. **Dual numbering**: The full draft and unified draft have nearly identical content but different table numbering. The unified draft has LaTeX formatting. Decide which to use as base (recommend: unified draft for LaTeX tables, full draft for equation formatting).
2. **Oracle ceiling placement**: Discussed before individual methods. This is good framing -- keep it.
3. **Attachment-Frequency**: The full draft uses $\widehat{p}_{\text{att}}(c\mid\sigma)$ with explicit frequency notation; unified uses $P(c\mid\sigma)$. Prefer the full draft's $\widehat{p}$ to distinguish empirical estimate from true probability.
4. **DE architecture details**: Unified draft specifies $d=128$, $K=20$ negatives, margin $\Delta=0.3$, two-layer MLP with ReLU. Full draft has Morgan FP + attachment embedding. Align: include specific architectural choices as they matter for reproducibility.
5. **HGB features**: Unified draft lists 4 feature groups specifically. Full draft is more general. Keep the unified draft's specific enumeration.
6. **Borda equation**: Full draft uses $|\mathcal{C}_i| + 1 - \rho_m(i,c)$. Unified draft uses $|V|+1$. These differ if $|V| \neq |\mathcal{C}_i|$. Verify: the candidate set size is $|\mathcal{C}_i|$, not vocabulary size $|V|$. Use $|\mathcal{C}_i|$.
7. **Score Blend lambda**: Both use $\lambda=0.95$. The full draft has the z-score normalization equation; unified draft references it. Keep the explicit z-score equation.
8. **Performance numbers**: The top-10 values belong in the Methods table but some prose describes them as "results." This is acceptable for base rankers as they characterize the methods. For the scorer's main result, defer to Section 4.
9. **Summary paragraph**: The gap analysis (Oracle - best base ranker = ~3%) that motivates the scorer is excellent. Keep it.

### Final Output Requirements
- ~1200-1500 words (longest section -- 5 methods to cover)
- Performance table with all 6 rows
- Clear "complementarity" framing that motivates fusion

# 3.5 Ablation: The Prior_Ranks Removal -- Spec

## Coverage
Leave-one-family-out ablation study revealing that 5 prior_ranks features actively harm generalization. Documents the ablation methodology, the shortcut learning mechanism (query-relative rank features fail to transfer across splits), rescue/lost analysis comparing 82-feature vs 77-feature models, and the final model configuration.

## Key Equations / Notation

Prior-rank feature definition (for each prior score $u_j$):
$$
r_j^{\text{prior}}(i,c) = 1 + \sum_{c'\in\mathcal{C}_i} \mathbb{I}[u_j(i,c') > u_j(i,c) \lor (u_j(i,c') = u_j(i,c) \land \tau_i(c') < \tau_i(c))]
$$

Feature sets:
$$
\mathcal{A}_{82} = \text{full feature index set}, \quad
\mathcal{R}_{\text{prior}} = \{r_{\text{backoff}}, r_{\text{cont}}, r_{\text{tP4}}, r_{\text{P1}}, r_{\text{P3}}\}
$$

Final 77-feature projection:
$$
\Phi_{ic}^{(77)} = \Pi_{\mathcal{A}_{82} \setminus \mathcal{R}_{\text{prior}}}(\Phi_{ic}^{(82)})
$$

Shortcut learning mechanism (conceptual, from Geirhos et al., 2020): rank features are query-relative; val set has 8 old_fragments, blind has 19 disjoint ones; rank distributions differ systematically.

## Figures / Tables Needed
- **Table 4**: Leave-one-family-out ablation results (Full/82: 0.8851, drop prior_ranks: 0.9243, drop prior_positives: 0.9037, drop query_stats: 0.9019, drop model_ranks: 0.8697, drop model_scores: 0.8610)
- **Table 5**: Rescue/lost analysis (82-feat: rescued=987, lost=596; 77-feat: rescued=424, lost=6)

## Dependencies
- Requires 3.4's feature set definition and model architecture
- Rescue/lost definitions ($\operatorname{Rescue}_i$, $\operatorname{Lost}_i$, $\operatorname{Net}$) are *defined* in 3.6.3 (evaluation protocol) but must be *referenced* here. Cross-reference: "As defined in Section 3.6.3, a rescue occurs when..."
- Final model (77-feature) output is used in Section 4 Results

## What Needs Updating from Existing Draft

### Issues (full draft lines 399-499, unified draft lines 258-319)
1. **Rescue/lost definition location problem**: Both the full draft and unified draft put rescue/lost *formulas* in both 3.5 (ablation) and 3.6.3 (evaluation). This is duplication. **Fix**: Define rescue/lost methodology once in 3.6.3 as part of the evaluation protocol. In 3.5, reference the definition and only show the *results* (Table 5).
2. **Shortcut learning explanation**: The Geirhos et al. 2020 reference is excellent. The mechanism (8 vs 19 fragments, rank distributions differ) is clearly explained. Keep this section verbatim -- it is one of the strongest analytical contributions.
3. **Ablation table**: The full draft Table 4 lists 5 comparison conditions. The unified draft matches. Add delta columns for clarity.
4. **Rescue/lost table**: The unified draft Table 5 has "Delta" row; the full draft has a cleaner format. Use the full draft format.
5. **Final model statement**: The unified draft ends with "secondary blind Top-10 = 0.9243 [0.9198, 0.9289]" and delta = "+0.0686 [+0.0633, +0.0739]". This is the primary result of the paper. Decide: does this stat go in 3.5 (as "final model specification") or in Section 4 (as "main blind performance")? **Recommendation**: State the final model's performance briefly in 3.5 for completeness, but reserve the detailed comparison table for Section 4 (Results).
6. **Per-fragment robustness**: The full draft mentions "all 19 fragments perform at or above baseline." This is an important sanity check. Keep it.
7. **Design rule**: "Per-query rank features are a generalization hazard; use raw scores, rates, and intrinsic properties instead." This should be stated prominently.

### Final Output Requirements
- ~800-1000 words
- Two tables (ablation + rescue/lost)
- Clear shortcut-learning narrative

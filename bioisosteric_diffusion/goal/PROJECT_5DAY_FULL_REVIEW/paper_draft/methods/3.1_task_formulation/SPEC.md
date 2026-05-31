# 3.1 Task Formulation -- Spec

## Coverage
Defines the closed-vocabulary, multi-positive learning-to-rank problem: query = (old fragment, attachment signature), candidate set = attachment-compatible subset of global vocabulary, task = rank candidates so true replacements appear at top.

## Key Equations / Notation

Query definition:
$$
q_i = (f_i^{\text{old}}, \sigma_i), \qquad \mathcal{Q} = \{q_i\}_{i=1}^N
$$

Vocabulary and candidate set:
$$
\mathcal{V}_{\text{train}}, \qquad
\mathcal{C}_i = \{c \in \mathcal{V}_{\text{train}} : \chi(c, \sigma_i) = 1\}
$$

Scoring function:
$$
s_\theta : (q_i, c) \mapsto s_\theta(i, c) \in \mathbb{R}
$$

Rank with tie-breaking:
$$
\rho_\theta(i,c) = 1 + \sum_{c'\in\mathcal{C}_i} \mathbb{I}[s_\theta(i,c') > s_\theta(i,c) \lor (s_\theta(i,c') = s_\theta(i,c) \land \tau_i(c') < \tau_i(c))]
$$

Positive set (structure-derived):
$$
\mathcal{P}_i = \{c \in \mathcal{C}_i : (f_i^{\text{old}}, \sigma_i, c) \in \mathcal{M}\}
$$

Binary label:
$$
y_{ic} = \mathbb{I}[c \in \mathcal{P}_i]
$$

Best positive rank:
$$
r_i^+(\theta) = \min_{c \in \mathcal{P}_i} \rho_\theta(i, c)
$$

Hit@K:
$$
h_i^{(K)}(\theta) = \mathbb{I}[r_i^+(\theta) \le K]
$$

Top@K and MRR:
$$
\operatorname{Top@K}(\theta; \mathcal{Q}) = \frac{1}{N}\sum_i h_i^{(K)}(\theta), \qquad
\operatorname{MRR}(\theta; \mathcal{Q}) = \frac{1}{N}\sum_i \frac{1}{r_i^+(\theta)}
$$

## Figures / Tables Needed
- None (text + equations only)
- May include a small schematic of the query-to-ranking pipeline (optional)

## Dependencies
- **No dependencies on other Methods subsections** (this is the foundation)
- Must establish notation used by ALL subsequent sections

## What Needs Updating from Existing Draft

### Issues in Full Draft (lines 43-132)
1. **Notation**: Full draft uses $q_i = (f_i^{\mathrm{old}}, \sigma_i)$ which is good. The unified draft uses $q = (f_{\text{old}}, \sigma)$ (no subscript). Standardize on subscript $i$ form for clarity in multi-query contexts.
2. **Tie-breaking**: The full draft has a detailed tie-breaking definition ($\tau_i(c)$). The unified draft omits it. Keep the tie-breaking detail -- it matters for reproducibility.
3. **Multi-positive convention**: Both drafts mention it briefly. Explicitly state: "For multi-positive queries, a hit is scored if any positive fragment appears within top K."
4. **Candidate compatibility function**: Full draft uses $\chi(c, \sigma_i) \in \{0,1\}$, unified draft uses $\text{compatible}(c, \sigma)$. Standardize on $\chi$ for mathematical formality.
5. **Structure-derived caveat**: Must include the explicit statement: "These labels are structure-derived: they indicate observed structural substitution, not activity preservation." This is present in both drafts -- keep it.

### Final Output Requirements
- Clean formal notation with all symbols defined
- ~500-700 words
- References: Hussain2010, Griffen2011 for MMP extraction context

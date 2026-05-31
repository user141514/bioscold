### 3.1 Task Formulation

We formalize scaffold-conditioned fragment replacement as a closed-vocabulary, multi-positive learning-to-rank problem. Let $\mathcal{Q} = \{q_i\}_{i=1}^{N}$ denote a query set. Each query pairs an old fragment with its attachment context,

$$
q_i = \bigl(f_i^{\mathrm{old}}, \sigma_i\bigr),
$$

where $f_i^{\mathrm{old}}$ is the fragment to be replaced and $\sigma_i$ is the attachment signature --- the atom and bond context through which the fragment is connected to the scaffold. The replacement vocabulary is constructed exclusively from the training split and denoted $\mathcal{V}_{\mathrm{train}}$. For query $q_i$, only candidates that are attachment-compatible with $\sigma_i$ are eligible:

$$
\mathcal{C}_i
=
\bigl\{c\in\mathcal{V}_{\mathrm{train}}:\chi(c,\sigma_i)=1\bigr\},
$$

where $\chi(c,\sigma_i)\in\{0,1\}$ is a deterministic attachment-compatibility predicate encoding valence and bond-order constraints. A scoring model with parameters $\theta$ assigns each candidate a real-valued score

$$
s_{\theta} : (q_i,c) \mapsto s_{\theta}(i,c)\in\mathbb{R},\qquad c\in\mathcal{C}_i.
$$

Candidates are ranked in decreasing order of $s_{\theta}(i,c)$. To make the ranking function well-defined even under score ties, let $\tau_i(c)$ be a fixed deterministic tie-breaking key, such as the candidate index in the frozen vocabulary. The induced rank is

$$
\rho_{\theta}(i,c)
=
1+
\sum_{c'\in\mathcal{C}_i}
\mathbb{I}\!\bigl[
 s_{\theta}(i,c')>s_{\theta}(i,c)
 \;\lor\;
 \bigl(s_{\theta}(i,c')=s_{\theta}(i,c)\land \tau_i(c')<\tau_i(c)\bigr)
\bigr].
$$

**Supervision labels.** Let $\mathcal{M}$ denote the set of structure-derived replacement triples obtained from matched molecular pair (MMP) extraction after the appropriate train/evaluation split has been fixed (construction details in Section 3.2). For query $q_i$, the positive set is

$$
\mathcal{P}_i
=
\bigl\{c\in\mathcal{C}_i:
(f_i^{\mathrm{old}},\sigma_i,c)\in\mathcal{M}\bigr\}.
$$

The corresponding candidate-level binary label is

$$
y_{ic} = \mathbb{I}\!\bigl[c\in\mathcal{P}_i\bigr], \qquad c\in\mathcal{C}_i.
$$

Queries can be multi-positive because more than one replacement fragment can be observed for the same old fragment and attachment context. These labels are **structure-derived**: they indicate observed structural substitution, not activity preservation. This is a limitation inherent to structure-based fragment replacement benchmarks and motivates the computational review proxy introduced in Section 3.7.

**Evaluation metrics.** For a scoring function $s_\theta$, the best positive rank for query $q_i$ is

$$
r_i^+(\theta)=\min_{c\in\mathcal{P}_i}\rho_\theta(i,c).
$$

The per-query hit at cutoff $K$ is

$$
h_i^{(K)}(\theta)=\mathbb{I}\!\bigl[r_i^+(\theta)\le K\bigr].
$$

For multi-positive queries, a hit is scored if any positive fragment appears within the top $K$, following the standard information retrieval convention of not penalising models for retrieving any of several valid replacements. The query-level Top-$K$ accuracy is

$$
\operatorname{Top@K}(\theta;\mathcal{Q})
=
\frac{1}{N}\sum_{i=1}^{N}h_i^{(K)}(\theta),
$$

and mean reciprocal rank is

$$
\operatorname{MRR}(\theta;\mathcal{Q})
=
\frac{1}{N}\sum_{i=1}^{N}\frac{1}{r_i^+(\theta)}.
$$

Top-10 is the primary metric; Top-1, Top-5, Top-20, Top-50, and MRR are reported as secondary diagnostics. All confidence intervals are computed by nonparametric bootstrap resampling over 5,000 replicates at the query level, so that all candidates belonging to a sampled query are retained together. This query-level evaluation mirrors the practical constraint that a medicinal chemist examines only the top handful of candidate suggestions.

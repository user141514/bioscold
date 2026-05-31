## 3.3 Base Ranking Methods

We implement five baseline ranking methods that represent distinct approaches to the fragment replacement problem, providing both comparative baselines and the input features for our candidate-level scorer (Section 3.4). These methods span a deliberate spectrum of complexity and inductive bias: from empirical frequency counting, through learned embedding similarity, to feature-based classification and parameter-free rank fusion. Each captures a fundamentally different signal---global replacement statistics, structural compatibility in a learned space, interpretable molecular properties, or their consensus---and the complementarity among them is itself a central finding that the Borda fusion (Section 3.3.4) and the downstream scorer both exploit.

All methods are evaluated on the secondary blind split (13,347 held-out queries) using query-level Top-10 accuracy: a candidate is considered correct if the ground-truth replacement fragment is ranked among the top 10 for its query. Confidence intervals are computed by nonparametric bootstrapping with 5,000 resamples over queries. Table 2 summarizes the blind Top-10 performance of each method.

\begin{table}[t]
\centering
\caption{Secondary blind Top-10 accuracy of the five base ranking methods, with 95\% bootstrap confidence intervals. The Oracle ceiling represents the fraction of queries where any base ranker places the correct candidate in its top 10.}
\label{tab:base_ranker_performance}
\small
\begin{tabular}{lcc}
\toprule
Method & Blind Top-10 & 95\% CI \\
\midrule
Attachment-Frequency & 0.6019 & [0.5933, 0.6104] \\
Dual Encoder (DE) & 0.8055 & [0.7986, 0.8122] \\
HGB & 0.7437 & [0.7356, 0.7516] \\
Borda(DE, HGB) & 0.8384 & [0.8321, 0.8447] \\
MLP (rank-only) & 0.8402 & [0.8339, 0.8466] \\
\addlinespace
Oracle ceiling & 0.8686 & --- \\
\bottomrule
\end{tabular}
\end{table}

**Oracle ceiling.** Before describing each ranker individually, it is useful to establish the best possible performance achievable from the base ranker set. The Oracle ceiling (0.8686) represents the fraction of secondary blind queries for which *at least one* base ranker places the correct fragment in its top 10. This is a theoretical upper bound on any downstream method that operates solely on the base rankers' outputs: even a perfect selector cannot recover a target that no base ranker surfaces. The gap between the Oracle and the best individual ranker (0.8686 vs.\ 0.8402) indicates that approximately 3\% of queries are individually recoverable by different rankers---queries where, for example, the Dual Encoder succeeds and the MLP fails, or vice versa---and quantifies the opportunity for a fusion or scoring method that can learn to select the right ranker's output on a per-query basis.

### 3.3.1 Attachment-Frequency Ranker

The Attachment-Frequency ranker estimates the empirical training-set prior for candidate $c$ under attachment signature $\sigma$. Let $n_{\mathrm{train}}(c,\sigma)$ be the number of training replacement pairs in which $c$ appears under $\sigma$. The score is

$$
s_{\mathrm{att}}(c,\sigma)
=
\widehat{p}_{\mathrm{att}}(c\mid\sigma)
=
\frac{n_{\mathrm{train}}(c,\sigma)}
       {\sum_{c'\in\mathcal{V}_{\mathrm{train}}} n_{\mathrm{train}}(c',\sigma)},
\label{eq:attach_freq}
$$

and candidates are ranked by descending $s_{\mathrm{att}}(c,\sigma)$. The method has no learned parameters and uses only training-set counts.

This approach embodies a straightforward but powerful empirical prior: fragments that have frequently appeared as replacements at a given attachment environment are more likely to be valid replacements again. The method captures the *global* replacement tendency of each fragment under each attachment context, information that is immediately useful but inherently coarse: it cannot distinguish between two candidates with identical frequency counts that differ in chemical compatibility with the specific query fragment. The Attachment-Frequency ranker therefore serves as the natural lower bound for all subsequent methods. On the secondary blind split, it achieves a Top-10 accuracy of 0.6019 (95\% CI: 0.5933--0.6104).

### 3.3.2 Dual Encoder Ranker

The Dual Encoder (DE) ranker replaces the frequency-based heuristic with a learned measure of structural compatibility. Its design follows the standard dual-encoder architecture for similarity learning: a query encoder and a candidate encoder that project their respective inputs into a shared embedding space, where compatibility is scored by cosine similarity.

The query encoder takes two inputs: the Morgan fingerprint (ECFP, radius 2, 2048 bits) of the original fragment $f_i^{\mathrm{old}}$ being replaced, and a learned embedding of the attachment signature $\sigma_i$. These are concatenated and passed through a two-layer MLP with hidden dimensionality $d=128$ and ReLU activation to produce a query embedding. The candidate encoder, independently, maps the candidate fragment's Morgan fingerprint through a separate two-layer MLP (same architecture, not weight-tied) to a candidate embedding. Writing $x(\cdot)$ for the Morgan fingerprint and $a(\sigma_i)$ for the attachment embedding,

$$
\mathbf{h}^{q}_{i}
=
g_{\phi}\!\left(x(f_i^{\mathrm{old}}),a(\sigma_i)\right),
\qquad
\mathbf{h}^{c}_{c}
=
h_{\psi}\!\left(x(c)\right).
\label{eq:de_encoders}
$$

The DE score is the cosine similarity between query and candidate embeddings:

$$
s_{\mathrm{DE}}(i,c)
=
\frac{\langle \mathbf{h}^{q}_{i},\mathbf{h}^{c}_{c}\rangle}
     {\lVert\mathbf{h}^{q}_{i}\rVert_2\,\lVert\mathbf{h}^{c}_{c}\rVert_2}.
\label{eq:de_score}
$$

For training, positives are paired with attachment-compatible negatives $\mathcal{N}_i\subset\mathcal{C}_i\setminus\mathcal{P}_i$, sampled uniformly from the candidate vocabulary. For each positive pair, we sample $K=20$ negatives that are attachment-compatible but not the observed replacement. The model is optimized with a margin-based ranking loss:

$$
\mathcal{L}_{\mathrm{DE}}
=
\sum_{i=1}^{N}
\sum_{c^{+}\in\mathcal{P}_i}
\sum_{c^{-}\in\mathcal{N}_i}
\left[\delta - s_{\mathrm{DE}}(i,c^{+}) + s_{\mathrm{DE}}(i,c^{-})\right]_{+},
\label{eq:de_loss}
$$

where $[t]_+=\max(t,0)$ and $\delta=0.3$ is the ranking margin. The objective encourages the model to assign higher similarity to the true replacement than to any sampled negative, learning an embedding space where structurally compatible query-candidate pairs cluster together.

The Dual Encoder provides a fundamentally different signal from the frequency-based approach. While $s_{\mathrm{att}}(c,\sigma)$ captures global empirical trends---``this candidate is common at this attachment''---the DE captures whether a *specific* candidate's molecular structure suits a *specific* query's chemical context. A candidate with low global frequency may still be the correct replacement if its substructure is complementary to the query fragment. This divergence in signal is reflected quantitatively: DE achieves a Top-10 accuracy of 0.8055 (95\% CI: 0.7986--0.8122) on the secondary blind split, a gain of +0.2036 over the frequency baseline, confirming that learned structural compatibility captures substantial signal beyond empirical frequency.

### 3.3.3 Histogram Gradient-Boosted Ranker (HGB)

The Histogram Gradient-Boosted (HGB) Ranker approaches fragment replacement as a supervised classification problem with explicit, interpretable molecular features. For each query-candidate pair $(q_i, c)$, we compute a feature vector $\mathbf{x}_{q_i,c}$ organized into four groups: (1) *frequency features:* $\widehat{p}_{\mathrm{att}}(c\mid\sigma)$, global replacement frequency $\text{count}(c)$, and the attachment frequency $\text{count}(c, \sigma)$; (2) *attachment signature features:* the bond order at the attachment point, whether the bond belongs to a ring system, the scaffold atom type at the attachment, and the attachment signature itself (one-hot encoded); (3) *molecular property descriptors:* heavy atom count, molecular weight, logP, TPSA, ring count, hydrogen bond donor and acceptor counts, and the number of rotatable bonds---computed for both the candidate and query fragments; and (4) *fingerprint similarity:* the Tanimoto coefficient between the Morgan fingerprints (radius 2, 2048 bits) of the original and candidate fragments. The HGB model is trained using scikit-learn's `HistGradientBoostingClassifier` with 200 boosting iterations, a maximum depth of 6, and a learning rate of 0.1. Candidates are ranked by the model's predicted probability of being a valid replacement.

The HGB ranker occupies a distinct methodological position in our framework. Unlike the Dual Encoder, whose latent embeddings trade interpretability for representational flexibility, the HGB model operates directly on physicochemical descriptors whose meanings are transparent to a medicinal chemist. This interpretability makes it the natural source for the Conservative Mode in our dual-mode workflow (Section 3.7), where chemically conservative proposals---those grounded in observable, nameable properties rather than opaque learned features---are preferred for high-confidence applications. On the secondary blind split, HGB achieves a Top-10 accuracy of 0.7437 (95\% CI: 0.7356--0.7516), establishing a strong feature-based baseline that, while lower than DE, captures a complementary signal.

### 3.3.4 Borda Fusion and Score Blending

**Borda fusion.** The Borda fusion method combines the DE and HGB rankers through parameter-free rank aggregation, testing whether their individual errors are independent enough that their consensus outperforms either alone. For query $q_i$ with candidate set $\mathcal{C}_i$, let $\rho_m(i,c)$ denote the rank of candidate $c$ assigned by method $m\in\{\mathrm{DE},\mathrm{HGB}\}$, with smaller ranks indicating higher-ranked candidates. The Borda score is

$$
S_{\mathrm{Borda}}(i,c)
= \sum_{m\in\{\mathrm{DE},\mathrm{HGB}\}}
\left(|\mathcal{C}_i| + 1 - \rho_m(i,c)\right),
\label{eq:borda}
$$

and candidates are re-ranked by descending $S_{\mathrm{Borda}}$.

The parameter-free design of Borda fusion is deliberate and methodologically motivated by our transform-heldout evaluation protocol (Section 3.2). Under a transform-heldout split, no query in the validation or secondary blind sets shares its $(f_i^{\mathrm{old}}, \sigma_i)$ combination with any training query. This means that any fusion weights optimized on a validation set would necessarily be tuned to ranking patterns that may not generalize to queries with unseen transform combinations---there is no guarantee that the optimal weight for DE versus HGB on validation queries is optimal for secondary blind queries. Borda fusion sidesteps this entirely by using equal, fixed weights that require no tuning, ensuring consistent behavior across all splits. This conservative choice is appropriate for a baseline method: if simple equal-weight fusion already demonstrates substantial gains, then the complementarity signal is robust and not an artifact of weight optimization.

The Borda results confirm this. On the secondary blind split, Borda fusion achieves a Top-10 accuracy of 0.8384 (95\% CI: 0.8321--0.8447), representing gains of +0.0329 over DE alone and +0.0947 over HGB alone. The improvement over DE is particularly notable: DE is the stronger individual ranker, yet fusing it in equal proportion with the weaker HGB still produces a net gain, confirming that the two methods make substantially different errors. The Borda consensus captures the approximately 3\% of queries where DE fails but HGB succeeds, approaching the Oracle ceiling without any learned component.

**Score Blend (MLP + HGB).** The Score Blend extends the fusion concept by introducing a lightweight learned aggregator: a rank-only MLP that takes as input three per-query rank values for each candidate---the ranks assigned by the DE, HGB, and Attachment-Frequency rankers---and produces a single score $s_{\mathrm{MLP}}(q_i, c)$. This MLP is a two-layer network with hidden dimensionality 32 and ReLU activations, trained as a binary classifier on the training split. It learns to weight the three base rankers' signals differently depending on the query context, adapting the fusion strategy to each query rather than applying the fixed Borda formula.

The final Score Blend combines the MLP score with a separately refit HGB score after within-query standardization. For any score vector $u(i,c)$ over $c\in\mathcal{C}_i$, define

$$
z_i\!\left(u(i,c)\right)
= \frac{u(i,c)-\mu_i(u)}{\sigma_i(u)+\varepsilon},
\qquad
\mu_i(u)=\frac{1}{|\mathcal{C}_i|}\sum_{c'\in\mathcal{C}_i}u(i,c'),
\label{eq:zscore}
$$

with $\sigma_i(u)$ the corresponding within-query standard deviation and $\varepsilon$ a small numerical stabilizer. The blended score is

$$
s_{\mathrm{blend}}(i,c)
= \lambda\, z_i\{s_{\mathrm{MLP}}\}(c)
+ (1-\lambda)\, z_i\{s_{\mathrm{HGB-refit}}\}(c),
\qquad \lambda=0.95,
\label{eq:score_blend}
$$

where $s_{\mathrm{HGB-refit}}$ is the predicted probability from an HGB model refit on the training data with the MLP output included as an additional feature. The heavy coefficient on the MLP term ($\lambda=0.95$) reflects that the rank-only aggregator captures the vast majority of predictive signal; the marginal HGB-refit contribution provides calibration smoothing rather than substantial new information.

On the secondary blind split, the rank-only MLP achieves a Top-10 accuracy of 0.8402 (95\% CI: 0.8339--0.8466). While this is a marginal improvement over Borda (+0.0018, not statistically significant at the Top-10 level), the MLP provides a statistically significant improvement in Mean Reciprocal Rank, indicating that its benefit is in elevating the correct candidate closer to the top of the ranked list within queries where it already appears in the top 10. The full Score Blend achieves 0.8558, establishing the strongest baseline from base ranker signals alone.

### 3.3.5 Summary: The Gap Between Base Rankers and the Oracle

Across all five methods, the best-performing individual ranker (MLP, 0.8402) and the strongest combination (Score Blend, 0.8558) fall short of the Oracle ceiling (0.8686). The residual gap---approximately 1.3 to 2.8 percentage points---represents queries where, even after rank fusion and learned aggregation, the base ranker signals are insufficient to surface the correct candidate. Closing this gap requires moving beyond rank-level and frequency-level features to a richer representation that incorporates fine-grained chemical constraints: the properties of the fragment-scaffold interface, the steric and electronic compatibility of the replacement, and the structural relationships among candidates within a query. This motivates the candidate-level scorer in Section 3.4, which augments the base ranker signals with 77 chemically motivated features and a dense gradient-boosted model capable of discriminating among candidates that the base rankers cannot separate.

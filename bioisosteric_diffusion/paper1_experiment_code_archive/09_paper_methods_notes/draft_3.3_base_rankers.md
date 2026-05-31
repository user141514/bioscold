## 3.3 Base Ranking Methods

We implement five baseline ranking methods that represent distinct approaches to the fragment replacement problem, providing both comparative baselines and the input features for our candidate-level scorer (Section 3.4). These methods span a deliberate spectrum of complexity and inductive bias: from empirical frequency counting, through learned embedding similarity, to feature-based classification and parameter-free rank fusion. Each captures a fundamentally different signal---global replacement statistics, structural compatibility in a learned space, interpretable molecular properties, or their consensus---and the complementarity among them is itself a central finding that the Borda fusion (Section 3.3.4) and the downstream scorer both exploit.

All methods are evaluated on the secondary blind split (13,347 held-out queries) using query-level Top-10 accuracy: a candidate is considered correct if the ground-truth replacement fragment is ranked among the top 10 for its query. Confidence intervals are computed by nonparametric bootstrapping with 5,000 resamples over queries. Table 2 summarizes the blind Top-10 performance of each method.

\begin{table}[t]
\centering
\caption{Blind Top-10 accuracy of the five base ranking methods, with 95\% bootstrap confidence intervals. The Oracle ceiling represents the fraction of queries where any base ranker places the correct candidate in its top 10.}
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

\vspace{0.5em}

\noindent\textbf{Oracle ceiling.} Before describing each ranker individually, it is useful to establish the best possible performance achievable from the base ranker set. The Oracle ceiling (0.8686) represents the fraction of blind queries for which \emph{at least one} base ranker places the correct fragment in its top 10. This is a theoretical upper bound on any downstream method that operates solely on the base rankers' outputs: even a perfect selector cannot recover a target that no base ranker surfaces. The gap between the Oracle and the best individual ranker (0.8686 vs.\ 0.8402) indicates that approximately 3\% of queries are individually recoverable by different rankers---queries where, for example, the Dual Encoder succeeds and the MLP fails, or vice versa---and quantifies the opportunity for a fusion or scoring method that can learn to select the right ranker's output on a per-query basis.

\subsection*{3.3.1 Attachment-Frequency Ranker}

The Attachment-Frequency ranker estimates the conditional probability of observing a candidate fragment $c$ given an attachment signature $\sigma$ in the training set:

\begin{equation}
P(c \mid \sigma) = \frac{\text{count}(c, \sigma)}{\text{count}(\sigma)},
\label{eq:attach_freq}
\end{equation}

\noindent where $\text{count}(c, \sigma)$ is the number of training matched molecular pairs in which fragment $c$ appears with attachment signature $\sigma$, and $\text{count}(\sigma)$ is the total number of training pairs bearing that signature. Candidates are ranked by descending $P(c \mid \sigma)$, and the method requires no learned parameters and no molecular featurization---only the training pair counts.

This approach embodies a straightforward but powerful empirical prior: fragments that have frequently appeared as replacements at a given attachment environment are more likely to be valid replacements again. The method captures the \emph{global} replacement tendency of each fragment under each attachment context, information that is immediately useful but inherently coarse: it cannot distinguish between two candidates with identical frequency counts that differ in chemical compatibility with the specific query fragment. The Attachment-Frequency ranker therefore serves as the natural lower bound for all subsequent methods. On the blind split, it achieves a Top-10 accuracy of 0.6019 (95\% CI: 0.5933--0.6104).

\subsection*{3.3.2 Dual Encoder Ranker}

The Dual Encoder (DE) ranker replaces the frequency-based heuristic with a learned measure of structural compatibility. Its design follows the standard dual-encoder architecture for similarity learning: a query encoder and a candidate encoder that project their respective inputs into a shared embedding space, where compatibility is scored by cosine similarity.

The query encoder takes two inputs: the Morgan fingerprint (ECFP, radius 2, 2048 bits) of the original fragment being replaced, and a learned embedding of the attachment signature $\sigma$. These are concatenated and passed through a two-layer MLP with hidden dimensionality $d=128$ and ReLU activation to produce a query embedding $\mathbf{e}_q \in \mathbb{R}^d$. The candidate encoder, independently, maps the candidate fragment's Morgan fingerprint through a separate two-layer MLP (same architecture, not weight-tied) to a candidate embedding $\mathbf{e}_c \in \mathbb{R}^d$. The replacement score is then the cosine similarity:

\begin{equation}
s_{\text{DE}}(q, c) = \cos(\mathbf{e}_q, \mathbf{e}_c) = \frac{\mathbf{e}_q \cdot \mathbf{e}_c}{\|\mathbf{e}_q\| \, \|\mathbf{e}_c\|}.
\label{eq:de_score}
\end{equation}

Training uses a contrastive objective over query-candidate pairs derived from the training MMP data. For each positive pair $(q, c^+)$---a query and its observed replacement---we sample $K=20$ negative candidates $(q, c^-)$ uniformly from the candidate vocabulary, ensuring the negatives are attachment-compatible but not the observed replacement. The model is optimized with a margin-based ranking loss:

\begin{equation}
\mathcal{L}_{\text{DE}} = \sum_{(q, c^+)} \sum_{k=1}^{K} \max\bigl(0, \; s_{\text{DE}}(q, c^-_k) - s_{\text{DE}}(q, c^+) + \Delta\bigr),
\label{eq:de_loss}
\end{equation}

\noindent with margin $\Delta = 0.3$. The objective encourages the model to assign higher similarity to the true replacement than to any sampled negative, learning an embedding space where structurally compatible query-candidate pairs cluster together.

The Dual Encoder provides a fundamentally different signal from the frequency-based approach. While $P(c \mid \sigma)$ captures global empirical trends---``this candidate is common at this attachment''---the DE captures whether a \emph{specific} candidate's molecular structure suits a \emph{specific} query's chemical context. A candidate with low global frequency may still be the correct replacement if its substructure is complementary to the query fragment. This divergence in signal is reflected quantitatively: DE achieves a Top-10 accuracy of 0.8055 (95\% CI: 0.7986--0.8122) on the blind split, a gain of +0.2036 over the frequency baseline, confirming that learned structural compatibility captures substantial signal beyond empirical frequency.

\subsection*{3.3.3 Histogram Gradient-Boosted Ranker}

The Histogram Gradient-Boosted (HGB) Ranker approaches fragment replacement as a supervised classification problem with explicit, interpretable molecular features. For each query-candidate pair $(q, c)$, we compute a feature vector $\mathbf{x}_{q,c}$ organized into four groups: (1) \emph{frequency features:} $P(c \mid \sigma)$, global replacement frequency $\text{count}(c)$, and the attachment frequency $\text{count}(c, \sigma)$; (2) \emph{attachment signature features:} the bond order at the attachment point, whether the bond belongs to a ring system, the scaffold atom type at the attachment, and the attachment signature itself (one-hot encoded); (3) \emph{molecular property descriptors:} heavy atom count, molecular weight, logP, TPSA, ring count, hydrogen bond donor and acceptor counts, and the number of rotatable bonds---computed for both the candidate and query fragments; and (4) \emph{fingerprint similarity:} the Tanimoto coefficient between the Morgan fingerprints (radius 2, 2048 bits) of the original and candidate fragments. The HGB model is trained using scikit-learn's \texttt{HistGradientBoostingClassifier} with 200 boosting iterations, a maximum depth of 6, and a learning rate of 0.1. Candidates are ranked by the model's predicted probability of being a valid replacement.

The HGB ranker occupies a distinct methodological position in our framework. Unlike the Dual Encoder, whose latent embeddings trade interpretability for representational flexibility, the HGB model operates directly on physicochemical descriptors whose meanings are transparent to a medicinal chemist. This makes it the natural source for the Conservative Mode in our dual-mode workflow (Section 3.6), where chemically conservative proposals---those grounded in observable, nameable properties rather than opaque learned features---are preferred for high-confidence applications. On the blind split, HGB achieves a Top-10 accuracy of 0.7437 (95\% CI: 0.7356--0.7516), establishing a strong feature-based baseline that, while lower than DE, captures a complementary signal.

\subsection*{3.3.4 Borda Fusion and Score Blending}

\textbf{Borda fusion.} The Borda fusion method combines the DE and HGB rankers through parameter-free rank aggregation, testing whether their individual errors are independent enough that their consensus outperforms either alone. For a query $q$ with candidate set of size $|V|$, let $\text{rank}_m(q, c)$ denote the rank assigned to candidate $c$ by method $m \in \{\text{DE}, \text{HGB}\}$. The Borda score is:

\begin{equation}
S_{\text{Borda}}(q, c) = \sum_{m \in \{\text{DE}, \text{HGB}\}} \bigl( |V| + 1 - \text{rank}_m(q, c) \bigr),
\label{eq:borda}
\end{equation}

\noindent and candidates are re-ranked by $S_{\text{Borda}}$ in descending order.

The parameter-free design of Borda fusion is deliberate and methodologically motivated by our transform-heldout evaluation protocol (Section 3.2.2). Under a transform-heldout split, no query in the validation or blind sets shares its (old\_fragment, attachment\_signature) combination with any training query. This means that any fusion weights optimized on a validation set would necessarily be tuned to ranking patterns that may not generalize to queries with unseen transform combinations---there is no guarantee that the optimal weight for DE versus HGB on validation queries is optimal for blind queries. Borda fusion sidesteps this entirely by using equal, fixed weights that require no tuning, ensuring consistent behavior across all splits. This conservative choice is appropriate for a baseline method: if simple equal-weight fusion already demonstrates substantial gains, then the complementarity signal is robust and not an artifact of weight optimization.

The Borda results confirm this. On the blind split, Borda fusion achieves a Top-10 accuracy of 0.8384 (95\% CI: 0.8321--0.8447), representing gains of +0.0329 over DE alone and +0.0947 over HGB alone. The improvement over DE is particularly notable: DE is the stronger individual ranker, yet fusing it in equal proportion with the weaker HGB still produces a net gain, confirming that the two methods make substantially different errors. The Borda consensus captures the approximately 3\% of queries where DE fails but HGB succeeds, approaching the Oracle ceiling without any learned component.

\textbf{Score Blend (MLP + HGB).} The Score Blend extends the fusion concept by introducing a lightweight learned aggregator: a rank-only MLP that takes as input three per-query rank values for each candidate---the ranks assigned by the DE, HGB, and Attachment-Frequency rankers---and produces a single score $s_{\text{MLP}}(q, c)$. This MLP is a two-layer network with hidden dimensionality 32 and ReLU activations, trained as a binary classifier on the training split. It learns to weight the three base rankers' signals differently depending on the query context, adapting the fusion strategy to each query rather than applying the fixed Borda formula.

The final Score Blend combines the MLP score with a separately refit HGB model:

\begin{equation}
s_{\text{blend}}(q, c) = 0.95 \cdot z\bigl(s_{\text{MLP}}(q, c)\bigr) + 0.05 \cdot z\bigl(s_{\text{HGB-refit}}(q, c)\bigr),
\label{eq:score_blend}
\end{equation}

\noindent where $z(\cdot)$ denotes per-query z-score normalization and $s_{\text{HGB-refit}}$ is the predicted probability from an HGB model refit on the training data with the MLP's outputs included as an additional feature. The heavy coefficient on the MLP term (0.95) reflects that the rank-only aggregator captures the vast majority of predictive signal; the marginal HGB-refit contribution provides calibration smoothing rather than substantial new information.

On the blind split, the rank-only MLP achieves a Top-10 accuracy of 0.8402 (95\% CI: 0.8339--0.8466). While this is a marginal improvement over Borda (+0.0018, not statistically significant at the Top-10 level), the MLP provides a statistically significant improvement in Mean Reciprocal Rank, indicating that its benefit is in elevating the correct candidate closer to the top of the ranked list within queries where it already appears in the top 10. The full Score Blend achieves 0.8558, establishing the pre-D4S strongest baseline and the highest performance achievable from base ranker signals alone.

\subsection*{3.3.5 Summary: The Gap Between Base Rankers and the Oracle}

Across all five methods, the best-performing individual ranker (MLP, 0.8402) and the strongest combination (Score Blend, 0.8558) fall short of the Oracle ceiling (0.8686). The residual gap---approximately 1.3 to 2.8 percentage points---represents queries where, even after rank fusion and learned aggregation, the base ranker signals are insufficient to surface the correct candidate. Closing this gap requires moving beyond rank-level and frequency-level features to a richer representation that incorporates fine-grained chemical constraints: the properties of the fragment-scaffold interface, the steric and electronic compatibility of the replacement, and the structural relationships among candidates within a query. This motivates the candidate-level scorer in Section 3.4, which augments the base ranker signals with 77 chemically motivated features and a dense gradient-boosted model capable of discriminating among candidates that the base rankers cannot separate.

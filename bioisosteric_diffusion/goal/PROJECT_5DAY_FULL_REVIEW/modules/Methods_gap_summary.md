# Module 10 — Methods Gap Analysis

## Starting Problem

The S3TEXT journal-facing manuscript (the base for S4GEN and S6R) has a Methods section that was written for the D4A1-era pipeline. It describes the Borda-based fusion with DE and HGB, but does NOT reflect the current S6R/D4S31-level algorithm (validation-selected MLP+HGB score blend, prior_rank removal, feature schema alignment). The task.md Section 6 defines 14 specific topics that Methods must cover, and this analysis compares current Methods content against that standard.

**Source manuscripts compared:**
- Current Methods: `paper_s3text_full_manuscript_journal_facing.md` Sections 3.1-3.6 (also carried forward to S4GEN and S6R drafts)
- Required content standard: `task/task.md` Section 6 ("Methods 蓝图必须回答" checklist)

---

## Detailed Gap Analysis

### Gap 1: Candidate Matrix Structure (NOT COVERED)

**Current Methods says**: "For each query, the candidate set C_q is the attachment-compatible subset of that train replacement vocabulary." (§3.1) Mentions "four repairs" in §3.2 but does NOT describe the matrix structure.

**Required**: Describe the candidate matrix: each row is a (query_id, candidate_SMILES, score, label) tuple. Rows are grouped by query. For each query-candidate pair, there are feature vectors (frequency features, Morgan similarity, 2D deltas, attachment features). The matrix is materialized as sharded JSONL files.

**Severity**: HIGH. Without this, reviewers cannot understand what the models actually score.

### Gap 2: Label Construction (INCOMPLETE)

**Current Methods says**: "Each query is associated with a positive set P_q of replacement fragments derived from structure-based replacement pairs extracted through matched molecular pair analysis on ChEMBL." (§3.1) Mentions decoy-to-positive ratio repairs in §3.2.

**Required**: Describe:
- Labels are binary per query-candidate pair (1 = positive from MMP, 0 = all other candidates).
- Multi-positive queries: a query can have 2+ known positive replacements. During evaluation, Top10 = 1 if ANY positive is in top 10.
- Decoy selection: 1:1 decoy-to-positive ratio, removal of wrong-positive overlaps, removal of duplicate decoys.
- The label does NOT imply activity preservation (already stated).

**Severity**: HIGH. Multi-positive handling is entirely absent from Methods.

### Gap 3: Closed Vocabulary Construction (INCOMPLETE)

**Current Methods says**: "We construct a closed global replacement vocabulary V from the training split." (§3.1) Mentions "152-fragment" in §3.5 analysis context.

**Required**: Describe:
- How the 152 training replacement fragments are selected (deduplication, frequency filtering, SMILES canonicalization).
- Why closed vocabulary: isolates ranking from generation.
- All evaluation uses only these 152 (or 150 for blind) fragments.
- Attachment-compatibility filtering: candidates not compatible with the query's attachment signature are excluded before scoring.

**Severity**: MEDIUM. The concept is described but lacks engineering detail needed for reproducibility.

### Gap 4: Base Ranker Score/Rank Generation (INCOMPLETE)

**Current Methods says**: Describes DE (embedding, cosine similarity), HGB (7 features, fragment-level), attachment-frequency baseline. Describes Borda score as `sum(|V| + 1 - rank)`.

**Required**: Describe:
- How each base ranker produces a score per query-candidate pair.
- How scores are converted to ranks within each query (descending score = rank 1).
- How ties are broken.
- The exact 7 features used by HGB: log_gf, log_af, morgan_tanimoto, heavy_atom_delta, cand_ha, old_ha, attachment_match.
- The 5 prior_rank features that were LATER REMOVED (backoff_logit_rank, cont_prior_rank, t_p4_logit_rank, p1_logit_rank, p3_logit_rank).

**Severity**: HIGH. The feature set and ranking mechanics are underspecified.

### Gap 5: D4S31 Final Scorer Feature Family (NOT COVERED)

**Current Methods says**: Only describes the D4A1-era HGB with 7 features.

**Required**: Describe the D4S31 final scorer:
- **Score features**: M5 MLP rank score (row_zscore).
- **Score blend**: `0.95 * z(M5 mlp rank score) + 0.05 * z(HGB refit score)`.
- **Validation selection**: blend weights selected on validation split, evaluated once on blind.
- **Feature families used** in the final HGB refit: ranks, scores, candidate frequencies, attachment frequencies, Morgan similarity, 2D property deltas (MW, LogP, TPSA, HBD, HBA, RotB).
- **Feature families NOT used** (because not in aligned matrix): ACCFG, electronic density, A4C features, functional group counts.
- **Score column standardization**: per-query row z-score normalization.

**Severity**: CRITICAL. The current Methods describes a D4A1-era model, not the actual best method.

### Gap 6: Why prior_ranks Were Removed (NOT COVERED)

**Current Methods says**: Nothing about prior_rank removal. (This was only added to S4GEN Discussion as a complementary finding.)

**Required**: Describe:
- Five prior_rank features encoded query-local ordering from frequency statistics.
- They severely impaired blind-set transfer despite improving validation performance.
- Removing them reduced hit-loss from 596 queries (D4S28R) to 6 queries (D4S31).
- Blind Top10 improved from 0.885 (D4S28R) to 0.924 (D4S31) after removal.
- This is a cross-split generalization design principle: features encoding query-local order from training statistics do not transfer to disjoint fragment distributions.

**Severity**: CRITICAL. This is the single most important algorithmic improvement and is completely absent from Methods.

### Gap 7: HistGB Training Procedure (INCOMPLETE)

**Current Methods says**: "HGB is a feature-based histogram gradient-boosted ranker trained on fragment-level frequency features." No parameters.

**Required**: Describe:
- max_depth=5, max_iter=100, learning_rate=0.1, class_weight=balanced.
- Validation early stopping (not used for final selection).
- Query-level ranking (HGB as pointwise ranker, not listwise).
- Training on all training shards (247 shards, 100,784 queries).
- The refit model used in the score blend vs the original D4A1 model.

**Severity**: MEDIUM. Reproducibility requires these details.

### Gap 8: Per-Query Sorting Procedure (NOT COVERED)

**Current Methods says**: Metrics are "averaged at the query level" but does not describe the sorting step.

**Required**: Describe:
- After scoring, candidates within each query are sorted descending by score.
- Top K = first K candidates in sorted order.
- If score tie, the tie-breaking rule (typically by candidate frequency or arbitrary).
- Primary metric: query-level Top10 = proportion of queries where at least one positive appears in top 10.
- Secondary: Top1/5/20/50, MRR.

**Severity**: MEDIUM. Sorting is implicit but should be explicit.

### Gap 9: Feature Schema Alignment (NOT COVERED)

**Current Methods says**: Nothing about feature alignment.

**Required**: Describe:
- Train, validation, and blind splits must have identical feature schemas (same columns, same dtypes).
- D4S28 bug: feature alignment failure caused blind T10=0.712 (incorrectly interpreted as generalization failure).
- Fix: ensure all feature columns present in train are present in val/blind; columns absent from any split are dropped during training.
- prior_rank features (which rely on training-set statistics) cannot be computed for blind where the fragment vocabulary differs, causing alignment errors.
- The fix to remove prior_ranks simultaneously solved the alignment bug AND improved generalization.

**Severity**: CRITICAL. The alignment bug story is essential for understanding the evolution and for reviewer trust.

### Gap 10: Rescue/Lost Definition (NOT COVERED)

**Current Methods says**: Nothing about rescue/lost.

**Required**: Define:
- **Rescue**: a query that was a miss (no Top10 hit) under a baseline method becomes a hit under the new method.
- **Lost**: a query that was a hit under a baseline method becomes a miss under the new method.
- **Net gain** = rescues - losses.
- **Gain/loss ratio** = rescues / losses (a key stability metric used throughout D4S series).
- Hit-loss count is reported for each algorithmic change (D4S28R lost=596, D4S31 lost=6).

**Severity**: HIGH. Rescue/lost is central to the algorithm optimization narrative but absent from Methods.

### Gap 11: Bootstrap CI Procedure (COVERED)

**Current Methods says**: "Statistical uncertainty is estimated by nonparametric bootstrap resampling over query ID with 5,000 replicates to form 95% confidence intervals [Efron1979]." (§3.4)

**Required**: Same. Already covered.

**Severity**: NONE. Properly described.

### Gap 12: Leakage Control (COVERED)

**Current Methods says**: "Transform-heldout split in which no (old_fragment, attachment_signature) transform appears in both train and test." Zero overlap verification. Separate secondary blind split. 13,347 blind queries, 150-fragment blind vocabulary. (§3.2)

**Required**: Same. Already covered.

**Severity**: NONE. Properly described.

---

## Summary of Gaps by Severity

| # | Gap | Current Status | Severity |
|---|-----|----------------|----------|
| 1 | Candidate matrix structure | Not described | HIGH |
| 2 | Label construction (multi-positive handling) | Incomplete | HIGH |
| 3 | Closed vocabulary construction | Incomplete | MEDIUM |
| 4 | Base ranker score/rank generation details | Incomplete | HIGH |
| 5 | D4S31 final scorer feature family | **Not described** | **CRITICAL** |
| 6 | Why prior_ranks were removed | **Not described** | **CRITICAL** |
| 7 | HistGB training hyperparameters | Incomplete | MEDIUM |
| 8 | Per-query sorting procedure | Not described | MEDIUM |
| 9 | Feature schema alignment (D4S28 bug story) | **Not described** | **CRITICAL** |
| 10 | Rescue/lost definitions | Not described | HIGH |
| 11 | Bootstrap CI procedure | Covered | NONE |
| 12 | Leakage control | Covered | NONE |

---

## Key Files

| File | Path |
|------|------|
| S3TEXT Journal-Facing Manuscript (Methods Section 3.1-3.6) | `plan_results/routeA_paper_s3text_journal_facing/paper_s3text_full_manuscript_journal_facing.md` |
| Task Definition (Required Methods Content Checklist) | `task/task.md` (Section 6) |
| S4GEN Polished Manuscript (Updated Methods) | `plan_results/routeA_paper_s4gen_general_journal_polish/paper_s4gen_polished_manuscript.md` |
| S6R Journal-Facing Manuscript (Methods) | `plan_results/routeA_paper_s6r_text_journal_facing/paper_s6r_text_journal_facing_main_manuscript.md` |
| Algorithm Optimization Synthesis (D4S28R/D4S31 details) | `plan_results/routeA_algorithm_optimization_synthesis_20260528.md` |
| Algorithm Optimization Current Verdict | `plan_results/ROUTEA_ALGORITHM_OPTIMIZATION_CURRENT_VERDICT.md` |

---

## Verdict

**The current Methods section describes a D4A1-era pipeline (HGB with 7 features, Borda fusion) that no longer reflects the actual best method. Three critical gaps — the D4S31 final scorer feature family, the prior_rank removal rationale, and the feature schema alignment story — are completely absent. The rescue/lost evaluation framework and multi-positive handling are also missing. Only bootstrap CI and leakage control are adequately covered. A major Methods rewrite is required before the paper can accurately represent the current algorithm.**

---

## Retracted/Negated Old Conclusions

- **The S4GEN Methods (derived from S3TEXT) was assumed adequate** because the manuscript reads coherently. However, it describes an outdated pipeline (Borda-as-main-method, D4A1-era HGB). It does not describe the score blend, prior_rank removal, or feature alignment.
- **The Borda-centric narrative was correct at the time of S3TEXT** but is now superseded by the S6R score-blend-centric narrative. The Methods section needs restructuring to reflect the new method hierarchy.
- **The feature-engineering finding (5 prior_rank features) was added only to Discussion** in S4GEN, but it MUST be in Methods because it is a core model design decision.

---

## Still-Credible Final Conclusions

1. **Methods must be completely rewritten** for the S6R-level algorithm (score blend, not Borda, as primary method).
2. **Three critical gaps**: D4S31 feature family, prior_rank removal rationale, feature schema alignment.
3. **Three high-severity gaps**: candidate matrix structure, label construction (multi-positive), base ranker details, rescue/lost definitions.
4. **Three medium-severity gaps**: closed vocabulary construction, HistGB hyperparameters, per-query sorting.
5. **Two topics are already correct**: bootstrap CI, leakage control.
6. **The Methods rewrite is the single most important pre-submission task** — without it, the paper describes the wrong method.

---

## Impact on Paper

- **Methods must be rewritten from scratch** for the S6R narrative. The current S3TEXT-based Methods describes a superseded algorithm.
- **The rewrite must add 10 new content areas** (the gaps above) and restructure the narrative around the score blend, not Borda.
- **Borda stays in Methods** as a fixed baseline definition (Section 3.3 or equivalent), not as the primary fusion method.
- **The prior_rank removal story** must be in Methods (as a model design decision, not only in Discussion).
- **The D4S28 feature alignment bug** should be described (in Methods or an appendix) to establish credibility about the cross-split generalization challenge.
- **Rescue/lost analysis** should be defined in Methods because it is the primary evaluation framework used throughout the D4S algorithm optimization.
- **Multi-positive queries** must be explicitly handled in the evaluation description: Top10 = 1 if ANY positive is in the top 10.
- **The candidate matrix** (sharded JSONL, row = query-candidate pair) must be described for reproducibility.
- **Once Methods is rewritten**, the Results section also needs updating to reference the correct method names and evaluation framework.

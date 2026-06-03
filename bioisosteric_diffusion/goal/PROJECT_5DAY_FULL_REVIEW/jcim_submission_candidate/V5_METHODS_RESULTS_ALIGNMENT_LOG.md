# V5 Methods/Results Alignment Log

## Purpose

Aligned the Methods, protocol tables, candidate-matrix definitions, and diagnostic sections with the Fresh Blind2 evidence hierarchy already established in the revised Abstract and Results.

## Multi-agent review

Ruflo swarm remained unavailable in this session:

- `mcp__ruflo__.agent_list` failed with `Transport closed`.

Because Ruflo was unavailable, three `multi_agent_v1` explorer reviewers were spawned:

- Reviewer A: Methods/protocol/candidate-matrix consistency.
- Reviewer B: Results/Discussion/Conclusion claim-boundary consistency.
- Reviewer C: stale reference and line-level conflict scan.

All substantive reviewer findings were applied.

## Fixes applied

### Candidate universe and task formulation

- Replaced the hard attachment-compatible candidate definition with a protocol-indexed full closed-vocabulary definition:
  `C_i = V_train^(p)`.
- Clarified that Fresh Blind2 uses the full Train2-derived 161-fragment vocabulary.
- Clarified that the original secondary-blind diagnostic matrix uses the earlier 150-fragment vocabulary.
- Moved attachment-compatible candidate subsets into sensitivity-analysis language rather than the primary task definition.

### Candidate-matrix construction

- Rewrote candidate construction to distinguish:
  - original fixed-150 secondary-blind diagnostic matrix,
  - Fresh Blind2 full-161 primary prospective matrix,
  - top-150 Blind2 comparability sensitivity.
- Added Table M1b for Fresh Blind2 full-161 construction:
  - Train2: 93,083 queries, 161 candidates/query.
  - Dev2/calibration: 27,415 queries, 161 candidates/query.
  - Blind2: 17,058 queries, 161 candidates/query.
  - Blind2 positive-set size sum: 28,330.
  - Blind2 zero in-matrix-positive queries: 0.
  - Random expected Top-10: 0.0987.
- Added metric guard that metric-bearing evaluations require at least one in-matrix positive per query.

### Protocol hierarchy

- Replaced stale Table M4 language that made secondary blind the primary performance protocol.
- Table M4 now labels:
  - Original secondary blind: diagnostic history / post-audit evidence / stress-test context.
  - Fresh Blind2: primary prospective evaluation.
  - A4C/activity diagnostics: boundary and workflow diagnostics only.
- Replaced old “Primary claims cite only secondary blind results” wording with Fresh Blind2 primary prospective wording.

### Prior-ranks language

- Renamed Section 3.5 to `Prior_Ranks as a Post-Audit Instability Diagnostic`.
- Replaced “Why Prior_Ranks Degrade Generalization” with “Why Prior_Ranks Were Suspected to be Unstable”.
- Replaced stronger “non-transferable shortcut” wording with “potentially unstable query-local frequency shortcut”.
- Clarified that removal of prior_ranks defines HistGB77; the prior_ranks features themselves define the 82-feature augmentation.

### A4C downweighting

- Compressed A4C Methods into a short supplementary workflow diagnostics section.
- Removed the main-text dual-mode workflow, provenance table, and alert-rate definitions from Methods.
- Stated that A4C annotations are not used for model fitting, feature selection, scorer selection, ranking claims, or medicinal-chemistry validation.

### Results consistency

- Added grouped uncertainty directly to Table 2:
  - query-level CI,
  - transform-key CI,
  - old-fragment CI.
- Reframed DE/HGB overlap as a secondary-blind diagnostic, correcting the old mismatch between 13,347-query counts and Fresh Blind2.
- Renamed the LambdaRank section to a future architecture path rather than a stronger-main-method claim.
- Merged activity-comparable and calibration sanity into one boundary-diagnostics section.

## Final conflict scan

No remaining hits for:

- `0.9243`
- `probability` / `probabilities`
- `valid replacement`
- `activity preservation`
- `wet-lab`
- `expert review`
- `locked`
- `final method`
- `fully prospective`
- `improves transfer`
- `removing prior_ranks improves`
- `Primary claims cite only secondary`
- `Secondary blind | 13,347 | **Primary`
- `Section 4.6`
- `Section 4.9`
- `non-transferable query-local`
- `Results are reported in Section 4.4`
- `All 77 features are computable`

The only remaining `primary performance` phrase is a negative boundary statement that canonical analysis does not support primary performance claims.

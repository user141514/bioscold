# BindingDB Hard-Negative 10k Diagnostic v3, Audited

Date: 2026-06-10

Purpose: test whether a frequency-free transferable-content prototype survives candidate-heldout and core-heldout diagnostics before any full 10-seed run.

This version supersedes the first v3 summary. Two evaluator issues were fixed before re-running:

- FullBlend alpha tuning now matches the outer split. Candidate-heldout uses inner candidate-heldout; core-heldout uses inner core-heldout.
- Hit@K now uses a stable query/candidate hash tie-breaker, so all-equal scores cannot inherit row-order bias.
- The evaluator now reports Hit@1, Hit@5, Hit@10, MRR, NDCG@10, random Hit@10 expectation, random-normalized Hit@10, and Hit@10 enrichment over random.

## Protocol Audit

| Audit | Status |
|---|---|
| Outer/inner tuning match | Passed. `fullblend_tuning` is `candidate_smiles` for candidate protocols and `core_key` for core-heldout. |
| Weak candidate-heldout | Passed. Test positive candidates have zero overlap with train rows, but other test candidates may appear in train rows. |
| Strict candidate-heldout | Added. Test candidate IDs have zero overlap with train rows. |
| CTCR identity | Constrained. CTCR is a prototype ensemble over the same no-frequency content features, not yet a distinct new algorithm. |
| Content distribution audit | Added. Positive/negative feature, CA score, CTCR score, and no-frequency LBC coefficient audits are written under `audit_10k_hard_*`. |
| Candidate-heldout metrics | Added. Strict candidate-heldout is interpreted by random-normalized Hit@10 and enrichment, not raw Hit@10 alone. |

## CTCR Prototype Boundary

CTCR currently differs from LBC no-freq only in score construction:

- LBC no-freq: logistic-regression probability on `morgan`, `bit_corr`, `dHeavy`, `dRings`, `dMW`, `dLogP`, and `dTPSA`.
- CTCR: query-normalized average of the no-frequency logistic decision function and a ridge learned-content score trained on the same seven features.

It does not yet introduce a different feature family, training target, hard-negative objective, or split-specific training protocol. Therefore CTCR cannot be called a new algorithm unless a later diagnostic creates separation from LBC no-freq and FullBlend.

## Candidate-Heldout, Weak, 3 Seeds

Definition: held-out positive `candidate_smiles` are absent from train rows; test negative candidates may still occur in train rows.

Split audit:

- mean evaluated queries: 8,261.3
- mean candidates/query: 68.447
- test positive candidate overlap with train rows: 0 in all seeds
- test any candidate overlap with train rows: 2,728-2,760 values
- FullBlend alpha: 0.0 in all seeds, tuned by inner candidate-heldout

| Method | Query Hit@10 | OF-macro Hit@10 |
|---|---:|---:|
| Frequency | 0.119057 | 0.092039 |
| CA | 0.501216 | 0.610010 |
| Blend | 0.491569 | 0.602908 |
| FullBlend | 0.786734 | 0.748852 |
| LBC | 0.749606 | 0.716048 |
| LBC no-freq | 0.780854 | 0.739795 |
| CTCR prototype | 0.783858 | 0.744864 |

Query metric audit:

| Method | Hit@1 | Hit@5 | Hit@10 | MRR | NDCG@10 | RN-Hit@10 | Enrich@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frequency | 0.000080 | 0.013844 | 0.119057 | 0.055470 | 0.032503 | -0.758472 | 0.238386 |
| CA | 0.153986 | 0.400856 | 0.501216 | 0.279319 | 0.199743 | 0.004281 | 1.004416 |
| FullBlend | 0.475865 | 0.685147 | 0.786734 | 0.577491 | 0.472132 | 0.574263 | 1.576614 |
| LBC | 0.349230 | 0.621835 | 0.749606 | 0.480664 | 0.402233 | 0.500021 | 1.502333 |
| LBC no-freq | 0.490490 | 0.687697 | 0.780854 | 0.586187 | 0.474505 | 0.562509 | 1.564849 |
| CTCR prototype | 0.482954 | 0.686178 | 0.783858 | 0.581812 | 0.473173 | 0.568472 | 1.570903 |

Random Hit@10 expectation is 0.499042. RN-Hit@10 means `(Hit@10 - random expectation) / (1 - random expectation)`.

Interpretation: frequency support collapses for positives, but FullBlend no longer collapses after fair alpha tuning because it selects pure learned content (`alpha=0`). CTCR is only marginally above LBC no-freq and below/near FullBlend, so this does not support a CTCR superiority claim.

## Candidate-Heldout, Strict, 3 Seeds

Definition: every `candidate_smiles` appearing in test rows is absent from train rows.

Split audit:

- mean evaluated queries: 8,261.3
- mean candidates/query: 23.345
- test positive candidate overlap with train rows: 0 in all seeds
- test any candidate overlap with train rows: 0 in all seeds
- score frequency nonzero rate: 0.000000
- FullBlend alpha: 0.0 in all seeds, tuned by inner candidate-heldout

| Method | Query Hit@10 | OF-macro Hit@10 |
|---|---:|---:|
| Frequency | 0.918332 | 0.900547 |
| CA | 0.828516 | 0.838617 |
| Blend | 0.828516 | 0.838617 |
| FullBlend | 0.952706 | 0.919754 |
| LBC | 0.954735 | 0.921478 |
| LBC no-freq | 0.950048 | 0.913454 |
| CTCR prototype | 0.951902 | 0.917221 |

Query metric audit:

| Method | Hit@1 | Hit@5 | Hit@10 | MRR | NDCG@10 | RN-Hit@10 | Enrich@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frequency | 0.182333 | 0.680444 | 0.918332 | 0.391865 | 0.393095 | -0.006389 | 0.999428 |
| CA | 0.290752 | 0.611850 | 0.828516 | 0.444484 | 0.394322 | -1.113307 | 0.901681 |
| FullBlend | 0.593115 | 0.856757 | 0.952706 | 0.706661 | 0.636861 | 0.417492 | 1.036837 |
| LBC | 0.592583 | 0.862542 | 0.954735 | 0.707525 | 0.638613 | 0.442393 | 1.039046 |
| LBC no-freq | 0.598124 | 0.851640 | 0.950048 | 0.707267 | 0.634391 | 0.384652 | 1.033945 |
| CTCR prototype | 0.596135 | 0.854378 | 0.951902 | 0.707221 | 0.636030 | 0.407510 | 1.035962 |

Random Hit@10 expectation is 0.918857.

Interpretation: strict candidate identity leakage is removed, but the strict partition leaves a smaller/easier test candidate space. Frequency has no nonzero frequency signal; its high raw Hit@10 is exactly random-level (`RN-Hit@10=-0.006389`, enrichment `0.999428`) under K=10 with only 23.3 candidates/query and 4.0 positives/query on average. This split is useful as a leakage audit, not as the main hard ranking benchmark.

## Core-Heldout, 3 Seeds

Definition: held-out positive `core_key` values are absent from train rows.

Split audit:

- mean evaluated queries: 3,478.7
- mean candidates/query: 64.479
- test positive core overlap with train rows: 0 in all seeds
- test any core overlap with train rows: 0 in all seeds
- FullBlend alpha: 0.25 in all seeds, tuned by inner core-heldout

| Method | Query Hit@10 | OF-macro Hit@10 |
|---|---:|---:|
| Frequency | 0.789533 | 0.717224 |
| CA | 0.656251 | 0.759141 |
| Blend | 0.739180 | 0.810379 |
| FullBlend | 0.950536 | 0.952091 |
| LBC | 0.938129 | 0.927235 |
| LBC no-freq | 0.940052 | 0.950753 |
| CTCR prototype | 0.941375 | 0.952178 |

Query metric audit:

| Method | Hit@1 | Hit@5 | Hit@10 | MRR | NDCG@10 | RN-Hit@10 | Enrich@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frequency | 0.355421 | 0.644433 | 0.789533 | 0.489189 | 0.384496 | 0.026919 | 1.007115 |
| CA | 0.264459 | 0.560458 | 0.656251 | 0.409583 | 0.269807 | -0.590928 | 0.837160 |
| FullBlend | 0.516602 | 0.832778 | 0.950536 | 0.652026 | 0.575285 | 0.771165 | 1.212577 |
| LBC | 0.531505 | 0.829093 | 0.938129 | 0.660254 | 0.564604 | 0.713241 | 1.196786 |
| LBC no-freq | 0.487748 | 0.806302 | 0.940052 | 0.625338 | 0.557299 | 0.722802 | 1.199191 |
| CTCR prototype | 0.485303 | 0.806638 | 0.941375 | 0.624278 | 0.558006 | 0.728980 | 1.200873 |

Random Hit@10 expectation is 0.783904.

Interpretation: FullBlend, LBC no-freq, and CTCR are effectively tied under core-heldout. The evidence supports transferable learned content, not CTCR-specific novelty.

## Content Distribution Mechanism

Candidate-heldout positive/negative mean contrasts:

| Metric | Positive mean | Negative mean | Positive - negative |
|---|---:|---:|---:|
| Morgan | 0.303182 | 0.284738 | 0.018444 |
| bit_corr | 0.442585 | 0.424901 | 0.017684 |
| dHeavy | 2.597557 | 1.148542 | 1.449016 |
| dRings | 0.365182 | 0.116878 | 0.248304 |
| dMW | 0.327371 | 0.176660 | 0.150712 |
| dLogP | 0.376093 | 0.339076 | 0.037017 |
| dTPSA | 1.436646 | 1.966685 | -0.530038 |
| CA score | 0.307489 | 0.344549 | -0.037060 |
| CTCR score | 0.878957 | -0.054273 | 0.933229 |

CA is weak because its coarse physchem penalty assigns higher average score to negatives than positives. In this BindingDB-hard matrix, positives are slightly more similar by Morgan/bit_corr but also have larger heavy-atom, ring, MW, and logP deltas. A fixed "smaller delta is better" CA rule is therefore miscalibrated.

No-frequency LBC learns the opposite weighting pattern under candidate-heldout:

| Feature | Standardized coefficient mean |
|---|---:|
| Morgan | 1.446547 |
| bit_corr | -1.852376 |
| dHeavy | 0.383780 |
| dRings | 0.294576 |
| dMW | 0.118704 |
| dLogP | -0.043260 |
| dTPSA | -0.068531 |

The mechanism is learned multivariate content reweighting, not simply adding frequency. `bit_corr` is useful as a conditional correction against Morgan, while learned physchem weights reverse CA's coarse penalty for the deltas that are enriched in positives.

## Decision

Do not run the full huge benchmark as a CTCR superiority run yet.

The honest claim after audit is:

> transferable learned content dominates hand-crafted CA and raw candidate frequency under BindingDB-hard candidate-heldout; CTCR is currently a calibrated no-frequency content prototype that does not yet separate from LBC no-freq or FullBlend.

This closes the LBC external-diagnostic question and opens a separate paper route:

- task: candidate-heldout fragment replacement ranking,
- problem: transferable content calibration after candidate support is cut,
- locked baselines: CA, FullBlend with candidate-heldout tuning (`alpha=0` in this diagnostic), full-content scalar, and LBC no-freq,
- algorithm gate: a new method must beat LBC no-freq and FullBlend by a meaningful margin under weak candidate-heldout, then survive strict candidate-heldout and core-heldout.

Next algorithmic work should create a real distinction from LBC no-freq, for example by adding context-conditioned content weighting, counterfactual hard-negative calibration, or candidate-support gating that can beat FullBlend under weak candidate-heldout without relying on candidate frequency.

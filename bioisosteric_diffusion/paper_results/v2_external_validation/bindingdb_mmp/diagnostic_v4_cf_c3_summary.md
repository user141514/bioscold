# BindingDB Hard-Negative 10k Diagnostic v4: CF-C3

Date: 2026-06-10

Purpose: test whether the first candidate-heldout content-calibration algorithm can separate from the strongest learned no-frequency baselines before any full run.

## Method

CF-C3 is a no-frequency ranker trained on counterfactual CA-matched candidate sets with context-calibrated content interactions.

It uses:

- content features: `morgan`, `bit_corr`, `dHeavy`, `dRings`, `dMW`, `dLogP`, `dTPSA`
- CA-bucket interactions from training-set CA score quantiles
- context interactions with `attachment_signature`, `endpoint`, and collapsed `target_key`
- CA-matched BCE training: all positives plus the five nearest-CA negatives per positive within the same query
- matched inner tuning of `C` over `{0.03, 0.1, 0.3, 1.0}` by OF-macro MRR

It does not use `candidate_smiles`, candidate frequency, or `query_id` as model features.

## Gate

Primary gate on weak candidate-heldout:

- CF-C3 must beat the strongest of FullBlend, fullcontent, and LBC no-freq by at least +0.03 OF-macro MRR or +0.03 OF-macro NDCG@10.
- CF-C3 must not reduce query Hit@1 versus the strongest learned baseline.

Result: **passed**.

| Metric | Strongest learned baseline | CF-C3 | Delta |
|---|---:|---:|---:|
| OF-macro MRR | 0.566657 | 0.661811 | +0.095154 |
| OF-macro NDCG@10 | 0.413571 | 0.530930 | +0.117359 |
| Query Hit@1 | 0.490490 | 0.605433 | +0.114943 |

## Weak Candidate-Heldout, 3 Seeds

Mean random Hit@10 expectation: 0.499042.

| Method | OF-macro MRR | OF-macro NDCG@10 | Query Hit@1 | Query MRR | Query NDCG@10 | Query RN-Hit@10 |
|---|---:|---:|---:|---:|---:|---:|
| CA | 0.360323 | 0.254157 | 0.153986 | 0.279319 | 0.199743 | 0.004281 |
| fullcontent | 0.558526 | 0.413209 | 0.475865 | 0.577491 | 0.472132 | 0.574263 |
| FullBlend | 0.558526 | 0.413209 | 0.475865 | 0.577491 | 0.472132 | 0.574263 |
| LBC no-freq | 0.566657 | 0.413571 | 0.490490 | 0.586187 | 0.474505 | 0.562509 |
| CTCR prototype | 0.563469 | 0.414004 | 0.482954 | 0.581812 | 0.473173 | 0.568472 |
| CF-C3 | 0.661811 | 0.530930 | 0.605433 | 0.691846 | 0.589694 | 0.761541 |

CF-C3 selected `C` values by seed: 0.30, 1.00, 0.03. Selected CA-matched training rows: 25,581; 27,350; 26,718.

## Strict Candidate-Heldout, 3 Seeds

Mean random Hit@10 expectation: 0.918857. This split is a leakage/random audit, not the primary hard-ranking benchmark.

| Method | OF-macro MRR | OF-macro NDCG@10 | Query Hit@1 | Query MRR | Query NDCG@10 | Query RN-Hit@10 |
|---|---:|---:|---:|---:|---:|---:|
| CA | 0.529355 | 0.425244 | 0.290752 | 0.444484 | 0.394322 | -1.113307 |
| fullcontent | 0.685294 | 0.568786 | 0.593115 | 0.706661 | 0.636861 | 0.417492 |
| FullBlend | 0.685294 | 0.568786 | 0.593115 | 0.706661 | 0.636861 | 0.417492 |
| LBC no-freq | 0.684221 | 0.563796 | 0.598124 | 0.707267 | 0.634391 | 0.384652 |
| CTCR prototype | 0.685319 | 0.566831 | 0.596135 | 0.707221 | 0.636030 | 0.407510 |
| CF-C3 | 0.778617 | 0.683864 | 0.710104 | 0.800162 | 0.737549 | 0.717022 |

CF-C3 remains above the learned no-frequency baselines even when all test candidate IDs are absent from training rows.

## Core-Heldout, 3 Seeds

Mean random Hit@10 expectation: 0.783904.

| Method | OF-macro MRR | OF-macro NDCG@10 | Query Hit@1 | Query MRR | Query NDCG@10 | Query RN-Hit@10 |
|---|---:|---:|---:|---:|---:|---:|
| CA | 0.502097 | 0.339295 | 0.264459 | 0.409583 | 0.269807 | -0.590928 |
| fullcontent | 0.648281 | 0.560920 | 0.483564 | 0.622874 | 0.558521 | 0.740955 |
| FullBlend | 0.663896 | 0.558104 | 0.516602 | 0.652026 | 0.575285 | 0.771165 |
| LBC no-freq | 0.653022 | 0.560756 | 0.487748 | 0.625338 | 0.557299 | 0.722802 |
| CTCR prototype | 0.651200 | 0.561335 | 0.485303 | 0.624278 | 0.558006 | 0.728980 |
| CF-C3 | 0.813806 | 0.691118 | 0.731167 | 0.801677 | 0.694865 | 0.755429 |

CF-C3 also separates under core-heldout, suggesting that the context-calibrated counterfactual signal is not merely memorizing candidate identity.

## Decision

CF-C3 passes the 10k v4 algorithm gate. The new-paper direction can move from a mechanism/benchmark seed to an algorithm-candidate diagnostic:

> Counterfactual context-calibrated content ranking improves candidate-heldout fragment replacement ranking beyond FullBlend, fullcontent, and LBC no-freq on the 10k BindingDB-hard diagnostic.

Next required diagnostic before a full huge run:

1. Add candidate-cluster-heldout to ensure generalization beyond exact candidate IDs.
2. Add ablations: no CA bucket, no context interactions, no CA-matched selection, no target context.
3. Re-run 10k v4 with ablations and the same matched inner tuning.

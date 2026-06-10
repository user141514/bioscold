# bindingdb_mmp_10k_hard_core_v3_tuned Score Distribution Audit

Protocol: `core_heldout`

## Split Audit

- seeds: 3
- mean train rows: 479169.3
- mean test rows: 224238.7
- mean evaluated queries: 3478.7
- mean candidates/query: 64.479
- mean positives/query: 10.108
- score_freq nonzero rate: 0.657317
- positive score_freq nonzero rate: 0.468677
- negative score_freq nonzero rate: 0.692381

## Positive vs Negative Mean Contrast

| metric | positive_mean | negative_mean | positive_minus_negative_mean | positive_p50 | negative_p50 |
| --- | --- | --- | --- | --- | --- |
| morgan | 0.310150 | 0.291095 | 0.019055 | 0.285669 | 0.268009 |
| bit_corr | 0.451695 | 0.431123 | 0.020572 | 0.443352 | 0.419037 |
| dHeavy | 2.591052 | 1.113118 | 1.477934 | 2.000000 | 0.666667 |
| dRings | 0.356100 | 0.108152 | 0.247948 | 0.000000 | 0.000000 |
| dMW | 0.317250 | 0.173918 | 0.143332 | 0.214560 | 0.105827 |
| dLogP | 0.390601 | 0.337250 | 0.053350 | 0.293092 | 0.244716 |
| dTPSA | 1.224448 | 1.811288 | -0.586840 | 0.283973 | 0.376927 |
| score_ca | 0.311913 | 0.349936 | -0.038023 | 0.289882 | 0.336732 |
| score_ctcr | 0.775176 | -0.144145 | 0.919321 | 0.583343 | -0.308984 |

## No-Frequency LBC Coefficients

| feature | standardized_coef_mean |
| --- | --- |
| morgan | 1.882628 |
| bit_corr | -2.014679 |
| dHeavy | 0.552013 |
| dRings | 0.251232 |
| dMW | 0.064838 |
| dLogP | -0.062880 |
| dTPSA | -0.073816 |

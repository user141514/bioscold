# bindingdb_mmp_10k_hard_candidate_strict_v3 Score Distribution Audit

Protocol: `candidate_strict_heldout`

## Split Audit

- seeds: 3
- mean train rows: 48737.3
- mean test rows: 192858.0
- mean evaluated queries: 8261.3
- mean candidates/query: 23.345
- mean positives/query: 3.979
- score_freq nonzero rate: 0.000000
- positive score_freq nonzero rate: 0.000000
- negative score_freq nonzero rate: 0.000000

## Positive vs Negative Mean Contrast

| metric | positive_mean | negative_mean | positive_minus_negative_mean | positive_p50 | negative_p50 |
| --- | --- | --- | --- | --- | --- |
| morgan | 0.303182 | 0.285702 | 0.017481 | 0.275903 | 0.266370 |
| bit_corr | 0.442585 | 0.425986 | 0.016598 | 0.432927 | 0.417821 |
| dHeavy | 2.597557 | 1.157540 | 1.440018 | 2.000000 | 1.000000 |
| dRings | 0.365182 | 0.118172 | 0.247010 | 0.000000 | 0.000000 |
| dMW | 0.327371 | 0.177603 | 0.149768 | 0.216590 | 0.102365 |
| dLogP | 0.376093 | 0.343857 | 0.032237 | 0.277299 | 0.247988 |
| dTPSA | 1.436646 | 1.905992 | -0.469346 | 0.323166 | 0.413924 |
| score_ca | 0.307489 | 0.344875 | -0.037386 | 0.282998 | 0.332146 |
| score_ctcr | 0.578041 | -0.118787 | 0.696828 | 0.480598 | -0.255270 |

## No-Frequency LBC Coefficients

| feature | standardized_coef_mean |
| --- | --- |
| morgan | 1.472588 |
| bit_corr | -1.881116 |
| dHeavy | 0.381572 |
| dRings | 0.292469 |
| dMW | 0.129441 |
| dLogP | -0.036033 |
| dTPSA | -0.067245 |

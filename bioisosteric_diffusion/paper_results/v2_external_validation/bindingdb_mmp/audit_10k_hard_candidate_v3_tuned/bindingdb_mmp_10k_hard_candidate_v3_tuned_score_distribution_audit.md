# bindingdb_mmp_10k_hard_candidate_v3_tuned Score Distribution Audit

Protocol: `candidate_heldout`

## Split Audit

- seeds: 3
- mean train rows: 50554.3
- mean test rows: 565395.7
- mean evaluated queries: 8261.3
- mean candidates/query: 68.447
- mean positives/query: 3.979
- score_freq nonzero rate: 0.265183
- positive score_freq nonzero rate: 0.000000
- negative score_freq nonzero rate: 0.281555

## Positive vs Negative Mean Contrast

| metric | positive_mean | negative_mean | positive_minus_negative_mean | positive_p50 | negative_p50 |
| --- | --- | --- | --- | --- | --- |
| morgan | 0.303182 | 0.284738 | 0.018444 | 0.275903 | 0.266013 |
| bit_corr | 0.442585 | 0.424901 | 0.017684 | 0.432927 | 0.416315 |
| dHeavy | 2.597557 | 1.148542 | 1.449016 | 2.000000 | 1.000000 |
| dRings | 0.365182 | 0.116878 | 0.248304 | 0.000000 | 0.000000 |
| dMW | 0.327371 | 0.176660 | 0.150712 | 0.216590 | 0.102640 |
| dLogP | 0.376093 | 0.339076 | 0.037017 | 0.277299 | 0.246267 |
| dTPSA | 1.436646 | 1.966685 | -0.530038 | 0.323166 | 0.413979 |
| score_ca | 0.307489 | 0.344549 | -0.037060 | 0.282998 | 0.332064 |
| score_ctcr | 0.878957 | -0.054273 | 0.933229 | 0.625954 | -0.184095 |

## No-Frequency LBC Coefficients

| feature | standardized_coef_mean |
| --- | --- |
| morgan | 1.446547 |
| bit_corr | -1.852376 |
| dHeavy | 0.383780 |
| dRings | 0.294576 |
| dMW | 0.118704 |
| dLogP | -0.043260 |
| dTPSA | -0.068531 |

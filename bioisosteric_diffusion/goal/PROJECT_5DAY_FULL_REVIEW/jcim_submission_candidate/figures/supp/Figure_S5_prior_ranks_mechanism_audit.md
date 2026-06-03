# Supplementary Figure S5 Audit

## Files

| File | Status |
|---|---|
| `Figure_S5_prior_ranks_mechanism_audit.svg` | PASS |
| `Figure_S5_prior_ranks_mechanism_audit.pdf` | PASS |
| `Figure_S5_prior_ranks_mechanism_audit.png` | PASS |
| `Figure_S5_prior_ranks_mechanism_caption.md` | PASS |

## Claim Boundary

- The figure is a post-audit targeted mechanism diagnostic, not prospective feature selection.
- Panel A supports the bounded claim that prior ranks are the dominant unstable prior-family component in this audit.
- Panel A does not support a claim that all prior information is harmful.
- Panel B supports a support-bin diagnostic: the 77-feature minus 82-feature gain is largest in low-support queries.
- The figure does not claim activity preservation, wet-lab validation, or universal improvement across chemical subspaces.

## Locked Values

| Quantity | Value | Source |
|---|---:|---|
| Drop prior ranks Delta | +0.0393 | `prior_ranks_targeted_ablation.csv` |
| Drop all non-rank prior families Delta | +0.0174 | `prior_ranks_targeted_ablation.csv` |
| Drop prior statistics Delta | +0.0151 | `prior_ranks_targeted_ablation.csv` |
| Drop prior scores Delta | +0.0073 | `prior_ranks_targeted_ablation.csv` |
| Drop PMI/conditional-prior contrast Delta | -0.0037 | `prior_ranks_targeted_ablation.csv` |
| Low-support Delta | +0.0488 | `prior_ranks_support_bin_audit.csv` |
| Medium-support Delta | +0.0344 | `prior_ranks_support_bin_audit.csv` |
| High-support Delta | +0.0346 | `prior_ranks_support_bin_audit.csv` |

## Verdict

PASS. The figure is suitable as Supplementary Figure S5 if cited as a diagnostic support figure rather than as a new primary model-performance figure.

# BindingDB Hard-Negative 10k Diagnostic

Date: 2026-06-09

This diagnostic was added after the first BindingDB feasibility benchmark produced near-ceiling content-aware (CA) scores, indicating that random or weakly constrained negatives were too easy.

## Hard-Negative Rule

For each positive query context, negatives are drawn preferentially from:

1. the same `target_key + endpoint + attachment_signature` but a different core,
2. then the same attachment signature,
3. then the global fragment pool as fallback.

In `ca_matched` mode, candidate negatives are ranked by the same CA proxy used by the evaluator and the most similar unsupported candidates are selected. These are stress-test negatives, not asserted inactive compounds.

## 10k Source-Row Comparison, 3 Seeds

| Matrix | Frequency query Hit@10 | CA query Hit@10 | Blend query Hit@10 | LBC query Hit@10 | LBC OF-macro Hit@10 |
|---|---:|---:|---:|---:|---:|
| Easy negatives | 0.847672 | 0.994591 | 0.998228 | 0.998341 | 0.998561 |
| CA-matched hard negatives | 0.848522 | 0.665755 | 0.749667 | 0.954230 | 0.948716 |

Mean evaluation coverage was unchanged at 3,376.0 queries and 1,009 old fragments per seed.

## Interpretation

The original easy-negative BindingDB matrix is suitable as a source and pipeline feasibility check, but not as the main external benchmark. The hard-negative variant is more diagnostic because CA no longer solves the task alone, while LBC retains a large margin over CA and blend on the same 10k source slice.

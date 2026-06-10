# Candidate-Heldout Content Calibration Paper Design

Date: 2026-06-10
Status: new-paper design seed, separate from the LBC manuscript

## Working Thesis

Candidate-heldout fragment replacement ranking is a distinct problem from OF-heldout support-conditioned ranking. Once positive candidate support is removed, hand-crafted content proxies can fail or invert under counterfactual hard negatives, while learned no-frequency content calibration restores ranking ability.

The paper should not start by claiming a named algorithm is superior. It should start from the audited anomaly:

- CA positive mean: 0.307489
- CA negative mean: 0.344549
- CTCR prototype positive mean: 0.878957
- CTCR prototype negative mean: -0.054273

This establishes the problem: similarity-like content is not enough; transferable content compatibility must be calibrated under hard candidate-heldout conditions.

## Candidate Titles

1. Learning Transferable Content Compatibility for Candidate-Heldout Fragment Replacement Ranking
2. When Similarity Fails: Counterfactual Content Calibration for Candidate-Heldout Fragment Replacement Ranking
3. Candidate-Heldout Fragment Replacement Ranking by Context-Conditioned Content Calibration

Use title 1 until an algorithm clearly separates from the strongest learned baselines. Use title 2 only if the counterfactual calibration model wins cleanly and the CA-inversion story remains central.

## Claim Boundary

Allowed claim after the current audit:

> Candidate-heldout diagnostics reveal a new fragment replacement ranking regime in which raw candidate support is unavailable, hand-crafted CA can invert, and learned no-frequency content scoring becomes necessary.

Not yet allowed:

> CTCR is a new superior algorithm.

The current CTCR prototype uses the same no-frequency feature family as LBC no-freq plus a query-normalized ridge/logistic score ensemble. It is a useful probe, not yet a distinct method.

## Failure Ledger

| Evidence | Intended hypothesis | Observed result | Failure type | Implication |
|---|---|---|---|---|
| Easy BindingDB external matrix | External benchmark validates LBC ranking | CA reached near-ceiling | Negative-design bottleneck | Easy negatives cannot test algorithmic content ranking. |
| Hard-negative 10k OF split | Hard negatives expose LBC superiority | FullBlend and no-frequency learned content remain strong | Claim bottleneck | The evidence supports learned content, not LBC-specific frequency use. |
| First v3 candidate-heldout | Frequency/FullBlend collapse proves CTCR direction | FullBlend collapse was protocol-sensitive | Split/tuning bottleneck | Inner tuning must match outer generalization axis. |
| Stable-tie strict candidate-heldout | Strict split tests novel candidate discovery | Frequency Hit@10 equals random expectation | Metric bottleneck | Hit@10 alone is misleading when candidate pools are small. |
| CA distribution audit | CA should be a reasonable content proxy | CA score is higher for negatives than positives | Representation/calibration bottleneck | Fixed physchem penalties are directionally wrong under hard negatives. |
| CTCR vs LBC no-freq | CTCR should be a new algorithm | CTCR only marginally differs | Algorithm identity bottleneck | Need structural ability beyond no-frequency logistic content scoring. |

## Problem Definition

Given a query fragment replacement context and a candidate replacement fragment, rank candidates so that BindingDB-supported replacements appear early.

Inputs:

- old fragment `f_old`
- candidate fragment `c`
- query context `z`, such as attachment signature, core class, target/endpoint family when available, and old-fragment chemotype cluster
- no raw candidate frequency for held-out positive candidates

Output:

- a candidate ranking for each query

Primary evaluation axis:

- weak candidate-heldout: held-out positive candidate IDs do not appear in train rows, while other test negative candidates may appear

Leakage audit axis:

- strict candidate-heldout: every test candidate ID is absent from train rows

Stress axes:

- candidate-cluster-heldout: hold out clusters of candidate fragments, not only exact candidate IDs
- core-heldout: hold out replacement cores
- temporal split: train on earlier evidence and test on later evidence

## Metric Policy

Hit@10 is retained for continuity but is not sufficient for candidate-heldout claims.

Primary metrics:

- OF-macro MRR
- OF-macro NDCG@10
- query random-normalized Hit@10

Secondary metrics:

- Hit@1
- Hit@5
- Hit@10
- Hit@10 enrichment over random
- candidates/query and positives/query distribution

Strict candidate-heldout must always report random Hit@10 expectation. A method with high raw Hit@10 but enrichment near 1 is treated as random-level.

## Locked Baseline Ladder

The proposed method must beat the strongest learned no-frequency baselines, not just Frequency or CA.

| Tier | Baseline | Role |
|---|---|---|
| B0 | Frequency | Support shortcut; expected to fail under weak candidate-heldout positives. |
| B1 | CA | Hand-crafted content proxy; currently inverted under BindingDB-hard. |
| B2 | fixed Blend | Legacy support/content blend sanity check. |
| B3 | FullBlend with matched candidate-heldout tuning | Strong learned-content/support blend; selects `alpha=0` in weak candidate-heldout. |
| B4 | full-content scalar | Ridge learned-content baseline with no candidate support. |
| B5 | LBC no-freq | Strongest current no-frequency learned content baseline. |
| B6 | CTCR prototype | Current calibrated ensemble probe; not a final method. |

Algorithm gate:

- weak candidate-heldout: proposed model > LBC no-freq and FullBlend by at least +0.03 OF-macro MRR or +0.03 OF-macro NDCG@10, with no degradation in Hit@1
- strict candidate-heldout: proposed model must exceed random-normalized Hit@10 of LBC no-freq/FullBlend, not merely raw Hit@10
- core-heldout: proposed model must remain competitive with FullBlend and LBC no-freq

## Algorithm Route A: Context-Conditioned Content Calibration

Short name: C3C, context-conditioned content calibration.

Target bottleneck:

CA fails because fixed global content weights are wrong; learned global weights help, but the correct weighting likely depends on replacement context.

Score:

```text
score(c, f, z) = w0^T x(c, f) + rho(z) * delta_w(z)^T x(c, f)
```

where:

- `x(c, f)` is the no-frequency content feature vector
- `w0` is the global no-frequency content weight
- `delta_w(z)` is a context residual
- `rho(z) = n_z / (n_z + tau)` shrinks low-support contexts back to the global model

Minimal context features:

- attachment signature
- attachment atom type
- aromatic versus aliphatic attachment
- ring versus chain context
- old-fragment chemotype cluster
- core class or core cluster when available

Do not use candidate ID or candidate frequency as a feature in candidate-heldout experiments.

Diagnostic v0:

- train global LBC no-freq
- train context residual linear/logistic models for contexts with enough support
- use shrinkage for all contexts
- evaluate under weak candidate-heldout, strict candidate-heldout, and core-heldout

Kill test:

- if C3C does not beat LBC no-freq or FullBlend under weak candidate-heldout, it is a context analysis, not an algorithm.

Fallback claim:

- learned content transfer is mostly global rather than context-specific.

## Algorithm Route B: Counterfactual CA-Matched Calibration

Short name: CF-Cal, counterfactual content calibration.

Target bottleneck:

The benchmark already constructs counterfactual decoys: same target/endpoint/attachment-compatible candidates that are CA-matched but unsupported. Training should explicitly learn residual evidence inside CA-matched candidate sets.

Core idea:

```text
learn P(y = 1 | content features, CA bucket, query/counterfactual set)
```

Minimal training forms:

1. CA-bucket residual BCE:
   - bucket rows by CA score quantile within each training split
   - train no-frequency logistic content model with CA-bucket fixed effects
   - balance queries, old fragments, and candidates

2. CA-matched set BCE:
   - for each positive candidate, sample CA-matched hard negatives from the same target/endpoint/attachment-compatible candidate set
   - train BCE on these matched sets
   - no pairwise margin loss initially

3. Residual ranker:
   - fit CA or full-content scalar first
   - train a residual classifier on errors within CA-matched neighborhoods

Why this is different from failed pairwise margin:

- the negative set is defined by the benchmark's counterfactual semantics, not by generic ranking difficulty
- the training target remains calibrated BCE, not an unconstrained pairwise ranking objective
- the model is forced to explain why CA-similar candidates diverge

Kill test:

- if CF-Cal improves Hit@10 but not Hit@1/MRR/NDCG@10, it is only exploiting K-width and should be rejected.

Fallback claim:

- counterfactual matching is valuable as an evaluation protocol even if the residual model does not beat learned baselines.

## Algorithm Route C: Reliability-Gated Learned Content

Short name: RGC, reliability-gated content.

Target bottleneck:

FullBlend already learns to shut frequency off under candidate-heldout. A useful algorithm may instead learn when global content is reliable versus when context residuals should dominate.

Score:

```text
score = g(z, diagnostics) * score_context_content
      + (1 - g(z, diagnostics)) * score_global_content
```

Gate inputs:

- context support count
- candidate pool size
- local CA variance
- disagreement between CA and no-frequency content
- number of positives per training context

Kill test:

- if the gate collapses to always global content, do not claim a gated algorithm.

Fallback claim:

- simple global content calibration is sufficient under current data scale.

## Experiment Ladder

### Phase 0: metric and audit lock

Done:

- matched inner FullBlend tuning
- weak and strict candidate-heldout
- stable Hit@K tie-breaking
- Hit@1/5/10, MRR, NDCG@10, random-normalized Hit@10
- CA/CTCR distribution audit

### Phase 1: 10k diagnostic v4

Status: completed for CF-C3 on 2026-06-10.

Add baselines and v0 methods:

- CA
- full-content scalar
- FullBlend with matched tuning
- LBC no-freq
- CTCR prototype
- C3C v0
- CF-Cal v0
- RGC v0 if C3C and CF-Cal both show partial wins

Splits:

- weak candidate-heldout
- strict candidate-heldout
- core-heldout

Decision:

- promote only methods that beat LBC no-freq and FullBlend under weak candidate-heldout on MRR/NDCG@10 and survive strict/core audits

CF-C3 result:

| Split | Gate result |
|---|---|
| weak candidate-heldout | Passed: CF-C3 beats the strongest learned baseline by +0.095154 OF-macro MRR and +0.117359 OF-macro NDCG@10. |
| strict candidate-heldout | Passed as leakage/random audit: CF-C3 improves OF-macro MRR to 0.778617 and query RN-Hit@10 to 0.717022. |
| core-heldout | Passed: CF-C3 improves OF-macro MRR to 0.813806 and OF-macro NDCG@10 to 0.691118. |

The current paper route can now be upgraded from a mechanism/benchmark seed to an algorithm-candidate paper, pending candidate-cluster-heldout and ablation diagnostics.

### Phase 2: candidate-cluster-heldout diagnostic

Purpose:

- prevent exact candidate ID heldout from being too narrow
- test generalization to unseen candidate chemotypes

Construction:

- cluster candidate fragments by Morgan/Tanimoto or scaffold/core key
- hold out clusters containing positive candidates
- report cluster overlap audit

### Phase 3: full BindingDB-hard run

Only run after Phase 1 or Phase 2 produces a real method separation.

Minimum:

- 10 seeds
- primary OF-macro MRR and NDCG@10
- Hit@1/5/10 and random-normalized Hit@10
- weak candidate-heldout as primary
- strict candidate-heldout as leakage audit
- core-heldout as transfer audit

### Phase 4: external robustness

Use only after the BindingDB-hard evidence is stable:

- ChEMBL temporal split as auxiliary time robustness
- SwissBioisostere overlap as replacement-reference sanity check, not label validation

## Figure Plan

| Figure | Message | Evidence |
|---|---|---|
| Fig. 1 | Candidate-heldout is a different problem axis from OF-heldout. | Split schematic plus support-removal audit. |
| Fig. 2 | Similarity can fail under hard negatives. | CA positive/negative inversion and feature distribution contrasts. |
| Fig. 3 | Context/counterfactual calibration method. | CF-C3 pipeline diagram. |
| Fig. 4 | Main diagnostic table. | Hit@1/5/10, MRR, NDCG@10, RN-Hit@10 across weak/strict/core splits. |
| Fig. 5 | Ablation and kill-test evidence. | No context, no counterfactual, no shrinkage, CA-bucket controls. |

## Paper Skeleton

### Abstract

Message: Existing fragment replacement ranking can look solved when candidate support or easy negatives dominate, but candidate-heldout hard negatives reveal that fixed similarity proxies can invert. We define candidate-heldout fragment replacement ranking and show that calibrated learned content is necessary; a new method must beat strong no-frequency learned baselines.

### Introduction

Paragraph roles:

1. Fragment replacement ranking needs candidates that generalize beyond historical support.
2. OF-heldout evaluation permits candidate support transfer and can obscure novel-candidate behavior.
3. BindingDB-hard candidate-heldout reveals a sharper failure: CA assigns higher scores to negatives than positives.
4. This motivates transferable content calibration rather than more frequency engineering.
5. Contributions: task/split, audit metrics, mechanism evidence, and CF-C3 as a calibrated content method that passes the first 10k gate.

### Method

Sections:

1. Candidate-heldout problem formulation
2. Hard-negative and split construction
3. Baseline ladder
4. Proposed calibration model
5. Training and hyperparameter tuning protocol

### Experiments

Sections:

1. Diagnostic setup and metrics
2. CA inversion and feature distribution audit
3. Main candidate-heldout ranking results
4. Strict candidate leakage audit
5. Core/candidate-cluster transfer
6. Ablations and failure analysis

## Claim Gates

| Outcome | Paper route | Claim |
|---|---|---|
| CF-C3 beats LBC no-freq and FullBlend by the pre-set margin | Algorithm-candidate paper | Counterfactual context-calibrated content ranking improves candidate-heldout fragment replacement ranking. |
| New methods tie LBC no-freq/FullBlend but all learned content beats CA | Mechanism paper | Candidate-heldout ranking is governed by learned transferable content, not hand-crafted similarity. |
| Strict/candidate-cluster splits expose metric or leakage problems | Benchmark paper | Candidate-heldout fragment replacement needs new split and metric controls. |
| No method beats strong baselines, but CA inversion is robust | Benchmark/mechanism paper | Hard counterfactual negatives reveal why simple similarity proxies fail. |

## Immediate Code Targets

1. Add candidate-cluster-heldout split and overlap audit.
2. Add CF-C3 ablations: no CA bucket, no context interactions, no CA-matched selection, no target context.
3. Re-run 10k v4 with the ablation ladder and matched inner tuning.
4. If CF-C3 survives candidate-cluster-heldout and ablations, run the full BindingDB-hard 10-seed benchmark.

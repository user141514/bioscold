# Evidence Bank

Catalogs all quantitative claims, tables, and evidence-bearing units in the existing manuscript. Each entry tagged with claim strength, motivation link, and audit status.

---

## Quantitative Claims (Locked Numbers — NOT to be changed)

| Claim ID | Claim | Evidence | Location | Claim Strength | Motivation Link | Audit |
|---|---|---|---|---|---|---|
| E01 | 82-feat scorer Top-10 = 0.8851, 95% CI [0.8796, 0.8903] | Table 1, 13,347 blind queries | §4.1 | High | Primary prospective result | LOCKED |
| E02 | 77-feat scorer Top-10 = 0.9243, 95% CI [0.9199, 0.9287] | Table 1, 13,347 blind queries | §4.1 | High (post-audit) | Primary post-audit result | LOCKED |
| E03 | ΔTop-10 (77-feat vs Score Blend) = +0.0686, 95% CI [+0.0638, +0.0733] | Paired bootstrap, B=5000 | §4.1, §5.1 | High | Primary comparison claim | LOCKED |
| E04 | ΔTop-10 (77-feat vs 82-feat) = +0.0393, 95% CI [+0.0354, +0.0432] | Paired bootstrap, B=5000 | §4.2 | High | Prior_ranks removal benefit | LOCKED |
| E05 | Attachment-Frequency Top-10 = 0.6019 | Table 1 | §4.1 | High | Baseline | LOCKED |
| E06 | HGB Top-10 = 0.7437 | Table 1 | §4.1 | High | Baseline | LOCKED |
| E07 | DE Top-10 = 0.8055 | Table 1 | §4.1 | High | Baseline | LOCKED |
| E08 | Borda(DE,HGB) Top-10 = 0.8384 | Table 1 | §4.1 | High | DE-HGB complementarity evidence | LOCKED |
| E09 | MLP Top-10 = 0.8402 | Table 1 | §4.1 | High | Baseline | LOCKED |
| E10 | Score Blend Top-10 = 0.8558 | Table 1 | §4.1 | High | Strongest baseline | LOCKED |
| E11 | Best-of-DE+HGB = 0.8686 | Table 1 | §4.1 | High (diagnostic only) | Diagnostic bound | LOCKED |
| E12 | Rescue/lost vs Score Blend: R=1016, L=101, net=+915 | Table 3 | §4.3 | High | Reliability evidence | LOCKED |
| E13 | Rescue/lost vs 82-feat scorer: R=626, L=102, net=+524 | Table 3 | §4.3 | High | Prior_ranks reliability improvement | LOCKED |
| E14 | Rescue-to-loss ratio = 10.06, CI [8.25, 12.52] | Table 3 | §4.3 | High | Effect size | LOCKED |
| E15 | 82-feat lost queries vs Score Blend: 596 → 77-feat: 101 (5.9× reduction) | Table 3 + Supplementary | §4.3 | High | Prior_ranks harm quantified | LOCKED |
| E16 | G2 alert rate = 46.85%, A4C coverage 100% | Table 4 | §4.5 | Moderate | A4C triage signal | LOCKED |
| E17 | G3 alert rate = 9.67%, A4C coverage 100% | Table 4 | §4.5 | Moderate | A4C triage signal | LOCKED |
| E18 | G4 A4C coverage = 5.63%, alert rate 17.60% among covered | Table 4 | §4.5 | Low (sparse coverage) | A4C limitation | LOCKED |
| E19 | 5 prior_ranks features removed (backoff_logit_rank, cont_prior_rank, t_p4_logit_rank, p1_logit_rank, p3_logit_rank) | §3.5.1, Table 2 | §3.5 | High | Mechanism evidence | LOCKED |
| E20 | 8 dev fragments → 19 blind fragments (fragment identity shift) | §3.5.2 | §3.5, §5.3 | High | Shortcut mechanism | LOCKED |
| E21 | Naive encoding bug Top-10 = 0.7120 → 0.8851 after frozen schema | §4.4 | §4.4 | High (implementation fix) | Schema alignment necessity | LOCKED |
| E22 | DE +0.2036 over Attachment-Frequency baseline | Table 1 | §4.1 | High | DE ranking strength | LOCKED |
| E23 | Borda +0.0329 over DE alone | Table 1 | §4.1 | High | Complementarity evidence | LOCKED |
| E24 | Drop model_ranks (F3): Top-10 = 0.8697, Δ = -0.0154 | Table 2 | §4.2 | Moderate | Ablation control | LOCKED |
| E25 | Drop model_scores (F1): Top-10 = 0.8610, Δ = -0.0241 | Table 2 | §4.2 | Moderate | Ablation control | LOCKED |

## Tables (Preserve as-is unless structural change needed)

| Table ID | Title | Location | Status |
|---|---|---|---|
| M1 | Candidate-matrix statistics | §3.2 | KEEP |
| M2 | Leakage verification | §3.2 | KEEP |
| M3 | Feature families with audit trail | §3.4.2 | KEEP |
| M4 | Evaluation protocol separation | §3.6.4 | KEEP |
| M5 | Provenance groups and definitions | §3.7.2 | KEEP |
| 1 | Secondary blind Top-10 accuracy | §4.1 | KEEP |
| 2 | Leave-one-family-out ablation | §4.2 | KEEP |
| 3 | Rescue/lost analysis | §4.3 | KEEP |
| 4 | A4C alert rates by provenance group | §4.5 | KEEP |

## Qualitative Claims (Scope-checked)

| Claim ID | Claim | Evidence Basis | Motivation Link | Overclaim Risk | Audit |
|---|---|---|---|---|---|
| Q01 | Transform-heldout split prevents memorization of transform identities | Table M2 (zero overlap) | Primary contribution | None — direct evidence | SAFE |
| Q02 | Prior_ranks are non-transferable shortcuts | §3.5.2 mechanism, Table 2 | Secondary contribution | None — mechanism is specific and testable | SAFE |
| Q03 | Candidate-level scoring succeeds where query-level routing fails | §5.2 analysis, AUC 0.61 router | Supporting insight | Low — empirical observation, claim is scoped | SAFE |
| Q04 | "Sparse prior-score rank transforms can become generalization hazards" | §3.5.2, design rule | Secondary contribution | Low — scoped to cross-split molecular ranking | SAFE |
| Q05 | A4C provides "coverage-limited computational triage" | Table 4, G4 5.63% coverage | Supporting | None — limitation explicitly stated | SAFE |
| Q06 | "No reliable G4-wide alert estimate is possible" | Table 4, 94.37% unknown | Supporting | None — honest limitation | SAFE |

## Evidence Gaps (Claims needing stronger anchoring)

| Gap ID | What's Missing | Current State | Fix |
|---|---|---|---|
| G01 | Per-fragment improvement visualization | Mentioned in text ("all 19 old-fragment strata are at baseline parity"), values in Supplementary Table S5 | Consider: add a small figure showing per-fragment ΔTop-10 for 77-feat vs 82-feat scorer |
| G02 | Query-level router failure quantified | AUC ≈ 0.61 mentioned in §5.2, no formal comparison table | Add: brief table comparing routing strategies attempted + failure mode |
| G03 | DE-HGB complementarity visualized | Borda improvement +0.0329 in Table 1, no overlap/disagreement visualization | Consider: Venn diagram or overlap heatmap of DE vs HGB top-10 hits |

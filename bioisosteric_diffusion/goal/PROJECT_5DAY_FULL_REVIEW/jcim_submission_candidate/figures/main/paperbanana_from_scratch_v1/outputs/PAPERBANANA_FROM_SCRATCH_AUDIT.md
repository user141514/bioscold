# PaperBanana From-Scratch Figure Suite Audit

## Scope

This folder contains a from-scratch redraw of Figures 1-5 using PaperBanana-style design constraints: faithfulness, conciseness, readability, and restrained academic aesthetics. It does not edit the frozen v4 SVGs.

## Key Decisions

- Figure 1 focuses only on task definition and leakage-controlled evaluation.
- Figure 2 focuses on candidate-level scoring architecture and the post-audit pruning ribbon.
- Figure 3 preserves the locked secondary blind Top-10 values and CIs.
- Figure 4 preserves the locked ablation and rescue/lost values.
- Figure 5 preserves the dual-mode provenance triage and A4C claim boundary.

## Locked Values Preserved

- Figure 3: Attachment 0.6019, HGB 0.7437, DE 0.8055, Borda 0.8384, MLP 0.8402, Score Blend 0.8558, 82-feature 0.8851, 77-feature 0.9243, Best-of-DE+HGB diagnostic 0.8686.
- Figure 4: Drop prior ranks +0.0393 CI [0.0354, 0.0432], Drop model ranks -0.0154, Drop model scores -0.0241; lost 596 -> 101; rescue/lost/net values 1016/101/+915 and 626/102/+524.
- Figure 5: G2 46.85%, G3 9.67%, G4 5.63% coverage and 17.60% among covered with 94.37% unknown.

## Claim Boundaries

- No activity-preserving, wet-lab, expert-validation, or toxicity-verdict claim is introduced.
- A4C remains computational triage only.
- 77-feature scorer remains post-audit/post-selection, not fully prospective.
- Best-of-DE+HGB remains diagnostic and is not called Oracle, ceiling, or upper bound.

# PaperBanana Integration Notes

## Status

PaperBanana was cloned to:

`E:\zuhui\bioisosteric_diffusion\external_tools\PaperBanana`

The requested conda environment is:

`E:\Anaconda3\envs\rag-env`

Smoke tests completed in `rag-env`:

- `conda run -n rag-env python --version` -> Python 3.11.14
- `conda run -n rag-env python skill\run.py --help` -> OK
- PaperBanana agent import smoke test -> `PAPERBANANA_IMPORT_SMOKE_TEST_OK`

The earlier `.venv` created inside PaperBanana is not used.

## Current Blocker

Full PaperBanana generation requires one of:

- `OPENROUTER_API_KEY`
- `GOOGLE_API_KEY`

Neither is currently set in the environment. Without an API key, the skill can be inspected and prepared, but cannot generate candidate images.

## Recommended Use For This Manuscript

Use PaperBanana for concept/style candidates for schematic figures:

- Figure 1: benchmark overview
- Figure 2: candidate-level architecture schematic
- Figure 5: provenance triage schematic

Do not use PaperBanana to regenerate locked numeric result figures:

- Figure 3: secondary blind Top-10 forest plot
- Figure 4: ablation/reliability audit

Those should remain deterministic Python/SVG outputs because they contain locked values, confidence intervals, and claim-boundary-sensitive evidence.

## Run Commands

After setting an API key, run:

```powershell
cd E:\zuhui\bioisosteric_diffusion\goal\PROJECT_5DAY_FULL_REVIEW\jcim_submission_candidate\figures\main\paperbanana_tooling

$env:OPENROUTER_API_KEY = "sk-or-v1-..."

.\RUN_PAPERBANANA_WITH_RAG_ENV.ps1 -Figure Figure1 -Candidates 1 -CriticRounds 1
.\RUN_PAPERBANANA_WITH_RAG_ENV.ps1 -Figure Figure2 -Candidates 1 -CriticRounds 1
.\RUN_PAPERBANANA_WITH_RAG_ENV.ps1 -Figure Figure5 -Candidates 1 -CriticRounds 1
```

Outputs will be written to:

`E:\zuhui\bioisosteric_diffusion\goal\PROJECT_5DAY_FULL_REVIEW\jcim_submission_candidate\figures\main\paperbanana_tooling\outputs`

## Manuscript-Safe Workflow

1. Generate one or more PaperBanana candidate PNGs for Figures 1/2/5.
2. Review them only for layout ideas and visual hierarchy.
3. Rebuild accepted ideas in editable SVG/PDF/PNG using the current manuscript figure scripts.
4. Preserve locked scientific content and claim boundaries.

PaperBanana should not be treated as an authoritative data plotting or claim-generation system for this paper.

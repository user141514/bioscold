# PaperBanana Generation Status

## Current status

`READY_BUT_BLOCKED_BY_API_KEY`

The PaperBanana reference datasets are installed and the figure-generation input briefs are prepared. The PaperBanana skill runner starts correctly inside `rag-env`, but generation cannot proceed until at least one model API key is configured.

## Verified local setup

- PaperBanana repo: `E:\zuhui\bioisosteric_diffusion\external_tools\PaperBanana`
- Conda environment: `rag-env`
- PaperBananaBench: available
- PaperBananaDiagramPDFs: available
- Dataset-aware retriever smoke test: passed
- Skill help command: passed

## Blocking condition

The generation runner stopped with:

```text
PaperBanana generation needs OPENROUTER_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY. Set one before running.
```

Current detected keys:

- `OPENROUTER_API_KEY`: missing
- `GOOGLE_API_KEY`: missing
- `OPENAI_API_KEY`: missing
- `ANTHROPIC_API_KEY`: missing

## Prepared inputs

The following figure briefs are ready:

- `inputs\Figure_1_content.md`
- `inputs\Figure_2_content.md`
- `inputs\Figure_3_content.md`
- `inputs\Figure_4_content.md`
- `inputs\Figure_5_content.md`
- `inputs\GLOBAL_STYLE.md`

## Resume commands

After setting a key in the same PowerShell session:

```powershell
cd E:\zuhui\bioisosteric_diffusion\goal\PROJECT_5DAY_FULL_REVIEW\jcim_submission_candidate\figures\main\paperbanana_generation_v1
$env:OPENROUTER_API_KEY="sk-or-v1-..."
.\RUN_PAPERBANANA_GENERATE.ps1 -Figure Figure2 -Candidates 1 -CriticRounds 1
```

Then run all figures:

```powershell
.\RUN_PAPERBANANA_GENERATE.ps1 -Figure All -Candidates 1 -CriticRounds 1
```

Figures 3 and 4 contain locked numerical evidence, so PaperBanana outputs for those figures should be treated as layout/design candidates only unless every number and CI is manually audited.

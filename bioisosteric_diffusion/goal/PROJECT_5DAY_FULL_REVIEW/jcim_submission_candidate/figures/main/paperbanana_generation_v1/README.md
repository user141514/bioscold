# PaperBanana Generation v1

This folder contains direct inputs for the PaperBanana skill CLI. Unlike the deterministic SVG harmonization workflow, this workflow asks PaperBanana to generate new candidate figures from text, using its local reference datasets:

- `E:\zuhui\bioisosteric_diffusion\external_tools\PaperBanana\data\PaperBananaBench`
- `E:\zuhui\bioisosteric_diffusion\external_tools\PaperBanana\data\PaperBananaDiagramPDFs`

## Run

Set an API key first:

```powershell
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
```

Then run:

```powershell
cd E:\zuhui\bioisosteric_diffusion\goal\PROJECT_5DAY_FULL_REVIEW\jcim_submission_candidate\figures\main\paperbanana_generation_v1
.\RUN_PAPERBANANA_GENERATE.ps1 -Figure Figure2 -Candidates 1 -CriticRounds 1
```

To generate all five candidate figures:

```powershell
.\RUN_PAPERBANANA_GENERATE.ps1 -Figure All -Candidates 1 -CriticRounds 1
```

## Caution

Figures 3 and 4 contain locked numeric evidence. PaperBanana candidates for these figures should be treated as layout drafts only. Final quantitative figures should be checked against the locked values before manuscript use.

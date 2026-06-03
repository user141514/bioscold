# PaperBanana DeepSeek Run Log

## Status

`STARTED_WITH_DEEPSEEK_TEXT_PROVIDER`

DeepSeek was added as an OpenAI-compatible text provider for PaperBanana. Because the provided key is a DeepSeek text-model key rather than an image-generation key, diagram figures are routed through the PaperBanana `plot` code-generation path (`TaskMode=PlotAll`) for now.

## Privacy handling

- The API key was stored in the Windows user environment variable `DEEPSEEK_API_KEY`.
- The key was not written into `configs/model_config.yaml`.
- The key is not included in this run log.
- `configs/model_config.yaml` remains gitignored by the PaperBanana repo.

## Local compatibility changes

- `external_tools/PaperBanana/utils/generation_utils.py`
  - Added Windows user-environment fallback for API-key lookup.
  - Added `deepseek/` provider routing.
  - Added DeepSeek chat retry helper.
- `external_tools/PaperBanana/skill/run.py`
  - Passed CLI `--task` into `ExpConfig.task_name`.
  - Fixed final image extraction to respect `diagram` vs `plot`.
- `RUN_PAPERBANANA_GENERATE.ps1`
  - Added DeepSeek key detection.
  - Added `PlotAll` mode.
  - Flattened captions before passing them through Windows command-line calls.

## DeepSeek smoke test

DeepSeek model routing passed with:

```text
clients=DeepSeek
response=OK
```

Working model name:

```text
deepseek/deepseek-v4-pro
```

## Figure 2 outputs

Raw PaperBanana candidate:

```text
outputs/paperbanana_figure_2_candidate.png
```

The raw candidate proved the PaperBanana + DeepSeek pipeline can run, but it had layout defects:

- ranker labels overlapped;
- audit ribbon sat too low;
- right-side text was clipped.

Cleaned editable candidate:

```text
outputs/Figure_2_paperbanana_cleaned_v3.svg
outputs/Figure_2_paperbanana_cleaned_v3.pdf
outputs/Figure_2_paperbanana_cleaned_v3.png
```

The cleaned v3 figure preserves the PaperBanana from-scratch concept but uses controlled matplotlib drawing to ensure no text overlap, no clipping, and editable SVG/PDF output.

## Current recommendation

Use `Figure_2_paperbanana_cleaned_v3` as the first from-scratch candidate for visual review.

Next recommended sequence:

1. Generate/clean Figure 1 from the same PaperBanana + controlled-vector workflow.
2. Generate/clean Figure 5.
3. Keep Figures 3 and 4 mostly locked, using PaperBanana only for alternative layout ideation because their numerical values are locked evidence.

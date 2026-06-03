param(
    [ValidateSet("Figure1", "Figure2", "Figure5")]
    [string]$Figure = "Figure2",
    [int]$Candidates = 1,
    [int]$CriticRounds = 1
)

$ErrorActionPreference = "Stop"

$repo = "E:\zuhui\bioisosteric_diffusion\external_tools\PaperBanana"
$work = "E:\zuhui\bioisosteric_diffusion\goal\PROJECT_5DAY_FULL_REVIEW\jcim_submission_candidate\figures\main\paperbanana_tooling"
$inputs = Join-Path $work "inputs"
$outputs = Join-Path $work "outputs"
New-Item -ItemType Directory -Path $outputs -Force | Out-Null

if (-not $env:OPENROUTER_API_KEY -and -not $env:GOOGLE_API_KEY) {
    Write-Error "PaperBanana needs OPENROUTER_API_KEY or GOOGLE_API_KEY. Set one before running generation."
}

switch ($Figure) {
    "Figure1" {
        $contentFile = Join-Path $inputs "figure1_paperbanana_content.md"
        $caption = Get-Content -Path (Join-Path $inputs "figure1_paperbanana_caption.txt") -Raw
        $output = Join-Path $outputs "paperbanana_figure1_candidate.png"
        $ratio = "16:9"
    }
    "Figure2" {
        $contentFile = Join-Path $inputs "figure2_paperbanana_content.md"
        $caption = Get-Content -Path (Join-Path $inputs "figure2_paperbanana_caption.txt") -Raw
        $output = Join-Path $outputs "paperbanana_figure2_candidate.png"
        $ratio = "16:9"
    }
    "Figure5" {
        $contentFile = Join-Path $inputs "figure5_paperbanana_content.md"
        $caption = Get-Content -Path (Join-Path $inputs "figure5_paperbanana_caption.txt") -Raw
        $output = Join-Path $outputs "paperbanana_figure5_candidate.png"
        $ratio = "16:9"
    }
}

Push-Location $repo
try {
    conda run -n rag-env python skill\run.py `
        --content-file $contentFile `
        --caption $caption `
        --task diagram `
        --output $output `
        --aspect-ratio $ratio `
        --num-candidates $Candidates `
        --max-critic-rounds $CriticRounds `
        --retrieval-setting none `
        --exp-mode demo_planner_critic
}
finally {
    Pop-Location
}

param(
    [ValidateSet("Figure1", "Figure2", "Figure3", "Figure4", "Figure5", "All")]
    [string]$Figure = "Figure2",
    [int]$Candidates = 1,
    [int]$CriticRounds = 1,
    [string]$Retrieval = "auto",
    [string]$ExpMode = "demo_full",
    [ValidateSet("Auto", "PlotAll")]
    [string]$TaskMode = "PlotAll",
    [string]$MainModel = "deepseek/deepseek-v4-pro",
    [string]$ImageModel = "deepseek/deepseek-v4-pro"
)

$ErrorActionPreference = "Stop"

$repo = "E:\zuhui\bioisosteric_diffusion\external_tools\PaperBanana"
$work = "E:\zuhui\bioisosteric_diffusion\goal\PROJECT_5DAY_FULL_REVIEW\jcim_submission_candidate\figures\main\paperbanana_generation_v1"
$inputs = Join-Path $work "inputs"
$outputs = Join-Path $work "outputs"
New-Item -ItemType Directory -Path $outputs -Force | Out-Null

if (-not $env:DEEPSEEK_API_KEY) {
    $userDeepSeekKey = [Environment]::GetEnvironmentVariable("DEEPSEEK_API_KEY", "User")
    if ($userDeepSeekKey) {
        $env:DEEPSEEK_API_KEY = $userDeepSeekKey
    }
}

if (-not $env:OPENROUTER_API_KEY -and -not $env:GOOGLE_API_KEY -and -not $env:OPENAI_API_KEY -and -not $env:DEEPSEEK_API_KEY) {
    throw "PaperBanana generation needs OPENROUTER_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY. Set one before running."
}

function Invoke-PaperBananaFigure {
    param(
        [int]$Number
    )
    $contentFile = Join-Path $inputs "Figure_${Number}_content.md"
    $captionFile = Join-Path $inputs "Figure_${Number}_caption.txt"
    $caption = ((Get-Content -Path $captionFile -Raw) -replace "\s+", " ").Trim()
    $task = if ($TaskMode -eq "PlotAll") { "plot" } elseif ($Number -in @(3,4)) { "plot" } else { "diagram" }
    $output = Join-Path $outputs "paperbanana_figure_${Number}_candidate.png"

    $paperBananaArgs = @(
        "skill\run.py",
        "--content-file", $contentFile,
        "--caption", $caption,
        "--task", $task,
        "--output", $output,
        "--aspect-ratio", "16:9",
        "--num-candidates", "$Candidates",
        "--max-critic-rounds", "$CriticRounds",
        "--retrieval-setting", $Retrieval,
        "--exp-mode", $ExpMode
    )
    if ($MainModel) {
        $paperBananaArgs += @("--main-model-name", $MainModel)
    }
    if ($ImageModel) {
        $paperBananaArgs += @("--image-gen-model-name", $ImageModel)
    }

    Push-Location $repo
    try {
        conda run -n rag-env python @paperBananaArgs
    }
    finally {
        Pop-Location
    }
}

$figures = if ($Figure -eq "All") { 1..5 } else { [int]($Figure -replace "Figure", "") }
foreach ($n in $figures) {
    Invoke-PaperBananaFigure -Number $n
}

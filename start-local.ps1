param(
    [Alias('ModelsDir')]
    [string] $FinalModels,
    [string] $FinalCode,
    [string] $ModelDir,
    [string] $YoloWeights,
    [string] $Python,
    [string] $DependencyPath,
    [ValidateRange(1, 65535)] [int] $Port = 8000
)

$ErrorActionPreference = 'Stop'
if (-not $Python) {
    $venvPy = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
    $Python = if (Test-Path -LiteralPath $venvPy) { $venvPy } else { 'python' }
}
$Python = (Get-Command $Python).Source

if ($DependencyPath) {
    $env:PYTHONPATH = (Resolve-Path -LiteralPath $DependencyPath).Path
}
if ($FinalCode) {
    $env:HIGHLIGHT_FINAL_CODE = (Resolve-Path -LiteralPath $FinalCode).Path
} else {
    $bundled = Join-Path $PSScriptRoot 'engine'
    if (Test-Path -LiteralPath (Join-Path $bundled 'run_raw.py')) {
        $env:HIGHLIGHT_FINAL_CODE = (Resolve-Path -LiteralPath $bundled).Path
    }
}
if ($FinalModels) {
    $env:HIGHLIGHT_FINAL_MODELS = (Resolve-Path -LiteralPath $FinalModels).Path
}
if ($ModelDir) {
    $env:HIGHLIGHT_MODEL_DIR = (Resolve-Path -LiteralPath $ModelDir).Path
}
if ($YoloWeights) {
    $env:HIGHLIGHT_YOLO_WEIGHTS = (Resolve-Path -LiteralPath $YoloWeights).Path
}

$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

Push-Location -LiteralPath $PSScriptRoot
try {
    & $Python -m uvicorn studio.app:app --host 127.0.0.1 --port $Port
    exit $LASTEXITCODE
} finally {
    Pop-Location
}

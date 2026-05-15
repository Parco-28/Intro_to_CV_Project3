# Part 3 Exploration Pipeline (Windows PowerShell)
# Usage: .\scripts\run_part3.ps1 <video_or_frames_path> [-Masks <masks_dir>]

param(
    [Parameter(Mandatory=$true)] [string]$Source,
    [string]$Masks = "results/part2/masks",
    [string]$Output = "results/part3"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (Test-Path ".venv\Scripts\Activate.ps1") { . ".venv\Scripts\Activate.ps1" }

Write-Host "=== Part 3: Exploration Pipeline ===" -ForegroundColor Cyan
python part3_exploration/pipeline.py `
    --video "$Source" `
    --masks "$Masks" `
    --output "$Output"

Write-Host "=== Part 3 Complete ===" -ForegroundColor Green
Write-Host "Results in: $Output"

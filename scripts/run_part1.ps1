# Part 1 Baseline Pipeline (Windows PowerShell)
# Usage: .\scripts\run_part1.ps1 <video_or_frames_path>

param(
    [Parameter(Mandatory=$true)] [string]$Source,
    [string]$Output = "results/part1",
    [string]$Model = "yolov8m-seg.pt",
    [double]$Conf = 0.35,
    [double]$MotionThreshold = 2.0,
    [string]$InpaintMethod = "telea"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (Test-Path ".venv\Scripts\Activate.ps1") { . ".venv\Scripts\Activate.ps1" }

Write-Host "=== Part 1: Baseline Pipeline ===" -ForegroundColor Cyan
python part1_baseline/pipeline.py `
    --video "$Source" `
    --output "$Output" `
    --model "$Model" `
    --conf $Conf `
    --motion-threshold $MotionThreshold `
    --inpaint-method $InpaintMethod

Write-Host "=== Part 1 Complete ===" -ForegroundColor Green
Write-Host "Results in: $Output"

# Part 2 SOTA Pipeline (Windows PowerShell)
# Usage: .\scripts\run_part2.ps1 <video_or_frames_path>

param(
    [Parameter(Mandatory=$true)] [string]$Source,
    [string]$Output = "results/part2",
    [switch]$NoSam2,
    [switch]$NoPropainter
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (Test-Path ".venv\Scripts\Activate.ps1") { . ".venv\Scripts\Activate.ps1" }

$flags = @()
if ($NoSam2) { $flags += "--no-sam2" }
if ($NoPropainter) { $flags += "--no-propainter" }

Write-Host "=== Part 2: SOTA Pipeline ===" -ForegroundColor Cyan
python part2_sota/pipeline.py --video "$Source" --output "$Output" @flags

Write-Host "=== Part 2 Complete ===" -ForegroundColor Green
Write-Host "Results in: $Output"

# Run all three parts sequentially (Windows PowerShell)
# Usage: .\scripts\run_all.ps1 <video_or_frames_path>

param(
    [Parameter(Mandatory=$true)] [string]$Source
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Video Object Removal & Inpainting" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

& "$PSScriptRoot\run_part1.ps1" -Source $Source
Write-Host ""
& "$PSScriptRoot\run_part2.ps1" -Source $Source
Write-Host ""
& "$PSScriptRoot\run_part3.ps1" -Source $Source

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  All parts complete!" -ForegroundColor Green
Write-Host "  Results in: results/" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

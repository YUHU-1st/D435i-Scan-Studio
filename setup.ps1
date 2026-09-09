$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $ProjectRoot ".uv-cache"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv was not found. Install it from https://docs.astral.sh/uv/."
}

uv sync --python 3.12 --extra dev
Write-Host ""
Write-Host "Installation complete. Check the camera:"
Write-Host "  .\run.ps1 --list-devices"
Write-Host "Start the scanner GUI:"
Write-Host "  .\run.ps1"

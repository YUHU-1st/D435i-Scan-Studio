$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $ProjectRoot ".uv-cache"

uv run --python 3.12 d435i-scan --config (Join-Path $ProjectRoot "config.example.yml") @args


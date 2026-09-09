param([switch]$Dev)
$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $ProjectRoot '.uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $ProjectRoot '.tools\python'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $ProjectRoot '.venv'

function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit $LASTEXITCODE): $Executable $($Arguments -join ' ')"
    }
}

try {
    if ($PSVersionTable.PSEdition -eq 'Core' -and -not $IsWindows) {
        throw 'This installer is for Windows 10/11. Use uv sync on other systems.'
    }
    if (-not [Environment]::Is64BitOperatingSystem) {
        throw '64-bit Windows is required by the Python/Open3D dependencies.'
    }
    foreach ($file in @('pyproject.toml', 'uv.lock')) {
        if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $file))) {
            throw "Missing $file. Download the complete repository ZIP and extract it before running setup."
        }
    }
    $uv = Join-Path $ProjectRoot '.tools\uv\uv.exe'
    if (-not (Test-Path -LiteralPath $uv)) {
        $existing = Get-Command uv -ErrorAction SilentlyContinue
        if ($existing) {
            $uv = $existing.Source
        } else {
            Write-Host 'Installing uv from the official installer...'
            $installDir = Join-Path $ProjectRoot '.tools\uv'
            New-Item -ItemType Directory -Force -Path $installDir | Out-Null
            $env:UV_INSTALL_DIR = $installDir
            $env:UV_NO_MODIFY_PATH = '1'
            $installer = Join-Path $env:TEMP 'd435i-uv-install.ps1'
            Invoke-WebRequest -Uri 'https://astral.sh/uv/install.ps1' -OutFile $installer -UseBasicParsing
            Invoke-Checked 'powershell.exe' @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $installer)
            if (-not (Test-Path -LiteralPath $uv)) { throw "uv.exe was not created at $uv" }
        }
    }
    Push-Location $ProjectRoot
    try {
        Invoke-Checked $uv @('python', 'install', '3.12')
        $syncArgs = @('sync', '--locked', '--python', '3.12')
        if ($Dev) { $syncArgs += @('--extra', 'dev') }
        Invoke-Checked $uv $syncArgs
        $python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
        if (-not (Test-Path -LiteralPath $python)) { throw 'The virtual environment was not created.' }
        Invoke-Checked $python @('-c', 'import sys; assert sys.version_info[:2] == (3, 12); import numpy, yaml, open3d, pyrealsense2; print("Python and core dependencies: OK")')
        Write-Host ''
        Write-Host 'Installation complete. Double-click run.cmd to start.'
        Write-Host 'Optional: install FreeCAD separately to enable STEP export.'
        Write-Host 'If camera detection fails, run: .\run.cmd --doctor'
    } finally {
        Pop-Location
    }
} catch {
    [Console]::Error.WriteLine("Setup failed: $($_.Exception.Message)`nCheck internet access, disk space, and the error above. You can retry setup.cmd without deleting the project.")
    exit 1
}

# install-windows.ps1 — pi-vision-struct 安装器（Windows）
# 用法: powershell -ExecutionPolicy Bypass -File install-windows.ps1 [-WithOmniparser] [-WithDom]
# 依赖: Miniforge（conda）；截图需 `pip install mss`
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { Write-Error "需要 python（安装 Miniforge 后自动获得）"; exit 1 }

$args = @()
if ($WithOmniparser) { $args += "--with-omniparser" }
if ($WithDom) { $args += "--with-dom" }

& python -u python/setup/vs_setup.py @args
exit $LASTEXITCODE

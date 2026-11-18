param([int]$Port = 8765, [string]$Storage = '')
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'scripts\storage.ps1')
$Storage = Get-BookStorage $Storage
Set-Location $PSScriptRoot
$python = Join-Path $Storage '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw 'Ambiente ausente. Configure o ambiente de treinamento primeiro.' }
$env:MPLBACKEND = 'Agg'
& $python -u -m src.app.server --storage $Storage --port $Port
if ($LASTEXITCODE -ne 0) { throw 'A aplicacao encerrou com erro.' }

param([int]$Port = 8765, [string]$Storage = 'D:\what-is-the-book-2')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = Join-Path $Storage '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw 'Ambiente ausente. Configure o ambiente de treinamento primeiro.' }
$env:MPLBACKEND = 'Agg'
& $python -u -m src.app.server --storage $Storage --port $Port
if ($LASTEXITCODE -ne 0) { throw 'A aplicacao encerrou com erro.' }

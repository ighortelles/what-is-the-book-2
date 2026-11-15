param(
    [switch]$Setup,
    [switch]$Smoke,
    [string]$Storage = 'D:\what-is-the-book-2'
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:UV_PROJECT_ENVIRONMENT = Join-Path $Storage '.venv'
$env:UV_CACHE_DIR = Join-Path $Storage 'uv-cache'
$env:MPLBACKEND = 'Agg'
$env:PYTHONUNBUFFERED = '1'
$python = Join-Path $env:UV_PROJECT_ENVIRONMENT 'Scripts\python.exe'
if ($Setup) {
    uv sync --locked --python 3.13 --extra training --no-dev
    if ($LASTEXITCODE -ne 0) { throw 'Environment setup failed' }
    # Windows PyPI wheels are CPU-only; use the CUDA 12.6 build for Pascal.
    uv pip install --python $python --no-deps --index-url https://download.pytorch.org/whl/cu126 'torch==2.7.1+cu126' 'torchvision==0.22.1+cu126'
    if ($LASTEXITCODE -ne 0) { throw 'CUDA installation failed' }
}
if (-not (Test-Path $python)) { throw 'Run with -Setup first' }
& $python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; x=torch.randn(64,64,device='cuda',requires_grad=True); (x@x).sum().backward(); torch.cuda.synchronize(); print(torch.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'CUDA kernel check failed' }
$arguments = @('-u', '-m', 'src.detection.train_preprocessing', '--run', '--config', 'configs/yolov8_preprocessing_local.json', '--output', (Join-Path $Storage 'runs\preprocessing_local'), '--cache', (Join-Path $Storage 'preprocessed'), '--device', '0')
if ($Smoke) { $arguments += @('--smoke', '--variant', 'original') }
# Direct Python execution preserves the explicitly installed CUDA wheels.
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }

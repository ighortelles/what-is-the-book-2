param(
    [string]$Storage = 'D:\what-is-the-book-2',
    [int]$WaitForInstallProcess = 0
)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $Storage | Out-Null
$statusPath = Join-Path $Storage 'training_status.json'
$lock = [System.IO.File]::Open((Join-Path $Storage 'training.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
function Set-TrainingStatus([string]$State, [string]$Detail) {
    $value = @{ state=$State; detail=$Detail; updated=(Get-Date -Format o); pid=$PID }
    [System.IO.File]::WriteAllText($statusPath, ($value | ConvertTo-Json), [System.Text.UTF8Encoding]::new($false))
    Write-Output "STATUS | $State | $Detail"
}
try {
    if ($WaitForInstallProcess -gt 0) {
        Set-TrainingStatus 'installing' 'Waiting for CUDA package installation'
        $installer = Get-Process -Id $WaitForInstallProcess -ErrorAction SilentlyContinue
        if ($installer) { $installer.WaitForExit() }
    }
    Set-TrainingStatus 'smoke' 'Checking CUDA and running a short training'
    & (Join-Path $PSScriptRoot 'train_local.ps1') -Smoke -Storage $Storage
    Set-TrainingStatus 'training' 'Running all preprocessing variants'
    & (Join-Path $PSScriptRoot 'train_local.ps1') -Storage $Storage
    Set-TrainingStatus 'completed' 'All preprocessing variants completed'
} catch {
    Set-TrainingStatus 'failed' $_.Exception.Message
    Write-Error $_
    exit 1
} finally {
    $lock.Dispose()
}

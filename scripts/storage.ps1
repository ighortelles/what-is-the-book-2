function Get-BookStorage([string]$Override = '') {
    $repoRoot = Split-Path $PSScriptRoot -Parent
    $value = $Override
    if (-not $value) { $value = $env:BOOK_STORAGE }
    $config = Join-Path $repoRoot 'settings.local.json'
    if (-not $value -and (Test-Path $config)) {
        $value = (Get-Content $config -Raw -Encoding UTF8 | ConvertFrom-Json).storage
    }
    if (-not $value) { $value = $repoRoot }
    if (-not [System.IO.Path]::IsPathRooted($value)) { $value = Join-Path $repoRoot $value }
    return [System.IO.Path]::GetFullPath($value)
}

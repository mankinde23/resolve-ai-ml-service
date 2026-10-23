
$ErrorActionPreference = "Stop"

$root = (Get-Location).Path

Write-Host "`nRestructuring Resolve AI ML Service..." -ForegroundColor Cyan

# Create required directories
$directories = @(
    "app",
    "artifacts",
    "tests"
)

foreach ($directory in $directories) {
    $path = Join-Path $root $directory

    if (-not (Test-Path -LiteralPath $path)) {
        New-Item -ItemType Directory -Path $path | Out-Null
        Write-Host "Created $directory/"
    }
}

# Move files without overwriting existing files
$moves = @(
    @{ Source = "app.py";         Destination = "app/main.py" },
    @{ Source = "model.pkl";      Destination = "artifacts/model.pkl" },
    @{ Source = "vectorizer.pkl"; Destination = "artifacts/vectorizer.pkl" }
)

foreach ($move in $moves) {
    $source = Join-Path $root $move.Source
    $destination = Join-Path $root $move.Destination

    if ((Test-Path -LiteralPath $source) -and
        (Test-Path -LiteralPath $destination)) {
        throw "Both source and destination exist: $($move.Source) -> $($move.Destination). Resolve this conflict manually."
    }

    if (Test-Path -LiteralPath $source) {
        Move-Item -LiteralPath $source -Destination $destination
        Write-Host "Moved $($move.Source) -> $($move.Destination)" -ForegroundColor Green
    }
    elseif (Test-Path -LiteralPath $destination) {
        Write-Host "Already organized: $($move.Destination)"
    }
    else {
        Write-Warning "File not found: $($move.Source)"
    }
}

# Create starter files only when missing
$starterFiles = @{
    "app/__init__.py" = ""
    "app/schemas.py" = @'
"""Request and response schemas for the prediction API."""
'@
    "app/model.py" = @'
"""Model loading and prediction logic."""
'@
    "tests/test_predict.py" = @'
"""Tests for the prediction endpoint."""
'@
}

foreach ($file in $starterFiles.Keys) {
    $path = Join-Path $root $file

    if (-not (Test-Path -LiteralPath $path)) {
        Set-Content -LiteralPath $path -Value $starterFiles[$file] -Encoding utf8
        Write-Host "Created $file" -ForegroundColor Green
    }
    else {
        Write-Host "Preserved existing $file"
    }
}

# Verify required root configuration files
foreach ($file in @("pyproject.toml", ".gitignore", "README.md")) {
    if (-not (Test-Path -LiteralPath (Join-Path $root $file))) {
        Write-Warning "Required root file is missing: $file"
    }
}

Write-Host "`nCurrent project tree:" -ForegroundColor Cyan
tree /A /F

Write-Host "`nRestructuring completed. Review the tree above." -ForegroundColor Green

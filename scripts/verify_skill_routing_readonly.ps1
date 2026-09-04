[CmdletBinding()]
param(
    [switch]$Quick,
    [switch]$KeepSandbox
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Get-RepositoryFingerprint {
    param([Parameter(Mandatory)][string]$Root)

    $relativeFiles = @(& git -C $Root ls-files -co --exclude-standard)
    if ($LASTEXITCODE -ne 0) {
        throw "git ls-files failed while fingerprinting $Root"
    }

    $builder = [Text.StringBuilder]::new()
    foreach ($relative in ($relativeFiles | Sort-Object -Unique)) {
        $fullPath = Join-Path $Root $relative
        $hash = if (Test-Path -LiteralPath $fullPath -PathType Leaf) {
            (Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash
        }
        else {
            "<missing>"
        }
        [void]$builder.Append($relative).Append("`0").Append($hash).Append("`n")
    }

    $bytes = [Text.Encoding]::UTF8.GetBytes($builder.ToString())
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($sha256.ComputeHash($bytes))).Replace("-", "")
    }
    finally {
        $sha256.Dispose()
    }
}

function Assert-LastExitCode {
    param([Parameter(Mandatory)][string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

$sourceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$temporaryRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$sandboxRoot = Join-Path $temporaryRoot ("corecoder-readonly-test-" + [guid]::NewGuid().ToString("N"))
$runtimeTemp = Join-Path $sandboxRoot ".runtime"
$originalLocation = (Get-Location).Path
$environmentNames = @(
    "CORECODER_MEMORY",
    "CORECODER_MEMORY_DIR",
    "CORECODER_SKILLS_DIR",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONPATH",
    "TEMP",
    "TMP"
)
$savedEnvironment = @{}
foreach ($name in $environmentNames) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

$beforeFingerprint = Get-RepositoryFingerprint -Root $sourceRoot
$testFailure = $null
$sourceChanged = $false

try {
    New-Item -ItemType Directory -Path $sandboxRoot | Out-Null
    Write-Host "[isolation] Copying the working tree to $sandboxRoot"
    & robocopy $sourceRoot $sandboxRoot /E /NFL /NDL /NJH /NJS /NP `
        /XD .git .venv venv __pycache__ .pytest_cache .mypy_cache .ruff_cache `
        /XF .env "*.pyc" | Out-Null
    if ($LASTEXITCODE -gt 7) {
        throw "robocopy failed with exit code $LASTEXITCODE"
    }

    New-Item -ItemType Directory -Path $runtimeTemp | Out-Null
    $env:CORECODER_MEMORY = "0"
    $env:CORECODER_MEMORY_DIR = Join-Path $runtimeTemp "memory"
    $env:CORECODER_SKILLS_DIR = Join-Path $runtimeTemp "user-skills"
    $env:PYTHONDONTWRITEBYTECODE = "1"
    $env:PYTHONPATH = ""
    $env:TEMP = $runtimeTemp
    $env:TMP = $runtimeTemp
    Set-Location $sandboxRoot

    Write-Host "[1/4] Static analysis (read-only source copy)"
    & python -m ruff check corecoder tests --no-cache
    Assert-LastExitCode -Step "Ruff"

    Write-Host "[2/4] Syntax/bytecode validation (writes only inside the copy)"
    & python -m compileall -q -f corecoder tests
    Assert-LastExitCode -Step "compileall"

    Write-Host "[3/4] Skill routing, security, and configuration regression tests"
    & python -m pytest -q tests/test_skills.py tests/test_security.py tests/test_config_env.py `
        -p no:cacheprovider --basetemp (Join-Path $runtimeTemp "pytest-focused")
    Assert-LastExitCode -Step "Focused pytest suite"

    if (-not $Quick) {
        Write-Host "[4/4] Full automated test suite"
        & python -m pytest -q -p no:cacheprovider `
            --basetemp (Join-Path $runtimeTemp "pytest-full")
        Assert-LastExitCode -Step "Full pytest suite"
    }
    else {
        Write-Host "[4/4] Full suite skipped because -Quick was supplied"
    }
}
catch {
    $testFailure = $_
}
finally {
    Set-Location $originalLocation
    foreach ($name in $environmentNames) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], "Process")
    }

    $afterFingerprint = Get-RepositoryFingerprint -Root $sourceRoot
    $sourceChanged = $beforeFingerprint -ne $afterFingerprint
    if ($sourceChanged) {
        Write-Host -ForegroundColor Red `
            "SOURCE TREE CHANGED: before=$beforeFingerprint after=$afterFingerprint"
    }
    else {
        Write-Host "[integrity] PASS: source fingerprint stayed $beforeFingerprint"
    }

    if ($KeepSandbox) {
        Write-Host "[cleanup] Sandbox retained at $sandboxRoot"
    }
    elseif (Test-Path -LiteralPath $sandboxRoot) {
        $resolvedSandbox = (Resolve-Path -LiteralPath $sandboxRoot).Path
        $expectedPrefix = $temporaryRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
        if (
            -not $resolvedSandbox.StartsWith($expectedPrefix, [StringComparison]::OrdinalIgnoreCase) -or
            -not (Split-Path -Leaf $resolvedSandbox).StartsWith("corecoder-readonly-test-")
        ) {
            throw "refusing to remove unexpected sandbox path: $resolvedSandbox"
        }
        Remove-Item -LiteralPath $resolvedSandbox -Recurse -Force
        Write-Host "[cleanup] Disposable sandbox removed"
    }
}

if ($sourceChanged) {
    throw "Read-only verification failed because the source tree changed"
}
if ($null -ne $testFailure) {
    throw $testFailure
}

Write-Host "PASS: all requested checks succeeded without modifying the source project"

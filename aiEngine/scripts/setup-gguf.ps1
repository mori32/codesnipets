param(
    [switch] $Force
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$modelDirectory = Join-Path $repositoryRoot "tests\fixtures\GGUF"
$fileName = "qwen2.5-0.5b-instruct-q8_0.gguf"
$destination = Join-Path $modelDirectory $fileName
$partial = "$destination.part"
$uri = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/${fileName}?download=true"
$expectedSha256 = "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e"

New-Item -ItemType Directory -Path $modelDirectory -Force | Out-Null

if ((Test-Path -LiteralPath $destination) -and -not $Force) {
    $existingHash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($existingHash -eq $expectedSha256) {
        Write-Host "GGUF already exists and matches SHA-256: $destination"
        exit 0
    }
    throw "Existing GGUF has an unexpected SHA-256. Use -Force to replace it: $destination"
}

# Windows PowerShell 5.1 may otherwise negotiate an obsolete TLS version.
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

try {
    if (Test-Path -LiteralPath $partial) {
        Remove-Item -LiteralPath $partial -Force
    }
    Write-Host "Downloading $uri"
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($null -ne $curl) {
        & $curl.Source --fail --location --retry 3 --retry-all-errors --retry-delay 2 `
            --user-agent "aiEngine-setup/1.0" --output $partial $uri
        if ($LASTEXITCODE -ne 0) {
            throw "curl failed with exit code $LASTEXITCODE"
        }
    }
    else {
        Invoke-WebRequest -Uri $uri -OutFile $partial -UseBasicParsing
    }

    $actualSha256 = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $expectedSha256) {
        throw "SHA-256 mismatch. Expected $expectedSha256, got $actualSha256"
    }

    Move-Item -LiteralPath $partial -Destination $destination -Force
    Write-Host "Installed verified GGUF: $destination"
}
finally {
    if (Test-Path -LiteralPath $partial) {
        Remove-Item -LiteralPath $partial -Force
    }
}

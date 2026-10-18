param(
    [string] $Ref = "v0.5.0",
    [string] $Generator = "Ninja",
    [ValidateSet("CPU", "CUDA")]
    [string] $Backend = "CPU",
    [switch] $Force,
    [switch] $SkipBuild
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Keep the runtime local to this checkout, but outside Git tracking. The source,
# build tree, install tree, and generated environment file are all ignored by
# .gitignore.
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$thirdPartyDirectory = Join-Path $repositoryRoot "third_party"
$sourceDirectory = Join-Path $thirdPartyDirectory "llama.cpp"
$backendSuffix = if ($Backend -eq "CUDA") { "-cuda" } else { "" }
$buildDirectory = Join-Path $thirdPartyDirectory "llama.cpp-build$backendSuffix"
$installDirectory = Join-Path $thirdPartyDirectory "llama.cpp-install$backendSuffix"
$archiveName = ".llama-cpp-$($Ref -replace '[^A-Za-z0-9._-]', '_').zip"
$archivePath = Join-Path $thirdPartyDirectory $archiveName
$partialArchivePath = "$archivePath.part"
$extractDirectory = Join-Path $thirdPartyDirectory ".llama-cpp-extract"
$environmentFile = Join-Path $thirdPartyDirectory "llama.cpp-env$backendSuffix.ps1"

function Get-ArchiveUri([string] $version) {
    if ($version.StartsWith("refs/", [StringComparison]::OrdinalIgnoreCase)) {
        return "https://codeload.github.com/ggml-org/llama.cpp/zip/$version"
    }
    if ($version.StartsWith("v", [StringComparison]::OrdinalIgnoreCase)) {
        return "https://codeload.github.com/ggml-org/llama.cpp/zip/refs/tags/$version"
    }
    return "https://codeload.github.com/ggml-org/llama.cpp/zip/refs/heads/$version"
}

function Find-VisualStudioRoot {
    $vswhereCandidates = @(
        "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe",
        "C:\Program Files\Microsoft Visual Studio\Installer\vswhere.exe"
    )
    $vswhere = $vswhereCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if ($null -ne $vswhere) {
        $installationPath = (& $vswhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath | Select-Object -First 1)
        if ($null -ne $installationPath) {
            $installationPath = $installationPath.Trim()
        }
        if ($installationPath -and (Test-Path -LiteralPath (Join-Path $installationPath "Common7\Tools\VsDevCmd.bat"))) {
            return $installationPath
        }
    }

    $fallback = Get-ChildItem -Path "C:\Program Files\Microsoft Visual Studio", `
        "C:\Program Files (x86)\Microsoft Visual Studio" -Filter "VsDevCmd.bat" -File -Recurse `
        -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $fallback) {
        return (Split-Path (Split-Path (Split-Path $fallback.FullName -Parent) -Parent) -Parent)
    }
    return $null
}

function Initialize-VisualStudioEnvironment([string] $installationPath) {
    $vsDevCmd = Join-Path $installationPath "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path -LiteralPath $vsDevCmd)) {
        return
    }

    $command = '"' + $vsDevCmd + '" -arch=x64 -host_arch=x64 && set'
    $environmentLines = & cmd.exe /d /s /c $command
    if ($LASTEXITCODE -ne 0) {
        throw "Visual Studio Developer environment initialization failed with exit code $LASTEXITCODE"
    }
    foreach ($line in $environmentLines) {
        $separator = $line.IndexOf("=", [StringComparison]::Ordinal)
        if ($separator -gt 0) {
            $name = $line.Substring(0, $separator)
            $value = $line.Substring($separator + 1)
            # cmd.exe can emit both PATH=... and Path=...; keep the
            # Developer Command Prompt value instead of overwriting it with
            # the original PowerShell value.  Set it explicitly because
            # PowerShell environment lookup is case-insensitive.
            if ($name -ceq "Path") {
                [Environment]::SetEnvironmentVariable("Path", $value, "Process")
                continue
            }
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
    Write-Host "Initialized Visual Studio Developer environment: $installationPath"
}

function ConvertTo-PowerShellString([string] $value) {
    return "'" + $value.Replace("'", "''") + "'"
}

function Invoke-Download([string] $uri, [string] $destination) {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Write-Host "Downloading llama.cpp $Ref"
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($null -ne $curl) {
        & $curl.Source --fail --location --retry 3 --retry-all-errors --retry-delay 2 `
            --user-agent "aiEngine-setup/1.0" --output $destination $uri
        if ($LASTEXITCODE -ne 0) {
            throw "curl failed with exit code $LASTEXITCODE"
        }
        return
    }
    Invoke-WebRequest -Uri $uri -OutFile $destination -UseBasicParsing
}

function Find-InstalledPath([string] $name) {
    $match = Get-ChildItem -LiteralPath $installDirectory -Recurse -File -Filter $name |
        Select-Object -First 1
    if ($null -eq $match) {
        throw "Could not find $name under $installDirectory after installing llama.cpp"
    }
    return $match.FullName
}

New-Item -ItemType Directory -Path $thirdPartyDirectory -Force | Out-Null

$visualStudioRoot = Find-VisualStudioRoot
if ($null -ne $visualStudioRoot) {
    Initialize-VisualStudioEnvironment $visualStudioRoot
}

$sourceHeader = Join-Path $sourceDirectory "include\llama.h"
if (-not (Test-Path -LiteralPath $sourceHeader)) {
    if ((Test-Path -LiteralPath $sourceDirectory) -and -not $Force) {
        throw "Existing llama.cpp source is incomplete. Remove it or rerun with -Force: $sourceDirectory"
    }

    if (Test-Path -LiteralPath $sourceDirectory) {
        Remove-Item -LiteralPath $sourceDirectory -Recurse -Force
    }
    if (Test-Path -LiteralPath $extractDirectory) {
        Remove-Item -LiteralPath $extractDirectory -Recurse -Force
    }
    if (Test-Path -LiteralPath $partialArchivePath) {
        Remove-Item -LiteralPath $partialArchivePath -Force
    }
    if (Test-Path -LiteralPath $archivePath) {
        Remove-Item -LiteralPath $archivePath -Force
    }

    $uri = Get-ArchiveUri $Ref
    try {
        Invoke-Download $uri $partialArchivePath
        # Expand-Archive validates the filename extension and accepts .zip,
        # but not the atomic-download suffix .zip.part.
        Move-Item -LiteralPath $partialArchivePath -Destination $archivePath -Force
        Expand-Archive -LiteralPath $archivePath -DestinationPath $extractDirectory -Force
        $extractedSource = Get-ChildItem -LiteralPath $extractDirectory -Directory |
            Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "include\llama.h") } |
            Select-Object -First 1
        if ($null -eq $extractedSource) {
            throw "Downloaded archive does not contain include\llama.h"
        }
        Move-Item -LiteralPath $extractedSource.FullName -Destination $sourceDirectory
        Write-Host "Installed llama.cpp source: $sourceDirectory"
    }
    finally {
        if (Test-Path -LiteralPath $partialArchivePath) {
            Remove-Item -LiteralPath $partialArchivePath -Force
        }
        if (Test-Path -LiteralPath $archivePath) {
            Remove-Item -LiteralPath $archivePath -Force
        }
        if (Test-Path -LiteralPath $extractDirectory) {
            Remove-Item -LiteralPath $extractDirectory -Recurse -Force
        }
    }
}
else {
    Write-Host "llama.cpp source already exists: $sourceDirectory"
}

if ($SkipBuild) {
    Write-Host "Download completed. Build was skipped by -SkipBuild: $sourceDirectory"
    exit 0
}

if (-not $SkipBuild) {
    $cmake = Get-Command cmake.exe -ErrorAction SilentlyContinue
    if ($null -eq $cmake) {
        throw "cmake.exe was not found. Install CMake and the MSVC C++ workload, then rerun this script."
    }
    if ($null -eq (Get-Command cl.exe -ErrorAction SilentlyContinue) -or
        $null -eq (Get-Command lib.exe -ErrorAction SilentlyContinue)) {
        throw "cl.exe/lib.exe were not found. Run this script from a Visual Studio Developer PowerShell or install the MSVC C++ workload."
    }
    if ($Generator -eq "Ninja" -and $null -eq (Get-Command ninja.exe -ErrorAction SilentlyContinue)) {
        throw "ninja.exe was not found. Install the CMake/Ninja component of Visual Studio or pass -Generator with another CMake generator."
    }

    $cudaCompiler = $null
    if ($Backend -eq "CUDA") {
        $cudaRoot = $env:CUDA_PATH
        if (-not $cudaRoot -or -not (Test-Path -LiteralPath $cudaRoot)) {
            throw "CUDA backend requested, but CUDA_PATH is not set to an installed NVIDIA CUDA Toolkit. Install the toolkit, open a new PowerShell, and rerun with -Backend CUDA."
        }
        $cudaCompiler = Join-Path $cudaRoot "bin\nvcc.exe"
        if (-not (Test-Path -LiteralPath $cudaCompiler)) {
            throw "CUDA backend requested, but nvcc.exe was not found at $cudaCompiler. Install a complete NVIDIA CUDA Toolkit and rerun."
        }
        $env:CUDACXX = $cudaCompiler
        $env:PATH = "$(Join-Path $cudaRoot 'bin');$env:PATH"
    }

    $cacheFile = Join-Path $buildDirectory "CMakeCache.txt"
    if (Test-Path -LiteralPath $cacheFile) {
        $configuredGenerator = Select-String -LiteralPath $cacheFile -Pattern '^CMAKE_GENERATOR:INTERNAL=' |
            Select-Object -First 1
        if ($null -ne $configuredGenerator) {
            $configuredGenerator = $configuredGenerator.Line.Substring($configuredGenerator.Line.IndexOf("=", [StringComparison]::Ordinal) + 1)
            if ($configuredGenerator -ne $Generator) {
                Write-Host "CMake generator changed from '$configuredGenerator' to '$Generator'; recreating the local build tree."
                Remove-Item -LiteralPath $buildDirectory -Recurse -Force
            }
        }
    }

    New-Item -ItemType Directory -Path $buildDirectory -Force | Out-Null
    New-Item -ItemType Directory -Path $installDirectory -Force | Out-Null

    # aiEngine links the llama library directly. Keep optional tools/server
    # disabled: cmake --install executes every generated install rule, not
    # only the target passed to cmake --build.
    $backendCmakeArgs = if ($Backend -eq "CUDA") {
        @("-DGGML_CUDA=ON", "-DGGML_VULKAN=OFF", "-DCMAKE_CUDA_COMPILER=$cudaCompiler")
    } else {
        @("-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF")
    }

    & $cmake.Source -G $Generator -S $sourceDirectory -B $buildDirectory `
        "-DCMAKE_INSTALL_PREFIX=$installDirectory" `
        "-DCMAKE_BUILD_TYPE=Release" `
        "-DBUILD_SHARED_LIBS=ON" `
        "-DGGML_NATIVE=ON" `
        @backendCmakeArgs `
        "-DLLAMA_BUILD_COMMON=OFF" `
        "-DLLAMA_BUILD_TESTS=OFF" `
        "-DLLAMA_BUILD_TOOLS=OFF" `
        "-DLLAMA_BUILD_EXAMPLES=OFF" `
        "-DLLAMA_BUILD_SERVER=OFF" `
        "-DLLAMA_BUILD_APP=OFF"
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed with exit code $LASTEXITCODE"
    }

    & $cmake.Source --build $buildDirectory --config Release --target llama --parallel
    if ($LASTEXITCODE -ne 0) {
        throw "llama.cpp build failed with exit code $LASTEXITCODE"
    }

    & $cmake.Source --install $buildDirectory --config Release
    if ($LASTEXITCODE -ne 0) {
        throw "llama.cpp install failed with exit code $LASTEXITCODE"
    }
}

$includePath = Find-InstalledPath "llama.h" | Split-Path -Parent
$libraryPath = Find-InstalledPath "llama.lib" | Split-Path -Parent
$runtime = Get-ChildItem -LiteralPath $installDirectory -Recurse -File -Filter "llama.dll" |
    Select-Object -First 1
$runtimePath = if ($null -ne $runtime) { $runtime.DirectoryName } else { $libraryPath }
$runtimePaths = @($runtimePath)
if ($Backend -eq "CUDA") {
    # CUDA 13.x places the DLLs used by llama.cpp in bin\x64 rather than
    # directly in bin. Ensure the generated runtime environment can load them.
    $runtimePaths += Join-Path $env:CUDA_PATH "bin\x64"
}
$runtimePaths += $env:PATH
$runtimeAndPath = $runtimePaths -join ";"

$environmentLines = @(
    "# Generated by scripts/setup-llama-cpp.ps1 for aiEngine.",
    "`$env:LLAMA_CPP_INCLUDE_DIR = $(ConvertTo-PowerShellString $includePath)",
    "`$env:LLAMA_CPP_LIB_DIR = $(ConvertTo-PowerShellString $libraryPath)",
    "`$env:LLAMA_CPP_LIB_NAME = 'llama'",
    "`$env:PATH = $(ConvertTo-PowerShellString $runtimeAndPath)"
)
if ($env:INCLUDE) {
    $environmentLines += "`$env:INCLUDE = $(ConvertTo-PowerShellString $env:INCLUDE)"
}
if ($env:LIB) {
    $environmentLines += "`$env:LIB = $(ConvertTo-PowerShellString $env:LIB)"
}
$environmentLines | Set-Content -LiteralPath $environmentFile -Encoding utf8

Write-Host ""
Write-Host "llama.cpp ($Backend backend) is ready. Load the Cargo environment with:"
Write-Host "  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass"
Write-Host "  . $environmentFile"
Write-Host "Then run:"
Write-Host "  cargo run -p ai_engine_cli --features llama-cpp -- --model tests\fixtures\GGUF\qwen2.5-0.5b-instruct-q8_0.gguf --context '庭で犬を' --dictionary examples\context-words.tsv --candidates 2"

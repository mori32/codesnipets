param(
    [string]$OutputDirectory = '',
    [string]$ModelIn = 'target/crf-wiki003-fixedseed-init-threads16-1k-20261003/initial-model.json',
    [ValidateRange(1,128)][int]$Threads = 8,
    [ValidateRange(1,256)][int]$NBest = 256,
    [ValidateRange(512,65536)][int]$MaxParameterMiB = 16384,
    [ValidateRange(1,100000)][int]$TestSampleSentences = 2000,
    [string]$EvaluationSampleIn = '',
    [ValidateRange(0.0,1.0)][double]$DeltaCutoff = 0,
    [ValidateRange(0.0,1.0)][double]$DeltaCutoffMax = 0.01,
    [ValidateRange(0,10000000)][int]$InitialBigramCount = 0,
    [ValidateRange(0.0,1.0)][double]$InitialBigramDelta = 0.9,
    [ValidateRange(0,10000000)][int]$InitialTrigramCount = 0,
    [ValidateRange(0.0,1.0)][double]$InitialTrigramWeight = 0.1,
    [switch]$BuildStaticInitialModel,
    [switch]$GoldPathAugmentation
)
$ErrorActionPreference = 'Stop'
if ($DeltaCutoff -gt $DeltaCutoffMax) { throw 'DeltaCutoff cannot exceed DeltaCutoffMax.' }
if ($EvaluationSampleIn) {
    throw 'EvaluationSampleIn is an evaluation-only option (requires epochs=0). Run a separate baseline/final evaluation with scripts/benchmark-crf.ps1.'
}
$crfRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $OutputDirectory) {
    $OutputDirectory = 'target/crf-wiki003-full-threads' + $Threads + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
}
$crfOutput = if ([IO.Path]::IsPathRooted($OutputDirectory)) {
    [IO.Path]::GetFullPath($OutputDirectory)
} else {
    [IO.Path]::GetFullPath((Join-Path $crfRoot $OutputDirectory))
}
if (-not $crfOutput.StartsWith($crfRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'CRF output must be inside this repository.'
}
$crfModel = if ($BuildStaticInitialModel) { $null } elseif ([IO.Path]::IsPathRooted($ModelIn)) { [IO.Path]::GetFullPath($ModelIn) } else { [IO.Path]::GetFullPath((Join-Path $crfRoot $ModelIn)) }
$crfSource = Join-Path $crfRoot 'corpus/wiki-ja/tagged-mecab-initial/jawiki-corpus-003.mecab-initial.jsonl.bz2'
$crfBuild = Join-Path $crfRoot 'target/release/ai_engine_crf.exe'
$crfFrozenExecutable = $crfOutput + '.exe'
$crfWrapperOut = $crfOutput + '.runner.stdout.log'
$crfWrapperErr = $crfOutput + '.runner.stderr.log'
$crfLaunchPath = $crfOutput + '.launch.json'
foreach ($crfInput in @($crfSource,$crfBuild)) {
    if (-not (Test-Path -LiteralPath $crfInput -PathType Leaf)) { throw "Input not found: $crfInput" }
}
if ($crfModel -and -not (Test-Path -LiteralPath $crfModel -PathType Leaf)) { throw "Initial model not found: $crfModel" }
if ($BuildStaticInitialModel -and $InitialBigramCount -eq 0 -and $InitialTrigramCount -eq 0) {
    throw 'BuildStaticInitialModel requires at least one nonzero initial n-gram count.'
}
if ($EvaluationSampleIn) {
    $crfEvaluationSample = if ([IO.Path]::IsPathRooted($EvaluationSampleIn)) {
        [IO.Path]::GetFullPath($EvaluationSampleIn)
    } else {
        [IO.Path]::GetFullPath((Join-Path $crfRoot $EvaluationSampleIn))
    }
    if (-not (Test-Path -LiteralPath $crfEvaluationSample -PathType Leaf)) { throw "Evaluation sample not found: $crfEvaluationSample" }
}
foreach ($crfPath in @($crfOutput,$crfFrozenExecutable,$crfWrapperOut,$crfWrapperErr,$crfLaunchPath,
    ($crfOutput + '.stdout.log'),($crfOutput + '.stderr.log'),($crfOutput + '.benchmark.json'))) {
    if (Test-Path -LiteralPath $crfPath) { throw "Output already exists: $crfPath" }
}
$crfFreeMiB = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024
# The learned table and a compaction replacement coexist briefly. With cutoff
# enabled, the replacement target is 75% of budget; otherwise it can be full size.
# Keep 4 GiB for the model, batch gradients, and runtime overhead. Checkpoint/TSV
# output now streams sparse maps and no longer materializes sorted table copies.
# Cutoff reuses the existing HashMap buckets after deleting weak entries; it
# no longer needs a second table-sized allocation. Keep 4 GiB for model, batch,
# page cache, and runtime overhead in addition to the learned-table budget.
$crfRequiredFreeMiB = [Math]::Ceiling($MaxParameterMiB + 4096)
if ($crfFreeMiB -lt $crfRequiredFreeMiB) {
    throw ("Not enough free RAM: free={0:N0} MiB, required={1:N0} MiB for a {2} MiB table budget." -f $crfFreeMiB,$crfRequiredFreeMiB,$MaxParameterMiB)
}
[IO.Directory]::CreateDirectory((Split-Path -Parent $crfOutput)) | Out-Null
Copy-Item -LiteralPath $crfBuild -Destination $crfFrozenExecutable -ErrorAction Stop
$crfArguments = @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"' + (Join-Path $PSScriptRoot 'benchmark-crf.ps1') + '"'),
    '-OutputDirectory', ('"' + $crfOutput + '"'),
    '-Executable', ('"' + $crfFrozenExecutable + '"'), '-FullCorpus', '-Epochs', 1,
    '-Threads', $Threads, '-BatchSize', 16, '-NBest', $NBest, '-PageBlockSize', 1000,
    '-ValidationSentences', $TestSampleSentences, '-EvaluationSentences', 1000,
    '-MaxParameterMiB', $MaxParameterMiB, '-DeltaCutoff', $DeltaCutoff,
    '-DeltaCutoffMax', $DeltaCutoffMax, '-InitialBigramCount', $InitialBigramCount,
    '-InitialBigramDelta', $InitialBigramDelta, '-InitialTrigramCount', $InitialTrigramCount,
    '-InitialTrigramWeight', $InitialTrigramWeight, '-SaveInitialModel'
)
if ($crfModel) { $crfArguments += @('-ModelIn', ('"' + $crfModel + '"')) }
if ($GoldPathAugmentation) { $crfArguments += '-GoldPathAugmentation' }
if ($EvaluationSampleIn) { $crfArguments += @('-EvaluationSampleIn', ('"' + $crfEvaluationSample + '"')) }
$crfPowerShell = Get-Command pwsh.exe -ErrorAction SilentlyContinue
if (-not $crfPowerShell) { $crfPowerShell = Get-Command powershell.exe -ErrorAction Stop }
$crfProcess = Start-Process -FilePath $crfPowerShell.Source -ArgumentList $crfArguments `
    -WorkingDirectory $crfRoot -WindowStyle Hidden -RedirectStandardOutput $crfWrapperOut `
    -RedirectStandardError $crfWrapperErr -PassThru
$crfProcessHandle = $crfProcess.Handle # Preserve the handle so the exit status is readable after an early exit.
$crfLaunch = [ordered]@{
    started_at=(Get-Date).ToString('o'); runner_pid=$crfProcess.Id; output=$crfOutput
    input=$crfSource; source_bytes=(Get-Item -LiteralPath $crfSource).Length
    model_in=$crfModel; initial_model_sha256=if ($crfModel) { (Get-FileHash -LiteralPath $crfModel -Algorithm SHA256).Hash } else { $null }
    executable=$crfFrozenExecutable; executable_sha256=(Get-FileHash -LiteralPath $crfFrozenExecutable -Algorithm SHA256).Hash
    threads=$Threads; nbest=$NBest; batch_size=16; page_block_size=1000; epochs=1
    full_corpus=$true; parameter_budget_mib=$MaxParameterMiB; free_memory_mib_at_launch=$crfFreeMiB
    estimated_peak_reserve_mib=$crfRequiredFreeMiB
    delta_cutoff=$DeltaCutoff; delta_cutoff_max=$DeltaCutoffMax
    initial_bigram_count=$InitialBigramCount; initial_bigram_delta=$InitialBigramDelta
    initial_trigram_count=$InitialTrigramCount; initial_trigram_weight=$InitialTrigramWeight
    build_static_initial_model=[bool]$BuildStaticInitialModel
    gold_path_augmentation=[bool]$GoldPathAugmentation
    evaluation_sample_in=if ($EvaluationSampleIn) { $crfEvaluationSample } else { $null }
    heldout_rule='fixed page-ID hash modulo 5 == 0; about 20%; no heldout CRF updates'
    test_sample_sentences=$TestSampleSentences
    stderr_log=($crfOutput + '.stderr.log'); runner_stdout_log=$crfWrapperOut
    note='Runs independently of the initiating terminal. When BuildStaticInitialModel is set, the CRF builds initial n-gram corrections from the configured statistics before scanning chunk003, and saves initial-model.json. No LLM/server.'
}
$crfLaunchStream = [IO.File]::Open($crfLaunchPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
$crfLaunchWriter = [IO.StreamWriter]::new($crfLaunchStream,[Text.UTF8Encoding]::new($false))
try { $crfLaunchWriter.WriteLine(($crfLaunch | ConvertTo-Json -Depth 4)) } finally { $crfLaunchWriter.Dispose() }
$crfLaunch | ConvertTo-Json -Depth 4
Start-Sleep -Seconds 1
$crfProcess.Refresh()
if ($crfProcess.HasExited) {
    $crfProcess.WaitForExit()
    $crfProcess.Refresh()
    if ($crfProcess.ExitCode -ne 0) {
        Get-Content -LiteralPath $crfWrapperErr -Tail 10
        throw "CRF runner failed at startup (exit $($crfProcess.ExitCode), handle $crfProcessHandle); see $crfWrapperErr"
    }
}

param(
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [int]$Sentences = 1000,
    [int]$NBest = 256,
    [int]$Epochs = 1,
    [int]$ShuffleBuffer = 256,
    [int]$MaxParameterMiB = 512,
    [double]$DeltaCutoff = 0,
    [double]$DeltaCutoffMax = 0.01,
    [int]$InitialBigramCount = 0,
    [double]$InitialBigramDelta = 0.9,
    [int]$InitialTrigramCount = 0,
    [double]$InitialTrigramWeight = 0.1,
    [int]$ProgressInterval = 250000,
    [int]$Threads = 16,
    [int]$BatchSize = 16,
    [int]$PageBlockSize = 1000,
    [int]$ValidationSentences = 200,
    [int]$EvaluationSentences = 1000,
    [string]$Executable = '',
    [string]$ModelIn = '',
    [string]$EvaluationSampleIn = '',
    [string]$ResumeFrom = '',
    [switch]$SaveInitialModel,
    [switch]$ReferenceKernels,
    [switch]$GoldPathAugmentation,
    [switch]$Hogwild,
    [switch]$FullCorpus
)
$ErrorActionPreference = 'Stop'
$crfRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $Executable) { $Executable = Join-Path $crfRoot 'target/release/ai_engine_crf.exe' }
$crfExecutable = [IO.Path]::GetFullPath($Executable)
if (-not (Test-Path -LiteralPath $crfExecutable -PathType Leaf)) { throw "Executable not found: $crfExecutable" }
$crfExecutableHash = (Get-FileHash -LiteralPath $crfExecutable -Algorithm SHA256).Hash
$crfOutput = if ([IO.Path]::IsPathRooted($OutputDirectory)) {
    [IO.Path]::GetFullPath($OutputDirectory)
} else {
    [IO.Path]::GetFullPath((Join-Path $crfRoot $OutputDirectory))
}
if (-not $crfOutput.StartsWith($crfRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Benchmark output must be inside this repository.'
}
$crfLogOut = $crfOutput + '.stdout.log'
$crfLogErr = $crfOutput + '.stderr.log'
$crfStatsPath = $crfOutput + '.benchmark.json'
foreach ($crfPath in @($crfOutput,$crfLogOut,$crfLogErr,$crfStatsPath)) {
    if (Test-Path -LiteralPath $crfPath) { throw "Output already exists: $crfPath" }
}
$crfParent = Split-Path -Parent $crfOutput
[IO.Directory]::CreateDirectory($crfParent) | Out-Null
$crfArguments = @(
    '--input', ('"' + (Join-Path $crfRoot 'corpus/wiki-ja/tagged-mecab-initial/jawiki-corpus-003.mecab-initial.jsonl.bz2') + '"'),
    '--corpus-statistics', ('"' + (Join-Path $crfRoot 'target/dictionary-build-userdic-20261003-run01/global') + '"'),
    '--user-additions', ('"' + (Join-Path $crfRoot 'dictionary/additions') + '"'),
    '--output', ('"' + $crfOutput + '"'),
    '--epochs', $Epochs, '--max-sentences', $Sentences,
    '--max-validation-sentences', $ValidationSentences, '--max-eval-sentences', $EvaluationSentences,
    '--nbest', $NBest, '--max-records', ([Math]::Max(20000,$Sentences * 10)),
    '--shuffle-buffer', $ShuffleBuffer, '--max-parameter-mib', $MaxParameterMiB,
    '--threads', $Threads, '--batch-size', $BatchSize, '--page-block-size', $PageBlockSize
)
$crfInvariantCulture = [Globalization.CultureInfo]::InvariantCulture
$crfArguments += @('--delta-cutoff', $DeltaCutoff.ToString('R', $crfInvariantCulture),
    '--delta-cutoff-max', $DeltaCutoffMax.ToString('R', $crfInvariantCulture))
$crfArguments += @('--initial-bigram-count', $InitialBigramCount,
    '--initial-bigram-delta', $InitialBigramDelta.ToString('R', $crfInvariantCulture),
    '--initial-trigram-count', $InitialTrigramCount,
    '--initial-trigram-weight', $InitialTrigramWeight.ToString('R', $crfInvariantCulture),
    '--progress-interval', $ProgressInterval)
if ($ModelIn) {
    if ([IO.Path]::IsPathRooted($ModelIn)) {
        $crfModelIn = [IO.Path]::GetFullPath($ModelIn)
    } else {
        $crfModelIn = [IO.Path]::GetFullPath((Join-Path $crfRoot $ModelIn))
    }
    if (-not (Test-Path -LiteralPath $crfModelIn -PathType Leaf)) { throw "Model not found: $crfModelIn" }
    $crfArguments += @('--model-in', ('"' + $crfModelIn + '"'))
}
if ($EvaluationSampleIn) {
    $crfEvaluationSample = if ([IO.Path]::IsPathRooted($EvaluationSampleIn)) {
        [IO.Path]::GetFullPath($EvaluationSampleIn)
    } else {
        [IO.Path]::GetFullPath((Join-Path $crfRoot $EvaluationSampleIn))
    }
    if (-not (Test-Path -LiteralPath $crfEvaluationSample -PathType Leaf)) { throw "Evaluation sample not found: $crfEvaluationSample" }
    $crfArguments += @('--evaluation-sample-in', ('"' + $crfEvaluationSample + '"'))
}
if ($ResumeFrom) {
    if ($ModelIn) { throw 'ResumeFrom cannot be combined with ModelIn.' }
    $crfResumePath = if ([IO.Path]::IsPathRooted($ResumeFrom)) { [IO.Path]::GetFullPath($ResumeFrom) }
        else { [IO.Path]::GetFullPath((Join-Path $crfRoot $ResumeFrom)) }
    $crfArguments += @('--resume-from', ('"' + $crfResumePath + '"'))
}
if ($SaveInitialModel) { $crfArguments += '--save-initial-model' }
if ($ReferenceKernels) { $crfArguments += '--reference-kernels' }
if ($GoldPathAugmentation) { $crfArguments += '--gold-path-augmentation' }
if ($Hogwild) { $crfArguments += '--hogwild' }
if ($FullCorpus) { $crfArguments += '--full-corpus' }
$crfTimer = [Diagnostics.Stopwatch]::StartNew()
$crfProcess = Start-Process -FilePath $crfExecutable `
    -ArgumentList $crfArguments -WorkingDirectory $crfRoot -WindowStyle Hidden `
    -RedirectStandardOutput $crfLogOut -RedirectStandardError $crfLogErr -PassThru
$crfProcessHandle = $crfProcess.Handle # Retain the handle so Windows PowerShell can read ExitCode after exit.
$crfPeakWorkingSet = [long]0
$crfPeakPrivate = [long]0
$crfSamples = 0
$crfLastNotice = 0
$crfCpuSeconds = [double]0
$crfPeakThreadCount = 0
while (-not $crfProcess.HasExited) {
    $crfProcess.Refresh()
    if ($crfProcess.HasExited) { break }
    $crfPeakWorkingSet = [Math]::Max($crfPeakWorkingSet,$crfProcess.PeakWorkingSet64)
    $crfPeakPrivate = [Math]::Max($crfPeakPrivate,$crfProcess.PrivateMemorySize64)
    $crfSamples++
    $crfCpuSeconds = [Math]::Max($crfCpuSeconds,$crfProcess.TotalProcessorTime.TotalSeconds)
    $crfPeakThreadCount = [Math]::Max($crfPeakThreadCount,$crfProcess.Threads.Count)
    if ($crfTimer.Elapsed.TotalSeconds - $crfLastNotice -ge 10) {
        Write-Output ("elapsed={0:N1}s peak_working_set={1:N1}MiB private={2:N1}MiB" -f `
            $crfTimer.Elapsed.TotalSeconds, ($crfPeakWorkingSet / 1MB), ($crfProcess.PrivateMemorySize64 / 1MB))
        Get-Content -LiteralPath $crfLogErr -Tail 2
        $crfLastNotice = $crfTimer.Elapsed.TotalSeconds
    }
    Start-Sleep -Milliseconds 500
}
$crfProcess.WaitForExit()
$crfExitCode = $crfProcess.ExitCode
if ($null -eq $crfExitCode) { throw "Process exit code unavailable (handle $crfProcessHandle); do not treat this as success." }
$crfTimer.Stop()
$crfStats = [ordered]@{
    sentences_requested=$Sentences; nbest=$NBest; epochs=$Epochs; shuffle_buffer=$ShuffleBuffer
    threads=$Threads; batch_size=$BatchSize; page_block_size=$PageBlockSize
    model_in=$ModelIn; save_initial_model=[bool]$SaveInitialModel
    evaluation_sample_in=$EvaluationSampleIn; gold_path_augmentation=[bool]$GoldPathAugmentation
    resume_from=$ResumeFrom
    full_corpus=[bool]$FullCorpus; validation_sample_capacity=$ValidationSentences
    reference_kernels=[bool]$ReferenceKernels
    evaluation_sample_capacity=$EvaluationSentences
    parameter_budget_mib=$MaxParameterMiB; parameter_budget_enforced=[bool](-not $Hogwild); exit_code=$crfExitCode
    delta_cutoff=$DeltaCutoff; delta_cutoff_max=$DeltaCutoffMax
    initial_bigram_count=$InitialBigramCount; initial_bigram_delta=$InitialBigramDelta
    initial_trigram_count=$InitialTrigramCount; initial_trigram_weight=$InitialTrigramWeight
    progress_interval=$ProgressInterval
    elapsed_seconds=$crfTimer.Elapsed.TotalSeconds
    peak_working_set_bytes=$crfPeakWorkingSet; sampled_peak_private_bytes=$crfPeakPrivate
    samples=$crfSamples; executable=$crfExecutable
    executable_sha256=$crfExecutableHash
    sampled_cpu_seconds=$crfCpuSeconds; peak_process_thread_count=$crfPeakThreadCount
    output=$crfOutput; note='Windows process working set includes checkpoint loading/export; private peak is sampled every 500ms.'
}
$crfStatsStream = [IO.File]::Open($crfStatsPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
$crfStatsWriter = [IO.StreamWriter]::new($crfStatsStream,[Text.UTF8Encoding]::new($false))
try { $crfStatsWriter.WriteLine(($crfStats | ConvertTo-Json -Depth 4)) } finally { $crfStatsWriter.Dispose() }
$crfStats | ConvertTo-Json -Depth 4
Get-Content -LiteralPath $crfLogErr -Tail 4
Get-Content -LiteralPath $crfLogOut
if ($crfExitCode -ne 0) { exit $crfExitCode }

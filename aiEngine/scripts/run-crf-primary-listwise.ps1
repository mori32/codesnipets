param(
    [Parameter(Mandatory=$true)][string]$RunBase,
    [Parameter(Mandatory=$true)][string]$WorkBase,
    [Parameter(Mandatory=$true)][string]$InitialModel
)

$ErrorActionPreference = 'Stop'
$crfRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$RunBase = [IO.Path]::GetFullPath($RunBase)
$WorkBase = [IO.Path]::GetFullPath($WorkBase)
$InitialModel = [IO.Path]::GetFullPath($InitialModel)
$inputFile = Join-Path $crfRoot 'corpus\wiki-ja\tagged-mecab-initial\jawiki-corpus-003.mecab-initial.jsonl.bz2'
$builtExe = Join-Path $crfRoot 'target\release\ai_engine_crf.exe'
$modelOutput = Join-Path $RunBase 'model'
$workDir = Join-Path $WorkBase 'work'
$frozenExe = Join-Path $RunBase 'ai_engine_crf.exe'
$stdoutLog = Join-Path $RunBase 'stdout.log'
$stderrLog = Join-Path $RunBase 'stderr.log'
$runnerLog = Join-Path $RunBase 'runner.log'
$launchPath = Join-Path $RunBase 'launch.json'
$benchmarkPath = Join-Path $RunBase 'benchmark.json'

foreach ($path in @($inputFile,$builtExe,$InitialModel)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Input not found: $path" }
}
foreach ($path in @($RunBase,$WorkBase)) {
    if (Test-Path -LiteralPath $path) { throw "Refusing to reuse existing run/work path: $path" }
}
$freeMiB = [double](Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024
$requiredMiB = 24576 + 4096
if ($freeMiB -lt $requiredMiB) { throw "Not enough free RAM: ${freeMiB} MiB available, ${requiredMiB} MiB required." }

[IO.Directory]::CreateDirectory((Split-Path -Parent $RunBase)) | Out-Null
[IO.Directory]::CreateDirectory((Split-Path -Parent $WorkBase)) | Out-Null
[IO.Directory]::CreateDirectory($RunBase) | Out-Null
[IO.Directory]::CreateDirectory($modelOutput) | Out-Null
[IO.Directory]::CreateDirectory($WorkBase) | Out-Null
[IO.Directory]::CreateDirectory($workDir) | Out-Null
Copy-Item -LiteralPath $builtExe -Destination $frozenExe

$sourceBytes = (Get-Item -LiteralPath $inputFile).Length
$initialHash = (Get-FileHash -LiteralPath $InitialModel -Algorithm SHA256).Hash
$exeHash = (Get-FileHash -LiteralPath $frozenExe -Algorithm SHA256).Hash
$rootReadme = @"
# CRF chunk003 primary-listwise run

- Status: preparing; see `launch.json`, `model\` and `stderr.log`.
- Source: `$inputFile` (the entire chunk003 file only).
- Initial model: `$InitialModel` (SHA-256 `$initialHash`); no learned weights/checkpoints are reused.
- Model output: `$modelOutput`.
- Work/cache/recovery: `$workDir`; normalized afresh against the selected refined POS IDs. No previous page cache is reused.
- Settings: 16 threads, batch 16, N=128, at most 2 epochs; epoch 2 uses epoch-1 final post-trigram top-1 misses; primary and trigram listwise, gold-path augmentation and missed feedback; LR .03, L2 0; learned table budget 24,576 MiB; word-bi/tri entry count caps 50m/100m trigger then retain half.
- Fixed evaluation: first 5,000 accepted normalized chunk003 sentences; reports include character-level accuracy.
- Deletion: derived experiment output. Keep while running or needed for recovery/result review. Source corpus and initial model are separate.
"@
Set-Content -LiteralPath (Join-Path $RunBase 'README.md') -Value $rootReadme -Encoding utf8
$workReadme = @"
# CRF chunk003 work data

- Status: in progress after the trainer creates its input report.
- Source: `$inputFile`; initial model: `$InitialModel`.
- Contents: fresh page-block cache, recovery snapshots, and epoch-specific `missed-chunk003` records.
- Previous run caches/checkpoints are not reused.
- Deletion: derived and regenerable; preserve until this run is complete and no recovery is needed.
"@
Set-Content -LiteralPath (Join-Path $WorkBase 'README.md') -Value $workReadme -Encoding utf8

$crfArguments = @(
    '--input', ('"' + $inputFile + '"'),
    '--model-in', ('"' + $InitialModel + '"'),
    '--output', ('"' + $modelOutput + '"'),
    '--work-dir', ('"' + $workDir + '"'),
    '--epochs', '2', '--full-corpus', '--threads', '16', '--batch-size', '16',
    '--nbest', '128', '--max-words', '48', '--max-nodes', '20000',
    '--max-edges', '1000000', '--max-astar-expansions', '500000',
    '--max-validation-sentences', '5000', '--max-eval-sentences', '1000',
    '--test-prefix-sentences', '5000', '--page-block-size', '1000',
    '--shuffle-buffer', '256', '--seed', '20261003',
    '--learning-rate', '0.03', '--l2', '0', '--max-parameter-mib', '24576',
    '--delta-cutoff', '0.0001', '--delta-cutoff-max', '0.01',
    '--recovery-interval', '250000', '--progress-interval', '250000',
    '--progress-interval-seconds', '1800', '--primary-nbest-listwise',
    '--gold-path-augmentation', '--missed-feedback', '--save-initial-model'
)
$process = Start-Process -FilePath $frozenExe -ArgumentList $crfArguments -WorkingDirectory $crfRoot `
    -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru
$handle = $process.Handle
$launch = [ordered]@{
    run_id = Split-Path -Leaf $RunBase
    status = 'running'
    started_at = (Get-Date).ToString('o')
    runner_pid = $PID
    crf_pid = $process.Id
    input = $inputFile
    input_bytes = $sourceBytes
    initial_model = $InitialModel
    initial_model_sha256 = $initialHash
    executable = $frozenExe
    executable_sha256 = $exeHash
    model_output = $modelOutput
    work = $workDir
    previous_normalized_cache_reused = $false
    previous_checkpoint_reused = $false
    threads = 16
    batch_size = 16
    nbest = 128
    epochs_max = 2
    epoch_2_missed_feedback_only = $true
    primary_objective = 'raw-nbest-listwise'
    word_bigram_and_trigram_batch_reduction = 'sum; no 1/16'
    other_primary_and_reading_length_trigram_reduction = 'mean over their defined eligible sentences'
    learning_rate = 0.03
    l2 = 0.0
    delta_cutoff = 0.0001
    delta_cutoff_max = 0.01
    max_parameter_mib = 24576
    word_bigram_count_trigger_keep_half = 50000000
    word_trigram_count_trigger_keep_half = 100000000
    test_prefix_sentences = 5000
    full_chunk003 = $true
    free_memory_mib_at_launch = [Math]::Round($freeMiB)
    stdout = $stdoutLog
    stderr = $stderrLog
    runner_log = $runnerLog
    note = 'Fresh normalization uses the new model-local word/POS IDs. Only chunk003 is read.'
}
$launch | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $launchPath -Encoding utf8
Add-Content -LiteralPath $runnerLog -Value ("started run_id={0} pid={1} free_mib={2:N0}" -f $launch.run_id,$process.Id,$freeMiB)
Set-Content -LiteralPath (Join-Path $RunBase 'README.md') -Value ($rootReadme -replace 'Status: preparing', 'Status: running') -Encoding utf8

$timer = [Diagnostics.Stopwatch]::StartNew()
$peakWorkingSet = [long]0
$peakPrivate = [long]0
$cpuSeconds = [double]0
$peakThreads = 0
$samples = 0
while (-not $process.HasExited) {
    $process.Refresh()
    if ($process.HasExited) { break }
    $peakWorkingSet = [Math]::Max($peakWorkingSet,$process.PeakWorkingSet64)
    $peakPrivate = [Math]::Max($peakPrivate,$process.PrivateMemorySize64)
    $cpuSeconds = [Math]::Max($cpuSeconds,$process.TotalProcessorTime.TotalSeconds)
    $peakThreads = [Math]::Max($peakThreads,$process.Threads.Count)
    $samples++
    Add-Content -LiteralPath $runnerLog -Value ("elapsed_seconds={0:N0} working_set_mib={1:N0} private_mib={2:N0} cpu_seconds={3:N0} threads={4}" -f `
        $timer.Elapsed.TotalSeconds,($process.WorkingSet64/1MB),($process.PrivateMemorySize64/1MB),$cpuSeconds,$process.Threads.Count)
    Start-Sleep -Seconds 30
}
$process.WaitForExit()
$exitCode = $process.ExitCode
$timer.Stop()
$benchmark = [ordered]@{
    run_id = $launch.run_id
    exit_code = $exitCode
    elapsed_seconds = $timer.Elapsed.TotalSeconds
    peak_working_set_bytes = $peakWorkingSet
    peak_sampled_private_bytes = $peakPrivate
    sampled_cpu_seconds = $cpuSeconds
    peak_process_thread_count = $peakThreads
    samples = $samples
    executable_sha256 = $exeHash
    completed_at = (Get-Date).ToString('o')
}
$benchmark | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $benchmarkPath -Encoding utf8
$launch.status = if ($exitCode -eq 0) { 'process-exit-0; verification pending' } else { 'failed' }
$launch.exit_code = $exitCode
$launch.completed_at = (Get-Date).ToString('o')
$launch | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $launchPath -Encoding utf8
Add-Content -LiteralPath $runnerLog -Value ("finished exit_code={0} elapsed_seconds={1:N1}" -f $exitCode,$timer.Elapsed.TotalSeconds)
if ($exitCode -eq 0) {
    Set-Content -LiteralPath (Join-Path $RunBase 'README.md') -Value ($rootReadme -replace 'Status: preparing', 'Status: process exited zero; inspect model/verification.json before declaring completion') -Encoding utf8
} else {
    Set-Content -LiteralPath (Join-Path $RunBase 'README.md') -Value ($rootReadme -replace 'Status: preparing', "Status: failed with exit code $exitCode; inspect stderr.log and latest committed recovery") -Encoding utf8
}
if ($exitCode -ne 0) { exit $exitCode }

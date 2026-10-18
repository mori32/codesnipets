param(
    [string] $AdditionDirectory = "dictionary\additions",
    [Parameter(Mandatory = $true)]
    [string] $SystemDictionary,
    [string] $OutputDictionary = "dictionary\build\user.dic",
    [string] $MecabDictIndex = "",
    [switch] $KeepMergedCsv
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repositoryRoot = Split-Path -Parent $PSScriptRoot
function Resolve-RepositoryPath([string] $path) {
    if ([IO.Path]::IsPathRooted($path)) { return $path }
    return Join-Path $repositoryRoot $path
}

$additionPath = Resolve-RepositoryPath $AdditionDirectory
$systemDictionaryPath = Resolve-RepositoryPath $SystemDictionary
$outputPath = Resolve-RepositoryPath $OutputDictionary
if (-not (Test-Path -LiteralPath $additionPath -PathType Container)) {
    throw "Addition directory was not found: $additionPath"
}
if (-not (Test-Path -LiteralPath $systemDictionaryPath -PathType Container)) {
    throw "System dictionary directory was not found: $systemDictionaryPath"
}

$columns = @(
    "surface", "left_id", "right_id", "cost",
    "pos1", "pos2", "pos3", "pos4", "pos5", "pos6",
    "base", "reading", "pronunciation"
)

function Get-Field([object] $row, [string] $name) {
    $property = $row.PSObject.Properties[$name]
    if ($null -eq $property) {
        throw "Addition CSV is missing '$name'"
    }
    return [string] $property.Value
}

function Get-AdditionKey([object] $row) {
    return (@(
        (Get-Field $row "surface"), (Get-Field $row "reading"),
        (Get-Field $row "pos1"), (Get-Field $row "pos2"),
        (Get-Field $row "pos3"), (Get-Field $row "pos4"),
        (Get-Field $row "pos5"), (Get-Field $row "pos6")
    ) -join ([char]0x1f))
}

$deduplicated = [Collections.Generic.List[object]]::new()
$seen = @{}
$inputFiles = @(
    Get-ChildItem -LiteralPath $additionPath -Filter "*-addition.csv" -File
    Get-ChildItem -LiteralPath $additionPath -Filter "*-new-words.txt" -File
) | Sort-Object Name
foreach ($inputFile in $inputFiles) {
    if ($inputFile.Name.EndsWith("-new-words.txt", [StringComparison]::OrdinalIgnoreCase)) {
        $fileRows = [Collections.Generic.List[object]]::new()
        foreach ($line in [IO.File]::ReadLines($inputFile.FullName, [Text.Encoding]::UTF8)) {
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            $values = $line.Split(',')
            if ($values.Count -ne $columns.Count) {
                throw "New-words row in '$($inputFile.Name)' has $($values.Count) fields; expected $($columns.Count)."
            }
            $row = [ordered]@{}
            for ($index = 0; $index -lt $columns.Count; $index++) {
                if ($values[$index].Contains('"') -or $values[$index].Contains("`r") -or $values[$index].Contains("`n")) {
                    throw "New-words row in '$($inputFile.Name)' contains an unsupported quote or line break."
                }
                $row[$columns[$index]] = $values[$index]
            }
            $fileRows.Add([pscustomobject]$row)
        }
    }
    else {
        $fileRows = @(Import-Csv -LiteralPath $inputFile.FullName)
    }
    foreach ($row in $fileRows) {
        foreach ($column in $columns) { $null = Get-Field $row $column }
        $key = Get-AdditionKey $row
        if ($seen.ContainsKey($key)) { continue }
        $seen[$key] = $true
        $deduplicated.Add($row)
    }
}
if ($deduplicated.Count -eq 0) {
    throw "No addition rows were found under $additionPath"
}

$outputDirectoryPath = Split-Path -Parent $outputPath
New-Item -ItemType Directory -Path $outputDirectoryPath -Force | Out-Null
$mergedPath = Join-Path $outputDirectoryPath "user-merged.csv"
$temporaryMergedPath = "$mergedPath.part"
$temporaryDictionaryPath = "$outputPath.part"
$temporaryBackupPath = "$temporaryDictionaryPath.previous"

try {
    # MeCab source CSV has no header. The dated addition files do have a
    # header so they remain self-describing when reviewed independently.
    # MeCab's CSV reader does not treat RFC-style quotes as CSV syntax; quotes
    # would become part of the feature strings. Serialize its input explicitly.
    $csvLines = foreach ($row in $deduplicated) {
        $values = foreach ($column in $columns) {
            $value = Get-Field $row $column
            if ($value.Contains(',') -or $value.Contains('"') -or
                $value.Contains("`r") -or $value.Contains("`n")) {
                throw "MeCab CSV field '$column' contains an unsupported delimiter or line break."
            }
            $value
        }
        [string]::Join(',', [string[]] $values)
    }
    # MeCab's dictionary CSV reader treats a UTF-8 BOM as part of the first
    # surface. PowerShell 5.1's `-Encoding utf8` adds that BOM, so write the
    # intermediate explicitly as UTF-8 without a BOM.
    [IO.File]::WriteAllLines(
        $temporaryMergedPath,
        [string[]] $csvLines,
        [Text.UTF8Encoding]::new($false)
    )
    Move-Item -LiteralPath $temporaryMergedPath -Destination $mergedPath -Force

    $indexPath = $MecabDictIndex
    if ([string]::IsNullOrWhiteSpace($indexPath)) {
        $command = Get-Command mecab-dict-index.exe -ErrorAction SilentlyContinue
        if ($null -eq $command) { $command = Get-Command mecab-dict-index -ErrorAction SilentlyContinue }
        if ($null -eq $command) {
            throw "mecab-dict-index was not found. Pass -MecabDictIndex explicitly."
        }
        $indexPath = $command.Source
    }
    else {
        $indexPath = Resolve-RepositoryPath $indexPath
    }
    if (-not (Test-Path -LiteralPath $indexPath -PathType Leaf)) {
        throw "mecab-dict-index was not found: $indexPath"
    }

    & $indexPath -d $systemDictionaryPath -u $temporaryDictionaryPath `
        -f UTF-8 -t UTF-8 $mergedPath
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $temporaryDictionaryPath -PathType Leaf)) {
        throw "mecab-dict-index failed with exit code $LASTEXITCODE"
    }

    if (Test-Path -LiteralPath $outputPath -PathType Leaf) {
        [IO.File]::Replace($temporaryDictionaryPath, $outputPath, $temporaryBackupPath, $true)
        if (Test-Path -LiteralPath $temporaryBackupPath) {
            Remove-Item -LiteralPath $temporaryBackupPath -Force
        }
    }
    else {
        Move-Item -LiteralPath $temporaryDictionaryPath -Destination $outputPath
    }
}
finally {
    if (Test-Path -LiteralPath $temporaryMergedPath) { Remove-Item -LiteralPath $temporaryMergedPath -Force }
    if (Test-Path -LiteralPath $temporaryDictionaryPath) { Remove-Item -LiteralPath $temporaryDictionaryPath -Force }
    if (Test-Path -LiteralPath $temporaryBackupPath) { Remove-Item -LiteralPath $temporaryBackupPath -Force }
    if (-not $KeepMergedCsv -and (Test-Path -LiteralPath $mergedPath)) { Remove-Item -LiteralPath $mergedPath -Force }
}

$manifestPath = "$outputPath.json"
[ordered]@{
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
    system_dictionary = [IO.Path]::GetFullPath($systemDictionaryPath)
    addition_files = @($inputFiles | ForEach-Object { $_.Name })
    rows = $deduplicated.Count
    output_dictionary = [IO.Path]::GetFullPath($outputPath)
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding utf8

Write-Host "Built $outputPath from $($deduplicated.Count) unique addition rows"

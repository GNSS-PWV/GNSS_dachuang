param(
    [Parameter(Mandatory = $true)]
    [string[]]$DataDir,
    [string]$OutputDir = "result_strict_train_torch_gpu",
    [int]$Epochs = 100,
    [int]$BatchSize = 128,
    [int]$NumWorkers = 0,
    [int]$MaxFiles = 0,
    [switch]$RequireAllSplits
)

$ErrorActionPreference = "Stop"
$python = "D:\Users\lenovo\anaconda3\envs\torch_gpu\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "torch_gpu Python not found: $python"
}

# The pip CUDA wheel and Conda MKL each ship libiomp5md.dll.  This process-local
# compatibility switch is for the Windows experimental environment only; do not
# copy it into the server job or treat the resulting run as a formal benchmark.
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$script = Join-Path $PSScriptRoot "strict_train.py"
$args = @($script)
foreach ($dir in $DataDir) {
    $args += @("--data_dir", $dir)
}
$args += @(
    "--output_dir", $OutputDir,
    "--epochs", $Epochs,
    "--batch_size", $BatchSize,
    "--num_workers", $NumWorkers
)
if ($MaxFiles -gt 0) {
    $args += @("--max_files", $MaxFiles)
}
if ($RequireAllSplits) {
    $args += "--require_all_splits"
}

& $python @args
exit $LASTEXITCODE

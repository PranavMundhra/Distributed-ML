param(
    [string]$Python = "",
    [int]$GatewayPort = 8000,
    [int]$WorkerPort = 8001,
    [int]$WorkerCount = 2
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $localPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $localPython) {
        $Python = $localPython
    } else {
        $Python = "python"
    }
}

& $Python (Join-Path $PSScriptRoot "run_local.py") --gateway-port $GatewayPort --worker-port $WorkerPort --worker-count $WorkerCount
exit $LASTEXITCODE

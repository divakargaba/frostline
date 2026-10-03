$ErrorActionPreference = 'Stop'
$repoPath = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$pythonPath = Join-Path $repoPath '.venv\Scripts\python.exe'
$dashboardPath = Join-Path $repoPath 'frontend\dist\index.html'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Create .venv and install requirements-research.txt first. See README.md.' }
if (-not (Test-Path -LiteralPath $dashboardPath)) { throw 'Build the dashboard first: cd frontend; npm ci; npm run build' }
$listener = Get-NetTCPConnection -State Listen -LocalPort 8000 -ErrorAction SilentlyContinue
if ($listener) {
    try {
        $health = Invoke-RestMethod 'http://127.0.0.1:8000/api/health' -TimeoutSec 3
        if ($health.status -eq 'ok' -and $health.mode -eq 'local research replay') {
            Write-Output 'Frostline is already running: http://127.0.0.1:8000'
            exit 0
        }
    } catch { }
    throw 'Port 8000 is occupied by another server. Stop it or use a different port in the uvicorn command.'
}
$logPath = Join-Path $repoPath 'logs'
New-Item -ItemType Directory -Path $logPath -Force | Out-Null
$process = Start-Process -FilePath $pythonPath -ArgumentList '-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', '8000' -WorkingDirectory $repoPath -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logPath 'dashboard.out.log') -RedirectStandardError (Join-Path $logPath 'dashboard.err.log') -PassThru
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    Start-Sleep -Milliseconds 300
    try {
        $health = Invoke-RestMethod 'http://127.0.0.1:8000/api/health' -TimeoutSec 1
        if ($health.status -eq 'ok') {
            Write-Output "Frostline is running: http://127.0.0.1:8000 (launcher PID $($process.Id))"
            exit 0
        }
    } catch { }
    if ($process.HasExited) { break }
}
throw "Frostline did not become ready. Inspect $logPath\dashboard.err.log."

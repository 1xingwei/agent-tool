$ErrorActionPreference = "Stop"

# Shells derived from Git Bash (and some sandboxes) export the same variable under
# two spellings -- `PATH` alongside Windows' `Path`, `http_proxy` alongside
# `HTTP_PROXY`. Windows resolves names case-insensitively, but Start-Process builds
# the child environment in a case-sensitive dictionary and aborts with a
# duplicate-key error on the extras. Drop the bash spelling whenever the canonical
# one is also present. (`Get-ChildItem Env:` cannot be used to find these: the
# provider itself throws on the duplicates.)
$canonical = @{
    "Path"       = "PATH"
    "HTTP_PROXY" = "http_proxy"
    "HTTPS_PROXY" = "https_proxy"
}
foreach ($name in $canonical.Keys) {
    if ((Test-Path "Env:\$name") -and (Test-Path "Env:\$($canonical[$name])")) {
        Remove-Item -LiteralPath "Env:\$($canonical[$name])" -ErrorAction SilentlyContinue
    }
}

$env:HTTP_PROXY = "http://127.0.0.1:7897"
$env:HTTPS_PROXY = "http://127.0.0.1:7897"

# This script lives in scripts/, so the repo root is one level further up.
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

# Runtime output lives outside the source tree; create the directories on demand.
$logs = Join-Path $root "logs"
$state = Join-Path $root "var"
New-Item -ItemType Directory -Force -Path $logs, $state | Out-Null

Start-Process -FilePath "$root\.venv\Scripts\python.exe" -ArgumentList "src/run_service.py" -WorkingDirectory $root -RedirectStandardOutput "$logs\service.log" -RedirectStandardError "$logs\service.err.log" -WindowStyle Hidden
Start-Process -FilePath "$root\.venv\Scripts\streamlit.exe" -ArgumentList "run src/streamlit_app.py --server.headless true" -WorkingDirectory $root -RedirectStandardOutput "$logs\streamlit.log" -RedirectStandardError "$logs\streamlit.err.log" -WindowStyle Hidden

# Wait for the API to answer. A 401 counts as "up": when AUTH_SECRET is set in
# .env every request is unauthenticated until a token is supplied, and
# Invoke-WebRequest raises an exception on any non-2xx response. Only a failure to
# get any HTTP response at all (connection refused) means the service is not up.
$apiUp = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 2
    try {
        Invoke-WebRequest -Uri "http://localhost:8080/info" -Method Get -TimeoutSec 4 -UseBasicParsing | Out-Null
        $apiUp = $true
    } catch {
        if ($_.Exception.Response) { $apiUp = $true }
    }
    if ($apiUp) { break }
}
if (-not $apiUp) { Write-Host "API FAILED - see logs\service.err.log"; exit 1 }
Write-Host "API OK: http://localhost:8080"

for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:8501" -Method Get -TimeoutSec 4 -UseBasicParsing
        if ($r.StatusCode -eq 200) { Write-Host "Chat OK: http://localhost:8501"; break }
    } catch {
        if ($i -eq 29) { Write-Host "Streamlit FAILED - see logs\streamlit.err.log"; exit 1 }
    }
}
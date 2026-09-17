$ErrorActionPreference = "Stop"
$env:HTTP_PROXY = "http://127.0.0.1:7897"
$env:HTTPS_PROXY = "http://127.0.0.1:7897"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

Start-Process -FilePath "$root\.venv\Scripts\python.exe" -ArgumentList "src/run_service.py" -WorkingDirectory $root -RedirectStandardOutput "$root\service.log" -RedirectStandardError "$root\service.err.log" -WindowStyle Hidden
Start-Process -FilePath "$root\.venv\Scripts\streamlit.exe" -ArgumentList "run src/streamlit_app.py --server.headless true" -WorkingDirectory $root -RedirectStandardOutput "$root\streamlit.log" -RedirectStandardError "$root\streamlit.err.log" -WindowStyle Hidden

for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 2
    try {
        Invoke-RestMethod -Uri "http://localhost:8080/info" -Method GET -TimeoutSec 4 | Out-Null
        Write-Host "API OK: http://localhost:8080"
        break
    } catch {
        if ($i -eq 59) { Write-Host "API FAILED - see service.err.log"; exit 1 }
    }
}
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:8501" -Method Get -TimeoutSec 4 -UseBasicParsing
        if ($r.StatusCode -eq 200) { Write-Host "Chat OK: http://localhost:8501"; break }
    } catch {
        if ($i -eq 29) { Write-Host "Streamlit FAILED - see streamlit.err.log"; exit 1 }
    }
}
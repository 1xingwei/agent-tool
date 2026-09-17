$ErrorActionPreference = "Stop"

# 从 Git Bash（以及某些沙箱）派生的 shell 会以两种拼写导出同一个变量——
# `PATH` 与 Windows 的 `Path` 并存，`http_proxy` 与 `HTTP_PROXY` 并存。
# Windows 解析名称时不区分大小写，但 Start-Process 会以区分大小写的字典
# 构建子进程环境，并在遇到多余项时因重复键错误而中止。
# 当规范拼写也存在时，丢弃 bash 拼写。（`Get-ChildItem Env:` 无法用于查找这些项：
# provider 自身会在遇到重复项时抛出异常。）
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

# 此脚本位于 scripts/ 下，因此仓库根目录在其上一级。
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

# 运行时输出位于源码树之外；按需创建目录。
$logs = Join-Path $root "logs"
$state = Join-Path $root "var"
New-Item -ItemType Directory -Force -Path $logs, $state | Out-Null

Start-Process -FilePath "$root\.venv\Scripts\python.exe" -ArgumentList "src/run_service.py" -WorkingDirectory $root -RedirectStandardOutput "$logs\service.log" -RedirectStandardError "$logs\service.err.log" -WindowStyle Hidden
Start-Process -FilePath "$root\.venv\Scripts\streamlit.exe" -ArgumentList "run src/streamlit_app.py --server.headless true" -WorkingDirectory $root -RedirectStandardOutput "$logs\streamlit.log" -RedirectStandardError "$logs\streamlit.err.log" -WindowStyle Hidden

# 等待 API 响应。401 也算「已启动」：当 .env 中设置了 AUTH_SECRET 时，
# 在提供 token 之前每个请求都未认证，而 Invoke-WebRequest 对任何非 2xx 响应
# 都会抛出异常。只有完全无法获得 HTTP 响应（连接被拒绝）才意味着服务未启动。
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
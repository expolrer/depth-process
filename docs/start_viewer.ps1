$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$port = 8767
$running = $false

try {
    $client = [System.Net.Sockets.TcpClient]::new()
    $client.Connect("127.0.0.1", $port)
    $running = $client.Connected
    $client.Dispose()
} catch {
    $running = $false
}

if (-not $running) {
    $pythonCandidates = @(
        "C:\Program Files\Python310\python.exe",
        (Get-Command python -ErrorAction SilentlyContinue).Source
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
    if (-not $pythonCandidates) {
        throw "Python was not found. Install Python or start serve_viewer.py manually."
    }
    Start-Process `
        -FilePath $pythonCandidates[0] `
        -ArgumentList "serve_viewer.py", "--port", "$port" `
        -WorkingDirectory $root `
        -WindowStyle Hidden
    Start-Sleep -Milliseconds 900
}

Start-Process "http://127.0.0.1:$port/?view=depth"

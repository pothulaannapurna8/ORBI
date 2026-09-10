$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendProcess = $null
$FrontendProcess = $null

function Start-Backend {
    if ($script:BackendProcess -and -not $script:BackendProcess.HasExited) {
        Write-Host "Backend is already running (PID $($script:BackendProcess.Id))."
        return
    }

    $python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $python)) {
        throw "Python environment not found at $python. Create it and install requirements first."
    }

    $script:BackendProcess = Start-Process `
        -FilePath $python `
        -ArgumentList "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload" `
        -WorkingDirectory $ProjectRoot `
        -PassThru

    Write-Host "Backend started at http://localhost:8000 (PID $($script:BackendProcess.Id))."
}

function Start-Frontend {
    if ($script:FrontendProcess -and -not $script:FrontendProcess.HasExited) {
        Write-Host "Frontend is already running (PID $($script:FrontendProcess.Id))."
        return
    }

    $npm = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $npm) {
        throw "npm was not found on PATH. Install Node.js before starting the frontend."
    }

    $script:FrontendProcess = Start-Process `
        -FilePath "cmd.exe" `
        -ArgumentList "/c", "npm", "run", "dev", "--", "--host", "0.0.0.0" `
        -WorkingDirectory (Join-Path $ProjectRoot "frontend") `
        -PassThru

    Write-Host "Frontend started at http://localhost:3000 (PID $($script:FrontendProcess.Id))."
}

function Stop-ServiceProcess {
    param(
        [Parameter(Mandatory = $true)]
        [ref]$ProcessReference,
        [Parameter(Mandatory = $true)]
        [string]$ServiceName
    )

    $process = $ProcessReference.Value
    if ($process -and -not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
        Write-Host "$ServiceName stopped."
    } else {
        Write-Host "$ServiceName is not running."
    }

    $ProcessReference.Value = $null
}

function Show-Status {
    $backendStatus = if ($script:BackendProcess -and -not $script:BackendProcess.HasExited) { "running (PID $($script:BackendProcess.Id))" } else { "stopped" }
    $frontendStatus = if ($script:FrontendProcess -and -not $script:FrontendProcess.HasExited) { "running (PID $($script:FrontendProcess.Id))" } else { "stopped" }
    Write-Host "Backend:  $backendStatus"
    Write-Host "Frontend: $frontendStatus"
}

Write-Host "PS26227 local services"
Write-Host "Commands: backend, frontend, both, status, stop, exit"

try {
    while ($true) {
        $command = (Read-Host "start>").Trim().ToLowerInvariant()

        switch ($command) {
            "backend" { Start-Backend }
            "frontend" { Start-Frontend }
            "both" {
                Start-Backend
                Start-Frontend
            }
            "status" { Show-Status }
            "stop" {
                Stop-ServiceProcess ([ref]$script:FrontendProcess) "Frontend"
                Stop-ServiceProcess ([ref]$script:BackendProcess) "Backend"
            }
            "exit" { break }
            "" { }
            default { Write-Host "Unknown command. Use backend, frontend, both, status, stop, or exit." }
        }

        if ($command -eq "exit") {
            break
        }
    }
}
finally {
    Stop-ServiceProcess ([ref]$script:FrontendProcess) "Frontend"
    Stop-ServiceProcess ([ref]$script:BackendProcess) "Backend"
}
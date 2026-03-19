# Done-it Chatbot – local start script
# Run this from the chatbot/ directory:  .\start.ps1

Set-Location $PSScriptRoot

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv venv
}

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install / update dependencies
Write-Host "Installing dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

# Check .env exists
if (-not (Test-Path ".env")) {
    Write-Host ""
    Write-Host "ERROR: .env file not found!" -ForegroundColor Red
    Write-Host "Copy .env.example to .env and add your HuggingFace token." -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

# Load .env into the current PowerShell session so Python inherits all variables
Get-Content ".env" | Where-Object { $_ -match '^\s*[^#]\S+=\S' } | ForEach-Object {
    $parts = $_ -split '=', 2
    $key   = $parts[0].Trim()
    $value = $parts[1].Trim()
    [System.Environment]::SetEnvironmentVariable($key, $value, 'Process')
    Write-Host "  Loaded: $key" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "Starting Done-it chatbot API on http://127.0.0.1:8000 ..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor Gray
Write-Host ""

python main.py

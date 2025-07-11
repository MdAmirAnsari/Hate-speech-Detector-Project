#!/usr/bin/env pwsh
"""
PowerShell script to start the Hate Speech Detection Streamlit interface
"""

Write-Host "🛡️ Starting Hate Speech Detection Web Interface..." -ForegroundColor Green
Write-Host ""

# Check if virtual environment is activated
if ($env:VIRTUAL_ENV -or $env:CONDA_DEFAULT_ENV) {
    Write-Host "✅ Virtual environment detected!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Virtual environment not detected. Activating..." -ForegroundColor Yellow
    & "..\hate-speech-env\Scripts\Activate.ps1"
}

Write-Host ""
Write-Host "🚀 Starting Streamlit server..." -ForegroundColor Blue
Write-Host "📱 The web interface will open in your default browser at:" -ForegroundColor Cyan
Write-Host "   http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "💡 To stop the server, press Ctrl+C" -ForegroundColor Yellow
Write-Host ""

# Start Streamlit
try {
    streamlit run streamlit_app.py --server.port 8501 --server.address localhost
} catch {
    Write-Host "❌ Error starting Streamlit: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Please make sure Streamlit is installed: pip install streamlit" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

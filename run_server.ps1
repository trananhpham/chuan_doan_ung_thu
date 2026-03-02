# ============================================================
# run_server.ps1 - Khởi động Flask Backend Server
# Breast Cancer Diagnosis System
# ============================================================

$PYTHON = "C:\Users\ADMIN\AppData\Local\Programs\Python\Python313\python.exe"
$ROOT   = $PSScriptRoot

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Breast Cancer Diagnosis System"             -ForegroundColor Cyan
Write-Host "  Đang khởi động server..."                   -ForegroundColor Cyan
Write-Host "  Truy cập: http://localhost:5000"            -ForegroundColor Yellow
Write-Host "  Dừng server: Ctrl+C"                        -ForegroundColor Gray
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $ROOT
& $PYTHON backend\app.py

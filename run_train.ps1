# ============================================================
# run_train.ps1 - Script chạy training cho cả 2 mô hình AI
# Breast Cancer Diagnosis System
#
# CÁCH DÙNG:
#   .\run_train.ps1 ultrasound   -> Train mô hình siêu âm
#   .\run_train.ps1 biopsy       -> Train mô hình vi thể
#   .\run_train.ps1 all          -> Train cả 2 mô hình
# ============================================================

$PYTHON = "C:\Users\ADMIN\AppData\Local\Programs\Python\Python313\python.exe"
$ROOT   = $PSScriptRoot
$AI_DIR = Join-Path $ROOT "ai_engine"

function Train-Ultrasound {
    Write-Host ""
    Write-Host "=============================================" -ForegroundColor Cyan
    Write-Host "  [1/2] TRAINING: EfficientNet-B3 (Siêu Âm)" -ForegroundColor Cyan
    Write-Host "  Dataset : data/Dataset_BUSI_with_GT/"       -ForegroundColor Gray
    Write-Host "  Output  : ai_engine/saved_models/ultrasound_efficientnet_b3.pth" -ForegroundColor Gray
    Write-Host "  Phase 1 : 15 epochs (top layer only)"       -ForegroundColor Gray
    Write-Host "  Phase 2 : 35 epochs (fine-tuning)"         -ForegroundColor Gray
    Write-Host "=============================================" -ForegroundColor Cyan
    Write-Host ""
    Set-Location $AI_DIR
    & $PYTHON train_ultrasound.py
    Set-Location $ROOT
    Write-Host ""
    Write-Host "[OK] Hoàn tất training Ultrasound!" -ForegroundColor Green
}

function Train-Biopsy {
    Write-Host ""
    Write-Host "=============================================" -ForegroundColor Magenta
    Write-Host "  [2/2] TRAINING: ResNet-50 (Vi thể)"        -ForegroundColor Magenta
    Write-Host "  Dataset : data/BreaKHis_v1/..."             -ForegroundColor Gray
    Write-Host "  Output  : ai_engine/saved_models/biopsy_resnet50.pth" -ForegroundColor Gray
    Write-Host "  Phase 1 : 10 epochs (top layer only)"       -ForegroundColor Gray
    Write-Host "  Phase 2 : 40 epochs (fine-tuning layer4)"  -ForegroundColor Gray
    Write-Host "=============================================" -ForegroundColor Magenta
    Write-Host ""
    Set-Location $AI_DIR
    & $PYTHON train_biopsy.py
    Set-Location $ROOT
    Write-Host ""
    Write-Host "[OK] Hoàn tất training Biopsy!" -ForegroundColor Green
}

# ---- Main ----
$mode = $args[0]

if (-not $mode) {
    Write-Host ""
    Write-Host "CÁCH DÙNG:" -ForegroundColor Yellow
    Write-Host "  .\run_train.ps1 ultrasound   -> Train model Siêu Âm (EfficientNet-B3)"
    Write-Host "  .\run_train.ps1 biopsy        -> Train model Vi Thể (ResNet-50)"
    Write-Host "  .\run_train.ps1 all           -> Train cả 2 model"
    Write-Host ""
    exit
}

switch ($mode.ToLower()) {
    "ultrasound" { Train-Ultrasound }
    "biopsy"     { Train-Biopsy }
    "all" {
        Train-Ultrasound
        Train-Biopsy
        Write-Host ""
        Write-Host "=============================================" -ForegroundColor Green
        Write-Host "  [DONE] Đã train xong cả 2 mô hình!"       -ForegroundColor Green
        Write-Host "  Khởi động server: .\run_server.ps1"        -ForegroundColor Green
        Write-Host "=============================================" -ForegroundColor Green
    }
    default {
        Write-Host "Lệnh không hợp lệ: $mode" -ForegroundColor Red
        Write-Host "Dùng: ultrasound | biopsy | all"
    }
}

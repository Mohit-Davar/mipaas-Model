# run_training.ps1
# PowerShell launcher for CheXpert Plus training.
# Usage: .\run_training.ps1 [-Model densenet121|tf_efficientnetv2_s|convnext_small|rad_dino|rad_dino_frozen|tri_nn_transformer] [extra train.py args]
param(
    [string]$Model = "densenet121",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Extra
)

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot

$pythonPath = "C:\ProgramData\Anaconda3\python.exe"
if (-not (Test-Path $pythonPath)) {
    $pythonPath = "python"
}

# Per-model presets (CNNs use the 512 px / batch 16 x 2 defaults from config.py)
$presets = @{
    "rad_dino"           = @("--model_name", "rad_dino", "--image_size", "518", "--batch_size", "8", "--accum_steps", "4",
                             "--epochs", "2", "--backbone_lr", "5e-5", "--head_lr", "1e-3", "--layer_decay", "0.75",
                             "--weight_decay", "0.05", "--dropout", "0.1")
    "rad_dino_frozen"    = @("--model_name", "rad_dino", "--run_name", "rad_dino_frozen", "--image_size", "518",
                             "--freeze_blocks", "12", "--batch_size", "32", "--accum_steps", "1", "--epochs", "1",
                             "--head_lr", "1e-3", "--dropout", "0.1")
    "tri_nn_transformer" = @("--model_name", "tri_nn_transformer", "--image_size", "384", "--batch_size", "8", "--accum_steps", "4")
}
$trainArgs = @("train.py")
if ($presets.ContainsKey($Model)) { $trainArgs += $presets[$Model] } else { $trainArgs += @("--model_name", $Model) }
if ($Extra) { $trainArgs += $Extra }

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " CheXpert Plus Training | $Model" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/3] Pre-flight QC audit (includes the label-integrity gate)..." -ForegroundColor Yellow
& $pythonPath qc_audit.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Pre-flight quality control audit failed! Training aborted."
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "[2/3] Hardware check..." -ForegroundColor Yellow
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader

Write-Host ""
Write-Host "[3/3] Training: $($trainArgs -join ' ')" -ForegroundColor Yellow
& $pythonPath @trainArgs

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Green
Write-Host " Training finished. Next: train the other models, then run_ensemble.bat" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Green

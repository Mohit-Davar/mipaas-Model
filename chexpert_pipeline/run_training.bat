@echo off
title CheXpert Plus - Training Engine
color 0B
cd /d "%~dp0"

rem Usage: run_training.bat [model] [extra train.py arguments]
rem   CNNs (512 px, batch 16 x 2):  densenet121 (default) ^| tf_efficientnetv2_s ^| convnext_small
rem   Transformer (518 px):         rad_dino         = RAD-DINO ViT-B/14, full fine-tune, 2 epochs (~3 h)
rem                                 rad_dino_frozen  = RAD-DINO frozen, head only, 1 epoch (~25 min baseline)
rem   Legacy:                       tri_nn_transformer (384 px, batch 8 x 4)
rem Recommended: densenet121 + tf_efficientnetv2_s + rad_dino, then run_ensemble.bat

set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

set PYTHON_EXE=C:\ProgramData\Anaconda3\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python

set MODEL=%~1
if "%MODEL%"=="" set MODEL=densenet121
set ARGS=--model_name %MODEL%
if /I "%MODEL%"=="rad_dino" set ARGS=--model_name rad_dino --image_size 518 --batch_size 8 --accum_steps 4 --epochs 2 --backbone_lr 5e-5 --head_lr 1e-3 --layer_decay 0.75 --weight_decay 0.05 --dropout 0.1
if /I "%MODEL%"=="rad_dino_frozen" set ARGS=--model_name rad_dino --run_name rad_dino_frozen --image_size 518 --freeze_blocks 12 --batch_size 32 --accum_steps 1 --epochs 1 --head_lr 1e-3 --dropout 0.1
if /I "%MODEL%"=="tri_nn_transformer" set ARGS=--model_name tri_nn_transformer --image_size 384 --batch_size 8 --accum_steps 4

echo =====================================================================
echo  CheXpert Plus Training  ^|  %MODEL%
echo  Measured on RTX 4070 SUPER (image cache required for these speeds):
echo    densenet121 / tf_efficientnetv2_s @512: ~25 min/epoch, 3 epochs
echo    convnext_small @512: ~38 min/epoch, 3 epochs
echo    rad_dino @518: ~82 min/epoch, 2 epochs ^| rad_dino_frozen: ~25 min
echo =====================================================================
echo.

echo [1/3] Pre-flight QC audit (includes the label-integrity gate)...
"%PYTHON_EXE%" qc_audit.py
if errorlevel 1 (
    echo [ERROR] Pre-flight audit failed - training aborted. See the messages above.
    pause
    exit /b 1
)

echo.
echo [2/3] Hardware check...
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
echo.

echo [3/3] Training: train.py %ARGS% %2 %3 %4 %5 %6 %7 %8 %9
"%PYTHON_EXE%" train.py %ARGS% %2 %3 %4 %5 %6 %7 %8 %9

echo.
echo =====================================================================
echo  Training finished. Checkpoints in ..\checkpoints, prediction CSVs in ..\outputs
echo  Next: train the other models, then run_ensemble.bat
echo =====================================================================
echo.
pause

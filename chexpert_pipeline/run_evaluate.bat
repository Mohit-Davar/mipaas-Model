@echo off
title CheXpert Plus - Model Evaluation ^& Calibration
color 0E
cd /d "%~dp0"

rem Usage: run_evaluate.bat [checkpoint.pth]   (default: ..\checkpoints\densenet121_best.pth)

echo =====================================================================
echo  CheXpert Plus: Offline Model Evaluation ^& Threshold Calibration
echo =====================================================================
echo.

set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set PATH=C:\ProgramData\Anaconda3;C:\ProgramData\Anaconda3\Scripts;C:\ProgramData\Anaconda3\Library\bin;%PATH%

set CHECKPOINT_PATH=%~1
if "%CHECKPOINT_PATH%"=="" set CHECKPOINT_PATH=..\checkpoints\densenet121_best.pth

if not exist "%CHECKPOINT_PATH%" (
    echo [ERROR] Checkpoint not found at %CHECKPOINT_PATH%
    echo Usage: run_evaluate.bat [path_to_pth]
    pause
    exit /b 1
)

echo --- Official CheXpert validation set (radiologist labels), with flip TTA ---
python evaluate.py --checkpoint "%CHECKPOINT_PATH%" --split valid --tta
echo.
echo --- Held-out test split (labeler labels), with flip TTA ---
python evaluate.py --checkpoint "%CHECKPOINT_PATH%" --split test --tta

echo.
echo =====================================================================
echo  Evaluation finished!
echo =====================================================================
echo.
pause

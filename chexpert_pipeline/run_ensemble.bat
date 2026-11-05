@echo off
setlocal EnableDelayedExpansion
title CheXpert Plus - Ensemble
color 0D
cd /d "%~dp0"

rem Averages every trained model whose prediction CSVs exist in ..\outputs
rem (densenet121, tf_efficientnetv2_s, convnext_small, rad_dino).

set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set PYTHON_EXE=C:\ProgramData\Anaconda3\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python

set O=..\outputs
for %%S in (valid test) do (
    set FILES=
    set N=0
    for %%M in (densenet121 tf_efficientnetv2_s convnext_small rad_dino) do (
        if exist "%O%\%%M_%%S_predictions.csv" (
            set FILES=!FILES! "%O%\%%M_%%S_predictions.csv"
            set /a N+=1
        )
    )
    echo.
    echo ===== Ensemble on %%S split: !N! model^(s^) =====
    if !N! LSS 2 (
        echo Need at least 2 trained models with %%S predictions in %O% - skipping.
    ) else (
        "%PYTHON_EXE%" ensemble.py --split %%S --method mean --pred_files !FILES!
    )
)
echo.
pause

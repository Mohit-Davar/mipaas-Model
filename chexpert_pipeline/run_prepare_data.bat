@echo off
title CheXpert Plus - Data Preparation
color 0E
cd /d "%~dp0"

rem Prerequisite: CheXpert v1.0 train.csv and valid.csv in D:\Yatharth\Mipaas\labels\
rem (Stanford AIMI; they ship with CheXpert-v1.0 and CheXpert-v1.0-small)

set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set PYTHON_EXE=C:\ProgramData\Anaconda3\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python

echo =====================================================================
echo  CheXpert Plus: Data Preparation (labels -^> QC -^> probe -^> cache)
echo =====================================================================

echo.
echo [1/4] Rebuilding split manifest from original CheXpert labels...
"%PYTHON_EXE%" tools\build_manifest_chexpert.py
if errorlevel 1 goto :fail

echo.
echo [2/4] Pre-flight QC audit (6 gates, incl. label integrity)...
"%PYTHON_EXE%" qc_audit.py
if errorlevel 1 goto :fail

echo.
echo [3/4] Label sanity probe (do the labels describe the images?)...
"%PYTHON_EXE%" tools\label_sanity_probe.py
if errorlevel 1 goto :fail

echo.
echo [4/4] Building / topping up the 512px image cache (C:\chexpert_cache\jpg512)...
echo       First full build: ~1.5-2 h (limited by the D: hard drive), ~13 GB. Resumable.
"%PYTHON_EXE%" tools\build_image_cache.py
if errorlevel 1 goto :fail

echo.
echo =====================================================================
echo  Data ready. Next:  run_training.bat densenet121
echo =====================================================================
pause
exit /b 0

:fail
echo.
echo [ERROR] Step failed - see the messages above. Training data is NOT ready.
pause
exit /b 1

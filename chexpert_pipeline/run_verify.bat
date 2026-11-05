@echo off
title CheXpert Plus - Automated Pipeline Verification
color 0A
cd /d "%~dp0"

echo =====================================================================
echo  Running Comprehensive Pipeline Verification Suite (8 End-to-End Tests)
echo =====================================================================
echo.

set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set PATH=C:\ProgramData\Anaconda3;C:\ProgramData\Anaconda3\Scripts;C:\ProgramData\Anaconda3\Library\bin;%PATH%

python verify_pipeline.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Verification suite detected errors!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo =====================================================================
echo  All 8 tests passed successfully! Codebase is ready for training.
echo =====================================================================
echo.
pause


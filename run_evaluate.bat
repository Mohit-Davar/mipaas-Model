@echo off
title CheXpert Plus - Model Evaluation ^& Calibration
cd /d "%~dp0chexpert_pipeline"
call .\run_evaluate.bat %*

@echo off
title CheXpert Plus - Training Engine
cd /d "%~dp0chexpert_pipeline"
call .\run_training.bat %*

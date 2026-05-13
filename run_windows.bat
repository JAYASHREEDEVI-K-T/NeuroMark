@echo off
title NeuroMark EEG — Starting...
echo.
echo  ==========================================================
echo   NeuroMark EEG - Seizure Stage Detection System
echo  ==========================================================
echo.

:: Check if venv exists
if not exist "venv\" (
    echo  [SETUP] Creating virtual environment...
    python -m venv venv
    echo  [SETUP] Installing dependencies - this may take a few minutes...
    call venv\Scripts\activate.bat
    pip install -r requirements.txt --quiet
    echo  [SETUP] Done!
    echo.
) else (
    call venv\Scripts\activate.bat
)

:: Check model
if not exist "models\model_6ch.keras" (
    echo  [WARNING] Model file not found!
    echo  Please download model_6ch.keras from your Kaggle notebook
    echo  and place it in the models\ folder.
    echo.
)

echo  Starting server at http://localhost:5000
echo  Press Ctrl+C to stop.
echo.

python app.py

pause

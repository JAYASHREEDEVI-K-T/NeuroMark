#!/bin/bash

echo ""
echo " =========================================================="
echo "  NeuroMark EEG — Seizure Stage Detection System"
echo " =========================================================="
echo ""

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo " [SETUP] Creating virtual environment..."
    python3 -m venv venv
    echo " [SETUP] Installing dependencies — this may take a few minutes..."
    source venv/bin/activate
    pip install -r requirements.txt --quiet
    echo " [SETUP] Done!"
    echo ""
else
    source venv/bin/activate
fi

# Check model
if [ ! -f "models/model_6ch.keras" ]; then
    echo " [WARNING] Model file not found!"
    echo " Please download model_6ch.keras from your Kaggle notebook"
    echo " and place it in the models/ folder."
    echo ""
fi

echo " Starting server at http://localhost:5000"
echo " Press Ctrl+C to stop."
echo ""

python app.py

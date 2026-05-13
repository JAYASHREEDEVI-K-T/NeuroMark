# NeuroMark — EEG Seizure Stage Detection System

A professional web application that takes any EDF session from the TUH EEG dataset,
runs full preprocessing + CNN inference, and displays a richly annotated result with
timeline, waveform, charts, and exportable tables.

---

## WHAT THIS DOES

1. You upload an `.edf` file (and optionally its `.csv_bi` annotation file)
2. The app:
   - Loads the EDF with MNE
   - Computes 6 bipolar channels (FP1-T3, T3-T5, A1-T3, FP2-T4, T4-T6, T4-A2)
   - Applies bandpass filter 0.5–40 Hz + 50 Hz notch
   - Segments into 5-second windows with 50% overlap
   - Labels each window (Ictal / Pre-ictal / Post-ictal / Inter-ictal)
   - Z-score normalizes each window
   - Runs inference with your trained 1D CNN
3. Shows: waveform view, color-coded prediction timeline, pie chart, confidence histogram,
   full window-by-window table with probabilities. Export as JSON or CSV.

---

## PREREQUISITES

- Python 3.10 or 3.11 (recommended)
- VS Code (recommended editor)
- No GPU required — runs on CPU

---

## STEP 1 — Download Your Model from Kaggle

1. Go to your Kaggle notebook
2. In the output panel, find `model_6ch.keras`
3. Download it
4. Place it inside the `models/` folder of this project:

```
neuromark_eeg/
    models/
        model_6ch.keras       ← PUT IT HERE
```

---

## STEP 2 — Set Up Python Environment

Open a terminal in VS Code (`Ctrl+`` ` or Terminal → New Terminal).

Navigate to the project folder:
```bash
cd path/to/neuromark_eeg
```

Create a virtual environment:
```bash
python -m venv venv
```

Activate it:
- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **Mac / Linux:**
  ```bash
  source venv/bin/activate
  ```

You should see `(venv)` in your terminal prompt.

---

## STEP 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs Flask, MNE, NumPy, TensorFlow (CPU), and SciPy.
TensorFlow download is ~400 MB — takes a few minutes on first install.

---

## STEP 4 — Run the App

```bash
python app.py
```

You will see:
```
════════════════════════════════════════════════════
  NeuroMark EEG — Seizure Stage Detection System
════════════════════════════════════════════════════

  Model path  : models/model_6ch.keras
  Model ready : ✓ YES

  Open in browser: http://localhost:5000
════════════════════════════════════════════════════
```

Open your browser and go to: **http://localhost:5000**

---

## STEP 5 — Analyze a Session

1. Click **Upload EDF File** (or drag & drop) — choose any `.edf` from the TUH dataset
2. Click the **CSV_BI Annotation File** row to optionally upload the matching `.csv_bi`
   (without it, all windows are treated as inter-ictal; with it, seizure periods are auto-labeled)
3. Select the **Montage** from the dropdown (or leave on Auto-detect)
4. Click **ANALYZE EEG**
5. Wait ~10–60 seconds depending on file size (no GPU needed)
6. Results appear with:
   - Color-coded waveform
   - Prediction timeline
   - Class distribution pie chart
   - Confidence histogram
   - Full window table with probabilities
   - Export to JSON / CSV

---

## PROJECT STRUCTURE

```
neuromark_eeg/
├── app.py                  ← Flask server (run this)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
├── core/
│   ├── __init__.py
│   ├── config.py           ← All constants (channels, sfreq, labels)
│   ├── loader.py           ← EDF loading, bipolar computation, annotation parsing
│   ├── preprocessor.py     ← Segmentation, labeling, normalization
│   └── predictor.py        ← Model loading, inference, result building
│
├── templates/
│   └── index.html          ← Full UI (single-file, no build step)
│
├── models/                 ← PUT model_6ch.keras HERE
├── uploads/                ← Temp files (auto-cleaned)
└── exports/                ← JSON results saved here
```

---

## TROUBLESHOOTING

**"Model not found" in the UI**
→ Download `model_6ch.keras` from Kaggle and put it in `models/`

**"None of the required EEG channels found"**
→ Your EDF uses a different montage. Try selecting a different montage in the dropdown.
→ Some EDF files use non-standard channel names — check your EDF header.

**MNE errors during loading**
→ Make sure MNE is installed: `pip install mne`
→ Some corrupted EDFs may not load — try a different session file.

**TensorFlow import errors on Windows**
→ Try: `pip install tensorflow-cpu` instead of `tensorflow`

**App runs but browser shows nothing**
→ Make sure you're visiting http://localhost:5000 (not 127.0.0.1:5000 — both should work)

**Slow analysis (>2 minutes)**
→ Normal for large files (>1 hour recordings) on CPU
→ Smaller files (10–15 min sessions) should take 15–30 seconds

---

## CHANNEL DEFINITIONS

The model was trained on 6 bipolar channels:

| # | Channel | Electrode A | Electrode B |
|---|---------|-------------|-------------|
| 1 | FP1-T3  | EEG FP1-REF | EEG T3-REF  |
| 2 | T3-T5   | EEG T3-REF  | EEG T5-REF  |
| 3 | A1-T3   | EEG A1-REF  | EEG T3-REF  |
| 4 | FP2-T4  | EEG FP2-REF | EEG T4-REF  |
| 5 | T4-T6   | EEG T4-REF  | EEG T6-REF  |
| 6 | T4-A2   | EEG T4-REF  | EEG A2-REF  |

For LE montage, `-REF` is replaced with `-LE`.

---

## LABEL CLASSES

| ID | Class       | Color  | Description                        |
|----|-------------|--------|------------------------------------|
| 0  | Inter-ictal | Green  | Normal brain activity              |
| 1  | Pre-ictal   | Yellow | 0–30s before seizure onset         |
| 2  | Ictal       | Red    | Active seizure period              |
| 3  | Post-ictal  | Blue   | 0–30s after seizure end            |

---

## SIGNAL PROCESSING PARAMETERS

- Sampling rate: 256 Hz (resampled)
- Window size: 5 seconds (1280 samples)
- Window overlap: 50% (step = 640 samples)
- Bandpass filter: 0.5–40 Hz (FIR)
- Notch filter: 50 Hz (powerline)
- Pre/post-ictal buffer: 30 seconds
- Normalization: per-window Z-score

---

*Built for the AI-Powered EEG Signal Cleaning and Seizure Stage Detection project.*
*Model trained on TUH EEG Dataset (Temple University Hospital).*

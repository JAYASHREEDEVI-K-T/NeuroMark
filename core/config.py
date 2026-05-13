"""
NeuroMark EEG — Configuration
All signal processing and model constants extracted from training notebook.
"""

# ── Signal settings ─────────────────────────────────────────────────────────
SFREQ        = 256          # Resample everything to 256 Hz
WINDOW_SEC   = 5            # Each window = 5 seconds
OVERLAP      = 0.5          # 50% overlap between consecutive windows
N_SAMPLES    = int(WINDOW_SEC * SFREQ)   # = 1280 samples per window
PRE_POST_SEC = 30           # Seconds before/after seizure = pre/post-ictal

# ── Bipolar channel definitions ─────────────────────────────────────────────
CHANNELS = [
    ("FP1-T3", "EEG FP1-REF", "EEG T3-REF"),
    ("T3-T5",  "EEG T3-REF",  "EEG T5-REF"),
    ("A1-T3",  "EEG A1-REF",  "EEG T3-REF"),
    ("FP2-T4", "EEG FP2-REF", "EEG T4-REF"),
    ("T4-T6",  "EEG T4-REF",  "EEG T6-REF"),
    ("T4-A2",  "EEG T4-REF",  "EEG A2-REF"),
]

CHANNELS_LE = [
    ("FP1-T3", "EEG FP1-LE", "EEG T3-LE"),
    ("T3-T5",  "EEG T3-LE",  "EEG T5-LE"),
    ("A1-T3",  "EEG A1-LE",  "EEG T3-LE"),
    ("FP2-T4", "EEG FP2-LE", "EEG T4-LE"),
    ("T4-T6",  "EEG T4-LE",  "EEG T6-LE"),
    ("T4-A2",  "EEG T4-LE",  "EEG A2-LE"),
]

N_CHANNELS = len(CHANNELS)

# ── Label mapping ───────────────────────────────────────────────────────────
LABEL_MAP = {
    0: "inter-ictal",
    1: "pre-ictal",
    2: "ictal",
    3: "post-ictal",
}

LABEL_COLORS = {
    0: "#22c55e",   # green   — inter-ictal
    1: "#eab308",   # yellow  — pre-ictal
    2: "#ef4444",   # red     — ictal
    3: "#3b82f6",   # blue    — post-ictal
}

LABEL_ICTAL = 2
LABEL_PRE   = 1
LABEL_POST  = 3
LABEL_INTER = 0

# ── Annotation parsing ──────────────────────────────────────────────────────
SEIZURE_LABELS = {"seiz"}
SKIP_LABELS    = {"artf", "eyem", "musc", "chew", "shiv", "elpp", "elst", "calb"}

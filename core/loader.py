"""
NeuroMark EEG — EDF Loader
Loads EDF file, computes 6 bipolar channels, applies bandpass + notch filter.
"""

import numpy as np
import mne
from pathlib import Path
from .config import (
    CHANNELS, CHANNELS_LE, N_CHANNELS, SFREQ,
    SEIZURE_LABELS, SKIP_LABELS
)

mne.set_log_level("ERROR")


def detect_montage(edf_path: str) -> str:
    """Detect montage type from file path or EDF content."""
    path_str = str(edf_path).lower()
    if "02_tcp_le" in path_str or "/le/" in path_str:
        return "02_tcp_le"
    elif "03_tcp_ar_a" in path_str:
        return "03_tcp_ar_a"
    else:
        return "01_tcp_ar"


def detect_montage_from_channels(ch_names: list) -> str:
    """
    Detect montage by inspecting actual channel names found in the EDF.
    More reliable than filename-based detection.
    """
    joined = " ".join(ch_names).upper()
    if "-LE" in joined:
        return "02_tcp_le"
    elif "-REF" in joined:
        return "01_tcp_ar"
    else:
        return "01_tcp_ar"


def load_edf_bipolar(edf_path: str, montage: str = None):
    """
    Load one EDF file and return a (N_CHANNELS, n_samples) array
    of bipolar channel signals.

    Parameters
    ----------
    edf_path : str — path to the .edf file
    montage  : str — optional montage override; auto-detected if None

    Returns
    -------
    data         : np.ndarray (N_CHANNELS, n_samples) float32
    sfreq        : float
    n_samples    : int
    ch_names     : list of str — the 6 bipolar channel names
    duration_sec : float — total recording duration
    zero_chs     : list of str — channels that had to be zeroed (missing electrodes)
    """

    # ── Step 1: Load EDF (all channels) ──────────────────────────────────────
    raw       = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    available = raw.ch_names

    # ── Step 2: Detect montage from ACTUAL channel names in the file ─────────
    # This is more reliable than guessing from the filename.
    # If user passed an explicit montage override, respect it.
    if montage and montage != "":
        # User explicitly chose a montage in the UI dropdown — use it
        ch_defs = CHANNELS_LE if "le" in montage.lower() else CHANNELS
    else:
        # Auto-detect from real channel names
        detected = detect_montage_from_channels(available)
        ch_defs  = CHANNELS_LE if "le" in detected.lower() else CHANNELS

    # ── Step 3: Collect the electrode names we need ───────────────────────────
    needed = []
    for (_, ea, eb) in ch_defs:
        if ea not in needed:
            needed.append(ea)
        if eb not in needed:
            needed.append(eb)

    present = [e for e in needed if e in available]
    missing = [e for e in needed if e not in available]

    if not present:
        raise ValueError(
            f"None of the required EEG channels found in this EDF file.\n"
            f"Required: {needed}\n"
            f"Available: {available}\n"
            f"This EDF may use a different montage. "
            f"Try selecting a montage manually in the dropdown."
        )

    # ── Step 4: Keep only the channels we need ────────────────────────────────
    raw.pick_channels(present)

    # ── Step 5: Resample to target frequency ─────────────────────────────────
    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ, verbose=False)

    # ── Step 6: Filter ────────────────────────────────────────────────────────
    raw.filter(0.5, 40.0, fir_design="firwin", verbose=False)
    raw.notch_filter(50.0, verbose=False)

    # ── Step 7: Build signal dictionary ──────────────────────────────────────
    eeg_data    = raw.get_data()
    signal_dict = {ch: eeg_data[i] for i, ch in enumerate(raw.ch_names)}
    n_samples    = eeg_data.shape[1]
    duration_sec = n_samples / SFREQ

    # ── Step 8: Compute 6 bipolar channels ───────────────────────────────────
    bipolar  = np.zeros((N_CHANNELS, n_samples), dtype=np.float32)
    ch_names = []
    zero_chs = []

    for ch_idx, (pair_name, ea, eb) in enumerate(ch_defs):
        ch_names.append(pair_name)
        if ea in signal_dict and eb in signal_dict:
            bipolar[ch_idx] = (signal_dict[ea] - signal_dict[eb]).astype(np.float32)
        else:
            bipolar[ch_idx] = 0.0
            zero_chs.append(pair_name)

    return bipolar, SFREQ, n_samples, ch_names, duration_sec, zero_chs


def parse_csvbi(csvbi_path: str, merge_gap_sec: float = 2.0):
    """
    Parse a .csv_bi annotation file and return merged seizure periods.

    Parameters
    ----------
    csvbi_path    : str — path to the .csv_bi file
    merge_gap_sec : float — merge seizures closer than this (seconds)

    Returns
    -------
    seizure_periods : list of (start, end) tuples in seconds
    all_annotations : list of dicts with all annotation rows
    """
    seizure_periods = []
    all_annotations = []

    with open(csvbi_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            if any(line.startswith(k)
                   for k in ["version", "bname", "duration", "channel"]):
                continue

            parts = line.split(",")
            if len(parts) < 4:
                continue

            try:
                start = float(parts[1].strip())
                end   = float(parts[2].strip())
                label = parts[3].strip().lower()
            except (ValueError, IndexError):
                continue

            all_annotations.append({
                "start": start,
                "end":   end,
                "label": label,
                "duration": end - start,
            })

            if label in SEIZURE_LABELS:
                seizure_periods.append((start, end))

    # Merge close seizure periods
    if len(seizure_periods) > 1:
        seizure_periods.sort(key=lambda x: x[0])
        merged = [seizure_periods[0]]
        for cur_s, cur_e in seizure_periods[1:]:
            last_s, last_e = merged[-1]
            if cur_s - last_e <= merge_gap_sec:
                merged[-1] = (last_s, max(last_e, cur_e))
            else:
                merged.append((cur_s, cur_e))
        seizure_periods = merged

    return seizure_periods, all_annotations
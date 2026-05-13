"""
NeuroMark EEG — Preprocessor
Segments EEG into windows, labels them, and normalizes.
"""

import numpy as np
from .config import (
    SFREQ, WINDOW_SEC, OVERLAP, N_SAMPLES, PRE_POST_SEC,
    LABEL_ICTAL, LABEL_PRE, LABEL_POST, LABEL_INTER,
    LABEL_MAP, N_CHANNELS,
)


def get_label_for_window(t_mid: float, seizure_periods: list) -> int:
    """
    Classify a window by its midpoint time.

    Priority: Ictal > Pre-ictal > Post-ictal > Inter-ictal
    """
    for (s_start, s_end) in seizure_periods:
        if s_start <= t_mid <= s_end:
            return LABEL_ICTAL
        if (s_start - PRE_POST_SEC) <= t_mid < s_start:
            return LABEL_PRE
        if s_end < t_mid <= (s_end + PRE_POST_SEC):
            return LABEL_POST
    return LABEL_INTER


def segment_and_label(bipolar_data: np.ndarray, seizure_periods: list):
    """
    Chop bipolar_data into overlapping 5-sec windows and label each.

    Parameters
    ----------
    bipolar_data    : np.ndarray (N_CHANNELS, n_samples)
    seizure_periods : list of (start, end) tuples in seconds

    Returns
    -------
    windows     : np.ndarray (n_windows, N_CHANNELS, N_SAMPLES)
    labels      : np.ndarray (n_windows,) int
    window_meta : list of dicts with timing info per window
    """
    n_total = bipolar_data.shape[1]
    step    = int(N_SAMPLES * (1 - OVERLAP))

    windows     = []
    labels      = []
    window_meta = []

    start = 0
    while start + N_SAMPLES <= n_total:
        end   = start + N_SAMPLES
        t_mid = (start + end) / 2 / SFREQ

        label = get_label_for_window(t_mid, seizure_periods)
        window = bipolar_data[:, start:end]

        windows.append(window)
        labels.append(label)
        window_meta.append({
            "t_start":   round(start / SFREQ, 3),
            "t_end":     round(end   / SFREQ, 3),
            "t_mid":     round(t_mid, 3),
            "label_id":  label,
            "label_str": LABEL_MAP[label],
            "sample_start": start,
            "sample_end":   end,
        })

        start += step

    if not windows:
        return (
            np.empty((0, N_CHANNELS, N_SAMPLES), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            [],
        )

    return (
        np.stack(windows).astype(np.float32),
        np.array(labels, dtype=np.int32),
        window_meta,
    )


def normalize_windows(X: np.ndarray) -> np.ndarray:
    """
    Z-score normalize each window independently.
    Mean and std computed across all channels and time points.
    """
    X_norm = X.copy()
    for i in range(len(X)):
        window = X[i]
        mu     = window.mean()
        sigma  = window.std() + 1e-8
        X_norm[i] = (window - mu) / sigma
    return X_norm.astype(np.float32)


def extract_waveform_sample(bipolar_data: np.ndarray,
                            max_points: int = 2000) -> dict:
    """
    Downsample the raw EEG for frontend waveform display.
    Returns dict mapping channel name → list of float values.
    """
    from .config import CHANNELS

    n_samples = bipolar_data.shape[1]
    step      = max(1, n_samples // max_points)
    indices   = np.arange(0, n_samples, step)
    times     = (indices / SFREQ).tolist()

    ch_data = {}
    for ch_idx, (ch_name, _, _) in enumerate(CHANNELS):
        raw_ch   = bipolar_data[ch_idx, indices]
        # Normalize to [-1, 1] for display
        ch_max   = np.abs(raw_ch).max() + 1e-8
        ch_norm  = (raw_ch / ch_max).tolist()
        ch_data[ch_name] = ch_norm

    return {"times": times, "channels": ch_data}

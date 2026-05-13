"""
NeuroMark EEG — Predictor
Loads trained CNN model and runs inference on EEG windows.
"""

import numpy as np
from pathlib import Path
from collections import Counter
from .config import LABEL_MAP, LABEL_COLORS, N_CHANNELS, N_SAMPLES

_model = None
_model_path = None


def get_model(model_path: str):
    """Load and cache the Keras model."""
    global _model, _model_path

    if _model is not None and _model_path == model_path:
        return _model

    try:
        import tensorflow as tf
        tf_major = int(tf.__version__.split(".")[0])
        _model = tf.keras.models.load_model(model_path)
        _model_path = model_path
        return _model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from '{model_path}'.\n"
            f"Error: {e}\n\n"
            f"Make sure you have downloaded model_6ch.keras from Kaggle "
            f"and placed it in the models/ folder."
        )


def predict_windows(windows: np.ndarray, model_path: str,
                    batch_size: int = 64) -> dict:
    """
    Run model inference on a batch of EEG windows.

    Parameters
    ----------
    windows    : np.ndarray (n_windows, N_CHANNELS, N_SAMPLES)
    model_path : str — path to .keras model file
    batch_size : int

    Returns
    -------
    dict with:
        predictions    : list of int (class index per window)
        probabilities  : list of list of float (4 softmax scores per window)
        confidence     : list of float (max probability per window)
    """
    model = get_model(model_path)

    n_batches  = (len(windows) + batch_size - 1) // batch_size
    all_probs  = []

    for i in range(n_batches):
        batch = windows[i * batch_size:(i + 1) * batch_size]
        probs = model.predict(batch, verbose=0)
        all_probs.append(probs)

    all_probs   = np.concatenate(all_probs, axis=0)
    predictions = np.argmax(all_probs, axis=1).tolist()
    confidence  = np.max(all_probs, axis=1).tolist()

    return {
        "predictions":   predictions,
        "probabilities": all_probs.tolist(),
        "confidence":    confidence,
    }


def build_results(
    window_meta: list,
    pred_result: dict,
    seizure_periods: list,
    duration_sec: float,
    ch_names: list,
    zero_chs: list,
) -> dict:
    """
    Merge window metadata with predictions into full result payload.
    """
    predictions  = pred_result["predictions"]
    probs        = pred_result["probabilities"]
    confidences  = pred_result["confidence"]

    # Per-window result
    windows_out = []
    for i, meta in enumerate(window_meta):
        pred_id = predictions[i]
        windows_out.append({
            **meta,
            "pred_label_id":  pred_id,
            "pred_label_str": LABEL_MAP[pred_id],
            "pred_color":     LABEL_COLORS[pred_id],
            "confidence":     round(confidences[i] * 100, 1),
            "probs": {
                LABEL_MAP[j]: round(probs[i][j] * 100, 1)
                for j in range(4)
            },
        })

    # Summary statistics
    counts = Counter(predictions)
    total  = len(predictions)

    summary = {
        "total_windows":   total,
        "duration_sec":    round(duration_sec, 1),
        "channels_used":   ch_names,
        "channels_zeroed": zero_chs,
        "seizure_periods": [
            {"start": round(s, 2), "end": round(e, 2), "duration": round(e - s, 2)}
            for s, e in seizure_periods
        ],
        "class_counts": {
            LABEL_MAP[k]: {
                "count":   counts.get(k, 0),
                "percent": round(counts.get(k, 0) / max(total, 1) * 100, 1),
                "color":   LABEL_COLORS[k],
            }
            for k in range(4)
        },
        "dominant_label": LABEL_MAP[max(counts, key=counts.get)] if counts else "unknown",
        "mean_confidence": round(float(np.mean(confidences)) * 100, 1),
    }

    return {
        "windows":  windows_out,
        "summary":  summary,
    }

"""
NeuroMark EEG — Flask Backend
Run with: python app.py
"""

# ── FIX: OpenBLAS memory crash on Windows ────────────────────────────────────
# Must be set BEFORE numpy / tensorflow are imported.
import os
os.environ["OPENBLAS_NUM_THREADS"]  = "1"
os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["MKL_NUM_THREADS"]       = "1"
os.environ["NUMEXPR_NUM_THREADS"]   = "1"
os.environ["VECLIB_MAXIMUM_THREADS"]= "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# ─────────────────────────────────────────────────────────────────────────────

import json
import uuid
import shutil
import tempfile
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from werkzeug.utils import secure_filename

from core.config import LABEL_MAP, LABEL_COLORS, N_CHANNELS, N_SAMPLES, SFREQ
from core.loader import load_edf_bipolar, parse_csvbi, detect_montage
from core.preprocessor import segment_and_label, normalize_windows, extract_waveform_sample
from core.predictor import predict_windows, build_results

# ── Flask setup ──────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
# Use system temp dir for uploads — avoids "no space on device" on the project drive
# Windows: C:\Users\<user>\AppData\Local\Temp\neuromark_uploads
# Mac/Linux: /tmp/neuromark_uploads
UPLOAD_DIR   = Path(tempfile.gettempdir()) / "neuromark_uploads"
MODEL_DIR    = BASE_DIR / "models"
EXPORTS_DIR  = BASE_DIR / "exports"
MODEL_FILE   = MODEL_DIR / "model_6ch.keras"

for d in [UPLOAD_DIR, MODEL_DIR, EXPORTS_DIR]:
    d.mkdir(exist_ok=True, parents=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB max upload


# ── JSON error handlers — prevents HTML pages from reaching the browser ──────
@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request", "detail": str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found", "detail": str(e)}), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large", "detail": "Max upload size is 500 MB."}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Server error", "detail": str(e)}), 500

@app.errorhandler(Exception)
def unhandled(e):
    app.logger.error(traceback.format_exc())
    return jsonify({"error": type(e).__name__, "detail": str(e)}), 500


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/favicon.ico")
def favicon():
    # Return a minimal 1x1 transparent ICO so the browser stops 404-ing
    ico = (
        b"\x00\x00\x01\x00\x01\x00\x01\x01\x00\x00\x01\x00"
        b"\x18\x00\x28\x00\x00\x00\x16\x00\x00\x00\x28\x00"
        b"\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x01\x00"
        b"\x18\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\xd4\xff\x00\x00\x00\x00\x00"
    )
    return Response(ico, mimetype="image/x-icon")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """Check if model is available."""
    model_ok   = MODEL_FILE.exists()
    model_size = round(MODEL_FILE.stat().st_size / (1024 * 1024), 1) if model_ok else 0
    return jsonify({
        "model_loaded": model_ok,
        "model_path":   str(MODEL_FILE),
        "model_size_mb": model_size,
        "ready": model_ok,
    })


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Main analysis endpoint.
    Expects multipart/form-data with:
        edf_file  : EDF file
        csvbi_file: CSV_BI annotation file (optional — if not provided, all inter-ictal)
        montage   : optional montage string
    """
    session_id = str(uuid.uuid4())[:8]
    tmp_dir    = UPLOAD_DIR / session_id
    tmp_dir.mkdir(exist_ok=True)

    try:
        # ── 1. Validate model ────────────────────────────────────────────────
        if not MODEL_FILE.exists():
            return jsonify({
                "error": "Model file not found",
                "detail": (
                    f"Please download 'model_6ch.keras' from your Kaggle "
                    f"notebook outputs and place it in the 'models/' folder.\n"
                    f"Expected path: {MODEL_FILE}"
                )
            }), 503

        # ── 2. Save uploaded files ───────────────────────────────────────────
        if "edf_file" not in request.files:
            return jsonify({"error": "No EDF file provided"}), 400

        edf_file = request.files["edf_file"]
        if not edf_file.filename.lower().endswith(".edf"):
            return jsonify({"error": "File must be a .edf file"}), 400

        edf_filename = secure_filename(edf_file.filename)
        edf_path     = tmp_dir / edf_filename
        edf_file.save(str(edf_path))

        csvbi_path = None
        if "csvbi_file" in request.files:
            csvbi_file = request.files["csvbi_file"]
            if csvbi_file.filename:
                csvbi_filename = secure_filename(csvbi_file.filename)
                csvbi_path     = tmp_dir / csvbi_filename
                csvbi_file.save(str(csvbi_path))

        montage = request.form.get("montage", None)

        # ── 3. Load EDF ──────────────────────────────────────────────────────
        app.logger.info(f"[{session_id}] Loading EDF: {edf_filename}")
        bipolar_data, sfreq, n_samples, ch_names, duration_sec, zero_chs = \
            load_edf_bipolar(str(edf_path), montage)

        app.logger.info(
            f"[{session_id}] Loaded — duration={duration_sec:.1f}s, "
            f"shape={bipolar_data.shape}"
        )

        # ── 4. Parse annotations ─────────────────────────────────────────────
        if csvbi_path and csvbi_path.exists():
            seizure_periods, all_annotations = parse_csvbi(str(csvbi_path))
            app.logger.info(
                f"[{session_id}] Annotations — {len(seizure_periods)} seizure period(s)"
            )
        else:
            seizure_periods  = []
            all_annotations  = []
            app.logger.info(f"[{session_id}] No annotations — treating as inter-ictal")

        # ── 5. Extract waveform sample for display ───────────────────────────
        waveform = extract_waveform_sample(bipolar_data, max_points=3000)

        # ── 6. Segment & label ───────────────────────────────────────────────
        windows, labels, window_meta = segment_and_label(bipolar_data, seizure_periods)

        if len(windows) == 0:
            return jsonify({
                "error": "No windows could be extracted from this EDF file.",
                "detail": f"Recording duration: {duration_sec:.1f}s. "
                          f"Minimum required: {5}s."
            }), 422

        app.logger.info(f"[{session_id}] Segmented — {len(windows)} windows")

        # ── 7. Normalize ─────────────────────────────────────────────────────
        windows_norm = normalize_windows(windows)

        # ── 8. Predict ───────────────────────────────────────────────────────
        app.logger.info(f"[{session_id}] Running inference...")
        pred_result = predict_windows(windows_norm, str(MODEL_FILE))
        app.logger.info(f"[{session_id}] Inference done.")

        # ── 9. Build result payload ──────────────────────────────────────────
        results = build_results(
            window_meta, pred_result,
            seizure_periods, duration_sec,
            ch_names, zero_chs,
        )

        results["waveform"]         = waveform
        results["session_id"]       = session_id
        results["filename"]         = edf_filename
        results["sfreq"]            = sfreq
        results["label_colors"]     = LABEL_COLORS
        results["label_map"]        = {str(k): v for k, v in LABEL_MAP.items()}
        results["has_annotations"]  = bool(csvbi_path)
        results["all_annotations"]  = all_annotations[:200]  # cap for payload

        # ── 10. Save result JSON for export ──────────────────────────────────
        export_path = EXPORTS_DIR / f"result_{session_id}.json"
        with open(export_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        return jsonify(results)

    except ValueError as e:
        return jsonify({"error": "Channel/format error", "detail": str(e)}), 422
    except RuntimeError as e:
        return jsonify({"error": "Model error", "detail": str(e)}), 500
    except OSError as e:
        if e.errno == 28 or "space" in str(e).lower() or "No space" in str(e):
            return jsonify({
                "error": "Disk full — no space left on device",
                "detail": (
                    f"Windows ran out of disk space writing the temp file.\n\n"
                    f"Temp folder used: {UPLOAD_DIR}\n\n"
                    f"Fix options:\n"
                    f"  1. Free up space on your C: drive (need ~2× the EDF file size)\n"
                    f"  2. Run Disk Cleanup (search 'Disk Cleanup' in Start)\n"
                    f"  3. Delete old files in {UPLOAD_DIR}\n"
                    f"  4. If C: is full, set TEMP env var to another drive before running:\n"
                    f"     set TEMP=D:\\Temp && python app.py"
                )
            }), 507
        return jsonify({"error": "File system error", "detail": str(e)}), 500
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": "Unexpected error", "detail": str(e)}), 500
    finally:
        # Clean up temp upload
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


@app.route("/api/export/<session_id>")
def api_export(session_id):
    """Download full JSON result for a session."""
    # Sanitize session_id
    safe_id = "".join(c for c in session_id if c.isalnum() or c == "-")[:16]
    filepath = EXPORTS_DIR / f"result_{safe_id}.json"
    if not filepath.exists():
        return jsonify({"error": "Result not found"}), 404
    return send_from_directory(
        str(EXPORTS_DIR),
        f"result_{safe_id}.json",
        as_attachment=True,
        download_name=f"neuromark_result_{safe_id}.json",
    )


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  NeuroMark EEG — Seizure Stage Detection System")
    print("═" * 60)
    print(f"\n  Model path  : {MODEL_FILE}")
    print(f"  Model ready : {'✓ YES' if MODEL_FILE.exists() else '✗ NO — place model_6ch.keras in models/'}")
    print(f"\n  Open in browser: http://localhost:5000")
    print("═" * 60 + "\n")

    app.run(debug=True, host="0.0.0.0", port=5000)
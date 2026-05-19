from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from ecg_ml.features import extract_ecg_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CAPTURE_FILE = PROJECT_ROOT / "tools" / "ecg_raw_capture.csv"

MODEL_FILE = PROJECT_ROOT / "models" / "trained" / "random_forest_mitdb.pkl"
SCALER_FILE = PROJECT_ROOT / "models" / "trained" / "scaler_mitdb.pkl"
ENCODER_FILE = PROJECT_ROOT / "models" / "trained" / "label_encoder_mitdb.pkl"


def load_ad8232_capture(path):
    df = pd.read_csv(path)
    df = df[df["lo_status"] == 0].copy()

    timestamps = df["timestamp_us"].values.astype(float)
    adc = df["adc"].values.astype(float)

    if len(timestamps) > 2:
        dt = np.diff(timestamps) / 1_000_000.0
        fs = 1.0 / np.mean(dt)
    else:
        fs = 100.0

    return adc, fs


def main():
    if not CAPTURE_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {CAPTURE_FILE}")

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    encoder = joblib.load(ENCODER_FILE)

    signal, fs = load_ad8232_capture(CAPTURE_FILE)

    features, result = extract_ecg_features(signal, fs)

    feature_df = pd.DataFrame([features])

    # Mesmas colunas usadas no treino
    expected_features = [
        "fs",
        "num_r_peaks",
        "num_rr_intervals",
        "rr_mean",
        "rr_std",
        "rr_min",
        "rr_max",
        "rr_cv",
        "bpm_mean",
        "bpm_std",
        "bpm_min",
        "bpm_max",
        "signal_mean",
        "signal_std",
        "signal_min",
        "signal_max",
        "signal_energy",
        "duration_sec",
        "peak_density",
    ]

    X = feature_df[expected_features]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_scaled = scaler.transform(X)

    pred_encoded = model.predict(X_scaled)[0]
    pred_label = encoder.inverse_transform([pred_encoded])[0]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[0]
        prob_table = list(zip(encoder.classes_, probs))
    else:
        prob_table = []

    print("\n=== PREDIÇÃO AD8232 ===")
    print(f"Arquivo: {CAPTURE_FILE}")
    print(f"FS estimada: {fs:.2f} Hz")
    print(f"Picos R detectados: {features['num_r_peaks']}")
    print(f"RR médio: {features['rr_mean']:.3f} s")
    print(f"BPM médio: {features['bpm_mean']:.1f}")
    print(f"RR CV: {features['rr_cv']:.3f}")
    print(f"\nClasse prevista: {pred_label}")

    if prob_table:
        print("\nProbabilidades:")
        for label, prob in prob_table:
            print(f"{label}: {prob:.3f}")

    print("\nFeatures usadas:")
    for k in expected_features:
        print(f"{k}: {features.get(k)}")


if __name__ == "__main__":
    main()
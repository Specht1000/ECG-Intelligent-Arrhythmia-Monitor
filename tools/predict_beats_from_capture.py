from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib

from scipy.signal import find_peaks, resample

from tensorflow.keras.models import load_model

sys.path.append(str(Path(__file__).resolve().parent))

from advanced_ecg_preprocessing import preprocess_ad8232


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_FILE = PROJECT_ROOT / "tools" / "ecg_raw_capture.csv"

MODEL_FILE = PROJECT_ROOT / "models" / "trained" / "heartbeat_cnn.keras"
ENCODER_FILE = PROJECT_ROOT / "models" / "trained" / "heartbeat_label_encoder.pkl"

CNN_BEAT_LEN = 252

PRE_R_SEC = 0.25
POST_R_SEC = 0.45


def load_capture(path):
    df = pd.read_csv(path)
    df = df[df["lo_status"] == 0].copy()

    adc = df["adc"].values.astype(np.float32)
    timestamps = df["timestamp_us"].values.astype(float)

    if len(timestamps) > 2:
        dt = np.diff(timestamps) / 1_000_000.0
        fs = 1.0 / np.mean(dt)
    else:
        fs = 100.0

    return adc, fs


def detect_r_peaks(signal, fs):
    x = signal.astype(float)

    energy = x ** 2

    threshold = np.mean(energy) + 1.1 * np.std(energy)

    peaks, _ = find_peaks(
        energy,
        height=threshold,
        distance=int(0.35 * fs)
    )

    refined = []
    radius = int(0.10 * fs)

    for p in peaks:
        start = max(0, p - radius)
        end = min(len(x), p + radius)

        if end > start:
            local = start + np.argmax(np.abs(x[start:end]))
            refined.append(local)

    return np.array(sorted(set(refined)))


def normalize_beat(beat):
    beat = beat.astype(np.float32)
    beat = beat - np.mean(beat)

    std = np.std(beat)

    if std > 1e-6:
        beat = beat / std

    return beat


def main():
    print("Carregando ECG AD8232...")

    raw_signal, fs = load_capture(CSV_FILE)

    print(f"FS estimada: {fs:.2f} Hz")
    print(f"Amostras: {len(raw_signal)}")

    processed = preprocess_ad8232(raw_signal, fs)

    r_peaks = detect_r_peaks(processed, fs)

    print(f"Picos R detectados: {len(r_peaks)}")

    model = load_model(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)

    pre_samples = int(PRE_R_SEC * fs)
    post_samples = int(POST_R_SEC * fs)

    predictions = []

    print("\n=== BATIMENTOS CLASSIFICADOS ===\n")

    for i, peak in enumerate(r_peaks):
        start = peak - pre_samples
        end = peak + post_samples

        if start < 0 or end > len(processed):
            continue

        beat = processed[start:end]

        if len(beat) < 10:
            continue

        beat = normalize_beat(beat)
        beat = resample(beat, CNN_BEAT_LEN)
        beat = normalize_beat(beat)

        beat_input = beat.reshape(1, CNN_BEAT_LEN, 1)

        pred = model.predict(beat_input, verbose=0)

        pred_class = int(np.argmax(pred))
        label = encoder.inverse_transform([pred_class])[0]
        confidence = float(np.max(pred))

        if confidence >= 0.70:
            predictions.append(label)

        print(f"Beat {i+1:02d} | {label:18s} | confidence={confidence:.3f}")

    print("\n=== RESUMO ===")

    if len(predictions) == 0:
        print("Nenhum batimento válido classificado.")
        return

    predictions = np.array(predictions)

    unique, counts = np.unique(predictions, return_counts=True)
    total = len(predictions)

    for label, count in zip(unique, counts):
        percent = 100.0 * count / total
        print(f"{label}: {count} ({percent:.1f}%)")

    normal_count = np.sum(predictions == "normal")
    supra_count = np.sum(predictions == "supraventricular")
    vent_count = np.sum(predictions == "ventricular")

    normal_pct = normal_count / total
    supra_pct = supra_count / total
    vent_pct = vent_count / total

    print("\n=== DECISÃO FINAL ===")

    if normal_pct >= 0.70:
        final_decision = "normal"

    elif vent_pct >= 0.20:
        final_decision = "alerta_ventricular"

    elif supra_pct >= 0.20:
        final_decision = "possivel_supraventricular"

    else:
        final_decision = "inconclusivo"

    print(f"Decisão: {final_decision}")


if __name__ == "__main__":
    main()
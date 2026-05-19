from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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
CONF_THRESHOLD = 0.70


COLORS = {
    "normal": "green",
    "supraventricular": "orange",
    "ventricular": "red",
    "low_confidence": "gray",
}


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


def classify_beats(processed, r_peaks, fs, model, encoder):
    pre_samples = int(PRE_R_SEC * fs)
    post_samples = int(POST_R_SEC * fs)

    classified = []

    for i, peak in enumerate(r_peaks):
        start = peak - pre_samples
        end = peak + post_samples

        if start < 0 or end > len(processed):
            continue

        beat = processed[start:end]

        beat = normalize_beat(beat)
        beat = resample(beat, CNN_BEAT_LEN)
        beat = normalize_beat(beat)

        beat_input = beat.reshape(1, CNN_BEAT_LEN, 1)

        pred = model.predict(beat_input, verbose=0)
        pred_class = int(np.argmax(pred))
        label = encoder.inverse_transform([pred_class])[0]
        confidence = float(np.max(pred))

        final_label = label if confidence >= CONF_THRESHOLD else "low_confidence"

        classified.append({
            "index": i + 1,
            "peak": peak,
            "label": label,
            "final_label": final_label,
            "confidence": confidence,
            "beat": beat,
        })

    return classified


def plot_full_ecg(processed, classified, fs):
    t = np.arange(len(processed)) / fs

    plt.figure(figsize=(14, 5))
    plt.plot(t, processed, color="black", linewidth=1, alpha=0.7, label="ECG processado")

    for item in classified:
        peak = item["peak"]
        label = item["final_label"]
        color = COLORS.get(label, "blue")

        plt.scatter(
            peak / fs,
            processed[peak],
            color=color,
            s=60,
            marker="x"
        )

        plt.text(
            peak / fs,
            processed[peak],
            item["label"][0].upper(),
            fontsize=9,
            color=color
        )

    plt.title("ECG AD8232 com batimentos classificados")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude processada")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_beats_grid(classified, max_beats=20):
    selected = classified[:max_beats]

    if not selected:
        print("Nenhum batimento para visualizar.")
        return

    cols = 4
    rows = int(np.ceil(len(selected) / cols))

    plt.figure(figsize=(14, 3 * rows))

    for idx, item in enumerate(selected):
        plt.subplot(rows, cols, idx + 1)

        label = item["final_label"]
        color = COLORS.get(label, "blue")

        plt.plot(item["beat"], color=color)
        plt.title(
            f"Beat {item['index']} | {item['label']}\nconf={item['confidence']:.2f}"
        )
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def print_summary(classified):
    valid = [x for x in classified if x["final_label"] != "low_confidence"]

    print("\n=== RESUMO COM CONFIANÇA >= 0.70 ===")

    if not valid:
        print("Nenhum batimento confiável.")
        return

    labels = np.array([x["final_label"] for x in valid])
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    for label, count in zip(unique, counts):
        print(f"{label}: {count} ({100 * count / total:.1f}%)")

    normal_pct = np.sum(labels == "normal") / total
    supra_pct = np.sum(labels == "supraventricular") / total
    vent_pct = np.sum(labels == "ventricular") / total

    print("\n=== DECISÃO FINAL ===")

    if normal_pct >= 0.70:
        decision = "normal"
    elif vent_pct >= 0.20:
        decision = "alerta_ventricular"
    elif supra_pct >= 0.20:
        decision = "possivel_supraventricular"
    else:
        decision = "inconclusivo"

    print(f"Decisão: {decision}")


def main():
    raw_signal, fs = load_capture(CSV_FILE)

    print(f"FS estimada: {fs:.2f} Hz")
    print(f"Amostras: {len(raw_signal)}")

    processed = preprocess_ad8232(raw_signal, fs)
    r_peaks = detect_r_peaks(processed, fs)

    print(f"Picos R detectados: {len(r_peaks)}")

    model = load_model(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)

    classified = classify_beats(processed, r_peaks, fs, model, encoder)

    for item in classified:
        print(
            f"Beat {item['index']:02d} | "
            f"{item['label']:18s} | "
            f"conf={item['confidence']:.3f} | "
            f"final={item['final_label']}"
        )

    print_summary(classified)

    plot_full_ecg(processed, classified, fs)
    plot_beats_grid(classified, max_beats=24)


if __name__ == "__main__":
    main()
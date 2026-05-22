from pathlib import Path
import sys
from collections import deque

import serial
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, resample
from tensorflow.keras.models import load_model

sys.path.append(str(Path(__file__).resolve().parent))

PORT = "COM9"
BAUD = 115200

FS = 250.0
WINDOW_SEC = 12
WINDOW_SIZE = int(FS * WINDOW_SEC)

CNN_BEAT_LEN = 252
PRE_R_SEC = 0.25
POST_R_SEC = 0.45
CONF_THRESHOLD = 0.70
PLOT_SCALE = 1.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = PROJECT_ROOT / "models" / "trained" / "heartbeat_cnn.keras"
ENCODER_FILE = PROJECT_ROOT / "models" / "trained" / "heartbeat_label_encoder.pkl"
QUALITY_MODEL_FILE = PROJECT_ROOT / "models" / "trained" / "signal_quality_model.pkl"

raw_buffer = deque(maxlen=WINDOW_SIZE)
classified_history = deque(maxlen=30)
last_classified_peak = -999999

model = load_model(MODEL_FILE)
encoder = joblib.load(ENCODER_FILE)
quality_model = joblib.load(QUALITY_MODEL_FILE)

ser = serial.Serial(PORT, BAUD, timeout=1)


def preprocess_raw_only(raw_array):
    signal = raw_array.astype(np.float32)

    signal = signal - np.mean(signal)

    std = np.std(signal)

    if std > 1e-6:
        signal = signal / std

    return signal


def extract_quality_features(raw_array):
    diff = np.diff(raw_array)

    features = np.array([[
        np.std(raw_array),
        np.max(raw_array) - np.min(raw_array),
        np.std(diff),
        np.sum((raw_array - np.mean(raw_array)) ** 2) / len(raw_array),
        np.sum((raw_array <= 10) | (raw_array >= 32760)),
    ]])

    return features


def detect_r_peaks(signal, fs):
    energy = signal ** 2
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
        end = min(len(signal), p + radius)

        if end > start:
            local = start + np.argmax(np.abs(signal[start:end]))
            refined.append(local)

    return np.array(sorted(set(refined)))


def normalize_beat(beat):
    beat = beat.astype(np.float32)
    beat -= np.mean(beat)

    std = np.std(beat)

    if std > 1e-6:
        beat /= std

    return beat


def classify_beat(processed, peak, fs):
    pre = int(PRE_R_SEC * fs)
    post = int(POST_R_SEC * fs)

    start = peak - pre
    end = peak + post

    if start < 0 or end > len(processed):
        return None

    beat = processed[start:end]

    beat = normalize_beat(beat)
    beat = resample(beat, CNN_BEAT_LEN)
    beat = normalize_beat(beat)

    x = beat.reshape(1, CNN_BEAT_LEN, 1)

    pred = model.predict(x, verbose=0)

    pred_class = int(np.argmax(pred))
    label = encoder.inverse_transform([pred_class])[0]
    confidence = float(np.max(pred))

    final_label = label if confidence >= CONF_THRESHOLD else "low_confidence"

    return final_label, label, confidence


plt.ion()

fig, ax = plt.subplots(figsize=(12, 5))

line, = ax.plot([], [], color="black")
scatter = ax.scatter([], [], color="green")

ax.set_title("ECG bruto normalizado + IA em tempo real")
ax.set_xlabel("Amostras")
ax.set_ylabel("Amplitude normalizada")
ax.grid(True)

print("Lendo ECG em tempo real SEM FILTRO... Ctrl+C para sair.")

try:
    while True:
        raw_line = ser.readline().decode(errors="ignore").strip()

        if not raw_line or raw_line.startswith("["):
            continue

        parts = raw_line.split(",")

        try:
            raw = int(parts[1])
        except Exception:
            continue

        raw_buffer.append(raw)

        if len(raw_buffer) < WINDOW_SIZE:
            continue

        raw_array = np.array(raw_buffer, dtype=np.float32)

        quality_features = extract_quality_features(raw_array)
        signal_quality = quality_model.predict(quality_features)[0]

        processed = preprocess_raw_only(raw_array)

        peaks = detect_r_peaks(processed, FS)

        if len(peaks) == 0:
            continue

        latest_peak = peaks[-1]
        global_index = len(raw_buffer) - WINDOW_SIZE + latest_peak

        if signal_quality != "bad":
            if abs(global_index - last_classified_peak) > int(0.30 * FS):
                result = classify_beat(processed, latest_peak, FS)

                if result is not None:
                    final_label, raw_label, confidence = result
                    last_classified_peak = global_index

                    classified_history.append(final_label)

                    print(
                        f"Quality={signal_quality:10s} | "
                        f"raw={raw_label:18s} | "
                        f"final={final_label:18s} | "
                        f"conf={confidence:.3f}"
                    )
        else:
            print("Quality=bad | classificação cardíaca ignorada")

        labels = [x for x in classified_history if x != "low_confidence"]

        if labels:
            unique, counts = np.unique(labels, return_counts=True)
            dominant = unique[np.argmax(counts)]
        else:
            dominant = "inconclusivo"

        processed_plot = processed / PLOT_SCALE

        x_axis = np.arange(len(processed_plot))
        line.set_data(x_axis, processed_plot)

        ax.set_xlim(0, len(processed_plot))

        ymin = np.min(processed_plot)
        ymax = np.max(processed_plot)

        ax.set_ylim(ymin - 1, ymax + 1)

        peak_y = processed_plot[peaks]

        scatter.remove()

        point_color = "green"

        if signal_quality == "acceptable":
            point_color = "orange"
        elif signal_quality == "bad":
            point_color = "red"

        scatter = ax.scatter(peaks, peak_y, color=point_color, marker="x")

        ax.set_title(
            f"ECG bruto normalizado | qualidade={signal_quality} | decisão={dominant}"
        )

        plt.pause(0.001)

except KeyboardInterrupt:
    print("Encerrado.")

finally:
    ser.close()
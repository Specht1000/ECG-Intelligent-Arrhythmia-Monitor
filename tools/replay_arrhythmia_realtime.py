from pathlib import Path
from collections import deque
import time

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, resample
from tensorflow.keras.models import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MITDB_FEATURE_FILE = PROJECT_ROOT / "data" / "processed" / "mitdb_feature_dataset.csv"
CNN_MODEL_FILE = PROJECT_ROOT / "models" / "trained" / "heartbeat_cnn.keras"
ENCODER_FILE = PROJECT_ROOT / "models" / "trained" / "heartbeat_label_encoder.pkl"

FS = 250.0
WINDOW_SEC = 12
WINDOW_SIZE = int(FS * WINDOW_SEC)

CNN_BEAT_LEN = 252
CONF_THRESHOLD = 0.70

SIM_LABEL = "ventricular"  # normal, supraventricular, ventricular
SIM_DURATION_SEC = 30


model = load_model(CNN_MODEL_FILE)
encoder = joblib.load(ENCODER_FILE)


def generate_synthetic_ecg(label, fs, duration_sec):
    t = np.arange(0, duration_sec, 1 / fs)

    if label == "normal":
        bpm = 75
        noise = 0.08
    elif label == "supraventricular":
        bpm = 115
        noise = 0.12
    elif label == "ventricular":
        bpm = 90
        noise = 0.18
    else:
        bpm = 75
        noise = 0.1

    rr = 60.0 / bpm
    signal = np.zeros_like(t)

    peak_times = np.arange(0.8, duration_sec, rr)

    for pt in peak_times:
        center = int(pt * fs)

        if center <= 0 or center >= len(signal):
            continue

        width = int(0.04 * fs)

        if label == "ventricular":
            width = int(0.09 * fs)

        start = max(0, center - 3 * width)
        end = min(len(signal), center + 3 * width)

        x = np.arange(start, end) - center

        qrs = np.exp(-(x ** 2) / (2 * width ** 2))

        if label == "ventricular":
            qrs *= 1.8
            qrs -= 0.7 * np.exp(-((x - width) ** 2) / (2 * (width * 1.5) ** 2))

        signal[start:end] += qrs

    baseline = 0.2 * np.sin(2 * np.pi * 0.25 * t)
    muscle_noise = noise * np.random.randn(len(t))

    signal = signal + baseline + muscle_noise
    signal = signal - np.mean(signal)

    std = np.std(signal)
    if std > 1e-6:
        signal = signal / std

    return signal


def detect_r_peaks(signal, fs):
    energy = signal ** 2
    threshold = np.mean(energy) + 1.2 * np.std(energy)

    peaks, _ = find_peaks(
        energy,
        height=threshold,
        distance=int(0.45 * fs)
    )

    refined = []
    radius = int(0.08 * fs)

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


def classify_beat(signal, peak, fs):
    pre = int(0.25 * fs)
    post = int(0.45 * fs)

    start = peak - pre
    end = peak + post

    if start < 0 or end > len(signal):
        return None

    beat = signal[start:end]
    beat = normalize_beat(beat)
    beat = resample(beat, CNN_BEAT_LEN)
    beat = normalize_beat(beat)

    x = beat.reshape(1, CNN_BEAT_LEN, 1)

    pred = model.predict(x, verbose=0)
    cls = int(np.argmax(pred))
    label = encoder.inverse_transform([cls])[0]
    confidence = float(np.max(pred))

    final = label if confidence >= CONF_THRESHOLD else "low_confidence"

    return final, label, confidence


def estimate_bpm(peaks, fs):
    if len(peaks) < 2:
        return 0.0

    rr = np.diff(peaks) / fs
    rr_valid = rr[(rr >= 0.35) & (rr <= 1.8)]

    if len(rr_valid) == 0:
        return 0.0

    return 60.0 / np.mean(rr_valid)


def main():
    signal = generate_synthetic_ecg(SIM_LABEL, FS, SIM_DURATION_SEC)

    buffer = deque(maxlen=WINDOW_SIZE)
    history = deque(maxlen=30)

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 5))

    line, = ax.plot([], [], color="black")
    scatter = ax.scatter([], [], color="green", marker="x")

    ax.set_xlabel("Amostras")
    ax.set_ylabel("Amplitude normalizada")
    ax.grid(True)

    print("Replay MIT-BIH/sintético em tempo real...")
    print(f"Classe simulada: {SIM_LABEL}")

    last_peak_global = -999999

    try:
        for i, sample in enumerate(signal):
            buffer.append(sample)

            if len(buffer) < WINDOW_SIZE:
                time.sleep(1 / FS)
                continue

            window = np.array(buffer, dtype=np.float32)

            peaks = detect_r_peaks(window, FS)
            bpm = estimate_bpm(peaks, FS)

            if len(peaks) > 0:
                latest_peak = peaks[-1]
                global_peak = i - WINDOW_SIZE + latest_peak

                if abs(global_peak - last_peak_global) > int(0.35 * FS):
                    result = classify_beat(window, latest_peak, FS)

                    if result is not None:
                        final, raw_label, conf = result
                        last_peak_global = global_peak

                        history.append(final)

                        print(
                            f"Beat | esperado={SIM_LABEL:18s} | "
                            f"raw={raw_label:18s} | "
                            f"final={final:18s} | "
                            f"conf={conf:.3f}"
                        )

            valid = [x for x in history if x != "low_confidence"]

            if valid:
                unique, counts = np.unique(valid, return_counts=True)
                decision = unique[np.argmax(counts)]
            else:
                decision = "inconclusivo"

            x_axis = np.arange(len(window))
            line.set_data(x_axis, window)

            ax.set_xlim(0, len(window))
            ax.set_ylim(np.min(window) - 1, np.max(window) + 1)

            scatter.remove()

            if len(peaks) > 0:
                scatter = ax.scatter(peaks, window[peaks], color="green", marker="x")
            else:
                scatter = ax.scatter([], [], color="green", marker="x")

            ax.set_title(
                f"Replay ECG | esperado={SIM_LABEL} | decisão={decision} | BPM={bpm:.1f}"
            )

            plt.pause(0.001)
            time.sleep(1 / FS)

    except KeyboardInterrupt:
        print("Encerrado.")


if __name__ == "__main__":
    main()
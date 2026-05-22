from pathlib import Path
from collections import deque
import time

import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.signal import resample
from tensorflow.keras.models import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]

X_FILE = PROJECT_ROOT / "data" / "processed" / "heartbeat_dataset" / "X_beats.npy"
Y_FILE = PROJECT_ROOT / "data" / "processed" / "heartbeat_dataset" / "y_beats.npy"

MODEL_FILE = PROJECT_ROOT / "models" / "trained" / "heartbeat_cnn.keras"
ENCODER_FILE = PROJECT_ROOT / "models" / "trained" / "heartbeat_label_encoder.pkl"

CNN_BEAT_LEN = 252
CONF_THRESHOLD = 0.70

TARGET_CLASS = "ventricular"  # normal / supraventricular / ventricular
MAX_BEATS = 200

model = load_model(MODEL_FILE)
encoder = joblib.load(ENCODER_FILE)

X = np.load(X_FILE)
y = np.load(Y_FILE, allow_pickle=True)

indices = np.where(y == TARGET_CLASS)[0]

if len(indices) == 0:
    raise RuntimeError(f"Nenhum batimento encontrado para {TARGET_CLASS}")

indices = indices[:MAX_BEATS]


def normalize(signal):
    signal = signal.astype(np.float32)
    signal -= np.mean(signal)

    std = np.std(signal)

    if std > 1e-6:
        signal /= std

    return signal


def classify_beat(beat):
    beat = normalize(beat)
    beat = resample(beat, CNN_BEAT_LEN)
    beat = normalize(beat)

    x = beat.reshape(1, CNN_BEAT_LEN, 1)

    pred = model.predict(x, verbose=0)

    cls = int(np.argmax(pred))
    label = encoder.inverse_transform([cls])[0]

    confidence = float(np.max(pred))

    final = label if confidence >= CONF_THRESHOLD else "low_confidence"

    return final, label, confidence


plt.ion()

fig, ax = plt.subplots(figsize=(12, 5))

line, = ax.plot([], [], color="black")
scatter = ax.scatter([], [], color="green", marker="x")

ax.set_xlabel("Amostras")
ax.set_ylabel("Amplitude normalizada")
ax.grid(True)

print("Replay MIT-BIH REAL por batimentos")
print(f"Classe alvo: {TARGET_CLASS}")
print(f"Batimentos disponíveis: {len(indices)}")

history = deque(maxlen=30)

try:
    for idx in indices:
        beat = X[idx].astype(np.float32)
        beat = normalize(beat)

        final, raw_label, conf = classify_beat(beat)

        history.append(final)

        valid = [x for x in history if x != "low_confidence"]

        if valid:
            unique, counts = np.unique(valid, return_counts=True)
            decision = unique[np.argmax(counts)]
        else:
            decision = "inconclusivo"

        print(
            f"Beat | esperado={TARGET_CLASS:18s} | "
            f"raw={raw_label:18s} | "
            f"final={final:18s} | "
            f"conf={conf:.3f}"
        )

        x_axis = np.arange(len(beat))

        line.set_data(x_axis, beat)

        ax.set_xlim(0, len(beat))
        ax.set_ylim(np.min(beat) - 1, np.max(beat) + 1)

        scatter.remove()

        peak = int(np.argmax(np.abs(beat)))

        scatter = ax.scatter(
            [peak],
            [beat[peak]],
            color="green",
            marker="x"
        )

        ax.set_title(
            f"MIT-BIH REAL | esperado={TARGET_CLASS} | decisão={decision}"
        )

        plt.pause(0.001)
        time.sleep(0.25)

except KeyboardInterrupt:
    print("Encerrado.")
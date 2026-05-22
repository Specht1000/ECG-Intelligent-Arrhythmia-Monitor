from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import resample

MODEL_PATH = "models/trained/heartbeat_cnn.keras"
BEATS_DIR = Path("data/real_beats")

CNN_BEAT_LEN = 252

CLASS_NAMES = [
    "normal",
    "supraventricular",
    "ventricular"
]

model = tf.keras.models.load_model(MODEL_PATH)

files = sorted(BEATS_DIR.glob("*.csv"))

print("=" * 50)
print("CLASSIFICAÇÃO DE BATIMENTOS REAIS")
print("=" * 50)

counts = {
    "normal": 0,
    "supraventricular": 0,
    "ventricular": 0,
    "low_confidence": 0,
}

for file in files:

    df = pd.read_csv(file)

    beat = df["value"].values.astype(np.float32)

    beat = resample(beat, CNN_BEAT_LEN)

    beat -= np.mean(beat)

    std = np.std(beat)
    if std > 1e-6:
        beat /= std

    x = beat.reshape(1, CNN_BEAT_LEN, 1)

    pred = model.predict(x, verbose=0)[0]

    idx = int(np.argmax(pred))

    label = CLASS_NAMES[idx]

    conf = float(pred[idx])

    final_label = label

    if conf < 0.70:
        final_label = "low_confidence"

    counts[final_label] += 1

    print(
        f"{file.name:<20} "
        f"class={label:<18} "
        f"conf={conf:.3f} "
        f"final={final_label}"
    )

print("\nResumo:")
for k, v in counts.items():
    print(f"{k}: {v}")

# Mostra só os primeiros 12 para não abrir 50 janelas
for file in files[:12]:

    df = pd.read_csv(file)

    beat = df["value"].values.astype(np.float32)
    beat = resample(beat, CNN_BEAT_LEN)

    beat -= np.mean(beat)

    std = np.std(beat)
    if std > 1e-6:
        beat /= std

    x = beat.reshape(1, CNN_BEAT_LEN, 1)

    pred = model.predict(x, verbose=0)[0]

    idx = int(np.argmax(pred))
    label = CLASS_NAMES[idx]
    conf = float(pred[idx])

    plt.figure(figsize=(8, 3))
    plt.plot(beat)
    plt.title(f"{file.name} | {label} | conf={conf:.3f}")
    plt.grid(True)
    plt.show()
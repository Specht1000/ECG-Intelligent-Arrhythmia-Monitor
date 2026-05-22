from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REAL_CAPTURE = PROJECT_ROOT / "data" / "real_captures" / "06_normal.csv"

print("Gerando figuras do paper...")

df = pd.read_csv(REAL_CAPTURE)

signal = df["adc"].values.astype(np.float32)

# =========================================================
# FIGURA 1 - SINAL BRUTO
# =========================================================

plt.figure(figsize=(12, 4))
plt.plot(signal[:2500], linewidth=1)

plt.title("Sinal ECG bruto adquirido com AD8232 + ADS1115")
plt.xlabel("Amostras")
plt.ylabel("ADC")

plt.grid(True)

file1 = OUT_DIR / "figure_raw_ecg.png"
plt.savefig(file1, dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# FIGURA 2 - NORMALIZAÇÃO
# =========================================================

norm = signal.copy()

norm -= np.mean(norm)

std = np.std(norm)
if std > 1e-6:
    norm /= std

plt.figure(figsize=(12, 4))
plt.plot(norm[:2500], linewidth=1)

plt.title("Sinal ECG normalizado")
plt.xlabel("Amostras")
plt.ylabel("Amplitude normalizada")

plt.grid(True)

file2 = OUT_DIR / "figure_normalized_ecg.png"
plt.savefig(file2, dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# FIGURA 3 - BATIMENTO VENTRICULAR
# =========================================================

beat_file = PROJECT_ROOT / "data" / "processed" / "heartbeat_dataset"

X = np.load(beat_file / "X_beats.npy")
y = np.load(beat_file / "y_beats.npy", allow_pickle=True)

ventricular_idx = np.where(y == "ventricular")[0][0]

beat = X[ventricular_idx]

beat -= np.mean(beat)

std = np.std(beat)
if std > 1e-6:
    beat /= std

plt.figure(figsize=(8, 4))
plt.plot(beat, linewidth=2)

peak = np.argmax(np.abs(beat))

plt.scatter(
    peak,
    beat[peak],
    color="red",
    marker="x",
    s=100
)

plt.title("Exemplo de batimento ventricular")
plt.xlabel("Amostras")
plt.ylabel("Amplitude normalizada")

plt.grid(True)

file3 = OUT_DIR / "figure_ventricular_beat.png"
plt.savefig(file3, dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# FIGURA 4 - BATIMENTO NORMAL
# =========================================================

normal_idx = np.where(y == "normal")[0][0]

beat = X[normal_idx]

beat -= np.mean(beat)

std = np.std(beat)
if std > 1e-6:
    beat /= std

plt.figure(figsize=(8, 4))
plt.plot(beat, linewidth=2)

peak = np.argmax(np.abs(beat))

plt.scatter(
    peak,
    beat[peak],
    color="green",
    marker="x",
    s=100
)

plt.title("Exemplo de batimento normal")
plt.xlabel("Amostras")
plt.ylabel("Amplitude normalizada")

plt.grid(True)

file4 = OUT_DIR / "figure_normal_beat.png"
plt.savefig(file4, dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# RESUMO
# =========================================================

print("\nFiguras geradas:")
print(file1)
print(file2)
print(file3)
print(file4)
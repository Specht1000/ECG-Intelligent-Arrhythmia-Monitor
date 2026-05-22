from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BEATS_DIR = Path("data/real_beats")
OUT_DIR = Path("reports/final")

OUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(BEATS_DIR.glob("*.csv"))

all_beats = []

for file in files:

    df = pd.read_csv(file)

    beat = df["value"].values.astype(np.float32)

    all_beats.append(beat)

all_beats = np.array(all_beats)

mean_beat = np.mean(all_beats, axis=0)
std_beat = np.std(all_beats, axis=0)

# =========================================================
# MÉTRICAS
# =========================================================

print("=" * 50)
print("RELATÓRIO FINAL")
print("=" * 50)

print(f"Batimentos reais analisados: {len(all_beats)}")
print(f"Amostras por beat: {all_beats.shape[1]}")

print("\nEstabilidade média:")
print(f"Média global: {np.mean(all_beats):.4f}")
print(f"Desvio padrão global: {np.std(all_beats):.4f}")

# =========================================================
# BEAT MÉDIO
# =========================================================

plt.figure(figsize=(12, 5))

plt.plot(mean_beat, label="Beat médio")

plt.fill_between(
    np.arange(len(mean_beat)),
    mean_beat - std_beat,
    mean_beat + std_beat,
    alpha=0.3,
    label="±1 desvio padrão"
)

plt.title("Batimento médio real - AD8232 + ADS1115")
plt.xlabel("Amostras")
plt.ylabel("Amplitude normalizada")

plt.grid(True)
plt.legend()

out1 = OUT_DIR / "mean_real_beat.png"

plt.savefig(out1, dpi=300, bbox_inches="tight")

print(f"\nFigura salva: {out1}")

# =========================================================
# SOBREPOSIÇÃO DOS BEATS
# =========================================================

plt.figure(figsize=(12, 5))

for beat in all_beats:
    plt.plot(beat, alpha=0.25)

plt.title("Sobreposição dos batimentos reais")
plt.xlabel("Amostras")
plt.ylabel("Amplitude normalizada")

plt.grid(True)

out2 = OUT_DIR / "overlay_real_beats.png"

plt.savefig(out2, dpi=300, bbox_inches="tight")

print(f"Figura salva: {out2}")

# =========================================================
# HISTOGRAMA
# =========================================================

plt.figure(figsize=(10, 5))

plt.hist(all_beats.flatten(), bins=100)

plt.title("Distribuição das amplitudes do ECG")
plt.xlabel("Amplitude")
plt.ylabel("Frequência")

plt.grid(True)

out3 = OUT_DIR / "histogram_real_ecg.png"

plt.savefig(out3, dpi=300, bbox_inches="tight")

print(f"Figura salva: {out3}")

plt.show()
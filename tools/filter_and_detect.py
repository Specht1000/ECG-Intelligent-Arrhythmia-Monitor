import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILE = "ecg_raw_capture.csv"
FS = 100  # pelo teu resultado: 3001 amostras / 30 s ≈ 100 Hz

df = pd.read_csv(FILE)
df = df[df["lo_status"] == 0].copy()

x = df["adc"].values.astype(float)

# Remove offset DC
x_centered = x - np.mean(x)

# Média móvel simples para suavizar
def moving_average(signal, window):
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")

# Suavização
y = moving_average(x_centered, 5)

# Realce simples: derivada + quadrado, estilo ideia do Pan-Tompkins
d = np.diff(y, prepend=y[0])
squared = d ** 2
integrated = moving_average(squared, 8)

# Limiar adaptado simples
threshold = np.mean(integrated) + 1.5 * np.std(integrated)

peaks = []
min_distance = int(0.35 * FS)  # evita contar dois picos muito próximos
last_peak = -min_distance

for i in range(1, len(integrated) - 1):
    if integrated[i] > threshold and integrated[i] > integrated[i - 1] and integrated[i] > integrated[i + 1]:
        if i - last_peak >= min_distance:
            peaks.append(i)
            last_peak = i

peaks = np.array(peaks)

duration_sec = len(x) / FS

if len(peaks) >= 2:
    rr = np.diff(peaks) / FS
    bpm = 60 / np.mean(rr)
else:
    bpm = np.nan

print("Amostras:", len(x))
print("Duração estimada:", duration_sec, "s")
print("Picos detectados:", len(peaks))
print("BPM estimado:", bpm)

plt.figure(figsize=(12, 5))
plt.plot(x_centered, label="ECG bruto centralizado", alpha=0.5)
plt.plot(y, label="ECG suavizado", linewidth=2)

if len(peaks) > 0:
    plt.scatter(peaks, y[peaks], marker="x", label="Picos detectados")

plt.title(f"ECG filtrado/suavizado - BPM estimado: {bpm:.1f}")
plt.xlabel("Amostras")
plt.ylabel("Amplitude relativa")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(integrated, label="Sinal integrado")
plt.axhline(threshold, linestyle="--", label="Limiar")
plt.title("Realce estilo Pan-Tompkins")
plt.xlabel("Amostras")
plt.grid(True)
plt.legend()
plt.show()
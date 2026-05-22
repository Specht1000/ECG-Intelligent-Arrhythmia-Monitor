from pathlib import Path
from collections import deque

import serial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PORT = "COM9"
BAUD = 115200

PRE = 70
POST = 180
MAX_BEATS = 50

OUT_DIR = Path("data/real_beats")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ser = serial.Serial(PORT, BAUD, timeout=1)

samples = []
pending_peaks = []
saved = 0
last_status = 0

plt.ion()
fig, ax = plt.subplots(figsize=(10, 4))

print("Capturando batimentos reais...")
print("IMPORTANTE: feche Termite e Serial Monitor antes.")
print("Aguardando RPEAK...")

try:
    while saved < MAX_BEATS:
        line = ser.readline().decode(errors="ignore").strip()

        if not line:
            continue

        # CSV: timestamp,raw,integrated,r_peak,bpm,threshold
        if not line.startswith("["):
            parts = line.split(",")

            if len(parts) >= 6:
                try:
                    raw = float(parts[1])
                    r_peak = int(parts[3])
                except Exception:
                    continue

                samples.append(raw)

                if r_peak == 1:
                    peak_index = len(samples) - 1
                    pending_peaks.append(peak_index)
                    print(f"RPEAK CSV detectado em amostra {peak_index}")

        # Log legível: [RPEAK] Pico R detectado...
        elif line.startswith("[RPEAK]"):
            peak_index = len(samples) - 1
            if peak_index > 0:
                pending_peaks.append(peak_index)
                print(f"RPEAK LOG detectado em amostra {peak_index}")

        # salva beats pendentes quando já existem amostras depois do pico
        new_pending = []

        for peak_index in pending_peaks:
            if len(samples) < peak_index + POST:
                new_pending.append(peak_index)
                continue

            start = peak_index - PRE
            end = peak_index + POST

            if start < 0 or end > len(samples):
                continue

            beat = np.array(samples[start:end], dtype=np.float32)

            if len(beat) != PRE + POST:
                continue

            beat -= np.mean(beat)

            std = np.std(beat)
            if std > 1e-6:
                beat /= std

            out_file = OUT_DIR / f"beat_{saved:05d}.csv"

            pd.DataFrame({
                "sample": np.arange(len(beat)),
                "value": beat
            }).to_csv(out_file, index=False)

            saved += 1

            ax.clear()
            ax.plot(beat)
            ax.set_title(f"Batimento real salvo #{saved}")
            ax.set_xlabel("Amostras")
            ax.set_ylabel("Amplitude normalizada")
            ax.grid(True)
            plt.pause(0.001)

            print(f"Beat salvo: {out_file}")

        pending_peaks = new_pending

        if len(samples) - last_status >= 500:
            last_status = len(samples)
            print(f"Amostras recebidas: {len(samples)} | beats salvos: {saved}")

except KeyboardInterrupt:
    print("Interrompido pelo usuário.")

finally:
    ser.close()
    print("Finalizado.")
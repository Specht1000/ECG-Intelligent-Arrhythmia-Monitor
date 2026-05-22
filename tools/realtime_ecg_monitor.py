import serial
import matplotlib.pyplot as plt
from collections import deque

PORT = "COM9"
BAUD = 115200
WINDOW_SIZE = 500

ser = serial.Serial(PORT, BAUD, timeout=1)

raw_buffer = deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
integrated_buffer = deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)

plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))

line_raw, = ax1.plot(list(raw_buffer))
line_integrated, = ax2.plot(list(integrated_buffer))

ax1.set_title("ECG bruto")
ax2.set_title("Pan-Tompkins integrado")

ax1.set_ylim(1000, 3000)
ax2.set_ylim(0, 50000)

ax1.set_xlim(0, WINDOW_SIZE)
ax2.set_xlim(0, WINDOW_SIZE)

ax1.grid(True)
ax2.grid(True)

print("Lendo serial... Ctrl+C para sair.")

while True:
    try:
        line = ser.readline().decode(errors="ignore").strip()

        if not line:
            continue

        if line.startswith("["):
            print(line)
            continue

        parts = line.split(",")

        if len(parts) < 5:
            print("Linha ignorada:", line)
            continue

        timestamp = int(parts[0])
        raw = float(parts[1])
        integrated = float(parts[2])
        r_peak = int(parts[3])
        bpm = float(parts[4])

        raw_buffer.append(raw)
        integrated_buffer.append(integrated)

        line_raw.set_ydata(list(raw_buffer))
        line_integrated.set_ydata(list(integrated_buffer))

        ymin = min(raw_buffer)
        ymax = max(raw_buffer)
        if ymax > ymin:
            ax1.set_ylim(ymin - 100, ymax + 100)

        imax = max(integrated_buffer)
        ax2.set_ylim(0, max(1000, imax * 1.2))

        ax1.set_title(f"ECG bruto | BPM={bpm:.1f}")

        if r_peak == 1:
            print(f"R PEAK | BPM={bpm:.1f}")

        plt.pause(0.001)

    except KeyboardInterrupt:
        break

    except Exception as e:
        print("Erro:", e)

ser.close()
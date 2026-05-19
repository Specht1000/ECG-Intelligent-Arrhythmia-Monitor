import serial
import matplotlib.pyplot as plt
from collections import deque

PORT = "COM9"
BAUD = 115200
WINDOW = 800

ser = serial.Serial(PORT, BAUD, timeout=1)

data = deque([0] * WINDOW, maxlen=WINDOW)

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(list(data))

ax.set_title("AD8232 - ECG bruto")
ax.set_xlabel("Amostras")
ax.set_ylabel("ADC")
ax.set_ylim(0, 4095)
ax.grid(True)

print("Lendo serial... Ctrl+C para sair.")

try:
    while True:
        line_raw = ser.readline().decode(errors="ignore").strip()

        if not line_raw:
            continue

        if line_raw.startswith("[") or line_raw.startswith("#"):
            print(line_raw)
            continue

        parts = line_raw.split(",")

        if len(parts) != 3:
            continue

        try:
            timestamp_us = int(parts[0])
            adc = int(parts[1])
            lo_status = int(parts[2])
        except ValueError:
            continue

        if lo_status == 1:
            print("Eletrodo solto")
            continue

        data.append(adc)

        line.set_ydata(list(data))
        line.set_xdata(range(len(data)))

        ymin = min(data)
        ymax = max(data)

        if ymax > ymin:
            ax.set_ylim(800, 4200)

        plt.pause(0.001)

except KeyboardInterrupt:
    print("Encerrado.")

finally:
    ser.close()
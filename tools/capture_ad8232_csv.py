import serial
import time
import csv

PORT = "COM9"
BAUD = 115200
DURATION_SEC = 30
OUT_FILE = "ecg_raw_capture.csv"

ser = serial.Serial(PORT, BAUD, timeout=1)

samples = []

print("Capturando por 30 segundos...")
start = time.time()

while time.time() - start < DURATION_SEC:
    line = ser.readline().decode(errors="ignore").strip()

    if not line:
        continue

    if line.startswith("[") or line.startswith("#"):
        continue

    parts = line.split(",")

    try:
        if len(parts) == 3:
            timestamp_us = int(parts[0])
            adc = int(parts[1])
            lo_status = int(parts[2])
        else:
            timestamp_us = int((time.time() - start) * 1_000_000)
            adc = int(line)
            lo_status = 0
    except ValueError:
        continue

    samples.append([timestamp_us, adc, lo_status])

ser.close()

with open(OUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp_us", "adc", "lo_status"])
    writer.writerows(samples)

print(f"Salvo em: {OUT_FILE}")
print(f"Amostras capturadas: {len(samples)}")
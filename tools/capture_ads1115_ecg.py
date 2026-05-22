import serial
import time
import csv

PORT = "COM9"
BAUD = 115200
DURATION_SEC = 30
OUT_FILE = "ecg_raw_capture.csv"

ser = serial.Serial(PORT, BAUD, timeout=1)

samples = []

print("Capturando ECG do ADS1115 por 30 segundos...")
start = time.time()

while time.time() - start < DURATION_SEC:
    line = ser.readline().decode(errors="ignore").strip()

    if not line or line.startswith("["):
        continue

    parts = line.split(",")

    try:
        timestamp_us = int(parts[0])
        raw = int(parts[1])
        lo_status = 0
    except:
        continue

    samples.append([timestamp_us, raw, lo_status])

ser.close()

with open(OUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp_us", "adc", "lo_status"])
    writer.writerows(samples)

print(f"Salvo em: {OUT_FILE}")
print(f"Amostras: {len(samples)}")
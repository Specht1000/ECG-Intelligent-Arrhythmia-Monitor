import serial
import time
import csv
from pathlib import Path

PORT = "COM9"
BAUD = 115200
DURATION_SEC = 30

LABEL = input("Label da captura (normal, noisy, movement, breathing): ").strip()
NAME = input("Nome do arquivo sem .csv: ").strip()

OUT_DIR = Path("../data/real_captures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / f"{NAME}_{LABEL}.csv"

ser = serial.Serial(PORT, BAUD, timeout=1)

samples = []

print(f"Capturando {DURATION_SEC}s...")
start = time.time()

while time.time() - start < DURATION_SEC:
    line = ser.readline().decode(errors="ignore").strip()

    if not line or line.startswith("["):
        continue

    parts = line.split(",")

    try:
        timestamp_us = int(parts[0])
        raw = int(parts[1])
        integrated = float(parts[2])
        r_peak = int(parts[3])
        bpm = float(parts[4])
        threshold = float(parts[5])
    except:
        continue

    samples.append([
        timestamp_us,
        raw,
        integrated,
        r_peak,
        bpm,
        threshold,
        LABEL
    ])

ser.close()

with open(OUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp_us",
        "adc",
        "integrated",
        "r_peak",
        "bpm",
        "threshold",
        "label"
    ])
    writer.writerows(samples)

print(f"Salvo em: {OUT_FILE}")
print(f"Amostras: {len(samples)}")
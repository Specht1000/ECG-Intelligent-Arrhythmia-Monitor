from pathlib import Path
import pandas as pd
import numpy as np

CAPTURE_DIR = Path("../data/real_captures")
FS = 250.0
MIN_RR_SEC = 0.50   # máximo 120 bpm
MAX_RR_SEC = 1.50   # mínimo 40 bpm

files = sorted(CAPTURE_DIR.glob("*.csv"))

if not files:
    print("Nenhuma captura encontrada.")
    exit()

for file in files:
    df = pd.read_csv(file)

    r_idx = np.where(df["r_peak"].values == 1)[0]
    rr = np.diff(r_idx) / FS

    rr_valid = rr[(rr >= MIN_RR_SEC) & (rr <= MAX_RR_SEC)]

    bpm_raw = 60 / np.mean(rr) if len(rr) > 0 else np.nan
    bpm_valid = 60 / np.mean(rr_valid) if len(rr_valid) > 0 else np.nan

    print("\n============================")
    print(file.name)
    print("============================")
    print("Label:", df["label"].iloc[0])
    print("Amostras:", len(df))
    print("ADC min:", df["adc"].min())
    print("ADC max:", df["adc"].max())
    print("ADC mean:", df["adc"].mean())
    print("R peaks brutos:", len(r_idx))
    print("RR válidos:", len(rr_valid))
    print("BPM bruto:", bpm_raw)
    print("BPM filtrado:", bpm_valid)
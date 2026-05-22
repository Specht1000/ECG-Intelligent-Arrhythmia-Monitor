from pathlib import Path
import numpy as np
import pandas as pd

CAPTURE_DIR = Path("data/real_captures")
OUT_FILE = Path("data/processed/signal_quality_dataset.csv")

# None = usa todos os CSVs da pasta data/real_captures
VALID_FILES = None

QUALITY_MAP = {
    "normal": "good",
    "breathing": "acceptable",
    "movement": "acceptable",
    "noisy": "bad",
}

FS = 250.0
WINDOW_SEC = 5
WINDOW_SIZE = int(FS * WINDOW_SEC)

rows = []

for file in sorted(CAPTURE_DIR.glob("*.csv")):
    if VALID_FILES is not None and file.name not in VALID_FILES:
        continue

    df = pd.read_csv(file)

    if "label" not in df.columns:
        print(f"Ignorando {file.name}: coluna label não encontrada")
        continue

    label = str(df["label"].iloc[0]).strip()
    quality = QUALITY_MAP.get(label, "unknown")

    if quality == "unknown":
        print(f"Ignorando {file.name}: label desconhecida ({label})")
        continue

    signal = df["adc"].values.astype(float)

    for start in range(0, len(signal) - WINDOW_SIZE + 1, WINDOW_SIZE):
        end = start + WINDOW_SIZE
        segment = signal[start:end]

        if len(segment) != WINDOW_SIZE:
            continue

        diff = np.diff(segment)

        rows.append({
            "file": file.name,
            "start_sec": start / FS,
            "label": label,
            "quality": quality,
            "mean": np.mean(segment),
            "std": np.std(segment),
            "min": np.min(segment),
            "max": np.max(segment),
            "range": np.max(segment) - np.min(segment),
            "diff_std": np.std(diff),
            "energy": np.sum((segment - np.mean(segment)) ** 2) / len(segment),
            "zero_or_clip": np.sum((segment <= 10) | (segment >= 32760)),
        })

out = pd.DataFrame(rows)

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_FILE, index=False)

print(out)
print("\nDistribuição:")
print(out["quality"].value_counts())
print(f"\nTotal de janelas: {len(out)}")
print(f"Salvo em: {OUT_FILE}")
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MITDB_DIR = PROJECT_ROOT / "data" / "raw" / "mit_bih"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "heartbeat_dataset"

X_FILE = OUT_DIR / "X_beats.npy"
Y_FILE = OUT_DIR / "y_beats.npy"
META_FILE = OUT_DIR / "metadata.csv"

PRE_R_SEC = 0.25
POST_R_SEC = 0.45

LABEL_MAP = {
    "N": "normal",
    "L": "normal",
    "R": "normal",
    "e": "normal",
    "j": "normal",

    "A": "supraventricular",
    "a": "supraventricular",
    "J": "supraventricular",
    "S": "supraventricular",

    "V": "ventricular",
    "E": "ventricular",
}


def normalize_beat(beat):
    beat = beat.astype(np.float32)
    beat = beat - np.mean(beat)

    std = np.std(beat)
    if std > 1e-6:
        beat = beat / std

    return beat


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = sorted([p.stem for p in MITDB_DIR.glob("*.hea")])
    records = [r for r in records if not r.startswith("x_")]

    if not records:
        raise FileNotFoundError(f"Nenhum registro MIT-BIH encontrado em {MITDB_DIR}")

    X = []
    y = []
    metadata = []

    print("Registros encontrados:")
    print(records)

    for record_name in records:
        print(f"Processando {record_name}...")

        record_path = MITDB_DIR / record_name

        record = wfdb.rdrecord(str(record_path))
        ann = wfdb.rdann(str(record_path), "atr")

        fs = int(record.fs)
        signal_names = record.sig_name

        if "MLII" in signal_names:
            channel = signal_names.index("MLII")
        else:
            channel = 0

        signal = record.p_signal[:, channel]

        pre_samples = int(PRE_R_SEC * fs)
        post_samples = int(POST_R_SEC * fs)
        beat_len = pre_samples + post_samples

        for sample, symbol in zip(ann.sample, ann.symbol):
            if symbol not in LABEL_MAP:
                continue

            start = sample - pre_samples
            end = sample + post_samples

            if start < 0 or end > len(signal):
                continue

            beat = signal[start:end]

            if len(beat) != beat_len:
                continue

            beat = normalize_beat(beat)

            X.append(beat)
            y.append(LABEL_MAP[symbol])

            metadata.append({
                "record": record_name,
                "sample": sample,
                "symbol": symbol,
                "label": LABEL_MAP[symbol],
                "channel": signal_names[channel],
                "fs": fs,
                "start": start,
                "end": end,
            })

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    meta_df = pd.DataFrame(metadata)

    np.save(X_FILE, X)
    np.save(Y_FILE, y)
    meta_df.to_csv(META_FILE, index=False)

    print("\nDataset salvo:")
    print(f"X: {X_FILE}")
    print(f"y: {Y_FILE}")
    print(f"metadata: {META_FILE}")

    print("\nFormato X:")
    print(X.shape)

    print("\nDistribuição das classes:")
    print(meta_df["label"].value_counts())


if __name__ == "__main__":
    main()
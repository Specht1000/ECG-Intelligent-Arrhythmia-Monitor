from pathlib import Path
import sys
import numpy as np
import pandas as pd
import wfdb

sys.path.append(str(Path(__file__).resolve().parent))

from ecg_ml.features import extract_ecg_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MITDB_DIR = PROJECT_ROOT / "data" / "raw" / "mit_bih"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "mitdb_feature_dataset.csv"

WINDOW_SEC = 30


def label_from_annotations(record_path, start_sample, end_sample):
    ann = wfdb.rdann(str(record_path), "atr")

    symbols = [
        s for sample, s in zip(ann.sample, ann.symbol)
        if start_sample <= sample < end_sample
    ]

    if len(symbols) == 0:
        return "unknown"

    ventricular = {"V", "E"}
    supraventricular = {"A", "a", "J", "S"}
    normal = {"N", "L", "R", "e", "j"}

    if any(s in ventricular for s in symbols):
        return "ventricular"

    if any(s in supraventricular for s in symbols):
        return "supraventricular"

    if all(s in normal for s in symbols):
        return "normal"

    return "other"


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    records = sorted([p.stem for p in MITDB_DIR.glob("*.hea")])

    if not records:
        raise FileNotFoundError(f"Nenhum .hea encontrado em {MITDB_DIR}")

    rows = []

    print(f"Registros encontrados: {records}")

    for record_name in records:
        print(f"Processando {record_name}...")

        record_path = MITDB_DIR / record_name
        record = wfdb.rdrecord(str(record_path))

        fs = int(record.fs)
        signal_names = record.sig_name

        if "MLII" in signal_names:
            ch = signal_names.index("MLII")
        else:
            ch = 0

        signal = record.p_signal[:, ch]
        window_size = int(WINDOW_SEC * fs)

        for start in range(0, len(signal) - window_size, window_size):
            end = start + window_size
            segment = signal[start:end]

            try:
                features, _ = extract_ecg_features(segment, fs)

                features["record_name"] = record_name
                features["start_sec"] = start / fs
                features["signal_column"] = signal_names[ch]
                features["label"] = label_from_annotations(record_path, start, end)

                rows.append(features)

            except Exception as e:
                print(f"Erro em {record_name} {start/fs:.1f}s: {e}")

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    df.to_csv(OUT_FILE, index=False)

    print(f"\nDataset salvo em: {OUT_FILE}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
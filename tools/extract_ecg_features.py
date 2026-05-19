import numpy as np
import pandas as pd
from ecg_pan_tompkins import load_ecg_csv, pan_tompkins

CSV_FILE = "ecg_raw_capture.csv"
OUT_FILE = "ecg_features.csv"


def extract_features(result):
    rr = result["rr_intervals"]
    r_peaks = result["r_peaks"]
    filtered = result["filtered"]
    fs = result["fs"]

    features = {}

    features["fs"] = fs
    features["num_r_peaks"] = len(r_peaks)

    if len(rr) > 0:
        bpm_values = 60.0 / rr

        features["rr_mean"] = np.mean(rr)
        features["rr_std"] = np.std(rr)
        features["rr_min"] = np.min(rr)
        features["rr_max"] = np.max(rr)
        features["rr_cv"] = np.std(rr) / np.mean(rr)

        features["bpm_mean"] = np.mean(bpm_values)
        features["bpm_std"] = np.std(bpm_values)
        features["bpm_min"] = np.min(bpm_values)
        features["bpm_max"] = np.max(bpm_values)
    else:
        features["rr_mean"] = np.nan
        features["rr_std"] = np.nan
        features["rr_min"] = np.nan
        features["rr_max"] = np.nan
        features["rr_cv"] = np.nan

        features["bpm_mean"] = np.nan
        features["bpm_std"] = np.nan
        features["bpm_min"] = np.nan
        features["bpm_max"] = np.nan

    features["signal_mean"] = np.mean(filtered)
    features["signal_std"] = np.std(filtered)
    features["signal_min"] = np.min(filtered)
    features["signal_max"] = np.max(filtered)
    features["signal_energy"] = np.sum(filtered ** 2) / len(filtered)

    duration_sec = len(filtered) / fs
    features["duration_sec"] = duration_sec
    features["peak_density"] = len(r_peaks) / duration_sec if duration_sec > 0 else np.nan

    return features


def classify_simple(features):
    bpm = features["bpm_mean"]
    rr_cv = features["rr_cv"]

    if np.isnan(bpm):
        return "invalid"

    if bpm < 60:
        return "bradycardia"

    if bpm > 100:
        return "tachycardia"

    if not np.isnan(rr_cv) and rr_cv > 0.20:
        return "possible_irregular_rhythm"

    return "normal"


def main():
    ecg, fs = load_ecg_csv(CSV_FILE)
    result = pan_tompkins(ecg, fs)

    features = extract_features(result)
    features["simple_label"] = classify_simple(features)

    df = pd.DataFrame([features])
    df.to_csv(OUT_FILE, index=False)

    print("Features extraídas:")
    for k, v in features.items():
        print(f"{k}: {v}")

    print(f"\nArquivo salvo em: {OUT_FILE}")


if __name__ == "__main__":
    main()
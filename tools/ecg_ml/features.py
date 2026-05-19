import numpy as np

from ecg_ml.pan_tompkins import detect_r_peaks, compute_rr_intervals


def extract_ecg_features(signal, fs):
    result = detect_r_peaks(signal, fs)

    r_peaks = result["r_peaks"]
    rr = compute_rr_intervals(r_peaks, fs)
    filtered = result["filtered"]

    features = {
        "fs": fs,
        "num_r_peaks": int(len(r_peaks)),
        "num_rr_intervals": int(len(rr)),
    }

    if len(rr) > 0:
        bpm = 60.0 / rr

        features.update({
            "rr_mean": float(np.mean(rr)),
            "rr_std": float(np.std(rr)),
            "rr_min": float(np.min(rr)),
            "rr_max": float(np.max(rr)),
            "rr_cv": float(np.std(rr) / np.mean(rr)) if np.mean(rr) > 0 else np.nan,
            "bpm_mean": float(np.mean(bpm)),
            "bpm_std": float(np.std(bpm)),
            "bpm_min": float(np.min(bpm)),
            "bpm_max": float(np.max(bpm)),
        })
    else:
        features.update({
            "rr_mean": np.nan,
            "rr_std": np.nan,
            "rr_min": np.nan,
            "rr_max": np.nan,
            "rr_cv": np.nan,
            "bpm_mean": np.nan,
            "bpm_std": np.nan,
            "bpm_min": np.nan,
            "bpm_max": np.nan,
        })

    features.update({
        "signal_mean": float(np.mean(filtered)),
        "signal_std": float(np.std(filtered)),
        "signal_min": float(np.min(filtered)),
        "signal_max": float(np.max(filtered)),
        "signal_energy": float(np.sum(filtered ** 2) / len(filtered)),
        "duration_sec": float(len(filtered) / fs),
        "peak_density": float(len(r_peaks) / (len(filtered) / fs)),
    })

    return features, result
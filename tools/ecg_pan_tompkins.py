import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


def load_ecg_csv(path: str):
    df = pd.read_csv(path)
    df = df[df["lo_status"] == 0].copy()

    timestamps = df["timestamp_us"].values.astype(float)
    adc = df["adc"].values.astype(float)

    if len(timestamps) > 2:
        dt = np.diff(timestamps) / 1_000_000.0
        fs = 1.0 / np.mean(dt)
    else:
        fs = 100.0

    return adc, fs


def bandpass_filter(signal, fs, lowcut=5.0, highcut=15.0, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype="bandpass")
    return filtfilt(b, a, signal)


def moving_window_integration(signal, fs, window_ms=150):
    window_size = int((window_ms / 1000.0) * fs)
    if window_size < 1:
        window_size = 1

    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode="same")


def pan_tompkins(signal, fs):
    # 1. Remove DC
    x = signal - np.mean(signal)

    # 2. Filtro passa-faixa 5–15 Hz
    filtered = bandpass_filter(x, fs)

    # 3. Derivada
    derivative = np.diff(filtered, prepend=filtered[0])

    # 4. Quadrado
    squared = derivative ** 2

    # 5. Janela móvel
    integrated = moving_window_integration(squared, fs)

    # 6. Detecção de picos no sinal integrado
    threshold = np.mean(integrated) + 1.2 * np.std(integrated)

    min_distance = int(0.35 * fs)

    peaks_integrated, _ = find_peaks(
        integrated,
        height=threshold,
        distance=min_distance
    )

    # 7. Refinamento: procurar pico R real no sinal filtrado perto do pico integrado
    r_peaks = []

    search_radius = int(0.12 * fs)

    for p in peaks_integrated:
        start = max(0, p - search_radius)
        end = min(len(filtered), p + search_radius)

        if end <= start:
            continue

        local_peak = start + np.argmax(filtered[start:end])
        r_peaks.append(local_peak)

    r_peaks = np.array(sorted(set(r_peaks)))

    # 8. BPM
    if len(r_peaks) >= 2:
        rr_intervals = np.diff(r_peaks) / fs
        bpm = 60.0 / np.mean(rr_intervals)
    else:
        rr_intervals = np.array([])
        bpm = np.nan

    result = {
        "fs": fs,
        "raw": signal,
        "centered": x,
        "filtered": filtered,
        "derivative": derivative,
        "squared": squared,
        "integrated": integrated,
        "threshold": threshold,
        "peaks_integrated": peaks_integrated,
        "r_peaks": r_peaks,
        "rr_intervals": rr_intervals,
        "bpm": bpm,
    }

    return result
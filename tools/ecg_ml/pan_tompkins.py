import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def bandpass_filter(signal, fs, lowcut=5.0, highcut=15.0, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="bandpass")
    return filtfilt(b, a, signal)


def moving_window_integration(signal, fs, window_ms=150):
    n = max(1, int((window_ms / 1000.0) * fs))
    window = np.ones(n) / n
    return np.convolve(signal, window, mode="same")


def detect_r_peaks(signal, fs):
    x = np.asarray(signal, dtype=float)
    x = x - np.mean(x)

    filtered = bandpass_filter(x, fs)

    derivative = np.diff(filtered, prepend=filtered[0])
    squared = derivative ** 2
    integrated = moving_window_integration(squared, fs)

    threshold = np.mean(integrated) + 1.2 * np.std(integrated)
    min_distance = int(0.35 * fs)

    peaks_integrated, props = find_peaks(
        integrated,
        height=threshold,
        distance=min_distance
    )

    r_peaks = []
    search_radius = int(0.12 * fs)

    for p in peaks_integrated:
        start = max(0, p - search_radius)
        end = min(len(filtered), p + search_radius)

        if end > start:
            local = start + np.argmax(filtered[start:end])
            r_peaks.append(local)

    r_peaks = np.array(sorted(set(r_peaks)))

    return {
        "filtered": filtered,
        "derivative": derivative,
        "squared": squared,
        "integrated": integrated,
        "threshold": threshold,
        "r_peaks": r_peaks,
        "peaks_integrated": peaks_integrated,
    }


def compute_rr_intervals(r_peaks, fs):
    if len(r_peaks) < 2:
        return np.array([])
    return np.diff(r_peaks) / fs
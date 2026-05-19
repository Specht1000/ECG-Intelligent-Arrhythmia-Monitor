import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def remove_dc(signal):
    signal = np.asarray(signal, dtype=float)
    return signal - np.mean(signal)


def bandpass_ecg(signal, fs, lowcut=0.5, highcut=40.0, order=3):
    nyq = fs / 2.0

    highcut = min(highcut, nyq - 1.0)

    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype="bandpass")
    return filtfilt(b, a, signal)


def notch_filter(signal, fs, freq=50.0, q=30.0):
    nyq = fs / 2.0

    if freq >= nyq:
        return signal

    w0 = freq / nyq
    b, a = iirnotch(w0, q)
    return filtfilt(b, a, signal)


def robust_normalize(signal):
    signal = np.asarray(signal, dtype=float)

    median = np.median(signal)
    centered = signal - median

    mad = np.median(np.abs(centered)) + 1e-8

    normalized = centered / mad

    return normalized


def preprocess_ad8232(signal, fs):
    x = remove_dc(signal)

    x = notch_filter(x, fs, freq=50.0)
    x = bandpass_ecg(x, fs, lowcut=0.5, highcut=40.0)

    x = robust_normalize(x)

    return x
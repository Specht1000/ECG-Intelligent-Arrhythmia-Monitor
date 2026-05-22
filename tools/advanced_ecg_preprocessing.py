import numpy as np
from scipy.signal import butter, filtfilt


def butter_bandpass(lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype="band")

    return b, a


def bandpass_filter(signal, lowcut=0.8, highcut=25.0, fs=250.0):
    b, a = butter_bandpass(lowcut, highcut, fs)

    return filtfilt(b, a, signal)


def remove_dc(signal):
    return signal - np.mean(signal)


def normalize_signal(signal):
    std = np.std(signal)

    if std > 1e-6:
        signal = signal / std

    return signal


def preprocess_ad8232(signal, fs=250.0):

    signal = signal.astype(np.float32)

    signal = remove_dc(signal)

    filtered = bandpass_filter(
        signal,
        lowcut=0.8,
        highcut=25.0,
        fs=fs
    )

    filtered = normalize_signal(filtered)

    return filtered